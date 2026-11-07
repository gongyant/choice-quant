"""Research cards for 600060 / 300487 / 000027 (as of 2026-10-16). 600060 is NOT in the 1500 sample: scored out-of-pool with the same frozen models."""
import sys, json, joblib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import p1lib as P1
from p1lib import L, H, V2
fm.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc"); plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]; plt.rcParams["axes.unicode_minus"] = False
meta = json.load(open(H/"models"/"model_meta_2026-10-16.json", encoding="utf-8")); fitted = {n: joblib.load(H/"models"/f"{n}_frozen_2026-10-16.joblib") for n in P1.MODELS}
X = L.build_data(); L.limit_flags(X); W = P1.world_from_X(X); i = len(W.cal) - 1; pool = np.where(W.elig[i])[0]; n_pool = len(pool)
lat = pd.read_csv(H/"latest_all_stocks_20261016.csv", dtype={"code": str}, encoding="utf-8-sig", keep_default_na=False, na_values=[""]).set_index("code")
# ---- 600060 out-of-pool ----
UM = pd.read_csv(V2/"data"/"universe_master.csv", dtype=str); u2 = UM[UM.code == "600060"].copy(); u2["list_date"] = pd.to_datetime(u2.list_date); u2["delist_date"] = pd.to_datetime(u2.delist_date)
oldRU, oldFA = L.RU, L.FA; L.RU, L.FA = H/"data_extra"/"raw", H/"data_extra"/"adj"
X2 = L.build_data(universe=u2, codes=["600060"]); L.RU, L.FA = oldRU, oldFA; L.limit_flags(X2)
f2, elig2, ADJ2 = L.compute_features(X2, X2.ADJ_unadj_based)
fo = {k: float(f2[k].values[i, 0]) for k in L.FEATS}; sco, xrow = P1.score_outsider(W, i, fitted, fo)
cards = {}
for c in ["600060", "300487", "000027"]:
    if c == "600060":
        XX, j, ff = X2, 0, f2; sc = sco; ins = False
        pool_scores = {n: P1.score_asof(W, i, fitted)[1][n] for n in P1.MODELS}
        rk = {n: int((pool_scores[n] > sc[n]).sum()) + 1 for n in P1.MODELS}; pc = {n: float((pool_scores[n] < sc[n]).mean()) for n in P1.MODELS}
        amt = float(ff["amt20"].values[i, 0]); t20 = float(pd.DataFrame(XX.TURN).rolling(20, min_periods=15).mean().values[i, 0]); elig = bool(elig2[i, 0])
        flags = ["不在1500样本内:用同一冻结模型对其做\"池外\"打分(特征分位数相对 2026-10-16 合格样本池计算),排名为插入样本池后的名次,非样本内结果",
                 "价格双源(新浪不复权 vs 腾讯不复权,1400个共同日收盘价逐日一致);复权因子仅新浪单源"]
    else:
        XX, j, ff = X, X.codes.index(c), W.f; r = lat.loc[c]; sc = {n: r[f"{k}_score"] for n, k in zip(P1.MODELS, ["ridge", "hgb", "lgbm"])}
        rk = {n: int(r[f"rank_{k}"]) for n, k in zip(P1.MODELS, ["ridge", "hgb", "lgbm"])}; pc = {n: float(r[f"pctile_{k}"]) for n, k in zip(P1.MODELS, ["ridge", "hgb", "lgbm"])}
        amt = float(r.amount20); t20 = float(r.turnover20); elig = bool(r.eligible_asof); flags = [x for x in str(r.quality_flags).split(";") if x and x != "nan"]
        flags.insert(0, "在1500样本内,样本内打分")
    px = pd.DataFrame({"date": XX.cal, "o": XX.O[:, j], "h": XX.Hh[:, j], "l": XX.L[:, j], "c": XX.C[:, j], "amt": XX.AMT[:, j], "F": XX.F[:, j]}).dropna(subset=["c"])
    for k in ["o", "h", "l", "c"]: px["a" + k] = px[k] / px.F    # corporate-action adjusted (raw / Sina cumulative factor)
    px["ma20"] = px.ac.rolling(20).mean(); px["ma60"] = px.ac.rolling(60).mean(); cur = px.iloc[-1]
    ret = lambda k: float(ff[k].values[i, j]); name = UM.set_index("code").loc[c, "name"]; ind = UM.set_index("code").loc[c]
    maxabs = float(px.ac.pct_change().abs().max()); nbar = len(px)
    cards[c] = dict(code=c, name=name, industry=f"{ind.industry_l1}/{ind.industry_l2}", board=ind.board, close=float(cur.c), last=str(cur.date.date()), r5=ret("r5"), r20=ret("r20"), r60=ret("r60"), vol20=ret("vol20"), amt20=amt, turn20=t20,
                    eligible=elig, scores={n: float(sc[n]) for n in P1.MODELS}, ranks=rk, pct=pc, flags=flags, bars=nbar, max_abs_ret=maxabs, in_sample=(c != "600060"))
    # ---- figure ----
    v = px.tail(250).reset_index(drop=True); fig = plt.figure(figsize=(13, 8.2)); gs = fig.add_gridspec(3, 3, height_ratios=[3, 1, 1.2], hspace=0.12, wspace=0.3)
    ax = fig.add_subplot(gs[0, :2]); up = v.ac >= v.ao
    ax.vlines(v.index, v.al, v.ah, color=np.where(up, "#d62728", "#2ca02c"), lw=0.8); ax.bar(v.index, (v.ac - v.ao).abs().clip(lower=1e-6), bottom=np.minimum(v.ao, v.ac), color=np.where(up, "#d62728", "#2ca02c"), width=0.7)
    ax.plot(v.index, v.ma20, color="#1f77b4", lw=1.2, label="MA20"); ax.plot(v.index, v.ma60, color="#ff7f0e", lw=1.2, label="MA60"); ax.legend(loc="upper left"); ax.set_xticks([])
    ax.set_title(f"{c} {name}  日K(复权价=不复权价/新浪复权因子, 近{len(v)}个交易日, 截至{cards[c]['last']})  最新不复权收盘 {cur.c:.2f}", fontsize=11); ax.grid(alpha=0.25)
    ax2 = fig.add_subplot(gs[1, :2], sharex=ax); ax2.bar(v.index, v.amt / 1e8, color=np.where(up, "#d62728", "#2ca02c"), width=0.7); ax2.set_ylabel("成交额(亿元)")
    tk = np.linspace(0, len(v) - 1, 7).astype(int); ax2.set_xticks(tk); ax2.set_xticklabels([str(v.date[t].date()) for t in tk], rotation=0, fontsize=8); ax2.grid(alpha=0.25)
    ax3 = fig.add_subplot(gs[:2, 2]); labs = ["HGB", "LGBM", "Ridge"]; vals = [pc["HGB"], pc["LGBM"], pc["Ridge"]]
    ax3.barh(labs, [x * 100 for x in vals], color=["#4c78a8", "#59a14f", "#b07aa1"]); ax3.axvline(50, color="gray", ls="--", lw=0.8); ax3.set_xlim(0, 100); ax3.invert_yaxis()
    for y, (l, x) in enumerate(zip(labs, vals)): ax3.text(min(x * 100 + 2, 62), y, f"第{rk[l]}名 / {n_pool + (1 if c == '600060' else 0)}  分位{x:.0%}", va="center", fontsize=9)
    ax3.set_title("研究排名分位(100=得分最高)\n仅排名,非概率/收益", fontsize=10); ax3.set_xlabel("在合格股票中的分位 (%)")
    ax4 = fig.add_subplot(gs[2, :]); ax4.axis("off")
    txt = (f"行业(巨潮最新分类): {cards[c]['industry']}    板块: {cards[c]['board']}    近5/20/60日(复权): {cards[c]['r5']:+.1%} / {cards[c]['r20']:+.1%} / {cards[c]['r60']:+.1%}    20日波动率(日): {cards[c]['vol20']:.2%}    20日成交额中位数: {amt/1e8:.2f}亿\n"
           f"模型原始score(未校准,非概率): HGB {sc['HGB']:.6f}  LGBM {sc['LGBM']:.6f}  Ridge {sc['Ridge']:.6f}    合格(eligible_asof): {elig}\n" + "说明: " + "; ".join(flags[:3]) + "\n风险: 5日收益基本不可预测,模型净超额统计上不显著;未计滑点/涨跌停成交失败;研究排名≠买入建议")
    ax4.text(0, 1, txt, va="top", fontsize=9, wrap=True); fig.savefig(H/"figures"/f"card_{c}.png", dpi=130, bbox_inches="tight"); plt.close(fig)
json.dump(cards, open(H/"scores"/"stock_cards_data_20261016.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
# ---- markdown ----
L_ = ["# 个股研究卡片(信号日 2026-10-16 收盘,仅为研究排名,**不是买入建议**)", "",
      f"- 排名范围:信号日合格样本池 {n_pool} 只(来自1500只分层抽样样本,非全市场);score 为模型原始输出,**未校准,不是概率也不是收益率**;分位 100% = 得分最高。",
      f"- 模型版本:{meta['model_version']};训练区间 {meta['train_first_date']}~{meta['train_last_date']}(训练行数 {meta['n_train_rows']:,});行业为巨潮最新分类(非历史时点)。", ""]
for c, d in cards.items():
    L_ += [f"## {c} {d['name']}", "", f"![card](../figures/card_{c}.png)", "", "| 项目 | 数值 |", "|---|---|",
           f"| 行业 / 板块 | {d['industry']} / {d['board']} |", f"| 数据日期(最后成交日) | {d['last']} |", f"| 最新收盘价(不复权) | {d['close']:.2f} 元 |",
           f"| 近5/20/60日收益(复权) | {d['r5']:+.2%} / {d['r20']:+.2%} / {d['r60']:+.2%} |", f"| 20日波动率(日收益标准差) | {d['vol20']:.2%} |", f"| 20日成交额中位数 | {d['amt20']/1e8:.2f} 亿元 |", f"| 20日平均换手率 | {d['turn20']:.2%} |",
           f"| 样本状态 | {'1500样本内' if d['in_sample'] else '**不在1500样本内(池外打分)**'};eligible_asof = {d['eligible']} |", f"| 数据 | {d['bars']} 个交易日(2021-01-04起),复权后最大单日绝对涨跌幅 {d['max_abs_ret']:.2%} |",
           f"| HGB score / 排名 / 分位 | {d['scores']['HGB']:.6f} / 第{d['ranks']['HGB']}名 / {d['pct']['HGB']:.1%} |", f"| LightGBM score / 排名 / 分位 | {d['scores']['LGBM']:.6f} / 第{d['ranks']['LGBM']}名 / {d['pct']['LGBM']:.1%} |",
           f"| Ridge score / 排名 / 分位 | {d['scores']['Ridge']:.6f} / 第{d['ranks']['Ridge']}名 / {d['pct']['Ridge']:.1%} |", f"| 数据质量标记 | {'; '.join(d['flags'])} |", "",
           "风险提示:", "- 5日相对收益基本不可预测:P0 修复后 HGB/LGBM 的扣费净超额统计上与 0 无法区分(t≈0.7),排名位置不代表后续会跑赢。",
           "- 交易成本(往返约0.3%)可吃掉大部分毛信号;未建模滑点、冲击成本、涨跌停/停牌导致的成交失败。", "- 价格数据主要来自新浪/腾讯公开接口(P0 已逐日交叉核验收盘价,但成交量单位与复权因子仍有单源风险);行业为最新分类,非历史时点。",
           "- 模型分数仅反映 11 个量价因子的横截面位置,不含基本面/消息面;score 不能当作预期收益率。", ""]
(H/"reports"/"stock_cards_20261016.md").write_text("\n".join(L_), encoding="utf-8")
for c, d in cards.items(): print(c, d["name"], d["industry"], d["close"], round(d["r5"], 4), round(d["r20"], 4), round(d["r60"], 4), round(d["vol20"], 4), round(d["amt20"] / 1e8, 2), d["eligible"], d["ranks"], {k: round(v, 3) for k, v in d["pct"].items()}, d["scores"], d["flags"])




