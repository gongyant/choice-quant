"""P1-A3 step 3: stock cards v2 (6 models). 600060 (out-of-pool): HGB/LGBM/Ridge scored by A1 frozen models; MLP scored with the saved track-B latest MLP checkpoints (verified on in-pool stocks);
GRU/TF NOT scored (need 60-day sequence inputs + cross-sectional amount z-scores + checkpoint scalers; not done -> stated, not fabricated)."""
import sys, json, joblib
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dl"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import p1lib as P1
from p1lib import L, H, V2
import dl_mlp
fm.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc"); plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]; plt.rcParams["axes.unicode_minus"] = False
meta = json.load(open(H / "models" / "model_meta_2026-10-16.json", encoding="utf-8")); fitted = {n: joblib.load(H / "models" / f"{n}_frozen_2026-10-16.joblib") for n in P1.MODELS}
X = L.build_data(); L.limit_flags(X); W = P1.world_from_X(X); i = len(W.cal) - 1; pool = np.where(W.elig[i])[0]; n_pool = len(pool)
lat = pd.read_csv(H / "latest_all_stocks_20261016_v2.csv", dtype={"code": str}, encoding="utf-8-sig", keep_default_na=False, na_values=[""]).set_index("code")
old = json.load(open(H / "scores" / "stock_cards_data_20261016.json", encoding="utf-8"))
# ---- MLP checkpoints ----
ck = {s: torch.load(H / "dl" / "checkpoints" / "mlp" / f"mlp_latest_s{s}_r999.pt", weights_only=False) for s in (42, 43, 44)}
k0 = ck[42]; print("ckpt type", type(k0), list(k0.keys())[:12] if isinstance(k0, dict) else "")
def mlp_pred(xrow):
    out = []
    for s in (42, 43, 44):
        M = ck[s]; M = M.get("model", M) if isinstance(M, dict) and "state" not in M else M; out.append(float(dl_mlp.predict(M, xrow.reshape(1, -1).astype(np.float32))[0]))
    return float(np.mean(out)), out
# verification on in-pool stocks: reproduce B's latest score file
dmlp = pd.read_csv(H / "scores" / "dl_mlp_scores_20261016.csv", dtype={"code": str}).set_index("code"); ver = []
for c in ["000027", "300487"] + list(dmlp.index[dmlp["rank"].notna()][:3]):
    j = X.codes.index(c); m_, seeds = mlp_pred(W.Xq[i][j]); ver.append((c, m_, float(dmlp.loc[c, "mlp_score_mean"]), abs(m_ - float(dmlp.loc[c, "mlp_score_mean"]))))
print("MLP checkpoint verification (recomputed vs B score file):", ver); assert max(v[3] for v in ver) < 1e-5
# ---- 600060 ----
UM = pd.read_csv(V2 / "data" / "universe_master.csv", dtype=str); u2 = UM[UM.code == "600060"].copy(); u2["list_date"] = pd.to_datetime(u2.list_date); u2["delist_date"] = pd.to_datetime(u2.delist_date)
oldRU, oldFA = L.RU, L.FA; L.RU, L.FA = H / "data_extra" / "raw", H / "data_extra" / "adj"; X2 = L.build_data(universe=u2, codes=["600060"]); L.RU, L.FA = oldRU, oldFA; L.limit_flags(X2)
f2, elig2, ADJ2 = L.compute_features(X2, X2.ADJ_unadj_based); fo = {k: float(f2[k].values[i, 0]) for k in L.FEATS}; sco, xrow = P1.score_outsider(W, i, fitted, fo)
assert all(abs(sco[n] - old["600060"]["scores"][n]) < 1e-12 for n in P1.MODELS), "A1 outsider scores reproduced"
mlp60, mlp60_seeds = mlp_pred(np.array(xrow)); psc = dmlp.loc[dmlp["rank"].notna(), "mlp_score_mean"].values
rk60 = int((psc > mlp60).sum()) + 1; pc60 = float((psc < mlp60).mean()); print("600060 MLP", mlp60, mlp60_seeds, rk60, pc60)
KEYS = [("HGB", "hgb"), ("LGBM", "lgbm"), ("Ridge", "ridge"), ("MLP", "dl_mlp"), ("GRU", "dl_gru"), ("TF", "dl_tf")]; cards = {}
for c in ["600060", "300487", "000027"]:
    if c == "600060":
        o = old[c]; sc = dict(o["scores"]); rk = dict(o["ranks"]); pc = dict(o["pct"]); sc["MLP"], rk["MLP"], pc["MLP"] = mlp60, rk60, pc60; sc["GRU"] = sc["TF"] = rk["GRU"] = rk["TF"] = pc["GRU"] = pc["TF"] = None
        XX, j, ff = X2, 0, f2; elig = o["eligible"]; amt, t20 = o["amt20"], o["turn20"]; flags = list(o["flags"]) + ["MLP 为池外打分:用轨道B保存的最新 MLP 检查点(3种子均值)对其11个横截面分位特征打分(检查点已在样本内股票上复现B的分数,误差<1e-5);GRU/Transformer 需要60日序列+全截面成交额z分数+检查点标准化参数,**未打分**(不编造)"]
    else:
        XX, j, ff = X, X.codes.index(c), W.f; r = lat.loc[c]; sc = {n: float(r[f"{k}_score"]) for n, k in KEYS}; rk = {n: int(r[f"rank_{k}"]) for n, k in KEYS}; pc = {n: float(r[f"pctile_{k}"]) for n, k in KEYS}
        amt = float(r.amount20); t20 = float(r.turnover20); elig = bool(r.eligible_asof); flags = [x for x in str(r.quality_flags).split(";") if x and x != "nan"]; flags.insert(0, "在1500样本内,样本内打分")
    px = pd.DataFrame({"date": XX.cal, "o": XX.O[:, j], "h": XX.Hh[:, j], "l": XX.L[:, j], "c": XX.C[:, j], "amt": XX.AMT[:, j], "F": XX.F[:, j]}).dropna(subset=["c"])
    for k in ["o", "h", "l", "c"]: px["a" + k] = px[k] / px.F
    px["ma20"] = px.ac.rolling(20).mean(); px["ma60"] = px.ac.rolling(60).mean(); cur = px.iloc[-1]; ret = lambda k: float(ff[k].values[i, j]); ind = UM.set_index("code").loc[c]; name = ind["name"]
    avail = [pc[n] for n in ["HGB", "LGBM", "MLP", "GRU", "TF"] if pc[n] is not None]; agree = float(np.std(avail)) if len(avail) == 5 else None
    cards[c] = dict(code=c, name=name, industry=f"{ind.industry_l1}/{ind.industry_l2}", board=ind.board, close=float(cur.c), last=str(cur.date.date()), r5=ret("r5"), r20=ret("r20"), r60=ret("r60"), vol20=ret("vol20"), amt20=amt, turn20=t20, eligible=elig, scores=sc, ranks=rk, pct=pc, flags=flags, bars=len(px), max_abs_ret=float(px.ac.pct_change().abs().max()), in_sample=(c != "600060"), agreement_std=agree)
    v = px.tail(250).reset_index(drop=True); fig = plt.figure(figsize=(13.5, 8.4)); gs = fig.add_gridspec(3, 3, height_ratios=[3, 1, 1.25], hspace=0.12, wspace=0.34); ax = fig.add_subplot(gs[0, :2]); up = v.ac >= v.ao
    ax.vlines(v.index, v.al, v.ah, color=np.where(up, "#d62728", "#2ca02c"), lw=0.8); ax.bar(v.index, (v.ac - v.ao).abs().clip(lower=1e-6), bottom=np.minimum(v.ao, v.ac), color=np.where(up, "#d62728", "#2ca02c"), width=0.7)
    ax.plot(v.index, v.ma20, color="#1f77b4", lw=1.2, label="MA20"); ax.plot(v.index, v.ma60, color="#ff7f0e", lw=1.2, label="MA60"); ax.legend(loc="upper left"); ax.set_xticks([]); ax.grid(alpha=0.25)
    ax.set_title(f"{c} {name}  日K(复权价=不复权价/新浪复权因子, 近{len(v)}个交易日, 截至{cards[c]['last']})  最新不复权收盘 {cur.c:.2f}", fontsize=11)
    ax2 = fig.add_subplot(gs[1, :2], sharex=ax); ax2.bar(v.index, v.amt / 1e8, color=np.where(up, "#d62728", "#2ca02c"), width=0.7); ax2.set_ylabel("成交额(亿元)"); tk = np.linspace(0, len(v) - 1, 7).astype(int); ax2.set_xticks(tk); ax2.set_xticklabels([str(v.date[t].date()) for t in tk], fontsize=8); ax2.grid(alpha=0.25)
    ax3 = fig.add_subplot(gs[:2, 2]); labs = [k for k, _ in KEYS]; vals = [pc[l] for l in labs]; colr = ["#4c78a8", "#59a14f", "#b07aa1", "#f28e2b", "#e15759", "#76b7b2"]
    ax3.barh(labs, [(x or 0) * 100 for x in vals], color=colr); ax3.axvline(50, color="gray", ls="--", lw=0.8); ax3.set_xlim(0, 118); ax3.invert_yaxis(); nn_ = n_pool + (1 if c == "600060" else 0)
    for y, (l, x) in enumerate(zip(labs, vals)): ax3.text(2 if x is None else min(x * 100 + 2, 62), y, "未打分(池外,无序列输入)" if x is None else f"第{rk[l]}名/{nn_} 分位{x:.0%}", va="center", fontsize=8.5)
    ax3.set_title("研究排名分位(100=得分最高)\n仅排名,非概率/收益", fontsize=10); ax3.set_xlabel("在合格股票中的分位 (%)"); ax4 = fig.add_subplot(gs[2, :]); ax4.axis("off")
    sct = "  ".join(f"{l} {sc[l]:.4f}" if sc[l] is not None else f"{l} 未打分" for l in labs)
    txt = (f"行业(巨潮最新分类): {cards[c]['industry']}    板块: {cards[c]['board']}    近5/20/60日(复权): {cards[c]['r5']:+.1%} / {cards[c]['r20']:+.1%} / {cards[c]['r60']:+.1%}    20日波动率(日): {cards[c]['vol20']:.2%}    20日成交额中位数: {amt/1e8:.2f}亿\n"
           f"模型原始score(未校准,非概率): {sct}    合格(eligible_asof): {elig}    五模型分歧(百分位std): {'%.3f' % agree if agree is not None else '不适用(GRU/TF缺失)'}\n" + "说明: " + "; ".join(x[:60] for x in flags[:3]) + "\n风险: 5日收益基本不可预测,所有模型扣费净超额统计上不显著;未计滑点/涨跌停成交失败;研究排名≠买入建议")
    ax4.text(0, 1, txt, va="top", fontsize=8.8, wrap=True); fig.savefig(H / "figures" / f"card_{c}_v2.png", dpi=130, bbox_inches="tight"); plt.close(fig)
json.dump(cards, open(H / "scores" / "stock_cards_data_20261016_v2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
L_ = ["# 个股研究卡片 v2(信号日 2026-10-16 收盘,含 MLP/GRU/Transformer;仅为研究排名,**不是买入建议**)", "",
      f"- 排名范围:信号日合格样本池 {n_pool} 只(来自1500只分层抽样样本,非全市场);score 为模型原始输出,**未校准,不是概率也不是收益率**;分位 100% = 得分最高。",
      f"- HGB/LGBM/Ridge:{meta['model_version']};MLP/GRU/TF:轨道B最新模型(训练截至 2026-06-16,验证截至 2026-09-22,标签最晚到期 2026-10-08);TF 为单种子42。行业为巨潮最新分类(非历史时点)。", ""]
for c, d in cards.items():
    L_ += [f"## {c} {d['name']}", "", f"![card](../figures/card_{c}_v2.png)", "", "| 项目 | 数值 |", "|---|---|", f"| 行业 / 板块 | {d['industry']} / {d['board']} |", f"| 数据日期(最后成交日) | {d['last']} |", f"| 最新收盘价(不复权) | {d['close']:.2f} 元 |", f"| 近5/20/60日收益(复权) | {d['r5']:+.2%} / {d['r20']:+.2%} / {d['r60']:+.2%} |",
           f"| 20日波动率(日收益标准差) | {d['vol20']:.2%} |", f"| 20日成交额中位数 | {d['amt20']/1e8:.2f} 亿元 |", f"| 20日平均换手率 | {d['turn20']:.2%} |", f"| 样本状态 | {'1500样本内' if d['in_sample'] else '**不在1500样本内(池外打分)**'};eligible_asof = {d['eligible']} |"]
    for l, _ in KEYS:
        L_.append(f"| {l} score / 排名 / 分位 | " + (f"{d['scores'][l]:.6f} / 第{d['ranks'][l]}名 / {d['pct'][l]:.1%} |" if d['scores'][l] is not None else "**未打分**(池外且GRU/Transformer需要序列输入,未能生成;不编造) |"))
    L_ += [f"| 五模型分歧(HGB/LGBM/MLP/GRU/TF 百分位的标准差) | {('%.3f' % d['agreement_std']) if d['agreement_std'] is not None else '不适用(GRU/TF 缺失)'};越小越一致,不是置信度 |", f"| 数据质量标记 | {'; '.join(d['flags'])} |", "",
           "风险提示:", "- 5日相对收益基本不可预测:6 个模型在 0.30% 往返成本下的净超额均与 0 无法区分(见 p1_dl_report.md);排名位置不代表后续会跑赢。", "- 交易成本可吃掉大部分毛信号;未建模滑点、冲击成本、涨跌停/停牌导致的成交失败。", "- 价格数据主要来自新浪/腾讯公开接口;成交量单位与复权因子仍有单源风险;行业为最新分类,非历史时点。", "- 模型分数仅反映量价特征的横截面位置,不含基本面/消息面;score 不能当作预期收益率。", ""]
(H / "reports" / "stock_cards_20261016_v2.md").write_text("\n".join(L_), encoding="utf-8")
for c, d in cards.items(): print(c, d["name"], d["ranks"], {k: (None if v is None else round(v, 3)) for k, v in d["pct"].items()}, d["agreement_std"])
