# -*- coding: utf-8 -*-
"""P3-B6 (v2 rules; generated from p3_report.py by _mk_v2.py): figures (figures/cand_<code>_<date>.png) + reports/daily_ai_report_<date>.md. Reads only P3 outputs; registry untouched."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
H = Path(__file__).resolve().parent; R = H / "results"; FIG = H / "figures"; REP = H / "reports"; FIG.mkdir(exist_ok=True); REP.mkdir(exist_ok=True)
FP = "C:/Windows/Fonts/msyh.ttc"; fm.fontManager.addfont(FP); plt.rcParams["font.family"] = fm.FontProperties(fname=FP).get_name(); plt.rcParams["axes.unicode_minus"] = False
FIVE = ["HGB", "LGBM", "MLP", "GRU", "TF"]
pc = lambda x: "—" if pd.isna(x) else f"{x:+.1%}"; pp = lambda x: "—" if pd.isna(x) else f"{x:.0%}"
def label(r):
    if r.top5_v2: return "继续观察"
    if r.watch_v2: return "暂不买入(下跌观察)"
    if r.top5: return "暂不买入(继续观察)" if r.trend == "下跌" else "继续观察"
    return "暂不买入(回避)"
def fig(code, r, z, HC, CT, date):
    codes = list(z["codes"]); j = codes.index(code); cal = pd.DatetimeIndex(z["cal"])[-z["C"].shape[0]:]; n = 120
    A, C, O, Hh, Lo = z["ADJf"][-n:, j], z["C"][-n:, j], z["O"][-n:, j], z["Hh"][-n:, j], z["Lo"][-n:, j]; d = cal[-n:]
    k = np.where(np.isfinite(C))[0][-1]; s = C[k] / A[k]; f = (A / np.where(np.isfinite(C), C, np.nan)) * s
    o, h, l, c = O * f, Hh * f, Lo * f, C * f
    fg, ax = plt.subplots(3, 1, figsize=(11, 10), gridspec_kw=dict(height_ratios=[3, 1.4, 1.4]))
    x = np.arange(n); up = c >= o
    ax[0].vlines(x, l, h, color=np.where(up, "#d62728", "#2ca02c"), lw=0.8); ax[0].bar(x, np.abs(c - o), bottom=np.minimum(o, c), color=np.where(up, "#d62728", "#2ca02c"), width=0.7)
    ma = pd.Series(A * s).rolling(20, min_periods=15).mean().values; ax[0].plot(x, ma, color="#1f77b4", lw=1, label="MA20")
    last = c[k]
    if pd.notna(r.calib_bin):
        for hz, xo, col in [("5d", n + 5, "#ff7f0e"), ("20d", n + 20, "#9467bd")]:
            t = CT[(CT.horizon == hz) & (CT.bin == int(r.calib_bin))].iloc[0]
            ax[0].vlines(xo, last * (1 + t.abs_q10), last * (1 + t.abs_q90), color=col, lw=6, alpha=0.5, label=f"{hz} 历史经验区间(10%-90%)"); ax[0].plot(xo, last * (1 + t.abs_q50), "o", color=col)
    ax[0].set_xlim(-1, n + 24); tk = list(range(0, n, 20)); ax[0].set_xticks(tk); ax[0].set_xticklabels([d[i].strftime("%m-%d") for i in tk])
    ax[0].set_title(f"{code} {r['name']}  收盘 {r.close_unadj:.2f}(截至 {date[:4]}-{date[4:6]}-{date[6:]},前复权口径对齐最新收盘)"); ax[0].legend(loc="upper left", fontsize=8); ax[0].grid(alpha=0.3)
    hc = HC[HC.code == code].sort_values("signal_date"); hd = pd.to_datetime(hc.signal_date)
    ax[1].plot(hd, hc.composite, "-o", ms=2, lw=0.8, label="历史 OOS 综合分位(5 模型)");
    if pd.notna(r.composite_pctile): ax[1].plot(pd.Timestamp(date), r.composite_pctile, "r*", ms=12, label="最新")
    ax[1].axhline(0.9, color="gray", ls="--", lw=0.7); ax[1].set_ylim(0, 1.05); ax[1].legend(fontsize=8, loc="lower left"); ax[1].grid(alpha=0.3); ax[1].set_ylabel("分位")
    ax[2].bar(hd, hc.realized_rel_ret_5d, width=4, color=np.where(hc.realized_rel_ret_5d >= 0, "#d62728", "#2ca02c")); ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_ylabel("实际5日相对收益"); ax[2].set_title("各历史信号日之后的实际 5 日相对池均值收益(事后已实现)", fontsize=9); ax[2].grid(alpha=0.3)
    ax[1].set_xlim(pd.Timestamp("2022-06-15"), pd.Timestamp(date) + pd.Timedelta(days=20)); ax[2].set_xlim(ax[1].get_xlim())
    fg.text(0.01, 0.005, "研究图表,不是买入推荐;区间为同组股票的历史经验分位,不是概率或预测价。", fontsize=8, color="gray"); fg.tight_layout()
    p = FIG / f"cand_{code}_{date}.png"; fg.savefig(p, dpi=110); plt.close(fg); assert p.stat().st_size > 20000, p; return p, len(hc)
def main(date):
    Rk = pd.read_csv(H / f"latest_research_ranking_{date}_v2.csv", dtype={"code": str}, encoding="utf-8-sig"); z = np.load(H / "data" / f"world_{date}.npz")
    CT = pd.read_csv(R / "calibration_table.csv", encoding="utf-8-sig"); SU = pd.read_csv(R / "calibration_supplement.csv", encoding="utf-8-sig"); CS = json.load(open(R / "calibration_summary.json", encoding="utf-8"))
    HC = pd.read_parquet(R / "hist_composite.parquet"); FU = json.load(open(R / f"fundamentals_{date}_v2.json", encoding="utf-8")); CI = json.load(open(R / "company_info_20261021_v2.json", encoding="utf-8"))
    Q = json.load(open(R / "data_quality_summary.json", encoding="utf-8")); MI = json.load(open(H / "data" / "merge_info.json", encoding="utf-8")); man = [json.loads(l) for l in open(H / "predictions_registry" / "manifest.jsonl", encoding="utf-8")][-1]
    top = Rk[Rk.top5_v2].sort_values("sel", ascending=False); wl = Rk[Rk.watch_v2].sort_values("sel", ascending=False); v1t = Rk[Rk.top5].sort_values("sel_score", ascending=False); av = Rk[Rk.avoid5].sort_values("avoid_score", ascending=False); ds = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    figs = {c: fig(c, r, z, HC, CT, date) for c, r in pd.concat([top, wl, av]).set_index("code").iterrows()}
    t = lambda hz, b: CT[(CT.horizon == hz) & (CT.bin == int(b))].iloc[0]; su = lambda hz, b: SU[(SU.horizon == hz) & (SU.bin == int(b))].iloc[0]
    t5, t20, s5, s20 = t("5d", 10), t("20d", 10), su("5d", 10), su("20d", 10)
    L = [f"# 每日 AI 研究报告(v2 规则) · 数据截至 {ds} 收盘", "",
         "> **这是研究记录,不是买入推荐。** 模型分数不是概率,也不是预期收益;下文的百分比是**历史上同类股票相对池子平均的经验频率**,不是精确概率。P2 结论仍然有效:模型信号**未被证明在扣除交易成本后能赚钱**。本报告不连券商、不下单。", "",
         "## 一句话回答", "",
         f"本次分析了样本内 **{len(Rk)} 只**股票(其中 {int(Rk.eligible_asof.sum())} 只满足可选池条件参与打分),数据截止 **{ds}**。按事先写好并登记的 v2 规则(`selection_rules_v2.json`,在计算前登记 sha256),重点观察的 " + str(len(top)) + " 只是:"
         + "、".join(f"**{r['name']}({r.code})**" for _, r in top.iterrows()) + "。",
         "入选理由:它们都不处于“20 日跌超 15% 且 MA20 下行”的下跌状态,也没有追高(20 日涨幅≤20%、无连续涨停);5 个冻结模型(HGB/LightGBM/MLP/GRU/Transformer)**一致**把它们排在前 5% 左右,模型分歧小,成交额足够(≥1 亿/日),非 ST、未涨停封板、未停牌,近期没有大涨。",
         f"**可靠性:低到中等。** 历史上最高分组(前 10%)的股票,5 日后“跑赢池均值超过 2%”的比例({t5.freq_outperform:.1%})**并不比全体({t5.base_outperform:.1%})高**;它们的优势主要体现在**更少大幅跑输**(跑输超过 2% 的比例 {t5.freq_underperform:.1%},全体 {t5.base_underperform:.1%})和平均相对收益略高(+{s5.mean_rel:.2%}/5 日)。按事先定义,情景都标为“预测可信度低”。v1 的 5 只中德福科技、日联科技、科威尔因下跌趋势移入“下跌观察名单”(暂不买入)。历史检验显示 v2 的收紧主要是风险控制,**并没有提高历史收益**(见 `history_vs_actual_report.md`)。", "",
         "## 数据与模型", "",
         f"- 数据:P0 原始价(2026-10-16 前)+ 本次增量 {', '.join(MI['new_trade_dates'])};{Q['本次成功增量采集']} 只全部采到,最后日期为 {ds} 的 {Q[f'最后日期 = {ds}']} 只;过期(停牌)非退市股 {Q[f'过期(最后日期 < {ds},不含已退市)']} 只;样本内历史已退市 {Q['已终止上市(样本内历史退市)']} 只。详见 `data_quality_latest.md`。",
         f"- 源:新浪原始日线经本机代理频繁 SSL 中断,SH/SZ 改用 P0 已记录的腾讯备用源,北交所用新浪;与 P0 重叠期(09-01~10-16)收盘价逐日完全一致(最大差 {MI['overlap_close_maxabs_max']}),成交额相对差 p99 {MI['overlap_amount_maxrel_p99']:.1e}。",
         f"- 模型:全部为冻结模型,未重训。HGB/LGBM/Ridge(2026-10-16 冻结 joblib);MLP、GRU 为 3 种子 latest checkpoint 平均;Transformer 只有 seed 42 的 latest checkpoint。用新数据复算 10-16 的分数与 P1 发布值一致(树模型差 <1e-16,GRU/TF Spearman 0.9999995)。",
         f"- 预测登记:`predictions_registry/{man['file']}`(sha256 `{man['sha256'][:16]}…`,{man['rows']} 行,生成于 {man['generated_at']}),只追加、不修改。", "",
         "## 总表", "", "| 类别 | 代码 | 名称 | 收盘 | 5日 | 20日 | 趋势 | 综合分位 | HGB | LGBM | TF | 分歧 | 20日成交额(亿) | 风险等级 | 观察优先级/结论 |", "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for k, (_, r) in enumerate(pd.concat([top, wl, av]).iterrows()):
        L.append(f"| {'重点观察' if r.top5_v2 else ('下跌观察' if r.watch_v2 else '回避')} | {r.code} | {r['name']} | {r.close_unadj:.2f} | {pc(r.chg_5d)} | {pc(r.chg_20d)} | {r.trend} | {pp(r.composite_pctile)} | {pp(r.pctile_HGB)} | {pp(r.pctile_LGBM)} | {pp(r.pctile_TF)} | {'—' if pd.isna(r.model_disagreement) else f'{r.model_disagreement:.2f}'} | {r.amt20/1e8:.2f} | {r.risk_level} | {(str(k+1)+' · ') if r.top5_v2 else ''}{label(r)} |")
    L += ["", "风险等级:高 = ST/退市风险、近 5 日≥2 次跌停或高波动且低流动性;中 = 高波动(池内前 20%)或成交额<1 亿或当日跌停;低 = 其余。", ""]
    def scen(r):
        if pd.isna(r.calib_bin): return ["- 5 日 / 20 日情景:该股未进入打分池,无校准组,**无法给出情景**。"]
        a, b = t("5d", r.calib_bin), t("20d", r.calib_bin); c = r.close_unadj
        return [f"- **5 日情景**(第 {int(r.calib_bin)} 组,历史 {int(a.n_obs)} 个样本):跑赢池均值>2% {a.freq_outperform:.0%} / 持平 {a.freq_flat:.0%} / 跑输>2% {a.freq_underperform:.0%}(全体 {a.base_outperform:.0%}/{1-a.base_outperform-a.base_underperform:.0%}/{a.base_underperform:.0%});绝对涨跌经验区间 10%/50%/90% = {a.abs_q10:+.1%}/{a.abs_q50:+.1%}/{a.abs_q90:+.1%},折合价格约 {c*(1+a.abs_q10):.2f} / {c*(1+a.abs_q50):.2f} / {c*(1+a.abs_q90):.2f}。可信度:**{a.credibility}**。",
                f"- **20 日情景**:跑赢>5% {b.freq_outperform:.0%} / 持平 {b.freq_flat:.0%} / 跑输>5% {b.freq_underperform:.0%}(全体 {b.base_outperform:.0%}/{1-b.base_outperform-b.base_underperform:.0%}/{b.base_underperform:.0%});绝对涨跌经验区间 {b.abs_q10:+.1%}/{b.abs_q50:+.1%}/{b.abs_q90:+.1%},约 {c*(1+b.abs_q10):.2f} / {c*(1+b.abs_q50):.2f} / {c*(1+b.abs_q90):.2f}。可信度:**{b.credibility}**。",
                "- 注意:区间是同组所有股票的历史分布,该股自身波动(vol20 池内分位 " + pp(r.vol20_pctile) + ")越高,实际区间可能越宽。"]
    def card(r):
        c = CI.get(r.code, {}); f = FU.get(r.code, {}); url1 = f"https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpInfo/stockid/{r.code}.phtml"; url2 = f"https://vip.stock.finance.sina.com.cn/corp/go.php/vCB_AllBulletin/stockid/{r.code}.phtml"
        L2 = [f"### {r['name']}({r.code}) · 收盘 {r.close_unadj:.2f} 元({ds}) · {r.exchange} {r.board} · 行业(最新分类) {r.industry_l1}", "", f"![{r.code}](../figures/cand_{r.code}_{date}.png)", "",
              f"- **主营**:{c.get('main') or '未取得可靠数据'}(来源:新浪财经公司资料 {url1},抓取 2026-10-21)"]
        if f and not f.get("error") and f.get("period"):
            yo = lambda k: "" if f.get(k + "_同比") is None else f",同比 {f[k+'_同比']:+.0%}"
            L2.append(f"- **最近财报**({f['period'][:4]}-{f['period'][4:6]}-{f['period'][6:]} 期累计):营业总收入 {f.get('营业总收入', np.nan)/1e8:.2f} 亿{yo('营业总收入')};归母净利润 {f.get('归母净利润', np.nan)/1e8:.3f} 亿{yo('归母净利润')};扣非净利润 {f.get('扣非净利润', np.nan)/1e8:.3f} 亿{yo('扣非净利润')}(来源:新浪财经财务摘要 {f['url']})")
        else: L2.append("- **最近财报**:未取得可靠数据")
        if c.get("ann"): L2.append("- **近期公告**(只列标题,不推断因果;来源 " + url2 + "):" + ";".join(f"{a} {b}" for a, b in c["ann"]))
        else: L2.append("- **近期公告**:未取得可靠数据(新浪公告页未返回列表)")
        L2.append(f"- **近期涨跌**:5 日 {pc(r.chg_5d)},20 日 {pc(r.chg_20d)},60 日 {pc(r.chg_60d)},趋势 **{r.trend}**(MA20 5 日斜率 {pc(r.ma20_slope_5d)},收盘相对 MA20 {pc(r.close_vs_ma20)})。**涨跌原因:未取得可靠的新闻证据**,上面公告仅供自行核对是否相关。")
        if pd.notna(r.composite_pctile):
            L2.append(f"- **模型看法**:HGB 第 {int(r.rank_HGB)} 名(分位 {pp(r.pctile_HGB)})、LightGBM 第 {int(r.rank_LGBM)} 名({pp(r.pctile_LGBM)})、Transformer 第 {int(r.rank_TF)} 名({pp(r.pctile_TF)});MLP {pp(r.pctile_MLP)}、GRU {pp(r.pctile_GRU)};综合分位 {pp(r.composite_pctile)},分歧 {r.model_disagreement:.2f};简单因子:5日反转 {pp(r.pctile_REV5)}、低波动 {pp(r.pctile_LOWVOL)}、三因子 {pp(r.pctile_F3)}。")
        else: L2.append("- **模型看法**:未进入 2026-10-21 打分池(成交额/历史/停牌条件不满足),无模型分数。")
        hc = HC[HC.code == r.code]; hi = hc[hc.bin == 10]
        L2.append(f"- **历史类似信号**:该股在 {len(hc)} 个历史 OOS 信号日中有 {len(hi)} 次进入最高组;那几次之后 5 日相对收益中位数 {pc(hi.realized_rel_ret_5d.median()) if len(hi) else '—'},为正的比例 {pp((hi.realized_rel_ret_5d > 0).mean()) if len(hi) else '—'}(单股样本少,仅作参考)。")
        L2 += scen(r)
        rk = []
        if r.high_vol: rk.append("波动高(池内前 20%)")
        if r.low_liquidity: rk.append("成交额<1 亿")
        if r.st_or_delist_risk: rk.append("ST/退市风险警示")
        if r.consec_limit_down: rk.append(f"近 5 日跌停 {int(r.n_limit_down_5d)} 次")
        if r.needs_access: rk.append("科创板/北交所需开通权限")
        if r.trend == "下跌": rk.append("处于下跌趋势(模型偏好近期下跌股,存在“抄在半山腰”的风险)")
        L2 += [f"- **风险**:{r.risk_level};" + ("、".join(rk) if rk else "无规则内的特别风险标记"), f"- **结论:{label(r)}**", ""]
        return L2
    L += ["## v1 与 v2 对比", "", "| | v1(selection_rules.json) | v2(selection_rules_v2.json) |", "|---|---|---|", "| Top5 | " + "、".join(f"{r['name']}({r.code})" for _, r in v1t.iterrows()) + " | " + "、".join(f"{r['name']}({r.code})" for _, r in top.iterrows()) + " |", f"| 通过过滤只数 | {int(Rk.passes_top_filters.sum())} | {int(Rk.passes_v2.sum())} |", "| 新增约束 | — | T1 20 日跌超 15% 且 MA20 下行 → 下跌观察;T2 20 日涨幅>20% 排除;T3 5 日内≥2 次涨停排除 |", "| 回避名单 | 同 v2 | 与 v1 相同(规则未改) |", "", f"今天 v1 通过的 {int(Rk.passes_top_filters.sum())} 只中,T1 排除 {int((Rk.passes_top_filters & Rk.T1_down).sum())} 只,T2、T3 各 {int((Rk.passes_top_filters & Rk.T2_chase20).sum())}、{int((Rk.passes_top_filters & Rk.T3_limitups).sum())} 只。v2 新进入的 " + "、".join(f"{r['name']}" for _, r in top.iterrows() if not r.top5) + " 综合分位略低于 v1 的被排除者。", "", "## 重点观察(卡片)", ""]
    for _, r in top.iterrows(): L += card(r)
    L += ["## 下跌观察名单(暂不买入)", "", "满足 v1 全部条件、模型也看好,但 20 日跌幅超过 15% 且 MA20 向下,按 v2 不进 Top5。历史上这类股票短期反弹并不少(探索期被排除者 5 日平均相对 +1.7%),所以这只是风险控制,不是“会继续跌”的判断。", ""]
    for _, r in wl.iterrows(): L += card(r)
    L += ["## 回避名单 5 只", "", "规则:模型一致看空(综合分位≤20% 且 ≥4 个模型分位≤30%),或 现名含 ST/退市风险,或 近 5 日跌停≥2 次;再按“看空程度+高波动+低流动性+ST+连续跌停”打分取前 5。注:四通股份(603838)按样本旧名“*ST四通”曾入选,但 10-17 已撤销退市风险警示、现名无 ST,已按现名剔除。", ""]
    for _, r in av.iterrows(): L += card(r)
    L += ["## 校准核查:为什么最高分组“跑赢 2%”的比例反而不高?", "",
          f"- **基准怎么定义**:基准 = 2022-07~2026-09 共 {CS['n_signal_dates']} 个信号日、全部 {CS['n_obs']:,} 个(股票,日期)样本中“5 日相对池均值 > +2%”的比例 = {t5.base_outperform:.1%}(20 日、>+5%:{t20.base_outperform:.1%})。10 个组样本数相同,所以基准就是 10 组的平均。",
          f"- **最高组(第 10 组)**:5 日跑赢>2% {t5.freq_outperform:.1%}(差 {t5.diff_vs_base:+.1%},95% 区间 [{t5.diff_ci_lo:+.1%}, {t5.diff_ci_hi:+.1%}],不显著 → 可信度低);但 **跑输>2% 只有 {t5.freq_underperform:.1%},比全体少 {abs(t5.under_minus_base):.1%}**(区间 [{t5.under_diff_ci_lo:+.1%}, {t5.under_diff_ci_hi:+.1%}],显著)。",
          f"- **中位数与均值**:最高组 5 日相对收益中位数 {s5.median_rel:+.2%}(最低组 {su('5d',1).median_rel:+.2%}),均值 {s5.mean_rel:+.2%}/5 日(区间 [{s5.mean_rel_ci_lo:+.2%}, {s5.mean_rel_ci_hi:+.2%}]);20 日均值 {s20.mean_rel:+.2%}(区间 [{s20.mean_rel_ci_lo:+.2%}, {s20.mean_rel_ci_hi:+.2%}])。所有组中位数都为负,是因为个股收益右偏(少数大涨拉高池均值)。",
          f"- **补充定义:跑赢“当天池中位数”的比例**(事后加的补充,非预注册):最高组 5 日 {s5.share_beat_pool_median:.1%}(区间 [{s5.share_beat_median_ci_lo:.1%}, {s5.share_beat_median_ci_hi:.1%}]),20 日 {s20.share_beat_pool_median:.1%}([{s20.share_beat_median_ci_lo:.1%}, {s20.share_beat_median_ci_hi:.1%}]);最低组 5 日 {su('5d',1).share_beat_pool_median:.1%}。",
          f"- **原因:两端尾部厚度不同**。最低组波动大(5 日相对收益标准差 {su('5d',1).std_rel:.1%},|相对收益|>2% 的比例 {su('5d',1).share_tails:.0%}),所以它“大涨”和“大跌”都更多;最高组更“稳”(标准差 {s5.std_rel:.1%},尾部 {s5.share_tails:.0%})。“跑赢 2%”这个定义只数右尾,于是高波动的低分组反而更高。RankIC 为正({CS['composite_rankic_5d_mean']:.3f},{CS['composite_rankic_5d_pos_share']:.0%} 的信号日为正)与此并不矛盾:排序信息主要来自**左尾(少跑输)和整体位置略高**。",
          "- **用大白话说**:模型选出的高分股,历史上并不更容易“大涨跑赢”,而是**更不容易“大跌跑输”**,平均每 5 天比池子平均多约 0.3 个百分点——这么小的差距,扣掉买卖成本后很可能就没了(P2 结论)。原定义与可信度标签保持不变,未调整阈值。", "",
          "### 校准表(第 1、5、10 组)", "", "| 期限 | 组 | 样本 | 跑赢 | 持平 | 跑输 | 跑赢-基准 [95%区间] | 均值相对收益 | 中位数相对收益 | 跑赢池中位数 | 绝对 10%/50%/90% | 可信度 |", "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|"]
    for hz in ("5d", "20d"):
        for b in (1, 5, 10):
            a, s = t(hz, b), su(hz, b); L.append(f"| {hz} | {b} | {int(a.n_obs)} | {a.freq_outperform:.1%} | {a.freq_flat:.1%} | {a.freq_underperform:.1%} | {a.diff_vs_base:+.1%} [{a.diff_ci_lo:+.1%},{a.diff_ci_hi:+.1%}] | {s.mean_rel:+.2%} | {s.median_rel:+.2%} | {s.share_beat_pool_median:.1%} | {a.abs_q10:+.1%}/{a.abs_q50:+.1%}/{a.abs_q90:+.1%} | {a.credibility} |")
    Y = pd.read_csv(R / "calibration_by_year.csv", encoding="utf-8-sig")
    L += ["", f"20 日另用非重叠子样本(每 4 个信号日取 1 个)验证:第 10 组跑赢差 {t20.nonoverlap_diff_vs_base:+.1%} [{t20.nonoverlap_diff_ci_lo:+.1%},{t20.nonoverlap_diff_ci_hi:+.1%}],同样不显著。",
          "分年度第 10 组 5 日跑赢>2% 频率(基准):" + ";".join(f"{int(y['year'])} {y['10']:.1%}({y['base']:.1%})" for _, y in Y[Y.horizon == "5d"].iterrows()) + "。各年度都没有稳定高于基准。", "",
          "## Top5 是否符合 selection_rules(v1 历史说明,保留)", "",
          "- 规则不是只按模型分数:先过滤(可选池、20 日成交额≥1 亿、非 ST/退市风险、未涨停封板、未停牌、未跌停、近 5 日跌停<2 次、≥4/5 模型分位≥0.70、分歧≤0.15、5 日涨幅≤15% 且 20 日涨幅≤30%、排除“下跌趋势且 5 日跌幅>8%”),共 " + str(int(Rk.passes_top_filters.sum())) + " 只通过;再用 sel = 综合分位 − 0.10×波动分位 − 0.05×[20 日涨幅>15%] 排序。",
          "- 德福科技 20 日 −32%、趋势判为“下跌”,但 5 日跌幅 −5.8% 未超过 −8%,所以按事先规则没有被“不自动抄跌”条件排除;日联科技(−23%)、科威尔(−18%)同理。这说明规则的抄跌约束偏松,而且模型本身偏好近期下跌的股票(简单因子 5 日反转分位也高)。**报告层面对这 3 只标“暂不买入(继续观察)”**,这是对规则结果加的一层保守提示,规则文件本身没有改;是否收紧(例如排除所有“下跌趋势”)留待第二批作为 v2 规则事先登记后再用。", "",
          "## 未来 1 周怎么核验", "", "- 行情更新后运行 `run_weekly_verify.ps1 20261029`(抓取→合并→`p3_verify_forward.py`),结果只追加写入 `verification/`。",
          f"- 以 {ds} 收盘为信号,按 P0 口径:2026-10-22 收盘作起点,2026-10-29 收盘(第 6 个交易日)算 5 日相对池均值收益;20 日约在 2026-11-19 前后(以交易所日历为准)。",
          f"- 核验方法:用 `predictions_registry/{man['file']}` 里登记的分数(不可修改),计算 5 只观察股和 5 只回避股的实际相对收益、是否落在上面的 10%~90% 区间、全池 RankIC;写入新文件,不改本报告。", "",
          "## 模型到底准不准", "",
          f"- 历史 OOS(2022-07~2026-09,207 个信号日)5 模型综合分位与 5 日相对收益的平均 RankIC = {CS['composite_rankic_5d_mean']:.3f},{CS['composite_rankic_5d_pos_share']:.0%} 的日子为正;20 日 RankIC {CS['composite_rankic_20d_mean']:.3f}。排序有一点信息,但很弱。",
          "- 对单只股票几乎没有确定性:最高组 5 日的绝对涨跌 10%~90% 区间约 −6% 到 +7%,和其他组差别不大。",
          "- P2 已检验:扣除交易成本后,基于这些分数的组合没有被证明能稳定赚钱;2022-07 之后的数据已被反复使用,结论属探索性。", "",
          "## 取不到 / 限制", "", "- 科威尔(688551)、力诺药包(301188)公告列表:新浪公告页未返回,未取得。", "- 涨跌原因的新闻证据:未检索到可靠来源,未编造;东方财富等源未使用。", "- 交易所官方名称列表(上交所/深交所/北交所)接口超时,现名改用腾讯行情接口(qt.gtimg.cn)取得。", "- 行业为巨潮最新分类,非历史时点。", ""]
    (REP / f"daily_ai_report_{date}_v2.md").write_text("\n".join(L), encoding="utf-8"); print("report lines", len(L), "figs", {k: (v[0].name, v[0].stat().st_size, v[1]) for k, v in figs.items()})
if __name__ == "__main__": main(sys.argv[1] if len(sys.argv) > 1 else "20261021")