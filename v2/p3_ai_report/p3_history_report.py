# -*- coding: utf-8 -*-
"""P3-B2: reports/history_vs_actual_report.md + reports/model_reliability_summary.md + figures/hist_*.png. Reads results/hist_* (from p3_v2.py)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
H = Path(__file__).resolve().parent; R = H / "results"; FIG = H / "figures"; REP = H / "reports"
FP = "C:/Windows/Fonts/msyh.ttc"; fm.fontManager.addfont(FP); plt.rcParams["font.family"] = fm.FontProperties(fname=FP).get_name(); plt.rcParams["axes.unicode_minus"] = False
pc = lambda x, k=1: "—" if pd.isna(x) else f"{x:+.{k}%}"; pp = lambda x: "—" if pd.isna(x) else f"{x:.0%}"
def main():
    PR = pd.read_csv(R / "hist_top5_by_period.csv", encoding="utf-8-sig"); S = pd.read_csv(R / "hist_top5_summary.csv", encoding="utf-8-sig"); PK = pd.read_csv(R / "hist_top5_picks.csv", dtype={"code": str}, encoding="utf-8-sig")
    Y = pd.read_csv(R / "hist_bins_by_year.csv", encoding="utf-8-sig", dtype={"year": str}); CC = pd.read_csv(R / "hist_v2_constraint_check.csv", encoding="utf-8-sig"); SU = pd.read_csv(R / "calibration_supplement.csv", encoding="utf-8-sig")
    U = pd.read_csv(H.parent / "data" / "sample_1500.csv", dtype=str).set_index("code").name; CN = pd.read_csv(H / "data" / "current_names.csv", dtype={"code": str}, encoding="utf-8-sig").set_index("code").name_now
    nm = lambda c: CN.get(c, U.get(c, ""))
    z = np.load(H / "data" / "world_20261021.npz"); cal = pd.DatetimeIndex(z["cal_full"]); A = z["ADJf_full"]; col = {c: j for j, c in enumerate(z["codes"])}
    # fig 1 cumulative
    fg, ax = plt.subplots(figsize=(11, 5))
    pool = PR[PR.strategy == "v1"].set_index("signal_date").pool_ret5; x = pd.to_datetime(pool.index)
    ax.plot(x, np.cumprod(1 + pool.values) - 1, color="k", lw=1.5, label="全池等权平均(成本前)")
    for st, c in [("v1", "#d62728"), ("v2", "#1f77b4"), ("F3基线", "#2ca02c")]:
        g = PR[PR.strategy == st].set_index("signal_date"); ax.plot(x, np.cumprod(1 + np.where(g.n_picks > 0, g.ret5, 0)) - 1, color=c, lw=1.3, label=f"{st} Top5 成本前")
        ax.plot(x, np.cumprod(1 + g.ret5_net_30bp.values) - 1, color=c, lw=1, ls="--", label=f"{st} Top5 扣 0.30% 往返成本")
    ax.axvline(pd.Timestamp("2025-01-01"), color="gray", ls=":"); ax.text(pd.Timestamp("2025-01-15"), ax.get_ylim()[1] * 0.9, "2025 起:探索期(非盲测)", fontsize=9, color="gray")
    ax.axhline(0, color="gray", lw=0.6); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3); ax.set_title("历史逐期 Top5(每周一期、持有 5 个交易日)累计收益;无候选的周期视为空仓")
    fg.tight_layout(); fg.savefig(FIG / "hist_cumulative.png", dpi=110); plt.close(fg)
    # fig 2 bins by year
    fg, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, hz in zip(axs, ["5d", "20d"]):
        for y, g in Y[Y.horizon == hz].groupby("year"): a.plot(g.bin, g.mean_rel, "-o", ms=3, label=y)
        a.axhline(0, color="k", lw=0.6); a.set_xlabel("综合分数组(1=最低,10=最高)"); a.set_title(f"{hz} 平均相对池均值收益(分年度)"); a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); a.grid(alpha=0.3); a.legend(fontsize=8)
    fg.tight_layout(); fg.savefig(FIG / "hist_bins_by_year.png", dpi=110); plt.close(fg)
    # examples: v1 picks, 3 best + 3 worst by 5d relative return (one per code)
    v1 = PK[PK.strategy == "v1"].dropna(subset=["realized_rel_ret_5d"]).copy(); v1["rank_pct"] = v1.composite
    best = v1.sort_values("realized_rel_ret_5d", ascending=False).drop_duplicates("code").head(3); worst = v1.sort_values("realized_rel_ret_5d").drop_duplicates("code").head(3)
    mid = v1.iloc[(v1.realized_rel_ret_5d - v1.realized_rel_ret_5d.median()).abs().argsort()[:1]]
    EX = pd.concat([best.assign(kind="好例子"), mid.assign(kind="普通例子"), worst.assign(kind="坏例子")]); EX["name"] = EX.code.map(nm)
    fg, axs = plt.subplots(2, 4, figsize=(15, 7)); axs = axs.ravel()
    for a, (_, e) in zip(axs, EX.iterrows()):
        i = cal.get_loc(pd.Timestamp(e.signal_date)); j = col[e.code]; s = A[i - 20:i + 22, j] / A[i + 1, j] - 1; xx = np.arange(-20, len(s) - 20)
        a.plot(xx, s, color="#444"); a.axvspan(1, 6, color="#ffbf00", alpha=0.25); a.axvline(0, color="r", ls="--", lw=0.8); a.axhline(0, color="gray", lw=0.5)
        a.set_title(f"{e.kind}:{e.code} {e['name']}\n{e.signal_date} Top5 第{int(e.rank_in_pick)},5日相对 {e.realized_rel_ret_5d:+.1%}", fontsize=9); a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); a.grid(alpha=0.3)
    axs[-1].axis("off"); axs[-1].text(0, 0.5, "红虚线 = 信号日\n黄色 = 持有的 5 个交易日\n纵轴 = 相对买入日(t+1)收盘的涨跌\n(前复权,事后实现)", fontsize=10)
    fg.tight_layout(); fg.savefig(FIG / "hist_examples.png", dpi=110); plt.close(fg)
    for f in ["hist_cumulative.png", "hist_bins_by_year.png", "hist_examples.png"]: assert (FIG / f).stat().st_size > 20000
    g = lambda st, pe: S[(S.strategy == st) & (S.period == pe)].iloc[0]
    L = ["# 历史预测 vs 真实涨跌(P3 第二批)", "", "> 研究复盘,不是买入推荐。**2025–2026 年不是盲测**:这段数据在 P1/P2 已被反复查看,只能算探索性复核;2022-07~2024-12 也属模型的走步样本外(walk-forward OOS),但规则设计时研究者已经见过 P1/P2 的汇总结果。所有收益为事后实现值,分数不是概率。", "",
         "## 1. 先说结论", "",
         f"- 207 个历史信号日(2022-07-01 ~ 2026-09-28,约每周一期)里,综合分数越高的组,之后 5 日和 20 日的**平均**相对表现越好,但差距很小:最高组 5 日平均比池子多 {SU[(SU.horizon=='5d')&(SU.bin==10)].mean_rel.iloc[0]:+.2%},最低组少 {abs(SU[(SU.horizon=='5d')&(SU.bin==1)].mean_rel.iloc[0]):.2%}。",
         f"- 按 v1 规则每期选出的 Top5,全期平均每期 5 日收益 {pc(g('v1','全期').mean_ret5,2)},同期全池平均 {pc(g('v1','全期').mean_pool5,2)};跑赢全池的期数占比(命中率)只有 {pp(g('v1','全期').hit_beat_pool)}。**累计(成本前){pc(g('v1','全期').cum_gross)},同期全池 {pc(g('v1','全期').cum_pool)};扣 0.30% 往返成本后 {pc(g('v1','全期').cum_net_30bp)}。**",
         f"- v2(收紧趋势、防追高)与 v1 差别很小:命中率 {pp(g('v2','全期').hit_beat_pool)},累计成本前 {pc(g('v2','全期').cum_gross)},扣成本后 {pc(g('v2','全期').cum_net_30bp)}。三因子(5 日反转+低波动+非流动性)基线 Top5 更差:成本前 {pc(g('F3基线','全期').cum_gross)}。",
         "- 一句话:**模型排序有一点点统计信息,但按规则挑 5 只股票,历史上没有稳定跑赢“随便买一篮子”,扣掉成本后是亏的。**", "",
         "## 2. 综合分数各组的实际表现(分年度)", "", "![分年度](../figures/hist_bins_by_year.png)", "",
         "下表:每组的 5 日相对池均值收益 平均 / 中位 / 跑赢当天池中位数比例 / 跑输(< −2%)比例。", ""]
    for hz in ["5d", "20d"]:
        L += [f"### {hz}", "", "| 年份 | 第1组 平均/中位/跑赢中位/跑输 | 第5组 | 第10组 |", "|---|---|---|---|"]
        for y, gg in Y[Y.horizon == hz].groupby("year"):
            f = lambda b: (lambda r: f"{r.mean_rel:+.2%} / {r.median_rel:+.2%} / {r.beat_median:.0%} / {r.under_share:.0%}")(gg[gg.bin == b].iloc[0])
            L.append(f"| {y}{'(探索期)' if y >= '2025' else ''} | {f(1)} | {f(5)} | {f(10)} |")
        L.append("")
    L += [f"20 日的跑输阈值为 −5%。2022 年只有下半年。", "", "## 3. 历史每期按规则选出的 Top5", "",
          "做法:每个历史信号日用当天已知的模型分数和行情,按 v1 / v2 规则挑最多 5 只(满足条件不足 5 只就少挑,没有就空仓),等权持有 5 个交易日(t+1 收盘买、t+6 收盘卖,P0 口径)。历史时点的 ST 状态拿不到,所以历史回放**没有**用 ST 过滤。对比:全池等权平均、三因子 F3 基线(同一流动性条件下 F3 最高的 5 只)。成本:往返 0.30%(P0/P2 的 flat 口径)和 0.60%(压力情形)。P2 的更细成本模型下结论更差。", "",
          "![累计](../figures/hist_cumulative.png)", "", "| 方案 | 期间 | 期数 | 有候选期数 | 平均只数 | 每期平均5日收益 | 同期全池 | 平均超额 | 命中率(跑赢全池) | 个股跑赢池中位数 | 累计成本前 | 累计全池 | 扣0.30% | 扣0.60% |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in S.sort_values(["period", "strategy"]).iterrows():
        L.append(f"| {r.strategy} | {r.period} | {int(r.n_periods)} | {int(r.n_periods_with_picks)} | {r.avg_picks:.1f} | {pc(r.mean_ret5,2)} | {pc(r.mean_pool5,2)} | {pc(r.mean_excess5,2)} | {pp(r.hit_beat_pool)} | {pp(r.pick_beat_median_share)} | {pc(r.cum_gross)} | {pc(r.cum_pool)} | {pc(r.cum_net_30bp)} | {pc(r.cum_net_60bp)} |")
    L += ["", "注意:设计期 v1/v2 很多周没有任何股票同时满足“5 个模型都看好、分歧小、成交额≥1 亿”等条件(122 期里只有约 50 期有候选),空仓期收益记 0,所以设计期的累计收益和全池不完全可比。", "",
          "### 3.1 逐期表(最近 30 期;全部 207 期见 `results/hist_top5_by_period.csv`,逐只明细见 `results/hist_top5_picks.csv`)", "", "| 信号日 | v1 只数 | v1 5日 | v2 只数 | v2 5日 | F3 5日 | 全池 5日 |", "|---|---:|---:|---:|---:|---:|---:|"]
    W = PR.pivot(index="signal_date", columns="strategy", values=["n_picks", "ret5"]); pl = PR.groupby("signal_date").pool_ret5.first()
    for sd in W.index[-30:]:
        L.append(f"| {sd} | {int(W.loc[sd, ('n_picks','v1')])} | {pc(W.loc[sd, ('ret5','v1')]) if W.loc[sd, ('n_picks','v1')] else '空仓'} | {int(W.loc[sd, ('n_picks','v2')])} | {pc(W.loc[sd, ('ret5','v2')]) if W.loc[sd, ('n_picks','v2')] else '空仓'} | {pc(W.loc[sd, ('ret5','F3基线')])} | {pc(pl[sd])} |")
    L += ["", "## 4. v2 每条新约束在历史上有没有用(5 日,v1 通过的股票中 被排除 vs 保留)", "", "| 期间 | 约束 | 被排除 n / 跑输比例 / 平均相对 | 保留 n / 跑输比例 / 平均相对 |", "|---|---|---|---|"]
    c5 = CC[CC.horizon == "5d"]
    for (pe, cn), gg in c5.groupby(["period", "constraint"], sort=False):
        e, k = gg[gg.side == "被排除"].iloc[0], gg[gg.side == "保留"].iloc[0]; fx = lambda r: f"{int(r.n)} / {pp(r.under_share)} / {pc(r.mean_rel,2)}" if r.n else "0(从未触发)"
        L.append(f"| {pe} | {cn} | {fx(e)} | {fx(k)} |")
    L += ["", "**解读**:T1(下跌趋势排除)在设计期只触发 3 次(都跑输),样本太少无法证明有用;在 2025–2026 探索期触发 215 次,被排除的股票反而**平均多赚 1.7%**、跑输比例更低——也就是说历史上“模型看好的下跌股”短期反弹更多。T2(20 日涨幅>20%)触发少、差别小;T3(5 日内≥2 次涨停)在 v1 已有的过滤下从未触发。**按事先约定,规则不因检验结果改动**;v2 的价值主要是风险控制(避免在趋势向下时抄底、避免追高),不是提高收益,而且历史上它可能**少赚**反弹。", "",
          "## 5. 真实例子(v1 规则历史选出的股票,好坏都有)", "", "![例子](../figures/hist_examples.png)", "", "| 类型 | 信号日 | 代码 | 名称(现名) | 当时 Top5 名次 | 综合分位 | 当时 20 日涨跌 | 之后 5 日涨跌 | 5 日相对池均值 | 之后 20 日涨跌 |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, e in EX.iterrows(): L.append(f"| {e.kind} | {e.signal_date} | {e.code} | {e['name']} | {int(e.rank_in_pick)} | {pp(e.composite)} | {pc(e.r20)} | {pc(e.realized_ret_5d)} | {pc(e.realized_rel_ret_5d)} | {pc(e.ret_20d)} |")
    L += ["", "例子是按事后结果挑的(最好 3 个、最差 3 个、1 个接近中位数),只用来说明“单只股票结果非常分散”,不代表平均水平。", "",
          "## 6. 限制", "", "- 2025–2026 不是盲测;2022-07 之后整体都已在 P1/P2 中被查看过。", "- 历史回放没有 ST 过滤(历史名称不可得),现行规则会更严。", "- 成本只用了简化的往返 0.30%/0.60%;P2 的显式成本(含价差、冲击)下,Top50 净年化已为负。", "- 每周一期、5 日持有,周与周之间不重叠;20 日表现有重叠,只作参考。", ""]
    (REP / "history_vs_actual_report.md").write_text("\n".join(L), encoding="utf-8")
    v = g("v1", "全期"); su10 = SU[(SU.horizon == "5d") & (SU.bin == 10)].iloc[0]
    M = ["# 模型到底准不准?(一页白话总结)", "", "> 研究结论,不是投资建议。截至 2026-10-21。", "",
         "**能不能用来做实际投资决策?——不能单独依赖。** 它最多是一个“初筛参考”,而且历史上扣掉交易成本后不赚钱。", "",
         "## 它有一点用的地方", f"- 排序方向是对的:历史 207 期里,综合分高的组平均表现略好(5 日 RankIC 约 0.08,约 74% 的周为正)。", f"- 高分股的主要特点是**少大跌**:最高组 5 日跑输池均值超过 2% 的比例 28%,全体 35%;跑赢当天池中位数的比例 {su10.share_beat_pool_median:.0%}(随机是 50%)。", "",
         "## 它不靠谱的地方", f"- **差距太小**:最高组平均每 5 天只比池子多约 {su10.mean_rel:.2%},而一次买卖的成本约 0.3%(P2 口径),优势基本被吃掉。",
         f"- **挑 5 只并不稳**:按规则历史逐期选出的 Top5,只有 {pp(v.hit_beat_pool)} 的周跑赢全池平均;累计成本前 {pc(v.cum_gross)},不如全池 {pc(v.cum_pool)};扣 0.30% 成本后 {pc(v.cum_net_30bp)}。",
         "- **单只股票几乎没法预测**:最高组的 5 日涨跌 80% 区间约 −6%~+7%,和普通股票差不多;模型分数**不是**上涨概率。",
         "- **P2 成本结论**:Top50 组合成本前年化约 15%,按显式成本(佣金、印花税、价差、冲击)计算后年化约 −3%;加上真实账户约束(不碰科创/北交所、成交额≥1 亿)后约 −10%。没有一种组合在成本和真实约束下显示出可交易的净超额。",
         "- 2025–2026 的结果已被反复查看,**没有真正的盲测**;要判断模型是否真的有效,只能看从 2026-10-21 开始登记、不再修改的前向预测(`predictions_registry/`,用 `p3_verify_forward.py` 核验)。", "",
         "## 如果还是想参考,怎么用才稳妥", "- 只把它当“观察名单”,结合公司基本面、公告自己判断;不要按分数重仓或频繁换股。", "- 关注“回避名单”的风险提示(ST、连续跌停、流动性差),这部分比“选牛股”更有实际意义。", "- 每周用前向核验看真实命中情况;连续几个月都无法跑赢全池,就说明信号不可用。", ""]
    (REP / "model_reliability_summary.md").write_text("\n".join(M), encoding="utf-8"); print("ok", len(L), len(M)); print(EX[["kind", "signal_date", "code", "name", "realized_rel_ret_5d"]].to_string())
if __name__ == "__main__": main()