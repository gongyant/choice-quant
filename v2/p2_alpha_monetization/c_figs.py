# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as Q
plt = Q.setup_cn_font(); R, FIG = Q.RES, Q.FIG; FIG.mkdir(exist_ok=True)
RN = pd.read_csv(R / "c_runs.csv"); SV = pd.read_csv(R / "c_seed_variance.csv"); PR = pd.read_csv(R / "c_paired.csv"); BY = pd.read_parquet(R / "c_by_date.parquet"); OR = pd.read_csv(R / "c_oracle_diag.csv"); PO = pd.read_csv(R / "c_posthoc.csv")
COL = {"T0": "#444444", "T0G": "#8c8c8c", "T1": "#1f77b4", "T2": "#2ca02c", "T3": "#d62728", "T4": "#9467bd", "LAGLGBM": "#ff7f0e"}
NAME = {"T0": "T0 基线", "T0G": "T0G g20几何", "T1": "T1 排序损失", "T2": "T2 多周期MSE", "T3": "T3 多周期排序", "T4": "T4 持续性正则", "LAGLGBM": "滞后堆叠LGBM"}
def save(fig, name):
    p = FIG / name; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); assert p.stat().st_size > 5000, name; print("saved", name, p.stat().st_size)
# 1. seed variance of the baseline and the lag-LGBM (207 signals)
def fig1():
    r = RN[(RN.window == "all") & (RN.variant.isin(["T0", "LAGLGBM"]))]; mets = [("rankic", "RankIC"), ("stability", "相邻信号日分数Spearman(稳定性)"), ("top50_overlap", "Top50相邻重合"), ("turnover_c2", "A0单边换手/期"), ("ann_net_c2", "C2净年化收益"), ("alpha_c2_hac_t", "因子调整alpha HAC t")]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for a, (m, lab) in zip(ax.ravel(), mets):
        for j, v in enumerate(["T0", "LAGLGBM"]):
            x = r[r.variant == v]; a.scatter(np.full(len(x), j) + np.linspace(-.12, .12, len(x)), x[m], color=COL[v], s=45, zorder=3)
            for _, q in x.iterrows(): a.annotate(str(int(q.seed)), (j + .05, q[m]), fontsize=8)
            a.hlines(x[m].mean(), j - .25, j + .25, color=COL[v])
        a.set_xticks([0, 1]); a.set_xticklabels(["Transformer(T0)", "滞后堆叠LGBM"]); a.set_title(lab); a.grid(alpha=.3)
        if m == "ann_net_c2": a.axhline(0, color="k", lw=.6)
    fig.suptitle("图1 种子方差(全207信号;T0的seed42=P1冻结结果;横线=种子均值)"); save(fig, "c_fig1_seed_variance.png")
# 2. variant ladder on the first 104 signals (seed mean with seed range)
def fig2():
    S4 = SV[SV.signals == "104"].set_index("variant"); order = [v for v in ["T0", "T0G", "T1", "T2", "T3", "T4", "LAGLGBM"] if v in S4.index]
    mets = [("rankic", "RankIC"), ("stability", "稳定性(相邻Spearman)"), ("top50_overlap", "Top50重合"), ("turnover_c2", "单边换手/期"), ("net_c2_per_period", "C2净收益/期"), ("alpha_c2_hac_t", "alpha HAC t(C2)")]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    for a, (m, lab) in zip(ax.ravel(), mets):
        for j, v in enumerate(order):
            mu, lo, hi = S4.loc[v, f"{m}_mean"], S4.loc[v, f"{m}_min"], S4.loc[v, f"{m}_max"]; a.errorbar(j, mu, yerr=[[mu - lo], [hi - mu]], fmt="o", color=COL[v], capsize=4)
        a.set_xticks(range(len(order))); a.set_xticklabels([NAME[v] for v in order], rotation=30, ha="right", fontsize=8); a.set_title(lab); a.grid(alpha=.3)
        if m in ("net_c2_per_period", "alpha_c2_hac_t"): a.axhline(0, color="k", lw=.6)
    fig.suptitle("图2 变体对比(前104信号,种子均值与种子极差;探索性,非样本外选型)"); save(fig, "c_fig2_variant_ladder.png")
# 3. forest plot of paired seed-mean differences
def fig3():
    P = PR[(PR.scope == "seed-mean") & (PR.signals.isin(["104"])) & (PR.metric.isin(["ic", "stab", "turn_c2", "net_c2", "alpha_c2"]))]
    labs = {"ic": "RankIC", "stab": "稳定性", "turn_c2": "换手", "net_c2": "C2净收益/期", "alpha_c2": "alpha/期(C2)"}
    pairs = list(dict.fromkeys(zip(P.A, P.B))); fig, ax = plt.subplots(1, 5, figsize=(16, 4.8), sharey=True)
    for a, m in zip(ax, labs):
        for j, (A_, B_) in enumerate(pairs):
            x = P[(P.A == A_) & (P.B == B_) & (P.metric == m)]
            if len(x): x = x.iloc[0]; a.errorbar(x.diff_mean, j, xerr=[[x.diff_mean - x.ci_lo], [x.ci_hi - x.diff_mean]], fmt="o", color="#d62728" if abs(x.hac_t) >= 1.96 else "#555555", capsize=3)
        a.axvline(0, color="k", lw=.7); a.set_title(labs[m]); a.grid(alpha=.3)
    ax[0].set_yticks(range(len(pairs))); ax[0].set_yticklabels([f"{a_}−{b_}" for a_, b_ in pairs]); fig.suptitle("图3 配对差(A−B,种子均值路径,95% block bootstrap CI;红=|HAC t|≥1.96;前104信号)"); save(fig, "c_fig3_paired_forest.png")
# 4. rolling stability by time
def fig4():
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for v in ["T0", "T3", "T4", "LAGLGBM"]:
        x = BY[BY.variant == v]
        if x.empty: continue
        g = x.groupby("k")[["stab", "turn_c2"]].mean().reset_index(); g = g[g.k < 104] if v in ("T3", "T4") else g
        for a, c in zip(ax, ["stab", "turn_c2"]): a.plot(g.k, g[c].rolling(8, min_periods=3).mean(), color=COL[v], label=NAME[v], lw=1.6)
    for a, t in zip(ax, ["相邻信号日分数Spearman(8期滑动均值,种子均值)", "A0单边换手/期(C2,8期滑动均值)"]):
        a.set_title(t); a.grid(alpha=.3)
        for k in range(0, 207, 4): a.axvline(k, color="#dddddd", lw=.4, zorder=0)
    ax[0].legend(ncol=4, fontsize=8); ax[1].set_xlabel("信号序号(浅线=retrain边界)"); fig.suptitle("图4 稳定性与换手随时间"); save(fig, "c_fig4_stability_time.png")
# 5. oracle diagnostic + posthoc smoothing
def fig5():
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    ax[0].plot(OR.noise_ar1_phi, OR.stability, "o-", label="稳定性"); ax[0].plot(OR.noise_ar1_phi, OR.turnover_c2, "s-", label="A0换手"); ax[0].set_title("ORACLE合成信号(仅诊断,含未来信息)\n固定IC信号+AR(1)噪声: 噪声持续性→稳定性/换手"); ax[0].set_xlabel("噪声AR(1)系数 φ"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(OR.noise_ar1_phi, OR.cost_c2_per_period, "d-", color="#d62728"); ax[1].set_title("同上:C2成本/期(与IC水平无关地随换手变化)"); ax[1].set_xlabel("φ"); ax[1].grid(alpha=.3)
    p = PO[PO["diag"].str.contains("EMA") & ~PO["diag"].str.contains("paired") & PO.get("rankic", pd.Series(dtype=float)).notna()]; base = RN[(RN.variant == "T0") & (RN.window == "all")]
    xs = [("T0", base)] + [("T0+EMA\n(事后诊断)", p)]
    for j, (lab, d) in enumerate(xs): ax[2].scatter(np.full(len(d), j), d.turnover_c2, color=["#444444", "#e377c2"][j], s=40); 
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels([x[0] for x in xs]); ax[2].set_title("事后诊断:因果EMA平滑分数→换手(单边/期)"); ax[2].grid(alpha=.3); save(fig, "c_fig5_oracle_posthoc.png")
# 6. cumulative RankIC and net, T0 vs LAGLGBM
def fig6():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for v in ["T0", "LAGLGBM"]:
        x = BY[BY.variant == v]; g = x.groupby("k")[["ic", "net_c2"]].mean()
        ax[0].plot(g.index, g.ic.cumsum(), color=COL[v], label=NAME[v]); ax[1].plot(g.index, (1 + g.net_c2).cumprod(), color=COL[v], label=NAME[v])
    ax[0].set_title("累计RankIC(种子均值)"); ax[1].set_title("C2净值(A0 Top50,种子均值路径)"); [a.legend() or a.grid(alpha=.3) for a in ax]; fig.suptitle("图6 匹配信息集:Transformer vs 滞后堆叠LightGBM(全207信号)"); save(fig, "c_fig6_lag_vs_tf.png")
if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6): f()