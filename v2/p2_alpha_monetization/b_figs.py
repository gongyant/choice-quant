# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
fm.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc"); plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]; plt.rcParams["axes.unicode_minus"] = False
import p2lib as Q
R, FIG = Q.RES, Q.FIG; FIG.mkdir(exist_ok=True)
P = pd.read_csv(R / "b_performance.csv"); P = P[P.window == "eval_k52+"]; EW = pd.read_csv(R / "b_ew_benchmark.csv"); PA = pd.read_csv(R / "b_paired.csv"); TV = pd.read_csv(R / "b_turnover_vs_prediction.csv"); FA = pd.read_csv(R / "b_factor_alpha.csv")
ST = pd.read_csv(R / "b_cost_stress.csv"); MODELS = Q.MODELS; COL = {"A0": "#444444", "A1_nested": "#1f77b4", "A2_nested": "#d62728"}; LAB = {"A0": "A0 朴素Top50", "A1_nested": "A1 缓冲(嵌套选参)", "A2_nested": "A2 成本感知优化(嵌套选参)"}
def save(fig, name):
    p = FIG / name; fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); assert p.stat().st_size > 20000, p; print("saved", name, p.stat().st_size)
def runs(m, s): return pd.read_parquet(R / "b_runs" / f"{m}_{s}.parquet")
# 1 NAV
fig, ax = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
for i, s in enumerate(["C1_flat", "C2_explicit"]):
    for j, m in enumerate(["HGB", "LGBM"]):
        a = ax[i, j]; r = runs(m, s)
        for v in ["A0", "A1_nested", "A2_nested"]:
            x = r[(r.variant == v) & (r.k >= 52)].sort_values("k"); d = pd.to_datetime(EW.signal_date.values[x.k.values]); a.plot(d, (1 + x.ret.values).cumprod(), color=COL[v], label=LAB[v], lw=1.6)
        e = EW[EW.k >= 52]; a.plot(pd.to_datetime(e.signal_date), (1 + e.ew_ret_c0).cumprod(), color="gray", ls="--", lw=1.2, label="等权全样本基准(零成本)")
        a.set_title(f"{m} · {'flat 0.30%往返' if s == 'C1_flat' else '显式价差+冲击成本'}"); a.grid(alpha=.3); a.set_ylabel("净值(5日一期复利, 评估窗k≥52)")
        if i == 0 and j == 0: a.legend(fontsize=8)
fig.suptitle("图1 A0/A1/A2 净值曲线(2025-2026 段为探索性复用)"); save(fig, "b_fig1_nav.png")
# 2 cost decomposition
comp = [("drag_commission_annual", "佣金"), ("drag_transfer_annual", "过户费"), ("drag_stamp_annual", "印花税"), ("drag_spread_annual", "价差(半价差)"), ("drag_impact_annual", "冲击")]
fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for j, m in enumerate(["HGB", "LGBM"]):
    a = ax[j]; xs = []; lab = []; bot = np.zeros(6); k = 0
    cols = ["#9ecae1", "#c6dbef", "#fdae6b", "#74c476", "#e377c2"]; data = []
    for v in ["A0", "A1_nested", "A2_nested"]:
        row = P[(P.scenario == "C2_explicit") & (P.model == m) & (P.variant == v)].iloc[0]; data.append([row[c] for c, _ in comp] + [P[(P.scenario == "C1_flat") & (P.model == m) & (P.variant == v)].iloc[0].drag_flat_annual])
    data = np.array(data); bot = np.zeros(3)
    for q, (c, n) in enumerate(comp): a.bar(np.arange(3) - 0.2, data[:, q], 0.38, bottom=bot, color=cols[q], label=n if j == 0 else None); bot += data[:, q]
    a.bar(np.arange(3) + 0.2, data[:, 5], 0.38, color="#bbbbbb", label="flat 0.30%口径" if j == 0 else None)
    for q in range(3): a.text(q - 0.2, bot[q] + 0.003, f"{bot[q]*100:.1f}%", ha="center", fontsize=9); a.text(q + 0.2, data[q, 5] + 0.003, f"{data[q,5]*100:.1f}%", ha="center", fontsize=9)
    a.set_xticks(range(3)); a.set_xticklabels(["A0", "A1嵌套", "A2嵌套"]); a.set_title(f"{m} 年化成本拖累(左柱=显式分解,右柱=flat)"); a.grid(alpha=.3, axis="y")
ax[0].set_ylabel("年化拖累(占净值,单利)"); ax[0].legend(fontsize=8); fig.suptitle("图2 成本拖累分解"); save(fig, "b_fig2_cost_decomposition.png")
# 3 grid scatter turnover vs net
G = pd.read_csv(R / "b_grid_fixed_params.csv"); G = G[G.window == "eval_k52+"]
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for i, s in enumerate(["C1_flat", "C2_explicit"]):
    for j, m in enumerate(["HGB", "LGBM"]):
        a = ax[i, j]; g = G[(G.scenario == s) & (G.model == m)]; pp = P[(P.scenario == s) & (P.model == m)]
        a1 = g[g.variant.str.startswith("A1")]; a2 = g[g.variant.str.startswith("A2")]
        a.scatter(a1.turnover_oneway_per_period, a1.net_mean_per_period * 100, c="#1f77b4", label="A1 网格(K_hold)", s=40)
        for _, r_ in a1.iterrows(): a.annotate(r_.variant.replace("A1_", ""), (r_.turnover_oneway_per_period, r_.net_mean_per_period * 100), fontsize=7)
        a.scatter(a2.turnover_oneway_per_period, a2.net_mean_per_period * 100, c="#d62728", marker="s", label="A2 网格(τ×cap×δ)", s=30)
        for v, mk_, c_ in [("A0", "*", "k"), ("R_A1", "x", "#6baed6"), ("S_A1", "+", "#6baed6"), ("R_A2", "x", "#fc9272"), ("S_A2", "+", "#fc9272")]:
            r_ = pp[pp.variant == v].iloc[0]; a.scatter(r_.turnover_oneway_per_period, r_.net_mean_per_period * 100, marker=mk_, c=c_, s=120 if v == "A0" else 70, label={"A0": "A0", "R_A1": "随机保留(匹配A1)", "S_A1": "随机跳过(匹配A1)", "R_A2": "随机保留(匹配A2)", "S_A2": "随机跳过(匹配A2)"}[v])
        a.set_title(f"{m} · {'flat' if s == 'C1_flat' else '显式成本'}"); a.set_xlabel("单边换手/期"); a.set_ylabel("净收益均值 %/5日"); a.grid(alpha=.3)
        if i == 0 and j == 0: a.legend(fontsize=7)
fig.suptitle("图3 换手 vs 净收益: 固定参数网格与匹配换手随机对照(评估窗k≥52)"); save(fig, "b_fig3_turnover_vs_net.png")
# 4 attribution
fig, ax = plt.subplots(2, 2, figsize=(15, 8), sharey="row")
for i, s in enumerate(["C1_flat", "C2_explicit"]):
    for j, (tag, nm) in enumerate([("R", "对照=随机保留(已校准至同换手)"), ("S", "对照=随机跳过(已校准至同换手)")]):
        a = ax[i, j]; t = TV[TV.scenario == s].copy(); t["lab"] = t.model + "\n" + t.family; t = t.sort_values(["family", "model"]); x = np.arange(len(t))
        a.bar(x, t[f"turnover_effect_{tag}_adj"] * 100, color="#9ecae1", label="换手效应(控制组-A0)"); a.bar(x, t[f"selection_effect_{tag}_adj"] * 100, bottom=t[f"turnover_effect_{tag}_adj"] * 100, color="#fb6a4a", label="选股效应(策略-控制组)")
        a.scatter(x, t.delta_net * 100, c="k", zorder=3, s=22, label="总改进 Δ净"); a.axhline(0, color="k", lw=.6); a.set_xticks(x); a.set_xticklabels(t.lab, fontsize=7); a.set_title(f"{s} · {nm}"); a.grid(alpha=.3, axis="y")
        if i == 0 and j == 0: a.legend(fontsize=8)
        if j == 0: a.set_ylabel("净收益Δ %/5日(相对A0)")
fig.suptitle("图4 相对A0的净改进: 换手节省 vs 选股成分"); save(fig, "b_fig4_attribution.png")
# 5 paired forest
fig, ax = plt.subplots(1, 2, figsize=(14, 7), sharey=True); cmps = ["A1_nested - A0", "A2_nested - A0", "A2_nested - A1_nested"]; cc = ["#1f77b4", "#d62728", "#2ca02c"]
for j, s in enumerate(["C1_flat", "C2_explicit"]):
    a = ax[j]; y = 0; yt = []; yl = []
    for m in MODELS:
        for q, c in enumerate(cmps):
            r_ = PA[(PA.scenario == s) & (PA.model == m) & (PA.comparison == c)].iloc[0]; a.errorbar(r_.net_diff_mean * 100, y, xerr=[[ (r_.net_diff_mean - r_.net_ci_lo) * 100], [(r_.net_ci_hi - r_.net_diff_mean) * 100]], fmt="o", color=cc[q], capsize=2, label=c if (m == MODELS[0]) else None); y += 1
        yt.append(y - 2); yl.append(m); y += 0.6
    a.axvline(0, color="k", lw=.7); a.set_yticks(yt); a.set_yticklabels(yl); a.set_title(f"{s}"); a.set_xlabel("净收益差 %/5日 (block bootstrap 95% CI)"); a.grid(alpha=.3, axis="x"); a.invert_yaxis()
ax[0].legend(fontsize=8); fig.suptitle("图5 配对差异(评估窗k≥52, 嵌套选参)"); save(fig, "b_fig5_paired_forest.png")
# 6 factor alpha t
fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for j, s in enumerate(["C1_flat", "C2_explicit"]):
    a = ax[j]
    for q, v in enumerate(["A0", "A1_nested", "A2_nested"]):
        t = FA[(FA.scenario == s) & (FA.variant == v)].set_index("model").loc[MODELS]; a.bar(np.arange(6) + (q - 1) * 0.27, t.alpha_hac_t, 0.27, color=COL[v], label=LAB[v])
    a.axhline(1.96, color="gray", ls="--", lw=.8); a.axhline(-1.96, color="gray", ls="--", lw=.8); a.set_xticks(range(6)); a.set_xticklabels(MODELS); a.set_title(f"{s}: 五因子调整后净alpha的HAC t"); a.grid(alpha=.3, axis="y")
ax[0].legend(fontsize=8); fig.suptitle("图6 因子调整净 alpha t 值(SMB/REV5/LVOL/MOM60/ILLIQ, 超额=策略-等权基准)"); save(fig, "b_fig6_factor_alpha_t.png")
# 7 scenarios
SC = ["C2_explicit", "C3_realistic", "C4_capacity_1e8"]; fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for j, (col, tt) in enumerate([("ann_net", "年化净收益"), ("sharpe_net", "净夏普")]):
    a = ax[j]; w = 0.26
    for q, v in enumerate(["A0", "A1_nested", "A2_nested"]):
        for u, s in enumerate(SC):
            t = P[(P.scenario == s) & (P.variant == v)].set_index("model").loc[MODELS]; a.bar(np.arange(6) + (q - 1) * 0.28 + 0, t[col] * (100 if col == "ann_net" else 1), 0.09, color=COL[v], alpha=[1, .65, .35][u], label=f"{LAB[v].split(' ')[0]}·{s}" if True else None, left=None) if False else None
    for q, v in enumerate(["A0", "A1_nested", "A2_nested"]):
        for u, s in enumerate(SC):
            t = P[(P.scenario == s) & (P.variant == v)].set_index("model").loc[MODELS]; a.bar(np.arange(6) + (q * 3 + u - 4) * 0.095, t[col] * (100 if col == "ann_net" else 1), 0.09, color=COL[v], alpha=[1, .65, .35][u], label=f"{v.split('_')[0]}·{s[:2]}")
    a.set_xticks(range(6)); a.set_xticklabels(MODELS); a.set_title(tt + "(C2显式/C3现实约束/C4容量1亿)"); a.grid(alpha=.3, axis="y"); a.axhline(0, color="k", lw=.6)
ax[0].legend(fontsize=6, ncol=3); ax[0].set_ylabel("%"); fig.suptitle("图7 现实流动性/准入约束与容量: C2→C3→C4"); save(fig, "b_fig7_constraints_capacity.png")
# 8 stress
fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True); order = ["hs0.5", "base", "hs2", "imp0.5", "imp2"]; nm = {"hs0.5": "半价差×0.5", "base": "基准", "hs2": "半价差×2", "imp0.5": "冲击×0.5", "imp2": "冲击×2"}
for j, m in enumerate(["HGB", "LGBM"]):
    a = ax[j]; st = ST[ST.model == m]
    for q, v in enumerate(["A0", "A1_nested", "A2_nested"]):
        vals = []
        for o in order:
            if o == "base": vals.append(P[(P.scenario == "C2_explicit") & (P.model == m) & (P.variant == v)].iloc[0].net_mean_per_period)
            else: vals.append(st[(st.strategy == v) & (st.stress == o)].iloc[0].net_mean)
        a.bar(np.arange(5) + (q - 1) * 0.27, np.array(vals) * 100, 0.27, color=COL[v], label=LAB[v])
    a.set_xticks(range(5)); a.set_xticklabels([nm[o] for o in order], fontsize=8); a.set_title(m); a.grid(alpha=.3, axis="y"); a.axhline(0, color="k", lw=.6)
ax[0].set_ylabel("净收益均值 %/5日"); ax[0].legend(fontsize=8); fig.suptitle("图8 成本假设敏感性(显式口径 C2)"); save(fig, "b_fig8_cost_stress.png")