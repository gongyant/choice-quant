# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, json
import p2lib as Q
plt = Q.setup_cn_font(); R = Q.RES; F = Q.FIG
MOD = Q.MODELS; COL = dict(zip(MOD, ["#1f77b4", "#ff7f0e", "#7f7f7f", "#2ca02c", "#d62728", "#9467bd"]))
def save(fig, name):
    fig.tight_layout(); fig.savefig(F / name, dpi=130); plt.close(fig)
def grouped(ax, df, cat_col, val="mean", lo="ci_lo", hi="ci_hi", models=MOD, cats=None):
    cats = cats or list(df[cat_col].unique()); w = 0.8 / len(models)
    for k, m in enumerate(models):
        d = df[df.model == m].set_index(cat_col).reindex(cats); x = np.arange(len(cats)) + k * w - 0.4 + w / 2
        ax.bar(x, d[val], w, color=COL.get(m, None), label=m, yerr=[d[val] - d[lo], d[hi] - d[val]], capsize=2, error_kw=dict(lw=0.8))
    ax.set_xticks(np.arange(len(cats))); ax.set_xticklabels(cats, rotation=0, fontsize=8); ax.axhline(0, color="k", lw=0.6)
# 1 IC by liquidity (+vol, exchange)
G = pd.read_csv(R / "rankic_by_group.csv")
fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, gv, tt in zip(axs, ["流动性(amt20 日内三分位)", "流动性(amt20 绝对档)"], ["按日内 amt20 三分位", "按 amt20 绝对档(元)"]):
    d = G[G.group_var == gv]; grouped(ax, d, "group"); ax.set_title(f"RankIC 分流动性组:{tt}(新口径;误差线=块自助95%CI)"); ax.set_ylabel("平均 RankIC")
axs[0].legend(ncol=3, fontsize=8); save(fig, "ic_by_liquidity.png")
fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, gv, tt in zip(axs, ["波动率(vol20 日内三分位)", "交易所"], ["按 vol20 三分位", "按交易所(北交所日均仅~35只)"]):
    d = G[G.group_var == gv]; grouped(ax, d, "group"); ax.set_title(f"RankIC 分组:{tt}"); ax.set_ylabel("平均 RankIC")
axs[0].legend(ncol=3, fontsize=8); save(fig, "ic_by_vol_exchange.png")
ind = G[(G.group_var.str.startswith("行业")) & G.model.isin(["HGB", "LGBM", "GRU"])]
fig, ax = plt.subplots(figsize=(13, 4.8)); cats = sorted(ind.group.unique()); grouped(ax, ind, "group", models=["HGB", "LGBM", "GRU"], cats=cats)
ax.set_title("RankIC 分行业(cninfo【最新】行业分类,非时点,仅作描述)"); ax.legend(); ax.tick_params(axis="x", rotation=30); save(fig, "ic_by_industry_latest.png")
# 2 deciles
D = pd.read_csv(R / "decile_means.csv")
fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
for ax, m in zip(axs.ravel(), MOD):
    for d, c, lab in [("new", COL[m], "新口径(下一可执行日)"), ("old_all", "#bbbbbb", "旧口径(t+1→t+6)")]:
        x = D[(D.model == m) & (D.definition == d)]; ax.bar(x.decile + (0.2 if d == "new" else -0.2), x.mean_rel_ret * 100, 0.4, color=c, label=lab)
    ax.axhline(0, color="k", lw=0.6); ax.set_title(f"{m} 十分位(D10=分数最高)"); ax.set_xticks(range(1, 11))
axs[0, 0].set_ylabel("5日相对收益均值(%)"); axs[1, 0].set_ylabel("5日相对收益均值(%)"); axs[0, 0].legend(fontsize=8); save(fig, "decile_returns.png")
# 3 partial IC
PI = pd.read_csv(R / "partial_rankic.csv"); sets = list(PI.control_set.unique())
fig, axs = plt.subplots(1, 2, figsize=(15, 4.8))
raw = PI[PI.control_set == sets[0]].groupby("model").raw_ic_same_sample_mean.first().reindex(MOD)
d0 = PI[PI.control_set == sets[3]]
grouped(axs[0], PI[PI.control_set.isin([sets[0], sets[1], sets[3], sets[4]])], "control_set", cats=[sets[0], sets[1], sets[3], sets[4]]); axs[0].set_title("偏 RankIC(控制集逐步加码;误差线=95%CI)"); axs[0].tick_params(axis="x", labelsize=7); axs[0].legend(ncol=3, fontsize=8)
axs[0].set_xticklabels([s.replace(" ", "\n") for s in [sets[0], sets[1], sets[3], sets[4]]], fontsize=7)
x = np.arange(6); axs[1].bar(x - 0.2, raw.values, 0.4, label="原始 RankIC", color="#bbbbbb"); axs[1].bar(x + 0.2, d0.set_index("model").reindex(MOD)["mean"].values, 0.4, label="四项全控后偏 RankIC", color="#1f77b4")
axs[1].set_xticks(x); axs[1].set_xticklabels(MOD); axs[1].set_title("原始 vs 四项全控(r5,r20,vol20,turn20,amihud20)"); axs[1].legend(); axs[1].axhline(0, color="k", lw=0.6); save(fig, "partial_ic.png")
# 4 factor attribution
A = pd.read_csv(R / "factor_attribution.csv"); a5 = A[(A.spec == "5因子") & (A.dependent == "D10-D1多空价差")]; a0 = A[(A.spec == "无因子(均值)") & (A.dependent == "D10-D1多空价差")]
names = list(a5.model); fig, axs = plt.subplots(1, 3, figsize=(17, 5))
x = np.arange(len(names)); a5 = a5.set_index("model").loc[names]; a0 = a0.set_index("model").loc[names]
axs[0].bar(x - 0.2, a0.alpha * 100, 0.4, color="#bbbbbb", label="未调整 D10-D1 均值"); axs[0].bar(x + 0.2, a5.alpha * 100, 0.4, color="#1f77b4", label="5因子调整后 alpha", yerr=[(a5.alpha - a5.alpha_ci_lo) * 100, (a5.alpha_ci_hi - a5.alpha) * 100], capsize=2)
axs[0].set_xticks(x); axs[0].set_xticklabels(names, rotation=40, ha="right", fontsize=7); axs[0].set_ylabel("每5日 %"); axs[0].set_title("D10-D1 多空价差:调整前后"); axs[0].legend(fontsize=8); axs[0].axhline(0, color="k", lw=0.6)
axs[1].bar(x, a5.alpha_hac_t, color=["#1f77b4" if m in MOD else "#999999" for m in names]); axs[1].axhline(1.96, color="r", ls="--", lw=0.8); axs[1].axhline(-1.96, color="r", ls="--", lw=0.8)
axs[1].set_xticks(x); axs[1].set_xticklabels(names, rotation=40, ha="right", fontsize=7); axs[1].set_title("alpha 的 HAC(4) t 值(灰=基线)")
axs[2].bar(x, a5.r2, color=["#2ca02c" if m in MOD else "#999999" for m in names]); axs[2].set_xticks(x); axs[2].set_xticklabels(names, rotation=40, ha="right", fontsize=7); axs[2].set_title("5因子回归 R²(基线应≈1:检验归因方法)")
save(fig, "factor_attribution.png")
# 5 old vs new, baselines
T = pd.read_csv(R / "rankic_old_vs_new.csv")
fig, axs = plt.subplots(1, 2, figsize=(15, 4.8))
for k, (d, lab) in enumerate([("old_all", "旧口径 t+1→t+6"), ("old_exec_t1", "旧口径·仅t+1可买"), ("new", "新口径 下一可执行日")]):
    x = T[(T.definition == d) & T.model.isin(MOD)].set_index("model").reindex(MOD); xx = np.arange(6) + (k - 1) * 0.27
    axs[0].bar(xx, x["mean"], 0.27, label=lab, yerr=[x["mean"] - x.ci_lo, x.ci_hi - x["mean"]], capsize=2)
axs[0].set_xticks(range(6)); axs[0].set_xticklabels(MOD); axs[0].legend(fontsize=8); axs[0].set_title("六模型 RankIC:新旧口径(95%块自助CI)"); axs[0].set_ylim(0, 0.12)
x = T[T.definition == "new"].set_index("model"); order = MOD + [m for m in x.index if m not in MOD]
cc = ["#1f77b4"] * 6 + ["#999999"] * (len(order) - 6)
axs[1].bar(range(len(order)), x.loc[order, "mean"], color=cc, yerr=[x.loc[order, "mean"] - x.loc[order, "ci_lo"], x.loc[order, "ci_hi"] - x.loc[order, "mean"]], capsize=2)
axs[1].set_xticks(range(len(order))); axs[1].set_xticklabels(order, rotation=40, ha="right", fontsize=7); axs[1].set_title("模型(蓝) vs 简单基线(灰) RankIC(新口径)"); save(fig, "rankic_old_vs_new_and_baselines.png")
# 7 subsets
S = pd.read_csv(R / "subset_robustness.csv"); subs = list(S.subset.unique())
fig, axs = plt.subplots(1, 2, figsize=(16, 5.2))
grouped(axs[0], S[S.model.isin(["HGB", "LGBM", "GRU", "TF"])], "subset", models=["HGB", "LGBM", "GRU", "TF"], cats=subs); axs[0].set_title("子集稳健性:RankIC(新口径)"); axs[0].legend()
d = S[S.model.isin(["HGB", "LGBM", "GRU", "TF"])].copy()
w = 0.2
for k, m in enumerate(["HGB", "LGBM", "GRU", "TF"]):
    x = d[d.model == m].set_index("subset").reindex(subs); axs[1].bar(np.arange(len(subs)) + k * w - 0.3, x.D10_mean * 100, w, color=COL[m], label=m)
axs[1].axhline(0, color="k", lw=0.6); axs[1].set_title("子集内 D10(多头十分位)相对池均值,每5日 %(扣费前)"); axs[1].set_ylabel("%")
for ax in axs: ax.set_xticks(range(len(subs))); ax.set_xticklabels([s.replace(":", ":\n").replace("(", "\n(") for s in subs], rotation=40, ha="right", fontsize=6)
save(fig, "subset_robustness.png")
# 8 half-year spreads
H = pd.read_csv(R / "decile_spread_by_period.csv"); H = H[(H.definition == "new") & (H.level == "half")]
fig, ax = plt.subplots(figsize=(12, 4.6))
for m in MOD + ["3F(REV5+LOWVOL+ILLIQ)"]:
    x = H[H.model == m]; ax.plot(x.period, x.spread_mean * 100, marker="o", label=m, color=COL.get(m, "k"), ls="-" if m in MOD else "--")
ax.axhline(0, color="k", lw=0.6); ax.set_title("D10-D1 价差分半年(每5日 %,新口径;2026H2 仅 2026-07~09 的信号)"); ax.legend(ncol=4, fontsize=8); save(fig, "decile_spread_by_halfyear.png")
# check
rep = {}
for p in sorted(F.glob("*.png")): rep[p.name] = p.stat().st_size
assert all(v > 20000 for v in rep.values()), rep
json.dump(rep, open(F / "_figure_check.json", "w"), indent=1); print(rep)
import matplotlib.font_manager as fm; print([f.name for f in fm.fontManager.ttflist if "Microsoft YaHei" in f.name][:2])