"""E0 / E1 / E3 comparison: old (INVALID_BEFORE_FIX) vs fixed, net excess vs same-regime equal weight, t / HAC / block bootstrap, cost sensitivity, delisted-group diagnostics."""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import p0lib as L
R = L.P0 / "results"; OUT = L.P0; S5 = ["Ridge", "HGB", "LGBM", "Momentum20", "Reversal5"]; ALL = S5 + ["EqualWeight"]
OLD = pd.read_csv(L.V2 / "results" / "baseline" / "walk_forward_returns.csv")
def stats_block(ex):
    ex = pd.Series(ex).dropna().values; t, p = L.tstat(ex); t4, p4 = L.nw_t(ex, 4); t8, p8 = L.nw_t(ex, 8); lo, hi = L.block_boot(ex)
    return dict(excess_mean=float(ex.mean()), t=t, p=p, hac4_t=t4, hac4_p=p4, hac8_t=t8, hac8_p=p8, boot_lo=lo, boot_hi=hi, n=len(ex))
def row_old(s):
    net = OLD[f"{s}_net"]; ew = OLD.EqualWeight_net; r = dict(version="OLD_baseline (INVALID_BEFORE_FIX)", strategy=s, periods=len(OLD))
    r.update({k: v for k, v in L.perf(net).items() if k != "periods"})
    if s != "EqualWeight":
        r.update(dict(gross_mean=float(OLD[f"{s}_gross"].mean()), net_mean=float(net.mean()), IC=float(OLD[f"{s}_IC"].mean()), RankIC=float(OLD[f"{s}_RankIC"].mean()), RankIC_IR=float(OLD[f"{s}_RankIC"].mean() / OLD[f"{s}_RankIC"].std()),
                      turnover_replaced=float(OLD[f"{s}_turnover"].mean()), buy_fill_rate=np.nan, cash_frac=np.nan))
        r.update({"net_ex_" + k: v for k, v in stats_block(net - ew).items()})
        g = stats_block(OLD[f"{s}_gross"] - OLD.universe_ew_gross); r.update(dict(gross_excess_mean=g["excess_mean"], gross_excess_t=g["t"]))
    else: r.update(dict(net_mean=float(net.mean()), turnover_replaced=np.nan))
    return r
def row_new(tag, label, df, s, c="c30"):
    net, gro = df[f"{s}_ret_{c}"], df[f"{s}_ret_c0"]; ewn, ewg = df[f"EqualWeight_ret_{c}"], df["EqualWeight_ret_c0"]
    r = dict(version=label, strategy=s, periods=len(df)); r.update({k: v for k, v in L.perf(net).items() if k != "periods"}); r["net_mean"] = float(net.mean()); r["gross_mean"] = float(gro.mean())
    if s != "EqualWeight": r.update(IC=float(df[f"{s}_IC"].mean()), RankIC=float(df[f"{s}_RankIC"].mean()), RankIC_IR=float(df[f"{s}_RankIC"].mean() / df[f"{s}_RankIC"].std()))
    r["turnover_replaced"] = float(df[f"{s}_replaced"].mean()); r["turnover_traded_value"] = float(df[f"{s}_turnover_value"].mean())
    r["buy_fill_rate"] = float(df[f"{s}_buy_filled"].sum() / max(1, df[f"{s}_buy_attempts"].sum())); r["cash_frac"] = float(df[f"{s}_cash_frac_post"].mean()); r["pending_exit_mean"] = float(df[f"{s}_n_pending_exit"].mean())
    rej = {k.replace(f"{s}_rej_", ""): float(df[k].sum()) for k in df.columns if k.startswith(f"{s}_rej_")}; r["buy_rejections"] = json.dumps(rej)
    if s != "EqualWeight":
        r.update({"net_ex_" + k: v for k, v in stats_block(net - ewn).items()}); g = stats_block(gro - ewg); r.update(dict(gross_excess_mean=g["excess_mean"], gross_excess_t=g["t"]))
    return r
VERS = {"fixed_sina_adj": "FIXED (raw exec, factor-adjusted returns, as-of-t pool)", "tx_qfq_full": "E1-V1 old Tencent-qfq data, fixed engine, full data (diagnostic)",
        "tx_qfq_rowfilter": "E1-V2 old Tencent-qfq data, fixed engine, anomalous rows filtered", "tx_qfq_drop3": "E1-V3 old Tencent-qfq data, fixed engine, 3 invalid-price stocks dropped"}
loaded = {t: pd.read_csv(R / t / "period_returns.csv") for t in VERS if (R / t / "period_returns.csv").exists()}
rows = [row_old(s) for s in S5 + ["EqualWeight"]]
for t, df in loaded.items():
    for s in ALL: rows.append(row_new(t, VERS[t], df, s))
cm = pd.DataFrame(rows); cm.to_csv(OUT / "comparative_metrics.csv", index=False, encoding="utf-8-sig")
# bootstrap intervals (net and gross excess), old and fixed
bi = []
for s in S5:
    for nm, ex in [("OLD net excess vs zero-cost EW (INVALID_BEFORE_FIX)", OLD[f"{s}_net"] - OLD.EqualWeight_net), ("OLD gross excess (INVALID_BEFORE_FIX)", OLD[f"{s}_gross"] - OLD.universe_ew_gross)]:
        b = stats_block(ex); bi.append(dict(version=nm, strategy=s, **b))
for t, df in loaded.items():
    for s in S5:
        for nm, ex in [("net(30bps RT) excess vs same-regime EW", df[f"{s}_ret_c30"] - df.EqualWeight_ret_c30), ("gross excess vs same-regime EW", df[f"{s}_ret_c0"] - df.EqualWeight_ret_c0)]:
            bi.append(dict(version=f"{t}: {nm}", strategy=s, **stats_block(ex)))
bi = pd.DataFrame(bi); bi.to_csv(OUT / "bootstrap_intervals.csv", index=False, encoding="utf-8-sig")
# reproduce GPT numbers on the old CSV
rep = {s: stats_block(OLD[f"{s}_net"] - OLD.EqualWeight_net) for s in ["HGB", "LGBM", "Ridge"]}
json.dump(dict(mine=rep, gpt_claimed={"HGB": dict(mean=0.001639, t=1.22, hac4_t=1.18, hac4_p=0.239, boot=[-0.000866, 0.004519]), "LGBM": dict(mean=0.000753, t=0.51, hac_p=0.604), "Ridge": dict(mean=-0.000859, t=-0.88, hac_p=0.470)},
               note="HAC p-values here use t(n-1); GPT's HAC p may use the normal distribution; bootstrap = moving blocks of 5 periods, 4000 draws, seed 20261018."), open(OUT / "repro_old_numbers.json", "w"), indent=1)
if "fixed_sina_adj" in loaded:
    df = loaded["fixed_sina_adj"]; tag = "fixed_sina_adj"
    # cost sensitivity (E2 minimal)
    cs = []
    for c, cl in [("c0", 0.0), ("c10", 0.001), ("c30", 0.003), ("c60", 0.006)]:
        for s in ALL:
            r = dict(round_trip_cost=cl, strategy=s); r.update({k: v for k, v in L.perf(df[f"{s}_ret_{c}"]).items()})
            if s != "EqualWeight": r.update({"net_ex_" + k: v for k, v in stats_block(df[f"{s}_ret_{c}"] - df[f"EqualWeight_ret_{c}"]).items()})
            cs.append(r)
    pd.DataFrame(cs).to_csv(OUT / "cost_sensitivity.csv", index=False, encoding="utf-8-sig")
    # E0: overlap with old pipeline
    old_sets = pickle.load(open(R / "old_replica" / "old_sets.pkl", "rb")); new = pickle.load(open(R / tag / "sets.pkl", "rb")); od = {r["signal_date"]: r for r in old_sets}
    jac = lambda a, b: len(set(a) & set(b)) / max(1, len(set(a) | set(b)))
    e0 = []
    for k, sd in enumerate(new["signal_dates"]):
        if sd not in od: continue
        o = od[sd]; row = dict(signal_date=sd, n_new_pool=len(new["pools"][k]), n_old_candidate_pool=len(o["old_candidate_pool"]), n_old_eligible_t=len(o["old_eligible_t_pool"]),
                              jaccard_pool_old_candidate_vs_new=jac(o["old_candidate_pool"], new["pools"][k]), jaccard_pool_old_eligible_t_vs_new=jac(o["old_eligible_t_pool"], new["pools"][k]))
        for s in S5: row[f"{s}_top50_jaccard"] = jac(o["topk"][s], new["topk"][s][k])
        e0.append(row)
    e0 = pd.DataFrame(e0); oi = OLD.set_index("signal_date"); ni = df.set_index("signal_date"); com = e0.signal_date
    for s in S5 + ["EqualWeight"]:
        e0[f"{s}_net_old"] = oi.loc[com, f"{s}_net"].values; e0[f"{s}_net_new"] = ni.loc[com, f"{s}_ret_c30"].values; e0[f"{s}_net_diff_new_minus_old"] = e0[f"{s}_net_new"] - e0[f"{s}_net_old"]
        if s != "EqualWeight": e0[f"{s}_RankIC_old"] = oi.loc[com, f"{s}_RankIC"].values; e0[f"{s}_RankIC_new"] = ni.loc[com, f"{s}_RankIC"].values; e0[f"{s}_RankIC_diff"] = e0[f"{s}_RankIC_new"] - e0[f"{s}_RankIC_old"]
    e0.to_csv(OUT / "e0_period_diffs.csv", index=False, encoding="utf-8-sig")
    sm = [dict(item="pool Jaccard old(candidate=eligible&tradable) vs new eligible_asof_t", mean=e0.jaccard_pool_old_candidate_vs_new.mean(), min=e0.jaccard_pool_old_candidate_vs_new.min())]
    sm.append(dict(item="pool Jaccard old eligible_t(no future filter) vs new", mean=e0.jaccard_pool_old_eligible_t_vs_new.mean(), min=e0.jaccard_pool_old_eligible_t_vs_new.min()))
    sm.append(dict(item="avg pool size old candidate / old eligible_t / new", mean=f"{e0.n_old_candidate_pool.mean():.1f} / {e0.n_old_eligible_t.mean():.1f} / {e0.n_new_pool.mean():.1f}"))
    for s in S5:
        d = e0[f"{s}_net_diff_new_minus_old"]; di = e0[f"{s}_RankIC_diff"]; t1 = L.tstat(d)[0]; t2 = L.tstat(di.dropna())[0]
        sm.append(dict(item=f"{s}: Top50 Jaccard old vs new", mean=e0[f"{s}_top50_jaccard"].mean(), min=e0[f"{s}_top50_jaccard"].min()))
        sm.append(dict(item=f"{s}: per-period net return diff new-old", mean=d.mean(), std=d.std(), paired_t=t1, mean_abs=d.abs().mean()))
        sm.append(dict(item=f"{s}: RankIC diff new-old", mean=di.mean(), std=di.std(), paired_t=t2))
    pd.DataFrame(sm).to_csv(OUT / "e0_summary.csv", index=False, encoding="utf-8-sig")
    # by year
    by = []
    for y, g in df.groupby(df.signal_date.str[:4]):
        for s in ALL:
            r = dict(year=y, strategy=s, periods=len(g), mean_net=float(g[f"{s}_ret_c30"].mean()), compounded_net=float((1 + g[f"{s}_ret_c30"]).prod() - 1))
            if s != "EqualWeight": r.update(RankIC=float(g[f"{s}_RankIC"].mean()), net_excess_mean=float((g[f"{s}_ret_c30"] - g.EqualWeight_ret_c30).mean()))
            by.append(r)
    pd.DataFrame(by).to_csv(OUT / "by_year_fixed.csv", index=False, encoding="utf-8-sig")
    # P0-D diagnostics
    dg = [dict(item="pool delisted-in-sample share (mean over periods)", value=float(df.pool_delist_share.mean())), dict(item="pool label mean listed (5d, mark-to-market)", value=float(df.pool_label_listed.mean())),
          dict(item="pool label mean delisted-in-sample (5d)", value=float(df.pool_label_delisted.mean())), dict(item="reweighted pool mean with delisted share=3.6% (universe-like)", value=float(0.964 * df.pool_label_listed.mean() + 0.036 * df.pool_label_delisted.mean())),
          dict(item="reweighted pool mean with delisted share as sampled", value=float((1 - df.pool_delist_share) @ df.pool_label_listed / len(df) + df.pool_delist_share @ df.pool_label_delisted.fillna(0) / len(df)))]
    for s in S5:
        dg += [dict(item=f"{s}: Top50 delisted share", value=float(df[f"{s}_delist_share"].mean())), dict(item=f"{s}: mean 5d label of picks - listed", value=float(df[f"{s}_label_listed"].mean())), dict(item=f"{s}: mean 5d label of picks - delisted-in-sample", value=float(df[f"{s}_label_delisted"].mean()))]
    pd.DataFrame(dg).to_csv(OUT / "delist_group_diagnostics.csv", index=False, encoding="utf-8-sig")
# E1 table
e1 = cm[cm.version.str.startswith(("OLD", "FIXED", "E1"))][["version", "strategy", "annualized_return", "sharpe", "max_drawdown", "RankIC", "net_ex_excess_mean", "net_ex_t", "net_ex_hac4_t", "net_ex_boot_lo", "net_ex_boot_hi", "turnover_replaced", "buy_fill_rate"]]
e1.to_csv(OUT / "e1_sensitivity.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); print(repro := json.dumps(rep, indent=0)[:1500]); print(e1.round(4).to_string())
