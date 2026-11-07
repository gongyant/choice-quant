# -*- coding: utf-8 -*-
"""P2-B aggregation: performance, paired tests, factor alpha, turnover-vs-prediction decomposition, constraints, cost stress."""
import json, glob
import numpy as np, pandas as pd
import p2lib as Q, p0lib as L
R = Q.RES; PPY = 252 / 5; EV = 52
MOD = Q.MODELS; SCEN = ["C1_flat", "C2_explicit", "C3_realistic", "C4_capacity_1e8"]
runs = {}; extras = {}
for m in MOD:
    for s in SCEN:
        f = R / "b_runs" / f"{m}_{s}.parquet"
        if f.exists(): runs[(m, s)] = pd.read_parquet(f); extras[(m, s)] = json.load(open(R / "b_runs" / f"{m}_{s}_extras.json", encoding="utf-8"))
EW = pd.read_csv(R / "b_ew_benchmark.csv"); FR = pd.read_csv(R / "factor_returns_by_date.csv", index_col=0, encoding="utf-8-sig"); FR.columns = [c.split("(")[0] for c in FR.columns]; fcols = list(FR.columns)
def series(m, s, v, col="ret"):
    d = runs[(m, s)]; x = d[d.variant == v].sort_values("k"); return x
def window(x, w): return x[x.k >= EV] if w == "eval_k52+" else x
def metrics(x, ew):
    r = x.ret.values; g = x.gross_ret.values; p = L.perf(r); pg = L.perf(g); cc = [c for c in x.columns if c.startswith("cost_") and c != "cost_total"]
    out = dict(n_periods=len(r), net_mean_per_period=float(r.mean()), gross_mean_per_period=float(g.mean()), ann_net=p["annualized_return"], ann_gross=pg["annualized_return"], ann_vol=p["annualized_vol"], sharpe_net=p["sharpe"], sharpe_gross=pg["sharpe"],
               max_drawdown=p["max_drawdown"], win_rate=p["win_rate"], turnover_oneway_per_period=float(x.turnover_value.mean()), turnover_oneway_annualized=float(x.turnover_value.mean() * PPY),
               cost_per_period=float((x.cost_total / x.nav_pre).mean()), cost_drag_annual=float((x.cost_total / x.nav_pre).mean() * PPY), n_hold=float(x.n_hold.mean()), cash_frac=float(x.cash_frac_post.mean()))
    for c in cc: out["drag_" + c[5:] + "_annual"] = float((x[c] / x.nav_pre).mean() * PPY)
    ex0 = r - ew.ew_ret_c0.values[x.k.values]; out["excess_vs_EWc0_mean"] = float(ex0.mean()); sm = Q.summarize(ex0); out.update(excess_hac_t=sm["hac_t"], excess_ci_lo=sm["ci_lo"], excess_ci_hi=sm["ci_hi"])
    ex3 = r - ew.ew_ret_c30.values[x.k.values]; out["excess_vs_EWc30_mean"] = float(ex3.mean()); out["excess_c30_hac_t"] = Q.nw_t(ex3)[0]
    return out
MAIN = ["A0", "A1_nested", "A2_nested", "R_A1", "S_A1", "R_A2", "S_A2"]
rows = []
for (m, s), d in runs.items():
    for v in d.variant.unique():
        if "|stress" in v: continue
        x = series(m, s, v)
        for w in ["eval_k52+", "full207"]:
            if w == "full207" and v in ("A1_nested", "A2_nested") : pass
            xx = window(x, w)
            if len(xx) < 20: continue
            rows.append(dict(scenario=s, model=m, variant=v, window=w, **metrics(xx, EW)))
PF = pd.DataFrame(rows); PF.to_csv(R / "b_performance.csv", index=False, encoding="utf-8-sig")
# ---- paired differences (net, gross, cost) on eval periods ----
def paired(m, s, a, b, w="eval_k52+"):
    xa, xb = window(series(m, s, a), w), window(series(m, s, b), w); out = {}
    for lab, col in [("net", "ret"), ("gross", "gross_ret")]:
        dd = xa[col].values - xb[col].values; out[lab] = Q.summarize(dd)
    dc = (xa.cost_total / xa.nav_pre).values - (xb.cost_total / xb.nav_pre).values; out["cost"] = Q.summarize(dc)
    dt = xa.turnover_value.values - xb.turnover_value.values; out["turn"] = dict(mean=float(dt.mean()))
    return out
rows = []
for (m, s) in runs:
    for a, b in [("A1_nested", "A0"), ("A2_nested", "A0"), ("A2_nested", "A1_nested"), ("A1_nested", "R_A1"), ("A1_nested", "S_A1"), ("A2_nested", "R_A2"), ("A2_nested", "S_A2"), ("R_A1", "A0"), ("S_A1", "A0"), ("R_A2", "A0"), ("S_A2", "A0")]:
        o = paired(m, s, a, b)
        rows.append(dict(scenario=s, model=m, comparison=f"{a} - {b}", n_dates=o["net"]["n_dates"], net_diff_mean=o["net"]["mean"], net_hac_t=o["net"]["hac_t"], net_ci_lo=o["net"]["ci_lo"], net_ci_hi=o["net"]["ci_hi"], net_pos_share=o["net"]["pos_share"],
                         gross_diff_mean=o["gross"]["mean"], gross_hac_t=o["gross"]["hac_t"], cost_diff_mean=o["cost"]["mean"], cost_hac_t=o["cost"]["hac_t"], turnover_diff_mean=o["turn"]["mean"]))
PR = pd.DataFrame(rows); PR.to_csv(R / "b_paired.csv", index=False, encoding="utf-8-sig")
# ---- turnover vs prediction decomposition ----
rows = []
for (m, s) in runs:
    a0 = window(series(m, s, "A0"), "eval_k52+")
    for fam in ("A1", "A2"):
        X_ = window(series(m, s, f"{fam}_nested"), "eval_k52+"); Rr = window(series(m, s, f"R_{fam}"), "eval_k52+"); Ss = window(series(m, s, f"S_{fam}"), "eval_k52+")
        net = lambda z: float(z.ret.mean()); gross = lambda z: float(z.gross_ret.mean()); cost = lambda z: float((z.cost_total / z.nav_pre).mean())
        dn = net(X_) - net(a0)
        r = dict(scenario=s, model=m, family=fam, net_A0=net(a0), net_X=net(X_), delta_net=dn, delta_gross=gross(X_) - gross(a0), delta_cost_saving=cost(a0) - cost(X_), turnover_A0=float(a0.turnover_value.mean()), turnover_X=float(X_.turnover_value.mean()))
        for tag, Cn in (("R", Rr), ("S", Ss)):
            r[f"net_{tag}"] = net(Cn); r[f"turnover_{tag}"] = float(Cn.turnover_value.mean()); r[f"turnover_effect_{tag}"] = net(Cn) - net(a0); r[f"selection_effect_{tag}"] = net(X_) - net(Cn)
            r[f"turnover_share_{tag}"] = (r[f"turnover_effect_{tag}"] / dn) if abs(dn) > 1e-4 else np.nan
            # exact turnover matching: the bisection calibration used one seed and is biased (S control ~10-16% too low), so the control is linearly re-weighted with A0 (its own mixture end-point) to hit X's turnover
            lam = (r["turnover_X"] - r[f"turnover_{tag}"]) / (r["turnover_A0"] - r[f"turnover_{tag}"]); adj = Cn.ret.values + lam * (a0.ret.values - Cn.ret.values); r[f"lambda_{tag}"] = lam; r[f"net_{tag}_adj"] = float(adj.mean())
            r[f"turnover_effect_{tag}_adj"] = float(adj.mean()) - net(a0); r[f"selection_effect_{tag}_adj"] = net(X_) - float(adj.mean()); r[f"turnover_share_{tag}_adj"] = (r[f"turnover_effect_{tag}_adj"] / dn) if abs(dn) > 1e-4 else np.nan
            sm2 = Q.summarize(X_.ret.values - adj); r[f"selection_{tag}_adj_hac_t"] = sm2["hac_t"]; r[f"selection_{tag}_adj_ci_lo"] = sm2["ci_lo"]; r[f"selection_{tag}_adj_ci_hi"] = sm2["ci_hi"]
        # paired HAC for the selection component vs each control
        for tag in ("R", "S"):
            dd = X_.ret.values - window(series(m, s, f"{tag}_{fam}"), "eval_k52+").ret.values; sm = Q.summarize(dd); r[f"selection_{tag}_hac_t"] = sm["hac_t"]; r[f"selection_{tag}_ci_lo"] = sm["ci_lo"]; r[f"selection_{tag}_ci_hi"] = sm["ci_hi"]
        rows.append(r)
pd.DataFrame(rows).to_csv(R / "b_turnover_vs_prediction.csv", index=False, encoding="utf-8-sig")
# ---- factor-adjusted net alpha (eval periods): excess over EW c0 on the five P2-A factors ----
rows = []
for (m, s) in runs:
    for v in MAIN:
        x = window(series(m, s, v), "eval_k52+"); idx = x.signal_idx.values; ex0 = x.ret.values - EW.ew_ret_c0.values[x.k.values]; Fm = FR.loc[idx, fcols].values
        beta, t, r2 = Q.ols_hac(ex0, Fm); lo, hi = Q.alpha_ci(ex0, Fm, B_=2000)
        rows.append(dict(scenario=s, model=m, variant=v, n=len(ex0), alpha_per_period=float(beta[0]), alpha_hac_t=float(t[0]), alpha_ci_lo=lo, alpha_ci_hi=hi, r2=float(r2), alpha_annualized_simple=float(beta[0] * PPY), **{"beta_" + c: float(beta[j + 1]) for j, c in enumerate(fcols)}, **{"t_" + c: float(t[j + 1]) for j, c in enumerate(fcols)}))
        # gross-of-cost alpha too
        yg = x.gross_ret.values - EW.ew_ret_c0.values[x.k.values]; bg, tg, _ = Q.ols_hac(yg, Fm); rows[-1].update(gross_alpha_per_period=float(bg[0]), gross_alpha_hac_t=float(tg[0]))
pd.DataFrame(rows).to_csv(R / "b_factor_alpha.csv", index=False, encoding="utf-8-sig")
# ---- nested selection log and optimism check ----
rows = []; opt = []
for (m, s), ex in extras.items():
    for fam in ("A1", "A2"):
        for b in ex["selection"][fam]: rows.append(dict(scenario=s, model=m, family=fam, block_start=b["block_start"], block_end=b["block_end"], chosen=b["chosen"], criterion_value=b["crit_chosen"], n_train_periods=b["n_train_periods"]))
        d = runs[(m, s)]; gv = [v for v in d.variant.unique() if v.startswith(fam + "_") and v != f"{fam}_nested" and "|" not in v and v not in ("A1_nested",)]
        gv = [v for v in gv if (v.startswith("A1_K") if fam == "A1" else v.startswith("A2_t"))]
        ev_sh = {v: L.perf(window(series(m, s, v), "eval_k52+").ret.values)["sharpe"] for v in gv}; best = max(ev_sh, key=ev_sh.get)
        opt.append(dict(scenario=s, model=m, family=fam, nested_eval_sharpe=L.perf(window(series(m, s, f"{fam}_nested"), "eval_k52+").ret.values)["sharpe"], best_fixed_eval_sharpe_hindsight=ev_sh[best], best_fixed_hindsight_variant=best, median_fixed_eval_sharpe=float(np.median(list(ev_sh.values()))), n_grid=len(gv)))
pd.DataFrame(rows).to_csv(R / "b_nested_selection.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(opt).to_csv(R / "b_nested_optimism_check.csv", index=False, encoding="utf-8-sig")
# ---- fixed-parameter grid (exploratory; NOT used for inference: chosen in hindsight) ----
G = PF[(PF.variant.str.startswith("A1_K")) | (PF.variant.str.startswith("A2_t"))]; G.to_csv(R / "b_grid_fixed_params.csv", index=False, encoding="utf-8-sig")
# ---- constraints / fills diagnostics ----
rows = []
for (m, s), d in runs.items():
    for v in ["A0", "A1_nested", "A2_nested"]:
        x = window(series(m, s, v), "eval_k52+"); rc = [c for c in x.columns if c.startswith("rej_")]
        rows.append(dict(scenario=s, model=m, variant=v, buy_fill_rate=float((x.buy_filled.sum() / x.buy_attempts.sum())) if x.buy_attempts.sum() else np.nan, cash_frac=float(x.cash_frac_post.mean()), n_hold=float(x.n_hold.mean()), max_weight=float(x.max_weight_post.max()), pending_exit_mean=float(x.n_pending_exit.mean()),
                         **{c + "_per_period": float(x[c].fillna(0).mean()) for c in rc}))
pd.DataFrame(rows).to_csv(R / "b_fill_diagnostics.csv", index=False, encoding="utf-8-sig")
# ---- cost stress ----
rows = []
for (m, s) in runs:
    d = runs[(m, s)]
    for v in d.variant.unique():
        if "|stress" in v:
            x = window(series(m, s, v), "eval_k52+"); base = v.split("|")[0]; rows.append(dict(scenario=s, model=m, strategy=base, stress=v.split("stress_")[1], net_mean=float(x.ret.mean()), sharpe_net=L.perf(x.ret.values)["sharpe"], cost_per_period=float((x.cost_total / x.nav_pre).mean()), turnover=float(x.turnover_value.mean())))
    for base in ["A0", "A1_nested", "A2_nested"]:
        if any("|stress" in v for v in d.variant.unique()):
            x = window(series(m, s, base), "eval_k52+"); rows.append(dict(scenario=s, model=m, strategy=base, stress="base", net_mean=float(x.ret.mean()), sharpe_net=L.perf(x.ret.values)["sharpe"], cost_per_period=float((x.cost_total / x.nav_pre).mean()), turnover=float(x.turnover_value.mean())))
pd.DataFrame(rows).to_csv(R / "b_cost_stress.csv", index=False, encoding="utf-8-sig")
# ---- extras: controls info ----
rows = []
for (m, s), ex in extras.items():
    for k, v in ex["controls"].items(): rows.append(dict(scenario=s, model=m, control=k, param=v["param"], target_turnover=v["target_turnover"], achieved_turnover=v["achieved_turnover"], seed_net_mean=float(np.mean(v["seed_eval_mean_net"])), seed_net_sd=float(np.std(v["seed_eval_mean_net"], ddof=1))))
pd.DataFrame(rows).to_csv(R / "b_controls_calibration.csv", index=False, encoding="utf-8-sig")
print("aggregated", len(PF), len(PR))