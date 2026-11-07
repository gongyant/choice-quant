# -*- coding: utf-8 -*-
"""P2-B library: protocol, nested time validation, run orchestration for one (model, scenario)."""
import json, time, itertools
import numpy as np, pandas as pd
import p2lib as Q, b_sim as S, b_data as BD
PROTO = json.load(open(Q.RES / "b_protocol.json", encoding="utf-8"))
BURN, BLOCK = PROTO["nested"]["burn_in"], PROTO["nested"]["block"]
A1_GRID = [("A1", k) for k in PROTO["grids"]["A1_K_hold"]]
A2_GRID = [("A2", t, c, d) for t in PROTO["grids"]["A2_tau"] for c in PROTO["grids"]["A2_cap"] for d in PROTO["grids"]["A2_delta"]]
A1_DEF = ("A1", PROTO["defaults_burn_in"]["A1_K_hold"]); _d = PROTO["defaults_burn_in"]["A2"]; A2_DEF = ("A2", _d["tau"], _d["cap"], _d["delta"])
def vname(th): return f"A1_K{th[1]}" if th[0] == "A1" else f"A2_t{th[1]}_c{th[2]}_d{'off' if th[3] is None else th[3]}"
def sharpe(x):
    x = np.asarray(x, float); sd = x.std(ddof=1); return float(x.mean() / sd) if sd > 0 else -np.inf

def nested_select(rets, default, burn_in=BURN, block=BLOCK):
    """rets: {theta: per-period net return array (length K)} from parameter-fixed runs. For the block starting at st the choice uses rets[:st] ONLY (expanding window).
    Returns (schedule list of theta per period, selection log)."""
    K = len(next(iter(rets.values()))); sched = [default] * K; log = []
    for st in range(burn_in, K, block):
        en = min(K, st + block); crit = {th: sharpe(r[:st]) for th, r in rets.items()}
        best = max(crit, key=lambda th: (crit[th], -list(rets).index(th)))
        for k in range(st, en): sched[k] = best
        log.append(dict(block_start=st, block_end=en, chosen=vname(best), crit_chosen=crit[best], n_train_periods=st, **{"sharpe_" + vname(th): v for th, v in crit.items()}))
    return sched, log

def make_cost(D, scen, hs_mult=None, imp_mult=None):
    if scen["cost"].startswith("flat"): return S.FlatCost(0.003)
    kw = {}
    if hs_mult is not None: kw["hs_mult"] = hs_mult
    if imp_mult is not None: kw["imp_mult"] = imp_mult
    return S.ExplicitCost(D["cal"], D["HS"], D["VOL"], D["ADV"], **kw)

class Recorder:
    """Wraps a decider and logs the DECISION (time s) separately from execution: holdings known at decision, targets, weights, ranks."""
    def __init__(self, inner, score_getter): self.inner, self.sg, self.rows = inner, score_getter, []
    def __call__(self, k, ctx):
        dec = self.inner(k, ctx); self.rows.append(dict(k=k, s=ctx["s"], held=np.where(ctx["units"] > 0)[0].copy(), tgt=np.asarray(dec["tgt"]).copy(), w=None if dec.get("w") is None else np.asarray(dec["w"]).copy(), meta=dec.get("meta", {})))
        return dec
class Sched:
    def __init__(self, deciders, sched): self.d, self.s = deciders, sched
    def __call__(self, k, ctx): return self.d[self.s[k]](k, ctx)

def run_model_scen(model, scen_name, D=None, with_stress=False, log_dir=None, verbose=True):
    t0 = time.time(); D = D or BD.build(); scen = PROTO["scenarios"][scen_name]; K = D["K"]; X, sig, ADJf = D["X"], D["sig"], D["ADJf"]
    cand = BD.cand_mask(D, dict(no_star=scen["no_star"], no_bj=scen["no_bj"], min_adv=scen["min_adv"])); sc = D["scores"][model]
    cost = make_cost(D, scen); eng = dict(capital=scen["capital"], pcap=scen["pcap"], ADV=D["ADV"], min_adv=scen["min_adv"])
    frames = {}; extras = dict(model=model, scenario=scen_name)
    def run(decider, name, rebalance=False, keep_logs=False, cst=None):
        o, lg, ex = S.simulate_ext(X, ADJf, list(sig), decider, cst or cost, rebalance=rebalance, name=name, keep_logs=keep_logs, **eng)
        o = o.copy(); o["variant"] = name; o["k"] = np.arange(len(o)); frames[name] = o; return o, lg, ex
    def mk(th, cst=None):
        c = cst or cost
        if th[0] == "A1": return S.A1Decider(sc, cand, sig, k_hold=th[1])
        return S.A2Decider(sc, cand, sig, ADJf, D["W"], X, c, D["y"], tau=th[1], cap=th[2], delta=th[3], capital=scen["capital"], **{k: v for k, v in PROTO["A2_fixed"].items() if k in ("lam", "M", "beta0", "beta_floor", "min_obs")})
    a0, a0lg, a0ex = run(S.A0Decider(sc, cand, sig), "A0", keep_logs=True); extras["A0_min_cash"] = a0ex["min_cash"]
    fixed = {}; ths = A1_GRID + A2_GRID; deciders = {}
    for th in ths:
        deciders[th] = mk(th); o, _, ex = run(deciders[th], vname(th), rebalance=(th[0] == "A2")); fixed[th] = o.ret.values
        if th[0] == "A2": extras.setdefault("a2_solver_fail", {})[vname(th)] = int(sum(not d["meta"]["success"] for d in ex["decisions"]))
    sel = {}; scheds = {}
    for fam, grid, dflt in [("A1", A1_GRID, A1_DEF), ("A2", A2_GRID, A2_DEF)]:
        sched, log = nested_select({th: fixed[th] for th in grid}, dflt); scheds[fam] = sched; sel[fam] = log
        dd = {th: mk(th) for th in set(sched)}; rec = Recorder(Sched(dd, sched), None)
        o, lg, ex = run(rec, f"{fam}_nested", rebalance=(fam == "A2"), keep_logs=True); extras[f"{fam}_nested_min_cash"] = ex["min_cash"]
        extras[f"{fam}_schedule"] = [vname(th) for th in sched]
        if log_dir is not None: save_logs(log_dir, model, scen_name, f"{fam}_nested", lg, rec.rows, X, sig, D)
    if log_dir is not None: save_logs(log_dir, model, scen_name, "A0", a0lg, None, X, sig, D)
    extras["selection"] = sel
    # --- turnover-matched random-retention controls (calibrated to the nested strategies' eval-period turnover) ---
    ev = slice(BURN, None); ctrl = {}
    for fam in ("A1", "A2"):
        target = float(frames[f"{fam}_nested"].turnover_value.values[ev].mean()); seed = PROTO["controls"]["random_retention"]["calibrate_seed"]; nit = PROTO["controls"]["random_retention"]["bisect_iters"]; nseed = PROTO["controls"]["random_retention"]["eval_seeds"]
        for tag, mkdec in [("R", lambda p, sd: S.RandRetainDecider(sc, cand, sig, p=p, seed=sd)), ("S", lambda p, sd: S.RandSkipDecider(sc, cand, sig, q=p, seed=sd))]:
            lo, hi = 0.0, 1.0       # R: p = retention prob (turnover falls as p rises); S: q = refresh prob (turnover rises with q)
            for _ in range(nit):
                mid = (lo + hi) / 2; o, _, _ = S.simulate_ext(X, ADJf, list(sig), mkdec(mid, seed), cost, keep_logs=False, **eng); tv = o.turnover_value.values[ev].mean()
                if (tv > target) == (tag == "R"): lo = mid
                else: hi = mid
            p = (lo + hi) / 2; outs = []
            for sd in range(1, nseed + 1):
                o, _, _ = S.simulate_ext(X, ADJf, list(sig), mkdec(p, sd), cost, keep_logs=False, **eng); outs.append(o)
            m = outs[0].copy()
            for c in ["ret", "gross_ret", "turnover_value", "cost_total", "nav_pre", "n_hold", "cash_frac_post"] + [c for c in outs[0].columns if c.startswith("cost_")]: m[c] = np.mean([o[c].values for o in outs], axis=0)
            m["variant"] = f"{tag}_{fam}"; m["k"] = np.arange(len(m)); frames[f"{tag}_{fam}"] = m
            ctrl[f"{tag}_{fam}"] = dict(param=p, target_turnover=target, achieved_turnover=float(m.turnover_value.values[ev].mean()), seed_eval_mean_net=[float(o.ret.values[ev].mean()) for o in outs])
    extras["controls"] = ctrl
    # --- cost stress (same schedules re-run under shifted cost assumptions) ---
    if with_stress and scen["cost"].startswith("explicit"):
        for tag, kw in [("hs0.5", dict(hs_mult=0.125)), ("hs2", dict(hs_mult=0.5)), ("imp0.5", dict(imp_mult=0.5)), ("imp2", dict(imp_mult=2.0))]:
            cs = make_cost(D, scen, **kw); eng2 = dict(eng)
            for nm, dec, rb in [("A0", S.A0Decider(sc, cand, sig), False), ("A1_nested", Sched({th: mk(th) for th in set(scheds["A1"])}, scheds["A1"]), False), ("A2_nested", Sched({th: mk(th, cs) for th in set(scheds["A2"])}, scheds["A2"]), True)]:
                o, _, _ = S.simulate_ext(X, ADJf, list(sig), dec, cs, rebalance=rb, keep_logs=False, **eng2); o = o.copy(); o["variant"] = f"{nm}|stress_{tag}"; o["k"] = np.arange(len(o)); frames[o.variant.iloc[0]] = o
    allp = pd.concat(frames.values(), ignore_index=True); allp["model"] = model; allp["scenario"] = scen_name
    for c in allp.columns:
        if allp[c].dtype == object and c not in ("variant", "model", "scenario"): allp[c] = pd.to_numeric(allp[c], errors="coerce")
    extras["seconds"] = round(time.time() - t0, 1)
    return allp, extras

def save_logs(log_dir, model, scen, variant, lg, rows, X, sig, D):
    log_dir.mkdir(parents=True, exist_ok=True)
    if len(lg): lg.to_csv(log_dir / f"orders_{model}_{scen}_{variant}.csv.gz", index=False, encoding="utf-8-sig")
    if rows is not None:
        out = []
        for r in rows:
            held = set(int(j) for j in r["held"]); w = r["w"]; n = len(r["tgt"])
            for q, j in enumerate(r["tgt"]): out.append(dict(k=r["k"], decision_date=str(X.cal[r["s"]].date()), code=X.codes[int(j)], target_weight=float(w[q]) if w is not None else 1.0 / n, held_at_decision=int(j) in held, **{f"meta_{a}": b for a, b in r["meta"].items() if np.isscalar(b)}))
            for j in held - set(int(x) for x in r["tgt"]): out.append(dict(k=r["k"], decision_date=str(X.cal[r["s"]].date()), code=X.codes[int(j)], target_weight=0.0, held_at_decision=True))
        pd.DataFrame(out).to_csv(log_dir / f"decisions_{model}_{scen}_{variant}.csv.gz", index=False, encoding="utf-8-sig")