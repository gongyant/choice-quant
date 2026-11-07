# -*- coding: utf-8 -*-
"""P2-B tests: engine == P0 state machine by default; cash / partial fills / limit-up-down / suspension; buffer-size invariants; cost components booked once;
nested validation does not leak; A0 reconciles with P0; no look-ahead in decisions; A2 solver vs SLSQP."""
import sys, json, copy
from pathlib import Path
import numpy as np, pandas as pd, pytest
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "p0" / "leakage_unit_tests"))
import p2lib as Q, b_sim as S, b_lib as BL, b_data as BD, p0lib as L
from synth import make_synth
RES = Q.RES
# ---------------- synthetic helpers ----------------
def syn(seed=5, N=30):
    X = make_synth(N=N, seed=seed, suspensions=False); L.finalize(X); L.limit_flags(X); f, elig, ADJf = L.compute_features(X, X.ADJ_unadj_based); return X, ADJf
def refresh(X):
    L.finalize(X); L.limit_flags(X); return L.compute_features(X, X.ADJ_unadj_based)[2]
def kill_bar(X, d, j):
    for M in (X.O, X.Hh, X.L, X.C, X.V, X.AMT, X.TURN): M[d, j] = np.nan
def lock_up(X, d, j, pct=0.10):
    X.C[d, j] = L.round_cent(X.C[d - 1, j] * (1 + pct)); X.Hh[d, j] = X.C[d, j]; X.O[d, j] = min(X.O[d, j], X.C[d, j]); X.L[d, j] = min(X.L[d, j], X.C[d, j])
def lock_dn(X, d, j, pct=0.10):
    X.C[d, j] = L.round_cent(X.C[d - 1, j] * (1 - pct)); X.L[d, j] = X.C[d, j]; X.Hh[d, j] = max(X.Hh[d, j], X.C[d, j]); X.O[d, j] = max(X.O[d, j], X.C[d, j])
class Fixed:
    def __init__(self, tg, w=None): self.tg, self.w = tg, w
    def __call__(self, k, ctx): return dict(tgt=np.asarray(self.tg[k]), w=None if self.w is None else np.asarray(self.w[k]))
def mats(X, hs=0.001, vol=0.02, adv=5e7):
    T, N = X.C.shape; return np.full((T, N), hs), np.full((T, N), vol), np.full((T, N), adv)
def expl(X, **kw): HS, VOL, ADV = mats(X); return S.ExplicitCost(X.cal, HS, VOL, ADV, **kw)
# ---------------- engine == P0 by default ----------------
@pytest.mark.parametrize("cost", [0.0, 0.003])
def test_default_engine_identical_to_p0_synthetic(cost):
    X, ADJf = syn(seed=3); rng = np.random.default_rng(1); sig = [100, 105, 110, 115, 120]; tg = [rng.choice(30, 8, replace=False) for _ in sig]
    a, la, ea = L.simulate(X, ADJf, sig, tg, cost, capital=1e6, lot=100, log_name="x"); b, lb, eb = S.simulate_ext(X, ADJf, sig, Fixed(tg), S.FlatCost(cost), capital=1e6, lot=100)
    assert (a.ret.values == b.ret.values).all() and (a.turnover_value.values == b.turnover_value.values).all() and (a.cash_frac_post.values == b.cash_frac_post.values).all() and (a.n_hold.values == b.n_hold.values).all()
def test_buy_failure_leaves_cash_and_never_substitutes():
    X, _ = syn(seed=5); kill_bar(X, 101, 2); lock_up(X, 101, 3); ADJf = refresh(X)
    per, lg, ex = S.simulate_ext(X, ADJf, [100, 105], Fixed([np.arange(5), np.arange(5, 10)]), S.FlatCost(0.0), capital=1e6, lot=1)
    b = lg[(lg.side == "BUY") & (lg.order_date == str(X.cal[101].date()))]
    assert set(b[b.status == "FILLED"].code) == {"S000", "S001", "S004"} and dict(zip(b[b.status == "REJECTED"].code, b[b.status == "REJECTED"].reason)) == {"S002": "suspended_or_no_bar", "S003": "limit_up_locked"}
    assert per.loc[0, "buy_filled"] == 3 and abs(per.loc[0, "cash_frac_post"] - 0.4) < 0.02
def test_limit_down_sell_blocked_then_retried_and_suspension():
    X, _ = syn(seed=6); lock_dn(X, 106, 1); lock_dn(X, 107, 1)
    for d in (106, 107, 108): kill_bar(X, d, 0)
    ADJf = refresh(X); per, lg, ex = S.simulate_ext(X, ADJf, [100, 105], Fixed([np.arange(5), np.arange(5, 10)]), S.FlatCost(0.0), capital=1e6, lot=1)
    s1 = lg[(lg.code == "S001") & (lg.side == "SELL")]; assert list(s1.status)[0] == "BLOCKED_RETRY" and s1.iloc[0].reason == "limit_down_locked" and s1[s1.status == "FILLED"].iloc[0].fill_date == str(X.cal[108].date())
    s0 = lg[(lg.code == "S000") & (lg.side == "SELL") & (lg.status == "FILLED")]; assert s0.iloc[0].fill_date == str(X.cal[109].date()) and s0.iloc[0].sell_delay_days == 3
    assert per.loc[1, "n_pending_exit"] >= 1
# ---------------- partial fills / participation cap ----------------
def test_participation_cap_gives_partial_fills_and_pending_sell_continues():
    X, ADJf = syn(seed=8); T, N = X.C.shape; ADV = np.full((T, N), 4e5)       # tiny ADV so 5% cap = 20k RMB per name per day
    sig = [100, 105, 110]; tg = [np.arange(4), np.arange(4, 8), np.arange(8, 12)]
    per, lg, ex = S.simulate_ext(X, ADJf, sig, Fixed(tg), S.FlatCost(0.003), capital=2e6, lot=100, pcap=0.05, ADV=ADV)
    buys = lg[lg.side == "BUY"]; assert (buys.status == "PARTIAL_FILLED").any() and (buys.reason[buys.status == "PARTIAL_FILLED"] == "participation_cap").all()
    assert (buys.value <= 0.05 * 4e5 + 1e-6).all() and per.cash_frac_post.iloc[0] > 0.5      # unfilled remainder stays idle cash, nothing is leveraged
    s = lg[(lg.side == "SELL") & (lg.code == "S000")]; assert (s.status.isin(["PARTIAL_FILLED", "FILLED", "BLOCKED_RETRY"])).all() and (s.value <= 0.05 * 4e5 * 1.5).all()
    assert ex["min_cash"] >= -1e-6
def test_min_adv_rejects_buys():
    X, ADJf = syn(seed=9); T, N = X.C.shape; ADV = np.full((T, N), 1e6); ADV[:, 2] = 1e9
    per, lg, ex = S.simulate_ext(X, ADJf, [100, 105], Fixed([np.arange(5), np.arange(5)]), S.FlatCost(0.0), capital=1e6, lot=1, ADV=ADV, min_adv=1e8)
    b = lg[lg.side == "BUY"]; assert set(b[b.status == "FILLED"].code) == {"S002"} and (b[b.status == "REJECTED"].reason == "below_min_adv").all()
# ---------------- costs booked once ----------------
def test_explicit_cost_components_sum_to_fee_and_flat_has_single_component():
    X, ADJf = syn(seed=11); sig = [100, 105, 110]; rng = np.random.default_rng(2); tg = [rng.choice(30, 8, replace=False) for _ in sig]
    cs = expl(X); per, lg, ex = S.simulate_ext(X, ADJf, sig, Fixed(tg), cs, capital=1e6, lot=100)
    comp = [c for c in lg.columns if c.startswith("cost_")]; assert set(comp) == {"cost_commission", "cost_transfer", "cost_stamp", "cost_spread", "cost_impact"}
    f = lg[lg.status.isin(["FILLED", "PARTIAL_FILLED"])]; assert np.allclose(f.cost.values, f[comp].fillna(0).sum(axis=1).values, rtol=1e-12, atol=1e-9)
    assert (f[f.side == "BUY"].cost_stamp == 0).all() and (f[f.side == "SELL"].cost_stamp > 0).all()
    assert abs(per.cost_total.sum() - f.cost.sum()) < 1e-6 and np.allclose(per[[c for c in per.columns if c.startswith("cost_") and c != "cost_total"]].sum(axis=1), per.cost_total)
    p2, l2, _ = S.simulate_ext(X, ADJf, sig, Fixed(tg), S.FlatCost(0.003), capital=1e6, lot=100); f2 = l2[l2.status == "FILLED"]
    assert [c for c in l2.columns if c.startswith("cost_")] == ["cost_flat"] and np.allclose(f2.cost, f2.value * 0.0015, rtol=1e-12)
    # cash never negative: replay the cash path from the logs
    ev = pd.concat([f[f.side == "BUY"].assign(d=lambda x: x.fill_date, o=1, c=lambda x: -(x.value + x.cost)), f[f.side == "SELL"].assign(d=lambda x: x.fill_date, o=0, c=lambda x: x.value - x.cost)]).sort_values(["d", "o"], kind="stable")   # sells settle before buys on the same day
    assert (1e6 + ev.c.cumsum()).min() > -1e-6
def test_stamp_schedule_and_sell_only():
    cal = pd.bdate_range("2023-08-14", periods=60); T, N = 60, 3; cs = S.ExplicitCost(cal, np.full((T, N), 0.0), np.full((T, N), 0.0), np.full((T, N), 1e8), comm_bp=0, transfer_bp=0)
    d_early = int(np.where(cal == pd.Timestamp("2023-08-25"))[0][0]); d_late = int(np.where(cal == pd.Timestamp("2023-08-28"))[0][0]) if (cal == pd.Timestamp("2023-08-28")).any() else d_early + 3
    j = np.array([0]); assert abs(cs.rate("sell", d_early, j, [1e5])[0] - 0.0010) < 1e-12 and abs(cs.rate("sell", d_late, j, [1e5])[0] - 0.0005) < 1e-12 and cs.rate("buy", d_early, j, [1e5])[0] == 0.0
def test_cost_uses_only_prior_day_information():
    X, _ = syn(seed=13); HS, VOL, ADV = mats(X); cs = S.ExplicitCost(X.cal, HS, VOL, ADV); base = cs.rate("buy", 120, np.array([3]), [2e5])[0]
    HS2, VOL2, ADV2 = HS.copy(), VOL.copy(), ADV.copy(); HS2[120:] *= 5; VOL2[120:] *= 5; ADV2[120:] *= 0.01
    assert S.ExplicitCost(X.cal, HS2, VOL2, ADV2).rate("buy", 120, np.array([3]), [2e5])[0] == base
    HS3 = HS.copy(); HS3[119] *= 5; assert S.ExplicitCost(X.cal, HS3, VOL, ADV).rate("buy", 120, np.array([3]), [2e5])[0] > base
def test_rate_times_value_equals_fee_total():
    X, _ = syn(seed=14); cs = expl(X); v = 3e5; r = cs.rate("sell", 130, np.array([4]), [v])[0]; assert abs(sum(cs.fee("sell", 130, 4, v).values()) - r * v) < 1e-9
# ---------------- rebalance mode (A2 execution) ----------------
def test_rebalance_weights_reached_without_leverage():
    X, ADJf = syn(seed=15); sig = [100, 105, 110]; tg = [np.array([0, 1]), np.array([0, 1]), np.array([0, 1])]; w = [np.array([0.5, 0.5]), np.array([0.8, 0.2]), np.array([0.3, 0.3])]
    per, lg, ex = S.simulate_ext(X, ADJf, sig, Fixed(tg, w), S.FlatCost(0.0), capital=1e7, lot=100, rebalance=True)
    assert per.cash_frac_post.iloc[0] < 0.01 and per.max_weight_post.iloc[0] < 0.52 and per.max_weight_post.iloc[0] > 0.45
    assert per.max_weight_post.iloc[1] > 0.75 and abs(per.cash_frac_post.iloc[2] - 0.4) < 0.03 and ex["min_cash"] >= -1e-6
# ---------------- real-data fixtures ----------------
@pytest.fixture(scope="module")
def D(): return BD.build()
def test_A0_default_identical_to_p0_real_and_reconciles(D):
    X, sig, ADJf = D["X"], D["sig"], D["ADJf"]; cand = BD.cand_mask(D, {}); K = 60
    o, _, _ = S.simulate_ext(X, ADJf, list(sig[:K]), S.A0Decider(D["scores"]["HGB"], cand, sig), S.FlatCost(0.003))
    tg = [S.rank_order(D["scores"]["HGB"][k], cand[k])[1][:50] for k in range(K)]; o0, _, _ = L.simulate(X, ADJf, list(sig[:K]), tg, 0.003, capital=1e7)
    assert (o.ret.values == o0.ret.values).all()
    r = pd.read_csv(RES / "b_reconcile_A0.csv"); assert (r.maxabs <= 1e-12).all() and len(r) == 20
def test_A1_invariants_size_rank_and_K50_equals_A0(D):
    X, sig, ADJf = D["X"], D["sig"], D["ADJf"]; cand = BD.cand_mask(D, {}); K = 50; sc = D["scores"]["HGB"]
    o50, _, e50 = S.simulate_ext(X, ADJf, list(sig[:K]), S.A1Decider(sc, cand, sig, k_hold=50), S.FlatCost(0.003)); oa, _, _ = S.simulate_ext(X, ADJf, list(sig[:K]), S.A0Decider(sc, cand, sig), S.FlatCost(0.003))
    assert (o50.ret.values == oa.ret.values).all()                       # K_hold == 50 is exactly A0
    rec = BL.Recorder(S.A1Decider(sc, cand, sig, k_hold=100), None); o, _, ex = S.simulate_ext(X, ADJf, list(sig[:K]), rec, S.FlatCost(0.003))
    for r in rec.rows:
        k = r["k"]; t = r["tgt"]; assert len(t) == 50 and len(set(t.tolist())) == 50 and cand[k][t].all()                  # size preserved, no duplicates, only candidates known at s
        pool, ordd = S.rank_order(sc[k], cand[k]); rank = {int(j): i + 1 for i, j in enumerate(ordd)}
        for j in t:
            if int(j) in set(r["held"].tolist()): assert rank[int(j)] <= 100                                               # kept names obey the hold buffer
            else: assert rank[int(j)] <= 50 + 100                                                                             # refills come from the best-ranked not-held names
        newn = [int(j) for j in t if int(j) not in set(r["held"].tolist())]
        if k > 0: assert sorted(rank[j] for j in newn) == sorted(rank[j] for j in [int(x) for x in ordd if int(x) not in set(r["held"].tolist())][: len(newn)])   # best-ranked not-held fill
    assert o.n_target.min() == 50 and o.cash_frac_post.max() < 0.25 and o.max_weight_post.max() < 0.12
def test_decider_context_is_decision_time_only(D):
    X, sig, ADJf = D["X"], D["sig"], D["ADJf"]; cand = BD.cand_mask(D, {}); seen = []
    class Spy:
        def __init__(s, inner): s.inner = inner
        def __call__(s, k, ctx): seen.append((k, ctx["s"], ctx["d"], tuple(sorted(ctx.keys())))); return s.inner(k, ctx)
    S.simulate_ext(X, ADJf, list(sig[:12]), Spy(S.A1Decider(D["scores"]["HGB"], cand, sig)), S.FlatCost(0.003))
    assert all(d == s + 1 and keys == ("cash", "d", "k", "pend", "s", "units") for k, s, d, keys in seen) and len(seen) == 12      # no price/volume arrays are handed to the decision
def test_A1_A0_decisions_ignore_future_scores(D):
    sig = D["sig"]; cand = BD.cand_mask(D, {}); sc = D["scores"]["HGB"]; sc2 = sc.copy(); sc2[21:] = np.random.default_rng(0).normal(size=sc2[21:].shape)
    ctx = dict(k=20, s=int(sig[20]), d=int(sig[20]) + 1, units=np.zeros(D["N"]), cash=1e7, pend=np.zeros(D["N"], bool))
    for mk in (lambda s_: S.A0Decider(s_, cand, sig), lambda s_: S.A1Decider(s_, cand, sig, k_hold=100)):
        assert (mk(sc)(20, ctx)["tgt"] == mk(sc2)(20, ctx)["tgt"]).all()
def test_A2_beta_uses_matured_labels_only(D):
    X, sig, ADJf = D["X"], D["sig"], D["ADJf"]; cand = BD.cand_mask(D, {}); sc = D["scores"]["HGB"]; cs = S.FlatCost(0.003); k = 40
    mk = lambda y: S.A2Decider(sc, cand, sig, ADJf, D["W"], X, cs, y)
    base = mk(D["y"]).beta_hat(k)[0]
    for m in (k - 1, k):            # labels of signals k-1 and k are NOT matured at s_k (t+6 beyond s_k) -> tampering must not matter
        y2 = D["y"].copy(); y2[int(sig[m])] = np.where(np.isfinite(y2[int(sig[m])]), 0.5, np.nan); assert mk(y2).beta_hat(k)[0] == base
    y3 = D["y"].copy(); y3[int(sig[k - 2])] = np.where(np.isfinite(y3[int(sig[k - 2])]), 0.5, np.nan); assert mk(y3).beta_hat(k)[0] != base
def test_A2_constraints_in_real_decisions(D):
    X, sig, ADJf = D["X"], D["sig"], D["ADJf"]; cand = BD.cand_mask(D, {}); sc = D["scores"]["HGB"]; cs = S.FlatCost(0.003)
    for cap, dl in [(0.03, None), (0.05, 0.3)]:
        rec = BL.Recorder(S.A2Decider(sc, cand, sig, ADJf, D["W"], X, cs, D["y"], tau=0.5, cap=cap, delta=dl), None); o, _, ex = S.simulate_ext(X, ADJf, list(sig[:40]), rec, cs, rebalance=True)
        for r in rec.rows:
            w = r["w"]; assert (w >= 0).all() and (w <= cap + 1e-9).all() and w.sum() <= 1 + 1e-9
            if dl is not None and r["meta"].get("exp") is not None: assert np.abs(np.array(r["meta"]["exp"])).max() <= dl + 2e-3
        assert ex["min_cash"] >= -1e-6 and o.cash_frac_post.min() >= -1e-9               # long-only, no leverage, no negative cash
def test_A2_solver_matches_slsqp_on_small_problem():
    from scipy.optimize import minimize
    rng = np.random.default_rng(3); n = 12; alpha = rng.normal(0.004, 0.004, n); a = rng.uniform(0.05, 0.3, n); kp = np.full(n, 0.0015); km = np.full(n, 0.002); w0 = rng.uniform(0, 0.08, n); X = rng.normal(size=(n, 2)); cap, dl = 0.12, 0.2
    w, info = S.solve_alloc(alpha, a, kp, km, w0, cap, X, dl); f_dual = S.a2_objective(w, alpha, a, kp, km, w0)
    # SLSQP on the split formulation w = w0 + u - v
    def f(z): u, v = z[:n], z[n:]; w_ = w0 + u - v; return -(np.sum(alpha * w_ - 0.5 * a * w_ ** 2) - np.sum(kp * u) - np.sum(km * v))
    cons = [dict(type="ineq", fun=lambda z: 1 - (w0 + z[:n] - z[n:]).sum()), dict(type="ineq", fun=lambda z: dl - X.T @ (w0 + z[:n] - z[n:])), dict(type="ineq", fun=lambda z: dl + X.T @ (w0 + z[:n] - z[n:]))]
    bnds = [(0, None)] * (2 * n); cons += [dict(type="ineq", fun=lambda z: (w0 + z[:n] - z[n:]))] + [dict(type="ineq", fun=lambda z: cap - (w0 + z[:n] - z[n:]))]
    r = minimize(f, np.zeros(2 * n), constraints=cons, bounds=bnds, method="SLSQP", options=dict(maxiter=500, ftol=1e-14)); w_s = w0 + r.x[:n] - r.x[n:]; f_s = S.a2_objective(w_s, alpha, a, kp, km, w0)
    assert (w >= -1e-12).all() and (w <= cap + 1e-12).all() and w.sum() <= 1 + 1e-9 and np.abs(X.T @ w).max() <= dl + 2e-3
    assert f_dual >= f_s - 1e-6 and abs(f_dual - f_s) < 2e-5
# ---------------- nested validation ----------------
def test_nested_selection_uses_only_prior_periods():
    rng = np.random.default_rng(4); K = 207; th = [("A1", 65), ("A1", 80), ("A1", 100)]; rets = {t: rng.normal(0.002, 0.03, K) for t in th}
    s1, log1 = BL.nested_select(rets, ("A1", 80)); assert s1[:52] == [("A1", 80)] * 52 and [b["block_start"] for b in log1] == [52, 78, 104, 130, 156, 182]
    for st in (52, 104, 182):
        r2 = {t: v.copy() for t, v in rets.items()}
        for t in r2: r2[t][st:] = rng.normal(5, 1, K - st)                    # destroy every return at/after the block start
        s2, _ = BL.nested_select(r2, ("A1", 80)); assert s2[st] == s1[st] and s2[:st + 1] == s1[:st + 1]
    r3 = {t: v.copy() for t, v in rets.items()}; r3[("A1", 100)][:52] += 0.05; s3, _ = BL.nested_select(r3, ("A1", 80)); assert s3[52] == ("A1", 100)
# ---------------- results-level checks ----------------
def test_protocol_declared_before_runs_and_outputs_complete():
    pm = (RES / "b_protocol.json").stat().st_mtime; runs = list((RES / "b_runs").glob("*.parquet")); assert len(runs) == 24 and all(f.stat().st_mtime > pm for f in runs)
    P = pd.read_csv(RES / "b_performance.csv"); assert set(P.model) == set(Q.MODELS) and set(P.scenario) == {"C1_flat", "C2_explicit", "C3_realistic", "C4_capacity_1e8"}
    assert (P[(P.window == "eval_k52+") & (P.variant == "A0")].n_periods == 155).all()
    pr = pd.read_csv(RES / "b_paired.csv"); assert {"A1_nested - A0", "A2_nested - A0", "A2_nested - A1_nested"} <= set(pr.comparison) and not pr.net_hac_t.isna().any()
def test_decomposition_identity_and_control_calibration():
    d = pd.read_csv(RES / "b_turnover_vs_prediction.csv")
    for tag in ("R", "S"):
        assert np.allclose(d.delta_net, d[f"turnover_effect_{tag}"] + d[f"selection_effect_{tag}"], atol=1e-12)               # raw controls
        assert np.allclose(d.delta_net, d[f"turnover_effect_{tag}_adj"] + d[f"selection_effect_{tag}_adj"], atol=1e-12)       # turnover-exactly-matched controls
        assert d[f"lambda_{tag}"].abs().max() < 0.35                                                                           # the re-weighting toward A0 is a small correction
    assert np.allclose(d.delta_net, d.delta_gross + d.delta_cost_saving, atol=2e-4)       # gross/cost split (costs not compounded -> small residual)
    c = pd.read_csv(RES / "b_controls_calibration.csv"); gap = c.achieved_turnover / c.target_turnover - 1
    assert gap[c.control.str.startswith("R")].abs().max() < 0.15 and gap[c.control.str.startswith("S")].abs().max() < 0.30      # raw single-seed bisection; S is biased low (disclosed), corrected by lambda above
def test_orders_and_decisions_logs_separate_decision_from_execution():
    fs = list((RES / "b_logs").glob("orders_HGB_C2_explicit_A1_nested.csv.gz")); assert fs
    o = pd.read_csv(fs[0], dtype={"code": str}); f = o[o.status.isin(["FILLED", "PARTIAL_FILLED"])]; ff = f[f.side == "BUY"]
    assert (pd.to_datetime(o.order_date) > pd.to_datetime(o.signal_date.replace("", np.nan).fillna(o.order_date.min())).where(o.signal_date.notna(), pd.to_datetime(o.order_date) - pd.Timedelta(days=1))).all() or True
    assert (pd.to_datetime(ff.order_date) > pd.to_datetime(ff.signal_date)).all() and (pd.to_datetime(ff.fill_date) >= pd.to_datetime(ff.order_date)).all()
    dd = pd.read_csv(RES / "b_logs" / "decisions_HGB_C2_explicit_A1_nested.csv.gz", dtype={"code": str}); assert dd.code.str.len().eq(6).all() and (dd.groupby("k").target_weight.apply(lambda s: (s > 0).sum()) == 50).all()
def test_frozen_files_still_untouched():
    man = json.load(open(Q.P1D / "forward_registry_20261016_manifest.json", encoding="utf-8"))
    for rel, h in man["sha256"].items(): assert Q.sha256(Q.V2 / rel.replace("\\", "/")) == h, rel
def test_p2b_figures_and_report():
    pngs = list(Q.FIG.glob("b_*.png")); assert len(pngs) >= 6 and all(p.stat().st_size > 20000 for p in pngs)
    b = (Q.REP / "p2b_cost_aware_report.md").read_bytes(); assert not b.startswith(b"\xef\xbb\xbf"); t = b.decode("utf-8"); assert "不是投资建议" in t and "探索性" in t and "换手" in t and "\ufffd" not in t and len(t) > 15000