# -*- coding: utf-8 -*-
"""Declare the P2-B protocol BEFORE the main runs (grids, nested-validation design, cost assumptions, constraint defaults). Writes results/b_protocol.json with sha256."""
import json, hashlib, datetime as dt
import p2lib as Q
P = dict(
  declared_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M (Asia/Singapore)"),
  note="Parameters are declared before the main grid runs. Only the cost-model shrink hs_mult=0.25 was set after an unshrunk Corwin-Schultz check showed median half-spreads of ~0.43% (implausible vs 1-2 ticks); it was fixed for all strategies and is stress-tested (0.5x/2x).",
  signals=dict(n=207, spacing_trading_days=5, horizon_days=5, first="2022-07-01", last="2026-09-28"),
  capital=1e7, lot=100, topn=50,
  strategies=dict(
    A0="naive Top50 (equal weight, buy rank<=50, sell when out of Top50; retained names not rebalanced) = P0 baseline_fixed state machine",
    A1="Top50 + hysteresis: buy rank<=50, held name kept while rank<=K_hold, target always exactly 50 names (kept names in best-rank order, remaining slots refilled by best-ranked NOT-held names)",
    A2="cost-aware long-only optimiser (own dual solver): alpha=beta_hat*z, beta_hat from matured labels only; diag 5d risk lam=25; turnover cost tau*side cost; cap per name; sum(w)<=1; optional |exposure|<=delta to log_amt20/vol20/r5; candidates top-100 by score + holdings"),
  grids=dict(A1_K_hold=[65, 80, 100, 150], A2_tau=[0.25, 0.5, 1.0], A2_cap=[0.03, 0.05], A2_delta=[None, 0.3]),
  defaults_burn_in=dict(A1_K_hold=80, A2=dict(tau=0.5, cap=0.05, delta=None)),
  A2_fixed=dict(lam=25.0, M=100, beta0=0.003, beta_floor=0.0005, min_obs=8, min_trade_frac=0.002),
  nested=dict(scheme="expanding window, anchored at first signal; burn-in 52 periods (default params); test blocks of 26 periods (k=52..77, 78..103, 104..129, 130..155, 156..181, 182..206); for each block the parameter maximising net Sharpe (mean/std of per-period net return) over ALL periods before the block start is used; selection counterfactuals are parameter-fixed full runs, evaluation is a SCHEDULED run with actual state carry-over",
              burn_in=52, block=26, criterion="net_sharpe_per_period"),
  scenarios=dict(
    C1_flat=dict(cost="flat 0.30% round trip (0.15% per side, p0lib convention)", capital=1e7, pcap=None, min_adv=None, no_star=False, no_bj=False),
    C2_explicit=dict(cost="explicit OHLCV model", capital=1e7, pcap=None, min_adv=None, no_star=False, no_bj=False),
    C3_realistic=dict(cost="explicit OHLCV model", capital=1e7, pcap=0.05, min_adv=1e8, no_star=True, no_bj=True),
    C4_capacity_1e8=dict(cost="explicit OHLCV model", capital=1e8, pcap=0.05, min_adv=1e8, no_star=True, no_bj=True)),
  access_defaults="C1/C2: STAR(科创板) and BJ(北交所) allowed, exactly as in the P0 pool/baseline (needed for A0 to reconcile with P0); C3/C4: switches no_star/no_bj ON (conservative retail-account default), pcap=5% of prior-day ADV(20d median amount), min ADV 1e8 for candidates and buys",
  explicit_cost=dict(commission_bp=2.5, transfer_bp=0.1, stamp_bp_before_2023_08_28=10.0, stamp_bp_from_2023_08_28=5.0, spread="0.25 x Corwin-Schultz 20d (half), floor half-tick/price, cap 2% full", impact="0.5 * vol20 * sqrt(value/ADV20), cap 5%", info_set="as of d-1"),
  cost_stress=dict(hs_mult=[0.5, 2.0], imp_mult=[0.5, 2.0], where="HGB and LGBM, C2, A0/A1_nested/A2 selected schedules re-run with the same schedule"),
  controls=dict(random_retention=dict(calibrate_seed=12345, bisect_iters=8, eval_seeds=20, description="R: each held name still in pool is kept with prob p regardless of rank; S (random skip): with prob q the naive Top50 is refreshed, else the portfolio is left untouched; p / q calibrated (bisection) so eval-period (k>=52) mean one-way turnover equals that of A1_nested / A2_nested; 20 seeds averaged")),
  models=Q.MODELS, hac_lag=4, block_bootstrap=dict(block=5, B=4000, seed=Q.SEED_BOOT),
  exploratory_flag="2025-2026 is re-used data; every 2025-2026 number is exploratory, not a blind test. Nested selection never uses periods at or after the block start.")
txt = json.dumps(P, ensure_ascii=False, indent=1); (Q.RES / "b_protocol.json").write_text(txt, encoding="utf-8")
print(hashlib.sha256(txt.encode("utf-8")).hexdigest()); print(txt[:300])