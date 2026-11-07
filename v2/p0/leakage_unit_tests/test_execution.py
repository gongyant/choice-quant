"""P0-B / P0-C / P0-E / P0-F execution-simulator and price-rule tests (synthetic fixtures + real-data sanity checks)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pytest
import p0lib as L
from synth import make_synth, pipeline
def prep(seed=5, N=30):
    X = make_synth(N=N, seed=seed, suspensions=False); return X
def run(X, sig, tg, cost=0.0, lot=1, cap=1e6, name="t"):
    L.finalize(X); L.limit_flags(X); f, elig, ADJf = L.compute_features(X, X.ADJ_unadj_based)
    return L.simulate(X, ADJf, sig, tg, cost, capital=cap, lot=lot, log_name=name)
def kill_bar(X, d, j):
    for M in (X.O, X.Hh, X.L, X.C, X.V, X.AMT, X.TURN): M[d, j] = np.nan
def lock_up(X, d, j, pct=0.10):
    X.C[d, j] = L.round_cent(X.C[d - 1, j] * (1 + pct)); X.Hh[d, j] = X.C[d, j]; X.O[d, j] = min(X.O[d, j], X.C[d, j]); X.L[d, j] = min(X.L[d, j], X.C[d, j])
def lock_dn(X, d, j, pct=0.10):
    X.C[d, j] = L.round_cent(X.C[d - 1, j] * (1 - pct)); X.L[d, j] = X.C[d, j]; X.Hh[d, j] = max(X.Hh[d, j], X.C[d, j]); X.O[d, j] = max(X.O[d, j], X.C[d, j])
def test_buy_failure_leaves_cash_and_never_substitutes_other_stock():            # P0-A / P0-B
    X = prep(); kill_bar(X, 101, 2); lock_up(X, 101, 3)
    per, lg, ex = run(X, [100, 105], [np.arange(5), np.arange(5, 10)])
    b = lg[(lg.side == "BUY") & (lg.order_date == str(X.cal[101].date()))]
    assert set(b[b.status == "FILLED"].code) == {"S000", "S001", "S004"}            # no substitute names were bought
    assert dict(zip(b[b.status == "REJECTED"].code, b[b.status == "REJECTED"].reason)) == {"S002": "suspended_or_no_bar", "S003": "limit_up_locked"}
    assert per.loc[0, "buy_filled"] == 3 and per.loc[0, "buy_attempts"] == 5 and abs(per.loc[0, "cash_frac_post"] - 0.4) < 0.02
    assert (b[b.status == "FILLED"].fill_date == str(X.cal[101].date())).all()     # buy at t+1 close on the calendar
def test_blocked_sell_is_delayed_and_position_keeps_risk():                      # P0-B / P0-F
    X = prep(seed=6); lock_dn(X, 106, 1); lock_dn(X, 107, 1)
    per, lg, ex = run(X, [100, 105], [np.arange(5), np.arange(5, 10)])
    s = lg[(lg.code == "S001") & (lg.side == "SELL")]
    assert list(s.status) == ["BLOCKED_RETRY", "FILLED"] and s.iloc[0].reason == "limit_down_locked"
    assert s.iloc[1].fill_date == str(X.cal[108].date()) and s.iloc[1].sell_delay_days == 2
    assert per.loc[1, "n_pending_exit"] == 1                                          # still held (and exposed to price moves) after the order day
    other = lg[(lg.side == "SELL") & (lg.code != "S001") & (lg.status == "FILLED")]
    assert (other.fill_date == str(X.cal[106].date())).all() and (other.sell_delay_days == 0).all()   # scheduled exit = signal t+6 on the calendar
def test_suspended_sell_waits_until_bar_exists():
    X = prep(seed=7)
    for d in (106, 107, 108): kill_bar(X, d, 0)
    per, lg, ex = run(X, [100, 105], [np.arange(5), np.arange(5, 10)])
    s = lg[(lg.code == "S000") & (lg.side == "SELL") & (lg.status == "FILLED")]
    assert s.iloc[0].fill_date == str(X.cal[109].date()) and s.iloc[0].sell_delay_days == 3
def test_ex_rights_day_is_not_a_limit_down_but_a_real_locked_limit_down_is():      # P0-F / P0-C
    X = prep(seed=8); t = 120; j = 0; prev = X.C[t - 1, j]
    X.C[t, j] = L.round_cent(prev * 0.56); X.L[t, j] = X.C[t, j]; X.Hh[t, j] = max(X.Hh[t, j], X.C[t, j]); X.O[t, j] = X.C[t, j]
    Y = __import__("copy").deepcopy(X)
    X.F[t:, j] = 0.58; L.finalize(X); L.limit_flags(X); assert not X.lim_dn[t, j]    # reference price = prev * 0.58 -> -44% raw drop is only ~-3.4% vs reference
    L.finalize(Y); L.limit_flags(Y); assert Y.lim_dn[t, j]                              # naive raw/prev rule (the old approach) would call it locked limit-down
    Z = __import__("copy").deepcopy(X); ref = prev * 0.58; Z.C[t, j] = L.round_cent(ref * 0.9); Z.L[t, j] = Z.C[t, j]; Z.O[t, j] = max(Z.O[t, j], Z.C[t, j]); Z.Hh[t, j] = max(Z.Hh[t, j], Z.C[t, j])
    L.finalize(Z); L.limit_flags(Z); assert Z.lim_dn[t, j]
def test_board_specific_limits_and_conservative_ST_5pct():
    X = prep(seed=9); t = 120
    lock_up(X, t, 0, 0.05); lock_up(X, t, 1, 0.07); lock_up(X, t, 2, 0.10)         # main board: exact 5% limit price (ST proxy) / 7% / 10%
    X.u.loc[X.codes[3], "board"] = "科创板"; X.u.loc[X.codes[4], "board"] = "科创板"; lock_up(X, t, 3, 0.10); lock_up(X, t, 4, 0.20)
    X.u.loc[X.codes[5], "board"] = "北交所"; X.u.loc[X.codes[6], "board"] = "北交所"; lock_up(X, t, 5, 0.20); lock_up(X, t, 6, 0.30)
    L.finalize(X); L.limit_flags(X); r = X.lim_up[t]
    assert r[0] and (not r[1]) and r[2] and (not r[3]) and r[4] and (not r[5]) and r[6]
def test_new_listing_no_limit_window():
    X = prep(seed=10); j = 0; X.u.loc[X.codes[j], "board"] = "创业板"; X.u.loc[X.codes[j], "list_date"] = X.cal[100]
    for M in (X.O, X.Hh, X.L, X.C, X.V, X.AMT, X.TURN): M[:100, j] = np.nan
    lock_up(X, 102, j, 0.20); lock_up(X, 110, j, 0.20); L.finalize(X); L.limit_flags(X)
    assert not X.lim_up[102, j] and X.lim_up[110, j]
def test_cost_is_charged_on_executed_trades_only():
    X = prep(seed=11); X.C[:] = 10.0; X.O[:] = 10.0; X.Hh[:] = 10.1; X.L[:] = 9.9
    per, lg, ex = run(X, [100, 105, 110], [np.arange(5)] * 3, cost=0.003, lot=1)
    assert abs(per.ret[0] + 0.0015 / 1.0015) < 2e-4 and abs(per.ret[1]) < 1e-9 and abs(per.ret[2]) < 1e-9   # retained names are not re-traded: no further cost
    per0, _, _ = run(X, [100, 105, 110], [np.arange(5)] * 3, cost=0.0, lot=1); assert abs(per0.ret).max() < 1e-9
def test_equal_weight_benchmark_pays_costs_in_same_simulator():                   # P0-E
    X = prep(seed=12); L.finalize(X); L.limit_flags(X); f, elig, ADJf = L.compute_features(X, X.ADJ_unadj_based)
    sig = [100, 105, 110]; pool = [np.arange(30)] * 3
    a, _, _ = L.simulate(X, ADJf, sig, pool, 0.0, capital=1e9, lot=100); b, _, _ = L.simulate(X, ADJf, sig, pool, 0.003, capital=1e9, lot=100)
    assert b.ret[0] < a.ret[0] and a.buy_attempts[0] == 30
def test_lot_size_100_shares():
    X = prep(seed=13); per, lg, ex = run(X, [100, 105], [np.arange(5), np.arange(5, 10)], cost=0.003, lot=100, cap=3e6, name="t")
    f = lg[(lg.side == "BUY") & (lg.status == "FILLED")]; assert len(f) > 0 and ((f.shares % 100) == 0).all()
def test_factor_table_and_adjusted_return_continuity():                            # P0-C
    cal = pd.bdate_range("2022-02-24", periods=8)
    fac = pd.DataFrame({"date": pd.to_datetime(["1900-01-01", "2022-03-01"]), "qfq_factor": [2.0, 1.0]})
    f = L.factor_on_dates(cal, fac); assert list(f[:4]) == [2.0, 2.0, 2.0, 1.0] and cal[3] == pd.Timestamp("2022-03-01")
    raw = np.array([10.0, 10.2, 5.2, 5.3]); F = np.array([2.0, 2.0, 1.0, 1.0]); adj = raw / F          # 10.2 -> 5.2 is a 2:1 split-like ex-rights day (+1.96% in total-return terms)
    assert abs(adj[2] / adj[1] - 1 - (5.2 * 2 / 10.2 - 1)) < 1e-12 and abs(adj[2] / adj[1] - 1) < 0.03 and (adj > 0).all()
def test_snap_pow10_volume_unit_inference():
    assert L.snap_pow10(1.002) == 1.0 and L.snap_pow10(99.4) == 100.0 and L.snap_pow10(0.0099) == 0.01 and np.isnan(L.snap_pow10(3.0))
RU3 = [c for c in ("600188", "600066", "000408") if (L.RU / f"{c}.parquet").exists() and (L.FA / f"{c}.parquet").exists()]
@pytest.mark.skipif(len(RU3) < 3, reason="unadjusted data for the 3 invalid-price stocks not downloaded")
def test_real_data_three_invalid_price_stocks_fixed():
    for c in RU3:
        d = L.load_raw(c); f = L.factor_on_dates(d.date, L.load_factor(c)); adj = d.close.values / f
        assert (d[["open", "high", "low", "close"]].values > 0).all() and (adj > 0).all()
        r = pd.Series(adj).pct_change().abs().max(); assert r <= 0.105, (c, r)      # within the 10% main-board limit (+ tick rounding)
@pytest.mark.skipif(not any(L.RU.glob("*.parquet")), reason="no unadjusted data")
def test_real_data_no_nonpositive_raw_prices_on_traded_bars():
    bad = []
    for p in list(L.RU.glob("*.parquet")):
        d = pd.read_parquet(p); v = d.volume if "volume" in d else d.volume_raw
        m = (v > 0) & (d.amount > 0)
        if ((d.loc[m, ["open", "high", "low", "close"]] <= 0).any(axis=1)).any(): bad.append(p.stem)
    assert bad == [], bad

