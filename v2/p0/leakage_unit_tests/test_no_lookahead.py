"""P0-A / P0-B / P0-E look-ahead tests.  Rule: anything known at signal time t (eligible_asof_t, features, ranks, scores, trained-model predictions)
must be IDENTICAL if all data dated after t is tampered with.  Negative controls prove the harness would catch a leak."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pytest
from sklearn.linear_model import Ridge
import p0lib as L
from synth import make_synth, pipeline, tamper_future, assert_asof
X0 = make_synth(); T = len(X0.cal); TS = [90, 110, 130, T - 20]
# ---- production functions under test -------------------------------------------------------------------------------------------------
def f_elig(X, t): return pipeline(X)["elig"][: t + 1]
def f_features(X, t): return np.concatenate([pipeline(X)["f"][k].values[: t + 1][None] for k in L.FEATS], axis=0)
def f_rank(X, t): return pipeline(X)["Xq"][: t + 1]
def f_scores_mom_rev(X, t):
    p = pipeline(X); return np.stack([p["f"]["r20"].values[t], -p["f"]["r5"].values[t]])
def f_limit_flags(X, t):   # limit status of bar t itself is known at t (uses prev close and ex-rights factor ratio up to t)
    pipeline(X); return np.stack([X.lim_up[: t + 1], X.lim_dn[: t + 1]])
def f_ridge_prediction(X, t, purge=12):
    p = pipeline(X); m_tr = p["elig"] & np.isfinite(p["yw"])
    if purge == 12: Xt, yt = L.train_rows(p["Xq"], p["yw"], m_tr, X.cal, t)
    else:
        lo = X.cal.searchsorted(X.cal[t] - pd.Timedelta(days=3 * 365)); rows = [(p["Xq"][d][m_tr[d]], p["yw"][d][m_tr[d]]) for d in range(lo, t - purge + 1) if m_tr[d].any()]
        Xt = np.concatenate([r[0] for r in rows]); yt = np.concatenate([r[1] for r in rows])
    m = Ridge(alpha=50.0).fit(Xt, yt); pool = np.where(p["elig"][t])[0]
    return m.predict(p["Xq"][t][pool])
# ---- negative controls: functions that deliberately use the future ------------------------------------------------------------------
def leaky_elig_old_style(X, t):
    """The ORIGINAL baseline.py bug (P0-A): signal-day pool filtered with t+1 buy volume / limit-up and t+6 exit volume."""
    p = pipeline(X); return (p["elig"] & X.traded[:, :] )[: t + 1] & np.vstack([X.traded[1:] & ~X.lim_up[1:] , np.zeros((1, X.C.shape[1]), bool)])[: t + 1]
def leaky_signal_next_return(X, t):
    p = pipeline(X); A = p["ADJf"]; return A[t + 1] / A[t] - 1
def leaky_elig_t6_exit(X, t):
    p = pipeline(X); out = p["elig"][: t + 1].copy(); out[-1] &= X.traded[min(t + 6, len(X.cal) - 1)]; return out
# ---- tests ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("t", TS)
@pytest.mark.parametrize("how", ["garbage", "nan"])
def test_eligible_asof_t_ignores_future(t, how): assert_asof(f_elig, X0, t, how)
@pytest.mark.parametrize("t", TS)
def test_features_ignore_future(t): assert_asof(f_features, X0, t, "garbage")
@pytest.mark.parametrize("t", TS)
def test_rank_features_ignore_future(t): assert_asof(f_rank, X0, t, "garbage")
@pytest.mark.parametrize("t", TS)
def test_momentum_reversal_scores_ignore_future(t): assert_asof(f_scores_mom_rev, X0, t, "garbage")
@pytest.mark.parametrize("t", TS[:3])
def test_limit_flags_of_past_bars_ignore_future(t): assert_asof(f_limit_flags, X0, t, "garbage")
@pytest.mark.parametrize("t", [120, 135])
def test_trained_model_prediction_ignores_future(t): assert_asof(f_ridge_prediction, X0, t, "garbage")
def test_history_rescaling_by_future_corporate_action_does_not_change_features():
    """Cumulative qfq factors are anchored at today: a FUTURE ex-date rescales the whole history. Return-type features must be invariant."""
    import copy
    Y = copy.deepcopy(X0); t = 120; Y.F[: t + 1] *= 2.0; L.finalize(Y)   # as if a 50% bonus happened after t
    a, b = pipeline(X0), pipeline(Y)
    for k in ["r1", "r5", "r20", "r60", "vol20", "ma_gap5", "ma_gap20", "volume_ratio5", "turnover5", "log_amt20", "range20"]:
        np.testing.assert_allclose(a["f"][k].values[: t + 1], b["f"][k].values[: t + 1], rtol=1e-9, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(a["elig"][: t + 1], b["elig"][: t + 1])
def test_label_uses_only_t1_to_t6_on_master_calendar():
    p = pipeline(X0); t = 115; Y = tamper_future(X0, t + 6, how="garbage"); q = pipeline(Y)   # tamper strictly after t+6
    np.testing.assert_allclose(p["lab"][t], q["lab"][t], equal_nan=True)
    Z = tamper_future(X0, t, how="garbage"); r = pipeline(Z)                                   # tamper after t: label MAY change (it is the target)
    assert not np.allclose(p["lab"][t], r["lab"][t], equal_nan=True)
    A = p["ADJf"]; np.testing.assert_allclose(p["lab"][t], A[t + 6] / A[t + 1] - 1, equal_nan=True)
def test_calendar_offset_differs_from_per_stock_row_shift():
    """P0-B: old code used groupby(code).shift(-6) (row offset). For a stock with a suspension gap that is NOT the calendar t+6."""
    X = make_synth(seed=3, suspensions=False); j = 0; a = 100; X.C[a:a + 3, j] = np.nan
    for M in (X.O, X.Hh, X.L, X.V, X.AMT, X.TURN): M[a:a + 3, j] = np.nan
    L.finalize(X); p = pipeline(X); rows = np.where(X.has_row[:, j])[0]; t = 98
    naive_exit_idx = rows[np.where(rows == t)[0][0] + 6]; assert naive_exit_idx != t + 6       # the row-shift lands on the wrong calendar day
    assert np.isclose(p["lab"][t, j], p["ADJf"][t + 6, j] / p["ADJf"][t + 1, j] - 1)
# ---- negative controls (the harness MUST fail on look-ahead) -------------------------------------------------------------------------
def test_NEGATIVE_old_style_future_filter_is_detected():
    with pytest.raises(AssertionError): assert_asof(leaky_elig_old_style, X0, 110, "garbage")
def test_NEGATIVE_next_day_return_signal_is_detected():
    with pytest.raises(AssertionError): assert_asof(leaky_signal_next_return, X0, 110, "garbage")
def test_NEGATIVE_t6_exit_filter_is_detected():
    with pytest.raises(AssertionError): assert_asof(leaky_elig_t6_exit, X0, 110, "nan")
def test_NEGATIVE_training_without_purge_is_detected():
    with pytest.raises(AssertionError): assert_asof(lambda X, t: f_ridge_prediction(X, t, purge=0), X0, 120, "garbage")

