# -*- coding: utf-8 -*-
"""P4 batch-1 tests: label maturity / no look-ahead / engine rules (synthetic + real cache)."""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd, pytest
P4 = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(P4)); import p4lib as P
cal = pd.bdate_range("2021-01-04", periods=1000)
@pytest.mark.parametrize("h", [5, 20])
def test_train_days_matured(h):
    for i in [300, 600, 999]:
        days = P.train_days(i, h, cal); assert P.label_end_index(max(days), h) < i and max(days) <= i - h - 2
def test_features_no_lookahead():
    # changing data AFTER t must not change features at t
    import types
    rng = np.random.default_rng(0); T, N = 200, 30
    def mk(seed):
        r = np.random.default_rng(seed); X = types.SimpleNamespace(); C = 10 * np.cumprod(1 + r.normal(0, .02, (T, N)), 0)
        X.C = C.copy(); X.O = C * 1.001; X.Hh = C * 1.01; X.L = C * 0.99; X.V = np.full((T, N), 1e6); X.AMT = X.C * X.V; X.TURN = np.full((T, N), 1.0); X.F = np.ones((T, N))
        X.traded = np.ones((T, N), bool); X.listed = X.traded; X.ADJ_unadj_based = X.C.copy(); X.u = pd.DataFrame(dict(industry_l2=["A"] * 15 + ["B"] * 15)); return X
    A = mk(1); f1 = P.features(A)[0]; B = mk(1)
    for k in ["C", "O", "Hh", "L", "V", "AMT", "TURN", "ADJ_unadj_based"]: getattr(B, k)[151:] *= 3.0
    f2 = P.features(B)[0]
    for k in f1:
        a, b = f1[k].values[:151], f2[k].values[:151]; assert np.allclose(np.nan_to_num(a), np.nan_to_num(b)), k
def test_labels_use_future_open_only():
    import types
    T, N = 50, 3; X = types.SimpleNamespace(O=np.tile(np.arange(1, T + 1, dtype=float)[:, None], (1, N)), F=np.ones((T, N)), traded=np.ones((T, N), bool))
    lab = P.labels(X, np.ones((T, N), bool), 5)[0]; assert np.isclose(lab[10, 0], X.O[16, 0] / X.O[11, 0] - 1) and np.isnan(lab[T - 6:]).all()
def _real():
    C = P4 / "cache"
    if not (C / "flags.npz").exists(): pytest.skip("cache not built")
    return C
def test_walkforward_meta_matured():
    for f in (P4 / "preds").glob("*_meta.json"):
        m = json.load(open(f, encoding="utf-8"))
        for r in m["retrains"]: assert r["label_end_max"] < r["retrain_signal_date"]
        assert m["final_train"]["last"] < "2026-10-21"
def test_preds_label_end_before_signal():
    for f in (P4 / "preds").glob("E1*_h*.parquet"):
        d = pd.read_parquet(f, columns=["signal_date", "label_end_max"]); assert (d.label_end_max < d.signal_date).all(), f
def test_engine_skips_and_costs():
    import types
    T, N = 30, 4; X = types.SimpleNamespace(C=np.full((T, N), 10.0), O=np.full((T, N), 10.0), F=np.ones((T, N)), traded=np.ones((T, N), bool), cal=pd.bdate_range("2024-01-01", periods=T), codes=list("abcd"))
    X.O[7:, 0] = 11.0; X.traded[2, 1] = False; olu = np.zeros((T, N), bool); olu[2, 2] = True; old = np.zeros((T, N), bool); old[7, 3] = True; X.O[8:, 3] = 9.0
    pr, ho = P.backtest(X, [1], [[0, 1, 2, 3]], 5, 4, 0.003, skip=None, olu=olu, old=old)
    st = dict(zip(ho.code, ho.status)); assert st["b"] == "SKIPPED_suspended_at_entry" and st["c"] == "SKIPPED_open_limit_up"
    a = ho[ho.code == "a"].iloc[0]; assert np.isclose(a.slot_ret_gross, 0.1) and np.isclose(a.slot_ret_net, 1.1 * 0.9985 * 0.9985 - 1)
    d = ho[ho.code == "d"].iloc[0]; assert d.exit_delay_days == 1 and np.isclose(d.slot_ret_gross, -0.1)
    assert np.isclose(pr.ret_gross.iloc[0], (0.1 - 0.1) / 4)
