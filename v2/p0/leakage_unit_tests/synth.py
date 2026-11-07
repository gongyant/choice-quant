"""Shared synthetic fixtures for the P0 unit tests (no network, no vendor data)."""
import sys, copy
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import p0lib as L
def make_synth(T=170, N=40, seed=0, suspensions=True):
    rng = np.random.default_rng(seed); cal = pd.bdate_range("2022-01-03", periods=T)
    C = np.round(10 * np.exp(np.cumsum(0.012 * rng.standard_normal((T, N)), axis=0)), 2).clip(min=0.5)
    O = np.round(C * (1 + 0.004 * rng.standard_normal((T, N))), 2); Hh = np.maximum(np.maximum(O, C), np.round(C * 1.01, 2)); Lw = np.minimum(np.minimum(O, C), np.round(C * 0.99, 2))
    V = rng.integers(8e6, 2e7, size=(T, N)).astype(float); AMT = V * C; TURN = V / 5e8
    if suspensions:
        for _ in range(25):
            j = rng.integers(0, N); a = rng.integers(70, T - 12); b = a + rng.integers(1, 5)
            for M in (O, Hh, Lw, C, V, AMT, TURN): M[a:b, j] = np.nan
    X = L.Data(); X.cal = cal; X.codes = [f"S{j:03d}" for j in range(N)]
    X.u = pd.DataFrame({"board": ["沪主板"] * N, "list_date": pd.NaT, "delist_date": pd.NaT}, index=X.codes)
    X.u["list_date"] = pd.to_datetime(X.u.list_date); X.u["delist_date"] = pd.to_datetime(X.u.delist_date)
    X.O, X.Hh, X.L, X.C, X.V, X.AMT, X.TURN = O, Hh, Lw, C, V, AMT, TURN; X.F = np.ones((T, N))
    return L.finalize(X)
def pipeline(X):
    """features -> eligibility -> rank features -> labels, exactly the production order."""
    L.finalize(X); L.limit_flags(X)
    f, elig, ADJf = L.compute_features(X, X.ADJ_unadj_based); lab, y, yw = L.make_labels(ADJf, elig); Xq = L.rank_features(f, elig)
    return dict(f=f, elig=elig, ADJf=ADJf, lab=lab, y=y, yw=yw, Xq=Xq)
def tamper_future(X, t, seed=1, how="garbage"):
    """Return a copy of X in which EVERYTHING dated after index t (prices, volumes, amounts, turnover, factors, traded status) is altered."""
    Y = copy.deepcopy(X); rng = np.random.default_rng(seed); T, N = X.C.shape
    for k in ["O", "Hh", "L", "C", "V", "AMT", "TURN"]:
        M = getattr(Y, k)
        if how == "nan": M[t + 1:] = np.nan
        else: M[t + 1:] = M[t + 1:] * rng.uniform(0.3, 3.0, size=M[t + 1:].shape)
    if how != "nan":
        Y.F[t + 1:] = Y.F[t + 1:] * 0.7
    return L.finalize(Y)
def assert_asof(fn, X, t, how="garbage"):
    """fn(X, t) must return the same thing on the real data and on data whose future (> t) was destroyed. Raises AssertionError otherwise."""
    a = fn(X, t); b = fn(tamper_future(X, t, how=how), t)
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-9, atol=1e-12, equal_nan=True)
