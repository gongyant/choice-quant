"""P1-A1 library: as-of-t scoring that reuses the P0-fixed pipeline (p0/p0lib.py) unchanged.
world_from_X(X)      -> features / eligibility / labels / rank-features from calendar matrices (features use only rows <= signal date;
                        labels use t+1..t+6 and are consumed only for training dates d <= i-12, i.e. label end <= i-6 < i).
train_frozen(W, i)   -> Ridge / HGB / LGBM with the P0 hyper-parameters (baseline.make_models), trained ONLY on rows d <= i-12, 3-year window.
score_asof(W, i, M)  -> model scores for every stock in the eligible_asof_t pool at index i.
"""
import sys, copy, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; V2 = H.parent
sys.path.insert(0, str(V2)); sys.path.insert(0, str(V2/"p0"))
import baseline as B
import p0lib as L
MODELS = ["Ridge", "HGB", "LGBM"]
def world_from_X(X):
    W = type("W", (), {})(); W.X = X
    f, elig, ADJf = L.compute_features(X, X.ADJ_unadj_based); lab, y, yw = L.make_labels(ADJf, elig)
    W.f, W.elig, W.ADJf, W.lab, W.y, W.yw = f, elig, ADJf, lab, y, yw
    W.Xq = L.rank_features(f, elig); W.cal = X.cal; return W
def train_frozen(W, i, seed=42):
    m_tr = W.elig & np.isfinite(W.yw)
    Xt, yt = L.train_rows(W.Xq, W.yw, m_tr, W.cal, i)
    lo = int(W.cal.searchsorted(W.cal[i] - pd.Timedelta(days=365 * 3))); hi = i - 12
    ms = B.make_models(seed); fitted = {}
    for n in MODELS:
        m = ms[n](); m.fit(Xt, yt); fitted[n] = m
    meta = dict(signal_index=int(i), signal_date=str(W.cal[i].date()), train_first_date=str(W.cal[lo].date()), train_last_date=str(W.cal[hi].date()),
                latest_label_end_date_used=str(W.cal[hi + L.H + 1].date()), n_train_rows=int(len(Xt)), n_train_dates=int(sum(m_tr[d].any() for d in range(lo, hi + 1))),
                seed=seed, purge_trading_days=12, window_years=3, horizon=L.H, features=L.FEATS, models=MODELS, label="t+1 close -> t+6 close adjusted return, de-meaned over eligible_asof_t pool, winsorised 1/99 per day")
    return fitted, meta
def score_asof(W, i, fitted):
    pool = np.where(W.elig[i])[0]; xs = W.Xq[i][pool]
    return pool, {n: fitted[n].predict(xs) for n in MODELS}
def score_outsider(W, i, fitted, f_out):
    """Score a stock NOT in the pool: its as-of-t feature values are ranked against the pool's as-of-t feature distribution."""
    pool = np.where(W.elig[i])[0]; row = []
    for k in L.FEATS:
        v = f_out[k]; pv = W.f[k].values[i][pool]; pv = pv[np.isfinite(pv)]
        row.append(0.0 if not np.isfinite(v) or len(pv) == 0 else (np.sum(pv < v) + 0.5 * np.sum(pv == v) + 0.5) / (len(pv) + 1) - 0.5)
    x = np.array(row)[None, :]; return {n: float(fitted[n].predict(x)[0]) for n in MODELS}, row
def file_sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
