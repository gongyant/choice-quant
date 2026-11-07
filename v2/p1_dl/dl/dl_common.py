"""P1-B common helpers. P0 code is imported READ-ONLY (v2/p0/p0lib.py); nothing in p0/ is modified."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
DL = Path(__file__).resolve().parent; P1 = DL.parent; V2 = P1.parent; P0 = V2 / "p0"
sys.path.insert(0, str(P0)); sys.path.insert(0, str(V2))
import p0lib as L
VAL_DAYS, GAP_DAYS, H = 60, 10, L.H
class World: pass
def build_world(codes=None):
    X = L.build_data(codes=codes); L.limit_flags(X)
    return world_from(X)
def world_from(X):
    W = World(); W.X = X; W.f, W.elig, W.ADJf = L.compute_features(X, X.ADJ_unadj_based)
    W.lab, W.y, W.yw = L.make_labels(W.ADJf, W.elig); W.Xq = L.rank_features(W.f, W.elig)
    W.m_tr = W.elig & np.isfinite(W.yw); W.cal = X.cal; W.T, W.N = W.elig.shape; return W
def split_train_val(Xq, yw, m_tr, cal, i, years=3, val_days=VAL_DAYS, gap=GAP_DAYS):
    """Training window = EXACTLY P0's (lo .. hi=i-12, 3y lookback). Inside it: validation = last `val_days` calendar rows (vs..hi); training = lo..te with te = vs-gap.
    Label of row d expires at close of d+6, so max train label expiry = te+6 < vs (before every validation date) and max val label expiry = hi+6 = i-6 < i."""
    lo = int(cal.searchsorted(cal[i] - pd.Timedelta(days=365 * years))); hi = i - 12; vs = hi - val_days + 1; te = vs - gap
    def collect(a, b):
        ds = [d for d in range(a, b + 1) if d >= 0 and m_tr[d].any()]
        if not ds: return np.zeros((0, Xq.shape[2])), np.zeros(0), np.zeros(0, int)
        return (np.concatenate([Xq[d][m_tr[d]] for d in ds]), np.concatenate([yw[d][m_tr[d]] for d in ds]), np.concatenate([np.full(int(m_tr[d].sum()), d) for d in ds]))
    Xt, yt, dt = collect(lo, te); Xv, yv, dv = collect(vs, hi)
    return dict(Xtr=Xt, ytr=yt, dtr=dt, Xva=Xv, yva=yv, dva=dv, lo=lo, te=te, vs=vs, hi=hi, label_expiry_train_max=int(dt.max()) + H + 1 if len(dt) else -1,
                label_expiry_val_max=int(dv.max()) + H + 1 if len(dv) else -1)