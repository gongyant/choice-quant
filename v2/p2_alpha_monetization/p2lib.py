# -*- coding: utf-8 -*-
"""P2-A shared library. All functions use only data <= t when building anything attached to signal date t.
Execution-day information (entry/exit tradability) is used ONLY to define the realised forward return, never the candidate pool or controls."""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as st

P2 = Path(__file__).resolve().parent; V2 = P2.parent; P1D = V2 / "p1_dl"
for p in (V2 / "p0", P1D, V2):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
RES, FIG, REP = P2 / "results", P2 / "figures", P2 / "reports"
MODELS = ["HGB", "LGBM", "Ridge", "MLP", "GRU", "TF"]
HP_ALL = P1D / "historical_predictions_all_models.parquet"
SEED_BOOT = 20261018; HAC_LAG = 4; BLOCK = 5; B = 4000

def L():
    import p0lib; return p0lib

def build_world():
    import p0lib as Lm, p1lib as P1
    X = Lm.build_data(); Lm.limit_flags(X); W = P1.world_from_X(X); return X, W

# ---------- as-of-t controls (rolling windows ending at t, inclusive) ----------
def rolling_turn20(TURN):
    return pd.DataFrame(TURN).rolling(20, min_periods=15).mean().values
def amihud20(r1, AMT):
    """Trailing 20-day mean of |r1| / amount (yuan) * 1e8, window ends at t. NaN on non-traded days is skipped (min 15 obs)."""
    a = (np.abs(pd.DataFrame(r1)) / pd.DataFrame(AMT).where(pd.DataFrame(AMT) > 0)) * 1e8
    a = a.replace([np.inf, -np.inf], np.nan)
    return a.rolling(20, min_periods=15).mean().values

# ---------- next-executable-day forward return ----------
def exec_forward(ADJf, buyable, sellable, sig_idx, hold=5, max_entry_wait=5, max_exit_wait=10):
    """Signal at close of t (row i). Entry day e = first row d>=i+1 (d<=i+1+max_entry_wait) with buyable[d,j] (bar present, traded, NOT closed locked at limit-up).
    Buy at close of e. Exit day = first row d>=e+hold (<= e+hold+max_exit_wait) with sellable[d,j] (traded, NOT locked limit-down); if none,
    forced exit at e+hold+max_exit_wait with last known (ffilled) adjusted close, flagged. Rows past the end of calendar -> NaN.
    Returns dict of arrays [len(sig_idx), N]: entry_idx, exit_idx, ret, forced."""
    T, N = ADJf.shape; cols = np.arange(N); K = len(sig_idx)
    E = np.full((K, N), -1); X_ = np.full((K, N), -1); R = np.full((K, N), np.nan); F = np.zeros((K, N), bool)
    for a, i in enumerate(sig_idx):
        e = np.full(N, -1)
        for k in range(max_entry_wait + 1):
            d = i + 1 + k
            if d >= T: break
            ok = (e < 0) & buyable[d]; e[ok] = d
        s0 = e + hold; x = np.full(N, -1)
        for k in range(max_exit_wait + 1):
            d = s0 + k; valid = (e >= 0) & (x < 0) & (d < T)
            ok = valid & sellable[np.minimum(d, T - 1), cols]; x[ok] = d[ok]
        forced = (e >= 0) & (x < 0) & (s0 + max_exit_wait < T); x[forced] = (s0 + max_exit_wait)[forced]
        good = (e >= 0) & (x >= 0)
        r = np.full(N, np.nan); r[good] = ADJf[x[good], cols[good]] / ADJf[e[good], cols[good]] - 1
        r[~np.isfinite(r)] = np.nan
        E[a], X_[a], R[a], F[a] = e, np.where(good, x, -1), r, forced & good
    return dict(entry_idx=E, exit_idx=X_, ret=R, forced=F)

# ---------- statistics ----------
def nw_t(x, lag=HAC_LAG):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 8: return np.nan, np.nan
    e = x - x.mean(); v = (e @ e) / n
    for l in range(1, min(lag, n - 1) + 1): v += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    se = np.sqrt(v / n)
    if se == 0: return np.nan, np.nan
    t = x.mean() / se; return float(t), float(2 * st.t.sf(abs(t), n - 1))
def block_ci(x, block=BLOCK, B=B, seed=SEED_BOOT):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 2 * block: return np.nan, np.nan
    rng = np.random.default_rng(seed); nb = int(np.ceil(n / block))
    s = rng.integers(0, n - block + 1, size=(B, nb)); idx = (s[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    m = x[idx].mean(axis=1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def summarize(x, label=None):
    """Mean of a per-date series with HAC(4) t, p, block-bootstrap 95% CI, share>0, n."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 8: return dict(n_dates=len(x), mean=float(x.mean()) if len(x) else np.nan, hac_t=np.nan, hac_p=np.nan, ci_lo=np.nan, ci_hi=np.nan, pos_share=np.nan, std=np.nan)
    t, p = nw_t(x); lo, hi = block_ci(x)
    return dict(n_dates=len(x), mean=float(x.mean()), std=float(x.std(ddof=1)), hac_t=t, hac_p=p, ci_lo=lo, ci_hi=hi, pos_share=float((x > 0).mean()))

def _rank(v):
    return st.rankdata(v)  # average ties
def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8: return np.nan
    ra, rb = _rank(a[m]), _rank(b[m]); ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb)); return float(ra @ rb / d) if d > 0 else np.nan
def partial_spearman(a, b, C):
    """Spearman correlation of a and b after OLS-residualising the RANKS of both on the ranks of columns of C (with intercept)."""
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(C).all(axis=1)
    if m.sum() < 30: return np.nan, np.nan, int(m.sum())
    ra, rb = _rank(a[m]), _rank(b[m]); Rc = np.column_stack([_rank(C[m][:, k]) for k in range(C.shape[1])])
    Z = np.column_stack([np.ones(m.sum()), Rc])
    ea = ra - Z @ np.linalg.lstsq(Z, ra, rcond=None)[0]; eb = rb - Z @ np.linalg.lstsq(Z, rb, rcond=None)[0]
    d = np.sqrt((ea @ ea) * (eb @ eb)); pc = float(ea @ eb / d) if d > 0 else np.nan
    # Fama-MacBeth style slope: standardised rank(score) coefficient in rank(ret) ~ rank(score)+controls
    Z2 = np.column_stack([Z, (ra - ra.mean()) / ra.std()]); beta = np.linalg.lstsq(Z2, (rb - rb.mean()) / rb.std(), rcond=None)[0][-1]
    return pc, float(beta), int(m.sum())

def wl_append(text):
    p = P1D / "worklog.md"
    import datetime as dt
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M (Asia/Singapore)")
    with open(p, "a", encoding="utf-8", newline="\n") as f: f.write(text.replace("@@T@@", ts))

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def setup_cn_font():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.font_manager as fm, matplotlib.pyplot as plt
    fm.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]; plt.rcParams["axes.unicode_minus"] = False
    return plt
# ---------- (P2-B) OLS with Newey-West errors and block-bootstrap alpha CI (same code as a_eval.py, shared here) ----------
def ols_hac(y, Xf, lag=HAC_LAG):
    y = np.asarray(y, float); Z = np.column_stack([np.ones(len(y)), Xf]); n, k = Z.shape
    beta = np.linalg.solve(Z.T @ Z, Z.T @ y); e = y - Z @ beta; U = Z * e[:, None]; S_ = U.T @ U
    for l in range(1, lag + 1): w = 1 - l / (lag + 1); G = U[l:].T @ U[:-l]; S_ += w * (G + G.T)
    Ai = np.linalg.inv(Z.T @ Z); cov = Ai @ S_ @ Ai; se = np.sqrt(np.diag(cov)); sst = (y - y.mean()) @ (y - y.mean())
    return beta, beta / se, (1 - (e @ e) / sst) if Z.shape[1] > 1 else 0.0
def alpha_ci(y, Xf, B_=2000, block=BLOCK, seed=SEED_BOOT):
    n = len(y); rng = np.random.default_rng(seed); nb = int(np.ceil(n / block)); s = rng.integers(0, n - block + 1, size=(B_, nb)); idx = (s[:, :, None] + np.arange(block)[None, None, :]).reshape(B_, -1)[:, :n]
    Z = np.column_stack([np.ones(n), Xf]); al = np.empty(B_)
    for b in range(B_): zb = Z[idx[b]]; al[b] = np.linalg.lstsq(zb, np.asarray(y, float)[idx[b]], rcond=None)[0][0]
    return float(np.percentile(al, 2.5)), float(np.percentile(al, 97.5))