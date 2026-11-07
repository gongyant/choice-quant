# -*- coding: utf-8 -*-
"""P5 shared helpers. Reads P4 caches/predictions READ-ONLY (never writes to p4_top5_quant). Engine = p4lib.backtest (unchanged)."""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
P5 = Path(__file__).resolve().parent; V2 = P5.parent; P4D = V2 / "p4_top5_quant"; sys.path.insert(0, str(P4D))
import p4lib as P
C = P4D / "cache"; PR = P4D / "preds"
def load():
    fl = np.load(C / "flags.npz"); X = pickle.load(open(C / "world.pkl", "rb")); LAB = pickle.load(open(C / "labels.pkl", "rb"))["lab"]
    W = type("W", (), {})(); W.fl, W.X, W.LAB = fl, X, LAB; W.cal = pd.DatetimeIndex(fl["cal"]); W.codes = list(fl["codes"]); W.col = {c: j for j, c in enumerate(W.codes)}
    W.elig, W.amt20, W.olu, W.old, W.st = fl["elig"], fl["amt20"], fl["olu"], fl["old"], fl["st"]; W.N = len(W.codes)
    H = pd.read_parquet(V2 / "p1_dl" / "historical_predictions_all_models.parquet", columns=["signal_date", "code", "model", "score", "eligible_asof"]); W.H = H[H.eligible_asof]
    W.sig = sorted(W.H.signal_date.unique()); W.sig_idx = [int(W.cal.get_loc(pd.Timestamp(s))) for s in W.sig]
    A = pd.DataFrame(np.where(X.traded, X.C / X.F, np.nan)).ffill(); W.A = A.values
    W.dd60 = (A / A.rolling(60, min_periods=40).max() - 1).values; W.r20 = (A / A.shift(20) - 1).values
    r1 = A.pct_change(fill_method=None); W.vol = r1.rolling(20, min_periods=15).std().values
    W.volp = pd.DataFrame(np.where(W.elig, W.vol, np.nan)).rank(axis=1, pct=True).values
    mk = r1.where(pd.DataFrame(W.elig)).mean(1); W.beta = (r1.rolling(60, min_periods=40).cov(mk) / mk.rolling(60, min_periods=40).var().values[:, None]).values if False else r1.rolling(60, min_periods=40).corr(mk).values * (r1.rolling(60, min_periods=40).std().values / mk.rolling(60, min_periods=40).std().values[:, None])
    TURN = np.where(X.TURN > 0, X.TURN, np.nan); W.fmcap = X.C * X.V * 100 / TURN       # float market cap proxy (turnover % = volume / float shares)
    W.ind = X.u.industry_l2.fillna("NA").values; W.names = X.u.name.values; W.board = X.u.board.values
    lv = np.full((len(W.sig), W.N), np.nan)
    for q, i in enumerate(W.sig_idx): lv[q, W.elig[i]] = -W.vol[i, W.elig[i]]
    W.lowvol = lv; return W
def mat(W, df):
    M = np.full((len(W.sig), W.N), np.nan); k = {s: q for q, s in enumerate(W.sig)}; r = df.signal_date.map(k).values; c = df.code.map(W.col).values; ok = ~pd.isna(c) & ~pd.isna(r)
    M[r[ok].astype(int), c[ok].astype(int)] = df.score.values[ok]; return M
def filt(W, i): return W.elig[i] & W.X.traded[i] & ~W.X.lim_up[i] & (np.nan_to_num(W.amt20[i]) >= 1e8) & ~W.st[i]
def cons(W, i): return (np.nan_to_num(W.volp[i], nan=1) <= 0.80) & (np.nan_to_num(W.dd60[i], nan=-1) > -0.35) & (np.nan_to_num(W.r20[i], nan=-1) > -0.25)
def pct(v, m): out = np.full(len(v), np.nan); out[m] = pd.Series(v[m]).rank(pct=True).values; return out
def plain(W, M, rows, K=5):
    out = []
    for q in rows:
        i = W.sig_idx[q]; s = np.where(filt(W, i), M[q], np.nan); ok = np.where(np.isfinite(s))[0]; out.append(ok[np.argsort(-s[ok], kind="stable")][:K])
    return out
def select(W, M, rows, buffer, blend, K=5, NB=40):
    """identical to p4_e2_eval.select (registered selection_rules_p4_v1)"""
    out, prev = [], []
    for q in rows:
        i = W.sig_idx[q]; f = filt(W, i) & np.isfinite(M[q]); s = M[q].copy()
        if blend: s = 0.5 * pct(M[q], f) + 0.5 * pct(W.lowvol[q], f & np.isfinite(W.lowvol[q]))
        s = np.where(f, s, np.nan); ok = np.where(np.isfinite(s))[0]; order = ok[np.argsort(-s[ok], kind="stable")]; rank = {j: r for r, j in enumerate(order)}
        ok_c = cons(W, i); pick, cnt = [], {}
        def add(j):
            if len(pick) < K and ok_c[j] and cnt.get(W.ind[j], 0) < 2 and j not in pick: pick.append(j); cnt[W.ind[j]] = cnt.get(W.ind[j], 0) + 1
        if buffer:
            for j in prev:
                if j in rank and rank[j] < NB: add(j)
        for j in order[:20]: add(j)
        out.append(np.array(pick, int)); prev = pick
    return out
def rows_for(W, h): return list(range(0, len(W.sig), 1 if h == 5 else 4))
def run(W, pk, rows, h, nm):
    si = [W.sig_idx[q] for q in rows]; pr, ho = P.backtest(W.X, si, pk, h, 5, 0.003, skip=W.st, olu=W.olu, old=W.old, names=W.names, strategy=nm)
    pr["horizon"] = h; pr["pool_ret"] = P.pool_bench(W.X, si, h, W.elig, W.olu); ho["horizon"] = h; return pr, ho
def hac_t(x, lag):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x); e = x - x.mean(); g0 = (e @ e) / n; s = g0
    for l in range(1, lag + 1): s += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    return x.mean() / np.sqrt(s / n)
def block_ci(x, block, B=5000, seed=20261022):
    x = np.asarray(x, float); n = len(x); rng = np.random.default_rng(seed); nb = int(np.ceil(n / block)); m = []
    for _ in range(B):
        st = rng.integers(0, n - block + 1, nb); m.append(x[(st[:, None] + np.arange(block)).ravel()[:n]].mean())
    return np.percentile(m, [2.5, 97.5])
