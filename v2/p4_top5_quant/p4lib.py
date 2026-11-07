# -*- coding: utf-8 -*-
"""P4 shared library (exploratory research only; no broker, no orders).
- load_world(): reuse P0 loader (p0lib.build_data/limit_flags) redirected IN MEMORY to P3 merged raw data (data to 2026-10-21). P0-P3 files are only read.
- features(): P0's 11 features + Alpha158-style price/volume factors, all computed from rows <= t.
- labels(): open-to-open adjusted returns t+1 open -> t+1+h open, de-meaned over the eligible_asof_t pool, graded deciles 0..9.
- st_proxy(): point-in-time ST proxy (historical ST names are not available): main board, a close locked at ~5% limit within the last 60 days
  and no |close/ref-1| > 5.6% in those 60 days.
- backtest(): one engine for every strategy: K equal-weight slots, cash allowed, buy at t+1 OPEN, exit at t+1+h OPEN (delayed while suspended /
  opening at limit-down), round-trip cost 0.30% (0.15% per executed side; retained names pay nothing), skip: ST-proxy at t, no bar at t+1, open at limit-up."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
P4 = Path(__file__).resolve().parent; V2 = P4.parent; P3 = V2 / "p3_ai_report"; MG = P3 / "data" / "merged"
sys.path[:0] = [str(V2 / "p0"), str(V2)]
import p0lib as L
END = "2026-10-21"
def load_world(end=END):
    L.D, L.RU, L.FA, L.LAST_TD = MG, MG / "raw_unadj_parquet", MG / "adj_factors", pd.Timestamp(end)
    X = L.build_data(); L.limit_flags(X); return X
BOARD_RATE = L.BOARD_RATE
def limit_prices(X):
    rate = np.array([BOARD_RATE.get(b, 0.10) for b in X.u.board.values])[None, :]
    up = L.round_cent(X.ref * (1 + rate)); dn = L.round_cent(X.ref * (1 - rate)); return up, dn
def open_flags(X):
    up, dn = limit_prices(X); tol = 0.005 + 1e-9; ok = X.traded & np.isfinite(X.ref) & ~X.nolimit_ipo
    return ok & (X.O >= up - tol), ok & (X.O <= dn + tol)
def st_proxy(X, win=60, rule_change="2026-07-06"):
    """Point-in-time ST proxy (historical ST names unavailable). Before 2026-07-06 (main-board ST limit 5%): main board, a close locked EXACTLY at the
    5% limit price (ex-rights ref, rounded to the cent) in the last `win` days and no |close/ref-1| > 5.6% in those days. From 2026-07-06 the ST limit
    became 10% (SSE/SZSE rules of 2026-04-24), so the proxy is frozen at its last pre-change value OR the current ST/*ST name (small look-ahead, conservative)."""
    main = np.array([b not in BOARD_RATE for b in X.u.board.values])[None, :]
    r = X.C / X.ref - 1; ok = X.traded & np.isfinite(r) & ~X.nolimit_ipo; tol = 0.005 + 1e-9
    u5 = L.round_cent(X.ref * 1.05); d5 = L.round_cent(X.ref * 0.95)
    lock5 = ok & (((np.abs(X.C - u5) <= tol) & (X.C >= X.Hh - tol)) | ((np.abs(X.C - d5) <= tol) & (X.C <= X.L + tol)))
    big = ok & (np.abs(r) > 0.056)
    a = pd.DataFrame(lock5.astype(float)).rolling(win, min_periods=1).sum().values > 0
    b = pd.DataFrame(big.astype(float)).rolling(win, min_periods=1).sum().values > 0
    st = main & a & ~b
    k = int(X.cal.searchsorted(pd.Timestamp(rule_change)))
    cur = (X.u.is_st_name.fillna(False) | X.u.name_has_delist_mark.fillna(False)).values.astype(bool)
    if k < len(X.cal): st[k:] = st[k - 1][None, :] | cur[None, :]
    return st
FE0 = list(L.FEATS)
def features(X):
    f0, elig, ADJf = L.compute_features(X, X.ADJ_unadj_based)
    tr = pd.DataFrame(X.traded); F = pd.DataFrame(X.F).ffill()
    C = pd.DataFrame(ADJf); Oa = pd.DataFrame(X.O) / F; Ha = pd.DataFrame(X.Hh) / F; La = pd.DataFrame(X.L) / F
    r1 = (C / C.shift(1) - 1).where(tr); V = pd.DataFrame(X.V).where(pd.DataFrame(X.V) > 0); A = pd.DataFrame(X.AMT).where(pd.DataFrame(X.AMT) > 0)
    f = {k: f0[k] for k in FE0}
    f["r10"] = (C / C.shift(10) - 1).where(tr); f["r120"] = (C / C.shift(120) - 1).where(tr)
    f["mom_60_5"] = (C.shift(5) / C.shift(60) - 1).where(tr)
    for k in [10, 60]: f[f"ma_gap{k}"] = (C / C.where(tr).rolling(k, min_periods=int(k * .7)).mean() - 1).where(tr)
    for k in [5, 60]: f[f"std{k}"] = r1.rolling(k, min_periods=max(3, int(k * .7))).std()
    f["vol_ratio_20_60"] = f["vol20"] / f["std60"]
    f["skew20"] = r1.rolling(20, min_periods=15).skew(); f["max_ret20"] = r1.rolling(20, min_periods=15).max(); f["min_ret20"] = r1.rolling(20, min_periods=15).min()
    for k in [20, 60]:
        hh = Ha.rolling(k, min_periods=int(k * .7)).max(); ll = La.rolling(k, min_periods=int(k * .7)).min()
        f[f"max_gap{k}"] = (hh / C - 1).where(tr); f[f"min_gap{k}"] = (C / ll - 1).where(tr); f[f"rsv{k}"] = ((C - ll) / (hh - ll)).where(tr)
    f["imax20"] = Ha.rolling(20, min_periods=15).apply(lambda a: 19 - np.nanargmax(a) if np.isfinite(a).any() else np.nan, raw=True) / 20
    f["imin20"] = La.rolling(20, min_periods=15).apply(lambda a: 19 - np.nanargmin(a) if np.isfinite(a).any() else np.nan, raw=True) / 20
    t = np.arange(20) - 9.5
    f["slope20"] = np.log(C).rolling(20, min_periods=20).apply(lambda a: np.dot(t, a) / np.dot(t, t), raw=True).where(tr)
    f["cntp20"] = (r1 > 0).astype(float).where(r1.notna()).rolling(20, min_periods=15).mean()
    f["sump20"] = r1.clip(lower=0).rolling(20, min_periods=15).sum() / r1.abs().rolling(20, min_periods=15).sum()
    lv = np.log(V); f["corr_pv20"] = C.rolling(20, min_periods=15).corr(lv); f["cord_rv20"] = r1.rolling(20, min_periods=15).corr(lv.diff())
    f["vma_5_20"] = V.rolling(5, min_periods=3).mean() / V.rolling(20, min_periods=15).mean(); f["vstd20"] = V.rolling(20, min_periods=15).std() / V.rolling(20, min_periods=15).mean()
    w = (r1.abs() * V); f["wvma20"] = w.rolling(20, min_periods=15).std() / w.rolling(20, min_periods=15).mean()
    f["kmid"] = ((C - Oa) / Oa).where(tr); f["klen"] = ((Ha - La) / Oa).where(tr); f["kup"] = ((Ha - np.maximum(Oa, C)) / Oa).where(tr); f["klow"] = ((np.minimum(Oa, C) - La) / Oa).where(tr)
    f["kmid5"] = f["kmid"].rolling(5, min_periods=3).mean()
    f["amihud20"] = (r1.abs() / A).rolling(20, min_periods=15).mean() * 1e8
    T_ = pd.DataFrame(X.TURN); f["turn20"] = T_.rolling(20, min_periods=15).mean(); f["turn_5_60"] = T_.rolling(5, min_periods=3).mean() / T_.rolling(60, min_periods=40).mean()
    f["amt_5_60"] = A.rolling(5, min_periods=3).mean() / A.rolling(60, min_periods=40).mean()
    # relative strength vs industry (industry_l2 = current CNINFO classification; industry membership is slow-moving, noted as a limitation)
    ind = X.u.industry_l2.fillna("NA").values
    for k in ["r5", "r20", "r60"]:
        v = f[k].where(pd.DataFrame(elig)); g = v.T.groupby(ind).transform("mean").T; f[f"{k}_vs_ind"] = f[k] - g
    for k in f: f[k] = f[k].replace([np.inf, -np.inf], np.nan)
    return f, elig, ADJf, f0["amt20"].values
def rank_feats(f, elig, names):
    out = np.zeros(elig.shape + (len(names),), np.float32)
    for q, k in enumerate(names):
        out[:, :, q] = np.nan_to_num(pd.DataFrame(np.where(elig, f[k].values, np.nan)).rank(axis=1, pct=True).values - 0.5, nan=0.0)
    return out
def labels(X, elig, h):
    Oa = np.where(X.traded, X.O / X.F, np.nan); T, N = Oa.shape; lab = np.full((T, N), np.nan)
    lab[:T - h - 1] = Oa[h + 1:] / Oa[1:T - h] - 1
    m = elig & np.isfinite(lab); ex = np.full((T, N), np.nan); exw = np.full((T, N), np.nan); gr = np.full((T, N), np.nan)
    for i in range(T):
        if m[i].sum() < 20: continue
        v = lab[i, m[i]] - lab[i, m[i]].mean(); ex[i, m[i]] = v; lo, hi = np.quantile(v, [.01, .99]); exw[i, m[i]] = np.clip(v, lo, hi)
        gr[i, m[i]] = np.minimum(9, np.floor(pd.Series(v).rank(method="first").values / (len(v) + 1e-9) * 10))
    return lab, ex, exw, gr
def label_end_index(i, h): return i + 1 + h      # last calendar row whose open the h-day label at row i uses
def train_days(i, h, cal, years=3, step=3):
    """Days whose label is fully matured at the close of i (label end i-1 < i), plus 1 extra day purge: d <= i-h-2."""
    lo = int(cal.searchsorted(cal[i] - pd.Timedelta(days=365 * years))); hi = i - h - 2
    return [d for d in range(hi, lo - 1, -step)][::-1]
def backtest(X, sig_idx, picks, h, K, cost_rt=0.003, skip=None, olu=None, old=None, names=None, strategy=""):
    """picks[k] = ordered column indices chosen at the close of sig_idx[k]; the first K are used. Returns per-period df, per-holding df."""
    T, N = X.C.shape; c2 = cost_rt / 2; Oa = np.where(X.traded, X.O / X.F, np.nan)
    rows, hold = [], []; prev = set()
    seq = [list(p)[:K] for p in picks]
    for k, i in enumerate(sig_idx):
        e = i + 1; x = i + 1 + h; cur = set(); rets = []; n_new = n_sell = 0; rej = {}
        nxt = set(seq[k + 1]) if k + 1 < len(seq) else set()
        for r_, j in enumerate(seq[k]):
            why = None
            if e >= T: why = "no_future_bar"
            elif skip is not None and skip[i, j]: why = "st_proxy"
            elif not X.traded[e, j]: why = "suspended_at_entry"
            elif olu[e, j]: why = "open_limit_up"
            if why: rej[why] = rej.get(why, 0) + 1; hold.append(dict(strategy=strategy, signal_date=str(X.cal[i].date()), rank=r_ + 1, code=X.codes[j], name=names[j] if names is not None else "", status="SKIPPED_" + why, slot_ret_gross=0.0, slot_ret_net=0.0)); continue
            d = x; delayed = 0
            while d < T and (not X.traded[d, j] or old[d, j]): d += 1; delayed += 1
            if d >= T:
                last = np.where(X.traded[:, j])[0]; dd = last[-1]; px = X.C[dd, j] / X.F[dd, j]; st = "MARKED_LAST_CLOSE_UNSOLD"
            else: px = Oa[d, j]; dd = d; st = "FILLED"
            g = px / Oa[e, j] - 1; held_before = j in prev; keep_after = j in nxt
            cb = 0 if held_before else c2; cs = 0 if keep_after else c2; net = (1 + g) * (1 - cb) * (1 - cs) - 1
            n_new += int(not held_before); n_sell += int(not keep_after); cur.add(j); rets.append((g, net))
            hold.append(dict(strategy=strategy, signal_date=str(X.cal[i].date()), rank=r_ + 1, code=X.codes[j], name=names[j] if names is not None else "", status=st,
                             entry_date=str(X.cal[e].date()), entry_open_unadj=float(X.O[e, j]), exit_date=str(X.cal[dd].date()), exit_delay_days=delayed,
                             slot_ret_gross=float(g), slot_ret_net=float(net), buy_cost=cb, sell_cost=cs, retained_from_prev=held_before))
        gs = sum(a for a, _ in rets) / K; ns = sum(b for _, b in rets) / K
        rows.append(dict(strategy=strategy, signal_date=str(X.cal[i].date()), n_picks=len(seq[k]), n_held=len(rets), ret_gross=gs, ret_net=ns, fees=gs - ns,
                         turnover=(n_new + n_sell) / (2 * K), **{f"rej_{a}": b for a, b in rej.items()}))
        prev = cur
    return pd.DataFrame(rows), pd.DataFrame(hold)
def pool_bench(X, sig_idx, h, elig, olu):
    T = X.C.shape[0]; Oa = np.where(X.traded, X.O / X.F, np.nan); out = []
    for i in sig_idx:
        e, x = i + 1, min(i + 1 + h, T - 1); m = elig[i] & X.traded[e] & ~olu[e]
        r = Oa[x, m] / Oa[e, m] - 1; out.append(np.nanmean(r))
    return np.array(out)
def mdd(r):
    eq = np.cumprod(1 + np.asarray(r)); return float((eq / np.maximum.accumulate(eq) - 1).min())
def ndcg_at(score, grade, k=5):
    m = np.isfinite(score) & np.isfinite(grade); s, g = score[m], grade[m]
    if len(s) < k * 2: return np.nan
    gain = 2 ** g - 1; disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = (gain[np.argsort(-s, kind="stable")[:k]] * disc).sum(); idcg = (np.sort(gain)[::-1][:k] * disc).sum(); return dcg / idcg
