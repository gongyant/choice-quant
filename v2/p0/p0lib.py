"""P0 shared library: unified trading calendar, unadjusted-price matrices, corporate-action factors, as-of-t eligibility,
limit-up/down inference, labels, and the order/position state machine.

Fix map (audit protocol Chapter 2):
  P0-A  eligible_asof_t uses ONLY information dated <= t (see compute_features / eligible). Execution feasibility (t+1 buy, t+6 sell) is
        decided by simulate() with *later* bars and can only REJECT an order (cash stays idle); it never swaps in a replacement stock.
  P0-B  Every t+1 / t+6 is an index on the single master calendar (matrix rows), not a per-stock row offset. Suspensions are NaN rows.
  P0-C  Fills use UNADJUSTED close; holding returns use adjusted-close ratios built from raw price / Sina cumulative factor
        (dividends/bonus/transfer/split). Prices <= 0 cannot occur (raw prices > 0 are required for a bar to count as traded).
  P0-D  list/delist dates are used for a historical daily listing mask; sample design flags are carried for group reporting.
  P0-E  Training-label de-meaning pool == IC pool == benchmark pool == eligible_asof_t. Equal-weight benchmark runs through the same
        simulator (same fills, costs, turnover).
  P0-F  Limit up/down inferred from RAW previous close adjusted for ex-rights (ref = prevC * F_t / F_prev), board-specific rate,
        rounded to the cent, no-limit windows for new listings; main board also tests the 5% (ST) price conservatively.
"""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
V2 = Path(__file__).resolve().parents[1]; P0 = V2 / "p0"; D = P0 / "data"
RU, FA, TXD = D / "raw_unadj_parquet", D / "adj_factors", D / "raw_unadj_tx"
H = 5
LAST_TD = pd.Timestamp("2026-10-16"); WIN_START = pd.Timestamp("2021-01-01")
FEATS = ["r1", "r5", "r20", "r60", "vol20", "ma_gap5", "ma_gap20", "volume_ratio5", "turnover5", "log_amt20", "range20"]
BOARD_RATE = {"创业板": 0.20, "科创板": 0.20, "北交所": 0.30}
def sha256(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def load_calendar():
    f = D / "trade_calendar.csv"
    if not f.exists():
        import akshare as ak
        c = pd.to_datetime(ak.tool_trade_date_hist_sina().trade_date); c = c[(c >= WIN_START) & (c <= LAST_TD)].sort_values()
        pd.DataFrame({"date": c}).to_csv(f, index=False)
    return pd.DatetimeIndex(pd.read_csv(f, parse_dates=["date"]).date)
def load_universe():
    return pd.read_csv(V2 / "data" / "sample_1500.csv", dtype={"code": str}, parse_dates=["list_date", "delist_date"])
def round_cent(x): return np.floor(x * 100 + 0.5 + 1e-9) / 100
def snap_pow10(r, tol=0.2):
    if not np.isfinite(r) or r <= 0: return np.nan
    k = np.round(np.log10(r)); return float(10.0 ** k) if abs(np.log10(r) - k) < tol else np.nan
def load_raw(code):
    f = RU / f"{code}.parquet"
    if not f.exists(): return None
    d = pd.read_parquet(f).sort_values("date").drop_duplicates("date", keep="last")
    if "volume_raw" in d.columns and "volume" not in d.columns: d = d.rename(columns={"volume_raw": "volume"})
    return d
def load_factor(code):
    f = FA / f"{code}.parquet"
    return pd.read_parquet(f) if f.exists() else None
def factor_on_dates(dates, fac):
    if fac is None or len(fac) == 0: return np.ones(len(dates))
    fac = fac.dropna(subset=["qfq_factor"]); fac = fac[fac.qfq_factor > 0].sort_values("date")
    if len(fac) == 0: return np.ones(len(dates))
    fd, fv = fac.date.values, fac.qfq_factor.values.astype(float)
    idx = np.searchsorted(fd, dates.values, side="right") - 1
    return np.where(idx >= 0, fv[np.clip(idx, 0, None)], fv[0])
class Data:
    """Calendar-indexed matrices [T dates x N stocks]. NaN = no bar (suspended / not listed / delisted / failed)."""
    pass
def build_data(universe=None, codes=None):
    cal = load_calendar(); u = universe if universe is not None else load_universe()
    codes = codes or [c for c in u.code if (RU / f"{c}.parquet").exists()]
    u = u.set_index("code").loc[codes]; T, N = len(cal), len(codes)
    A = {k: np.full((T, N), np.nan) for k in ["O", "Hh", "L", "C", "V", "AMT", "TURN", "F"]}
    meta = []
    for j, c in enumerate(codes):
        d = load_raw(c); src = d.source.iloc[0]
        n_before = len(d); d = d[d.date.isin(cal)]; pos = cal.get_indexer(d.date)
        close, vol, amt = d.close.astype(float).values, d.volume.astype(float).values, d.amount.astype(float).values
        ok = (vol > 0) & (amt > 0) & (close > 0)
        ratio = float(np.median(amt[ok] / (close[ok] * vol[ok]))) if ok.sum() >= 10 else np.nan
        mult = snap_pow10(ratio)
        dflt = 100.0 if src.startswith("tencent") else 1.0   # documented vendor unit: Tencent = lots(手), Sina = shares
        used = mult if np.isfinite(mult) else dflt
        A["O"][pos, j] = d.open.values; A["Hh"][pos, j] = d.high.values; A["L"][pos, j] = d.low.values; A["C"][pos, j] = close
        A["V"][pos, j] = vol * used; A["AMT"][pos, j] = amt
        A["TURN"][pos, j] = pd.to_numeric(d.turnover, errors="coerce").values if "turnover" in d else np.nan
        fac = load_factor(c); fs = factor_on_dates(cal, fac); A["F"][:, j] = fs
        meta.append(dict(code=c, source=src, rows=n_before, rows_in_cal=len(d), amount_ratio_median=ratio, unit_mult_inferred=mult, unit_mult_used=used,
                         unit_default_by_source=dflt, unit_unverified=not np.isfinite(mult),
                         factor_rows=0 if fac is None else len(fac), factor_status=("missing" if fac is None else ("empty_table(sina_not_available_or_no_actions)" if len(fac) == 0 else "ok")),
                         n_factor_changes_in_window=int((np.diff(fs) != 0).sum())))
    X = Data(); X.cal, X.codes, X.u = cal, codes, u; X.meta = pd.DataFrame(meta)
    for k, v in A.items(): setattr(X, k, v)
    return finalize(X)
def finalize(X):
    """Derived masks from the raw matrices (shared by build_data and the synthetic unit-test fixtures)."""
    u, cal = X.u, X.cal
    X.has_row = np.isfinite(X.C)
    X.traded = X.has_row & (X.V > 0) & (X.C > 0)
    X.ADJ_unadj_based = np.where(X.has_row, X.C / X.F, np.nan)   # corporate-action adjusted close (dividends reinvested)
    # historical listing mask (P0-D): list_date <= t and (delist_date > t or none) -- uses only dated facts
    ld = u.list_date.values.astype("datetime64[ns]"); dd = u.delist_date.values.astype("datetime64[ns]"); cd = cal.values[:, None]
    X.listed = (np.isnat(ld)[None, :] | (cd >= ld[None, :])) & (np.isnat(dd)[None, :] | (cd < dd[None, :]))
    return X
def ffill_mat(M):
    return pd.DataFrame(M).ffill().values
def limit_flags(X):
    """P0-F. Returns lim_up, lim_dn (bool [T,N]) = bar closed locked at the limit price (buy impossible at up / sell impossible at down)."""
    T, N = X.C.shape
    Cf = ffill_mat(X.C); prevC = np.vstack([np.full((1, N), np.nan), Cf[:-1]])
    Fat = np.where(X.has_row, X.F, np.nan); prevF = np.vstack([np.full((1, N), np.nan), ffill_mat(Fat)[:-1]])
    ref = prevC * X.F / prevF     # ex-rights reference price: prev raw close adjusted for dividend/bonus/transfer/split on t
    X.ref = ref
    board = X.u.board.values; rate = np.array([BOARD_RATE.get(b, 0.10) for b in board]); main = np.array([b not in BOARD_RATE for b in board])
    # no-limit windows for new listings (first 5 trading bars for STAR/ChiNext/main(after 2023-04-10), first bar for BJ and pre-2023 main board)
    k = np.cumsum(X.has_row, axis=0) - 1
    ipo_listed = np.array([(pd.notna(l) and l >= WIN_START) for l in X.u.list_date.values])
    first_date = np.array([X.cal[np.argmax(X.has_row[:, j])] if X.has_row[:, j].any() else pd.NaT for j in range(N)])
    near_ipo = np.array([ipo_listed[j] and pd.notna(first_date[j]) and (first_date[j] - X.u.list_date.values[j]).days <= 5 for j in range(N)])
    reg_main = (X.cal >= pd.Timestamp("2023-04-10"))[:, None]
    width = np.where(main[None, :], np.where(reg_main, 5, 1), np.where(board[None, :] == "北交所", 1, 5))
    nolim = near_ipo[None, :] & (k >= 0) & (k < width)
    X.nolimit_ipo = nolim
    ok = X.traded & np.isfinite(ref)
    up10 = round_cent(ref * (1 + rate[None, :])); dn10 = round_cent(ref * (1 - rate[None, :]))
    up5 = round_cent(ref * 1.05); dn5 = round_cent(ref * 0.95)
    tol = 0.005 + 1e-9
    atH = X.C >= X.Hh - tol; atL = X.C <= X.L + tol
    up = ok & atH & (X.C >= up10 - tol); dn = ok & atL & (X.C <= dn10 + tol)
    # ST (5% on main board) -- no historical ST list: conservative proxy = close exactly equals the 5% limit price AND equals the day's high/low.
    up |= ok & main[None, :] & atH & (np.abs(X.C - up5) <= tol); dn |= ok & main[None, :] & atL & (np.abs(X.C - dn5) <= tol)
    up &= ~nolim; dn &= ~nolim
    X.lim_up, X.lim_dn = up, dn
    return up, dn
def compute_features(X, ADJ):
    """As-of-t features computed on calendar-indexed matrices. ADJ = adjusted close [T,N] (NaN on non-bar days). Uses data <= t only."""
    ADJf = pd.DataFrame(ffill_mat(ADJ)); tr = pd.DataFrame(X.traded)
    ADJt = ADJf.where(tr)
    r1 = (ADJf / ADJf.shift(1) - 1).where(tr)
    f = {"r1": r1, "r5": (ADJf / ADJf.shift(5) - 1).where(tr), "r20": (ADJf / ADJf.shift(20) - 1).where(tr), "r60": (ADJf / ADJf.shift(60) - 1).where(tr)}
    f["vol20"] = r1.rolling(20, min_periods=15).std()
    f["ma_gap5"] = (ADJf / ADJt.rolling(5, min_periods=3).mean() - 1).where(tr); f["ma_gap20"] = (ADJf / ADJt.rolling(20, min_periods=15).mean() - 1).where(tr)
    vs = pd.DataFrame(X.V).where(pd.DataFrame(X.V) > 0)
    f["volume_ratio5"] = vs / vs.rolling(5, min_periods=3).mean(); f["turnover5"] = pd.DataFrame(X.TURN).rolling(5, min_periods=3).mean()
    amt20 = pd.DataFrame(X.AMT).rolling(20, min_periods=15).median(); f["log_amt20"] = np.log1p(amt20)
    f["range20"] = ((pd.DataFrame(X.Hh) - pd.DataFrame(X.L)) / pd.DataFrame(X.C)).rolling(20, min_periods=15).mean()
    for k in f: f[k] = f[k].replace([np.inf, -np.inf], np.nan)
    f["amt20"] = amt20
    # P0-A/D/E: pool known at the close of t -- bar at t, historically listed, liquid (20d median turnover amount >= 2e7), enough history
    elig = (tr & pd.DataFrame(X.listed) & (amt20 >= 2e7) & f["r60"].notna() & f["vol20"].notna() & pd.DataFrame(np.isfinite(ADJ) & (ADJ > 0))).values
    return f, elig, ADJf.values
def make_labels(ADJf, elig, h=H):
    """5-day label from t+1 close to t+6 close on the master calendar, market-neutralised over the SAME eligible_asof_t pool (P0-A/B/E)."""
    T, N = ADJf.shape; lab = np.full((T, N), np.nan)
    lab[:T - h - 1] = ADJf[h + 1:] / ADJf[1:T - h] - 1
    lab[~np.isfinite(lab)] = np.nan
    m = elig & np.isfinite(lab); y = np.full((T, N), np.nan); yw = np.full((T, N), np.nan)
    for i in range(T):
        if m[i].sum() < 5: continue
        v = lab[i, m[i]]; yy = v - v.mean(); lo, hi = np.quantile(yy, [0.01, 0.99])
        y[i, m[i]] = yy; yw[i, m[i]] = np.clip(yy, lo, hi)
    return lab, y, yw
def rank_features(f, elig):
    cols = [pd.DataFrame(np.where(elig, f[k].values, np.nan)).rank(axis=1, pct=True).values - 0.5 for k in FEATS]
    return np.nan_to_num(np.stack(cols, axis=2), nan=0.0)
def train_rows(Xq, yw, m_tr, cal, i, years=3):
    """Training rows available at signal index i: label windows (t+1..t+6) must END before i-5 (purge): train dates d <= i-12; 3-year lookback."""
    lo = cal.searchsorted(cal[i] - pd.Timedelta(days=365 * years)); hi = i - 12
    rows = [(Xq[d][m_tr[d]], yw[d][m_tr[d]]) for d in range(lo, hi + 1) if m_tr[d].any()]
    if not rows: return np.zeros((0, Xq.shape[2])), np.zeros(0)
    return np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows])
def simulate(X, ADJf, sig_idx, targets, cost_rt, capital=1e7, lot=100, log_name=None, haircut=0.0):
    """Position/cash state machine on the master calendar (P0-B, P0-F).
    Signal at close of calendar index i; orders at close of i+1; scheduled exit of a name = the next rebalance order day (i+6).
    Buy rejected (suspended/no bar, locked limit-up close, no valid price, lot too small): cash idle, NO replacement name.
    Sell blocked (suspended / limit-down locked): position kept, retried every following trading day (sell delay), keeps market risk.
    Retained names are not rebalanced. Cost: cost_rt is a ROUND-TRIP rate, charged cost_rt/2 on each executed buy value and sell value."""
    C, F, tr, lu, ld = X.C, X.F, X.traded, X.lim_up, X.lim_dn
    T, N = C.shape; c2 = cost_rt / 2.0
    units = np.zeros(N); cash = float(capital); pend = np.zeros(N, bool); plan = np.full(N, -1)
    order_k = {s + 1: k for k, s in enumerate(sig_idx)}; d0, dend = sig_idx[0] + 1, sig_idx[-1] + H + 1
    navs_pre, per, logs = [], [], []
    last_bar = np.where(X.has_row.any(0), T - 1 - np.argmax(X.has_row[::-1], axis=0), -1)
    end_map = {}
    if haircut > 0:   # delisting stress: a position still held when its stock's data ends (delisted, never sellable again) is written down once by `haircut` the day after the final bar
        for j in np.where((last_bar >= 0) & (last_bar < T - 1))[0]: end_map.setdefault(int(last_bar[j]) + 1, []).append(j)
    def lg(**kw):
        if log_name is not None: kw["strategy"] = log_name; logs.append(kw)
    def sell(d, j, planned_day, k):
        nonlocal cash
        if tr[d, j] and not ld[d, j]:
            val = units[j] * ADJf[d, j]; fee = val * c2; cash += val - fee
            lg(signal_date=str(X.cal[sig_idx[k]].date()) if k is not None else "", order_date=str(X.cal[planned_day].date()), code=X.codes[j], side="SELL", planned_date=str(X.cal[planned_day].date()),
               fill_date=str(X.cal[d].date()), fill_price=float(C[d, j]), status="FILLED", reason="", sell_delay_days=int(d - planned_day), shares=float(units[j] / F[d, j]), value=float(val), cost=float(fee))
            units[j] = 0.0; pend[j] = False; plan[j] = -1; return True, ""
        return False, ("suspended_or_no_bar" if not tr[d, j] else "limit_down_locked")
    for d in range(d0, dend + 1):
        k = order_k.get(d)
        for j in end_map.get(d, []):
            if units[j] > 0: units[j] *= (1.0 - haircut)
        if k is not None:
            val = np.nan_to_num(units * ADJf[d]); nav = cash + val.sum(); navs_pre.append(nav)
            tgt = targets[k]; tset = np.zeros(N, bool); tset[tgt] = True; held = units > 0
            st = dict(k=k, n_target=len(tgt), replaced=float((tset & ~held).sum()) / max(1, len(tgt)), nav_pre=nav)
            sell_val = 0.0; n_due = n_ok = 0
            pend[held & tset] = False; plan[held & tset] = -1
            for j in np.where(held & ~tset)[0]:
                if plan[j] < 0: plan[j] = d
                n_due += 1 if plan[j] == d else 0
                v = units[j] * ADJf[d, j]; ok, why = sell(d, j, plan[j], k)
                if ok: sell_val += v; n_ok += int(plan[j] == -1 and True)
                else:
                    pend[j] = True; lg(signal_date=str(X.cal[sig_idx[k]].date()), order_date=str(X.cal[d].date()), code=X.codes[j], side="SELL", planned_date=str(X.cal[d].date()), fill_date="", fill_price=np.nan, status="BLOCKED_RETRY", reason=why, sell_delay_days=0, shares=float(units[j] / F[d, j]), value=float(v), cost=0.0)
            new = np.where(tset & ~held)[0]; tv = nav / max(1, len(tgt)); cand = []
            n_buy_att, n_buy_ok = len(new), 0
            rej = {}
            for j in new:
                why = None
                if not tr[d, j]: why = "suspended_or_no_bar"
                elif lu[d, j]: why = "limit_up_locked"
                elif not (np.isfinite(ADJf[d, j]) and C[d, j] > 0): why = "no_valid_price"
                if why: rej[why] = rej.get(why, 0) + 1; lg(signal_date=str(X.cal[sig_idx[k]].date()), order_date=str(X.cal[d].date()), code=X.codes[j], side="BUY", planned_date=str(X.cal[d].date()), fill_date="", fill_price=float(C[d, j]) if np.isfinite(C[d, j]) else np.nan, status="REJECTED", reason=why, sell_delay_days=0, shares=0.0, value=0.0, cost=0.0)
                else: cand.append(j)
            buy_val = 0.0
            if cand:
                s = min(1.0, cash / (tv * (1 + c2) * len(cand))) if tv > 0 else 0.0
                for j in cand:
                    sh = np.floor(tv * s / (1 + c2) / (C[d, j] * lot)) * lot; v = sh * C[d, j]
                    if sh <= 0: rej["lot_too_small"] = rej.get("lot_too_small", 0) + 1; lg(signal_date=str(X.cal[sig_idx[k]].date()), order_date=str(X.cal[d].date()), code=X.codes[j], side="BUY", planned_date=str(X.cal[d].date()), fill_date="", fill_price=float(C[d, j]), status="REJECTED", reason="lot_too_small_or_no_cash", sell_delay_days=0, shares=0.0, value=0.0, cost=0.0); continue
                    fee = v * c2; cash -= v + fee; units[j] = sh * F[d, j]; buy_val += v; n_buy_ok += 1
                    lg(signal_date=str(X.cal[sig_idx[k]].date()), order_date=str(X.cal[d].date()), code=X.codes[j], side="BUY", planned_date=str(X.cal[d].date()), fill_date=str(X.cal[d].date()), fill_price=float(C[d, j]), status="FILLED", reason="", sell_delay_days=0, shares=float(sh), value=float(v), cost=float(fee))
            navp = cash + np.nansum(units * ADJf[d])
            st.update(buy_attempts=n_buy_att, buy_filled=n_buy_ok, sell_due_new=int(n_due), turnover_value=(buy_val + sell_val) / (2 * nav), cash_frac_post=cash / navp, n_hold=int((units > 0).sum()),
                      n_pending_exit=int((pend & (units > 0)).sum()), **{f"rej_{a}": b for a, b in rej.items()})
            per.append(st)
        else:
            for j in np.where(pend & (units > 0))[0]: sell(d, j, plan[j], None)
    navE = cash + np.nansum(units * ADJf[dend]); navs = navs_pre + [navE]
    rets = np.array(navs[1:]) / np.array(navs[:-1]) - 1
    stuck = (units > 0) & (last_bar < dend)
    out = pd.DataFrame(per); out["ret"] = rets; out["signal_idx"] = sig_idx
    return out, pd.DataFrame(logs), dict(terminal_unsold=int(stuck.sum()), terminal_unsold_codes=[X.codes[j] for j in np.where(stuck)[0]])
def perf(r, h=H):
    r = pd.Series(r).dropna().astype(float); ppy = 252 / h; eq = (1 + r).cumprod()
    if len(r) < 3: return {"periods": int(len(r))}
    ann = float(eq.iloc[-1] ** (ppy / len(r)) - 1); mdd = float((eq / eq.cummax() - 1).min())
    return dict(periods=int(len(r)), total_return=float(eq.iloc[-1] - 1), annualized_return=ann, annualized_vol=float(r.std() * np.sqrt(ppy)),
                sharpe=float(r.mean() / r.std() * np.sqrt(ppy)) if r.std() > 0 else np.nan, max_drawdown=mdd, win_rate=float((r > 0).mean()))
from scipy import stats as _st
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    t = x.mean() / (x.std(ddof=1) / np.sqrt(n)); return float(t), float(2 * _st.t.sf(abs(t), n - 1))
def nw_t(x, lag):
    """Newey-West (Bartlett) HAC t-stat for the mean; p-value from t(n-1)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x); e = x - x.mean(); g0 = (e @ e) / n; v = g0
    for l in range(1, lag + 1): v += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    se = np.sqrt(v / n); t = x.mean() / se; return float(t), float(2 * _st.t.sf(abs(t), n - 1))
def block_boot(x, block=5, B=4000, seed=20261018):
    """Moving-block bootstrap of the mean, 95% percentile interval."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x); rng = np.random.default_rng(seed); nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, nb)); idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    m = x[idx].mean(axis=1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
