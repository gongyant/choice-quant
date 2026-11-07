# -*- coding: utf-8 -*-
"""P2-B execution engine.  `simulate_ext` is p0lib.simulate's state machine (same order of operations, same buy/sell rejection and retry rules,
same lot sizing arithmetic) extended ONLY by switches that are OFF by default:
  * decide(k, ctx) callback: the portfolio decision for signal k is taken at the CLOSE of signal day s using only the holdings snapshot known then
    (units/cash before any order of day s+1) -- decision (time s) and execution (day s+1 ...) are separate objects and are both logged;
  * cost model object (flat round-trip rate == p0lib, or an explicit OHLCV spread/impact/commission/stamp model); every cost component is booked once;
  * optional participation cap (fraction of PRIOR-day ADV, partial fills), minimum ADV for buys;
  * optional weight-driven rebalancing (A2): partial reduce / top-up of retained names.
With defaults (flat cost, no caps, rebalance=False, equal-weight targets) the output is bit-identical to p0lib.simulate (tests/test_p2b.py)."""
import numpy as np, pandas as pd
import p2lib as Q
import p0lib as L

# ======================= cost models =======================
class FlatCost:
    kind = "flat"
    def __init__(self, rt): self.rt = rt; self.c2 = rt / 2.0
    def rate(self, side, d, j, value): return np.full(len(j), self.c2)
    def fee(self, side, d, j, value): return {"flat": value * self.c2}
    def describe(self): return dict(kind="flat", round_trip=self.rt)

class ExplicitCost:
    """Explicit OHLCV-based execution cost. Fills are at the (unadjusted) close; every component below is a cash cost booked once per executed side:
       commission  : comm_bp per side (default 2.5bp; no minimum fee modelled, order sizes >> 5 RMB min)
       transfer    : transfer_bp per side (0.1bp)
       stamp       : sell side only, 10bp before 2023-08-28, 5bp from 2023-08-28 (actual A-share schedule)
       spread      : half-spread = 0.5 * S, S = trailing-20d mean Corwin-Schultz (2012) high-low spread estimate (negatives -> 0) floored at one tick/price
                     (0.01/close) and capped at 2%, then SHRUNK by hs_mult=0.25 (declared a priori: raw Corwin-Schultz on A-share daily bars gives median half-spreads of ~0.43%, far above 1-2 ticks, because it also absorbs
                     limit-hit/gap volatility; the 0.25 shrink gives a median half-spread ~10bp; stress runs use 0.5x/2x). Matrices are as of d-1 (strictly prior information)
       impact      : square-root law  Y * sigma_d * sqrt(value / ADV)  with sigma_d = vol20 (as of d-1), ADV = 20d median turnover amount (as of d-1), capped at impact_cap
       No separate slippage/'cost buffer' is added on top (flat 0.30% is the ALTERNATIVE regime, never stacked)."""
    kind = "explicit"
    def __init__(self, cal, HS, VOL, ADV, comm_bp=2.5, transfer_bp=0.1, stamp_bp_early=10.0, stamp_bp_late=5.0, stamp_change="2023-08-28", Y=0.5, impact_cap=0.05, hs_mult=0.25, imp_mult=1.0):
        self.cal, self.HS, self.VOL, self.ADV = cal, HS, VOL, ADV; self.comm = comm_bp * 1e-4; self.tr = transfer_bp * 1e-4
        self.s_early, self.s_late = stamp_bp_early * 1e-4, stamp_bp_late * 1e-4; self.chg = pd.Timestamp(stamp_change); self.Y = Y; self.icap = impact_cap
        self.hs_mult, self.imp_mult = hs_mult, imp_mult
        self.params = dict(comm_bp=comm_bp, transfer_bp=transfer_bp, stamp_bp_early=stamp_bp_early, stamp_bp_late=stamp_bp_late, stamp_change=stamp_change, Y=Y, impact_cap=impact_cap, hs_mult=hs_mult, imp_mult=imp_mult)
    def stamp(self, d): return self.s_early if self.cal[d] < self.chg else self.s_late
    def _parts(self, side, d, j, value):
        j = np.atleast_1d(j); value = np.atleast_1d(np.asarray(value, float)); p = d - 1     # prior-day information only
        hs = self.hs_mult * self.HS[p, j]; imp = np.minimum(self.imp_mult * self.Y * self.VOL[p, j] * np.sqrt(np.maximum(value, 0) / self.ADV[p, j]), self.icap)
        st = self.stamp(d) if side == "sell" else 0.0
        return dict(commission=np.full(len(j), self.comm), transfer=np.full(len(j), self.tr), stamp=np.full(len(j), st), spread=hs, impact=imp)
    def rate(self, side, d, j, value):
        return sum(self._parts(side, d, j, value).values())
    def fee(self, side, d, j, value):
        parts = self._parts(side, d, j, value); return {k: float(v[0]) * value for k, v in parts.items()}
    def describe(self): return dict(kind="explicit", **self.params)

def corwin_schultz_half_spread(Hh, Lw, C, window=20, cap=0.02):
    """Daily Corwin-Schultz spread from unadjusted high/low of days t-1 and t; 20d trailing mean (min 10 obs) as of t; floor = 1 tick / close; returns HALF spread matrix."""
    H, Lo = np.where(Hh > 0, Hh, np.nan), np.where(Lw > 0, Lw, np.nan)
    hl = np.log(H / Lo) ** 2; beta = hl + np.vstack([np.full((1, H.shape[1]), np.nan), hl[:-1]])
    H2 = np.fmax(H, np.vstack([np.full((1, H.shape[1]), np.nan), H[:-1]])); L2 = np.fmin(Lo, np.vstack([np.full((1, H.shape[1]), np.nan), Lo[:-1]]))
    gamma = np.log(H2 / L2) ** 2; k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    S = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)); S = np.where(np.isfinite(S), np.maximum(S, 0.0), np.nan)
    S20 = pd.DataFrame(S).rolling(window, min_periods=10).mean().values
    floor = 0.01 / np.where(C > 0, C, np.nan)
    Cf = pd.DataFrame(floor).ffill().values
    S20 = np.fmax(S20, Cf); S20 = np.minimum(S20, cap)
    return 0.5 * pd.DataFrame(S20).ffill().values

# ======================= engine =======================
def simulate_ext(X, ADJf, sig_idx, decide, cost, capital=1e7, lot=100, rebalance=False, pcap=None, ADV=None, min_adv=None, min_trade_frac=0.002, name="", H=L.H, keep_logs=True):
    C, F, tr, lu, ld = X.C, X.F, X.traded, X.lim_up, X.lim_dn
    T, N = C.shape
    units = np.zeros(N); cash = float(capital); pend = np.zeros(N, bool); plan = np.full(N, -1)
    order_k = {s + 1: k for k, s in enumerate(sig_idx)}; d0, dend = sig_idx[0] + 1, sig_idx[-1] + H + 1
    navs_pre, per, logs, decisions = [], [], [], []; acc = {}; min_cash = [cash]
    last_bar = np.where(X.has_row.any(0), T - 1 - np.argmax(X.has_row[::-1], axis=0), -1)
    def lg(**kw):
        if keep_logs: kw["strategy"] = name; logs.append(kw)
    def book(comps):
        t = 0.0
        for a, b in comps.items(): acc[a] = acc.get(a, 0.0) + b; t += b
        return t
    def dstr(i): return str(X.cal[i].date())
    def sell(d, j, planned_day, k, shares_limit=None, reduce_only=False):
        """Sell (all of / at most shares_limit of) the position at the close of d. Returns (position_closed, reason). Same rules as p0lib.sell: blocked if no bar or locked limit-down."""
        nonlocal cash
        if tr[d, j] and not ld[d, j]:
            raw = units[j] / F[d, j]; qty = raw
            if shares_limit is not None: qty = min(qty, shares_limit)
            partial_cap = False
            if pcap:
                capsh = np.floor(pcap * ADV[d - 1, j] / C[d, j] / lot) * lot
                if qty > capsh: qty, partial_cap = capsh, True
            if qty <= 0: return False, "participation_cap"
            frac = qty / raw; u_sold = units[j] if frac >= 1.0 else units[j] * frac
            val = u_sold * ADJf[d, j]; comps = cost.fee("sell", d, j, val); fee = book(comps); cash += val - fee; min_cash[0] = min(min_cash[0], cash)
            full = frac >= 1.0
            lg(signal_date=dstr(sig_idx[k]) if k is not None else "", order_date=dstr(planned_day), code=X.codes[j], side="SELL", planned_date=dstr(planned_day), fill_date=dstr(d), fill_price=float(C[d, j]),
               status="FILLED" if full else "PARTIAL_FILLED", reason="" if not partial_cap else "participation_cap", sell_delay_days=int(d - planned_day), shares=float(qty), value=float(val), cost=float(fee), **{"cost_" + a: b for a, b in comps.items()})
            if full: units[j] = 0.0; pend[j] = False; plan[j] = -1
            else: units[j] -= u_sold
            return full, ("" if full else "partial_fill")
        return False, ("suspended_or_no_bar" if not tr[d, j] else "limit_down_locked")
    for d in range(d0, dend + 1):
        k = order_k.get(d)
        if k is not None:
            if per: per[-1].update({"cost_" + a: b for a, b in acc.items()}); acc.clear()
            val = np.nan_to_num(units * ADJf[d]); nav = cash + val.sum(); navs_pre.append(nav)
            ctx = dict(k=k, s=int(sig_idx[k]), d=d, units=units.copy(), cash=float(cash), pend=pend.copy())
            dec = decide(k, ctx); tgt = np.asarray(dec["tgt"], dtype=int); w = dec.get("w"); decisions.append(dict(k=k, s=int(sig_idx[k]), tgt=tgt.copy(), w=None if w is None else np.asarray(w, float).copy(), meta=dec.get("meta", {})))
            tset = np.zeros(N, bool); tset[tgt] = True; held = units > 0
            st = dict(k=k, n_target=len(tgt), replaced=float((tset & ~held).sum()) / max(1, len(tgt)), nav_pre=nav)
            sell_val = 0.0; n_due = n_ok = 0
            pend[held & tset] = False; plan[held & tset] = -1
            for j in np.where(held & ~tset)[0]:
                if plan[j] < 0: plan[j] = d
                n_due += 1 if plan[j] == d else 0
                v = units[j] * ADJf[d, j]; u_before = units[j]; ok, why = sell(d, j, plan[j], k)
                if ok: sell_val += v; n_ok += int(plan[j] == -1 and True)
                else:
                    pend[j] = True
                    if units[j] < u_before: sell_val += (u_before - units[j]) * ADJf[d, j]      # partial (participation-capped) fill
                    else: lg(signal_date=dstr(sig_idx[k]), order_date=dstr(d), code=X.codes[j], side="SELL", planned_date=dstr(d), fill_date="", fill_price=np.nan, status="BLOCKED_RETRY", reason=why, sell_delay_days=0, shares=float(units[j] / F[d, j]), value=float(v), cost=0.0)
            if rebalance:
                desired = np.zeros(N); desired[tgt] = (np.asarray(w, float) * nav)
                for j in np.where(held & tset)[0]:
                    if not np.isfinite(C[d, j]): continue
                    cur = units[j] * ADJf[d, j]; delta = desired[j] - cur; band = max(min_trade_frac * nav, lot * C[d, j])
                    if delta < -band:
                        sh = np.floor(-delta / C[d, j] / lot) * lot
                        if sh > 0:
                            u_b = units[j]; ok, why = sell(d, j, d, k, shares_limit=sh)
                            if units[j] < u_b: sell_val += (u_b - units[j]) * ADJf[d, j]
                            else: lg(signal_date=dstr(sig_idx[k]), order_date=dstr(d), code=X.codes[j], side="SELL", planned_date=dstr(d), fill_date="", fill_price=np.nan, status="BLOCKED_REDUCE", reason=why, sell_delay_days=0, shares=float(sh), value=float(-delta), cost=0.0)
                    pend[j] = False
                buy_list = [j for j in tgt if (units[j] <= 0 or (desired[j] - units[j] * ADJf[d, j]) > max(min_trade_frac * nav, lot * (C[d, j] if np.isfinite(C[d, j]) else 1e9)))]
                new = np.array(sorted(buy_list), dtype=int)
                tv_vec = np.array([desired[j] - (units[j] * ADJf[d, j] if units[j] > 0 else 0.0) for j in new]) if len(new) else np.zeros(0)
            else:
                new = np.where(tset & ~held)[0]; tv = nav / max(1, len(tgt)); tv_vec = np.full(len(new), tv)
            cand, cand_tv = [], []; n_buy_att, n_buy_ok = len(new), 0; rej = {}
            for jj, j in enumerate(new):
                why = None
                if not tr[d, j]: why = "suspended_or_no_bar"
                elif lu[d, j]: why = "limit_up_locked"
                elif not (np.isfinite(ADJf[d, j]) and C[d, j] > 0): why = "no_valid_price"
                elif min_adv is not None and not (ADV[d - 1, j] >= min_adv): why = "below_min_adv"
                if why: rej[why] = rej.get(why, 0) + 1; lg(signal_date=dstr(sig_idx[k]), order_date=dstr(d), code=X.codes[j], side="BUY", planned_date=dstr(d), fill_date="", fill_price=float(C[d, j]) if np.isfinite(C[d, j]) else np.nan, status="REJECTED", reason=why, sell_delay_days=0, shares=0.0, value=0.0, cost=0.0)
                else: cand.append(j); cand_tv.append(tv_vec[jj])
            buy_val = 0.0
            if cand:
                cand = np.array(cand, dtype=int); cand_tv = np.array(cand_tv, float); rate = cost.rate("buy", d, cand, cand_tv)
                if not rebalance and np.all(rate == rate[0]): denom = cand_tv[0] * (1 + rate[0]) * len(cand)      # identical arithmetic to p0lib
                else: denom = float((cand_tv * (1 + rate)).sum())
                s = min(1.0, cash / denom) if denom > 0 else 0.0
                for q, j in enumerate(cand):
                    tvj = cand_tv[q]; sh = np.floor(tvj * s / (1 + rate[q]) / (C[d, j] * lot)) * lot; capped = False
                    if pcap:
                        capsh = np.floor(pcap * ADV[d - 1, j] / (C[d, j] * lot)) * lot
                        if sh > capsh: sh, capped = capsh, True
                    v = sh * C[d, j]
                    if sh <= 0:
                        rej["lot_too_small"] = rej.get("lot_too_small", 0) + 1; lg(signal_date=dstr(sig_idx[k]), order_date=dstr(d), code=X.codes[j], side="BUY", planned_date=dstr(d), fill_date="", fill_price=float(C[d, j]), status="REJECTED", reason="lot_too_small_or_no_cash" if not capped else "participation_cap", sell_delay_days=0, shares=0.0, value=0.0, cost=0.0); continue
                    comps = cost.fee("buy", d, j, v); fee = book(comps); cash -= v + fee; min_cash[0] = min(min_cash[0], cash); units[j] += sh * F[d, j]; buy_val += v; n_buy_ok += 1
                    lg(signal_date=dstr(sig_idx[k]), order_date=dstr(d), code=X.codes[j], side="BUY", planned_date=dstr(d), fill_date=dstr(d), fill_price=float(C[d, j]), status="FILLED" if not capped else "PARTIAL_FILLED", reason="" if not capped else "participation_cap", sell_delay_days=0, shares=float(sh), value=float(v), cost=float(fee), target_value=float(tvj), **{"cost_" + a: b for a, b in comps.items()})
                    if capped: rej["partial_cap"] = rej.get("partial_cap", 0) + 1
            navp = cash + np.nansum(units * ADJf[d]); wmax = float(np.nanmax(units * ADJf[d]) / navp) if (units > 0).any() else 0.0
            st.update(buy_attempts=n_buy_att, buy_filled=n_buy_ok, sell_due_new=int(n_due), turnover_value=(buy_val + sell_val) / (2 * nav), buy_value=buy_val, sell_value=sell_val, cash_frac_post=cash / navp, n_hold=int((units > 0).sum()),
                      n_pending_exit=int((pend & (units > 0)).sum()), max_weight_post=wmax, **{f"rej_{a}": b for a, b in rej.items()})
            per.append(st)
        else:
            for j in np.where(pend & (units > 0))[0]: sell(d, j, plan[j], None)
    if per: per[-1].update({"cost_" + a: b for a, b in acc.items()})
    navE = cash + np.nansum(units * ADJf[dend]); navs = navs_pre + [navE]
    rets = np.array(navs[1:]) / np.array(navs[:-1]) - 1
    stuck = (units > 0) & (last_bar < dend)
    out = pd.DataFrame(per); out["ret"] = rets; out["signal_idx"] = sig_idx; out["nav_next"] = navs[1:]
    cc = [c for c in out.columns if c.startswith("cost_")]
    for c in cc: out[c] = out[c].fillna(0.0)
    out["cost_total"] = out[cc].sum(axis=1) if cc else 0.0
    out["gross_ret"] = (out.nav_next + out.cost_total) / out.nav_pre - 1      # return before costs paid in the period (costs not compounded)
    return out, pd.DataFrame(logs), dict(terminal_unsold=int(stuck.sum()), min_cash=float(min_cash[0]), decisions=decisions)

# ======================= deciders (decision at signal close s, only information <= s) =======================
def rank_order(score_row, cand_mask):
    """pool = indices with cand_mask in ascending order; rank 1 = highest score (stable, NaN last) -- identical to p0 baseline_fixed Top-K selection."""
    pool = np.where(cand_mask)[0]; s = np.nan_to_num(score_row[pool], nan=-1e9); order = np.argsort(-s, kind="stable")
    return pool, pool[order]
class A0Decider:
    def __init__(self, scores, cand, sig_idx, n=50): self.S, self.cand, self.n = scores, cand, n
    def __call__(self, k, ctx):
        pool, ordd = rank_order(self.S[k], self.cand[k]); return dict(tgt=ordd[: self.n], meta=dict(n_pool=len(pool)))
class A1Decider:
    """Hysteresis: buy rank<=N; a HELD name stays while its rank (today, among candidates) <= K_hold, kept in best-rank order up to N names;
    remaining slots are refilled with the best-ranked NOT-held names, so the target always has exactly N names (N = pool size if the pool is smaller); no cash-driven resizing.
    K_hold == N reproduces A0."""
    def __init__(self, scores, cand, sig_idx, n=50, k_hold=80): self.S, self.cand, self.n, self.K = scores, cand, n, k_hold
    def __call__(self, k, ctx):
        pool, ordd = rank_order(self.S[k], self.cand[k]); rank = {int(j): r + 1 for r, j in enumerate(ordd)}; held = np.where(ctx["units"] > 0)[0]
        keep = sorted([int(j) for j in held if int(j) in rank and rank[int(j)] <= self.K], key=lambda j: rank[j])[: self.n]; ks = set(keep)
        fill = [int(j) for j in ordd if int(j) not in ks and not (ctx["units"][j] > 0)][: max(0, min(self.n, len(pool)) - len(keep))]
        tgt = np.array(keep + fill, dtype=int); return dict(tgt=tgt, meta=dict(n_keep=len(keep), n_fill=len(fill), n_pool=len(pool)))
class RandRetainDecider:
    """Turnover-matched naive control: each held name still in the pool is retained with probability p regardless of its rank; refill with best-ranked not-held names."""
    def __init__(self, scores, cand, sig_idx, n=50, p=0.5, seed=0): self.S, self.cand, self.n, self.p = scores, cand, n, p; self.rng = np.random.default_rng(seed)
    def __call__(self, k, ctx):
        pool, ordd = rank_order(self.S[k], self.cand[k]); inpool = set(int(j) for j in pool); held = [int(j) for j in np.where(ctx["units"] > 0)[0] if int(j) in inpool]
        u = self.rng.random(len(held)); keep = [j for j, x in zip(held, u) if x < self.p]
        if len(keep) > self.n: keep = [keep[i] for i in self.rng.permutation(len(keep))[: self.n]]
        ks = set(keep); fill = [int(j) for j in ordd if int(j) not in ks and not (ctx["units"][j] > 0)][: max(0, min(self.n, len(pool)) - len(keep))]
        return dict(tgt=np.array(keep + fill, dtype=int), meta=dict(n_keep=len(keep)))

class RandSkipDecider:
    """Second turnover-matched naive control (stale-but-clean): with probability q the naive Top-N is refreshed, otherwise the target is simply the current holdings (no trading except pending exits)."""
    def __init__(self, scores, cand, sig_idx, n=50, q=0.5, seed=0): self.S, self.cand, self.n, self.q = scores, cand, n, q; self.rng = np.random.default_rng(seed)
    def __call__(self, k, ctx):
        held = np.where(ctx["units"] > 0)[0]
        if k == 0 or len(held) == 0 or self.rng.random() < self.q:
            pool, ordd = rank_order(self.S[k], self.cand[k]); return dict(tgt=ordd[: self.n], meta=dict(refresh=1))
        return dict(tgt=held, meta=dict(refresh=0))

# ---------- A2: cost-aware, risk-constrained long-only optimiser (own dual solver; no external QP library) ----------
def solve_alloc(alpha, a, kp, km, w0, cap, X=None, delta=None, budget=1.0):
    """max  sum(alpha*w - 0.5*a*w^2 - kp*(w-w0)^+ - km*(w0-w)^+)  s.t. 0<=w<=cap, sum(w)<=budget, |X'w|<=delta (columns of X).
    Lagrangian dual (concave separable primal given multipliers): w_j(theta) = clip(w_unc, 0, cap) in closed form; the dual is smooth (strongly concave primal)
    and is minimised with L-BFGS-B over theta=[mu, nu+, nu-] >= 0. Returns (w, info)."""
    from scipy.optimize import minimize
    n = len(alpha); m = 0 if X is None else X.shape[1]; sc = 1e-3
    def w_of(th):
        mu = th[0] * sc; b = alpha - mu
        if m: nu = (th[1:1 + m] - th[1 + m:1 + 2 * m]) * sc; b = b - X @ nu
        w1 = (b - kp) / a; w2 = (b + km) / a
        wu = np.where(w1 > w0, w1, np.where(w2 < w0, w2, w0)); return np.clip(wu, 0.0, cap)
    def obj_grad(th):
        w = w_of(th); mu = th[0] * sc; b = alpha - mu
        if m: nu = (th[1:1 + m] - th[1 + m:1 + 2 * m]) * sc; b = b - X @ nu
        prim = b * w - 0.5 * a * w * w - kp * np.maximum(w - w0, 0) - km * np.maximum(w0 - w, 0)
        g = prim.sum() + mu * budget
        gr = np.zeros(len(th)); gr[0] = (budget - w.sum()) * sc
        if m:
            g += delta * (th[1:1 + m] + th[1 + m:1 + 2 * m]).sum() * sc; xw = X.T @ w
            gr[1:1 + m] = (delta - xw) * sc; gr[1 + m:1 + 2 * m] = (delta + xw) * sc
        return g, gr
    th0 = np.zeros(1 + 2 * m); r = minimize(obj_grad, th0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * len(th0), options=dict(maxiter=300, ftol=1e-14, gtol=1e-10))
    w = w_of(r.x); s = w.sum()
    if s > budget * (1 + 1e-9): w = w * (budget / s)
    return w, dict(success=bool(r.success), n_iter=int(r.nit), sum_w=float(w.sum()), exp=None if not m else (X.T @ w).tolist())
def a2_objective(w, alpha, a, kp, km, w0): return float((alpha * w - 0.5 * a * w * w - kp * np.maximum(w - w0, 0) - km * np.maximum(w0 - w, 0)).sum())

class A2Decider:
    """Cost-aware long-only optimiser. alpha_j = beta_hat_k * z_j, z = uniform-rank score standardised to unit variance within the candidate pool at s,
    beta_hat_k = expanding mean of the per-period slope of the realised relative 5d return (t+1->t+6 label) on z over periods m<=k-2 only (labels already matured at s);
    before 8 matured periods beta0 (declared constant). Risk = diagonal 5-day variance 5*vol20^2 (as of s) times risk aversion lam; turnover cost = tau * one-way cost rate
    (side specific, from the same cost model that executes); box cap per name; sum(w)<=1 (no leverage, long-only); optional |exposure|<=delta (std. units) to log_amt20, vol20, r5."""
    def __init__(self, scores, cand, sig_idx, ADJf, W, X, cost, y_rel, tau=0.5, cap=0.05, delta=None, lam=25.0, M=100, beta0=0.003, beta_floor=0.0005, min_obs=8, capital=1e7, n_hold_ref=50):
        self.S, self.cand, self.sig, self.cost, self.tau, self.cap, self.delta, self.lam, self.M = scores, cand, sig_idx, cost, tau, cap, delta, lam, M
        self.ADJf, self.vol, self.beta0, self.bfloor, self.min_obs, self.cap_cap, self.ref = ADJf, W.f["vol20"].values, beta0, beta_floor, min_obs, capital, n_hold_ref
        self.expo = [W.f[c].values for c in ("log_amt20", "vol20", "r5")]
        self.slopes = []
        for k, s in enumerate(sig_idx):      # slope of realised label on z, available only after label maturity (used with lag >= 2 periods)
            pool = np.where(cand[k])[0]; z = self.z_of(self.S[k], pool); yy = y_rel[s][pool]; ok = np.isfinite(yy) & np.isfinite(z)
            self.slopes.append(float((z[ok] * yy[ok]).sum() / (z[ok] ** 2).sum()) if ok.sum() > 20 else np.nan)
    @staticmethod
    def z_of(score_row, pool):
        sc = np.nan_to_num(score_row[pool], nan=-1e9); r = pd.Series(sc).rank(method="average").values; pct = (r - 0.5) / len(pool)
        z = (pct - 0.5) * np.sqrt(12.0); return z
    def beta_hat(self, k):
        past = [b for b in self.slopes[: max(0, k - 1)] if np.isfinite(b)]       # periods m <= k-2
        if len(past) < self.min_obs: return self.beta0, len(past)
        return max(float(np.mean(past)), self.bfloor), len(past)
    def __call__(self, k, ctx):
        s = self.sig[k]; pool = np.where(self.cand[k])[0]; z = self.z_of(self.S[k], pool); beta, nb = self.beta_hat(k)
        held = np.where(ctx["units"] > 0)[0]; inpool = np.zeros(len(self.S[k]), bool); inpool[pool] = True
        order = np.argsort(-z, kind="stable"); topM = pool[order[: self.M]]
        cs = np.unique(np.concatenate([topM, np.array([j for j in held if inpool[j]], dtype=int)])).astype(int)
        zi = pd.Series(z, index=pool).reindex(cs).values; alpha = beta * zi
        nav_s = ctx["cash"] + np.nansum(ctx["units"] * self.ADJf[s]); w0 = np.nan_to_num(ctx["units"][cs] * self.ADJf[s][cs]) / nav_s
        sig5 = np.sqrt(5.0) * np.maximum(self.vol[s][cs], 0.01); a = self.lam * sig5 ** 2; nominal = nav_s / self.ref
        d_est = s + 1; kp = self.tau * self.cost.rate("buy", d_est, cs, np.full(len(cs), nominal)); km = self.tau * self.cost.rate("sell", d_est, cs, np.full(len(cs), nominal))
        Xm = None
        if self.delta is not None:
            cols = []
            for E in self.expo:
                e = E[s][pool]; mu_, sd_ = np.nanmean(e), np.nanstd(e); zz = np.clip((E[s][cs] - mu_) / sd_, -3, 3); cols.append(np.nan_to_num(zz))
            Xm = np.column_stack(cols)
        w, info = solve_alloc(alpha, a, kp, km, w0, self.cap, Xm, self.delta)
        keep = w > 1e-4; return dict(tgt=cs[keep], w=w[keep], meta=dict(beta_hat=beta, n_matured=nb, n_cand=len(cs), sum_w=float(w[keep].sum()), n_names=int(keep.sum()), success=info["success"], exp=info["exp"]))