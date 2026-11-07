# -*- coding: utf-8 -*-
"""P6 shared helpers. Reads P4 caches and P5 results READ-ONLY. Engine logic mirrored (not imported for accounting) for independent reconciliation."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
P6 = Path(__file__).resolve().parent; V2 = P6.parent; P4D = V2 / "p4_top5_quant"; P5D = V2 / "p5_best_top5"
sys.path.insert(0, str(P4D)); sys.path.insert(0, str(P5D))
import p4lib as P
import p5_common as Q5
C = P4D / "cache"; PR = P4D / "preds"
MERGERS = {"002013": dict(name="中航机电", acquirer="600372", acquirer_name="中航电子(现中航机载)", ratio=0.6647,
                          record_date="2023-03-16", new_share_list_date="2023-04-14",
                          acq_close_record=17.56, acq_open_listing=17.25,
                          source="巨潮 1216135561.PDF: 1股002013换0.6647股600372, 股权登记日2023-03-16, 3-17摘牌")}
def load():
    return Q5.load()
def filt(W, i): return Q5.filt(W, i)
def cons(W, i): return Q5.cons(W, i)
def plain(W, M, rows, K=5): return Q5.plain(W, M, rows, K)
def select_cons_only(W, M, rows, K=5):
    """Top20 -> constraints (volp/dd60/r20/industry<=2) -> Top5. NO buffer."""
    out = []
    for q in rows:
        i = W.sig_idx[q]; f = filt(W, i) & np.isfinite(M[q]); s = np.where(f, M[q], np.nan)
        ok = np.where(np.isfinite(s))[0]; order = ok[np.argsort(-s[ok], kind="stable")]; ok_c = cons(W, i); pick, cnt = [], {}
        for j in order[:20]:
            if len(pick) < K and ok_c[j] and cnt.get(W.ind[j], 0) < 2 and j not in pick:
                pick.append(j); cnt[W.ind[j]] = cnt.get(W.ind[j], 0) + 1
        out.append(np.array(pick, int))
    return out
def select_buffer_only(W, M, rows, K=5, NB=40):
    """Top20 -> Top5 with buffer (keep held while rank<40). NO volp/dd60/r20/industry constraints."""
    out, prev = [], []
    for q in rows:
        i = W.sig_idx[q]; f = filt(W, i) & np.isfinite(M[q]); s = np.where(f, M[q], np.nan)
        ok = np.where(np.isfinite(s))[0]; order = ok[np.argsort(-s[ok], kind="stable")]; rank = {j: r for r, j in enumerate(order)}; pick = []
        def add(j):
            if len(pick) < K and j not in pick: pick.append(j)
        for j in prev:
            if j in rank and rank[j] < NB: add(j)
        for j in order[:20]: add(j)
        out.append(np.array(pick, int)); prev = pick
    return out
def select_both(W, M, rows, K=5, NB=40):
    """Identical to P5 B1 / S4 selection module: Top20 -> constraints -> buffer. (= p5_common.select with blend=False)"""
    return Q5.select(W, M, rows, buffer=True, blend=False, K=K, NB=NB)
def rows_for(W, h): return Q5.rows_for(W, h)
def hac_t(x, lag): return Q5.hac_t(x, lag)
def block_ci(x, block, B=5000, seed=20261024): return Q5.block_ci(x, block, B=B, seed=seed)
def mdd(r): return P.mdd(r)


COST_RT = 0.003; C2 = COST_RT / 2; K = 5; INIT = 1_000_000.0

def independent_backtest(W, picks, rows, h, nm, apply_mergers=True):
    """Independent cash/equity ledger (same economics as p4lib.backtest + optional merger revaluation)."""
    import numpy as np, pandas as pd
    X = W.X; P = __import__("p4lib", fromlist=["pool_bench"])  # already loaded as module via Q
    import p4lib as P
    T = X.C.shape[0]; cal = W.cal; codes = W.codes; col = W.col
    Oa = np.where(X.traded, X.O / X.F, np.nan); Ca = np.where(X.traded, X.C / X.F, np.nan)
    si = [W.sig_idx[q] for q in rows]; seq = [list(p)[:K] for p in picks]
    equity = INIT; eq_curve = []; periods = []; trades = []; prev = set()
    for k, i in enumerate(si):
        e = i + 1; x = i + 1 + h; nxt = set(seq[k + 1]) if k + 1 < len(seq) else set()
        slot_cap = equity / K; period_end_value = 0.0; n_new = n_sell = 0; cur = set(); n_held = 0
        slot_rets_g, slot_rets_n = [], []
        for r_, j in enumerate(seq[k]):
            code = codes[j]; name = W.names[j]; why = None
            if e >= T: why = "no_future_bar"
            elif W.st[i, j]: why = "st_proxy"
            elif not X.traded[e, j]: why = "suspended_at_entry"
            elif W.olu[e, j]: why = "open_limit_up"
            if why:
                trades.append(dict(strategy=nm, horizon=h, signal_date=str(cal[i].date()), rank=r_+1, code=code, name=name, status="SKIPPED_"+why,
                                   entry_date="", exit_date="", entry_open_adj=np.nan, exit_px_adj=np.nan, slot_ret_gross=0.0, slot_ret_net=0.0,
                                   buy_cost=0.0, sell_cost=0.0, slot_start_cash=slot_cap, slot_end_value=slot_cap, retained_from_prev=False, keep_into_next=False, merger_note=""))
                period_end_value += slot_cap; continue
            d = x; delayed = 0
            while d < T and (not X.traded[d, j] or W.old[d, j]): d += 1; delayed += 1
            if d >= T:
                last = np.where(X.traded[:, j])[0]; dd = int(last[-1]); px = Ca[dd, j]; st = "MARKED_LAST_CLOSE_UNSOLD"
            else: px = Oa[d, j]; dd = d; st = "FILLED"
            merger_note = ""
            if apply_mergers and st == "MARKED_LAST_CLOSE_UNSOLD" and code in MERGERS:
                m = MERGERS[code]; last = int(np.where(X.traded[:, j])[0].max()); fadj = X.F[last, j] / X.F[e, j]
                ret = m["ratio"] * m["acq_close_record"] * fadj / X.O[e, j] - 1
                px = Oa[e, j] * (1 + ret); st = "MERGER_SWAP_RECORD"
                merger_note = f"swap {m['ratio']}*{m['acquirer']}@{m['acq_close_record']}; ret={ret:.6f}"
            g = px / Oa[e, j] - 1
            held_before = j in prev; keep_after = j in nxt
            cb = 0.0 if held_before else C2; cs = 0.0 if keep_after else C2
            end_after_sell = slot_cap * (1 - cb) * (1 + g) * (1 - cs); net = end_after_sell / slot_cap - 1
            n_new += int(not held_before); n_sell += int(not keep_after); cur.add(j); n_held += 1
            slot_rets_g.append(g); slot_rets_n.append(net); period_end_value += end_after_sell
            trades.append(dict(strategy=nm, horizon=h, signal_date=str(cal[i].date()), rank=r_+1, code=code, name=name, status=st,
                               entry_date=str(cal[e].date()), exit_date=str(cal[dd].date()) if dd >= 0 else "",
                               entry_open_adj=float(Oa[e, j]), exit_px_adj=float(px), entry_open_unadj=float(X.O[e, j]),
                               exit_delay_days=delayed, slot_ret_gross=float(g), slot_ret_net=float(net),
                               buy_cost=cb, sell_cost=cs, slot_start_cash=slot_cap, slot_end_value=end_after_sell,
                               retained_from_prev=held_before, keep_into_next=keep_after, merger_note=merger_note))
        gs = sum(slot_rets_g) / K; ns = sum(slot_rets_n) / K; equity = equity * (1 + ns)
        pool = P.pool_bench(X, [i], h, W.elig, W.olu)[0]
        periods.append(dict(strategy=nm, horizon=h, signal_date=str(cal[i].date()), n_picks=len(seq[k]), n_held=n_held,
                            ret_gross=gs, ret_net=ns, fees=gs - ns, turnover=(n_new + n_sell) / (2 * K),
                            equity_start=equity / (1 + ns), equity_end=equity, pool_ret=pool,
                            top5=";".join(codes[j] for j in seq[k]), holding_days=h,
                            entry_date=str(cal[e].date()) if e < T else "", planned_exit_date=str(cal[x].date()) if x < T else "",
                            next_signal_date=str(cal[si[k+1]].date()) if k+1 < len(si) else "",
                            overlap_with_next="none (exit open = next entry open)"))
        eq_curve.append(dict(strategy=nm, horizon=h, signal_date=str(cal[i].date()), equity=equity, ret_net=ns))
        prev = cur
    PER = pd.DataFrame(periods); EQ = pd.DataFrame(eq_curve)
    if len(EQ): EQ["drawdown"] = EQ.equity / EQ.equity.cummax() - 1
    return PER, pd.DataFrame(trades), EQ
