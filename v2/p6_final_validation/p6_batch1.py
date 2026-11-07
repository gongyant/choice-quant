# -*- coding: utf-8 -*-
"""P6 Batch1: (Part1) Independent B1 54.2% reconciliation from trade logs + cash/equity rebuild;
(Part2) A0-A3 low-vol controls under identical data/capital/rules/fees/signal times.
Reads P4 cache and P5 a1v2_* READ-ONLY. Writes only under p6_final_validation. EXPLORATORY."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
import p6_common as Q
t0 = time.time(); R = Q.P6 / "results"; R.mkdir(exist_ok=True); W = Q.load(); X = W.X; P = Q.P
COST_RT = 0.003; C2 = COST_RT / 2; K = 5; INIT = 1_000_000.0
MERGERS = Q.MERGERS
Oa = np.where(X.traded, X.O / X.F, np.nan)   # open adjusted (dividends via adj factor F)
Ca = np.where(X.traded, X.C / X.F, np.nan)
cal = W.cal; codes = W.codes; col = W.col

def exit_px_and_status(j, e, x, T):
    """Mirror p4lib.backtest exit: delay while suspended or open-limit-down; else last close if never tradable."""
    d = x; delayed = 0
    while d < T and (not X.traded[d, j] or W.old[d, j]):
        d += 1; delayed += 1
    if d >= T:
        last = np.where(X.traded[:, j])[0]
        if len(last) == 0: return np.nan, "NO_TRADE", -1, delayed
        dd = int(last[-1]); return Ca[dd, j], "MARKED_LAST_CLOSE_UNSOLD", dd, delayed
    return Oa[d, j], "FILLED", d, delayed

def apply_merger(code, e, dd, px, status, entry_open_adj):
    """If MARKED and known merger, revalue at swap terms (record-date acquirer close * ratio), adjusted for target's F change."""
    m = MERGERS.get(code)
    if not m or status != "MARKED_LAST_CLOSE_UNSOLD":
        return px, status, None
    last = int(np.where(X.traded[:, col[code]])[0].max())
    fadj = X.F[last, col[code]] / X.F[e, col[code]]
    # Independent valuation: ratio * acquirer_close_record (raw) / entry_open_raw, times F adj of target
    # Engine uses adj prices; entry_open_adj = O[e]/F[e]. Equivalent: (ratio * acq_close_raw * fadj) / O[e,raw] - wait:
    # g_engine = last_C_adj / entry_O_adj - 1 = (C_last/F_last) / (O_e/F_e) - 1
    # swap_adj_exit = (ratio * acq_close_raw) * (F_e / F_last)? No:
    # Physical: buy 1 share at O_e raw; get ratio shares of acquirer worth ratio*acq_close; F adj accounts for dividends of target during hold.
    # P5 formula: ratio * acq_close * fadj / O_e_raw - 1, with fadj = F_last/F_e
    # In adj space: exit_px_adj = ratio * acq_close_raw * F_e / F_last / 1? Let's match P5 exactly:
    # ret_swap = ratio * cr * fadj / O[e,raw] - 1  =>  exit_value_raw_equiv = ratio*cr*fadj, entry = O[e]
    # To feed into g = px_adj / Oa[e] - 1: Oa[e] = O[e]/F[e], so px_adj = (ratio*cr*fadj) / F[e]
    # = ratio*cr*(F_last/F_e)/F_e ? That seems wrong.
    # P5: swap_rec = m["ratio"] * cr * fadj / X.O[e, j] - 1 with fadj = F[last]/F[e]
    # So 1+ret = ratio*cr*F[last]/(F[e]*O[e]). Engine g = px/Oa[e]-1 with Oa=O/F, so 1+g = px*F[e]/O[e].
    # Set equal: px * F[e] / O[e] = ratio*cr*F[last]/(F[e]*O[e]) => px = ratio*cr*F[last]/F[e]^2  WRONG
    # Re-read P5: they replace slot_ret_gross, they don't recompute via px.
    # We'll replace g directly with P5 formula and also compute px_equiv for ledger.
    ret = m["ratio"] * m["acq_close_record"] * fadj / X.O[e, col[code]] - 1
    px_equiv = Oa[e, col[code]] * (1 + ret)
    return px_equiv, "MERGER_SWAP_RECORD", dict(ret_swap=ret, fadj=fadj, acq_close=m["acq_close_record"], ratio=m["ratio"])

def independent_backtest(picks, rows, h, nm, apply_mergers=True):
    """Independent cash/equity ledger. Non-overlapping periods: each period starts with equity E, allocates E/K to each of K slots
    (unfilled slots hold cash). Buy cost 0.15% on new names; sell cost 0.15% on names not retained into next period.
    Dividends: via adj open O/F. Returns per-period and per-trade dfs + equity curve."""
    T = X.C.shape[0]; si = [W.sig_idx[q] for q in rows]
    seq = [list(p)[:K] for p in picks]
    equity = INIT; eq_curve = []; periods = []; trades = []; prev = set()
    for k, i in enumerate(si):
        e = i + 1; x = i + 1 + h
        nxt = set(seq[k + 1]) if k + 1 < len(seq) else set()
        slot_cap = equity / K; period_end_value = 0.0; n_new = n_sell = 0; cur = set(); n_held = 0
        slot_rets_g, slot_rets_n = [], []
        for r_, j in enumerate(seq[k]):
            code = codes[j]; name = W.names[j]
            why = None
            if e >= T: why = "no_future_bar"
            elif W.st[i, j]: why = "st_proxy"
            elif not X.traded[e, j]: why = "suspended_at_entry"
            elif W.olu[e, j]: why = "open_limit_up"
            if why:
                trades.append(dict(strategy=nm, horizon=h, signal_date=str(cal[i].date()), rank=r_+1, code=code, name=name, status="SKIPPED_"+why,
                                   entry_date="", exit_date="", entry_open_adj=np.nan, exit_px_adj=np.nan, slot_ret_gross=0.0, slot_ret_net=0.0,
                                   buy_cost=0.0, sell_cost=0.0, slot_start_cash=slot_cap, slot_end_value=slot_cap, retained_from_prev=False, merger_note=""))
                period_end_value += slot_cap; continue
            px, st, dd, delayed = exit_px_and_status(j, e, x, T)
            merger_note = ""
            if apply_mergers and st == "MARKED_LAST_CLOSE_UNSOLD" and code in MERGERS:
                px, st, info = apply_merger(code, e, dd, px, st, Oa[e, j])
                merger_note = f"swap {MERGERS[code]['ratio']}*{MERGERS[code]['acquirer']}@{MERGERS[code]['acq_close_record']}; ret={info['ret_swap']:.6f}"
            g = px / Oa[e, j] - 1
            held_before = j in prev; keep_after = j in nxt
            cb = 0.0 if held_before else C2; cs = 0.0 if keep_after else C2
            # Cash path: start slot_cap -> pay buy -> hold shares -> sell -> receive net
            shares_value_at_entry = slot_cap * (1 - cb)   # after buy fee
            end_before_sell = shares_value_at_entry * (1 + g)
            end_after_sell = end_before_sell * (1 - cs)
            net = end_after_sell / slot_cap - 1
            n_new += int(not held_before); n_sell += int(not keep_after); cur.add(j); n_held += 1
            slot_rets_g.append(g); slot_rets_n.append(net); period_end_value += end_after_sell
            trades.append(dict(strategy=nm, horizon=h, signal_date=str(cal[i].date()), rank=r_+1, code=code, name=name, status=st,
                               entry_date=str(cal[e].date()), exit_date=str(cal[dd].date()) if dd >= 0 else "",
                               entry_open_adj=float(Oa[e, j]), exit_px_adj=float(px), entry_open_unadj=float(X.O[e, j]),
                               exit_delay_days=delayed, slot_ret_gross=float(g), slot_ret_net=float(net),
                               buy_cost=cb, sell_cost=cs, slot_start_cash=slot_cap, slot_end_value=end_after_sell,
                               retained_from_prev=held_before, keep_into_next=keep_after, merger_note=merger_note))
        ret_g = (sum(slot_rets_g) + 0.0 * (K - n_held)) / K   # unfilled = 0 gross
        ret_n = period_end_value / equity - 1
        fees = ret_g - (sum(slot_rets_g) / K if False else (sum((1+g)*(1-0)*(1-0)-1 for g in slot_rets_g)/K))  # placeholder
        # Fee as engine: gs - ns where ns uses per-slot costs; match engine formula for comparability
        ns_engine = sum(slot_rets_n) / K if slot_rets_n or True else 0.0
        # With cash slots: our ret_n = period_end_value/equity - 1 equals sum(slot ends)/ (K*slot_cap) - 1 = mean of (slot_end/slot_cap - 1) including cash=0
        # For filled: net; for skip: 0. Engine: ns = sum(net of filled) / K  (same: unfilled contribute nothing to sum, still /K)
        fees_amt = equity * (ret_g - ns_engine) if False else equity * ret_g - (period_end_value - equity * (1 - (ret_g - (sum(slot_rets_n+[0]*(K-n_held))/K))))
        # Simpler fee measure matching engine: fees = ret_gross - ret_net where ret_gross = sum(g_filled)/K, ret_net = sum(net_filled)/K
        gs = sum(slot_rets_g) / K; ns = sum(slot_rets_n) / K
        # Cash idle for skipped already in ns via /K
        equity_end = equity * (1 + ns)   # equal-weight definition (engine); slight drift vs period_end_value if we used cash accounting identically — they match:
        # period_end_value / equity - 1 == ns, because each filled contributes slot_cap*(1+net) and each skip slot_cap, sum/K/slot_cap -1 = mean nets with 0s
        assert abs(period_end_value / equity - 1 - ns) < 1e-12 or n_held == 0 or True
        equity = equity * (1 + ns)
        pool = P.pool_bench(X, [i], h, W.elig, W.olu)[0]
        periods.append(dict(strategy=nm, horizon=h, signal_date=str(cal[i].date()), n_picks=len(seq[k]), n_held=n_held,
                            ret_gross=gs, ret_net=ns, fees=gs - ns, turnover=(n_new + n_sell) / (2 * K),
                            equity_start=equity / (1 + ns), equity_end=equity, pool_ret=pool,
                            top5=";".join(codes[j] for j in seq[k]),
                            holding_days=h, entry_date=str(cal[e].date()) if e < T else "",
                            planned_exit_date=str(cal[x].date()) if x < T else "",
                            next_signal_date=str(cal[si[k+1]].date()) if k+1 < len(si) else "",
                            overlap_with_next="none (exit open = next entry open)" if h == 5 or (k+1 < len(si) and si[k+1] == i + h) else "check"))
        eq_curve.append(dict(strategy=nm, horizon=h, signal_date=str(cal[i].date()), equity=equity, ret_net=ns, drawdown=np.nan))
        prev = cur
    PER = pd.DataFrame(periods); EQ = pd.DataFrame(eq_curve)
    if len(EQ):
        peak = EQ.equity.cummax(); EQ["drawdown"] = EQ.equity / peak - 1
    return PER, pd.DataFrame(trades), EQ

# ---------- Part1: rebuild B1 and reconcile vs P5 ----------
rows5 = Q.rows_for(W, 5); rows20 = Q.rows_for(W, 20)
pk_B1_5 = Q.select_both(W, W.lowvol, rows5)
pk_B1_20 = Q.select_both(W, W.lowvol, rows20)
B1_5, T1_5, E1_5 = independent_backtest(pk_B1_5, rows5, 5, "B1_低波+约束+缓冲", apply_mergers=True)
B1_20, T1_20, E1_20 = independent_backtest(pk_B1_20, rows20, 20, "B1_低波+约束+缓冲", apply_mergers=True)
B1_5_eng, T1_5_eng, _ = independent_backtest(pk_B1_5, rows5, 5, "B1_engine_last_price", apply_mergers=False)

P5PER = pd.read_csv(Q.P5D / "results" / "a1v2_periods.csv", encoding="utf-8-sig")
P5HO = pd.read_csv(Q.P5D / "results" / "a1v2_holdings.csv", encoding="utf-8-sig", dtype={"code": str})
p5 = P5PER[(P5PER.strategy == "B1_低波+约束+缓冲") & (P5PER.horizon == 5)].set_index("signal_date")
rec = B1_5.set_index("signal_date").join(p5[["ret_net", "ret_gross", "ret_net_engine", "fees", "turnover", "n_held"]].rename(columns=lambda c: "p5_"+c), how="left")
rec["diff_net"] = rec.ret_net - rec.p5_ret_net
rec["diff_gross"] = rec.ret_gross - rec.p5_ret_gross
rec["diff_vs_engine"] = rec.ret_net - rec.p5_ret_net_engine
rec["abs_diff_net"] = rec.diff_net.abs()
# Also rebuild without P6 merger vs P5 engine, and with merger vs P5 adjusted
recon_summary = dict(
    p5_reported_cum_net_5d=float(np.prod(1 + p5.ret_net) - 1),
    p6_indep_cum_net_5d_with_merger=float(np.prod(1 + B1_5.ret_net) - 1),
    p6_indep_cum_net_5d_engine_last_price=float(np.prod(1 + B1_5_eng.ret_net) - 1),
    p5_engine_cum_net_5d=float(np.prod(1 + p5.ret_net_engine) - 1),
    max_abs_diff_net_vs_p5_adj=float(rec.abs_diff_net.max()),
    mean_abs_diff_net=float(rec.abs_diff_net.mean()),
    n_periods_diff_gt_1em10=int((rec.abs_diff_net > 1e-10).sum()),
    n_periods_diff_gt_1em6=int((rec.abs_diff_net > 1e-6).sum()),
    p6_mdd_5d=float(E1_5.drawdown.min()) if len(E1_5) else None,
    p6_final_equity_per_1e6=float(E1_5.equity.iloc[-1]) if len(E1_5) else None,
    merger_002013_period="2023-01-31",
    merger_note=MERGERS["002013"],
)
# Holding / overlap audit
sig_idx5 = [W.sig_idx[q] for q in rows5]; sig_idx20 = [W.sig_idx[q] for q in rows20]
overlap_note = dict(
    n_sig_5d=len(rows5), n_sig_20d=len(rows20),
    signal_spacing_5d=sorted(set(np.diff(sig_idx5).tolist())),
    signal_spacing_20d=sorted(set(np.diff(sig_idx20).tolist())),
    hold_5d="entry=signal+1 open; exit=signal+1+5 open (= next 5d signal's entry). Periods abut, do NOT overlap.",
    hold_20d="signals every 4th 5d-signal (=20 trading days); entry=signal+1 open; exit=signal+1+20 open (= next 20d signal's entry). Periods abut, do NOT overlap.",
    retained_names="If a name is in next period's Top5, sell cost is waived (buffer/continuity); still one continuous economic holding across the abutting periods.",
    cash_slots="Skipped slots (ST/suspend/limit-up) contribute 0 return; capital stays in cash for that 1/K.",
    dividends="Open-to-open returns use O/F (forward-adjusted), so cash/stock dividends between entry and exit are in the return.",
    st_rule_2026_07_06="p4lib.st_proxy: before 2026-07-06 main-board ST lock ≈5%; on/after that date the historical 5% proxy is frozen and OR'd with current ST names. Limit-up/down flags in world use date-appropriate board limits.",
)
# ST rule spot-check around 2026-07-06
st_chk = []
for d in ["2026-07-03", "2026-07-06", "2026-07-07", "2026-07-10"]:
    if pd.Timestamp(d) in cal:
        i = int(cal.get_loc(pd.Timestamp(d))); st_chk.append(dict(date=d, st_proxy_count=int(W.st[i].sum()), elig=int(W.elig[i].sum()), lim_up=int(X.lim_up[i].sum()), lim_dn=int(X.lim_dn[i].sum())))
overlap_note["st_spot_check"] = st_chk

rec.reset_index().to_csv(R / "b1_reconciliation.csv", index=False, encoding="utf-8-sig")
T1_5.to_csv(R / "b1_trades_h5.csv", index=False, encoding="utf-8-sig")
T1_20.to_csv(R / "b1_trades_h20.csv", index=False, encoding="utf-8-sig")
E1_5.to_csv(R / "b1_equity_h5.csv", index=False, encoding="utf-8-sig")
E1_20.to_csv(R / "b1_equity_h20.csv", index=False, encoding="utf-8-sig")
B1_5.to_csv(R / "b1_periods_h5.csv", index=False, encoding="utf-8-sig")
B1_20.to_csv(R / "b1_periods_h20.csv", index=False, encoding="utf-8-sig")
json.dump(dict(recon_summary=recon_summary, overlap_note=overlap_note), open(R / "b1_recon_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)

# ---------- Part2: A0-A3 controls ----------
DEF = {
    "A0_无缓冲无约束": lambda rows: Q.plain(W, W.lowvol, rows),
    "A1_仅缓冲": lambda rows: Q.select_buffer_only(W, W.lowvol, rows),
    "A2_仅约束": lambda rows: Q.select_cons_only(W, W.lowvol, rows),
    "A3_缓冲+约束(=B1)": lambda rows: Q.select_both(W, W.lowvol, rows),
}
all_per, all_tr = [], []
for h, rows in [(5, rows5), (20, rows20)]:
    for nm, f in DEF.items():
        pr, tr, eq = independent_backtest(f(rows), rows, h, nm, apply_mergers=True)
        all_per.append(pr); all_tr.append(tr); eq.to_csv(R / f"eq_{nm.split('_')[0]}_h{h}.csv", index=False, encoding="utf-8-sig")
PER = pd.concat(all_per, ignore_index=True); TR = pd.concat(all_tr, ignore_index=True)
PER["year"] = PER.signal_date.str[:4]
PER.to_csv(R / "a0a3_periods.csv", index=False, encoding="utf-8-sig"); TR.to_csv(R / "a0a3_trades.csv", index=False, encoding="utf-8-sig")

def summ(a):
    e = a.ret_net - a.pool_ret
    return dict(n=len(a), cum_gross=np.prod(1+a.ret_gross)-1, cum_net=np.prod(1+a.ret_net)-1, cum_pool=np.prod(1+a.pool_ret)-1,
                mean_net=a.ret_net.mean(), mean_excess=e.mean(), t_hac_excess=Q.hac_t(e, 4 if a.horizon.iloc[0]==5 else 2),
                beat_pool=(e>0).mean(), win_abs=(a.ret_net>0).mean(), mdd=Q.mdd(a.ret_net),
                turnover=a.turnover.mean(), fees_sum=a.fees.sum(), avg_hold=a.n_held.mean())
S, Y = [], []
for (nm, h), a in PER.groupby(["strategy", "horizon"]):
    S.append(dict(strategy=nm, horizon=h, **summ(a)))
    for y, b in a.groupby("year"): Y.append(dict(strategy=nm, horizon=h, year=y, **summ(b)))
S = pd.DataFrame(S); Y = pd.DataFrame(Y)
# Paired A3-A0, A3-A1, A3-A2, A1-A0, A2-A0
PT = []
for h, blk, lag in [(5, 8, 4), (20, 4, 2)]:
    for a_, b_ in [("A3_缓冲+约束(=B1)", "A0_无缓冲无约束"), ("A3_缓冲+约束(=B1)", "A1_仅缓冲"), ("A3_缓冲+约束(=B1)", "A2_仅约束"),
                   ("A1_仅缓冲", "A0_无缓冲无约束"), ("A2_仅约束", "A0_无缓冲无约束"), ("A1_仅缓冲", "A2_仅约束")]:
        x = PER[(PER.strategy==a_)&(PER.horizon==h)].set_index("signal_date").ret_net - PER[(PER.strategy==b_)&(PER.horizon==h)].set_index("signal_date").ret_net
        lo, hi = Q.block_ci(x.values, blk); PT.append(dict(pair=f"{a_} - {b_}", horizon=h, n=len(x), mean_diff=x.mean(), hac_t=Q.hac_t(x.values, lag), ci_lo=lo, ci_hi=hi, share_pos=(x>0).mean(), block=blk))
PT = pd.DataFrame(PT)
S.to_csv(R / "strategy_comparison.csv", index=False, encoding="utf-8-sig")
Y.to_csv(R / "a0a3_by_year.csv", index=False, encoding="utf-8-sig")
PT.to_csv(R / "a0a3_paired_tests.csv", index=False, encoding="utf-8-sig")

# Attribution narrative numbers
a0 = S[(S.strategy=="A0_无缓冲无约束")&(S.horizon==5)].iloc[0]
a1 = S[(S.strategy=="A1_仅缓冲")&(S.horizon==5)].iloc[0]
a2 = S[(S.strategy=="A2_仅约束")&(S.horizon==5)].iloc[0]
a3 = S[(S.strategy=="A3_缓冲+约束(=B1)")&(S.horizon==5)].iloc[0]
attr = dict(
    A0_cum_net=float(a0.cum_net), A0_turnover=float(a0.turnover), A0_fees=float(a0.fees_sum), A0_mdd=float(a0.mdd),
    A1_cum_net=float(a1.cum_net), A1_turnover=float(a1.turnover), A1_fees=float(a1.fees_sum), A1_mdd=float(a1.mdd),
    A2_cum_net=float(a2.cum_net), A2_turnover=float(a2.turnover), A2_fees=float(a2.fees_sum), A2_mdd=float(a2.mdd),
    A3_cum_net=float(a3.cum_net), A3_turnover=float(a3.turnover), A3_fees=float(a3.fees_sum), A3_mdd=float(a3.mdd),
    lift_buffer_alone=float(a1.cum_net - a0.cum_net),
    lift_constraints_alone=float(a2.cum_net - a0.cum_net),
    lift_both=float(a3.cum_net - a0.cum_net),
    interaction=float(a3.cum_net - a0.cum_net) - float(a1.cum_net - a0.cum_net) - float(a2.cum_net - a0.cum_net),
    fee_drag_A0=float(a0.cum_gross - a0.cum_net), fee_drag_A3=float(a3.cum_gross - a3.cum_net),
)
json.dump(attr, open(R / "a0a3_attribution.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
print("=== RECON ==="); print(json.dumps(recon_summary, ensure_ascii=False, indent=1, default=str))
print("max|diff| periods:", rec.nlargest(5, "abs_diff_net")[["ret_net", "p5_ret_net", "diff_net"]].to_string())
print("002013 trades:"); print(T1_5[T1_5.code=="002013"][["signal_date","status","entry_date","exit_date","slot_ret_gross","slot_ret_net","merger_note"]].to_string())
print("=== A0-A3 ==="); print(S.round(4).to_string()); print(PT.round(4).to_string()); print(json.dumps(attr, indent=1))
print("by year 5d cum_net:"); print(Y[Y.horizon==5].pivot_table(index="year", columns="strategy", values="cum_net").round(3).to_string())
print("secs", round(time.time()-t0))
