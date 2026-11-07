"""P0-C / P0-Q data audit on the UNADJUSTED data: quality_event_log, coverage classification, volume-unit check, cross-source check, corporate-action table."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import p0lib as L
R = L.P0 / "results"; R.mkdir(exist_ok=True)
X = L.build_data(); L.limit_flags(X); cal = X.cal; T, N = X.C.shape; u = X.u
old = pd.read_parquet(L.V2 / "data" / "panel_1500.parquet", columns=["code", "date", "open", "high", "low", "close"])
rate = {"创业板": .20, "科创板": .20, "北交所": .30}
def tx_raw(code):
    f = L.TXD / f"{code}.parquet"
    if not f.exists(): return None
    d = pd.read_parquet(f); d = d.rename(columns={"volume_raw": "volume"}) if "volume_raw" in d.columns else d
    return d.set_index("date")
# ------------------------------------------------------------------ 1. quality_event_log.csv
rows = []; Cf = L.ffill_mat(X.C); oldg = {c: g.sort_values("date").reset_index(drop=True) for c, g in old.groupby("code")}
for j, c in enumerate(X.codes):
    g = oldg.get(c)
    if g is None or len(g) < 2: continue
    lim = rate.get(u.board.iloc[j], .10); ret_old = g.close.pct_change().replace([np.inf, -np.inf], np.nan)
    inf_mask = np.isinf(g.close.pct_change()); ld = u.list_date.iloc[j]
    ipo = np.arange(len(g)) < 5 if (pd.notna(ld) and ld >= L.WIN_START) else np.zeros(len(g), bool)
    ext_old = ((g.close.pct_change().abs() > lim + 0.02) & ~ipo).values | inf_mask.values
    nonpos = (g[["open", "high", "low", "close"]] <= 0).any(axis=1).values
    # new scan on corrected series: raw close vs ex-rights reference price beyond the exchange limit (and not in a no-limit window)
    ref = X.ref[:, j]; Cj = X.C[:, j]; up = L.round_cent(ref * (1 + lim)); dn = L.round_cent(ref * (1 - lim))
    new_bad = X.traded[:, j] & np.isfinite(ref) & ((Cj > up + 0.0051) | (Cj < dn - 0.0051)) & ~X.nolimit_ipo[:, j]
    oldmap = {d: k for k, d in enumerate(g.date)}
    cand = set(g.date[ext_old | nonpos]) | set(cal[new_bad])
    for d in sorted(cand):
        t = cal.get_loc(d) if d in cal else None
        if t is None: continue
        k = oldmap.get(d); in_old = bool(k is not None and (ext_old[k] or nonpos[k])); in_new = bool(new_bad[t])
        prevC = Cf[t - 1, j] if t > 0 else np.nan; r_ref = Cj[t] / ref[t] - 1 if np.isfinite(ref[t]) and np.isfinite(Cj[t]) else np.nan
        prev_rows = np.where(X.has_row[:t, j])[0]; gap = int(t - prev_rows[-1]) if len(prev_rows) else -1
        exr = bool(np.isfinite(ref[t]) and abs(ref[t] / prevC - 1) > 1e-9)
        f0 = tx_raw(c); txc = float(f0.close.get(d, np.nan)) if f0 is not None else np.nan
        within = bool(np.isfinite(Cj[t]) and np.isfinite(ref[t]) and (Cj[t] <= L.round_cent(ref[t] * (1 + lim)) + 0.0051) and (Cj[t] >= L.round_cent(ref[t] * (1 - lim)) - 0.0051))
        if nonpos[k if k is not None else 0] and k is not None and nonpos[k]: cause = "qfq_nonpositive_price(additive_qfq_artifact)"
        elif X.nolimit_ipo[t, j]: cause = "no_limit_window_new_listing(genuine)"
        elif within: cause = "additive_qfq_artifact" if in_old else "within_limit"
        elif gap > 5: cause = "resumption_after_long_suspension(possible_no_limit)"
        else: cause = "unexplained_after_correction(manual_check)"
        action = {"additive_qfq_artifact": "old qfq return discarded; raw/factor-adjusted return used", "qfq_nonpositive_price(additive_qfq_artifact)": "old qfq price discarded; raw/factor-adjusted series used (all >0)",
                  "no_limit_window_new_listing(genuine)": "kept (genuine move); no limit-lock assumed", "resumption_after_long_suspension(possible_no_limit)": "kept; flagged; label winsorised 1/99",
                  "unexplained_after_correction(manual_check)": "kept; flagged for manual check; label winsorised 1/99; BLOCKER-LIST", "within_limit": "none"}[cause]
        rows.append(dict(code=c, name=u.name.iloc[j], board=u.board.iloc[j], event_date=str(d.date()), found_by=("old_qfq+corrected_scan" if in_old and in_new else "old_qfq_flag" if in_old else "corrected_scan_only"),
                         old_qfq_open=g.open[k] if k is not None else np.nan, old_qfq_high=g.high[k] if k is not None else np.nan, old_qfq_low=g.low[k] if k is not None else np.nan, old_qfq_close=g.close[k] if k is not None else np.nan,
                         old_qfq_ret=float(ret_old[k]) if k is not None and np.isfinite(ret_old[k]) else (np.inf if k is not None and inf_mask[k] else np.nan),
                         raw_open=X.O[t, j], raw_high=X.Hh[t, j], raw_low=X.L[t, j], raw_close=Cj[t], raw_prev_close=prevC, raw_volume_shares=X.V[t, j], raw_amount=X.AMT[t, j],
                         unit_mult_used=X.meta.unit_mult_used.iloc[j], source_primary=X.meta.source.iloc[j], tencent_raw_close=txc, tencent_vs_primary_close_diff=(txc - Cj[t]) if np.isfinite(txc) else np.nan,
                         gap_trading_days_since_prev_bar=gap, ex_rights_today=exr, factor_prev=X.F[t - 1, j], factor_today=X.F[t, j], ex_ref_price=ref[t],
                         limit_rate=lim, limit_up_price=float(L.round_cent(ref[t] * (1 + lim))) if np.isfinite(ref[t]) else np.nan, limit_down_price=float(L.round_cent(ref[t] * (1 - lim))) if np.isfinite(ref[t]) else np.nan,
                         limit_should_apply=bool(not X.nolimit_ipo[t, j]), closed_locked_limit_up=bool(X.lim_up[t, j]), closed_locked_limit_down=bool(X.lim_dn[t, j]),
                         raw_ret_vs_prev_close=Cj[t] / prevC - 1 if np.isfinite(prevC) else np.nan, corrected_ret_vs_ex_ref=r_ref,
                         ret_error_old_minus_corrected=(float(ret_old[k]) - r_ref) if (k is not None and np.isfinite(ret_old[k]) and np.isfinite(r_ref)) else np.nan, cause=cause, action=action))
ev = pd.DataFrame(rows); ev.to_csv(L.P0 / "quality_event_log.csv", index=False, encoding="utf-8-sig")
# ------------------------------------------------------------------ 2. coverage classification (stale end / coverage<95%)
cl = []
q_old = pd.read_csv(L.V2 / "reports" / "quality_per_stock.csv", dtype={"code": str}).set_index("code")
for j, c in enumerate(X.codes):
    hr = X.has_row[:, j]; first_t = int(np.argmax(hr)); last_t = int(T - 1 - np.argmax(hr[::-1])); ld, dd = u.list_date.iloc[j], u.delist_date.iloc[j]
    start = max(L.WIN_START, ld) if pd.notna(ld) else L.WIN_START; end = min(L.LAST_TD, dd) if pd.notna(dd) else L.LAST_TD
    exp = cal[(cal >= start) & (cal <= end)]; have = set(cal[hr]); miss = [d for d in exp if d not in have]; cov = 1 - len(miss) / max(1, len(exp))
    stale = cal[last_t] < end - pd.Timedelta(days=5); late = (cal[first_t] - exp.min()).days if len(exp) else 0
    f0 = tx_raw(c); cov_tx = np.nan
    if f0 is not None and len(exp):
        txs = set(f0.index); cov_tx = 1 - len([d for d in exp if d not in txs]) / len(exp)
    cause = "ok"
    if stale or cov < 0.95:
        gapd = (dd - cal[last_t]).days if pd.notna(dd) else None
        if pd.notna(dd) and stale and gapd > 70: cause = "delisted_stock: data ends >70d before delist date (check: merger/halt/source gap)"
        elif pd.notna(dd) and stale and gapd > 10: cause = "delisted_stock: last bar 10-70d before termination date (consistent with delisting-arrangement period / halt; normal)"
        elif pd.notna(dd) and stale: cause = "delisted_stock: data ends at delisting (normal)"
        elif stale: cause = "listed_status_but_data_ends_early (long suspension or source stop; check)"
        elif late > 5 and pd.notna(ld) and ld >= L.WIN_START: cause = "new_listing_inside_window (normal)"
        elif late > 5: cause = "late_start_vs_calendar (source starts late)"
        else: cause = "internal_gaps_suspensions"
        if np.isfinite(cov_tx) and abs(cov_tx - cov) > 0.02: cause += f" | sources disagree (tencent_cov={cov_tx:.3f})"
    cl.append(dict(code=c, name=u.name.iloc[j], status=u.status.iloc[j], list_date=ld, delist_date=dd, first_bar=str(cal[first_t].date()), last_bar=str(cal[last_t].date()), expected_days=len(exp), missing_days=len(miss),
                   coverage=round(cov, 4), coverage_tencent_raw=cov_tx, stale_end=bool(stale), old_flag_stale=bool(q_old.stale_end.get(c, False)) if c in q_old.index else None, old_coverage=q_old.coverage.get(c, np.nan), cause=cause))
cl = pd.DataFrame(cl); cl.to_csv(L.P0 / "coverage_classification.csv", index=False, encoding="utf-8-sig")
# ------------------------------------------------------------------ 3. volume-unit check (raw prices) + cross-source
vu, cs = [], []
for j, c in enumerate(X.codes):
    m = X.meta.iloc[j]; f0 = tx_raw(c); d = L.load_raw(c); row = dict(code=c, source=m.source, amount_over_close_volume_median=m.amount_ratio_median, inferred_multiplier=m.unit_mult_inferred, used_multiplier=m.unit_mult_used, vendor_default=m.unit_default_by_source, unverified=m.unit_unverified)
    if f0 is not None:
        s = d.set_index("date"); com = s.index.intersection(f0.index); com = [x for x in com if s.volume[x] > 0 and f0.volume[x] > 0]
        s = s.loc[com]; f0 = f0.loc[com]
        if len(com) >= 10:
            rv = np.median([s.volume[x] / f0.volume[x] for x in com]); row["volume_ratio_primary_over_tencent_raw_median"] = rv; row["tencent_volume_to_shares_multiplier"] = L.snap_pow10(1.0 / rv) if m.source.startswith("sina") else np.nan
            cs.append(dict(code=c, common_days=len(com), close_equal_frac=float(np.mean([abs(s.close[x] - f0.close[x]) < 0.0051 for x in com])), close_max_abs_diff=float(max(abs(s.close[x] - f0.close[x]) for x in com)),
                           amount_rel_diff_median=float(np.median([abs(s.amount[x] - f0.amount[x]) / max(1.0, s.amount[x]) for x in com])), primary_source=m.source))
    vu.append(row)
vu = pd.DataFrame(vu); vu.to_csv(L.P0 / "volume_unit_check.csv", index=False, encoding="utf-8-sig"); cs = pd.DataFrame(cs); cs.to_csv(L.P0 / "cross_source_check.csv", index=False, encoding="utf-8-sig")
# ------------------------------------------------------------------ 4. corporate action (adjustment factor) table
ca = []
for j, c in enumerate(X.codes):
    fac = L.load_factor(c)
    if fac is None or len(fac) == 0: continue
    fac = fac[fac.date >= pd.Timestamp("2020-12-01")].sort_values("date").reset_index(drop=True)
    allf = L.load_factor(c).sort_values("date").reset_index(drop=True)
    for _, r in fac.iterrows():
        prev = allf[allf.date < r.date]; fprev = prev.qfq_factor.iloc[-1] if len(prev) else np.nan
        t = cal.searchsorted(r.date); pc = Cf[t - 1, j] if 0 < t < T else np.nan
        ca.append(dict(code=c, ex_date=str(r.date.date()), factor_before=fprev, factor_after=r.qfq_factor, ratio_after_over_before=r.qfq_factor / fprev if fprev == fprev else np.nan,
                       prev_raw_close=pc, implied_cash_equiv_per_share=(pc * (1 - r.qfq_factor / fprev)) if (fprev == fprev and np.isfinite(pc)) else np.nan, in_sample_window=bool(r.date >= L.WIN_START),
                       source=r.source, fetched_at=r.fetched_at))
ca = pd.DataFrame(ca); ca.to_csv(L.P0 / "corporate_actions.csv", index=False, encoding="utf-8-sig")
summ = dict(stocks=N, panel_bars=int(X.has_row.sum()), events_total=len(ev), events_by_cause=ev.cause.value_counts().to_dict(), events_by_found_by=ev.found_by.value_counts().to_dict(),
            stocks_with_events=int(ev.code.nunique()), stocks_old_flag_extreme=int((q_old.extreme_returns > 0).sum()), old_extreme_records=int(q_old.extreme_returns.sum()),
            coverage_classification=cl.cause.value_counts().to_dict(), stale_end_n=int(cl.stale_end.sum()), coverage_lt95_n=int((cl.coverage < 0.95).sum()),
            raw_nonpositive_traded_bars=int((X.traded & ((X.O <= 0) | (X.Hh <= 0) | (X.L <= 0) | (X.C <= 0))).sum()), adj_nonpositive=int((X.ADJ_unadj_based <= 0).sum()),
            sources=X.meta.source.value_counts().to_dict(), unit_unverified_codes=X.meta[X.meta.unit_unverified].code.tolist(), unit_mult_used_counts={str(k): int(v) for k, v in X.meta.unit_mult_used.value_counts().items()},
            factor_status=X.meta.factor_status.value_counts().to_dict(), cross_source=dict(stocks=len(cs), close_equal_frac_mean=float(cs.close_equal_frac.mean()) if len(cs) else None, close_equal_frac_min=float(cs.close_equal_frac.min()) if len(cs) else None),
            corporate_action_rows=len(ca), limit_up_bars=int(X.lim_up.sum()), limit_dn_bars=int(X.lim_dn.sum()))
json.dump(summ, open(L.P0 / "data_audit_summary.json", "w"), ensure_ascii=False, indent=1, default=str); print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
