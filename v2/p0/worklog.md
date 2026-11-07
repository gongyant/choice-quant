# P0 worklog (D:\choice_quant_starter\v2\p0)
Times are Asia/Singapore (UTC+8), PC clock.

## Stage 1 - reproduce old numbers (done)
- Old results/baseline/walk_forward_returns.csv (207 periods): net excess = X_net - EqualWeight_net (EW has zero cost in old code line 105).
  HGB mean 0.001639/period t=1.22; LGBM 0.000753 t=0.508; Ridge -0.000859 t=-0.875; Momentum20 -0.011594 t=-4.98; Reversal5 -0.008006 t=-3.47. = GPT numbers (match).
- Old reported HGB "excess_tstat 2.65" is gross (code line 116: gross - universe_ew_gross). Confirmed from backtest_metrics.json.
- old_replica.py re-runs the unchanged old logic into p0/results/old_replica: max |diff| vs old CSV = 1e-16 (bit-for-bit reproducible).
- quality_per_stock.csv: 255 stocks with extreme returns, 1227 extreme records (sum of extreme_returns) - different units.

## Stage 2 - root cause of the 3 invalid-price stocks (done; root_cause_3stocks.py)
- Tencent raw (adjust="") == Sina raw (adjust="") exactly (max |close diff| = 0.0) for all three.
- Tencent qfq is an ADDITIVE (cash-equivalent) back-adjustment: price_qfq = raw - cumulative(per-share ex-rights amount of later ex-dates). Step of the offset at each ex-date == implied cash amount
  (e.g. 600188 2023-07-17: 14.16/share incl. the transfer-share effect). Cumulative later adjustments (11.32 for 600188, 7.50 for 600066, 8.19 for 000408 at 2021-01-04) exceed the 2021 raw prices (10.19/16.40/7.91)
  -> qfq prices <= 0 (52/1/19 days) -> pct_change explodes (542.86%, 141.67%, inf) and 123/109/85 days with |ret|>11%.
- Sina qfq-factor table is MULTIPLICATIVE (qfq = raw / factor_as_of_date). raw/factor reproduces Sina's qfq to <=0.005 (cent rounding), no non-positive prices, max |daily ret| 10.02%/10.04%/10.02%.
- Fix: execution & limit rules on RAW price; returns/features on raw/factor adjusted close (dividends reinvested at ex-date); ex-rights reference price for limit calc = prev raw close * F_t/F_prev.

## Stage 3 - unadjusted downloads (running in background)
- collect_unadj.py: Sina raw+qfq-factor primary; Sina raw fails (JSONDecodeError) for ~27% of SH/SZ codes -> Tencent raw fallback (source recorded in file). Tencent raw cross-check phase B.

## [17:38] Interrupted-resume check
- Phase A (Sina raw/Tencent fallback + Sina qfq-factor) cached for all 1498 codes (raw_unadj_parquet=1498, adj_factors=1498). A/A2 processes finishing; phase B (Tencent raw cross-check) still running (~474 of ~1400 SH/SZ files), not blocking.
- Unit tests (leakage_unit_tests): 45 passed incl. 4 negative controls (run on partial data; to rerun at end).
- Old numbers reproduced independently (HGB net t=1.22, HAC4 1.178, bootstrap [-0.000866,0.004519] = GPT).

## [17:38] Stage 4/6: launched baseline_fixed.py for fixed_sina_adj + E1 variants (tx_qfq_full / rowfilter / drop3) in background; logs p0\logs\run_*.log. 689009 factor derived from dividend announcements (open-vs-ref within 0.7%).

## [17:47] Stage 6: all 4 walk-forward runs done (207 periods each, ~440 s). Running compare_stats.py.

## [17:48] Stage 6 results: fixed HGB ann 7.85%, net excess/period +0.10% (t=0.72, HAC4 0.67, boot95 [-0.18%,0.40%]); EW(net,same regime) ann 3.25%. delisting_stress.py shows conclusion robust to stuck-position writedown 0/50/100%.

## [17:49] Stage 5: pytest 45 passed (incl. 4 NEGATIVE controls that must fail on look-ahead; real-data no-nonpositive test over all 1498 stocks) -> leakage_unit_tests\pytest_result.txt. quality_event_log.csv: 1272 events / 266 stocks (1158 additive-qfq artifacts corrected, 104 resumption-after-suspension, 10 unexplained, mostly 1-tick ref-price rounding).

## [18:10] Stage 3 done: Sina raw (1274 codes) + Tencent-raw fallback (206+ codes), Sina qfq-factor 1498/1498 (689009 derived from announcements); Tencent raw cross-check 1413/1413 SH/SZ. Re-running p0_data_audit.py (event log, coverage, unit, cross-source, corporate actions).

## [18:23] Stage 3/7: data audit final. Tencent raw vs Sina raw: 1203 independent stocks, close equal on 100% of common days (max abs diff 0.0), amount rel diff median 3e-7; Tencent raw volume unit is lots for 89 / shares for 1114 stocks (per-stock, resolved by amount/(raw close*volume)); 0 non-positive raw/adjusted prices; unit unverified only 920196 (1 bar) and 600317 (9 bars). Next: tables, manifest, report.

## [18:25] Stage 7 done: p0_audit_report.md, audit_manifest.json, trade_log.csv written. P0 judged PASS for engineering fixes (with blockers listed in report section 9).
