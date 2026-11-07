"""Write audit_manifest.json (source files, script sha256, timestamps, seeds, trial counts) and the consolidated trade_log.csv."""
import sys, json, glob, datetime as dt, hashlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import p0lib as L
P0, V2 = L.P0, L.V2
def dir_hash(files):
    h = hashlib.sha256()
    for f in sorted(files): h.update(Path(f).name.encode()); h.update(L.sha256(f).encode())
    return h.hexdigest()
# trade log: planned vs actual
tl = pd.read_csv(P0 / "results" / "fixed_sina_adj" / "orders_primary_cost30bps.csv", dtype={"code": str})
tl["planned_ref_close"] = tl.fill_price; tl.loc[tl.status != "FILLED", "fill_price"] = np.nan
tl["visible_at_signal"] = "eligible_asof_t=True (bar at t, listed, 20d median amount>=2e7, 60d history); score rank by model at close t"
tl["cost_round_trip_assumed"] = 0.003
cols = ["strategy", "signal_date", "order_date", "code", "side", "planned_date", "planned_ref_close", "fill_date", "fill_price", "status", "reason", "sell_delay_days", "shares", "value", "cost", "cost_round_trip_assumed", "visible_at_signal"]
tl[cols].to_csv(P0 / "trade_log.csv", index=False, encoding="utf-8-sig")
scripts = ["p0lib.py", "baseline_fixed.py", "old_replica.py", "collect_unadj.py", "p0_data_audit.py", "compare_stats.py", "root_cause_3stocks.py", "dividend_crosscheck.py", "make_manifest.py", "build_factor_fallback.py", "delisting_stress.py", "make_tables.py", "collect_unadj.py", "leakage_unit_tests/synth.py", "leakage_unit_tests/test_no_lookahead.py", "leakage_unit_tests/test_execution.py"]
man = dict(created=dt.datetime.now().isoformat(timespec="seconds"), timezone="Asia/Singapore", python=sys.version.split()[0], as_of_data_end=str(L.LAST_TD.date()), test_period="signals 2022-07-01 .. 2026-09-28 (first_test 2022-07-01, step 5 trading days)",
           seeds=dict(model_seed=42, bootstrap_seed=20261018, bootstrap_block=5, bootstrap_draws=4000, sampling_seed_original=20261016),
           scripts={s: dict(sha256=L.sha256(P0 / s), bytes=(P0 / s).stat().st_size) for s in scripts if (P0 / s).exists()},
           old_scripts_untouched={s: L.sha256(V2 / s) for s in ["baseline.py", "quality_report.py", "sample_universe.py"]},
           source_files={"sample_1500.csv": L.sha256(V2 / "data/sample_1500.csv"), "panel_1500.parquet": L.sha256(V2 / "data/panel_1500.parquet"), "old walk_forward_returns.csv": L.sha256(V2 / "results/baseline/walk_forward_returns.csv"),
                         "old backtest_metrics.json": L.sha256(V2 / "results/baseline/backtest_metrics.json"), "quality_per_stock.csv": L.sha256(V2 / "reports/quality_per_stock.csv"), "trade_calendar.csv": L.sha256(L.D / "trade_calendar.csv")},
           data_dirs={"raw_parquet_old_qfq(Tencent/Sina)": dict(files=len(list((V2 / "data/raw_parquet").glob("*.parquet"))), tree_sha256=dir_hash(glob.glob(str(V2 / "data/raw_parquet/*.parquet")))),
                      "raw_unadj_parquet(Sina primary/Tencent fallback)": dict(files=len(list(L.RU.glob("*.parquet"))), tree_sha256=dir_hash(glob.glob(str(L.RU / "*.parquet")))),
                      "adj_factors(Sina qfq-factor)": dict(files=len(list(L.FA.glob("*.parquet"))), tree_sha256=dir_hash(glob.glob(str(L.FA / "*.parquet")))),
                      "raw_unadj_tx(Tencent raw cross-check)": dict(files=len(list(L.TXD.glob("*.parquet"))), tree_sha256=dir_hash(glob.glob(str(L.TXD / "*.parquet"))))},
           data_sources=["Sina ak.stock_zh_a_daily adjust='' / 'qfq-factor'", "Tencent ak.stock_zh_a_hist_tx adjust='' / 'qfq'", "Sina ak.stock_history_dividend_detail", "Sina ak.tool_trade_date_hist_sina (calendar)", "cninfo industry (unchanged old files)"],
           trials=dict(hyperparameter_search="none: one fixed parameter set per model (same as old baseline.py)", models=3, benchmark_strategies=2, equal_weight_benchmark=1, topk=50, data_versions_run=["fixed_sina_adj", "tx_qfq_full", "tx_qfq_rowfilter", "tx_qfq_drop3"] + ["old_replica"], cost_levels=[0.0, 0.001, 0.003, 0.006],
                       full_walk_forward_runs=5, simulated_portfolio_paths=5 * 6 * 4 + 6 * 4 * 0, multiple_testing_note="no model/parameter selection was done on test results; statistics are descriptive. 5 strategy x 4 data versions compared -> treat any single t as exploratory."),
           result_files=sorted(str(Path(f).relative_to(P0)) for f in glob.glob(str(P0 / "*.csv")) + glob.glob(str(P0 / "*.json")) + glob.glob(str(P0 / "*.md"))))
json.dump(man, open(P0 / "audit_manifest.json", "w"), ensure_ascii=False, indent=1); print("manifest ok", len(man["scripts"]), "scripts;", len(tl), "trade-log rows")

