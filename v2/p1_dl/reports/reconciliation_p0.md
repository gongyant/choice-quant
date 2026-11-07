# 与 P0 结果对账(results\fixed_sina_adj)

walk-forward 重放使用 p0lib.py + baseline.make_models(同参数/seed 42/每4期再训练/3年窗口/purge 12日/首个测试日 2022-07-01),并记录每日×每股分数。下表为重放结果与 P0 `period_returns.csv`、`sets.pkl`、`orders_primary_cost30bps.csv` 的差异(207 期)。

| 检查项 | 模型 | 最大绝对差 | 备注 |
|---|---|---:|---|
| signal dates identical | - | 0.00e+00 | 207 periods 2022-07-01..2026-09-28 |
| n_pool | - | 0.00e+00 |  |
| net period return c30 (simulate rerun vs period_returns.csv) | Ridge | 9.71e-17 |  |
| Top50 set identical to sets.pkl (periods differing) | Ridge | 0.00e+00 | 207/207 identical |
| RankIC per period | Ridge | 9.89e-17 |  |
| IC per period | Ridge | 9.71e-17 |  |
| turnover_value (30bps) | Ridge | 5.55e-17 |  |
| cash_frac_post (30bps) | Ridge | 9.97e-17 |  |
| buy_filled (30bps) | Ridge | 0.00e+00 |  |
| n_hold (30bps) | Ridge | 0.00e+00 |  |
| order log identical to orders_primary_cost30bps.csv | Ridge | 0.00e+00 | rows 15257 vs 15257 |
| net period return c30 (simulate rerun vs period_returns.csv) | HGB | 9.89e-17 |  |
| Top50 set identical to sets.pkl (periods differing) | HGB | 0.00e+00 | 207/207 identical |
| RankIC per period | HGB | 9.71e-17 |  |
| IC per period | HGB | 9.71e-17 |  |
| turnover_value (30bps) | HGB | 5.55e-17 |  |
| cash_frac_post (30bps) | HGB | 9.93e-17 |  |
| buy_filled (30bps) | HGB | 0.00e+00 |  |
| n_hold (30bps) | HGB | 0.00e+00 |  |
| order log identical to orders_primary_cost30bps.csv | HGB | 0.00e+00 | rows 13735 vs 13735 |
| net period return c30 (simulate rerun vs period_returns.csv) | LGBM | 9.89e-17 |  |
| Top50 set identical to sets.pkl (periods differing) | LGBM | 0.00e+00 | 207/207 identical |
| RankIC per period | LGBM | 9.71e-17 |  |
| IC per period | LGBM | 9.71e-17 |  |
| turnover_value (30bps) | LGBM | 5.55e-17 |  |
| cash_frac_post (30bps) | LGBM | 9.89e-17 |  |
| buy_filled (30bps) | LGBM | 0.00e+00 |  |
| n_hold (30bps) | LGBM | 0.00e+00 |  |
| order log identical to orders_primary_cost30bps.csv | LGBM | 0.00e+00 | rows 14045 vs 14045 |
| net period return c30 | EqualWeight | 9.97e-17 |  |

结论:所有差异 ≤ 1.0e-16(浮点舍入),Top50 集合 207/207 期完全一致,订单日志逐行一致。