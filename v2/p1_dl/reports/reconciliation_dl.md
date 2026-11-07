# DL 模型对账(重放 vs 轨道B results\{mlp,gru,tf}\period_returns.csv)

方法:用 OOS 预测 parquet(`scores\dl_*_oos_predictions.parquet`)里每期 Top50(按 rank_in_pool 排序),经 P0 同一个 `p0lib.simulate`(1000万元、整手、往返0.30%)重新模拟,与 B 的 `period_returns.csv`、`orders_*_cost30bps.csv` 逐期/逐行比较;IC/RankIC 用 parquet 分数与 P0 标签 y 重新计算。

| 检查项 | 模型 | 最大绝对差 | 备注 |
|---|---|---:|---|
| net period return c30 (rerun of simulate from OOS Top50 vs dl results period_returns.csv) | MLP | 9.89e-17 | 207 periods |
| turnover_value (30bps) | MLP | 5.55e-17 |  |
| cash_frac_post (30bps) | MLP | 9.89e-17 |  |
| buy_filled (30bps) | MLP | 0.00e+00 |  |
| n_hold (30bps) | MLP | 0.00e+00 |  |
| order log identical to B orders_*_cost30bps.csv | MLP | 0.00e+00 | rows 14361 vs 14361 |
| IC per period (recomputed from OOS scores vs period_returns) | MLP | 9.71e-17 |  |
| RankIC per period | MLP | 9.71e-17 |  |
| EqualWeight net ret c30 vs P0 file | MLP | 0.00e+00 |  |
| buy_status FILLED/REJECTED count vs B order log BUY rows | MLP | 0.00e+00 | 7194 vs 7194 |
| realized_ret_5d equals A2 (HGB rows) label | MLP | 0.00e+00 |  |
| net period return c30 (rerun of simulate from OOS Top50 vs dl results period_returns.csv) | GRU | 9.71e-17 | 207 periods |
| turnover_value (30bps) | GRU | 1.11e-16 |  |
| cash_frac_post (30bps) | GRU | 9.93e-17 |  |
| buy_filled (30bps) | GRU | 0.00e+00 |  |
| n_hold (30bps) | GRU | 0.00e+00 |  |
| order log identical to B orders_*_cost30bps.csv | GRU | 0.00e+00 | rows 15131 vs 15131 |
| IC per period (recomputed from OOS scores vs period_returns) | GRU | 9.80e-17 |  |
| RankIC per period | GRU | 9.78e-17 |  |
| EqualWeight net ret c30 vs P0 file | GRU | 0.00e+00 |  |
| buy_status FILLED/REJECTED count vs B order log BUY rows | GRU | 0.00e+00 | 7513 vs 7513 |
| realized_ret_5d equals A2 (HGB rows) label | GRU | 0.00e+00 |  |
| net period return c30 (rerun of simulate from OOS Top50 vs dl results period_returns.csv) | TF | 9.89e-17 | 207 periods |
| turnover_value (30bps) | TF | 1.11e-16 |  |
| cash_frac_post (30bps) | TF | 9.80e-17 |  |
| buy_filled (30bps) | TF | 0.00e+00 |  |
| n_hold (30bps) | TF | 0.00e+00 |  |
| order log identical to B orders_*_cost30bps.csv | TF | 0.00e+00 | rows 13950 vs 13950 |
| IC per period (recomputed from OOS scores vs period_returns) | TF | 9.47e-17 |  |
| RankIC per period | TF | 9.95e-17 |  |
| EqualWeight net ret c30 vs P0 file | TF | 0.00e+00 |  |
| buy_status FILLED/REJECTED count vs B order log BUY rows | TF | 0.00e+00 | 6920 vs 6920 |
| realized_ret_5d equals A2 (HGB rows) label | TF | 0.00e+00 |  |

结论:所有数值差异 ≤ 1.1e-16(0 或浮点舍入);订单日志逐行一致(=0)。DL 模型的逐期净收益可由 OOS 预测文件独立复现。