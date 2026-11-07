### 旧版(INVALID_BEFORE_FIX，仅作对照)

| 版本 | 策略 | 年化 | 夏普 | 最大回撤 | RankIC | 净超额均值/期 | 常规t | HAC4 t | Bootstrap 95%区间(净超额/期) | 换手(替换比例) | 买单成交率 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OLD_baseline (INVALID_BEFORE_FIX) | Ridge | 1.6% | 0.20 | -43.5% | 0.0723 | -0.086% | -0.87 | -0.72 | [-0.32%, 0.15%] | 0.62 | - |
| OLD_baseline (INVALID_BEFORE_FIX) | HGB | 13.9% | 0.57 | -36.6% | 0.0694 | 0.164% | 1.22 | 1.18 | [-0.09%, 0.45%] | 0.64 | - |
| OLD_baseline (INVALID_BEFORE_FIX) | LGBM | 8.8% | 0.42 | -36.7% | 0.0697 | 0.075% | 0.51 | 0.52 | [-0.20%, 0.38%] | 0.64 | - |
| OLD_baseline (INVALID_BEFORE_FIX) | Momentum20 | -42.8% | -1.32 | -90.1% | -0.0546 | -1.159% | -4.97 | -4.39 | [-1.68%, -0.65%] | 0.52 | - |
| OLD_baseline (INVALID_BEFORE_FIX) | Reversal5 | -31.6% | -0.80 | -80.4% | 0.0254 | -0.801% | -3.47 | -3.09 | [-1.30%, -0.28%] | 0.92 | - |
| OLD_baseline (INVALID_BEFORE_FIX) | EqualWeight | 6.3% | 0.36 | -35.3% | - | -% | - | - | - | - | - |

### 修复版(30bps往返成本，同制度等权基准)

| 版本 | 策略 | 年化 | 夏普 | 最大回撤 | RankIC | 净超额均值/期 | 常规t | HAC4 t | Bootstrap 95%区间(净超额/期) | 换手(替换比例) | 买单成交率 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FIXED (raw exec, factor-adjusted returns, as-of-t pool) | Ridge | -6.3% | -0.17 | -48.3% | 0.0771 | -0.208% | -1.67 | -1.38 | [-0.53%, 0.08%] | 0.66 | 0.990 |
| FIXED (raw exec, factor-adjusted returns, as-of-t pool) | HGB | 7.9% | 0.40 | -35.0% | 0.0746 | 0.100% | 0.72 | 0.67 | [-0.18%, 0.40%] | 0.67 | 0.982 |
| FIXED (raw exec, factor-adjusted returns, as-of-t pool) | LGBM | 7.2% | 0.38 | -35.0% | 0.0745 | 0.091% | 0.64 | 0.62 | [-0.20%, 0.38%] | 0.67 | 0.979 |
| FIXED (raw exec, factor-adjusted returns, as-of-t pool) | Momentum20 | -44.7% | -1.54 | -91.5% | -0.0554 | -1.184% | -5.64 | -4.98 | [-1.67%, -0.74%] | 0.53 | 0.905 |
| FIXED (raw exec, factor-adjusted returns, as-of-t pool) | Reversal5 | -27.0% | -1.29 | -73.7% | 0.0265 | -0.705% | -4.18 | -3.77 | [-1.10%, -0.35%] | 0.92 | 0.977 |
| FIXED (raw exec, factor-adjusted returns, as-of-t pool) | EqualWeight | 3.2% | 0.25 | -35.7% | - | -% | - | - | - | 0.02 | 0.957 |

