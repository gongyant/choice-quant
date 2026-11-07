# P2-A 冻结预测文件核验 (a_verify)

- 结论:**全部通过**
- 文件:`D:\choice_quant_starter\v2\p1_dl\historical_predictions_all_models.parquet`,1,860,516 行,sha256 `e411b80a0296a247d5da1259e63aefb157c3803f802cbe8abc94860a18634804`

## 检查项

| 检查 | 结果 | 细节 |
|---|---|---|
| 总行数=1,860,516 | 通过 | 1860516 |
| 键(signal_date,code,model_id)唯一 | 通过 |  |
| 键(signal_date,code,model)唯一 | 通过 |  |
| 模型集合=6 | 通过 | ['GRU', 'HGB', 'LGBM', 'MLP', 'Ridge', 'TF'] |
| 每模型 310,086 行 | 通过 | {'GRU': 310086, 'HGB': 310086, 'LGBM': 310086, 'MLP': 310086, 'Ridge': 310086, 'TF': 310086} |
| 代码均为6位字符串(前导零保留) | 通过 |  |
| 以0开头代码存在且未丢零 | 通过 | 430 |
| 六模型 signal_date 集合相同且=207 | 通过 | 207 |
| signal_date 都在 P0 主日历内 | 通过 |  |
| 相邻信号相隔 5 个交易日(5 日标签不重叠) | 通过 | [np.int64(5)] |
| 六模型 eligible 行数一致 | 通过 | {'GRU': 238781, 'HGB': 238781, 'LGBM': 238781, 'MLP': 238781, 'Ridge': 238781, 'TF': 238781} |
| eligible 行 score 无缺失(全部六模型) | 通过 | {'GRU': 0, 'HGB': 0, 'LGBM': 0, 'MLP': 0, 'Ridge': 0, 'TF': 0} |
| 每个 model_id 内训练元数据恒定 | 通过 |  |
| 每模型 52 个 model_id(每块 4 期再训练) | 通过 | {'GRU': 52, 'HGB': 52, 'LGBM': 52, 'MLP': 52, 'Ridge': 52, 'TF': 52} |
| train_label_end_date(训练标签到期) < 该块首个 signal_date(全部模型) | 通过 | min gap days=8 |
| train_end_date <= train_label_end_date(全部) | 通过 |  |
| DL: train_end < val_end <= label_end(max_label_expiry) | 通过 |  |
| DL: trained_asof == 该块首个 signal_date | 通过 | 0 |
| 树/线性: 无 trained_asof/val_end 字段(表中为空,A2 训练无验证集) | 通过 |  |
| DL 训练窗口结束日整体早于同块 HGB(留出验证期) | 通过 |  |
| DL val_end == 同块 HGB train_end | 通过 |  |
| HGB: realized_ret_5d / rel 与独立重算(t+1收盘→t+6收盘)一致 | 通过 | {'max_abs_diff_ret': 0.0, 'max_abs_diff_rel': 0.0, 'nan_pattern_mismatch': 0, 'n': 238781} |
| LGBM: realized_ret_5d / rel 与独立重算(t+1收盘→t+6收盘)一致 | 通过 | {'max_abs_diff_ret': 0.0, 'max_abs_diff_rel': 0.0, 'nan_pattern_mismatch': 0, 'n': 238781} |
| Ridge: realized_ret_5d / rel 与独立重算(t+1收盘→t+6收盘)一致 | 通过 | {'max_abs_diff_ret': 0.0, 'max_abs_diff_rel': 0.0, 'nan_pattern_mismatch': 0, 'n': 238781} |
| MLP: realized_ret_5d / rel 与独立重算(t+1收盘→t+6收盘)一致 | 通过 | {'max_abs_diff_ret': 0.0, 'max_abs_diff_rel': 0.0, 'nan_pattern_mismatch': 0, 'n': 238781} |
| GRU: realized_ret_5d / rel 与独立重算(t+1收盘→t+6收盘)一致 | 通过 | {'max_abs_diff_ret': 0.0, 'max_abs_diff_rel': 0.0, 'nan_pattern_mismatch': 0, 'n': 238781} |
| TF: realized_ret_5d / rel 与独立重算(t+1收盘→t+6收盘)一致 | 通过 | {'max_abs_diff_ret': 0.0, 'max_abs_diff_rel': 0.0, 'nan_pattern_mismatch': 0, 'n': 238781} |
| order_date=cal[i+1],exit_date=cal[i+6] | 通过 |  |
| MLP: OOS parquet 行数 310,086 / 键唯一 / 代码6位 / 与总表分数一致 | 通过 | {'path': 'D:\\choice_quant_starter\\v2\\p1_dl\\scores\\dl_mlp_oos_predictions.parquet', 'bytes': 12578158, 'rows': 310086, 'key_unique': True, 'n_dates': 207, ' |
| GRU: OOS parquet 行数 310,086 / 键唯一 / 代码6位 / 与总表分数一致 | 通过 | {'path': 'D:\\choice_quant_starter\\v2\\p1_dl\\scores\\dl_gru_oos_predictions.parquet', 'bytes': 12542386, 'rows': 310086, 'key_unique': True, 'n_dates': 207, ' |
| TF: OOS parquet 行数 310,086 / 键唯一 / 代码6位 / 与总表分数一致 | 通过 | {'path': 'D:\\choice_quant_starter\\v2\\p1_dl\\scores\\dl_tf_oos_predictions.parquet', 'bytes': 8997527, 'rows': 310086, 'key_unique': True, 'n_dates': 207, 'n_ |
| GRU/TF 与 HGB 的 OOS 覆盖(日期×代码)在冻结文件中完全相同 → 共同样本=全部 eligible 行 | 通过 | {'eligible_rows': 238781, 'common_all_six_scores': 238781, 'common_with_new_def_return': 238658, 'per_model_scores_eligible': {'HGB': 238781, 'LGBM': 238781, 'R |
| forward_registry manifest sha256 全部匹配(未被改动) | 通过 | {'n': 22, 'bad': []} |

## 标签定义

realized_ret_5d = 复权收盘(t+1) -> 复权收盘(t+6) 的收益(P0 主日历行号 i+1 → i+6;复权用 Sina 前复权因子,停牌日用最近已知价前向填充,label_uses_stale_price 标记);realized_rel_ret_5d = 减去当日 eligible 池均值。无论 t+1 是否涨停封板/停牌,都按此口径计入(未做可成交过滤)。

## 训练元数据说明

HGB/LGBM/Ridge: train_end_date=该块最后训练信号日(label 到期 +8 日历日左右 < 首个信号);DL: train_end 早于树模型约 3 个月(留出 val 块用于早停),val_end 与树模型 train_end 同日,train_label_end=max_label_expiry=val_end+6 交易日

## 覆盖说明

冻结总表里六模型覆盖相同(207 个信号日 × 1498 码,eligible 238,781 行)。差异不在覆盖而在训练窗口:DL 的 train_end 比 HGB 早约 3 个月(早停验证块),且 DL 为 3 seeds(Transformer 仅 seed 42)取均值;共同样本比较仍按 (signal_date,code) 交集进行。

## 记录缺口(非不一致)

- n_train_rows 仅 HGB/LGBM/Ridge 记录;MLP/GRU/TF 在总表中为空(DL 训练行数未写入冻结表,属记录缺失,不影响分数)

## 不一致记录

- 无