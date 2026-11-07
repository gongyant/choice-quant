# research_screener_20261016.csv 列说明(研究排名,不是买入推荐;分数不是概率或收益;P2 结论:信号未被证明可在成本后变现)

生成:`c_screener.py`(.venv),只读取已冻结产物,不重训、不改 P0/P1/forward_registry。1498 行 = P0 世界内全部股票(P1 latest 1500 行中 2 只已退市且无行情的 *ST 股不在 P0 世界内);eligible_asof=True 的 1178 只为 2026-10-16 可选池。

| 列 | 含义 |
|---|---|
| code, name, exchange, board | 真实代码/现名/交易所/板块 |
| industry_l1/l2, industry_source | 巨潮行业分类**最新生效版本,非历史时点** |
| score_/rank_/pctile_{HGB,LGBM,Ridge,MLP,GRU,TF} | P1 2026-10-16 冻结分数、排名(1=最高)、池内分位(1=最高),与 P1 发布值一致;TF 为 seed42 单种子 |
| n_models_top50_of6, models_in_top50 | 六模型中排名进入 Top50 的模型数与名单 |
| pctile_mean_6, pctile_std_6 | 六模型分位均值 / 标准差(越小越一致) |
| n_models_top50_of5_p1, pctile_std_5_p1 | P1 原表的五模型版本(不含 Ridge),保留对照 |
| tf3_signal_date, tf3_seeds, tf3_pctile_mean/std, tf3_rank_of_mean, tf_s42_pctile_latest_oos | Transformer T0 seeds 42/43/44 在**最近一个 OOS 信号日(2026-09-28)**的池内分位均值/标准差;P2-C 未保存模型状态,禁止重训,故不是 10-16 分数 |
| r5, r20, r60, vol20, turn20, amihud20, amt20 (+ _pctile) | P2-A 风格暴露,截至 2026-10-16 收盘(amt20 = 20日成交额中位数,元) |
| base_REV5/LOWVOL/ILLIQ_pctile, base3_pctile_mean | 三因子基线 = 5日反转(-r5)、低波动(-vol20)、Amihud 非流动性 的池内分位及平均(P2-A 基线定义) |
| limit_up_locked_t, limit_down_locked_t, traded_t | 2026-10-16 收盘涨停/跌停封板、当日是否有成交(P0-F 口径) |
| st_name_current | 现名含 ST(现名非时点) |
| needs_special_access, star_board, bj_exchange | 科创板/北交所,需投资者准入 |
| low_liquidity_flag | amt20 < 5,000 万元(仪表盘可自定阈值) |
| a2w_{model}, a2w_mean6 | P2-B A2 规则研究权重:嵌套选出的最近一块参数(见 _meta.json)、C2 显式成本(成本输入取 10-16)、空仓起步、资本 1e7;**仅研究模拟**,未考虑 10-17 之后的成交可行性 |
| research_note | 免责声明 |