# sector_summary.csv 说明
- 行业 = 巨潮(cninfo)行业分类**最新生效版本**,不是历史时点分类;已退市股票多为"未分类/cninfo无数据"。
- **同花顺(THS)行业/概念:无成分数据**(v2\data\boards 里只有 board_catalog.csv 和空的 board_member_failures.csv,没有成分股映射),因此本表不含 THS 分组,也没有编造。Choice 板块成员需用户手工导出(见 v2\docs\choice_export_checklist.md)。
- 分组 Rank IC:每个信号日,在组内 eligible 且标签已到期的股票上计算 score 与 realized_rel_ret_5d(相对同日全体 eligible 池等权)的 Spearman,组内股票<10 的日期不计;rank_ic_mean 为各日均值,tstat 为朴素 t(未做 HAC,期间重叠的5日标签,仅描述)。
- top50_net_ret_per_trade:strategy_trades.csv 中该组股票的已平仓逐笔净收益(含0.30%往返成本、分红再投资口径)。top50_gross_ret_5d_mean:该组 Top50 入选记录的 P0 标签收益均值(未计成本、未区分是否成交)。
- group_period_vol_ann / maxdd:每个信号日取该组 Top50 入选股的 realized_ret_5d 等权均值,得到时间序列(只含有入选的期),年化波动=std*sqrt(252/5),最大回撤按该序列连乘;**不是可交易组合净值**。
- 若 ic_reliability 为"不可靠",请勿解读。多组×多模型比较存在多重检验问题。
