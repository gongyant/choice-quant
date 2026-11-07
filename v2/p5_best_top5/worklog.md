# P5 worklog
- 10-22 16:5x 批次A开始(只读 P4 缓存/预测,CPU)。B0=池等权,B1=低波动Top5,B2=低波动+同一约束/缓冲模块,B3=S4。
- 批次A完成,结果 results\a1_*/a2_*/a3_*,报告 reports\p5_batchA_report.md。批次B未开始。
- 10-22 17:0x 按 t17u 修正定义重跑 A1 -> a1v2_*(旧 a1_* 保留,报告改名 p5_batchA_report_v1_assumed_defs.md)。002013 按换股 1:0.6647 估值。
- 新浪 10-22 收盘快照 + 东财公告 -> p5_batchB.py:五股对比、B1/B3 10-22 名单登记(sha256)、forward_registry、buy_criterion_p5_v1(全部不通过)。
- p5_verify_forward.py(首跑 data_end 10-21, 0 行) / run_weekly_verify_p5.ps1 / dashboard 8505(AppTest 无异常) / reports\p5_final_report.md / bundle。
