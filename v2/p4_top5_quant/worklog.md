# P4 worklog

- 2026-10-21 23:20 SGT 批次1启动:审计 P0-P3 可复用接口;新建 p4lib.py(复用 p0lib 加载器,内存重定向到 P3 merged 数据至 2026-10-21)、p4_build.py(特征/标签缓存)。所有结果为探索性。

- 23:05 build 完成(50 特征;ST 代理按 2026-07-06 主板 ST 涨跌幅 5%->10% 规则切换处理)。
- 23:43 6 个 E1 walk-forward(lrank/xendcg/reg x h5/h20)+ p4_eval 完成,CPU,无 GPU。
- 23:45 E1reg h5/h20 最新排名登记 predictions_registry(sha 87bb087a… / 6b414dbd…),买入门槛未通过。pytest 7/7 通过。
- 23:50 reports\e0_e1_report.md 写成。结论:E1 未改善 Top5 费后表现。

## 批次2
- 10-21 23:56 selection_rules_p4_v1.json 登记(sha 4f3fdc80…),早于评估。
- 东方财富业绩报表+资产负债表在 box 上拉取(PC 访问 SSL 失败),10-22 09:3x 分块传输,sha c97bd57b… 校验一致,解压到 fin_raw。
- 09:40 p4_fin_build 完成(1289/1498 只有数据,覆盖 89%->99%,法定滞后使用 0%)。
- 09:42-09:53 E2reg/E2lrank x h5/h20 训练(CPU),p4_e2_eval,p4_latest_e2 S4 登记(h5 sha 1de07a3e…),pytest 11/11。
- reports\e2_report.md 写成。结论:无策略通过买入标准;S4 20 日净 +34.7% 高于 P3 v2,但低于低波动因子 +46.1%,且不显著。

## 批次3
- 10:15 启动 StockMixer 串行 GPU 训练(SME2 h5 -> h20 -> SME1 h5 -> h20),一次一个 GPU 作业。
- 10:22-11:26 SME2 h5/h20、SME1 h5/h20 串行 GPU 训练完成(每个 15-23 分钟);p4_sm_eval 完成;pytest 14/14 通过。
- 结论:StockMixer 无增量价值,SM-S4 不如 S4 和低波动因子;买入标准全部未通过;不升级 MASTER,不输出 SM 最新 Top5。报告 reports\stockmixer_report.md。

## 最终交付(10-22 11:4x)
- 无新训练。10-22 未收盘,数据仍截至 10-21,未更新。
- p4_final_prep.py:统一对比表 results\p4_comparison_all.csv、S4 经验校准 s4_calibration.csv、个股历史、净值图 figures\p4_equity_h5/h20.png。
- 低波动 Top5 登记 p4_pred_20261021_LOWVOL_top5.csv(sha 66d14be5…)。p4_verify_forward.py + run_weekly_verify_p4.ps1(追加到 p3_ai_report\verification\p4_forward_verify.csv)。
- 仪表盘 dashboard\p4_app.py,run_dashboard_p4.ps1,http://127.0.0.1:8504,健康检查 ok,AppTest 无异常。
- 报告 reports\p4_model_comparison_report.md、p4_top5_report.md。pytest 14/14。
- p4_review_bundle.zip:94 文件,3,133,219 字节,sha256 061852187cb2656cef6d05005754db28c1ec3652e46d6f2ee5c13f191969bdc2。
