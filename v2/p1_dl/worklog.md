# P1-A1 worklog (Asia/Singapore, PC clock; written 2026-10-18 21:19)
Directory: D:\choice_quant_starter\v2\p1_dl\  (track B writes under p1_dl\dl\ and tests\test_dl_mlp.py; not touched here)

## Steps
1. verify_p0_inputs.py: sha256 of sample_1500.csv, trade_calendar.csv, p0lib.py, baseline_fixed.py, old baseline.py and tree-sha256 of raw_unadj_parquet (1498 files), adj_factors (1498), raw_unadj_tx (1413) ALL match p0/audit_manifest.json.
   Calendar 1400 days 2021-01-04..2026-10-16; 1498 raw files, 1,761,682 bars, 0 non-positive closes; missing: 600485 (*ST信威), 600677 (*ST航通) (delisted, Sina/Tencent returned empty tables; fetch_status.csv: 'ValueError: 接口返回空表' x2 each).
2. fetch_extra.py: 600060 (not in the 1500 sample) -> Sina raw + qfq-factor + independent Tencent raw: 1400 bars 2021-01-04..2026-10-16, 28 factor rows, close diff Sina vs Tencent = 0.0 on 1400 common days. Stored under data_extra\ only.
3. score_latest.py (frozen models, P0 hyper-parameters via baseline.make_models, seed 42): built P0 matrices (9 s), pool at 2026-10-16 = 1178; trained on d in [2023-10-17, 2026-09-22] (purge 12 trading days, label end <= 2026-10-08); 850,768 rows over 715 dates; train time 2.9 s (Ridge+HGB+LGBM).
   model_version = p1A1-2026-10-16-HGB_LGBM_Ridge-seed42-train2023-10-17_2026-09-22-p0lib63e45932
4. latest_all_stocks_20261016.csv: 1500 rows, unique codes, 6-char strings; eligible 1178; ineligible 322: {'已退市': 206, '流动性不足': 91, '历史不足': 15, '最新日无成交': 8, '无行情数据': 2}.
5. top_bottom.py -> reports/top_bottom_20261016.md/.csv (HGB & LightGBM Top20/Bottom20, overlaps).
6. cards.py -> reports/stock_cards_20261016.md + figures/card_*.png (600060 scored out-of-pool; 300487/000027 in-sample).
7. pytest: tests/test_latest_scoring.py 9 passed; P0 leakage_unit_tests re-run: 45 passed (outputs saved in tests/).

## Files
| file | bytes | lines |
|---|---:|---:|
| `latest_all_stocks_20261016.csv` | 905,413 | 1501 |
| `reports/top_bottom_20261016.md` | 12,127 | 115 |
| `reports/top_bottom_20261016.csv` | 12,939 | 81 |
| `reports/stock_cards_20261016.md` | 5,180 | 82 |
| `figures/card_600060.png` | 176,373 | - |
| `figures/card_300487.png` | 160,305 | - |
| `figures/card_000027.png` | 160,827 | - |
| `models/model_meta_2026-10-16.json` | 1,227 | 40 |
| `models/HGB_frozen_2026-10-16.joblib` | 285,960 | - |
| `models/LGBM_frozen_2026-10-16.joblib` | 324,569 | - |
| `models/Ridge_frozen_2026-10-16.joblib` | 641 | - |
| `scores/latest_rank_features_20261016.csv` | 270,348 | 1179 |
| `scores/stock_cards_data_20261016.json` | 2,827 | 112 |
| `p0_input_verification.json` | 1,462 | 64 |
| `tests/test_latest_scoring.py` | 4,980 | 63 |
| `tests/pytest_result_A1.txt` | 1,193 | 17 |
| `tests/pytest_result_p0_rerun.txt` | 106 | 3 |
| `p1lib.py` | 2,810 | 41 |
| `score_latest.py` | 9,103 | 94 |
| `top_bottom.py` | 3,339 | 30 |
| `cards.py` | 9,642 | 77 |
| `verify_p0_inputs.py` | 2,674 | 30 |
| `fetch_extra.py` | 2,308 | 25 |



# ===== A2 (batch 2) 2026-10-18 21:38 =====
1. pip install streamlit plotly kaleido into project .venv only (streamlit 1.66.0, plotly 7.1.0); numpy/pandas unchanged.
2. build_history.py: replayed P0 baseline_fixed walk-forward (same p0lib/make_models/seed 42/retrain every 4 periods/3y/purge 12) recording day x stock outputs; 207 signals 2022-07-01..2026-09-28, 52 retrain blocks, 156 model_ids; 930,258 rows; 135 s.
   Reconciliation vs P0 results/fixed_sina_adj: all max abs diffs <= 1e-16; Top50 sets 207/207 identical; order logs row-identical (reports/reconciliation_p0.md).
3. summaries.py -> per_stock_summary.csv (1500 rows), sector_summary.csv (63 rows), figures, notes. THS board members absent (only board_catalog.csv) -> not included.
4. stock_dashboard.py + run_dashboard.ps1; smoke: AppTest 13 steps PASS (all pages, 3 quick stocks, 000006/002829/920699/600485, search); headless server /_stcore/health = 200, root 200 (server stopped afterwards). run_dashboard.ps1 uses --server.headless true (otherwise streamlit blocks on first-run email prompt) and opens the browser itself.
5. pytest: tests/test_history_a2.py 10 passed + test_latest_scoring.py 9 passed = 19 passed; P0 leakage_unit_tests re-run 45 passed.

## Files (A2)
| file | bytes | lines |
|---|---:|---:|
| `historical_predictions.parquet` | 44,909,983 | 930,258 rows |
| `historical_predictions_sample.csv` | 4,172,897 | 13483 |
| `historical_predictions_summary.csv` | 131,630 | 622 |
| `per_stock_summary.csv` | 1,280,470 | 1501 |
| `sector_summary.csv` | 18,676 | 64 |
| `strategy_trades.csv` | 9,945,838 | 20705 |
| `reports/recent20_metrics.csv` | 10,096 | 61 |
| `reports/recent20_metrics.md` | 7,363 | 86 |
| `reports/recent20_summary.csv` | 689 | 4 |
| `reports/reconciliation_p0.csv` | 1,785 | 31 |
| `reports/reconciliation_p0.md` | 2,334 | 38 |
| `reports/sector_summary_notes.md` | 1,352 | 7 |
| `figures/sector_rankic_hgb.png` | 60,032 | - |
| `figures/rolling_rankic_models.png` | 177,714 | - |
| `stock_dashboard.py` | 21,170 | 177 |
| `run_dashboard.ps1` | 741 | 10 |
| `smoke_dashboard.py` | 1,733 | 25 |
| `build_history.py` | 15,632 | 142 |
| `summaries.py` | 11,869 | 91 |
| `tests/test_history_a2.py` | 8,447 | 83 |
| `tests/pytest_result_A2.txt` | 2,152 | 28 |
| `tests/pytest_result_p0_rerun_A2.txt` | 106 | 3 |

## A3 最终整合(2026-10-18 ~ 2026-10-19)
- [2026-10-19 08:51] 子步骤1-3 (补记,实际完成于 2026-10-18 夜间):`a3\a3_latest.py` 生成 `latest_all_stocks_20261016_v2.csv`(1500行/50列,eligible 与 A1 一致 1178,DL 分数/排名/分位 + model_agreement_pctile_std=五模型(HGB,LGBM,MLP,GRU,TF)百分位 std)和 `reports\top_bottom_20261016_v2.md/csv`、`multi_model_top50_20261016.csv`(≥3模型Top50 共28只);`a3\a3_history.py` 生成 `historical_predictions_all_models.parquet`(1,860,516行,键唯一)、`per_stock_summary_v2.csv`、`sector_summary_v2.csv`、`strategy_trades_v2.csv`、`historical_predictions_summary_v2.csv`、`reports\recent20_metrics_v2.*`、`reconciliation_dl.*`(与B period_returns/订单日志最大差≤1.1e-16)、模型对比/成本敏感/分年/配对汇总、`figures\cumulative_net_models.png`;`a3\a3_cards.py` 生成个股卡片 v2(600060 MLP 池外打分经检查点复现验证;GRU/TF 未打分);`stock_dashboard_v2.py`+`smoke_dashboard_v2.py`,`run_dashboard.ps1` 指向 v2(旧版 `run_dashboard_v1.ps1`),AppTest 13/13,无头启动 /_stcore/health 与首页 HTTP 200;`a3\a3_registry.py` 生成 `forward_registry_20261016.csv`(300行)+manifest(22个文件sha256,已冻结)。
- [2026-10-19 08:51] 中断恢复(2026-10-19 早晨):核验已有产物完整(manifest 22/22 sha256 一致;v2 表1500行;parquet 1,860,516行无重复键)。
- [2026-10-19 08:51] 子步骤4:`a3\a3_report.py` 重写并运行,生成 `p1_dl_report.md`(UTF-8 无BOM;含真实名称 Top20/Bottom20、卡片、行业、最近20期、六模型对比、数据质量、风险、前向登记、诚实结论)。
- [2026-10-19 08:51] 子步骤5:新增 `tests\test_final_a3.py`(16项)。全部 P1 测试 65 passed(A1 9 + A2 10 + B1 11 + B2 10 + B3 9 + A3 16),P0 leakage_unit_tests 45 passed;输出 `tests\pytest_result_final.txt`。
- [2026-10-19 08:53] 子步骤6:`a3\a3_bundle.py` 生成 `review_bundle.zip`(10,112,660 B,74个文件,解压后约38MB;sha256 90e975e4…1fc81)与 `README_bundle.md`(含文件清单/含义/sha256、排除项、复现步骤、局限、审计提问);zip 内逐文件 sha256 复核 73/73 一致;不含原始行情/面板/完整历史预测/模型文件;历史预测仅含最近12个信号日(2026-07-10..2026-09-28)合格行抽样 86,118 行。
- [2026-10-19 08:53] 子步骤7:仪表盘 v2 最终冒烟:AppTest 13/13 PASS;run_dashboard.ps1 -NoBrowser 无头启动 /_stcore/health 与首页均 HTTP 200,已停止服务。A3 全部完成。
- 2026-10-19 12:22 (Asia/Singapore) [P2-A 进度1] 新目录 p2_alpha_monetization\ 建立;p2lib.py(下一可执行日前向收益 exec_forward、HAC/块自助/偏秩相关)、a_panel.py(面板 310,086 行,eligible 238,781,与冻结表 eligible 0 不一致)、a_verify.py(a_verify.json/md:冻结总表 1,860,516 行/键唯一/6位代码/标签独立重算差=0/DL OOS 分数一致/forward_registry 22 个 sha256 全匹配;唯一记录:DL 的 n_train_rows 在总表为空)、a_eval.py(新旧口径、分组IC、偏IC、基线、十分位、子集、归因,耗时~60s)已跑通;a_diag.py 解释新口径提升来源;a_figs.py 9 张图(中文字体检查无缺字)。P0/P1 输出未改动。下一步:报告、tests\test_p2a.py、重跑 P0 45 + P1 65。
- 2026-10-19 12:28 (Asia/Singapore) [P2-A 完成] reports\p2a_alpha_anatomy_report.md(a_report.py 由 results\*.csv 自动生成)、figures 9 张图(非空检查通过)、tests\test_p2a.py 24 项通过;重跑 P0 leakage_unit_tests 45 项通过、P1 tests 65 项通过(结果存 p2_alpha_monetization\tests\pytest_{p2a,p0_rerun,p1_rerun}.txt),未改动 P0/P1 输出与 forward_registry(manifest 22 个 sha256 复核一致)。关键结论:新口径 RankIC 六模型 0.074–0.081(t 7.5–8.7),但不高于低波动/三因子基线;四项控制后偏 IC 约保留 37–48%(Ridge≈0);价差主要在空头端,5 因子 alpha(D10−D1)树/DL 每5日约 0.8–1.2%(t 4.3–5.5),多头端 0.2–0.33%/5日≈成本同量级;amt20≥1e8 子集多头端超额≈0/负;2026 走弱。2025–2026 仅探索性。
- 2026-10-19 12:43 (Asia/Singapore) [P2-B 进度1] 新增 b_sim.py(p0lib.simulate 状态机的扩展版 simulate_ext:默认开关关闭时与 p0lib.simulate 逐位一致;含决策回调、显式成本模型、参与率上限/最小ADV、A2 权重再平衡)、b_data.py、b_lib.py、b_run.py、b_protocol.py(results\b_protocol.json 在主跑前声明:A1 K_hold 网格 {65,80,100,150}、A2 网格 12 组、嵌套验证 burn-in 52 期+26 期块、净夏普准则、4 套情景 C1 flat/C2 显式成本/C3 现实约束/C4 容量1e8)。A0 对账(b_bench.py):六模型 simulate_ext 与 p0lib.simulate 最大差 0,与冻结 P0/轨道B period_returns(c30) 最大差 ~1e-16,Top50 集合与冻结 in_top50 完全一致;EW 基准与 P0 文件差 ~1e-16。主跑(24 个 模型×情景 任务,3 并发,单进程约 0.9GB 内存;第一次 12 并发 OOM 已清理重跑)进行中。

## [P2-B 进度2] 2026-10-19 13:15 (UTC+8)
- 24个作业(6模型×4场景:C1 flat/C2 显式/C3 现实约束/C4 容量1亿)全部完成;首次launcher因日志命名bug在无附加参数的作业上崩溃,已修复后仅补跑缺失的9个(结果文件幂等,未改动已完成结果)。
- b_agg.py 已跑通:b_performance/b_paired/b_turnover_vs_prediction/b_factor_alpha/b_nested_*/b_cost_stress 等CSV已生成;匹配换手控制组发现S(随机跳过)单种子二分标定系统性偏低10-16%,已加入向A0线性重加权的精确换手校准列(保留原始列,披露)。
- b_figs.py 8张图已生成并检查非空;tests\test_p2b.py 已分段通过(引擎/成本/A1不变式/A2/嵌套),待全量运行 + P0/P1/P2-A 回归。

## [P2-B 完成] 2026-10-19 13:25 (UTC+8)
- 中断后检查:无残留后台进程;24个作业结果、8张图、报告均已落盘,未重做已验证部分。
- reports\p2b_cost_aware_report.md(b_report.py 自动由CSV生成);tests\test_p2b.py 24 项通过;重跑 P0 45 + P1 65 + P2-A 24 全部通过(输出 tests\pytest_{p2b,p0_rerun_b,p1_rerun_b,p2a_rerun_b}.txt);forward_registry manifest 22 个 sha256 与 test_p2b 校验一致,P0/P1 输出未改动。
- 关键结论:C2 显式成本 6模型均值 A0 毛15.2%/净-3.2%,A1 净1.2%,A2 毛13.7%/净9.7%;改善主要来自成本/换手下降(S对照选股效应均值0.021%/期,|t|>1.96 为0/12);因子调整净alpha无正显著,A2多数为负;C3现实约束下毛收益缩至3-7%,三种策略相对等权超额均为负;A1选参在134/144个块选到网格上界K150(披露)。2025-2026 为探索性,非投资建议。

## [P2-C 进度1] 2026-10-19 13:53 (UTC+8)
- 新增(均在 p2_alpha_monetization\,P0/P1 文件与 forward_registry 未改动):c_tf.py(P1 TFNet 配方的只读 import 扩展:多周期头 MHNet、逐日 pairwise 排序损失、持续性惩罚、g20 purge 几何,split() 内断言标签到期边际)、c_wf.py(按 4 信号 retrain 块存 checkpoint、可续跑)、c_super.py(监督重启)、c_lgbm.py(滞后堆叠 LightGBM,同一 60×(6通道+mask)=420 列)、c_eval.py/c_agg.py(评估)、c_protocol.py(results\c_protocol.json,声明于任何变体结果之前)。
- 复现检查:T0 seed 42 第 0 个 retrain 块用新训练器重训,与 P1 冻结 checkpoint 的预测最大差 8.6e-6(相关 1.0),确认配方一致(results\c_repro_r000.json)。
- 算力实测与减配声明(先于任何变体结果):单个 52 块 Transformer 全程约 35 分钟(随 n_train 增长,块越晚越慢);GPU 为单卡时间片,双进程总吞吐仅约 1.2×,且 Windows 提交内存只够 1 个 GPU worker 外加 CPU 作业(2–3 个进程曾触发 MemoryError/WinError 1455)。全量 3 种子×5 变体×52 块约 9 小时,故:T0 的 seed 43/44 跑满 52 块;消融变体 T1/T2/T3/T0G/T4 只训练前 26 个 retrain 块(信号 0–103,2022-07 起约 2 年),先 seed 42/43,时间允许再 seed 44;变体对基线的配对比较只用这 104 个信号。LightGBM 滞后堆叠 3 个种子全 52 块。
- 运行中:c_super.py(1 个 GPU worker,T0 seed 43 已到第 30/52 块)+ c_lgbm.py(seed 42 → 43 → 44,每块约 7 秒)。
## [P2-C 进度2:seed 43/44 第一批结果] 2026-10-19 14:08 (UTC+8)
- T0(P1 配方)seed 43 已跑满 52 块/207 信号;seed 44 进行中(7/52 块 = 28 信号)。滞后堆叠 LightGBM seed 42、43 已完成全 52 块,seed 44 进行中(每块约 7 秒;早停最优树数多为 15–130 棵,信号弱)。
- 初步(只看信号层;A0/C1/C2 净收益与因子 alpha 待全部完成后由 c_agg.py 统一计算):RankIC 按 P2-A 新口径(执行日前向收益,截面去均值)
  - T0 s42(P1 冻结):207 信号 RankIC 0.0810(HAC t 8.42),相邻信号日 Spearman 0.643,Top50 相邻重合 0.334
  - T0 s43:RankIC 0.0796(t 8.55),稳定性 0.659,Top50 重合 0.326
  - T0 s44(仅前 28 信号,不可比):RankIC 0.0736
  - LAGLGBM s42:RankIC 0.0869(t 8.27),稳定性 0.686,重合 0.284;s43:0.0809(t 7.96),0.685,0.285
- 初读:seed 42 与 43 的 Transformer RankIC 很接近(差 0.0014),seed 42 不像是"幸运种子";滞后堆叠 LGBM 在同一信息集下 RankIC 并不低于 Transformer。以上为点估计,配对检验和 block bootstrap 待做。
## [P2-C 进度3:T0 三种子完成 + 滞后堆叠 LGBM 配对] 2026-10-19 14:39 (UTC+8)
- 中断后检查:GPU worker(pid 61100,c_super 管理)一直在跑,没有重启;已完成的块没有重做。中断杀掉的 LAGLGBM_F200 进程从第 3 块续跑并已完成;卡死的 _q.py 已结束。T0 seed 44 在 14:36 跑满 52 块,队列已自动进入 T1 seed 42(26 块)。
- T0 三种子(207 信号;A0 Top50,C2 显式成本;seed 42 = P1 冻结结果,与 P2-B A0 净收益/换手对账差 <1e-16):
  - RankIC:s42 0.0810 / s43 0.0796 / s44 0.0778,均值 0.0795,std 0.0016;HAC t 7.7–8.5
  - 稳定性:0.643 / 0.659 / 0.653
  - Top50 重合:0.334 / 0.326 / 0.325
  - 换手/期:0.646 / 0.653 / 0.649
  - C2 净年化:−2.5% / +0.7% / −2.9%,均值 −1.6%,std 1.9%
  - C2 alpha t:−0.87 / −1.21 / −1.17
  - 跨种子同日分数 Spearman 0.77
- 滞后堆叠 LGBM(3 种子)− T0,207 信号种子均值配对:
  - RankIC 差 +0.0036(t 0.67,不显著)
  - 稳定性 +0.035(t 2.35)
  - Top50 重合 −0.046(t −3.0)
  - 换手 +0.035(t 2.46)
  - C2 净收益 −10bp/期(t −0.91)
  - C2 alpha 差 −26bp/期(t −2.58)
  - LGBM 自身 C2 alpha t 为 −2.5 到 −3.5
- 事后诊断:T0 分数做因果 EMA(0.5) 后,换手 0.65→0.42–0.43,RankIC 降 0.003(t −1.2 到 −1.6),C2 净收益 +4 到 +20bp/期(t 0.6–2.0)。固定 200 树的 LGBM(无早停,seed 42)RankIC 0.078,换手 0.75,更差。
- 剩余队列:5 个消融变体 × seed 42/43(×44 若时间允许),每个 26 块。
## [P2-C 进度4] 2026-10-19 15:05 (UTC+8)
- 第二次中断后没有杀进程,也没有重启;GPU worker pid 61100 连续运行。回归测试已全部重跑通过:P0 45、P1 65、P2-A 24、P2-B 24(tests\pytest_*_rerun_c.txt)。c_figs.py 生成 6 张图,无缺字形警告;c_report.py 已能从 CSV 生成报告。
- T1(排序损失,5 日头)seed 42 前 26 块完成(约 18 分钟)。前 104 信号上 T1−T0 配对(仅 seed 42,初步):
  - RankIC +0.0198(HAC t 2.09,CI [0.0003, 0.039])
  - 稳定性 +0.121(t 6.6)
  - Top50 重合 +0.189
  - 换手 −0.174/期(t −8.2)
  - C2 净 +11bp/期(t 0.69,不显著)
  - C2 alpha 差 −23bp/期(t −1.27)
  - T1 自身 C2 alpha t −3.69,疑似因子暴露变化,待查 beta
- 正在跑 T2 s42(第 21/26 块)。剩余队列:T3、T0G、T4 的 s42,然后 5 个变体的 s43;每个约 12–18 分钟。
## [P2-C 进度5] 2026-10-19 15:32 (UTC+8)
- seed 42 前 26 块已完成 T1、T2、T3,正在跑 T0G(第 8/26 块),之后是 T4,再之后是 5 个变体的 seed 43。
- 前 104 信号,仅 seed 42 的初步结果:
  - T2(多周期 MSE):RankIC 0.0783,换手 0.594,C2 alpha t −2.83
  - T3(多周期排序损失):RankIC 0.1074,换手 0.417,C2 alpha t −3.37
  - T3−T2(只换损失):RankIC +0.029(t 2.41),稳定性 +0.100(t 8.8),换手 −0.177(t −9.4),C2 净 +29bp/期(t 1.56),alpha 差 −11bp(t −0.67)
  - T3−T1(多周期头+g20 几何,混杂):RankIC +0.005(t 1.0)
- 初读:排序损失(T1、T3)稳定地提高 RankIC 和分数持续性、降低换手,但因子调整后的 alpha 没有改善,组合向 SMB 倾斜更多。多周期头本身的效果要等 T0G 对照(T2−T0G)。
- 时间安排:seed 42 预计约 16:00 全部完成;seed 43 一套约 85 分钟。按 30 分钟规则,设定 17:30 为训练截止,届时未完成的(变体, 种子)不进入表格,并写明。
## [P2-C 进度6] 2026-10-19 16:05 (UTC+8)
- 用户确认取消 17:30 的训练截止:seed 43 的 5 个消融变体(T1/T2/T3/T0G/T4,前 26 块)全部跑完,进程正常推进期间不减配。T4 seed 42 已到第 24/26 块。
- 测试状态:test_p2c.py 28 项中 27 项通过;剩下 1 项要求 c_paired.csv 中包含所有变体,需等聚合重跑。
## [P2-C 进度7:seed 42 全部变体完成] 2026-10-19 16:10 (UTC+8)
前 104 信号、仅 seed 42 的配对结果(A−B),逐项列出 RankIC、稳定性、换手、C2 净收益/期、C2 alpha/期,括号内为 HAC t:

- **T0G−T0**(只改几何)
  - RankIC −0.006(−1.06)
  - 稳定性 +0.009(0.80)
  - 换手 +0.016(1.15)
  - C2 净 −6bp(−0.54)
  - alpha −15bp(−0.97)
  - 结论:全部不显著
- **T1−T0**(只改为排序损失)
  - RankIC +0.020(2.09)
  - 稳定性 +0.121(6.6)
  - 换手 −0.174(−8.2)
  - C2 净 +11bp(0.69)
  - alpha −23bp(−1.27)
- **T2−T0G**(只加多周期头)
  - RankIC +0.002(0.30)
  - 稳定性 +0.052(3.1)
  - 换手 −0.069(−4.3)
  - C2 净 −9bp(−0.88)
  - alpha +6bp(0.52)
- **T3−T2**(多周期下换为排序损失)
  - RankIC +0.029(2.41)
  - 稳定性 +0.100(8.8)
  - 换手 −0.177(−9.4)
  - C2 净 +29bp(1.56)
  - alpha −11bp(−0.67)
- **T4−T0**(持续性正则 λ=0.5)
  - RankIC −0.042(−4.05)
  - 稳定性 +0.246(10.2)
  - 换手 −0.368(−16.5)
  - C2 净 +18bp(1.16)
  - alpha +8bp(0.44)
  - 结论:换手大幅下降,但预测力被明显牺牲
- **LAGLGBM−T0**(三种子在前 104 信号上取 seed42 配对)
  - RankIC −0.001(−0.12)
  - 换手 +0.059(3.3)
  - alpha −25bp(−1.68)

队列已进入 seed 43(T1 → T2 → T3 → T0G → T4,预计约 17:35 完成)。
## [P2-C 进度8] 2026-10-19 16:34 (UTC+8)
- T1 seed 43 完成(16:04–16:24)。T2 seed 43 进行中,第 20/26 块。之后依次是 T3、T0G、T4 的 seed 43,预计约 17:35 全部完成。worker pid 61100 一直正常推进。
## [P2-C 进度9] 2026-10-19 16:55 (UTC+8)
- seed 43 的 T1、T2、T3 已完成;T0G s43 于 16:53 开始,之后是 T4 s43,预计约 17:35 完成。不减配,进程没有重启。
## [P2-C 进度10] 2026-10-19 17:12 (UTC+8)
- T0G seed 43 已完成;T4 seed 43 于约 17:06 开始,预计约 17:30 完成。报告生成器已加入因子暴露表、按单因素自动生成的结论、实际执行与减配说明;完成后统一重跑聚合、出图、报告和测试。
## [P2-C 完成] 2026-10-19 17:47 (UTC+8)

### 运行与产出
- 训练:T0 seed 43/44 跑满 52 块(seed 42 = P1 冻结结果,对账差 <1e-16);滞后堆叠 LGBM seed 42/43/44 跑满 52 块;5 个消融变体的 seed 42/43 跑满前 26 块(104 信号)。消融的 seed 44 未纳入:监督进程在 seed 43 完成后自动开始了 T1 s44,按"不杀进程"的指示没有终止它;它的块不完整,不进入任何表格。
- 输出:
  - results\c_runs / c_seed_variance / c_paired / c_posthoc / c_oracle_diag / c_head_rankic / c_coverage / c_reconcile / c_block_meta(.csv)和 c_by_date.parquet
  - figures\c_fig1–6(非空,无缺字形警告)
  - reports\p2c_transformer_ablation_report.md
- 测试:test_p2c 28 通过;回归 P0 45、P1 65、P2-A 24、P2-B 24 全部通过;manifest 22 个 sha256 一致。

### 种子方差(T0,207 信号)
- RankIC 0.0795±0.0016(0.0778–0.0810)
- 稳定性 0.652±0.008
- Top50 重合 0.329
- 换手 0.649/期
- C1 净年化 6.6%,C2 净年化 −1.6%±2.0%(−2.9% 到 +0.7%)
- C2 alpha t −1.08
- 跨种子分数相关 0.77

### LAGLGBM − T0(207 信号,种子均值)
- RankIC +0.0036(t 0.71,不显著)
- 稳定性 +0.035(t 2.55)
- Top50 重合 −0.045(t −3.1)
- 换手 +0.038(t 2.74)
- C2 净 −7bp/期(t −0.65)
- C2 alpha −23bp/期(t −2.27)
- 结论:匹配信息集下 Transformer 没有 RankIC 优势

### 消融(前 104 信号,两种子均值)
各变体自身指标:

| 变体 | RankIC | 稳定性 | 换手 | C2 alpha t |
|---|---|---|---|---|
| T0 | 0.0830 | 0.687 | 0.671 | −1.48 |
| T0G | 0.0786 | 0.705 | 0.668 | −3.25 |
| T1 | 0.1049 | 0.810 | 0.497 | −3.56 |
| T2 | 0.0790 | 0.750 | 0.595 | −2.75 |
| T3 | 0.1073 | 0.863 | 0.412 | −4.22 |
| T4 | 0.0279 | 0.943 | 0.190 | −0.72 |
| LAGLGBM | 0.0795 | 0.711 | 0.698 | −5.03 |

单因素配对(A−B):

| 配对 | RankIC 差(t) | 稳定性差(t) | 换手差(t) | C2 净差/期(t) | C2 alpha 差/期(t) |
|---|---|---|---|---|---|
| T0G−T0 | −0.005(−0.86) | — | — | −8bp | — |
| T1−T0 | +0.022(2.40) | +0.115 | −0.166 | +14bp(1.13) | −12bp(−1.06) |
| T2−T0G | +0.0004(0.10) | +0.046(3.67) | −0.073 | +5bp(1.01) | +9bp(1.31) |
| T3−T2 | +0.028(2.78) | +0.112 | −0.183 | +18bp(1.22) | −15bp(−1.34) |
| T4−T0 | −0.055(−4.40) | +0.249 | −0.473 | +13bp(0.72) | +15bp(0.79) |

### 结论
- 排序损失提高 RankIC 和稳定性、降低换手,但组合 SMB 暴露上升,因子调整 alpha 没有改善。
- 多周期头只提高稳定性。
- 持续性正则牺牲预测力,且对种子很敏感。
- 没有任何变体的 C2 净收益或 alpha 显著改善。
- 全部结果为探索性。
## [P2 交付 7–9 进度1] 2026-10-19 17:47 (UTC+8)
- 按用户"必要时接着跑完"的意见,seed 44 消融队列(T1→T2→T3→T0G→T4)继续运行,没有停;T1 s44 已到第 13/26 块左右,预计约 19:15 全部完成。
- 交付 7 已完成:
  - 新增 stock_dashboard_p2.py(只读),6 个页面:概览;P2-A 分组 IC/偏 IC/十分位/因子归因/新旧口径;P2-B 净值/成本分解/换手 vs 预测/因子 alpha/配对;P2-C 种子方差/消融/配对/lag-LGBM vs TF/事后诊断;P1 逐股入口(链接 8501 的 v2 看板,另有只读的快速查询);报告与图。
  - 新增 run_dashboard_p2.ps1(端口 8502)和 smoke_dashboard_p2.py。
  - AppTest 22/22 通过;无头 HTTP 检查 /_stcore/health 与根页面均为 200(tests\http_dashboard_p2.txt);tests\test_dashboard_p2.py 5 项通过。
  - P1 的 v1/v2 看板未改动。
## [P2 交付 7–9 进度2] 2026-10-19 18:08 (UTC+8)
- seed 44 消融:T1 s44 已完成(17:38–17:57),T2 s44 进行中。
- 交付 9 的打包脚本 c_bundle.py 已写好并试运行:197 个文件,21.3 MB。包含逐文件 sha256、整包 sha256、面板 3 个日期的抽样;凭据/隐私扫描只命中冒烟输出里的 Temp 路径,已改为不记录 stderr 后清除。正式包等 seed 44 刷新后重建。
- 已保存只有 seed 42/43 时的快照 results\c_{runs,seed_variance,paired}_s42s43.csv;报告新增 §5b,对照加入 seed 44 前后的结果。
[P2 交付 7–9 进度3 / t12u 研究选股器] 2026-10-19 18:18
- 新增 c_screener.py → results\research_screener_20261016.csv(1498 行,可选池 1178;真实代码+名称)+ _columns.md + _meta.json。六模型冻结分数/排名与 P1 发布一致;TF 三种子(42/43/44)用最近 OOS 信号日 2026-09-28(未保存模型、不重训,已注明);三因子基线 = REV5/LOWVOL/ILLIQ;A2 研究权重用各模型 C2 嵌套最近一块参数(HGB t0.5 c0.03, LGBM t1.0 c0.03, Ridge t0.5 c0.05, MLP/GRU/TF t1.0 c0.03),空仓起步,持股 92–100 只,GRU 权重和 0.748,其余 1.0。
- stock_dashboard_p2.py 新增「研究选股器」页(侧栏第二位):顶部免责声明、行业(最新非时点)/交易所/流动性/一致性/排除涨停·ST·科创北交 筛选与排序、个股历史 OOS 分位、实际后续收益散点、复权K线。smoke 32/32,tests\test_screener_p2.py 9 passed,HTTP 200 复测通过。
- seed 44:T3 s44 进行中(18:12 block 8/26);之后 T0G、T4。pid 61100 正常,未杀/未重启。
[P2 交付 7–9 进度4] 2026-10-19 18:40
- seed 44:T3 s44 于 18:29 完成(26 块);T0G s44 进行中,之后 T4 s44。pid 61100 正常,未杀/未重启。等待期间无其他改动。

[P2 交付 7–9 完成] 2026-10-19 19:22
- seed 44 全部完成:T4 s44 19:14 DONE,supervisor 19:14:51 "ALL JOBS DONE",worker 自然退出(全程未杀/未重启)。
- 已重跑 c_agg / c_figs / c_report(§5b 保留 s42s43 对照;§10 测试表新增仪表盘/冒烟/HTTP)。三种子均值:T1 IC 0.1029、T3 0.1085、T2 0.0800、T0G 0.0790、T4 0.0319(跨种子相关 0.25)。配对检验中显著结果都没有变方向;符号变化的只有原本不显著的项(T0G−T0 重合/换手、T1−T0 C1 净、T3−T1 alpha)。新变为显著的:T1−T0 alpha C2 t −1.06→−2.00、C1 −1.93→−2.91;T3−T0 alpha C1 −1.88→−2.25(均为负)。
- 测试:test_p2c 28、P0 45、P1 65、P2-A 24、P2-B 24、dashboard+screener 14(其中 screener 9)全部通过;smoke 32/32;HTTP 200(19:17 冷启动)。
- 打包:旧包改名 p2_review_bundle_v1.zip(+ .sha256、README_p2_bundle_v1.md)。新 p2_review_bundle.zip 21.80 MB、206 个文件、sha256 4076f1e1d10ec1e38bbaa9ae5c5c69fffef10e1bcac421a1c819f14658fb9d2a,扫描命中 0。_patch.py 已删除。
[P3 第一批 进度1] 2026-10-21 19:38
- 新目录 p3_ai_report。selection_rules.json 已在计算前写定(19:3x,sha256 0B9CDEF5…)。增量采集 p3_fetch.py:新浪原始日线经本机代理频繁 SSL EOF/挂起,改用 P0 已记录的腾讯备用源(北交所仍用新浪),新浪 qfq-factor 取复权因子;py_mini_racer 非线程安全 → 8 进程分片单线程 + socket 超时 12s。写入 p3_ai_report\data\incr_raw,不覆盖 P0 raw。
- 已写 p3_merge.py / p3_score.py / p3_calib.py / p3_rank.py(待数据)。

## 2026-10-21 P3 第一批收尾(执行器)
- 重跑 p3_fin.py(含 603822 ST嘉澳)。新增 p3_calib_supp.py → results/calibration_supplement.csv(事后补充诊断:中位数/均值相对收益、跑赢池中位数比例、std、尾部占比;原定义不变)及 results/hist_composite.parquet。
- results/company_info_20261021.json(新浪主营+公告标题,688551 公告未取得)。
- p3_report.py → figures/cand_<code>_20261021.png ×10(均>130KB)与 reports/daily_ai_report_20261021.md;下跌趋势 3 只报告层标“暂不买入(继续观察)”,规则文件未改。
- tests/test_p3_first.py:7 passed。pred_20261021 注册表未改(sha 与 manifest 一致,只读)。删除 _patch.py。
## 2026-10-21 20:18–20:35 P3 第二批(执行器)
- selection_rules_v2.json 20:18:34 写入并登记 rules_manifest.jsonl(sha256 87dc4156…),只读;之后才运行 p3_v2.py。v1 规则/结果/pred_20261021 未改。
- p3_feat_hist.py → data/feature_panel_hist.parquet(208 日特征);p3_v2.py → latest_research_ranking_20261021_v2.csv、results/top5_v2/watch_v2/avoid5_v2、hist_top5_*、hist_v2_constraint_check、hist_bins_by_year。
- p3_history_report.py → reports/history_vs_actual_report.md、model_reliability_summary.md、figures/hist_*.png。
- p3_report_v2.py(由 p3_report.py 派生)+ p3_fin_v2.py + company_info_20261021_v2.json → reports/daily_ai_report_20261021_v2.md。
- p3_verify_forward.py + run_weekly_verify.ps1(verification/ 只追加;当前 0 个到期)。
- dashboard/(每日AI选股报告.py + pages/1_历史预测vs实际.py)+ run_dashboard_p3.ps1(8503)+ smoke_dashboard_p3.py:16/16(含 HTTP 200)。
- tests:test_p3_first 7 + test_p3_second 8 = 15 passed。