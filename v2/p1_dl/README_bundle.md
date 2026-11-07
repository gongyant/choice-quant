# review_bundle 说明(P1 轨道A/B 终稿评审包)

生成时间:2026-10-19 08:52(Asia/Singapore)。本包供独立审计:检查数据质量修复、walk-forward 无前视、DL 模型实现、对账与统计结论。**研究排名≠买入建议;模型 score 不是概率/收益率;未下任何真实订单。**

## 1. 一句话结论

修复 P0 数据/成交问题后,HGB/LightGBM/Ridge/MLP/GRU/Transformer 六个模型在 0.30% 往返成本下的 Top50 策略,**没有任何一个相对同制度等权基准的净超额统计显著**(HAC t 均 |t|<1.7,bootstrap 含 0;配对差异均不显著);Rank IC≈0.07 为正但被成本吃掉。2025–2026 非盲测;前向验证从 2026-10-19 起(见 forward_registry)。

## 2. 文件清单(含含义、大小、sha256)

| 路径(包内) | 含义 | 字节数 | sha256 |
|---|---|---:|---|
| `p0/p0_audit_report.md` | P0 修复/审计报告(根因、修复、修复前后对比) | 19,125 | `836b1d32c9983976d9974700877fc3e16b43d6f1948811d19320683c7a0bc1ff` |
| `p0/comparative_metrics.csv` | P0 修复前后/各策略对比指标 | 19,255 | `86cf276c0684978ec58e9f925ad62ee202c153c0968271133d4c34b2f7a424f4` |
| `p0/bootstrap_intervals.csv` | P0 5期移动区块 bootstrap 区间 | 12,514 | `f7ecd64cbd14c90b43de1124d1fc09ac0178400252f097ae3f7f64eefd3c2279` |
| `p0/cost_sensitivity.csv` | P0 成本敏感性 | 7,429 | `7a84e6ca58ff709b32ac8f190fd3f121987324418af030e81282e028ba147b49` |
| `p0/audit_manifest.json` | P0 文件 sha256 清单 | 5,638 | `46831929276a213219edd03e5cf6a9ab03df1ed084e75bb64dda11165872bc17` |
| `p0/quality_event_log.csv` | P0 异常事件日志(1272条/266只;含少量原始/复权价格字段,仅为事件证据) | 527,044 | `287aa1101b9166a02fbc2edad1a2b21b17b3d04e8d1785d069b7d752f61aef08` |
| `p1_dl/latest_all_stocks_20261016_v2.csv` | 2026-10-16 全部1500只样本主表(6模型分数/排名/分位+模型分歧) | 1,563,071 | `70ccb2ff00b0b4571f742a64f56a53e68175c68ab84115b282ea33d4400ea313` |
| `p1_dl/per_stock_summary_v2.csv` | 分个股汇总(含 DL 模型 Spearman/样本数,<30 标不可靠) | 2,179,333 | `7cb429c8113cd3e4c3400dcd523420723173dfb5597ce0d48aff39f4f0b2a57b` |
| `p1_dl/sector_summary_v2.csv` | 分行业/交易所/板块 Rank IC 等(6模型) | 36,584 | `3d80b5ac7d393f28f16fdc9b56aecc0ce7e3d094aac8f48ae4f63c9e09a2ad79` |
| `p1_dl/p1_dl_report.md` | P1 总报告(中文) | 47,643 | `d14fed1ac497c58f179877218594c9a428a209029f012ddb14eff39b6b008199` |
| `p1_dl/dl/dl_models_report.md` | 轨道B 深度学习模型报告(MLP/GRU/TF) | 10,785 | `f64feb9032205a9a50eafa3a659164bc74bb79254749ebc65d4aace6352b4ba8` |
| `p1_dl/forward_registry_20261016.csv` | 前向验证冻结登记:6模型 Top50(300行) | 80,678 | `25dec2927014865ee2a6ebcbb8809e40347192b810ee75cc9242342a8765bf52` |
| `p1_dl/forward_registry_20261016_manifest.json` | 前向登记 manifest(22个关键文件 sha256 + 预先声明的评估规则) | 4,222 | `28728c6f12e3b2a2d8dc79c661ff1854681c05785815d14d884d6ab698fcefae` |
| `p1_dl/historical_predictions_all_models_last12dates_eligible.csv` | 完整历史预测的抽样:最近12个信号日、合格行(已去掉价格/股数列) | 32,334,742 | `9e1c9ae90630ed6242f0a92220cf9a621e33023c16a2342263345377d2df6c4f` |
| `p1_dl/dl/results/mlp/side_by_side_c30.csv` | 轨道B MLP 结果:side_by_side_c30.csv | 5,294 | `ae89e29882a849e76a47881cfaeaad60a6075ea4b90d7348d3b993bc19a62e21` |
| `p1_dl/dl/results/mlp/cost_sensitivity.csv` | 轨道B MLP 结果:cost_sensitivity.csv | 9,795 | `9a91972f02d1ec2f5ae611436d021c6e1875e67be741b9782ec21bf655aa84b6` |
| `p1_dl/dl/results/mlp/by_year.csv` | 轨道B MLP 结果:by_year.csv | 5,017 | `0bdf6edac896152219ff9bdd697c51e05bfbe28c89dba28b73045475cda7033b` |
| `p1_dl/dl/results/mlp/paired_vs_p0.csv` | 轨道B MLP 结果:paired_vs_p0.csv | 853 | `b1dcc1e9fca4f08b320a1b3f1e5acbbdf3aca1049d906f47e42c4b3e10df2115` |
| `p1_dl/dl/results/gru/side_by_side_c30.csv` | 轨道B GRU 结果:side_by_side_c30.csv | 4,233 | `09ea6a7ffdbb32314cf803a1d53eb6a69e6450a2403784cb5f6e8ce98b0dc22c` |
| `p1_dl/dl/results/gru/cost_sensitivity.csv` | 轨道B GRU 结果:cost_sensitivity.csv | 7,782 | `99b45151a664fd02d22a63a6e0a03a0f759b528c7e7160865c8efa3dd2910608` |
| `p1_dl/dl/results/gru/by_year.csv` | 轨道B GRU 结果:by_year.csv | 3,998 | `6d05afb60fc2232a1760117eceafa879535384e811dc4a1b69c4e58abbc32015` |
| `p1_dl/dl/results/gru/paired.csv` | 轨道B GRU 结果:paired.csv | 1,258 | `bb33499e9c44e379524d7e7f3c41892e824d72a348cef7f951bd15ed7821bbc8` |
| `p1_dl/dl/results/tf/side_by_side_c30.csv` | 轨道B TF 结果:side_by_side_c30.csv | 3,709 | `3147e2a48aa0cd3e39cea6f1d8392ecf9d8b1eb85137d21612bde6f8daf28953` |
| `p1_dl/dl/results/tf/cost_sensitivity.csv` | 轨道B TF 结果:cost_sensitivity.csv | 6,744 | `b8be5918647aff749b113f980b13fb50786478b316d6ec751a892798b8106543` |
| `p1_dl/dl/results/tf/by_year.csv` | 轨道B TF 结果:by_year.csv | 3,485 | `9ae8b4e37ca0cd1a358839ef47f9684883bd568f1b04e93207f76ae4af64b246` |
| `p1_dl/dl/results/tf/paired.csv` | 轨道B TF 结果:paired.csv | 1,655 | `5ef12cd4fad9ae2b97bcfbdcef03852a138fd579607dc5a2fe0af35c8742329d` |
| `p1_dl/reports/recent20_metrics_v2.csv` | 最近20期逐期指标(6模型) | 18,491 | `848ee241793498e26d44767171ecbbaf02b2645fbf305f68cd056e3bb79212fc` |
| `p1_dl/reports/recent20_metrics_v2.md` | 最近20期明细(Markdown) | 13,529 | `18d804e3d53c7186d05e2c48a3825eedf5e87293c66a0d42d9733e7227529575` |
| `p1_dl/reports/recent20_summary_v2.csv` | 最近20期汇总 | 1,224 | `9dc5a399aea81e8acfc4fbe6ed0eacd7dc64c16b1fb81ec618ea9b36190afc58` |
| `p1_dl/reports/top_bottom_20261016_v2.md` | 五模型 Top20/Bottom20、重合、≥3模型Top50 | 32,941 | `663d4c72a3dda5dc8965b9ac40d5f7f760196596c795b05b2a59cf4031f93691` |
| `p1_dl/reports/top_bottom_20261016_v2.csv` | Top/Bottom 表(CSV) | 33,260 | `993c60f9aa7be67f3e47f1702490212cd4fdea6ac67f9574d32f05c81e006a84` |
| `p1_dl/reports/multi_model_top50_20261016.csv` | ≥3个模型同时进入Top50的28只 | 4,081 | `7bbd6f5036a46a172740a249da8873566775bb12eb51541a638a10bb70663eae` |
| `p1_dl/reports/stock_cards_20261016_v2.md` | 三只个股卡片报告 v2 | 6,188 | `790e4793d1da22e469005c1d88606afdbbe1ae7a34ded0de046818f220922f54` |
| `p1_dl/reports/reconciliation_dl.md` | DL 对账报告 | 2,873 | `46005d7da1432c213ea1020627a50db70313c13513c3b498a115c9a931045434` |
| `p1_dl/reports/reconciliation_dl.csv` | DL 对账明细 | 2,034 | `39ac0f9d6446241dae9916b05b7d80eb8f44b46710f4a1961b8d9791e4f2809f` |
| `p1_dl/reports/model_comparison_v2.csv` | 六模型对比汇总 | 7,991 | `272ffb417527fb0d4c85ed2b4d24852b7fa6457ce054f19bfa665754a72d3e1e` |
| `p1_dl/reports/cost_sensitivity_all_models.csv` | 六模型成本敏感性 | 14,735 | `3e5541456f7c14c1526bf537ffb326eeaecf2fa16dbc4e297a14757a8524faab` |
| `p1_dl/reports/by_year_all_models.csv` | 六模型分年度 | 7,352 | `8cc41aae8e53a10ca4036edafe2ebd8fd2e7bb7005f6e777d48e412a67519481` |
| `p1_dl/reports/paired_tests_all_models.csv` | 配对检验汇总 | 3,549 | `bdb4baffa378b6341c6cb0105fc799e3a8cb3df10294d713202d417ccf70f034` |
| `p1_dl/reports/cumulative_net_models.csv` | 累计净值数据 | 57,957 | `1e81a9c9a94d1ea244f73ec563a821b721e103b07a86abec86ed92051f51e2c3` |
| `p1_dl/figures/card_600060_v2.png` | 600060 海信视像卡片 | 199,329 | `32f56cd0da61864d6397b87e0d1534a9702b668ebd5abd478fb2b6c7c5c1ad55` |
| `p1_dl/figures/card_300487_v2.png` | 300487 蓝晓科技卡片 | 175,617 | `20278a887f3e6474cd8aa519d7a8284fd72d376e832808a5e7644bacdf996cc6` |
| `p1_dl/figures/card_000027_v2.png` | 000027 深圳能源卡片 | 173,601 | `b3f0eda0e52a270d83fecfbe10658c0f18b4af9e071a09fa732a30527978ae1e` |
| `p1_dl/figures/sector_rankic_hgb.png` | HGB 分行业 Rank IC | 60,032 | `c880d02652a368ee5b3b13f3e166379276d8b3183bb4c2e4b708dd6491b0663f` |
| `p1_dl/figures/rolling_rankic_models.png` | P0 三模型滚动 Rank IC | 177,714 | `3438aa340381959ee9ca200205ed16c6fd9121fb8362b93af5f3198237aed16f` |
| `p1_dl/figures/cumulative_net_models.png` | 六模型累计净值 | 232,705 | `0bb0b9da419c3990141a3d9b7d5e430f2818a2df44cb27df06eeca517dbac72f` |
| `p1_dl/tests/pytest_result_final.txt` | 最终测试输出(P1 65 + P0 45) | 6,238 | `f43b1785367b4bdb1b0ccbd356c414e637c2be2cf23d7a42429f9b452d501a74` |
| `p1_dl/tests/test_final_a3.py` | A3 最终整合测试 | 7,830 | `6dcdac975f769eb730b5b79fcb18835f79baaf75014634fca2a94885fb7cf7a1` |
| `p0/p0lib.py` | P0 核心库(数据/特征/标签/模拟器) | 19,715 | `63e45932c310ef22613118a6dfcf34506a50f047dc0f21d18b56bd032299d427` |
| `p0/baseline_fixed.py` | P0 修复版 walk-forward | 7,676 | `812a97fe34b5071873363b175950d316baaf0eceae614d6cc482f940ca6cd8d8` |
| `baseline.py` | make_models(Ridge/HGB/LGBM 参数) | 10,389 | `fbd01e4199a48ac463d4f6da511d24cbc9139c57dc86b837f23394301de6c1b8` |
| `p1_dl/p1lib.py` | A1 冻结模型打分库 | 2,810 | `76cd32ebd2b22006d5f96cbd1708c5dab975583ed89399a5da0385f52c3e2aa8` |
| `p1_dl/score_latest.py` | A1 最新日打分 | 9,103 | `bc02dda424ebdb1c55d1a842418a9af2fb1e709704f309aae3d16b543d873e21` |
| `p1_dl/top_bottom.py` | A1 Top/Bottom | 3,341 | `fc717a29364d44808df80c07147648d9b278ed242632493d3489186c7fa4382b` |
| `p1_dl/cards.py` | A1 卡片 | 9,651 | `5795bc5e455ad8d1dc76badb34aa1d019b81d9f9c2413ebc09be905d3744450c` |
| `p1_dl/build_history.py` | A2 历史预测重放 | 15,632 | `1fd7e22aba225b98ad401ed3018a4e0ca7ee12c160397a1c12904a926f3c4087` |
| `p1_dl/summaries.py` | A2 汇总 | 11,869 | `5711df50e6aab6bc6b0fced2a72c01da2e73909425322fae0a143a7f4b9f1ac4` |
| `p1_dl/stock_dashboard_v2.py` | 仪表盘 v2 | 25,478 | `0edbb22e609f617b4b6ccfe26813382e9f1d46675d91361ebcb221ba4df1ff07` |
| `p1_dl/run_dashboard.ps1` | 仪表盘启动脚本(指向 v2) | 794 | `93f607dad547c7023d06064229e1dbf9d49769cff476690c349adf30f1960945` |
| `p1_dl/smoke_dashboard_v2.py` | 仪表盘冒烟测试 v2 | 1,747 | `ca0cb4036d5d81fecea405ba19d72d034d30cb03fda2c63c7e50ae42e2a15d84` |
| `p1_dl/dl/dl_common.py` | B 公共函数 | 1,967 | `3524dd9281d739638e785f2afea61d6bda4031272c6d285f17ac361eaeb27f85` |
| `p1_dl/dl/dl_mlp.py` | B MLP | 4,025 | `f537f2c967ba9ee1e93f3e86213c59d77d34a965ff3269668f3425da3c858151` |
| `p1_dl/dl/dl_seq.py` | B GRU/Transformer | 9,794 | `d4da918d7aee7924f8775c7de4ed6988382d22e5c31ab866443c2440ba9a9c62` |
| `p1_dl/dl/run_mlp_wf.py` | B MLP walk-forward | 13,037 | `67be0dac46d64da6d4dbb7433880769db95943b0579d748497b5bf66bb55b773` |
| `p1_dl/dl/run_seq_wf.py` | B 序列模型 walk-forward | 12,880 | `14c5418fe46f05f2f838e1a5cf866d91571fc7f42f3fa76130584fac2c9361c6` |
| `p1_dl/dl/dl_compare.py` | B 对比/统计检验 | 7,270 | `bf4373fe59dcdd8e7a23d686389059d4e807819b2874310d93775617e543b4fc` |
| `p1_dl/dl/dl_compare_seq.py` | B 对比(序列) | 6,389 | `7a4625899b0a6f16aefe068933d296a4f9f833741eba8b054382fb9f7bd5717d` |
| `p1_dl/a3/a3_latest.py` | A3 v2 主表+Top/Bottom | 10,286 | `f8580f2c2bf93d420216a0dc47c3df2bf4917ed0fe4add966cda9435bf566f3a` |
| `p1_dl/a3/a3_history.py` | A3 全模型历史表/对账/汇总 | 23,824 | `42e83b0765ed6d1f2e7c620f0b0d4d79dfc8a51f997f672e0d79fccd1354baf6` |
| `p1_dl/a3/a3_cards.py` | A3 卡片 v2 | 11,339 | `37ffbcc5978435fd9c9a11eff2822feeedfe541bff80111e5d73033af337f259` |
| `p1_dl/a3/a3_registry.py` | A3 前向登记 | 4,115 | `022f1f8872a235e81441d35f675bf2c360801913d73a26920c104de04a00bd31` |
| `p1_dl/a3/a3_report.py` | A3 报告生成 | 23,124 | `809178662e3b9e5afee074989543348da601f03bf56ba8860c9b73253f63d5ea` |
| `p1_dl/a3/make_dashboard_v2.py` | A3 仪表盘 v2 生成脚本 | 12,257 | `fe2d80370c1d97296e160934284ec215d68bccaef12aacba528924400488f7b5` |

## 3. 刻意排除的内容

- 原始行情明细(`v2\p0\data\raw_unadj_parquet`、`adj_factors`、Tencent/Sina 原始 parquet)、约 176 万行日线面板:**数据供应商再分发风险**,不放入。
- 凭据/账户/个人信息:不含;包内代码仅保留 `D:\choice_quant_starter\...` 形式的项目相对路径描述(已做关键字扫描,命中见下)。
- 完整历史预测大文件:`D:\choice_quant_starter\v2\p1_dl\historical_predictions_all_models.parquet`(1,860,516 行 = 6 模型 × 207 信号日 × 1498 只,约 98 MB;A2 版 `historical_predictions.parquet` 930,258 行 = HGB/LGBM/Ridge)和 `strategy_trades_v2.csv`(含成交价格)仅在本机,包内只放最近 12 个信号日、合格行的抽样 CSV(去掉成交价/股数列)。
- 训练好的模型文件(joblib / .pt 检查点)未放入,sha256 记录在 `forward_registry_20261016_manifest.json`。
- 代码凭据关键字扫描命中:0 处(无)。

## 4. 复现步骤(本机,PowerShell,项目根 `D:\choice_quant_starter`)

1. 环境:`.venv`(Python 3.13,pandas/numpy/scikit-learn/lightgbm/torch/streamlit/plotly);P0 数据在 `v2\p0\data`(不在包内)。GRU/Transformer 训练用独立 CUDA venv(RTX 5070 Ti),CPU 不保证逐位一致。
2. P0:`v2\p0\baseline_fixed.py`(walk-forward 207 期)→ `results\fixed_sina_adj`;P0 无前视单测:`cd v2\p0; ..\..\.venv\Scripts\python.exe -m pytest leakage_unit_tests`(45 项)。
3. A1/A2:`score_latest.py`(冻结模型打分 2026-10-16)、`build_history.py`(重放并记录每日×每股分数,与 P0 对账 ≤1e-16)、`summaries.py`。
4. B:`dl\run_mlp_wf.py`、`dl\run_seq_wf.py`(GRU/Transformer)、`dl\dl_compare*.py`。
5. A3:`a3\a3_latest.py` → `a3\a3_history.py` → `a3\a3_cards.py` → `a3\a3_report.py`;`a3\a3_registry.py` 只能运行一次(已冻结,脚本拒绝覆盖)。
6. 测试:`cd v2\p1_dl; ..\..\.venv\Scripts\python.exe -m pytest tests -v`(65 项)。仪表盘:`.\run_dashboard.ps1`(http://127.0.0.1:8501)。

## 5. 已知局限(审计时请重点看)

- 2025–2026 数据已多次用于模型比较,**不是盲测**。
- 1500 只分层抽样样本,含 13.9% 已退市股(2 只无数据);幸存者/抽样偏差;无完整历史退市池。
- 行业为巨潮最新分类,非历史时点;无历史 ST 清单(涨跌停 5% 仅近似);复权因子单源(新浪);价格 295 只样本单源;920196/600317 成交量单位无法核验。
- Transformer 仅种子 42;GRU 三种子年化 -2.8%~+4.5%,种子噪声大;MLP/GRU 为 3 种子分数均值;GRU/TF 用 GPU,与 CPU 不保证逐位一致。
- 往返成本 0.30% 为粗略假设,未建模冲击成本;无调参、无因子/模型消融、无行业/市值中性化;score 未校准。
- 600060 海信视像不在 1500 样本:HGB/LGBM/Ridge/MLP 为池外打分,GRU/Transformer 未打分(需序列输入)。

## 6. 建议的审计提问

1. 训练样本的标签到期日是否严格早于每个信号日?(检查 `historical_predictions_all_models` 的 train_end_date / train_label_end_date;测试 `test_training_ends_before_label_window...`、`test_*_oos_parquet_models_trained_before_prediction`)
2. `eligible_asof_t` 是否只用 t 日及以前的信息?买不到的订单是否真的留作现金而不是换股?(`p0lib.simulate`、`leakage_unit_tests`)
3. DL 对账是否真正独立?(`reports\reconciliation_dl.md`:用 OOS 预测 Top50 经同一模拟器重放,与 B 的 period_returns/订单日志最大差 ≤1.1e-16)
4. 净超额显著性是否被重叠标签高估?(报告了 HAC(4) 与 5 期区块 bootstrap;配对检验见 `paired_tests_all_models.csv`)
5. 6 个模型、多指标、多子样本的多重比较如何处理?(未做校正;所有结论为"无显著",前向验证将用冻结登记)
6. 成本假设 0.30% 是否合理?0.60% 时全部为负(`cost_sensitivity_all_models.csv`)。
7. 行业/板块 Rank IC 差异是否只是噪声?(组内样本少、非历史时点、多重比较)
8. 前向登记是否可被事后修改?(manifest 记录 22 个文件 sha256;v2 主表 sha256 写入每个登记行)

## 7. 校验

解压后在包根目录用 `Get-FileHash -Algorithm SHA256 <路径>` 与上表核对。zip 内含本 README(README 自身 sha256 不在表中)。
