# P2 审稿包(p2_review_bundle.zip)

生成:2026-10-19 17:48(Asia/Singapore),`c_bundle.py`。只读研究材料;不是投资建议;模型分数不是概率;2022-07 以后的样本外数据已被前序阶段反复使用,**所有结论均为探索性**。

- 整包 sha256:`57e9dd65ef00665041ae34562e3a2d0e74cac4c8b5ffbdb7ca94451635e9e94b`(同见 `p2_review_bundle.zip.sha256`);大小 21.32 MB;文件数 197(含包内 README)。
- 包内的 README 副本不含整包 sha256(无法自引用),其余内容相同。

## 内容
- 脚本:P2-A(`a_*.py`)、P2-B(`b_*.py`)、P2-C(`c_*.py`)、共享库 `p2lib.py`、只读看板 `stock_dashboard_p2.py` + `run_dashboard_p2.ps1` + `smoke_dashboard_p2.py`。
- 参数/协议:`results/b_protocol.json`、`results/c_protocol.json`、`a_verify.json`、`results/c_repro_r000.json` 等。
- 结果:`results/*.csv`(全部统计输出)、`results/b_runs/*.parquet`(P2-B 每期组合结果,看板净值页使用)、`results/c_by_date.parquet`(P2-C 逐信号日指标)、`results/b_logs` 抽样(TF C2 的 A0/A2 订单与决策日志 3 个文件)、运行日志 `results/*.log`。
- 面板抽样:`results/sample_p2a_panel_3dates.csv` = `p2a_panel.parquet`(完整 310086 行 × 42 列,41 MB,未打包)中信号行号 [np.int64(360), np.int64(875), np.int64(1390)] 三个日期的全部行。
- 测试与输出:`tests/*.py`、`tests/*.txt`;图 `figures/*.png`;三份报告 `reports/p2a_alpha_anatomy_report.md`、`p2b_cost_aware_report.md`、`p2c_transformer_ablation_report.md`。
- **未打包**:原始行情(P0 `data/`)、P0/P1 代码与冻结产物(forward_registry 由 manifest sha256 校验)、P1/P2-C 模型 checkpoint、完整 P2-A 面板、其余 P2-B 订单日志。

## 复现步骤(Windows PowerShell,项目根 `D:\choice_quant_starter`)
1. 需要 P0 数据与 P1 冻结产物在原路径(`v2\p0`、`v2\p1_dl`);venv:`.venv`(pandas/lightgbm/CPU torch/streamlit)、`.venv_cuda`(torch cu128,P2-C Transformer 训练)。
2. P2-A:`python a_panel.py; python a_verify.py; python a_eval.py; python a_figs.py; python a_report.py`。
3. P2-B:`python b_protocol.py; python b_bench.py; python b_launch.py`(24 个模型×情景作业)→ `python b_agg.py; python b_exposure.py; python b_figs.py; python b_report.py`。
4. P2-C:`python c_protocol.py; python c_repro.py`;GPU:`..\..\.venv_cuda\Scripts\python.exe c_super.py 1`(按 retrain 块存 checkpoint、可续跑);CPU:`python c_lgbm.py --seeds 42 43 44`(事后敏感性 `--fixed 200`);`python c_agg.py; python c_figs.py; python c_report.py`。
5. 测试:`pytest tests\test_p2a.py tests\test_p2b.py tests\test_p2c.py tests\test_dashboard_p2.py`;P0 `pytest ..\p0\leakage_unit_tests`;P1 `pytest ..\p1_dl\tests`。看板:`.\run_dashboard_p2.ps1` → http://127.0.0.1:8502。

## 主要局限
- 样本外期 2022-07 至 2026-10 只有 207 个周度信号,且已被 P0/P1/P2 反复使用;所有检验都是探索性的,没有新的保留期。
- P2-B 的成本模型(Corwin-Schultz 半价差 + 平方根冲击)是参数化假设;C3/C4 约束下毛收益明显塌缩;没有任何策略的因子调整净 alpha 显著为正。
- P2-C:消融变体只在前 104 个信号(前 26 个 retrain 块)上训练(事先声明的减配);种子数少;一次只改一个因素,但 T3−T0 等打包比较不可分解;持续性正则只试了一个 λ。
- 候选池、行业为最新而非时点数据(P2-A 已披露);ST 状态部分依赖当前名称。

## 给独立审稿人的问题
1. 执行日前向收益(下一可买日入场、跌停顺延卖出)的定义是否仍有隐性前视?请抽查 `p2lib.exec_forward` 与 `sample_p2a_panel_3dates.csv`。
2. P2-B 嵌套时间验证(burn-in 52 期、26 期块、扩张窗口)是否足以排除参数选择乐观偏差?`b_nested_optimism_check.csv` 的结论可信吗?
3. 显式成本模型的量级是否合理(尤其小盘、北交所的半价差与冲击),是否会系统性地偏向某类策略?
4. P2-C 的 20 日标签 purge 推导(回看窗口不计入 embargo,只看标签到期)是否正确?`c_tf.split` 中的两条断言是否充分?
5. 排序损失提高 RankIC 的同时显著增加 SMB 暴露:这是新增的预测力,还是更高效地加载了小盘因子?五因子回归是否足以回答?
6. 配对检验采用 HAC(4) 与 5 期 block bootstrap;在 104/207 个周度观测、多重比较的情况下,哪些“显著”结论应当被校正后放弃?
7. 滞后堆叠 LightGBM 早停常只保留极少的树:这是否让“Transformer 无受控优势”的比较对 LightGBM 不公平(或反之)?

## 凭据 / 隐私扫描
- 规则:AWS key、私钥头、GitHub/类 OpenAI token、Bearer、password/secret/api_key 赋值、邮箱、带电话上下文的手机号、`C:\Users\<name>` 路径。命中 2 条:
  - `c_bundle.py`:user_home → `C:\\Users\\<name>``
  - `tests/smoke_dashboard_p2.txt`:user_home → `C:\Users\Administrator`

## 逐文件 sha256

| 文件 | 字节 | sha256 |
|---|---|---|
| `a_diag.py` | 2147 | `0c5d67d2e650b2869fe649dcb9c483923967c2f38e99c86040db71d9f3df5cea` |
| `a_eval.py` | 17159 | `f22c6a1a6e86fcce253e13d7ee9855feccac7103a522be642da7dd51ad62acbf` |
| `a_figs.py` | 8582 | `87d3fe45660cf6d15e11223a05f7cd8da465207f7f9d12eab243910b3944411e` |
| `a_panel.py` | 4051 | `9c8494bad79b514afb1f6bb97c5d068fcdbeca67c12be1dde4c34e92c3e1517c` |
| `a_report.py` | 29260 | `667c63f88c4dbf4da5ce9b843cbea6e562541b9874390ee290ef78a2b10d3cf5` |
| `a_verify.py` | 12076 | `8b2defa79e23b6df435004f0373f2c899090112f2c4e0c5d2721aeff62348c37` |
| `b_agg.py` | 11590 | `d76a6823331b200369523bdf351c35108ca4a015ba34e007d4d917f1abd21942` |
| `b_bench.py` | 2365 | `231b86eafb76f76d84793e3bca184c727e45bb6c1407fc24e2cc19bfcf3f3a92` |
| `b_data.py` | 1727 | `cee271b6d6e40edc2dc5ab544091ae4df7c456506674d304276b47d47f8b3596` |
| `b_exposure.py` | 1794 | `c0b4236af9c8ae1a83a2a03304fc2fd93592005144b1565e4b86ddb0c5362be0` |
| `b_figs.py` | 10831 | `81c7fcb2e9c6a3ec7f8a3fcdbe9381eb1df146da9e602627117e7c8e0f0a1b2d` |
| `b_launch.py` | 1109 | `92d0441817449e2ad1317f8519f28d2a2a2f05048756fd45cd45bb4c0a0db459` |
| `b_lib.py` | 9153 | `61b0f26ab98bcffddc43479cbe6d21b1b6d142a7b465a66d4028068b10a0a8f8` |
| `b_protocol.py` | 4280 | `1f0b1e6dd8b696c876c15065d7ec54e4e20654bbe539b6ff50d6601cebc215ff` |
| `b_report.py` | 27449 | `d7821dc669cb15da58aa9d559fcaa3c0ad705b76f3359d0dd209daa0bf47b8f0` |
| `b_run.py` | 955 | `875e8bbdbd289d442f6146d2fc097db8a8a61f299e63fa1cb7f6d1824c5d9355` |
| `b_sim.py` | 24658 | `d5f4b917c06d9a2c9e3b911aca8194aa03cf9aac83b859d9d4b3a8c3a6c24b0f` |
| `c_agg.py` | 11428 | `4e3077b9289c245ab2d33c25dd5bf8aca632ecdc6403252732a7fab11020b64a` |
| `c_bundle.py` | 9480 | `5b6f26985b88c4368918612755ed7eae08a467d24c6bb59b000c7f492373ae71` |
| `c_eval.py` | 8492 | `1f1d557f33bf23a2df89161f6323dbe5d8d1270ec487dc9c7b9ce0f196f1c8c6` |
| `c_figs.py` | 7295 | `0722148ec5f0f106239337aef11dc20ab4e8526785a8308781b7fc821a47d5f8` |
| `c_lgbm.py` | 3656 | `c569a351dd6dfe8392f6a617b9e731777345591943a0a9cb6350d43620cf868b` |
| `c_protocol.py` | 4573 | `f4d7015f2b7fcce1898a0a7e0d557d6936176c8edd442572910989ff9bf71810` |
| `c_report.py` | 24038 | `407ff869d7983aba5802cb1053d610d03c0ae7dd4bf3927fcaf1b4f822ec9416` |
| `c_repro.py` | 2197 | `37f03381bd4a861de97afdd6047a139f6aef68c9024c4e0c893bda9710e5c1ff` |
| `c_super.py` | 1413 | `985b175b012937438835c6284a0e658fd06c4e5c9e00182b36b2659e20f7309c` |
| `c_tf.py` | 10152 | `f2b7b9581cd3da68f8cd5664fdea139f345e032f05c18dbe6590ca233c0abe05` |
| `c_wf.py` | 5067 | `642a26e0473332924f86a46a2d119c28988d7f37b0e99410516417f4a553785a` |
| `p2lib.py` | 7692 | `01f7598aa3ae9118a734cddd3ab46260b6353bf2d27dbed4a4ebd793544f8af4` |
| `smoke_dashboard_p2.py` | 2297 | `68329f11be5f7867e620ab2a7ad51ffe009137decdf82ef77a72303c93595c2b` |
| `stock_dashboard_p2.py` | 12484 | `a8fbeb030c070e7c8d54b0d4348c56bdd96ee028ec6aa2cc1f6f76eca0f14129` |
| `c_regress.ps1` | 724 | `c019bdb6e0d1f1a69fbf649fb8c80c6b20d5bfdd63028e876274a35a8271a0fe` |
| `run_dashboard_p2.ps1` | 769 | `71ce7d97c59d0af99c1fc671c0260b222cec3f1f3bfd00312366aa06919ace12` |
| `a_verify.json` | 12371 | `fca2c885abb57d42e341402074c94bcb4099550c7bf265a5fc171d94026e544e` |
| `a_verify.md` | 5459 | `f8528fc3c8407f48ac43f4179b9ffc461144cb1547b1c42a8d80e7051b108660` |
| `results/a_verify_train_meta.csv` | 25733 | `defe633f3671382c296cd207a300ea86382d1f3b88ce8112e22c24b41ee8b510` |
| `results/b_controls_calibration.csv` | 11391 | `3fbb3bf1614ac3d8b7b8c4633e1948be95c3fb7b2d770b5c64846cbe9e0429d8` |
| `results/b_cost_stress.csv` | 3474 | `a04ed7d6444b90b5cf4bef70c2f3882c04a999a8e2234a921d7a111593520e60` |
| `results/b_ew_benchmark.csv` | 12917 | `36a47007c101fddf07b8e92fbc062a10756b352b4fab1d44edfbfce41ade9f23` |
| `results/b_exposures.csv` | 1711 | `693fcba07915d61f06c7774c497007686af0b5b4cdc5131fc7e657373e3faa4e` |
| `results/b_factor_alpha.csv` | 66252 | `0a3801dc471b644a63494644d42651b409cf088e38d077a9aebe4a6bdeb5913a` |
| `results/b_fill_diagnostics.csv` | 12401 | `77af14e87519cf4efdb5337fa08eb72a56586790506dfd69c37d447289c1537e` |
| `results/b_grid_fixed_params.csv` | 433345 | `9873a922858f1195ce4bc58c99a07c23ba2460d50cf7203801f7fbbaee98d31a` |
| `results/b_nested_optimism_check.csv` | 4759 | `984a53cc33003cf474a6243c5a9a7fb4488f79d729a8fcde05c6559d07ad3def` |
| `results/b_nested_selection.csv` | 19081 | `8591aa760f87d853cbd587312cf7df07b6acd2d21ff2787103f88c892270b949` |
| `results/b_paired.csv` | 64968 | `454b97c60945785648025b3bc7f1076e472a691e2241da6945806bfd2895768c` |
| `results/b_performance.csv` | 619157 | `7ba522bde24ad5f59b84bc93bc272a49f7baf88a46460efec11db7e4f0c98e95` |
| `results/b_reconcile_A0.csv` | 1337 | `5d21e79bf17d4d11ddba7e84249b793c005a92f8f3d217e7197ad5d2883c8f28` |
| `results/b_turnover_vs_prediction.csv` | 41240 | `67c7c9468b29969555b484f8339f051e5d4d0795953f8f2083d3d73ae5d1532e` |
| `results/baseline_paired_ic.csv` | 7458 | `0bb92f914788586c551e989502cabc226e559bac9dbdac4876824dae7e0fc2ca` |
| `results/c_block_meta.csv` | 34035 | `6919123e806e5b3ed37da84c5c3aa7c756d73c779ee20b9ec6b9502c48456258` |
| `results/c_coverage.csv` | 698 | `2238d3ba010ced77a5940b1d7d20b0577355e7f434794538f30d0cb467da7406` |
| `results/c_head_rankic.csv` | 517 | `adebec3b84b6e3756076bca99d6f80acc8a77c81f8b2af3bf02272d4f101f568` |
| `results/c_oracle_diag.csv` | 3870 | `3cbe52dab887eaa010f2497cbb48047db04a29c223dc1d8f098e2d5f3471fac8` |
| `results/c_paired.csv` | 70699 | `05e1f0f3e84dd015a15454ca237b952dc5cfdac21c78f6805675b26e5ab18437` |
| `results/c_posthoc.csv` | 16377 | `cf6758dabee2816193f80360756d1f0e3520307619adbd4c76965feb4f4b83d7` |
| `results/c_reconcile.csv` | 1492 | `0e101c43aaa327bcb1ba1a762598361b6d1c312da4333c4b028f48936abf905f` |
| `results/c_runs.csv` | 24776 | `140d319c53d304c9edbce0ce5405ca4f7c67adfefa14feddee91d36cc858dc86` |
| `results/c_seed_variance.csv` | 15916 | `5ad9d9e8a8fafe60115e0ca9274ace482ecd6dcd028a5fe2bed5f4c84d25ea7a` |
| `results/decile_means.csv` | 15215 | `d9868a618f391895b54a46a88f64e18911ec2126ff0b136f59473e35674d2faa` |
| `results/decile_spread_by_period.csv` | 27985 | `4ac764d8f5adb53c92feacaf82016529cce5d06ea8e3e14707c7c69c77c33531` |
| `results/decile_summary.csv` | 11961 | `0729916258c47eb2f12426677793a0052d322fb60e046139608f072ec70d4e2c` |
| `results/execution_by_decile.csv` | 9234 | `57e092e407034c0226f756a1fe85203cffa11d3cc66387907614ddb211f54298` |
| `results/factor_attribution.csv` | 19319 | `a3b7ff6686654d19908d9361b088d213928f776e8b2494e2de73f087c52ea4a6` |
| `results/factor_returns_by_date.csv` | 23235 | `3ffbb2b192e5042eb8b51902f8c040a95479b7f6bb1f726ad43852e2584f7d2b` |
| `results/group_completeness.csv` | 190 | `e18db6ac6da6e175949cc870ec00646873a238a04dfda296b9ffe4003f0c4e08` |
| `results/partial_rankic.csv` | 9747 | `12b5edc90a260fdfc201eb66aaac5962add14b5fd558f8e1a1f8b160cfad0353` |
| `results/partial_rankic_baselines.csv` | 1163 | `cad56de20c42e8fc69795320bd1bb772eea5ac1fe72bc0b4767fadbf5b52c5ba` |
| `results/partial_rankic_by_date.csv` | 610393 | `14c81ce094f527540c59c5857ec9319acaa6f041086019d8d16355ef88152340` |
| `results/rankic_by_date.csv` | 399035 | `054d69441f9a64c287dba275fd0a0eb1dd96fd941fc813cec0dd46af68e5fb79` |
| `results/rankic_by_group.csv` | 38479 | `3f90c90bd620aa624698ee09efb7f6898e04ff818f192b2eefd2b893a11cbeac` |
| `results/rankic_clean_timing_rows.csv` | 1210 | `db31be0738e762bccd5134ece50676a7e71e7107df8ef2350b75c939ea49ff91` |
| `results/rankic_old_minus_new_paired.csv` | 1101 | `ae3dad4bf6d52c96d5a843e6a90302d8693b79caed88f580e5275d2f2bef7c7f` |
| `results/rankic_old_vs_new.csv` | 6984 | `d9645e666a0cd192ac895ca927a79c02bbdce48e4fdf9c8cec4cd8fb1219c09d` |
| `results/score_corr_with_baselines.csv` | 1402 | `92f3c9e9387b2b8bcc5b2d3e254c91fc6cccccd874b4668a216f0e319f116ff8` |
| `results/subset_robustness.csv` | 17986 | `971bff135cf4b7885e36367473650bd98fb6a0f65590866d37a90810cb7de13f` |
| `results/timing_diagnostic.csv` | 3797 | `b11affb3b019051383c718b7094b79d1f087ec3a3eddc977b6488296746fc92d` |
| `results/b_protocol.json` | 4508 | `9a2d771fabe9e0bdbe9e06a218135fdc8950de56f3243da0d72460f432983b80` |
| `results/c_protocol.json` | 5607 | `3d287bcf4238157dc56c6faeea3de92223135ab0186243efc3e7a0341d8b9521` |
| `results/c_repro_r000.json` | 305 | `342dd86c5ed6566dd13fb1a0d65a20f515474d60eb1c621c1428c0708fcde04a` |
| `results/eval_meta.json` | 121 | `9e3c8bf1a632e0ef1824bac2d549e82daec94ab12d5aa96ea80f7a74eea74682` |
| `results/p2a_panel_meta.json` | 451 | `e76df248548e39f8f949e001271b5c02113c115cf28646ff1e88fea2ff0f6d55` |
| `results/a_eval.log` | 110 | `bca015a7d61689f18619b386875c31429d3b2a841316db59c0ca233cc8b60c6c` |
| `results/a_figs.log` | 735 | `b3381dd866d3b2a3575c53a923d95eaeaef6676402da7aac5f38cd748de7ac05` |
| `results/a_panel.log` | 453 | `a3a808c7444ca358fc9ac2bcad405585713c5102e5ac5915706c0c63eadcf11c` |
| `results/a_verify.log` | 837 | `63671afa5f84105ce66c696c226603866873101857cd9917546d9d384f8dcee5` |
| `results/b_launch.log` | 1808 | `88888ebf3563910f341f04e147a36a4bd2b6616c3d314d28245ca4ec1ac9ea80` |
| `results/b_launch2.log` | 253 | `b6af69d389b915fd63e51adc6a6aa263a4f86442025fec5a0b5969a1d1ec630a` |
| `results/c_run.log` | 33991 | `8831b3db651f0a08205e763ad67c5fbb38a6e51fba64349c2699c572c07ff3e8` |
| `results/c_super.log` | 80 | `8102620d5d1547eebede2c3f031fb3ecc85f1661fd867e8ee9a7a32386805691` |
| `results/b_runs/GRU_C1_flat.parquet` | 538348 | `1c967ac8f0cd91b2e79bd96148fb90488a2713ebf889d0ffeb97eaaeee2de2b7` |
| `results/b_runs/GRU_C1_flat_extras.json` | 18706 | `6bae521dbd0a2054602894b3d445ecd20be6206433f775d1c74af9a9680737ad` |
| `results/b_runs/GRU_C2_explicit.parquet` | 720389 | `75d85417749f4831893aab75cf82f36fa09a1e438c47b5a16b7b9c4047eab53f` |
| `results/b_runs/GRU_C2_explicit_extras.json` | 18599 | `4835c0c8d1d58bb06b123ee7d8516d906c73681f1b965a789de563762f5ce140` |
| `results/b_runs/GRU_C3_realistic.parquet` | 718557 | `fd5bb1d805f0c4b9ccff899667377ce1b7a10c4b5509cf78cf1ea54ac1ead450` |
| `results/b_runs/GRU_C3_realistic_extras.json` | 18640 | `0c5e3c85cfd912f231dedf2302a07192ec15b304f05e46e6a5fbbbe799f082c5` |
| `results/b_runs/GRU_C4_capacity_1e8.parquet` | 709654 | `69b72b38cde6c27571d4238eb8028db54426aa4d8dcad317f23730146012829e` |
| `results/b_runs/GRU_C4_capacity_1e8_extras.json` | 18745 | `c51511c1a6785b63c262ec1d60c7796f0a5dd142989702bdb2997efa641a4ed7` |
| `results/b_runs/HGB_C1_flat.parquet` | 537313 | `ce0581230ed13192fcc0ab9078acc27b7bdda19e1d4bcb43eeeed4f8eae709ce` |
| `results/b_runs/HGB_C1_flat_extras.json` | 18641 | `daed889d2eb88b73b5bef80571c19c26921408f988fcfdb0227ea2f5d1ea4b35` |
| `results/b_runs/HGB_C2_explicit.parquet` | 1086153 | `dbc1f02de5b1861951cd62eb466ed7fa6ae6ca640be803185f8047d804342b07` |
| `results/b_runs/HGB_C2_explicit_extras.json` | 18725 | `6dfc8cef15c3a40e6f02cfc62eb4f37c1a6207957d43bec76d6c2de41d29070d` |
| `results/b_runs/HGB_C3_realistic.parquet` | 717154 | `afdd7bb211704ef5ccd6d0b550567f67f75f1e44bf9028e12c8ea32a03dbaee8` |
| `results/b_runs/HGB_C3_realistic_extras.json` | 18768 | `14f58ba12b58b674f05d11ed0781820edaa444a5a0a8d0392334fa61876f7663` |
| `results/b_runs/HGB_C4_capacity_1e8.parquet` | 715129 | `334f571caf4501671c8d8754c2c80cbff7ccfe53efbde9880c18d12912d2f4da` |
| `results/b_runs/HGB_C4_capacity_1e8_extras.json` | 18750 | `9dbc93132156b03460db98fb48cf9d30430c0d89cbcb7fc2f21891163c4ed514` |
| `results/b_runs/LGBM_C1_flat.parquet` | 537958 | `b4868d19f699e584f1e59e14b94c3b5bfcb0032d46755abf928f0bc848ec6a4c` |
| `results/b_runs/LGBM_C1_flat_extras.json` | 18658 | `ed443ed509f478390c3f57ff615ad034da98ef61c31b192cf3cbaf318bbf91cc` |
| `results/b_runs/LGBM_C2_explicit.parquet` | 1085919 | `d1fbf8e4bab0437f1a2993d05366fb9af0fe5f330c10fd3199ca81a1549cbd5e` |
| `results/b_runs/LGBM_C2_explicit_extras.json` | 18710 | `691d3c6b9a691dea8c78e62511e106f6e8a8d677661941422397db449aa0aafa` |
| `results/b_runs/LGBM_C3_realistic.parquet` | 698688 | `19a086aced1f742d213b19a45505f5f9d26bb25275e2f1378a26d53158c30f6f` |
| `results/b_runs/LGBM_C3_realistic_extras.json` | 18696 | `0e457e6acb8cfafb5806982bcc920c40f025fd7fba25ccb6cc7fe64fc7787079` |
| `results/b_runs/LGBM_C4_capacity_1e8.parquet` | 711057 | `a2510f525d94fd1df50d4cba51fc9fd99f1b48260c7df75f3a987091f63080e9` |
| `results/b_runs/LGBM_C4_capacity_1e8_extras.json` | 18706 | `e9930be9062345d69c86492008a5f3183b574e509cfaec565449b646c8bc85e1` |
| `results/b_runs/log_GRU_C4_capacity_1e8.txt` | 295 | `51b9f84b04834a5cdd24d025011dcc32528677f439f0d66f684fac029de1831b` |
| `results/b_runs/log_MLP_C3_realistic.txt` | 291 | `93c013f9d5646728fc9bb193810a22ddbb8e82066e627adcbb6a081f0558271b` |
| `results/b_runs/log_MLP_C4_capacity_1e8.txt` | 296 | `d8ad547fc7cfc108ccf34d9f8f5c1d4ce8449810b37432206f13ab580e022a29` |
| `results/b_runs/log_Ridge_C3_realistic.txt` | 293 | `2f719fe1be7eb5ae652d59174d753745a9c542b026277e234beb059aed29d932` |
| `results/b_runs/log_Ridge_C4_capacity_1e8.txt` | 296 | `b42b97ffa4e52154b471e8431d25f8f7dff6c6bec2f742ce74d66a9267032a77` |
| `results/b_runs/log_TF_C3_realistic.txt` | 291 | `01290db15007bb1ee35652b24119292fec7bbb25dd942fac983bce0ed355f9f4` |
| `results/b_runs/log_TF_C4_capacity_1e8.txt` | 294 | `211e59c3c15a69b0f1544f428014a907fd5da0d426314e01d289d8c4bba7c03d` |
| `results/b_runs/MLP_C1_flat.parquet` | 531947 | `df6caa590fb25edcd0bb94af0ef2e08dc0445a2c301b1ac19c98f8446ac16281` |
| `results/b_runs/MLP_C1_flat_extras.json` | 18533 | `7961a63821f21efe163fafba238c77a42e0a764314c93e37eecdc7b605b2cc8f` |
| `results/b_runs/MLP_C2_explicit.parquet` | 714006 | `0f3faa123fbd7fb376359ee82c07566711fb6bc93d97bbc6e132d082235891ee` |
| `results/b_runs/MLP_C2_explicit_extras.json` | 18597 | `6c8b552fee3b67f0fdc48db980ea1fb0643c6167fa4b3f69ecf8395d9663363a` |
| `results/b_runs/MLP_C3_realistic.parquet` | 718112 | `efaa4ea249186a9a3f558e15bad193d216a384edecdfbef03d038230fafd3533` |
| `results/b_runs/MLP_C3_realistic_extras.json` | 18697 | `f3628306562e02cdf52c0b34425b6419e2aa5e19c39dba47c218cd67df141d17` |
| `results/b_runs/MLP_C4_capacity_1e8.parquet` | 719547 | `7d1ddd4029188a0e87e9b2b934c88bd416a9965c808d3351df204a28facf6be0` |
| `results/b_runs/MLP_C4_capacity_1e8_extras.json` | 18739 | `80e534fbd5d282ca59c766bfd8a9ce0aa0ecafb64aa74fde07bca939ce1ac470` |
| `results/b_runs/Ridge_C1_flat.parquet` | 529951 | `63e8112763abca7c40c262fa5583282e6c7d6b2d7c0edc7c19ba09d6cabacbd5` |
| `results/b_runs/Ridge_C1_flat_extras.json` | 18674 | `9bf793e71b109ed064b952ef5097c85e0ead970d1145a7ad086109ac0fc4da46` |
| `results/b_runs/Ridge_C2_explicit.parquet` | 713509 | `2436064a440638da4ddcbdbc39babd6b978189788b79887345c000f5da3c64a8` |
| `results/b_runs/Ridge_C2_explicit_extras.json` | 18696 | `dd679138020367f06f9e6c4a49522d3de04a685de2273dcf4ea55c46113c5c29` |
| `results/b_runs/Ridge_C3_realistic.parquet` | 711978 | `290a8e875a2475b9d1b1b4d2c8e99564247e085768ceeb2c3b05464c9226b24b` |
| `results/b_runs/Ridge_C3_realistic_extras.json` | 18693 | `5ab3d29a5cbe637305819c35bc530d7cad74f7876ede1f908532a8a485f35137` |
| `results/b_runs/Ridge_C4_capacity_1e8.parquet` | 703376 | `cbb11467ff92ee23eeac44b2963e9ad45df246ac989078a3c75cd05da338457a` |
| `results/b_runs/Ridge_C4_capacity_1e8_extras.json` | 18689 | `0bb9f355c0b5cc2e2978e7c3df5c78b7cab857b432e653b9dd345e7f02c9d096` |
| `results/b_runs/TF_C1_flat.parquet` | 537399 | `c0af3fc5f95c5b0d7da3dcd2075b3a3718363e8e39f5f8f30dbe5051b9240fc4` |
| `results/b_runs/TF_C1_flat_extras.json` | 18646 | `8805d96b5fbc84f100dc743d8345627b1da35dbc35fd814e14144fe2284d6440` |
| `results/b_runs/TF_C2_explicit.parquet` | 718251 | `c05bbd296574b4a8963784e85b6df27eb828f8020f054463f1cccc72e32dbf73` |
| `results/b_runs/TF_C2_explicit_extras.json` | 18601 | `7d2e41f62dfec5c3da9f1aafa870dc77644df6b4b0d530883cb22d14c1618875` |
| `results/b_runs/TF_C3_realistic.parquet` | 716685 | `5ffd7c4115c90375422562ce0f6e433885fa8b6c0df36b17820d32b93309639d` |
| `results/b_runs/TF_C3_realistic_extras.json` | 18704 | `69c62b37b4c595a963a3b999eecc9f13d0b691dc03f47710003f9e669c1554ee` |
| `results/b_runs/TF_C4_capacity_1e8.parquet` | 720321 | `2828f3d81294821c55bf8d95bd1b0f051ebd822465ec13416b357f94a661864e` |
| `results/b_runs/TF_C4_capacity_1e8_extras.json` | 18712 | `32668713bd2721ca2e93a3f31aaadf76ed2349b6f5e2fb1f390fb9f8dd5fc129` |
| `results/c_by_date.parquet` | 230490 | `4cab2344d9dc3ac09e0162bc89f21b31423f672b97bb8b0a3ae796c839d54107` |
| `results/b_logs/decisions_TF_C2_explicit_A2_nested.csv.gz` | 347884 | `827a4a6c46fa5008933567651ac65afeaac6ea78e2ddd5aae05c9869f784f267` |
| `results/b_logs/orders_TF_C2_explicit_A0.csv.gz` | 779292 | `bda57f9e849e507579f48a43e569699f204210c6e2c63a7c8047ae37781a41c7` |
| `results/b_logs/orders_TF_C2_explicit_A2_nested.csv.gz` | 671975 | `0a2da71af43afa7409c49eb11834279b257c16f4d6943e2dc1aea481f09241ba` |
| `tests/test_dashboard_p2.py` | 1365 | `5fd9fdb862dbbefca33d2ac7ea377e93eb93c52ab6e798a27cd8459fd93fee9d` |
| `tests/test_p2a.py` | 9342 | `6420f7a467e430b01de98ec1729e0b6dd6e6ce7bfebd2db820bed9076f8d0062` |
| `tests/test_p2b.py` | 19743 | `e037c67a5439093221eee9af638b23e5d19407184b392c6534e8a560dc48c8a6` |
| `tests/test_p2c.py` | 10393 | `5f25d242b8e8d1ee76b42f5725d2487c769fac74e12c4b8e31572d0e458b312f` |
| `tests/http_dashboard_p2.txt` | 62 | `763acdb7122f72892b44364834450126170b2dc9cfaa62f2617547cfa4bf7cad` |
| `tests/pytest_dashboard_p2.txt` | 103 | `02c6a22c3c231603f8b8023ede3274c4757a83bc9adb55ec29328b56bb932d76` |
| `tests/pytest_p0_rerun.txt` | 101 | `911745005243c1f051b08bc41257858a533ae104fcb1c0cb08bb6db4bed891d0` |
| `tests/pytest_p0_rerun_b.txt` | 204 | `87fa43533b348c055a26f86d3a55240de6e04eb434666a6a3d3459a76bf4ae99` |
| `tests/pytest_p0_rerun_c.txt` | 104 | `1e6f29a4e603537a5577f7e0a48875a6010e92c4a93d47592ec317fe7efaf33a` |
| `tests/pytest_p1_rerun.txt` | 102 | `5a2e3c731a8917f1a4a4fa611a1e81fab7455c326a9dced7b7381a5b20861aed` |
| `tests/pytest_p1_rerun_b.txt` | 226 | `e2767be6339b560da504e4bee6a0c6899274e678f17604a9ccf00b1866142162` |
| `tests/pytest_p1_rerun_c.txt` | 105 | `52a7f49014df88c592b60a769143ea5c37909b15cd4f5aaeb2420f9863086683` |
| `tests/pytest_p2a.txt` | 102 | `885f27288e6bb11632147bd9fadfdcfb1af48d8cc55689bc2d22e438ae85172f` |
| `tests/pytest_p2a_rerun_b.txt` | 206 | `f6db47ef8a0c0269b677bc90939f2df415057bfdde2eb6a3d898e2ee3531baae` |
| `tests/pytest_p2a_rerun_c.txt` | 105 | `99884a096099e7091e0610252af442c9478c49b2fb2daf45e7337742c6582765` |
| `tests/pytest_p2b.txt` | 206 | `e9c5bdfbda980728ec90fe8445024e0d15604e0141856c439d26c43f7820ddd6` |
| `tests/pytest_p2b_rerun_c.txt` | 105 | `597bb7007fae299e3cd1636c98b275aba8857b098419f367ea826bd7794b95a6` |
| `tests/pytest_p2c.txt` | 104 | `92337ea76eee927a18c4d5e4a3147b92d7478558beef988d7043065e66184ad7` |
| `tests/smoke_dashboard_p2.txt` | 1488 | `f87a3e4f1ca7561ab3c506c071ec423f772bf67bb010b8e9a5e9861f20cc8e19` |
| `figures/_figure_check.json` | 326 | `81d74adff75979f00a6ec84e0b3bbb863b0e1289e98dc5df60acac5610f88547` |
| `figures/b_fig1_nav.png` | 328193 | `b48678838c6e25a9d7dc1e0bcc0843d45fbc071dc6f3c9d457e9955a9b53cd8c` |
| `figures/b_fig2_cost_decomposition.png` | 59938 | `f43f3c750ce631b04cdabfad54cc20101f17def2bb14646889860d3b58cbef32` |
| `figures/b_fig3_turnover_vs_net.png` | 129613 | `3f385a514c275e1884d87258f9d878d0d90c059d36d11dfd30e9f719931e4b8c` |
| `figures/b_fig4_attribution.png` | 121015 | `3a4dbdd7014ec8bdb23da25ae13a49b34ffecd2f2baefc23ab3b592e3a49262c` |
| `figures/b_fig5_paired_forest.png` | 61226 | `f83b000bb202323e72fa9edca054b1207eee1349470543bca4515f74fc9dab1f` |
| `figures/b_fig6_factor_alpha_t.png` | 59254 | `18b590975b6b963b477d23584375e2926696be30b8153bfdc7f51c1ba9e9801a` |
| `figures/b_fig7_constraints_capacity.png` | 56277 | `c1f832fedbc991452be8d7af724927fc5e5244ea40d69d6fbe68eb5c3b061089` |
| `figures/b_fig8_cost_stress.png` | 44983 | `46109f25ad747e00fc5fff7be9bc3f538f6176df50a9e1457bb00a94b5d10516` |
| `figures/c_fig1_seed_variance.png` | 126246 | `2706c2dcd5e3db4e6ea8583714bedba59c66228a69e5c5d4ed18c5556d2b4a43` |
| `figures/c_fig2_variant_ladder.png` | 63545 | `2599dd4989efbe537f2ba2a7bbbda38240a68681bcc18a28c5a4f949628a44b9` |
| `figures/c_fig3_paired_forest.png` | 43382 | `1ce949a0dd39aa06db66aa20be9db87123a82ab668f4458718f74b870f4d774d` |
| `figures/c_fig4_stability_time.png` | 201737 | `d3d68cc92630f63ee26c429614ca643ac6713e1122498fa8259df9d822a5e9ca` |
| `figures/c_fig5_oracle_posthoc.png` | 103735 | `cf1c3f00e868eca8636b9c13cfa917d994c0ed3bd5a8a2149c63b43546f7da79` |
| `figures/c_fig6_lag_vs_tf.png` | 128125 | `dcaa2e905bed5643fab40d87bf7941f9a17edc86871c5ec35b168088dcbdc99a` |
| `figures/decile_returns.png` | 85162 | `afd55a666a50e61505f7b48d66cec7198a7ee1cf17ee0cc79ed4bf82fb05ca3b` |
| `figures/decile_spread_by_halfyear.png` | 150480 | `f51916492c561559a0f0ade9ab51167b3b61e21a12df3f374c1221438e549f6a` |
| `figures/factor_attribution.png` | 96183 | `e8387e08b94d505fe741d5ddca40a84c4b7dc80e0a6704666200f00cb72023d8` |
| `figures/ic_by_industry_latest.png` | 44043 | `191dd43734229d42e56317663015e464227234386ed5f6f62c6c1677eadde5ec` |
| `figures/ic_by_liquidity.png` | 61752 | `6433f8946d417802ee7019cfeccfcc4afe2e55d1278ac05c653fcaf2227926bc` |
| `figures/ic_by_vol_exchange.png` | 50221 | `30858ca34abbb1044eb0fcfa0b3876fd7c30ffaf88da4a85d390654df4a77a76` |
| `figures/partial_ic.png` | 58536 | `16d0928ad17a107b0b806516c13ad10c61e116b1191ec9e8f93cf58a8938b399` |
| `figures/rankic_old_vs_new_and_baselines.png` | 62316 | `797a6eb2321b7e41a3b688a862bede5f271712f6cfe747f1caa00850d9635ece` |
| `figures/subset_robustness.png` | 93555 | `97b200a0c8991a79320138e036a9eb7bc71a7f6f241f0fa2e050addce7371556` |
| `reports/p2a_alpha_anatomy_report.md` | 49765 | `d6fc69f853c87aae807bc10bb58acc77eb05f9d36a4bccc95b0042831d8c97bc` |
| `reports/p2b_cost_aware_report.md` | 36271 | `2dd2f9378f7058841a3db854504d415f9d46bcf91b0de5635900370081430135` |
| `reports/p2c_transformer_ablation_report.md` | 33568 | `de3e1a30c9820a9e9f980d3db50f27241809fb157e2b4ba4c2d6a0951ca38ba5` |
| `results/sample_p2a_panel_3dates.csv` | 2031988 | `b7ea0e5bad149cec5f7a7b533f9d5804707bd64459ae20cb195a2aa34e03fa83` |
