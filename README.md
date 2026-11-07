# Choice Quant · A 股量化研究系统

**Choice / public-market A-share research stack.** 每天给出约 5 只重点研究股票、写清入选理由，并做前向验证。这是研究记录，不是自动交易系统。

样本大约 **1498** 只股票（分层抽样，含一定比例已退市股票），覆盖 2022 年之后的日线与可核对的财务公告。代码按阶段放在 `v2/p0` … `v2/p6_final_validation`。根目录还保留最早的教学原型（约 9 只股票），用来说明数据从哪来、回测最小闭环长什么样。

本页数字都来自仓库里已经写好的阶段报告（尤其是 `v2/p6_final_validation/reports/` 与 `v2/p5_best_top5/reports/`）。历史回测在多个阶段被反复查看，**一律当作探索性结果**，不是盲测，也不能外推成未来收益。

---

## 已上传的报告与数据

上一轮精简上传已经带上大部分 P4–P6 的 markdown 报告、worklog 和 review bundle，例如 `v2/p4_top5_quant/reports/`、`v2/p4_top5_quant/worklog.md`、`v2/p4_top5_quant/p4_review_bundle.zip`，以及 P5、P6 下同名的 `reports/`、`worklog.md`、`p5_review_bundle.zip`、`p6_review_bundle.zip`。这次只补当时缺的明细 CSV 和 P1 复核包，约 18 MB。`predictions_registry` / `forward_registry` 里已冻结的预测行保持原样。

本次新增路径：

- `v2/p1_dl/review_bundle.zip`
- `v2/p2_alpha_monetization/results/b_controls_calibration.csv`
- `v2/p3_ai_report/results/calibration_by_year.csv`
- `v2/p3_ai_report/results/calibration_supplement.csv`
- `v2/p3_ai_report/results/calibration_table.csv`
- `v2/p4_top5_quant/results/bt_holdings_e0_e1.csv`
- `v2/p4_top5_quant/results/bt_holdings_e2.csv`
- `v2/p4_top5_quant/results/bt_holdings_sm.csv`
- `v2/p4_top5_quant/results/bt_summary_e0_e1.csv`
- `v2/p4_top5_quant/results/bt_summary_e2.csv`
- `v2/p4_top5_quant/results/bt_summary_sm.csv`
- `v2/p4_top5_quant/results/buy_gate_e2.csv`
- `v2/p4_top5_quant/results/buy_gate_sm.csv`
- `v2/p4_top5_quant/results/p4_latest_top5_all.csv`
- `v2/p4_top5_quant/results/s4_calibration.csv`
- `v2/p5_best_top5/results/b5_buy_gate.csv`
- `v2/p6_final_validation/results/b1_reconciliation.csv`
- `v2/p6_final_validation/results/buy_gate.csv`
- `v2/p6_final_validation/results/current_holdings.csv`
- `v2/p6_final_validation/results/latest_top5.csv`
- `v2/p6_final_validation/results/strategy_comparison.csv`

继续留在本机、未进入本仓库的有：`.venv`、原始行情 parquet、`fin_raw/`，以及 `.pt` / `.pth` 模型权重。

---

## 请先读

- **研究排名不是买入推荐。** 模型分、`select_score`、综合分只是排序用的分数，不是上涨概率，也不是预期收益率。
- **不做真实下单，不连接券商，不读取交易账户，不使用杠杆。** 仪表盘和脚本都是只读研究界面。
- **当前没有达到预先登记的买入标准。** 见下文「关键结论」。页面上出现的股票名称，只表示「在某一套冻结规则下被排进观察名单」。
- 历史回测里的净收益、回撤、t 统计量，是在固定成本假设和已复用样本上算出来的。换一套费用、股票池或持有期，结论可以变号。

若你只想知道「现在该不该按这个名单买」：按项目自己写下的标准，**不该**。名单可以当作阅读公告和风险的起点。

---

## 这个系统每天在做什么

1. 用当时已经看得到的行情和公告，给股票池打一个研究排序。
2. 按冻结规则筛出大约 5 只重点观察股票，并写下入选理由、主要风险和数据缺口。
3. 把当天的预测写进只追加的登记表（`predictions_registry` / `forward_registry`），之后不再改历史行。
4. 等持有期结束，用实际价格核对当时的登记，看有没有跑赢股票池。到期记录同样只追加。

信号日收盘之后才能形成名单。成交假设是下一交易日开盘（停牌、开盘涨跌停会顺延），不会假设用「当天收盘后才知道的信息」在同一收盘价成交。

---

## 当前结论（给赶时间的访客）

这些句子对应 P4–P6 的正式报告，不是新的计算结果。

1. **更复杂的模型没有把 RankIC 变成扣费后的 Top5 盈利。** 树模型、MLP、GRU、Transformer 的截面排序相关（RankIC）大致在 0.07 附近，和「低波动」这类简单风格因子同一量级。换成每天只持有 5 只、再扣约 0.30% 双边成本之后，机器学习方案的净累计多为负，统计上也不显著优于股票池。P2 的成本感知组合进一步说明：正的平均 RankIC 主要来自不容易做空的那一端，多头腿扣完成本后所剩不多。
2. **B1（低波动 + 持仓缓冲，并带波动/回撤/行业约束）的历史数字可以对上账。** P6 用独立资金台账重放 P5：5 日持有、约 207 期、起始 100 万的净累计与 P5 报告同为 **+54.2%**（逐期差在浮点噪声内）。其中约 2.5 个百分点来自 002013 换股吸收合并的重估，不是引擎原来的「最后成交价」口径（该口径约为 +51.7%）。
3. **这笔改善主要来自降低换手，不是来自更强的预测。** 同一数据上拆开看：只有缓冲（A1）净累计约 +50.5%，只有约束（A2）约 −12.3%，缓冲加约束（A3，即 B1）约 +54.2%。A3 相对 A1 的配对检验不显著。换手从大约 0.39 降到大约 0.08，费用从大约 24% 降到大约 5%。
4. **相对股票池的超额仍然不显著。** B1 相对合格池的超额 t 约为 0.19，自助区间含 0。绝对净值变好，不等于跑赢了股票池。
5. **预先登记的买入标准当前全部不通过。** P5 的六条标准要同时满足（回测 t、置信区间、多数年份为正、回撤、足够的前向到期期数、前向超额）。写报告时前向到期期数为 0。因此最新名单的标签是「研究候选，未达到买入条件」。

P6 在 **2026-10-23** 冻结的 B1 研究 Top5 为：爱尔眼科（300015）、三峡能源（600905）、双汇发展（000895）、中国交建（601800）、光大银行（601818）。这是观察名单，不是下单清单。更早一天（10-22）的登记没有被改写。

更完整的表在：

- `v2/p6_final_validation/reports/p6_final_report.md`
- `v2/p6_final_validation/reports/p6_batch1_report.md`
- `v2/p5_best_top5/reports/p5_final_report.md`
- `v2/p4_top5_quant/reports/p4_model_comparison_report.md`
- `v2/p2_alpha_monetization/reports/p2b_cost_aware_report.md`
- `v2/p1_dl/dl/dl_models_report.md`
- `v2/p0/p0_audit_report.md`

---

## 目录怎么读

仓库根目录是 v0.1 教学原型。真正按阶段推进的研究都在 `v2/`。

| 路径 | 一句话 |
|---|---|
| `collect.py` / `model_backtest.py` / `choice_import.py` | 最小原型：公开日线或 Choice 导出 CSV，滚动梯度提升，样本外回测。股票池示例只有约 9 只，不能代表全市场。 |
| `universe.csv` / `choice_modules.csv` | 原型用的代码列表示例，以及 Choice 界面模块名称备忘。板块代码不是股票代码。 |
| `tests/` | 原型回测的基本断言。 |
| `v2/data`、`v2/docs`、`v2/reports` | 抽样股票池、行业表、数据字典、早期质量报告。日线大文件不在本仓库。 |
| `v2/p0` | 审计并修复前视、复权伪影、成交与退市处理；同制度比较后，修复版模型没有显著净超额。 |
| `v2/p1_dl` | 1498 只股票的逐股分数、历史预测总表，以及 MLP / GRU / Transformer 与梯度提升的对照。 |
| `v2/p2_alpha_monetization` | 解释「有 RankIC」为什么还变不成扣费后的组合收益；含成本、换手和 Transformer 消融。 |
| `v2/p3_ai_report` | 每日研究简报：冻结模型打分、点时财务、数据质量，以及只追加的预测登记。 |
| `v2/p4_top5_quant` | 把研究目标收成 Top5：排序学习、低波动对照、行业/波动/回撤约束和持仓缓冲。 |
| `v2/p5_best_top5` | 在同一回测引擎上比较 B0–B3，并预先写下六条买入标准；当时全部不通过。 |
| `v2/p6_final_validation` | 独立对账 B1，把收益拆成「缓冲」和「约束」，并维持 8506 研究看板与前向登记。 |

各阶段报告在对应目录的 `reports/`。规则冻结文件例如 `v2/p6_final_validation/selection_rules_p6_B1_v1.json` 与 `v2/p5_best_top5/buy_criterion_p5_v1.json`。

---

## 各阶段在回答的问题

**P0 · 数字能不能信。** 旧原型里一度出现过更高的年化，复算后发现净超额并不显著，前复权伪影还会把结论方向带偏。P0 改成信号日当时可知的股票池、原始价成交、公司行为逐笔记录，并用故意写错的前视用例做反测试。结论是：工程修复可以通过，但修复后的模型相对同制度等权没有可交易的净超额。

**P1 · 复杂网络有没有多挖出信号。** 在同一 1498 只股票、同一 207 个五日信号、同一成本和回测器上，比较梯度提升、MLP、GRU 和小 Transformer。序列模型没有稳定超过树模型；RankIC 略高也不等于 Top 组合扣费后更赚。看板在本机 `8501`。

**P2 · 排序相关能不能变成钱。** 低波动和三因子基线的 RankIC 并不低于这些模型。多头端的价差和往返成本差不多。把换手降下来，净收益会好看很多，但和「随机少交易」对照之后，改善主要是少付费用，因子调整后没有显著的净 alpha。看板在 `8502`。

**P3 · 怎么写成每天能读的研究报告。** 不重训模型。用冻结权重给最新交易日打分，财务只用公告日之后才可用的报表，输出大约 5 只观察股、风险提示和「历史同类股票」的经验频率。预测写入 `predictions_registry`，文件哈希记在 manifest 里。看板在 `8503`。

**P4 · 如果每天只留 5 只会怎样。** 同一引擎比较多种 LightGBM / 排序损失、低波动因子，以及「约束 + 缓冲」。机器学习 Top5 扣费后仍明显弱于简单低波动；S4 一类混合规则也没有证明稳定跑赢股票池。看板在 `8504`。

**P5 · 哪一套规则最值得继续看，以及什么叫「可以买」。** B1 是低波动再加约束和缓冲，历史净累计最高，但相对股票池的 t 值仍然很小。六条买入标准在计算前写进 `buy_criterion_p5_v1.json`。八个「方案 × 期限」组合全部标记为不构成买入依据。看板在 `8505`。

**P6 · 54% 是不是算错了，以及现在的 5 只是谁。** 独立复现确认 B1 的 +54.2% 可以逐期对上；拆解说明主因是缓冲降低换手。相对股票池超额仍不显著，买入标准仍不通过。最新研究名单与规则、前向表放在本目录。看板在 `8506`。

---

## 如何在本机运行

开发机原路径是 `D:\choice_quant_starter`（Windows，PowerShell）。本仓库是其中去掉虚拟环境、原始行情和大权重之后的副本，目录结构与本机一致。

### 1. 虚拟环境与教学原型

根目录 `requirements.txt` 只覆盖原型脚本（pandas、numpy、scikit-learn、akshare、pytest、Excel 读写）。看板还需要本机 `.venv` 里已经装过的 **streamlit**（日志里为 1.66）以及作图依赖。重训深度学习才需要 `.venv_cuda` 里的 PyTorch，浏览已有报告不需要 GPU。

```powershell
cd D:\choice_quant_starter
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install streamlit
python -m pytest -q
```

原型闭环（会访问公开行情接口；接口不可用时改用你自己有权导出的 Choice CSV）：

```powershell
python collect.py --start 20220101 --end 20261016
python model_backtest.py
python choice_import.py data\choice\ChoiceDaily.csv
```

Linux / macOS 把激活命令换成 `python3 -m venv .venv` 与 `source .venv/bin/activate`。仪表盘脚本是 PowerShell，里面写死了 `D:\choice_quant_starter\.venv\Scripts\python.exe`。在其他路径或其他系统上，请改脚本里的 `$py`，或直接用下面的 `streamlit run`。

### 2. 研究看板（只读，本机 127.0.0.1）

不要同时启动两个占用同一端口的脚本。P1 的 v1 与 v2 都使用 8501。

| 端口 | 脚本 | 页面 |
|---|---|---|
| 8501 | `v2/p1_dl/run_dashboard.ps1` | 六模型逐股研究看板（v1 为 `run_dashboard_v1.ps1`） |
| 8502 | `v2/p2_alpha_monetization/run_dashboard_p2.ps1` | P2 成本与阿尔法诊断 |
| 8503 | `v2/p3_ai_report/run_dashboard_p3.ps1` | 每日 AI 研究简报 |
| 8504 | `v2/p4_top5_quant/run_dashboard_p4.ps1` | Top5 模型与规则对比 |
| 8505 | `v2/p5_best_top5/run_dashboard_p5.ps1` | B0–B3、五股卡片、买入标准 |
| 8506 | `v2/p6_final_validation/run_dashboard_p6.ps1` | B1 对账、归因、最新研究 Top5、前向是否到期 |

当前阶段入口：

```powershell
cd D:\choice_quant_starter
.\v2\p6_final_validation\run_dashboard_p6.ps1
```

浏览器打开 `http://127.0.0.1:8506`。停止用 Ctrl+C。`-NoBrowser` 可以不自动弹出浏览器。

若 Python 路径不同，在 `v2/p6_final_validation/dashboard` 下等价于：

```powershell
$env:PYTHONIOENCODING = "utf-8"
streamlit run p6_app.py --server.address 127.0.0.1 --server.port 8506 --server.headless true
```

周度前向核验（只追加已到期的结果，不改历史登记）：

```powershell
.\v2\p6_final_validation\run_weekly_verify_p6.ps1
```

P3–P5 分别有 `run_weekly_verify.ps1`、`run_weekly_verify_p4.ps1`、`run_weekly_verify_p5.ps1`。P6 的周核验会先跑上游已有的核对，再写自己的到期行。

重训、重抓全市场日线、重拉财务原始表，依赖下面「未上传内容」。只阅读 `reports/` 和已入库的 CSV，不需要这些大文件。

---

## 数据、时点与合规

**点时财务。** 利润表、资产负债表按报告期抓取，但特征值要等到公告日之后的第一个交易日才能用。P5 审计过 2026 年三季报：10 月 21 日实际使用的最大公告日是 10 月 20 日，公告日当天及之后的报表没有进入当天特征。缺失显示为「未取得」，不拿后验数字回填。

**禁止前视。** 训练窗只用标签已经走完的样本；标签结束日早于下一次信号日。股票池按信号日 `eligible_asof_t` 过滤，而不是用「今天仍在市」的名单回看历史。P0 有无前视单元测试，其中包括故意泄漏、必须失败的反例。随机打乱日期切分在这个项目里是错误做法。

**冻结预测登记。** 每个研究日的分数、规则文件哈希和股票名单写入 `predictions_registry` 或 `forward_registry`，manifest 记录 sha256。后续阶段明确不改已经冻结的前一日文件（例如 10 月 22 日的 P5 登记在 P6 中保持原样）。前向验证表只追加到期行。这样「当时怎么说」和「后来实际怎样」可以分开核对。

**数据从哪来，什么不能做。**

- 优先使用你自己账号已获授权的 Choice 导出（数据浏览器、历史成分、Excel）。导出许可不允许的字段不要进仓库，也不要进公开结果。
- 没有 Choice 权限时，原型和后续增量使用公开接口（AKShare，以及项目里记录过的腾讯 / 新浪日线备用源）。请自行限速，不要绕过登录或重放私有接口。
- Choice 板块代码（如截图里的 `861226`）不是六位股票代码。DDE、暗盘资金、增仓排名是供应商算法，不等于交易所逐笔资金流；没有历史权限就不要编造历史特征，更不要把今天的资金流倒填到过去。
- 前复权序列会因后来的分红送转被重算。严谨比较使用点时可得的公司行为；P0 记录了旧加法式前复权会污染特征和收益。
- 行业分类若没有历史成员生效日，长期回测存在幸存者偏差和行业口径偏差。样本含明显高于全市场的退市股比例，结论只代表这个样本。
- 成本是研究假设（P0 之后常见口径为双边约 0.30%，含佣金、印花税和粗估冲击），不是某家券商的真实费率。停牌、涨跌停、整手、流动性参与率只做了部分近似。

---

## 这个仓库里没有什么

精简包来自本机 `D:\choice_quant_starter`，上传前已排除体积大或不能公开再分发的内容。克隆之后**不能**直接重训或重跑全历史回测，除非你在本机准备好下列目录：

| 未入库内容 | 本机大致位置 | 用途 |
|---|---|---|
| `.venv` / `.venv_cuda` | 仓库根目录 | 解释器与已安装包；CUDA 环境只为 GRU / Transformer 训练 |
| 原始日线缓存 | `v2/data/raw_parquet/`、面板 parquet、`data/raw/` | 按股票缓存的 OHLCV，断点续跑和特征用 |
| `fin_raw/` | `v2/p4_top5_quant/fin_raw/` | 东方财富业绩与资产负债表原始表 |
| `preds/` | `v2/p4_top5_quant/preds/` | 各模型最新与历史预测 parquet |
| `c_checkpoints/` 与 `p1_dl/dl/checkpoints/` | P2-C 与 P1 | `*.pt` / `*.npz` 训练断点与冻结权重 |
| 其他 `*.pkl`、`*.joblib`、大 `*.parquet` | P0/P1 模型与预测总表 | 冻结的 HGB / LightGBM / Ridge 与全量分数矩阵 |

`.gitignore` 会继续排除上述类型，避免以后误提交。已入库的是脚本、测试、规则 JSON、预测登记 CSV、质量与对账表，以及各阶段 markdown 报告。报告里的结论可以读；要复现训练，需要你自己的授权数据和本机权重。

---

## 已知限制

- 2022-07 至 2026-09 这段样本在 P0 到 P6 中被反复使用，回测显著性是探索性的。真正没有被规则挑选过程看过的检验，只来自登记日之后新到期的前向记录。
- 静态或近似的 ST、涨跌停、行业分类，和交易所逐日制度并不完全一致。
- 没有模拟真实账户的最低佣金、融资融券、冲击成本的订单簿，也没有组合层面的资金曲线约束以外的容量模型（P2 的 C3/C4 只是研究情景）。
- 原型 README 时代的 9 只房地产股票只是格式示例。不要用根目录一次运行的 `exports/research_candidates.csv` 当作全市场结论。

---

## 许可与声明

代码以研究复现为目的公开。行情、财务和 Choice 导出数据的再分发以你的数据供应商协议为准；本仓库故意不包含原始行情和大表。

**本仓库全部输出均为研究记录，不构成投资建议、招揽或实盘指令。作者与贡献者不因他人依据这些排名、报告或回测数字作出的交易而承担责任。**
