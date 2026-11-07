"""P1-A3: build review_bundle.zip + README_bundle.md (no raw price detail, no 1.76M-row panel, no raw parquet, no full historical predictions)."""
import hashlib, zipfile, re, json, datetime as dt
from pathlib import Path
import pandas as pd
H = Path(__file__).resolve().parent.parent; V2 = H.parent; OUT = H / "review_bundle.zip"
sha = lambda b: hashlib.sha256(b).hexdigest()
# ---- sample of full historical predictions: last 12 signal dates, eligible rows, no raw price/share columns ----
hp = pd.read_parquet(H / "historical_predictions_all_models.parquet"); last = sorted(hp.signal_date.unique())[-12:]
smp = hp[hp.signal_date.isin(last) & hp.eligible_asof].drop(columns=["buy_price_unadj", "buy_shares", "buy_fill_date"]); tmp = H / "scores" / "_bundle_tmp"; tmp.mkdir(exist_ok=True)
sp = tmp / "historical_predictions_all_models_last12dates_eligible.csv"; smp.to_csv(sp, index=False, encoding="utf-8-sig"); print("sample rows", len(smp), last[0], last[-1], sp.stat().st_size)
DR = "dl/results"
items = [  # (source path, arcname relative to bundle root, description)
 (V2/"p0/p0_audit_report.md", "p0/p0_audit_report.md", "P0 修复/审计报告(根因、修复、修复前后对比)"),
 (V2/"p0/comparative_metrics.csv", "p0/comparative_metrics.csv", "P0 修复前后/各策略对比指标"),
 (V2/"p0/bootstrap_intervals.csv", "p0/bootstrap_intervals.csv", "P0 5期移动区块 bootstrap 区间"),
 (V2/"p0/cost_sensitivity.csv", "p0/cost_sensitivity.csv", "P0 成本敏感性"),
 (V2/"p0/audit_manifest.json", "p0/audit_manifest.json", "P0 文件 sha256 清单"),
 (V2/"p0/quality_event_log.csv", "p0/quality_event_log.csv", "P0 异常事件日志(1272条/266只;含少量原始/复权价格字段,仅为事件证据)"),
 (H/"latest_all_stocks_20261016_v2.csv", "p1_dl/latest_all_stocks_20261016_v2.csv", "2026-10-16 全部1500只样本主表(6模型分数/排名/分位+模型分歧)"),
 (H/"per_stock_summary_v2.csv", "p1_dl/per_stock_summary_v2.csv", "分个股汇总(含 DL 模型 Spearman/样本数,<30 标不可靠)"),
 (H/"sector_summary_v2.csv", "p1_dl/sector_summary_v2.csv", "分行业/交易所/板块 Rank IC 等(6模型)"),
 (H/"p1_dl_report.md", "p1_dl/p1_dl_report.md", "P1 总报告(中文)"),
 (H/"dl/dl_models_report.md", "p1_dl/dl/dl_models_report.md", "轨道B 深度学习模型报告(MLP/GRU/TF)"),
 (H/"forward_registry_20261016.csv", "p1_dl/forward_registry_20261016.csv", "前向验证冻结登记:6模型 Top50(300行)"),
 (H/"forward_registry_20261016_manifest.json", "p1_dl/forward_registry_20261016_manifest.json", "前向登记 manifest(22个关键文件 sha256 + 预先声明的评估规则)"),
 (sp, "p1_dl/historical_predictions_all_models_last12dates_eligible.csv", "完整历史预测的抽样:最近12个信号日、合格行(已去掉价格/股数列)"),
]
for m in ["mlp", "gru", "tf"]:
    for f in ["side_by_side_c30.csv", "cost_sensitivity.csv", "by_year.csv", "paired_vs_p0.csv" if m == "mlp" else "paired.csv"]: items.append((H/DR/m/f, f"p1_dl/{DR}/{m}/{f}", f"轨道B {m.upper()} 结果:{f}"))
for f, d in [("recent20_metrics_v2.csv", "最近20期逐期指标(6模型)"), ("recent20_metrics_v2.md", "最近20期明细(Markdown)"), ("recent20_summary_v2.csv", "最近20期汇总"), ("top_bottom_20261016_v2.md", "五模型 Top20/Bottom20、重合、≥3模型Top50"), ("top_bottom_20261016_v2.csv", "Top/Bottom 表(CSV)"),
             ("multi_model_top50_20261016.csv", "≥3个模型同时进入Top50的28只"), ("stock_cards_20261016_v2.md", "三只个股卡片报告 v2"), ("reconciliation_dl.md", "DL 对账报告"), ("reconciliation_dl.csv", "DL 对账明细"), ("model_comparison_v2.csv", "六模型对比汇总"),
             ("cost_sensitivity_all_models.csv", "六模型成本敏感性"), ("by_year_all_models.csv", "六模型分年度"), ("paired_tests_all_models.csv", "配对检验汇总"), ("cumulative_net_models.csv", "累计净值数据")]: items.append((H/"reports"/f, f"p1_dl/reports/{f}", d))
for f, d in [("card_600060_v2.png", "600060 海信视像卡片"), ("card_300487_v2.png", "300487 蓝晓科技卡片"), ("card_000027_v2.png", "000027 深圳能源卡片"), ("sector_rankic_hgb.png", "HGB 分行业 Rank IC"), ("rolling_rankic_models.png", "P0 三模型滚动 Rank IC"), ("cumulative_net_models.png", "六模型累计净值")]: items.append((H/"figures"/f, f"p1_dl/figures/{f}", d))
for f, d in [("tests/pytest_result_final.txt", "最终测试输出(P1 65 + P0 45)"), ("tests/test_final_a3.py", "A3 最终整合测试")]: items.append((H/f, f"p1_dl/{f.replace(chr(92), '/')}", d))
code = [(V2/"p0/p0lib.py", "p0/p0lib.py", "P0 核心库(数据/特征/标签/模拟器)"), (V2/"p0/baseline_fixed.py", "p0/baseline_fixed.py", "P0 修复版 walk-forward"), (V2/"baseline.py", "baseline.py", "make_models(Ridge/HGB/LGBM 参数)")]
for f, d in [("p1lib.py", "A1 冻结模型打分库"), ("score_latest.py", "A1 最新日打分"), ("top_bottom.py", "A1 Top/Bottom"), ("cards.py", "A1 卡片"), ("build_history.py", "A2 历史预测重放"), ("summaries.py", "A2 汇总"), ("stock_dashboard_v2.py", "仪表盘 v2"), ("run_dashboard.ps1", "仪表盘启动脚本(指向 v2)"), ("smoke_dashboard_v2.py", "仪表盘冒烟测试 v2"),
             ("dl/dl_common.py", "B 公共函数"), ("dl/dl_mlp.py", "B MLP"), ("dl/dl_seq.py", "B GRU/Transformer"), ("dl/run_mlp_wf.py", "B MLP walk-forward"), ("dl/run_seq_wf.py", "B 序列模型 walk-forward"), ("dl/dl_compare.py", "B 对比/统计检验"), ("dl/dl_compare_seq.py", "B 对比(序列)"),
             ("a3/a3_latest.py", "A3 v2 主表+Top/Bottom"), ("a3/a3_history.py", "A3 全模型历史表/对账/汇总"), ("a3/a3_cards.py", "A3 卡片 v2"), ("a3/a3_registry.py", "A3 前向登记"), ("a3/a3_report.py", "A3 报告生成"), ("a3/make_dashboard_v2.py", "A3 仪表盘 v2 生成脚本")]: code.append((H/f, "p1_dl/" + f, d))
items += code
bad = []
for p, a, d in items:
    if not p.exists(): bad.append(str(p))
assert not bad, bad
pat = re.compile(r"(api[_-]?key|secret|passwd|password|token\s*=|authorization|bearer\s)", re.I); hits = []
for p, a, d in code:
    t = p.read_text(encoding="utf-8", errors="ignore")
    for i, l in enumerate(t.splitlines(), 1):
        if pat.search(l): hits.append((a, i, l.strip()[:120]))
print("credential-like hits in code:", hits)
paths = [(a, p) for p, a, d in items]; rows = []
size_total = 0
for p, a, d in items:
    b = p.read_bytes(); rows.append((a, len(b), sha(b), d)); size_total += len(b)
now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
R = ["# review_bundle 说明(P1 轨道A/B 终稿评审包)", "", f"生成时间:{now}(Asia/Singapore)。本包供独立审计:检查数据质量修复、walk-forward 无前视、DL 模型实现、对账与统计结论。**研究排名≠买入建议;模型 score 不是概率/收益率;未下任何真实订单。**", "",
     "## 1. 一句话结论", "", "修复 P0 数据/成交问题后,HGB/LightGBM/Ridge/MLP/GRU/Transformer 六个模型在 0.30% 往返成本下的 Top50 策略,**没有任何一个相对同制度等权基准的净超额统计显著**(HAC t 均 |t|<1.7,bootstrap 含 0;配对差异均不显著);Rank IC≈0.07 为正但被成本吃掉。2025–2026 非盲测;前向验证从 2026-10-19 起(见 forward_registry)。", "",
     "## 2. 文件清单(含含义、大小、sha256)", "", "| 路径(包内) | 含义 | 字节数 | sha256 |", "|---|---|---:|---|"]
for a, n, h, d in rows: R.append(f"| `{a}` | {d} | {n:,} | `{h}` |")
R += ["", "## 3. 刻意排除的内容", "", "- 原始行情明细(`v2\\p0\\data\\raw_unadj_parquet`、`adj_factors`、Tencent/Sina 原始 parquet)、约 176 万行日线面板:**数据供应商再分发风险**,不放入。",
      "- 凭据/账户/个人信息:不含;包内代码仅保留 `D:\\choice_quant_starter\\...` 形式的项目相对路径描述(已做关键字扫描,命中见下)。",
      "- 完整历史预测大文件:`D:\\choice_quant_starter\\v2\\p1_dl\\historical_predictions_all_models.parquet`(1,860,516 行 = 6 模型 × 207 信号日 × 1498 只,约 98 MB;A2 版 `historical_predictions.parquet` 930,258 行 = HGB/LGBM/Ridge)和 `strategy_trades_v2.csv`(含成交价格)仅在本机,包内只放最近 12 个信号日、合格行的抽样 CSV(去掉成交价/股数列)。",
      "- 训练好的模型文件(joblib / .pt 检查点)未放入,sha256 记录在 `forward_registry_20261016_manifest.json`。", f"- 代码凭据关键字扫描命中:{len(hits)} 处" + ("(" + "; ".join(f"{a}:{i}" for a, i, _ in hits[:10]) + ",均需人工确认为无害变量名)" if hits else "(无)") + "。", "",
      "## 4. 复现步骤(本机,PowerShell,项目根 `D:\\choice_quant_starter`)", "",
      "1. 环境:`.venv`(Python 3.13,pandas/numpy/scikit-learn/lightgbm/torch/streamlit/plotly);P0 数据在 `v2\\p0\\data`(不在包内)。GRU/Transformer 训练用独立 CUDA venv(RTX 5070 Ti),CPU 不保证逐位一致。",
      "2. P0:`v2\\p0\\baseline_fixed.py`(walk-forward 207 期)→ `results\\fixed_sina_adj`;P0 无前视单测:`cd v2\\p0; ..\\..\\.venv\\Scripts\\python.exe -m pytest leakage_unit_tests`(45 项)。",
      "3. A1/A2:`score_latest.py`(冻结模型打分 2026-10-16)、`build_history.py`(重放并记录每日×每股分数,与 P0 对账 ≤1e-16)、`summaries.py`。",
      "4. B:`dl\\run_mlp_wf.py`、`dl\\run_seq_wf.py`(GRU/Transformer)、`dl\\dl_compare*.py`。",
      "5. A3:`a3\\a3_latest.py` → `a3\\a3_history.py` → `a3\\a3_cards.py` → `a3\\a3_report.py`;`a3\\a3_registry.py` 只能运行一次(已冻结,脚本拒绝覆盖)。",
      "6. 测试:`cd v2\\p1_dl; ..\\..\\.venv\\Scripts\\python.exe -m pytest tests -v`(65 项)。仪表盘:`.\\run_dashboard.ps1`(http://127.0.0.1:8501)。", "",
      "## 5. 已知局限(审计时请重点看)", "",
      "- 2025–2026 数据已多次用于模型比较,**不是盲测**。", "- 1500 只分层抽样样本,含 13.9% 已退市股(2 只无数据);幸存者/抽样偏差;无完整历史退市池。", "- 行业为巨潮最新分类,非历史时点;无历史 ST 清单(涨跌停 5% 仅近似);复权因子单源(新浪);价格 295 只样本单源;920196/600317 成交量单位无法核验。",
      "- Transformer 仅种子 42;GRU 三种子年化 -2.8%~+4.5%,种子噪声大;MLP/GRU 为 3 种子分数均值;GRU/TF 用 GPU,与 CPU 不保证逐位一致。", "- 往返成本 0.30% 为粗略假设,未建模冲击成本;无调参、无因子/模型消融、无行业/市值中性化;score 未校准。",
      "- 600060 海信视像不在 1500 样本:HGB/LGBM/Ridge/MLP 为池外打分,GRU/Transformer 未打分(需序列输入)。", "",
      "## 6. 建议的审计提问", "",
      "1. 训练样本的标签到期日是否严格早于每个信号日?(检查 `historical_predictions_all_models` 的 train_end_date / train_label_end_date;测试 `test_training_ends_before_label_window...`、`test_*_oos_parquet_models_trained_before_prediction`)",
      "2. `eligible_asof_t` 是否只用 t 日及以前的信息?买不到的订单是否真的留作现金而不是换股?(`p0lib.simulate`、`leakage_unit_tests`)",
      "3. DL 对账是否真正独立?(`reports\\reconciliation_dl.md`:用 OOS 预测 Top50 经同一模拟器重放,与 B 的 period_returns/订单日志最大差 ≤1.1e-16)",
      "4. 净超额显著性是否被重叠标签高估?(报告了 HAC(4) 与 5 期区块 bootstrap;配对检验见 `paired_tests_all_models.csv`)",
      "5. 6 个模型、多指标、多子样本的多重比较如何处理?(未做校正;所有结论为\"无显著\",前向验证将用冻结登记)",
      "6. 成本假设 0.30% 是否合理?0.60% 时全部为负(`cost_sensitivity_all_models.csv`)。", "7. 行业/板块 Rank IC 差异是否只是噪声?(组内样本少、非历史时点、多重比较)", "8. 前向登记是否可被事后修改?(manifest 记录 22 个文件 sha256;v2 主表 sha256 写入每个登记行)", "",
      "## 7. 校验", "", "解压后在包根目录用 `Get-FileHash -Algorithm SHA256 <路径>` 与上表核对。zip 内含本 README(README 自身 sha256 不在表中)。", ""]
readme = "\n".join(R); (H / "README_bundle.md").write_text(readme, encoding="utf-8")
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for a, p in paths: z.write(p, "review_bundle/" + a)
    z.writestr("review_bundle/README_bundle.md", readme.encode("utf-8"))
sp.unlink(); tmp.rmdir()
print("zip", OUT.stat().st_size, "uncompressed", size_total, "files", len(rows) + 1, "sha256", sha(OUT.read_bytes()))
