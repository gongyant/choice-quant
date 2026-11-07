# -*- coding: utf-8 -*-
"""Build p2_review_bundle.zip + README_p2_bundle.md (per-file sha256, whole-zip sha256), with a credential / privacy scan. Reads only; writes the zip, README, a panel sample and the scan report."""
import sys, re, json, hashlib, zipfile, datetime as dt, io
from pathlib import Path
import pandas as pd
H = Path(__file__).resolve().parent; R = H / "results"; OUT = H / "p2_review_bundle.zip"; RD = H / "README_p2_bundle.md"
def sha(b): return hashlib.sha256(b).hexdigest()
def collect():
    fs = []
    fs += sorted(p for p in H.glob("*.py")) + sorted(H.glob("*.ps1"))
    fs += [H / "a_verify.json", H / "a_verify.md"]
    fs += sorted(R.glob("*.csv")) + sorted(R.glob("*.json")) + sorted(R.glob("*.log")) + sorted(R.glob("*.md"))
    fs += sorted((R / "b_runs").glob("*")) + [R / "c_by_date.parquet"]
    fs += [p for p in sorted((R / "b_logs").glob("*")) if p.name in ("orders_TF_C2_explicit_A0.csv.gz", "decisions_TF_C2_explicit_A2_nested.csv.gz", "orders_TF_C2_explicit_A2_nested.csv.gz")]
    fs += sorted((H / "tests").glob("*.py")) + sorted((H / "tests").glob("*.txt"))
    fs += sorted((H / "figures").glob("*"))
    fs += sorted((H / "reports").glob("p2*.md"))
    return [p for p in fs if p.is_file() and p.name not in ("_patch.py", "_q.py", "c_bundle_scan.json") and p != OUT]
def panel_sample():
    P = pd.read_parquet(R / "p2a_panel.parquet"); ii = sorted(P.i.unique()); keep = [ii[0], ii[len(ii) // 2], ii[-1]]
    b = io.BytesIO(); P[P.i.isin(keep)].to_csv(b, index=False, encoding="utf-8-sig"); return b.getvalue(), keep, P.shape
PAT = {"aws_key": r"AKIA[0-9A-Z]{16}", "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "github_token": r"gh[pousr]_[A-Za-z0-9]{30,}", "openai_like": r"sk-[A-Za-z0-9]{20,}", "bearer": r"[Bb]earer\s+[A-Za-z0-9\-_\.]{20,}",
       "password_assign": r"(?i)(password|passwd|pwd|secret|api_?key|access_?token)\s*[:=]\s*['\"][^'\"]{4,}['\"]", "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "cn_mobile": r"(?<!\d)1[3-9]\d{9}(?!\d)", "cn_id": r"(?:身份证|ID ?card)[^\n]{0,10}\d{17}[\dXx]", "user_home": r"(?i)C:\\\\?Users\\\\?[^\\\\\s\"'<]+"}
def scan(name, data):
    if name.endswith((".png", ".parquet", ".gz", ".zip")): return []
    t = data.decode("utf-8", errors="ignore"); hits = []
    for k, p in PAT.items():
        for m in re.finditer(p, t):
            s = m.group(0)
            if k == "cn_mobile" and not re.search(r"(?i)(tel|phone|mobile|手机|电话)", t[max(0, m.start() - 20):m.start()]): continue      # long numbers in CSVs are numeric data
            hits.append(dict(file=name, kind=k, match=s[:60]))
    return hits
def main():
    files = collect(); entries = []; hits = []; t0 = dt.datetime.now()
    smp, keep, shp = panel_sample()
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            rel = p.relative_to(H).as_posix(); b = p.read_bytes(); z.writestr("p2_alpha_monetization/" + rel, b); entries.append((rel, len(b), sha(b))); hits += scan(rel, b)
        rel = "results/sample_p2a_panel_3dates.csv"; z.writestr("p2_alpha_monetization/" + rel, smp); entries.append((rel, len(smp), sha(smp))); hits += scan(rel, smp)
        readme = build_readme(entries, hits, keep, shp, zip_sha=None); z.writestr("p2_alpha_monetization/README_p2_bundle.md", readme.encode("utf-8"))
    zs = sha(OUT.read_bytes()); RD.write_text(build_readme(entries, hits, keep, shp, zip_sha=zs, size=OUT.stat().st_size), encoding="utf-8")
    (H / "p2_review_bundle.zip.sha256").write_text(f"{zs}  p2_review_bundle.zip\n", encoding="utf-8")
    json.dump(dict(n_files=len(entries) + 1, zip_bytes=OUT.stat().st_size, zip_sha256=zs, scan_hits=hits), open(R / "c_bundle_scan.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("files", len(entries) + 1, "zip MB", round(OUT.stat().st_size / 1e6, 2), "sha256", zs, "scan hits", len(hits)); [print(h) for h in hits[:30]]
def build_readme(entries, hits, keep, shp, zip_sha=None, size=None):
    L = ["# P2 审稿包(p2_review_bundle.zip)", "", f"生成:{dt.datetime.now():%Y-%m-%d %H:%M}(Asia/Singapore),`c_bundle.py`。只读研究材料;不是投资建议;模型分数不是概率;2022-07 以后的样本外数据已被前序阶段反复使用,**所有结论均为探索性**。", ""]
    if zip_sha: L += [f"- 整包 sha256:`{zip_sha}`(同见 `p2_review_bundle.zip.sha256`);大小 {size/1e6:.2f} MB;文件数 {len(entries)+1}(含包内 README)。", "- 包内的 README 副本不含整包 sha256(无法自引用),其余内容相同。", ""]
    else: L += ["- 整包 sha256 见包外 `README_p2_bundle.md` / `p2_review_bundle.zip.sha256`。", ""]
    L += ["## 内容", "- 研究选股器(t12u):`c_screener.py` → `results/research_screener_20261016.csv`(1498 行真实代码+名称,列说明见 `research_screener_20261016_columns.md`,参数见 `_meta.json`)与仪表盘「研究选股器」页。**研究排名,不是买入推荐;分数不是概率或收益;A2 权重只是研究模拟;P2 结论是信号未被证明可在成本后变现。** TF 三种子平均取最近 OOS 信号日 2026-09-28(未保存种子 43/44 模型,不重训)。", "- 脚本:P2-A(`a_*.py`)、P2-B(`b_*.py`)、P2-C(`c_*.py`)、共享库 `p2lib.py`、只读看板 `stock_dashboard_p2.py` + `run_dashboard_p2.ps1` + `smoke_dashboard_p2.py`。",
          "- 参数/协议:`results/b_protocol.json`、`results/c_protocol.json`、`a_verify.json`、`results/c_repro_r000.json` 等。", "- 结果:`results/*.csv`(全部统计输出)、`results/b_runs/*.parquet`(P2-B 每期组合结果,看板净值页使用)、`results/c_by_date.parquet`(P2-C 逐信号日指标)、`results/b_logs` 抽样(TF C2 的 A0/A2 订单与决策日志 3 个文件)、运行日志 `results/*.log`。",
          f"- 面板抽样:`results/sample_p2a_panel_3dates.csv` = `p2a_panel.parquet`(完整 {shp[0]} 行 × {shp[1]} 列,41 MB,未打包)中信号行号 {keep} 三个日期的全部行。", "- 测试与输出:`tests/*.py`、`tests/*.txt`;图 `figures/*.png`;三份报告 `reports/p2a_alpha_anatomy_report.md`、`p2b_cost_aware_report.md`、`p2c_transformer_ablation_report.md`。",
          "- **未打包**:原始行情(P0 `data/`)、P0/P1 代码与冻结产物(forward_registry 由 manifest sha256 校验)、P1/P2-C 模型 checkpoint、完整 P2-A 面板、其余 P2-B 订单日志。", "",
          "## 复现步骤(Windows PowerShell,项目根 `D:\\choice_quant_starter`)", "1. 需要 P0 数据与 P1 冻结产物在原路径(`v2\\p0`、`v2\\p1_dl`);venv:`.venv`(pandas/lightgbm/CPU torch/streamlit)、`.venv_cuda`(torch cu128,P2-C Transformer 训练)。",
          "2. P2-A:`python a_panel.py; python a_verify.py; python a_eval.py; python a_figs.py; python a_report.py`。", "3. P2-B:`python b_protocol.py; python b_bench.py; python b_launch.py`(24 个模型×情景作业)→ `python b_agg.py; python b_exposure.py; python b_figs.py; python b_report.py`。",
          "4. P2-C:`python c_protocol.py; python c_repro.py`;GPU:`..\\..\\.venv_cuda\\Scripts\\python.exe c_super.py 1`(按 retrain 块存 checkpoint、可续跑);CPU:`python c_lgbm.py --seeds 42 43 44`(事后敏感性 `--fixed 200`);`python c_agg.py; python c_figs.py; python c_report.py`。",
          "5. 测试:`pytest tests\\test_p2a.py tests\\test_p2b.py tests\\test_p2c.py tests\\test_dashboard_p2.py tests\\test_screener_p2.py`;P0 `pytest ..\\p0\\leakage_unit_tests`;P1 `pytest ..\\p1_dl\\tests`。看板:`.\\run_dashboard_p2.ps1` → http://127.0.0.1:8502。", "",
          "## 主要局限", "- 样本外期 2022-07 至 2026-10 只有 207 个周度信号,且已被 P0/P1/P2 反复使用;所有检验都是探索性的,没有新的保留期。", "- P2-B 的成本模型(Corwin-Schultz 半价差 + 平方根冲击)是参数化假设;C3/C4 约束下毛收益明显塌缩;没有任何策略的因子调整净 alpha 显著为正。",
          "- P2-C:消融变体只在前 104 个信号(前 26 个 retrain 块)上训练(事先声明的减配);种子数少;一次只改一个因素,但 T3−T0 等打包比较不可分解;持续性正则只试了一个 λ。", "- 候选池、行业为最新而非时点数据(P2-A 已披露);ST 状态部分依赖当前名称。", "",
          "## 给独立审稿人的问题", "0. 研究选股器的 A2 研究权重以 10-16 收盘为输入、空仓起步,未考虑 10-19 开盘可成交性;是否应在审阅中把它视为纯描述性输出?", "1. 执行日前向收益(下一可买日入场、跌停顺延卖出)的定义是否仍有隐性前视?请抽查 `p2lib.exec_forward` 与 `sample_p2a_panel_3dates.csv`。",
          "2. P2-B 嵌套时间验证(burn-in 52 期、26 期块、扩张窗口)是否足以排除参数选择乐观偏差?`b_nested_optimism_check.csv` 的结论可信吗?", "3. 显式成本模型的量级是否合理(尤其小盘、北交所的半价差与冲击),是否会系统性地偏向某类策略?",
          "4. P2-C 的 20 日标签 purge 推导(回看窗口不计入 embargo,只看标签到期)是否正确?`c_tf.split` 中的两条断言是否充分?", "5. 排序损失提高 RankIC 的同时显著增加 SMB 暴露:这是新增的预测力,还是更高效地加载了小盘因子?五因子回归是否足以回答?",
          "6. 配对检验采用 HAC(4) 与 5 期 block bootstrap;在 104/207 个周度观测、多重比较的情况下,哪些“显著”结论应当被校正后放弃?", "7. 滞后堆叠 LightGBM 早停常只保留极少的树:这是否让“Transformer 无受控优势”的比较对 LightGBM 不公平(或反之)?", "",
          "## 凭据 / 隐私扫描", f"- 规则:AWS key、私钥头、GitHub/类 OpenAI token、Bearer、password/secret/api_key 赋值、邮箱、带电话上下文的手机号、`C:\\Users\\<name>` 路径。命中 {len(hits)} 条" + (":" if hits else ",未发现凭据或个人信息。")]
    for h in hits[:50]: L.append(f"  - `{h['file']}`:{h['kind']} → `{h['match']}`")
    L += ["", "## 逐文件 sha256", "", "| 文件 | 字节 | sha256 |", "|---|---|---|"] + [f"| `{r}` | {n} | `{s}` |" for r, n, s in entries]
    return "\n".join(L) + "\n"
if __name__ == "__main__": main()