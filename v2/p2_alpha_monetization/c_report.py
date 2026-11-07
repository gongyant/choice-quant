# -*- coding: utf-8 -*-
"""P2-C report generator: reads results/c_*.csv and writes reports/p2c_transformer_ablation_report.md (all numbers come from the CSVs)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as Q
R = Q.RES
RN = pd.read_csv(R / "c_runs.csv"); SV = pd.read_csv(R / "c_seed_variance.csv"); PR = pd.read_csv(R / "c_paired.csv"); PO = pd.read_csv(R / "c_posthoc.csv"); OR = pd.read_csv(R / "c_oracle_diag.csv")
CV = pd.read_csv(R / "c_coverage.csv"); RC = pd.read_csv(R / "c_reconcile.csv"); BM = pd.read_csv(R / "c_block_meta.csv"); HI = pd.read_csv(R / "c_head_rankic.csv") if (R / "c_head_rankic.csv").stat().st_size > 5 else pd.DataFrame()
PRO = json.load(open(R / "c_protocol.json", encoding="utf-8")); REP = json.load(open(R / "c_repro_r000.json", encoding="utf-8"))
NAME = {"T0": "T0 P1基线", "T0G": "T0G 基线+g20几何", "T1": "T1 排序损失(5d)", "T2": "T2 多周期头MSE", "T3": "T3 多周期排序损失", "T4": "T4 持续性正则", "LAGLGBM": "滞后堆叠LGBM"}
MET = {"ic": "RankIC", "stab": "稳定性", "top50_overlap": "Top50重合", "turn_c2": "换手/期", "net_c1": "C1净收益/期", "net_c2": "C2净收益/期", "gross_c2": "毛收益/期", "cost_c2": "C2成本/期", "alpha_c1": "C1 alpha/期", "alpha_c2": "C2 alpha/期"}
pc = lambda x, d=2: "—" if pd.isna(x) else f"{x*100:.{d}f}%"; f = lambda x, d=3: "—" if pd.isna(x) else f"{x:.{d}f}"; bp = lambda x: "—" if pd.isna(x) else f"{x*1e4:.1f}bp"
def md(rows, head): return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"
def sig(r): return "显著" if (abs(r.hac_t) >= 1.96 and (r.ci_lo > 0 or r.ci_hi < 0)) else ("边缘" if abs(r.hac_t) >= 1.65 else "不显著")
def fmt_diff(m, x): return bp(x) if m in ("net_c1", "net_c2", "gross_c2", "cost_c2", "alpha_c1", "alpha_c2") else f(x, 4)
def pr_rows(A, B, sig_, scope="seed-mean", metrics=("ic", "stab", "top50_overlap", "turn_c2", "net_c1", "net_c2", "alpha_c2")):
    x = PR[(PR.A == A) & (PR.B == B) & (PR.signals.astype(str) == sig_) & (PR.scope == scope)]; out = []
    for m in metrics:
        y = x[x.metric == m]
        if len(y): y = y.iloc[0]; out.append([MET[m], fmt_diff(m, y.diff_mean), f(y.hac_t, 2), f"[{fmt_diff(m, y.ci_lo)}, {fmt_diff(m, y.ci_hi)}]", pc(y.share_A_gt_B, 0) if pd.notna(y.share_A_gt_B) else "—", sig(y)])
    return out
def get_pr(A, B, m, sig_="104", scope="seed-mean"):
    x = PR[(PR.A == A) & (PR.B == B) & (PR.signals.astype(str) == sig_) & (PR.scope == scope) & (PR.metric == m)]; return x.iloc[0] if len(x) else None
L = []; w = L.append
used = CV[CV.used]; nseeds = {v: sorted(used[used.variant == v].seed.tolist()) for v in used.variant.unique()}
w("# P2-C Transformer 受控消融:种子方差、匹配信息集 LightGBM、多周期头、排序损失、持续性正则\n")
w("> 生成:`c_report.py` 从 `results\\c_*.csv` 自动生成;所有数字来自这些 CSV。2022-07 至 2026 的样本外期都是复用过的历史数据,**所有结论均为探索性**;本报告不是投资建议,分数不是概率,也不宣称模型创新或经济优势。\n")
# ---------------- summary ----------------
w("## 0. 摘要(先说结论)\n")
s207 = SV[(SV.variant == "T0") & (SV.signals.astype(str) == "207")].iloc[0]; l207 = SV[(SV.variant == "LAGLGBM") & (SV.signals.astype(str) == "207")].iloc[0]
w(f"- **种子方差(T0 = P1 配方,207 信号,种子 {s207.seeds})**:RankIC 均值 {f(s207.rankic_mean,4)}(std {f(s207.rankic_std,4)},区间 {f(s207.rankic_min,4)}–{f(s207.rankic_max,4)});稳定性 {f(s207.stability_mean)}±{f(s207.stability_std)};Top50 重合 {f(s207.top50_overlap_mean)}±{f(s207.top50_overlap_std)};A0 换手 {f(s207.turnover_c2_mean)}/期;C2 净年化 {pc(s207.ann_net_c2_mean,1)}(std {pc(s207.ann_net_c2_std,1)},区间 {pc(s207.ann_net_c2_min,1)}~{pc(s207.ann_net_c2_max,1)});C2 alpha HAC t 均值 {f(s207.alpha_c2_hac_t_mean,2)}。跨种子同日分数 Spearman {f(s207.cross_seed_rank_corr,3)}(LGBM {f(l207.cross_seed_rank_corr,3)})。P1 的 seed 42 落在种子分布之内,不是“幸运种子”,但组合层净收益的种子间差异与其均值同量级。")
for sg in ("207", "104"):
    a = get_pr("LAGLGBM", "T0", "ic", sg); b = get_pr("LAGLGBM", "T0", "net_c2", sg); c = get_pr("LAGLGBM", "T0", "turn_c2", sg); d = get_pr("LAGLGBM", "T0", "alpha_c2", sg)
    if a is not None: w(f"- **滞后堆叠 LightGBM − Transformer({sg} 信号,种子均值路径配对)**:RankIC 差 {f(a.diff_mean,4)}(t {f(a.hac_t,2)},{sig(a)});换手差 {f(c.diff_mean,3)}(t {f(c.hac_t,2)});C2 净收益差 {bp(b.diff_mean)}/期(t {f(b.hac_t,2)},{sig(b)});C2 alpha 差 {bp(d.diff_mean)}/期(t {f(d.hac_t,2)},{sig(d)})。")
w("- 同一 60×7 信息集下,Transformer 的序列结构**没有表现出受控的 RankIC 优势**;两者差异主要在组合层(换手/Top50 重合),见 §4。")
for A, B in [("T0G", "T0"), ("T1", "T0"), ("T2", "T0G"), ("T3", "T2"), ("T4", "T0"), ("T3", "T0")]:
    a = get_pr(A, B, "ic"); s_ = get_pr(A, B, "stab"); t_ = get_pr(A, B, "turn_c2"); n_ = get_pr(A, B, "net_c2")
    if a is None:
        a1 = get_pr(A, B, "ic", scope=f"seed{42}")
        if a1 is None: w(f"- {A}−{B}:**未完成**(见 §8 减配)。"); continue
        a, s_, t_, n_ = a1, get_pr(A, B, "stab", scope="seed42"), get_pr(A, B, "turn_c2", scope="seed42"), get_pr(A, B, "net_c2", scope="seed42"); tag = "仅 seed 42"
    else: tag = "种子均值"
    w(f"- **{A}−{B}**({tag},前 104 信号):RankIC {f(a.diff_mean,4)}(t {f(a.hac_t,2)},{sig(a)});稳定性 {f(s_.diff_mean,3)}(t {f(s_.hac_t,2)},{sig(s_)});换手 {f(t_.diff_mean,3)}(t {f(t_.hac_t,2)});C2 净 {bp(n_.diff_mean)}/期(t {f(n_.hac_t,2)},{sig(n_)})。")
amax = RN[RN.window.isin(["all", "first104"])].alpha_c2_hac_t.max(); npos = int((RN[RN.window.isin(["all", "first104"])].alpha_c2_hac_t >= 1.96).sum())
w(f"- 所有组合层结论都要放在 P2-B 的背景下理解:A0 Top50 周频换手约 0.65/期,C2 成本吞掉大部分毛收益;各(变体, 种子, 窗口)的 C2 因子调整 alpha HAC t 最大为 {f(amax,2)},t≥1.96 的个数为 {npos},没有任何变体显示显著为正的因子调整 alpha。\n")
# ---------------- design ----------------
w("## 1. 范围与红线\n")
w("- P1 的 Transformer 架构、checkpoint、分数、P0/P1 文件与 `forward_registry_20261016*` 均未改动;新代码以只读 import 方式复用 P1 的 `dl_common`/`dl_seq`(`c_tf.py`),所有新产物在 `p2_alpha_monetization\\` 下(`c_*.py`、`c_checkpoints\\`、`results\\c_*`、`figures\\c_fig*.png`)。`tests\\test_p2c.py` 校验 manifest 22 个 sha256 与 P1 文件 mtime。")
w(f"- 协议 `results\\c_protocol.json` 声明于 {PRO['declared_at']},早于任何消融变体的第一个 checkpoint(测试校验);减配决定也写在其中。未在评估期上选择模型或超参:所有变体只有一个预先给定的配置(如持续性 λ=0.5、排序损失形式、LGBM 早停规则),没有网格。\n")
w("## 2. 设计\n")
w("### 2.1 受控变体(一次只改一个因素)\n")
rows = []
for v, sp in PRO["variants"].items(): rows.append([v, NAME.get(v, v), str(sp["heads"]), sp["loss"], sp["geom"], sp["persist"], ",".join(map(str, nseeds.get(v, []))) or "—"])
rows.append(["LAGLGBM", NAME["LAGLGBM"], "[5]", "LightGBM L2 + 早停", "p1", 0, ",".join(map(str, nseeds.get("LAGLGBM", []))) or "—"])
w(md(rows, ["代号", "说明", "输出头", "损失", "切分几何", "持续性λ", "完成种子"]))
w("单因素链:T0→T0G(只改 purge 几何)→T2(只加多周期头)→T3(只换排序损失);T0→T1(只换损失,批次按整日组织是排序损失的必要条件);T0→T4(只加持续性正则);T0 vs LAGLGBM(只换模型族,信息集、样本行、切分、retrain 频率、目标完全一致)。T3−T0、T3−T1 为打包/混杂比较,仅作参考。所有变体的主分数都是 5 日头。\n")
w("### 2.2 多周期标签的切分推导(不机械套用“回看+horizon”)\n")
w("- 行 d 的特征只用 ≤d 的数据(60 日回看窗口完全向后),因此**回看窗口不会把信息带到未来,不需要计入 embargo**。泄漏只可能来自标签:行 d 的 h 日标签区间是 close(d+1)→close(d+1+h),在 d+1+h 才实现。")
w("- P1(h=5)几何:验证段末行 hi=i−12,验证 60 日,训练段末 te=vs−10。最晚验证标签到期 hi+1+5=i−6(距信号日 6 个交易日的余量);最晚训练标签到期 te+6=vs−4(早于验证段首行 4 日)。")
w("- 对 Hmax=20,保持**相同余量**:要求 hi+1+20 ≤ i−6 ⇒ hi=i−27;要求 te+1+20 < vs−3 ⇒ 训练/验证间隔 25 日。即两处各加 20−5=15 日,而不是 60+20。`c_tf.split()` 对每个 head 断言这两条到期约束;测试还证明若用 P1 几何训练 20 日头,断言会触发(即存在标签泄漏)。")
w("- 因为 g20 几何本身改变了训练/验证样本(验证段向前移 15 日、训练段少 15+15 日),单独设立 T0G 作为几何对照,使“多周期头”的效果(T2−T0G)不与“purge 变长”混在一起。\n")
w("### 2.3 损失与正则\n")
w(f"- 排序损失:{PRO['rank_loss']}。\n- 持续性正则:{PRO['persistence']}(第一次实现中把分母方差 detach 后,模型把分数尺度压到 3.7e-5 来“满足”正则,已改为不 detach;测试校验尺度不变性)。\n- 滞后堆叠 LightGBM:{PRO['lag_lgbm']['features']};{PRO['lag_lgbm']['params']};{PRO['lag_lgbm']['split']}。\n")
w("### 2.4 度量与检验\n")
w("- RankIC:P2-A 新口径(信号日可得的候选池,执行日前向收益截面去均值,逐日 Spearman),HAC(4) t 与 5 期 block bootstrap 95% CI。\n- 稳定性:相邻信号日(间隔 5 个交易日)在两日都在候选池的股票上分数的 Spearman;Top50 重合:相邻两日 A0 Top50 的交集/50;并区分同一 retrain 块内与跨块的转移。\n- 组合:P2-B 的 A0(Top50 等权,P0 模拟器 `simulate_ext`,资本 1000 万),C1 flat 0.30% 双边、C2 显式 OHLCV 成本;换手为单边/期。\n- 因子 alpha:净收益减等权 c0 基准,对 P2-A 五因子(SMB、REV5、LVOL、MOM60、ILLIQ)HAC 回归截距;两策略的 alpha 差 = 净收益差序列对因子回归的截距(OLS 线性,测试校验)。\n- 配对:同一信号日 A−B 的差序列,HAC t、block bootstrap CI、A>B 日占比;“种子均值”= 先对各种子的逐日序列取均值(等权组合各种子策略),再配对;另给出每个种子的配对结果检验方向一致性。\n")
# ---------------- repro ----------------
w("## 3. 复现与对账\n")
w(f"- 新训练器重训 T0 seed 42 第 0 个 retrain 块:与 P1 冻结 checkpoint 预测最大绝对差 {REP['max_abs_mine_vs_p1ckpt']:.1e},相关 {REP['corr']:.10f},最优 epoch {REP['best_epoch_mine']} vs {REP['best_epoch_p1']},n_train {REP['n_train_mine']} 相同。")
rr = [[r.check, f(r.p2b, 6), f(r.p2c, 6), f"{r['diff']:.1e}" if pd.notna(r['diff']) else "—"] for _, r in RC.iterrows()]
w(md(rr, ["对账项", "P2-B/P2-A", "P2-C", "差"]))
w(f"- 覆盖:{md([[r.variant, r.seed, r.n_covered, r.n_needed, '是' if r.used else '否', f(r.finite_share_of_pool, 4)] for _, r in CV.iterrows()], ['变体', '种子', '已覆盖信号', '需要', '使用', '候选池内有分数比例'])}")
# ---------------- seed variance ----------------
w("## 4. 种子方差与匹配信息集 LightGBM\n")
w("### 4.1 每个种子(207 信号,window=all;eval_k52+ 见 CSV)\n")
x = RN[(RN.window == "all") & RN.variant.isin(["T0", "LAGLGBM"])]
w(md([[NAME[r.variant], int(r.seed), f(r.rankic, 4), f(r.rankic_hac_t, 2), f(r.stability), f(r.stab_within_block), f(r.stab_cross_boundary), f(r.top50_overlap), f(r.turnover_c2), pc(r.ann_gross_c2, 1), pc(r.ann_net_c1, 1), pc(r.ann_net_c2, 1), bp(r.alpha_c2), f(r.alpha_c2_hac_t, 2)] for _, r in x.iterrows()],
      ["模型", "种子", "RankIC", "HAC t", "稳定性", "块内", "跨块", "Top50重合", "换手/期", "毛年化C2", "净年化C1", "净年化C2", "alpha/期C2", "alpha t"]))
w("### 4.2 种子统计(均值 ± std,[最小, 最大],t 分布 95% 区间)\n")
rows = []
for _, r in SV.iterrows():
    rows.append([NAME.get(r.variant, r.variant), r.signals, r.seeds] + [f"{f(r[m+'_mean'],4)}±{f(r[m+'_std'],4)} [{f(r[m+'_min'],4)},{f(r[m+'_max'],4)}] ({f(r[m+'_ci_lo'],4)},{f(r[m+'_ci_hi'],4)})" for m in ("rankic", "stability", "turnover_c2")] + [f"{pc(r.ann_net_c2_mean,1)}±{pc(r.ann_net_c2_std,1)} [{pc(r.ann_net_c2_min,1)},{pc(r.ann_net_c2_max,1)}]", f"{f(r.alpha_c2_hac_t_mean,2)}±{f(r.alpha_c2_hac_t_std,2)}", f(r.cross_seed_rank_corr)])
w(md(rows, ["变体", "信号", "种子", "RankIC", "稳定性", "换手/期", "C2净年化", "alpha t(C2)", "跨种子分数相关"]))
w("注:3 个种子的 t 区间自由度只有 2,非常宽;区间只用于提示量级,不能据此宣称种子间“没有差异”。\n")
for sg in ("207", "104"):
    w(f"### 4.3 滞后堆叠 LGBM − Transformer 配对({sg} 信号,种子均值路径)\n"); w(md(pr_rows("LAGLGBM", "T0", sg), ["指标", "差(A−B)", "HAC t", "95% block CI", "A>B 日占比", "判定"]))
    per = PR[(PR.A == "LAGLGBM") & (PR.B == "T0") & (PR.signals.astype(str) == sg) & PR.scope.str.startswith("seed4") & PR.metric.isin(["ic", "net_c2", "turn_c2"])]
    w("逐种子:" + ";".join(f"{r.scope} {MET[r.metric]} {fmt_diff(r.metric, r.diff_mean)}(t {f(r.hac_t,2)})" for _, r in per.iterrows()) + "\n")
lm = BM[BM.variant == "LAGLGBM"]
if len(lm): w(f"- LGBM 早停的最优树数:中位数 {lm.best_iteration.median():.0f},最小 {lm.best_iteration.min():.0f},最大 {lm.best_iteration.max():.0f};后段 retrain 块常只保留个位数棵树(验证段 MSE 很快不再下降)。这是预先声明的早停规则的结果,不是调参;事后的固定 200 棵树敏感性见 §6。\n")
# ---------------- ablations ----------------
w("## 5. 消融变体(前 104 信号 = 前 26 个 retrain 块)\n")
S4 = SV[SV.signals.astype(str) == "104"]
rows = [[NAME.get(r.variant, r.variant), r.seeds, f(r.rankic_mean, 4), f(r.rankic_hac_t_mean, 2), f(r.stability_mean), f(r.top50_overlap_mean), f(r.turnover_c2_mean), pc(r.ann_gross_c2_mean, 1), pc(r.ann_net_c1_mean, 1), pc(r.ann_net_c2_mean, 1), f(r.alpha_c2_hac_t_mean, 2), f(r.cross_seed_rank_corr)] for _, r in S4.iterrows()]
w(md(rows, ["变体", "种子", "RankIC", "HAC t(均)", "稳定性", "Top50重合", "换手/期", "毛年化C2", "净年化C1", "净年化C2", "alpha t(C2,均)", "跨种子相关"]))
for A, B, desc, kind in PR[["A", "B", "desc", "kind"]].drop_duplicates().itertuples(index=False):
    if A == "LAGLGBM": continue
    sc = "seed-mean" if len(PR[(PR.A == A) & (PR.B == B) & (PR.scope == "seed-mean")]) else "seed42"
    w(f"### {A} − {B}:{desc}({kind};{'种子均值' if sc == 'seed-mean' else '仅 seed 42'})\n"); w(md(pr_rows(A, B, "104", sc), ["指标", "差(A−B)", "HAC t", "95% block CI", "A>B 日占比", "判定"]))
    per = PR[(PR.A == A) & (PR.B == B) & PR.scope.str.startswith("seed4") & PR.metric.isin(["ic", "stab", "net_c2"])]
    w("逐种子:" + ";".join(f"{r.scope} {MET[r.metric]} {fmt_diff(r.metric, r.diff_mean)}(t {f(r.hac_t,2)})" for _, r in per.iterrows()) + "\n")
fb = [c for c in RN.columns if c.startswith("beta_")]
xb = RN[((RN.window == "all") & ~RN.variant.isin(["T0", "LAGLGBM"])) | (RN.window == "first104")].groupby("variant")[fb + ["alpha_c2", "alpha_c2_hac_t"]].mean()
w("### 因子暴露(C2 净超额对五因子回归,前 104 信号,种子均值)\n"); w(md([[NAME.get(v, v)] + [f(r[c], 2) for c in fb] + [bp(r.alpha_c2), f(r.alpha_c2_hac_t, 2)] for v, r in xb.iterrows()], ["变体"] + [c[5:] for c in fb] + ["alpha/期", "alpha t"]))
if len(HI): w("### 多周期头各自的 RankIC(对 5 日实现收益,信息性)\n"); w(md([[r.variant, int(r.seed), int(r.horizon), f(r.rankic_vs_5d_realised, 4)] for _, r in HI.iterrows()], ["变体", "种子", "头(日)", "RankIC vs 5日实现"]))
tm = BM[BM.variant.isin(["T0", "T0G", "T1", "T2", "T3", "T4"]) & (BM.retrain < 26)].groupby("variant").agg(best_epoch=("best_epoch", "mean"), epochs=("epochs_run", "mean"), sec=("seconds", "mean"), n=("seconds", "size"))
w("训练诊断(前 26 块,所有已完成种子):" + ";".join(f"{v} 平均最优 epoch {r.best_epoch:.1f}/运行 {r.epochs:.1f}/每块 {r.sec:.0f}s(n={r.n})" for v, r in tm.iterrows()) + "\n")
# ---------------- seed 44 update vs 42/43 snapshot ----------------
if (R / "c_paired_s42s43.csv").exists():
    P0_ = pd.read_csv(R / "c_paired_s42s43.csv"); S0_ = pd.read_csv(R / "c_seed_variance_s42s43.csv")
    w("## 5b. 加入消融 seed 44 前后对照(前 104 信号)\n")
    w("快照 `results\\c_*_s42s43.csv` 保存了只有 seed 42/43 时的结果(17:40 生成),这里与当前结果并列,两者都没有删改。\n")
    rows = []
    for A, B in [("T0G", "T0"), ("T1", "T0"), ("T2", "T0G"), ("T3", "T2"), ("T4", "T0"), ("T3", "T0"), ("LAGLGBM", "T0")]:
        for m in ("ic", "stab", "turn_c2", "net_c2", "alpha_c2"):
            def pick(D):
                x = D[(D.A == A) & (D.B == B) & (D.signals.astype(str) == "104") & (D.metric == m) & (D.scope == "seed-mean")]
                if not len(x): x = D[(D.A == A) & (D.B == B) & (D.signals.astype(str) == "104") & (D.metric == m) & (D.scope == "seed42")]
                return x.iloc[0] if len(x) else None
            a, b = pick(P0_), pick(PR)
            if a is None or b is None: continue
            rows.append([f"{A}−{B}", MET[m], f"{fmt_diff(m, a.diff_mean)}(t {f(a.hac_t,2)},{int(a.n_seeds)} 种子)", f"{fmt_diff(m, b.diff_mean)}(t {f(b.hac_t,2)},{int(b.n_seeds)} 种子)", "一致" if np.sign(a.diff_mean) == np.sign(b.diff_mean) and sig(a) == sig(b) else ("方向一致" if np.sign(a.diff_mean) == np.sign(b.diff_mean) else "方向改变")])
    w(md(rows, ["比较", "指标", "seed 42/43", "当前(含 seed 44)", "判定变化"]))
    rows = []
    for v in ["T1", "T2", "T3", "T0G", "T4"]:
        a = S0_[(S0_.variant == v) & (S0_.signals.astype(str) == "104")]; b = SV[(SV.variant == v) & (SV.signals.astype(str) == "104")]
        if len(a) and len(b): a, b = a.iloc[0], b.iloc[0]; rows.append([NAME[v], a.seeds, f"{f(a.rankic_mean,4)}±{f(a.rankic_std,4)}", b.seeds, f"{f(b.rankic_mean,4)}±{f(b.rankic_std,4)}", f"{f(b.turnover_c2_mean,3)}±{f(b.turnover_c2_std,3)}", f"{pc(b.ann_net_c2_mean,1)}±{pc(b.ann_net_c2_std,1)}", f(b.alpha_c2_hac_t_mean, 2), f(b.cross_seed_rank_corr)])
    w(md(rows, ["变体", "原种子", "原 RankIC", "现种子", "现 RankIC", "现换手", "现 C2 净年化", "现 alpha t", "现跨种子相关"]))
# ---------------- post-hoc ----------------
w("## 6. 事后诊断(post-hoc,非预先声明的检验,不用于选择)\n")
P1_ = PO[PO.get("rankic").notna()] if "rankic" in PO.columns else PO
w(md([[r["diag"], int(r.seed), f(r.rankic, 4), f(r.stability), f(r.top50_overlap), f(r.turnover_c2), pc(r.ann_net_c2, 1), f(r.alpha_c2_hac_t, 2)] for _, r in P1_.iterrows()], ["诊断", "种子", "RankIC", "稳定性", "Top50重合", "换手/期", "C2净年化", "alpha t"]))
pp = PO[PO["diag"].str.contains("paired") & PO.metric.isin(["ic", "turn_c2", "net_c2"])] if "metric" in PO.columns else pd.DataFrame()
if len(pp): w("配对:" + ";".join(f"{r['diag'].replace(' [post-hoc paired]','')} s{int(r.seed)} {MET[r.metric]} {fmt_diff(r.metric, r.diff_mean)}(t {f(r.hac_t,2)})" for _, r in pp.iterrows()) + "\n")
w("### ORACLE 合成信号(仅诊断,构造中使用未来收益,不是策略)\n")
w(md([[f(r.rho_true_signal, 2), f(r.noise_ar1_phi, 2), f(r.rankic, 4), f(r.stability), f(r.top50_overlap), f(r.turnover_c2), bp(r.cost_c2_per_period)] for _, r in OR.iterrows()], ["真实信号载荷ρ", "噪声AR(1) φ", "RankIC", "稳定性", "Top50重合", "换手/期", "C2成本/期"]))
w("解读:在 RankIC 固定(约 0.05)时,换手几乎完全由分数中**噪声的持续性**决定;Transformer 稳定性约 0.65,对应 oracle 曲线上 φ≈0.6–0.7 的区域。降低换手可以靠让分数更“慢”(持续性正则、平滑),但这本身不产生 RankIC。该诊断净收益含未来信息,不报告。\n")
# ---------------- limitations ----------------
w("## 7. 结论(诚实版)\n")
w(f"1. P1 Transformer 的 seed 42 结果在 3 个种子的分布之内(RankIC {f(s207.rankic_min,4)}–{f(s207.rankic_max,4)});信号层种子方差小(std {f(s207.rankic_std,4)}),组合层种子方差相对均值很大(C2 净年化 std {pc(s207.ann_net_c2_std,1)},均值 {pc(s207.ann_net_c2_mean,1)})。跨种子同日分数相关 {f(s207.cross_seed_rank_corr,2)},Top50 名单在种子之间差异明显,单种子的组合层数字不宜过度解读。")
w("2. 匹配信息集下滞后堆叠 LightGBM 的 RankIC 与 Transformer 无显著差异;没有受控证据支持“Transformer 的序列建模带来额外预测力”。")
def eff(A, B, lab):
    sc = "seed-mean" if get_pr(A, B, "ic") is not None else "seed42"; a, s_, t_, n_, al = [get_pr(A, B, m, scope=sc) for m in ("ic", "stab", "turn_c2", "net_c2", "alpha_c2")]
    if a is None: return f"{lab}:未完成"
    return f"{lab}:RankIC {f(a.diff_mean,4)}(t {f(a.hac_t,2)},{sig(a)}),稳定性 {f(s_.diff_mean,3)}({sig(s_)}),换手 {f(t_.diff_mean,3)}({sig(t_)}),C2 净 {bp(n_.diff_mean)}/期({sig(n_)}),alpha 差 {bp(al.diff_mean)}/期({sig(al)})" + ("" if sc == "seed-mean" else "[仅 seed 42]")
w("3. 单因素效果(前 104 信号,种子均值路径):" + ";".join(eff(A, B, l) for A, B, l in [("T0G", "T0", "加长 purge(g20)"), ("T2", "T0G", "多周期头"), ("T1", "T0", "排序损失(单头)"), ("T3", "T2", "排序损失(多头)"), ("T4", "T0", "持续性正则")]) + "。")
w("   读法:排序损失是唯一同时提高 RankIC 和稳定性、降低换手的单因素;多周期头本身主要提高稳定性,不提高 RankIC;持续性正则以大幅牺牲 RankIC 换取换手下降;purge 加长基本无影响。但没有任何单因素在 C2 成本下带来显著的净收益或因子调整 alpha 改善,排序损失的组合还明显增加了 SMB(小盘)暴露(见因子暴露表),所以 RankIC 的提升不能直接解读为经济优势。")
t4 = SV[(SV.variant == "T4") & (SV.signals.astype(str) == "104")]
if len(t4): t4 = t4.iloc[0]; w(f"   持续性正则对种子很敏感:T4 的 RankIC 各种子在 {f(t4.rankic_min,4)}–{f(t4.rankic_max,4)} 之间,跨种子同日分数相关仅 {f(t4.cross_seed_rank_corr,2)},换手 {f(t4.turnover_c2_min,3)}–{f(t4.turnover_c2_max,3)}。λ=0.5 时模型接近“记住上一期的排序”,预测力不稳定。")
w("5. 多周期头各自的 RankIC 见 §5;多周期 MSE 模型的 20 日头对 5 日实现收益的 RankIC 反而高于 5 日头,排序损失模型四个头几乎相同。这只是信息性观察,没有据此选头,主分数始终是预先声明的 5 日头。")
w("4. 所有 2022-07 以后的数据都已被前序阶段反复使用,本阶段结果一律为探索性。\n")
w("## 8. 局限与减配说明\n")
cr = PRO["compute_reduction"]; w(f"- 原因:{cr['reason']}\n- T0:{cr['T0_seeds']}。\n- 消融:{cr['ablation_variants']}。\n- 选块:{cr['block_choice']}。")
miss = CV[~CV.used]; w("- 实际未完成的(变体, 种子):" + (", ".join(f"{r.variant}:s{r.seed}({r.n_covered}/{r.n_needed})" for _, r in miss.iterrows()) or "无") + "。未完成的不进入任何表格。")
s44 = CV[(CV.seed == 44) & CV.variant.isin(["T1", "T2", "T3", "T0G", "T4"])]; s44ok = bool(len(s44) == 5 and s44.used.all())
if s44ok: w("- 实际执行:T0 三种子 × 52 块、LAGLGBM 三种子 × 52 块全部完成;5 个消融变体 seed 42/43/44 各跑完前 26 块(seed 44 按用户“必要时接着跑完”的意见补跑,17:38–19:14 全部完成;原 seed 42/43 的数字保留在 `results\\c_*_s42s43.csv` 并在 §5b 对照)。3 个种子的 t 区间自由度只有 2,仍应以逐种子方向是否一致为主要参考。前 104 信号(2022-07 至 2024-08)整体是下跌市,所有变体的绝对净年化都为负,应看配对差与超额,不看绝对水平。")
else:
    w("- 实际执行:T0 三种子 × 52 块、LAGLGBM 三种子 × 52 块全部完成;5 个消融变体完成 seed 42 和 43 的前 26 块(用户确认取消 17:30 截止后 seed 43 全部跑完)。消融变体的 seed 44 按协议属于“时间允许再跑”,不纳入本报告:seed 43 完成后,监督进程按队列自动开始了 T1 seed 44,为遵守“不杀进程”的指示没有终止它;报告生成时它的块不完整,所以不进入任何表格(覆盖表中标为未使用;以后可从检查点续跑,再重跑 c_agg/c_report);因此消融的种子统计只有 2 个种子,t 区间(自由度 1)几乎没有信息量,应以逐种子方向是否一致为主要参考。前 104 信号(2022-07 至 2024-08)整体是下跌市,所有变体的绝对净年化都为负,应看配对差与超额,不看绝对水平。")
w("- 未做:额外种子 45/46;消融变体在后 26 块(2024-08 以后)的结果;多个持续性 λ 的扫描(只有一个预先给定的 λ)。\n- 前 104 信号只覆盖约 2 年,配对检验功效有限;3 个种子不足以可靠估计种子方差。\n- LGBM 早停在后段经常只保留极少树,其分数离散(大量并列),Top50 选择受并列打破规则影响;固定 200 树的事后敏感性仅 seed 42。\n- 候选池、成本模型、因子均沿用 P2-A/P2-B;行业为最新而非时点数据(P2-A 已披露)。\n")
w("## 9. 文件与复现\n")
w("- 脚本:`c_tf.py`(训练器/变体/几何)、`c_wf.py`(walk-forward 驱动,可续跑)、`c_super.py`(监督)、`c_lgbm.py`(滞后堆叠 LGBM,`--fixed 200` 为事后敏感性)、`c_repro.py`、`c_protocol.py`(参数文件 `results\\c_protocol.json`)、`c_eval.py`、`c_agg.py`、`c_figs.py`、`c_report.py`、`tests\\test_p2c.py`。")
w("- 复现顺序(PowerShell,工作目录 `p2_alpha_monetization`):`..\\..\\.venv_cuda\\Scripts\\python.exe c_super.py 1`(或 `c_wf.py --jobs T0:43:0,...`)→ `..\\..\\.venv\\Scripts\\python.exe c_lgbm.py --seeds 42 43 44` → `c_agg.py` → `c_figs.py` → `c_report.py` → `pytest tests\\test_p2c.py`。")
w("- 结果:`results\\c_runs.csv`、`c_seed_variance.csv`、`c_paired.csv`、`c_posthoc.csv`、`c_oracle_diag.csv`、`c_head_rankic.csv`、`c_coverage.csv`、`c_reconcile.csv`、`c_block_meta.csv`、`c_by_date.parquet`、`c_repro_r000.json`、`c_run.log`;图 `figures\\c_fig1..6_*.png`。\n")
tp = R.parent / "tests"; summ = []
for fn in ("pytest_p2c.txt", "pytest_p0_rerun_c.txt", "pytest_p1_rerun_c.txt", "pytest_p2a_rerun_c.txt", "pytest_p2b_rerun_c.txt", "pytest_dashboard_p2.txt", "smoke_dashboard_p2.txt", "http_dashboard_p2.txt"):
    p = tp / fn
    if p.exists(): last = [l for l in p.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()][-1:] or ["(空)"]; last = last[0]; summ.append([fn, last.strip("= ")])
w("## 10. 测试\n"); w(md(summ, ["文件", "结果"]) if summ else "(测试结果待写入)\n")
(Q.REP / "p2c_transformer_ablation_report.md").write_text("\n".join(L), encoding="utf-8"); print("report written", sum(len(x) for x in L))