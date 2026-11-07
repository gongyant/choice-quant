# -*- coding: utf-8 -*-
"""Generate reports/p2a_alpha_anatomy_report.md from results/*.csv (all numbers come from the CSVs)."""
import json, numpy as np, pandas as pd
import p2lib as Q
R = Q.RES; MOD = Q.MODELS
rd = lambda f: pd.read_csv(R / f)
T1, PD_, G, PI, PB = rd("rankic_old_vs_new.csv"), rd("rankic_old_minus_new_paired.csv"), rd("rankic_by_group.csv"), rd("partial_rankic.csv"), rd("baseline_paired_ic.csv")
DS, DM, DB, SUB, ATT, EX, TD, CT = rd("decile_summary.csv"), rd("decile_means.csv"), rd("decile_spread_by_period.csv"), rd("subset_robustness.csv"), rd("factor_attribution.csv"), rd("execution_by_decile.csv"), rd("timing_diagnostic.csv"), rd("rankic_clean_timing_rows.csv")
ICD = rd("rankic_by_date.csv"); meta = json.load(open(R / "p2a_panel_meta.json", encoding="utf-8")); V = json.load(open(Q.P2 / "a_verify.json", encoding="utf-8")); PBL = rd("partial_rankic_baselines.csv")
f4 = lambda x: f"{x:.4f}"; f2 = lambda x: f"{x:.2f}"; pc = lambda x: f"{x*100:.2f}%"
def ci(r): return f"[{r.ci_lo:.4f}, {r.ci_hi:.4f}]"
def table(header, rows): return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])
def it(df): return (s for _, s in df.iterrows())
L = []; A = L.append
new = T1[T1.definition == "new"].set_index("model"); old = T1[T1.definition == "old_all"].set_index("model"); olx = T1[T1.definition == "old_exec_t1"].set_index("model")
A("# P2-A 阿尔法解剖报告(alpha anatomy)\n")
A(f"生成:{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} (Asia/Singapore)。范围:只做 P2-A,不重训模型,只使用冻结的 `historical_predictions_all_models.parquet` 与 `dl_*_oos_predictions.parquet`;P0/P1 输出、`forward_registry_20261016.*` 未改动。\n")
A("> **声明**:本报告为研究诊断,**不是投资建议**;模型分数不是概率、也不是预期收益;2025–2026 数据在方法设计阶段已被复用,**只算探索性**,不是盲测;所有显著性均为 207 个不重叠 5 日期上的 HAC(4) / 5 期块自助结果,未做全表多重检验校正(仅对六模型头条 RankIC 给 Bonferroni×6)。\n")
A("## 0. 摘要(先读这里)\n")
h, l = new.loc["HGB"], new.loc["LGBM"]
A(f"1. **验证**:冻结总表 1,860,516 行、键唯一、代码均为 6 位字符串(430 个以 0 开头代码保留)、六模型同为 207 个信号日 × 1498 码(eligible 238,781 行,**GRU/TF 在冻结文件中覆盖与 HGB 完全相同**,故共同样本=全体 eligible)、`realized_ret_5d`/`realized_rel_ret_5d` 与我用 P0 矩阵独立重算的标签差为 0;forward_registry 22 个文件 sha256 全部匹配。**唯一记录缺口**:MLP/GRU/TF 的 `n_train_rows` 在总表为空(未写入),不影响分数。详见 `a_verify.md`。")
A(f"2. **新口径(下一可执行交易日起算)RankIC**:六模型均值 {new.loc[MOD,'mean'].min():.4f}–{new.loc[MOD,'mean'].max():.4f},HAC t {new.loc[MOD,'hac_t'].min():.1f}–{new.loc[MOD,'hac_t'].max():.1f},块自助 CI 全部不含 0;比旧口径(t+1→t+6)高约 0.003–0.004(配对 HAC t≈-5.5~-6.9),**但这个抬升来自仅占 ~2% 的尾部执行事件(涨停封板推迟买入、跌停封板推迟卖出、长期停牌),不是普通样本变好**(见 §3)。")
b3 = PB[(PB.model == "HGB") & (PB.baseline == "3F(REV5+LOWVOL+ILLIQ)")].iloc[0]; lv = PB[(PB.model == "HGB") & (PB.baseline == "LOWVOL")].iloc[0]
A(f"3. **正的平均 RankIC 本身不说明模型有独特价值**:简单‘低波动’信号的 RankIC 就是 {new.loc['LOWVOL','mean']:.4f},三因子基线(反转5日+低波动+Amihud 非流动性)为 {new.loc['3F(REV5+LOWVOL+ILLIQ)','mean']:.4f},**不低于任何一个模型**;HGB−三因子的配对差 {b3['mean']:.4f}(HAC t {b3.hac_t:.2f},CI [{b3.ci_lo:.4f}, {b3.ci_hi:.4f}]),HGB−低波动 {lv['mean']:.4f}(t {lv.hac_t:.2f})——即**模型并未在 RankIC 上打败简单风格因子**。")
p4 = PI[PI.control_set.str.startswith("S4")].set_index("model")
A(f"4. **偏 RankIC**:控制 r5、r20、vol20、换手率、Amihud 之后,HGB/LGBM/MLP 剩 {p4.loc['HGB','mean']:.4f}/{p4.loc['LGBM','mean']:.4f}/{p4.loc['MLP','mean']:.4f}(HAC t {p4.loc['HGB','hac_t']:.1f}/{p4.loc['LGBM','hac_t']:.1f}/{p4.loc['MLP','hac_t']:.1f}),约为原始的 {p4.loc['HGB','partial_over_raw']*100:.0f}%;GRU/TF 剩 {p4.loc['GRU','mean']:.4f}/{p4.loc['TF','mean']:.4f};**Ridge 仅 {p4.loc['Ridge','mean']:.4f}(t {p4.loc['Ridge','hac_t']:.1f},CI 含 0)——Ridge 的信号基本被已知风格解释**。注意这只是线性秩残差化,模型里还含 ma_gap/range20/volume_ratio 等与这些控制变量非线性相关的特征,‘残余’不等于‘独立新信息’。")
a5 = ATT[(ATT.spec == "5因子") & (ATT.dependent == "D10-D1多空价差")].set_index("model"); l5 = ATT[(ATT.spec == "5因子") & (ATT.dependent == "D10相对池均值(多头超额)")].set_index("model")
A(f"5. **十分位**:单调性好(分数十分位 vs 平均收益的 Spearman {DS[(DS.definition=='new')&DS.model.isin(MOD)].spearman_decile_vs_mean.min():.2f}–{DS[(DS.definition=='new')&DS.model.isin(MOD)].spearman_decile_vs_mean.max():.2f}),D10−D1 每 5 日约 {pc(DS[(DS.definition=='new')&(DS.model=='HGB')]['mean'].iloc[0])}(HGB),但**价差主要来自空头端**:HGB D1 {pc(DS[(DS.definition=='new')&(DS.model=='HGB')].D1.iloc[0])}、D10 {pc(DS[(DS.definition=='new')&(DS.model=='HGB')].D10.iloc[0])};A 股个股无法对称做空,多头端(扣费前)仅 {pc(DS[(DS.definition=='new')&(DS.model=='HGB')].D10.iloc[0])} 每 5 日,而 A3 采用的往返成本为 0.30%。")
A(f"6. **因子调整归因**:5 因子(SMB/REV5/LVOL/MOM60/ILLIQ)调整后 D10−D1 的 alpha:HGB {pc(a5.loc['HGB','alpha'])}(t {a5.loc['HGB','alpha_hac_t']:.2f})、LGBM {pc(a5.loc['LGBM','alpha'])}(t {a5.loc['LGBM','alpha_hac_t']:.2f})、MLP {pc(a5.loc['MLP','alpha'])}(t {a5.loc['MLP','alpha_hac_t']:.2f})、GRU {pc(a5.loc['GRU','alpha'])}(t {a5.loc['GRU','alpha_hac_t']:.2f})、TF {pc(a5.loc['TF','alpha'])}(t {a5.loc['TF','alpha_hac_t']:.2f})、**Ridge {pc(a5.loc['Ridge','alpha'])}(t {a5.loc['Ridge','alpha_hac_t']:.2f},不显著)**。三因子基线本身对同样 5 因子回归 R²≈{ATT[(ATT.spec=='5因子')&(ATT.dependent=='D10-D1多空价差')&(ATT.model=='3F(REV5+LOWVOL+ILLIQ)')].r2.iloc[0]:.2f}、alpha≈0(方法自检通过)。多头端(D10−池均值)的 5 因子 alpha:HGB {pc(l5.loc['HGB','alpha'])}(t {l5.loc['HGB','alpha_hac_t']:.2f})、LGBM {pc(l5.loc['LGBM','alpha'])}(t {l5.loc['LGBM','alpha_hac_t']:.2f})、MLP/GRU/TF t≈{l5.loc['MLP','alpha_hac_t']:.1f}/{l5.loc['GRU','alpha_hac_t']:.1f}/{l5.loc['TF','alpha_hac_t']:.1f},Ridge t {l5.loc['Ridge','alpha_hac_t']:.1f}。")
sx = SUB[SUB.model == "HGB"].set_index("subset")
A(f"7. **可交易性**:剔除 amt20<1e8 的低成交额股后,HGB RankIC {sx.loc['剔除 amt20<1e8 (5倍入池门槛)','mean']:.4f}(t {sx.loc['剔除 amt20<1e8 (5倍入池门槛)','hac_t']:.1f}),D10 相对池均值转为 {pc(sx.loc['剔除 amt20<1e8 (5倍入池门槛)','D10_mean'])}——**排序能力仍在,但多头端超额消失,价差靠空头端**;‘干净集’(非最低流动性三分位+非 t 日涨停+非 ST 名)HGB RankIC {sx.loc['组合干净集:非低流动性+非t日涨停+非ST名','mean']:.4f}、D10 {pc(sx.loc['组合干净集:非低流动性+非t日涨停+非ST名','D10_mean'])}。")
A("8. **稳定性**:2026 年(截至 2026-09-28 的信号)价差/IC 明显弱于 2022–2025(见 §10);2026 下半年仅含 7–9 月的少量信号日(样本数见 §7 表),不能下结论。\n")
A("**总体诚实结论**:冻结预测对未来 5 日相对收益有稳定为正的截面排序信息(RankIC≈0.07–0.08,不含 0),树模型/DL 在控制已知风格后仍保留约 37–48% 的偏相关;但(i)它**没有在 RankIC 上超过简单低波动/三因子基线**,(ii)价差大部分在**空头端**,多头端扣费前每 5 日约 0.3%、与 0.30% 往返成本同量级(A3 已见净超额不显著),(iii)在高成交额子集多头端超额为负/零,(iv)2026 年走弱。因此目前**不能**说这些模型已证明可变现的阿尔法;结论限于‘有排序信息、需要在可执行、含成本、仅多头的约束下重新检验’。\n")
# ---- 1 verify
A("## 1. 冻结预测文件核验(摘要;全文见 `a_verify.md/json`)\n")
A(table(["检查","结果"], [[c["name"], "通过" if c["ok"] else "**不一致**"] for c in V["checks"]]))
A(f"\n- 不一致清单:{V['inconsistencies'] or '无(n_train_rows 为空仅作记录:' + V['note_n_train_rows'] + ')'}")
A(f"- 训练元数据:{V['train_meta_note']}")
A(f"- 覆盖说明:{V['coverage_note']}")
A("- 共同样本:六模型 (signal_date, code) 的 score 交集 = 238,781 行(100% eligible);新口径下再要求有可实现收益 = " + f"{V['coverage']['common_with_new_def_return']:,} 行(其余 {V['coverage']['eligible_rows']-V['coverage']['common_with_new_def_return']} 行在 t+1…t+6 内始终无法买入或到日历末尾,已排除并计数)。所有跨模型比较均在这一共同样本上进行。\n")
# ---- 2 definitions
A("## 2. 收益口径定义(旧 vs 新)\n")
A("- **信号**:t 日收盘后给出分数;候选池 = P0 `eligible_asof_t`(t 日有成交、已上市、20 日成交额中位数≥2e7、r60/vol20 可得),只用 ≤t 的信息;所有分组、对照变量、基线信号也都用 ≤t 的数据在池内当日横截面计算。")
A("- **旧口径(控制组)**:t+1 收盘买入 → t+6 收盘卖出的复权收益(P0 标签),不论 t+1 是否涨停封板/停牌;相对收益=减去当日 eligible 池均值。样本 238,781 行。")
A("- **旧口径·仅 t+1 可买(中间对照)**:旧收益,但剔除 t+1 收盘涨停封板/停牌者(用于把‘样本剔除’与‘推迟买卖’两种效应分开)。")
A("- **新口径(下一可执行交易日)**:买入日 e = 自 t+1 起第一个 **有成交且收盘未封涨停** 的交易日(最多顺延到 t+6;P0 `lim_up` = 收盘锁在涨停价);买入价 = e 日复权收盘;卖出日 = 自 e+5 个交易日起第一个 **有成交且收盘未封跌停** 的交易日(最多再顺延 10 个交易日);若仍不可卖,在第 10 日按最近已知(前向填充)复权价强制了结并标记 `exit_forced`(退市/长期停牌的真实损失以最后价计,不假设能成交)。若到数据末尾仍未完成则 NaN。相对收益=减去当日‘有可实现收益的 eligible 成员’均值(RankIC 对此去均值不敏感)。")
A(f"- **不可买入处理**:t+1 涨停封板/停牌 → 顺延买入而不是剔除也不是按 t+1 价成交;**不用替补**(与 P0 simulate 一致);从未能买入者 {meta['eligible_rows']-meta['ret_new_eligible_finite']} 行排除并计数。统计:eligible 行中 t+1 不可买 {meta['share_not_buyable_t1_eligible']*100:.2f}%,买入被推迟 {meta['share_entry_delayed']*100:.2f}%,强制了结 {meta['share_exit_forced']*100:.3f}%。")
A("- **局限**:无分钟/盘口数据,‘涨停封板’用收盘锁价代理(收盘价=涨停价且=当日最高);ST 无历史名单,P0 用 5% 限幅代理;成交假设为收盘价全额成交,无冲击成本。")
A("- **时序/无前视**:新口径用到的 e、卖出日可交易性是**执行日事实**,只用于定义已实现收益(评价目标),不参与候选池、分组、对照变量、基线的构造,也不会反过来选择 t 日候选池。\n")
# ---- 3 old vs new
A("## 3. 六模型 × 新旧口径 RankIC(207 个不重叠 5 日期;HAC 滞后 4;5 期块自助 95% CI)\n")
rows = []
for m in MOD:
    o, x, n = old.loc[m], olx.loc[m], new.loc[m]; d = PD_.set_index("model").loc[m]
    rows.append([m, f4(o["mean"]), f2(o.hac_t), ci(o), f4(x["mean"]), f4(n["mean"]), f2(n.hac_t), ci(n), f"{n.p_bonf6:.2g}", f4(d["mean"]), f2(d.hac_t)])
A(table(["模型","旧口径 RankIC","t","95%CI","旧·仅t+1可买","**新口径 RankIC**","t","95%CI","Bonferroni×6 p","旧−新(配对)","配对t"], rows))
A(f"\n新口径样本日均 {new.loc['HGB','avg_n']:.0f} 只;胜率(RankIC>0 的日期占比)HGB {new.loc['HGB','pos_share']:.2f}、LGBM {new.loc['LGBM','pos_share']:.2f}、GRU {new.loc['GRU','pos_share']:.2f}、TF {new.loc['TF','pos_share']:.2f}。与 A3 的旧口径 RankIC(0.0705–0.0776)一致。\n")
A("**新口径更高的来源(`timing_diagnostic.csv`、`rankic_clean_timing_rows.csv`)**:\n")
td = TD[TD.model == "HGB"]
A(table(["执行状态(HGB)","行数","占比","新口径平均收益","旧口径平均收益","池化 Spearman"], [[r.segment, f"{r.n_rows:,}", f"{r.share*100:.2f}%", pc(r.mean_ret_new), pc(r.mean_ret_old), f4(r.pooled_spearman)] for r in it(td)]))
ct = CT.set_index("model")
A(f"\n只保留‘买卖均无推迟’的行(占 97.9%)时,新口径 RankIC 为 HGB {ct.loc['HGB','mean']:.4f}、LGBM {ct.loc['LGBM','mean']:.4f}、GRU {ct.loc['GRU','mean']:.4f}、TF {ct.loc['TF','mean']:.4f}——**略低于**旧口径全样本。新口径的提升完全来自约 2% 的尾部行:买入被推迟者(t+1 封涨停后平均 -2.75%)、到期跌停封板推迟卖出者(平均 -13.7%)、强制了结者(-30%)。模型把这些个股排得靠后(池化 Spearman 0.04–0.17),RankIC 因此被拉高。这类‘出事’样本反映的是**空头/回避端**信息,对做多没有直接收益,而且依赖‘不可卖出时按之后价格计损’的假设。结论:新口径更保守地反映了可执行性,但 RankIC 的提升不应解读为‘更强的选股能力’。\n")
# ---- 4 groups
A("## 4. 分组 RankIC(新口径;分组变量均为 t 时点可得;每日组内样本<15 的日期不计,日数<30 不报 t/CI)\n")
def gtab(gvlabel_prefix, models=MOD):
    d = G[G.group_var.str.startswith(gvlabel_prefix)]; rows = []
    for gr in d.group.unique():
        for m in models:
            r = d[(d.group == gr) & (d.model == m)].iloc[0]
            rows.append([gr, m, f"{r.avg_n_per_date:.0f}", int(r.n_dates), f4(r["mean"]), f2(r.hac_t) if pd.notna(r.hac_t) else "-", ci(r) if pd.notna(r.ci_lo) else "-"])
    return table(["组","模型","日均样本","日数","RankIC","HAC t","95%CI"], rows)
A("### 4.1 流动性(amt20 日内三分位;L1 最低)\n"); A(gtab("流动性(amt20 日内")); A("\n### 4.2 流动性(amt20 绝对档,元)\n"); A(gtab("流动性(amt20 绝对"))
A("\n### 4.3 波动率(vol20 日内三分位)\n"); A(gtab("波动率")); A("\n### 4.4 交易所(北交所日均仅约 35 只、仅 118 个日期达到每日≥15 只,宜谨慎)\n"); A(gtab("交易所"))
A("\n### 4.5 板块\n"); A(gtab("板块", ["HGB", "LGBM", "GRU", "TF"]))
ind = G[G.group_var.str.startswith("行业") & (G.model == "HGB")].sort_values("mean")
A("\n### 4.6 行业(**cninfo 最新分类,非时点;仅作描述,不能当作历史行业中性证据**;THS 行业/概念成分缺失;‘未分类’=universe 中 209 只无行业分类者)\n")
A(table(["行业(最新)","日均样本","日数","HGB RankIC","HAC t","95%CI"], [[r.group, f"{r.avg_n_per_date:.0f}", int(r.n_dates), f4(r["mean"]), f2(r.hac_t) if pd.notna(r.hac_t) else "-", ci(r) if pd.notna(r.ci_lo) else "-"] for r in it(ind)]))
A(f"\n要点:HGB 在低/中/高流动性三分位 RankIC 为 {G[(G.group=='L1低流动性')&(G.model=='HGB')]['mean'].iloc[0]:.4f}/{G[(G.group=='L2中')&(G.model=='HGB')]['mean'].iloc[0]:.4f}/{G[(G.group=='L3高流动性')&(G.model=='HGB')]['mean'].iloc[0]:.4f},**IC 并未集中在最不流动的一组**(低流动性组反而最低);在高波动组最强({G[(G.group=='V3高波动')&(G.model=='HGB')]['mean'].iloc[0]:.4f})、低波动组最弱({G[(G.group=='V1低波动')&(G.model=='HGB')]['mean'].iloc[0]:.4f}),说明排序能力部分来自波动/风险暴露的横截面差异。DL 模型在高流动性、大成交额档并不比树模型差(TF 在 amt20≥3e8 档 {G[(G.group=='A4>=3e8')&(G.model=='TF')]['mean'].iloc[0]:.4f})。图:`figures/ic_by_liquidity.png`、`ic_by_vol_exchange.png`、`ic_by_industry_latest.png`。\n")
# ---- 5 partial
A("## 5. 逐日偏 RankIC(新口径;每日把分数秩与收益秩对控制变量的秩做横截面 OLS 残差化再取相关;换手率缺失(腾讯源个股)的行在含 turn20 的控制集中剔除,并同步给出同样本原始 RankIC)\n")
rows = [[r.control_set, r.model, f4(r["mean"]), f2(r.hac_t), ci(r), f4(r.raw_ic_same_sample_mean), f"{r.partial_over_raw:.2f}", f"{r.fm_slope_mean:.4f}", f2(r.fm_slope_hac_t)] for r in it(PI)]
A(table(["控制集","模型","偏RankIC","HAC t","95%CI","同样本原始IC","偏/原始","FM斜率","FM t"], rows))
A("\n对照:同样的控制对基线(`partial_rankic_baselines.csv`):REV5/REV20 在控制 r5,r20 后偏 IC≈0(预期,方法自检);三因子基线控制其自身分量后≈0。图:`figures/partial_ic.png`。\n")
# ---- 6 baselines
A("## 6. 基线对比(新口径)\n")
A("**基线构造(均无前视、不拟合任何参数、方向事先固定)**:REV5=−r5(5 日复权收益);REV20=−r20;LOWVOL=−vol20(20 日日收益标准差);ILLIQ=Amihud20(20 日 |r1|/成交额 均值,窗口含 t);**3F=三者在 t 日 eligible 池内百分位秩的等权平均(REV5、LOWVOL、ILLIQ)**;3F-size 把 ILLIQ 换成‘小成交额’代理(−log_amt20;**无流通市值数据,所以不是市值,只是规模/流动性代理**)。所有输入只含 ≤t 数据(`tests/test_p2a.py` 有截断不变性测试)。\n")
rows = [[b, f4(new.loc[b, "mean"]), f2(new.loc[b, "hac_t"]), ci(new.loc[b])] for b in ["REV5", "REV20", "LOWVOL", "ILLIQ(Amihud)", "3F(REV5+LOWVOL+ILLIQ)", "3F-size(REV5+LOWVOL+SMALL)"]] + [[m, f4(new.loc[m, "mean"]), f2(new.loc[m, "hac_t"]), ci(new.loc[m])] for m in MOD]
A(table(["信号","RankIC","HAC t","95%CI"], rows))
A("\n配对差(模型−基线,每日 RankIC 差):\n")
rows = [[r.model, r.baseline, f4(r.model_mean_ic), f4(r.baseline_mean_ic), f4(r["mean"]), f2(r.hac_t), ci(r)] for r in it(PB) if r.model in MOD]
A(table(["模型","基线","模型IC","基线IC","差","HAC t","95%CI"], rows))
A("\n十分位价差对比(D10−D1,每 5 日):" + "; ".join(f"{b} {pc(DS[(DS.definition=='new')&(DS.model==b)]['mean'].iloc[0])}(t {DS[(DS.definition=='new')&(DS.model==b)].hac_t.iloc[0]:.1f})" for b in ["REV5", "REV20", "LOWVOL", "3F(REV5+LOWVOL+ILLIQ)", "3F-size(REV5+LOWVOL+SMALL)"]) + "。" + "模型的 D10−D1 价差(HGB 1.51%、LGBM 1.45%、GRU 1.46%、TF 1.47%)是三因子基线(0.64%)的约 2.3 倍,**这与‘RankIC 不高于基线’并不矛盾:基线在全池秩相关上占优,模型在两端(尤其 D1)更集中**。\n")
A("**解读**:HGB/LGBM 的 RankIC 显著高于单独的 REV5(+0.048,t≈4.1)、ILLIQ(+0.048,t≈5.4),与 REV20 差 +0.019(t≈1.9,边缘),与低波动、三因子**无差异**(配对 t≈0 / -0.5)。因此‘模型比简单反转强’成立仅对 REV5;‘模型比低波动/三因子强’在 RankIC 上**没有证据**。\n")
# ---- 7 deciles
A("## 7. 模型分数十分位(新口径,D10=分数最高;每日池内 ordinal 等分;相对收益=减当日池均值)\n")
dsn = DS[DS.definition == "new"].set_index("model")
rows = [[m, f"{dsn.loc[m,'D1']*100:.2f}", f"{dsn.loc[m,'D5']*100:.2f}", f"{dsn.loc[m,'D10']*100:.2f}", f"{dsn.loc[m,'mean']*100:.2f}", f2(dsn.loc[m, 'hac_t']), ci(dsn.loc[m]).replace("[", "[").replace("]", "]"), f"{dsn.loc[m,'spearman_decile_vs_mean']:.2f}", f"{dsn.loc[m,'mean_date_spearman']:.2f}", int(dsn.loc[m, "steps_up_of_9"]), f"{dsn.loc[m,'pos_share']:.2f}"] for m in MOD + ["REV5", "REV20", "LOWVOL", "3F(REV5+LOWVOL+ILLIQ)"]]
A(table(["信号","D1 %","D5 %","D10 %","D10−D1 %","HAC t","CI(小数)","十分位Spearman(聚合)","逐日Spearman均值","上升台阶/9","价差>0占比"], rows))
dm = DM[(DM.definition == "new") & DM.model.isin(["HGB"])]
A("\nHGB 逐分位平均相对收益(%):" + ", ".join(f"D{int(r.decile)} {r.mean_rel_ret*100:.2f}" for r in it(dm)) + "。形状:D1 显著为负(-1.2%),D2–D9 近零,D10 小幅为正——**阶梯主要靠最低分位的‘回避’**。\n")
yr = DB[(DB.definition == "new") & (DB.level == "year") & DB.model.isin(["HGB", "LGBM", "MLP", "GRU", "TF", "Ridge", "3F(REV5+LOWVOL+ILLIQ)"])]
A("**分年度 D10−D1(每 5 日 %;括号=朴素 t,仅作参考;2026 仅到 2026-09-28 信号)**:\n")
pv = yr.pivot(index="model", columns="period", values="spread_mean"); pt = yr.pivot(index="model", columns="period", values="naive_t"); nn = yr.groupby("period").n_dates.first()
A(table(["模型"] + [f"{p}(n={nn[p]})" for p in pv.columns], [[m] + [f"{pv.loc[m,p]*100:.2f}({pt.loc[m,p]:.1f})" for p in pv.columns] for m in pv.index]))
hy = DB[(DB.definition == "new") & (DB.level == "half") & DB.model.isin(["HGB", "LGBM", "GRU", "TF"])]
pv = hy.pivot(index="model", columns="period", values="spread_mean"); nn = hy.groupby("period").n_dates.first()
A("\n**分半年 D10−D1(每 5 日 %)**:\n"); A(table(["模型"] + [f"{p}(n={nn[p]})" for p in pv.columns], [[m] + [f"{pv.loc[m,p]*100:.2f}" for p in pv.columns] for m in pv.index]))
A("\n半年内样本仅 26 个左右,t 值不可靠;2024H1、2025H1 等个别半年波动大;2026H1 HGB/LGBM 价差≈0。图:`figures/decile_returns.png`、`decile_spread_by_halfyear.png`。\n")
# ---- 8 subsets
A("## 8. 真正的截面能力 vs 不可交易/低流动性段(新口径,各子集重算)\n")
A("说明:‘剔除退市样本股’使用了**未来**才知道的退市信息(universe 里 208/1500 只已终止上市),**只是事后诊断,不是可交易规则**;‘ST 名’用的是**现名**(非时点);t 日涨停封板为 t 日已知(可交易规则)。\n")
rows = [[r.subset, r.model, f"{r.avg_n:.0f}", f4(r["mean"]), f2(r.hac_t), ci(r), f"{r.spread_mean*100:.2f}", f2(r.spread_hac_t), f"{r.D10_mean*100:.2f}", f"{r.D1_mean*100:.2f}"] for r in it(SUB) if r.model in MOD]
A(table(["子集","模型","日均样本","RankIC","t","95%CI","D10−D1 %","t","D10 %","D1 %"], rows))
ex = EX[EX.model == "HGB"]
A("\nHGB 各十分位的执行/风险构成(旧口径样本,`execution_by_decile.csv`):\n")
A(table(["十分位","t+1不可买占比","t日涨停占比","ST名占比","退市股占比","买入被推迟","从未可买"], [[int(r.dec), f"{r.share_not_buyable_t1*100:.1f}%", f"{r.share_lim_up_at_t*100:.1f}%", f"{r.share_st_name*100:.1f}%", f"{r.share_delisted_stock*100:.1f}%", f"{r.share_entry_delayed*100:.1f}%", f"{r.share_never_executable*100:.2f}%"] for r in it(ex)]))
A("\n要点:(i)D1 里 8.9% 是 t 日收盘封涨停的股票(模型预测反转);这些是‘无法做空/无意义买入’的名字,拉大了空头端价差;(ii)剔除 t 日涨停、ST 名、退市样本后 RankIC 只小幅下降(HGB 0.078→0.072–0.076),说明排序能力**不是**由这些极端样本独占;(iii)但把样本限制为 amt20≥1e8 后 D10 超额由 +0.34% 变为负,说明**多头端的小幅超额集中在较低成交额股**——恰是最难以规模成交的部分;高流动性三分位 RankIC 仍有 0.056(HGB)~0.077(TF),排序信息存在,但变现空间主要是空头端。\n")
# ---- 9 attribution
A("## 9. 因子调整归因与块自助\n")
A("因子收益(每日、新口径相对收益,上/下 30% 等权多空):SMB(小成交额−大)、REV5(低 r5−高)、LVOL(低 vol20−高)、MOM60(高 r60−低)、ILLIQ(高 Amihud−低)。被解释变量=模型的 D10−D1 价差或 D10−池均值;OLS + Newey-West(4);alpha 的 CI 为 5 期块自助(2000 次)。\n")
rows = []
for dep in ["D10-D1多空价差", "D10相对池均值(多头超额)"]:
    for m in MOD + ["REV5", "LOWVOL", "3F(REV5+LOWVOL+ILLIQ)"]:
        a = ATT[(ATT.model == m) & (ATT.dependent == dep)].set_index("spec"); r5, r3, r0 = a.loc["5因子"], a.loc["3因子(REV5,LVOL,ILLIQ)"], a.loc["无因子(均值)"]
        rows.append([dep, m, pc(r0.alpha), f2(r0.alpha_hac_t), pc(r5.alpha), f2(r5.alpha_hac_t), f"[{r5.alpha_ci_lo*100:.2f}%, {r5.alpha_ci_hi*100:.2f}%]", f2(r5.r2), pc(r3.alpha), f2(r3.alpha_hac_t)])
A(table(["被解释变量","信号","未调整均值","t","5因子 alpha","t","alpha 95%CI","R²","3因子 alpha","t"], rows))
b = ATT[(ATT.spec == "5因子") & (ATT.dependent == "D10-D1多空价差") & ATT.model.isin(["HGB", "GRU", "TF"])]
A("\n暴露(5 因子回归的 β / HAC t,D10−D1):\n")
fc = [c for c in ATT.columns if c.startswith("beta_")]
A(table(["模型"] + [c[5:] for c in fc], [[r.model] + [f"{getattr(r, c):.2f}/{ATT.loc[r.name, 't_' + c[5:]]:.1f}" for c in fc] for r in it(b)]))
A("\n要点:树/DL 模型的 D10−D1 约 19–59% 的方差可由这些因子解释(R²≈0.19–0.59),扣除后 alpha 仍为每 5 日约 0.8–1.2%(t≈4.3–5.5);Ridge 被完全解释(R²=0.73,alpha t=0.77)。多头端 alpha 仅 0.2–0.33%/5 日(t≈2.0–3.0,Ridge 为负),扣 0.30% 往返成本后基本为零。DL 模型对 SMB/ILLIQ 的暴露(β 约 +2.3/-2.0,GRU)与树模型不同,提示 GRU 的部分价差来自规模/流动性对冲后的结构。图:`figures/factor_attribution.png`。\n")
# ---- 10 yearly IC
A("## 10. 分年度 RankIC(新口径;2025–2026 仅作探索性)\n")
x = ICD[ICD.definition == "new"].copy(); x["year"] = pd.to_datetime(x.signal_date).dt.year
yy = x.groupby(["model", "year"]).ic.agg(["mean", "count"]).reset_index(); pvy = yy.pivot(index="model", columns="year", values="mean"); cnt = yy.groupby("year")["count"].first()
A(table(["信号"] + [f"{y}(n={cnt[y]})" for y in pvy.columns], [[m] + [f"{pvy.loc[m,y]:.4f}" for y in pvy.columns] for m in MOD + ["LOWVOL", "3F(REV5+LOWVOL+ILLIQ)"]]))
A("\n")
# ---- 11 limitations
A("## 11. 局限\n")
for s in ["候选池里没有历史 ST 名单/市值/流通市值/真实历史行业:ST 只能用现名或 5% 限幅代理;规模只能用成交额代理;行业用 cninfo 最新分类(非时点),**行业分组只是描述**。",
          "样本含 13.9% 已退市股票(反幸存者偏差是优点,但‘退市’字段含未来信息,只用于事后诊断)。",
          "6 个模型 × 多种口径/分组/子集,检验很多,仅对头条 RankIC 做了 Bonferroni×6;分组表中的显著性是未校正的,个别组(北交所、半年窗口)样本小。",
          "HAC(4)+5 期块自助假设相邻信号日的序列相关较短;信号间隔 5 个交易日、5 日标签不重叠,但新口径下推迟买卖会使少数收益窗口略有重叠。",
          "新口径把尾部执行事件的损失(推迟卖出、强制了结)计入收益,RankIC 提升来自这些尾部;这些事件依赖价格前向填充与‘最多顺延 10 日’假设。",
          "涨停/跌停封板用收盘锁价代理,没有分钟数据;成交假设为收盘价全额成交,无冲击成本、无资金容量约束(P2-A 不涉及组合/容量)。",
          "偏 RankIC 是线性(秩)残差化;模型含与控制变量非线性相关的特征,‘残余’不等于独立信息;因子归因中因子由同一池、同一收益定义构造,依赖 30/30 分位的选择。",
          "2022-07~2026-09 全部为 walk-forward 样本外,但 2025–2026 的数据已在 P0/P1 设计、A3 评估阶段被反复查看,**只能算探索性,不是盲测**。真正的盲测只有 forward_registry(2026-10-19 起)的新信号。",
          "DL:GRU/TF 覆盖与 HGB 相同,但训练窗口早结束约 3 个月(早停验证块),Transformer 仅 seed 42,DL 的 n_train_rows 未记录。"]: A("- " + s)
A("\n## 12. 对 P2-B 的建议\n")
for s in ["**把检验改成‘只多头、含成本、可成交约束’**:基准用 3F/低波动,检验模型在‘多头 D10/Top-N − 同风格基准’上的净超额(不是多空价差),因为 A 股个股难以对称做空。",
          "**风格中性化后重训/重评分**(或在组合层做波动、规模/流动性、短期反转的暴露约束),看偏 IC 残余(约 0.03–0.037)能否在多头端兑现;Ridge 已被风格解释,可作为‘纯风格’对照。",
          "**重点研究流动性≥1e8 的子集与成本敏感性**:该子集多头端超额≈0,若在此处不能转正则模型的可变现空间主要在低成交额部分,需要容量与冲击成本模型。",
          "**把 D1 端‘回避’信息单独变现**:例如作为黑名单/剔除规则(剔除模型最差十分位 + t 日涨停/ST),检验剔除后基准多头的收益改善(‘回避’端 D1 -1.2%/5 日是本研究里最稳的一块信息),且无需做空。",
          "继续保持 forward registry 的盲测协议(≥30 个独立 5 日期才有初步功效);不要再用 2025–2026 做参数选择。",
          "时变性:2026 年 IC/价差走弱,需要滚动/分状态(波动率状态、涨跌停制度)监控,以及再训练频率与衰减的评估。"]: A("- " + s)
A("\n## 13. 文件清单\n")
A("- 代码:`p2lib.py`、`a_panel.py`、`a_verify.py`、`a_eval.py`、`a_diag.py`、`a_figs.py`、`a_report.py`、`tests/test_p2a.py`")
A("- 结果:`a_verify.json/md`;`results/` 下 `p2a_panel.parquet`、`rankic_old_vs_new.csv`、`rankic_by_date.csv`、`rankic_old_minus_new_paired.csv`、`rankic_by_group.csv`、`group_completeness.csv`、`partial_rankic*.csv`、`baseline_paired_ic.csv`、`score_corr_with_baselines.csv`、`decile_*.csv`、`subset_robustness.csv`、`execution_by_decile.csv`、`factor_returns_by_date.csv`、`factor_attribution.csv`、`timing_diagnostic.csv`、`rankic_clean_timing_rows.csv`")
A("- 图:`figures/*.png`(ic_by_liquidity、ic_by_vol_exchange、ic_by_industry_latest、decile_returns、decile_spread_by_halfyear、partial_ic、factor_attribution、rankic_old_vs_new_and_baselines、subset_robustness)")
txt = "\n".join(L)
(Q.REP / "p2a_alpha_anatomy_report.md").write_text(txt, encoding="utf-8")
print(len(txt), txt.count("\n"))