# -*- coding: utf-8 -*-
import json, numpy as np, pandas as pd
import p2lib as Q
R = Q.RES; MODELS = Q.MODELS; EV = 52
P = pd.read_csv(R / "b_performance.csv"); PE = P[P.window == "eval_k52+"]; PA = pd.read_csv(R / "b_paired.csv"); TV = pd.read_csv(R / "b_turnover_vs_prediction.csv"); FA = pd.read_csv(R / "b_factor_alpha.csv")
NS = pd.read_csv(R / "b_nested_selection.csv"); OP = pd.read_csv(R / "b_nested_optimism_check.csv"); FD = pd.read_csv(R / "b_fill_diagnostics.csv"); ST = pd.read_csv(R / "b_cost_stress.csv"); CC = pd.read_csv(R / "b_controls_calibration.csv")
EXP = pd.read_csv(R / "b_exposures.csv"); EW = pd.read_csv(R / "b_ew_benchmark.csv"); REC = pd.read_csv(R / "b_reconcile_A0.csv"); PR = json.load(open(R / "b_protocol.json", encoding="utf-8"))
V3 = ["A0", "A1_nested", "A2_nested"]; VN = {"A0": "A0", "A1_nested": "A1(嵌套)", "A2_nested": "A2(嵌套)"}
pc = lambda x, d=2: "—" if pd.isna(x) else f"{x*100:.{d}f}%"; f2 = lambda x, d=2: "—" if pd.isna(x) else f"{x:.{d}f}"
def md(rows, head):
    o = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]; o += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]; return "\n".join(o) + "\n"
def g(s, m, v, col, df=PE): return df[(df.scenario == s) & (df.model == m) & (df.variant == v)].iloc[0][col]
def mean_over(s, v, col): return float(PE[(PE.scenario == s) & (PE.variant == v)][col].mean())
def perf_table(s):
    rows = []
    for m in MODELS:
        for v in V3:
            r = PE[(PE.scenario == s) & (PE.model == m) & (PE.variant == v)].iloc[0]
            rows.append([m, VN[v], pc(r.ann_gross, 1), pc(r.ann_net, 1), f2(r.sharpe_net), pc(r.max_drawdown, 1), f2(r.turnover_oneway_per_period), pc(r.cost_drag_annual, 1), f"{r.excess_vs_EWc0_mean*100:.3f}% (t={f2(r.excess_hac_t)})", f2(r.n_hold, 0)])
    return md(rows, ["模型", "策略", "毛年化", "净年化", "净夏普", "最大回撤", "单边换手/期", "年化成本拖累", "净超额/期 vs 等权(HAC t)", "平均持仓数"])
def decomp_table():
    rows = []
    for m in MODELS:
        for v in V3:
            r = PE[(PE.scenario == "C2_explicit") & (PE.model == m) & (PE.variant == v)].iloc[0]; fl = g("C1_flat", m, v, "drag_flat_annual")
            rows.append([m, VN[v], pc(r.drag_commission_annual), pc(r.drag_transfer_annual), pc(r.drag_stamp_annual), pc(r.drag_spread_annual), pc(r.drag_impact_annual), pc(r.cost_drag_annual), pc(fl)])
    return md(rows, ["模型", "策略", "佣金", "过户费", "印花税", "半价差", "冲击", "显式合计(C2)", "flat 0.30%口径(C1)"])
def fa_table(s):
    rows = []
    for m in MODELS:
        for v in V3:
            r = FA[(FA.scenario == s) & (FA.model == m) & (FA.variant == v)].iloc[0]
            rows.append([m, VN[v], f"{r.alpha_per_period*100:.3f}%", f2(r.alpha_hac_t), f"[{r.alpha_ci_lo*100:.2f}%, {r.alpha_ci_hi*100:.2f}%]", f2(r.r2), f2(r.beta_SMB), f2(r.beta_LVOL), f2(r.gross_alpha_hac_t)])
    return md(rows, ["模型", "策略", "净alpha/期", "HAC t", "bootstrap 95%CI", "R²", "β_SMB", "β_LVOL", "毛alpha t"])
def paired_table(s):
    rows = []
    for m in MODELS:
        for c in ["A1_nested - A0", "A2_nested - A0", "A2_nested - A1_nested"]:
            r = PA[(PA.scenario == s) & (PA.model == m) & (PA.comparison == c)].iloc[0]
            rows.append([m, c.replace("_nested", "(嵌套)"), f"{r.net_diff_mean*100:.3f}%", f2(r.net_hac_t), f"[{r.net_ci_lo*100:.2f}%, {r.net_ci_hi*100:.2f}%]", f"{r.gross_diff_mean*100:.3f}%", f"{r.cost_diff_mean*100:.3f}%", f2(r.turnover_diff_mean, 3)])
    return md(rows, ["模型", "比较", "净差/期", "HAC t", "block bootstrap 95%CI", "毛差/期", "成本差/期", "换手差"])
def attr_table(s):
    rows = []
    for _, r in TV[TV.scenario == s].iterrows():
        rows.append([r.model, r.family, f"{r.delta_net*100:.3f}%", f"{r.delta_gross*100:.3f}%", f"{r.delta_cost_saving*100:.3f}%", f"{r.turnover_effect_S_adj*100:.3f}%", f"{r.selection_effect_S_adj*100:.3f}% (t={f2(r.selection_S_adj_hac_t)})", f"{r.turnover_effect_R_adj*100:.3f}%", f"{r.selection_effect_R_adj*100:.3f}% (t={f2(r.selection_R_adj_hac_t)})"])
    return md(rows, ["模型", "策略", "Δ净(X−A0)/期", "Δ毛", "Δ成本节省", "换手效应(S对照)", "选股效应(S对照,t)", "换手效应(R对照)", "选股效应(R对照,t)"])
def c34_table():
    rows = []; SS = ("C2_explicit", "C3_realistic", "C4_capacity_1e8")
    for m in MODELS:
        for v in V3:
            a = [g(s, m, v, "ann_gross") for s in SS]; b = [g(s, m, v, "ann_net") for s in SS]; sh = [g(s, m, v, "sharpe_net") for s in SS]
            fr = [FD[(FD.scenario == s) & (FD.model == m) & (FD.variant == v)].iloc[0] for s in SS[1:]]
            rows.append([m, VN[v], pc(a[0], 1), pc(a[1], 1), pc(a[2], 1), pc(b[0], 1), pc(b[1], 1), pc(b[2], 1), f2(sh[1]), f2(sh[2]), pc(fr[0].buy_fill_rate, 1), pc(fr[1].cash_frac, 1)])
    return md(rows, ["模型", "策略", "C2毛", "C3毛", "C4毛", "C2净", "C3净", "C4净", "C3夏普", "C4夏普", "C3买入成交率", "C4平均现金占比"])
def stress_table():
    rows = []; nm = {"hs0.5": "半价差×0.5", "base": "基准", "hs2": "半价差×2", "imp0.5": "冲击×0.5", "imp2": "冲击×2"}
    for m in ("HGB", "LGBM"):
        for v in V3:
            r = [pc(g("C2_explicit", m, v, "ann_net"), 1) if o == "base" else pc(ST[(ST.model == m) & (ST.strategy == v) & (ST.stress == o)].iloc[0].net_mean * 50.4, 1) for o in nm]
            rows.append([m, VN[v]] + r)
    return md(rows, ["模型", "策略"] + list(nm.values()))
def nested_table():
    rows = []
    for m in MODELS:
        for fam in ("A1", "A2"):
            x = NS[(NS.scenario == "C2_explicit") & (NS.model == m) & (NS.family == fam)].sort_values("block_start"); o = OP[(OP.scenario == "C2_explicit") & (OP.model == m) & (OP.family == fam)].iloc[0]
            rows.append([m, fam, " → ".join(c.replace("A1_", "").replace("A2_", "").replace("_doff", "").replace("_d0.3", "|δ0.3") for c in x.chosen), f2(o.nested_eval_sharpe), f2(o.best_fixed_eval_sharpe_hindsight) + f" ({o.best_fixed_hindsight_variant.replace('_doff','')})", f2(o.median_fixed_eval_sharpe)])
    return md(rows, ["模型", "族", "C2 各嵌套块选中参数(52/78/104/130/156/182起)", "嵌套净夏普", "事后最优固定参数夏普(偷看)", "网格中位夏普"])
def year_table():
    ew = EW.set_index("k"); rows = []
    for m in ("HGB", "LGBM"):
        r = pd.read_parquet(R / "b_runs" / f"{m}_C2_explicit.parquet")
        for v in V3:
            x = r[(r.variant == v) & (r.k >= EV)].copy(); x["yr"] = pd.to_datetime(ew.signal_date.loc[x.k.values].values).year; x["ex"] = x.ret.values - ew.ew_ret_c0.loc[x.k.values].values
            gg = x.groupby("yr"); rows.append([m, VN[v]] + [f"{gg.ret.mean()[y]*100:.2f}% / {gg.ex.mean()[y]*100:.2f}% (n={gg.ret.size()[y]})" for y in (2023, 2024, 2025, 2026)])
    return md(rows, ["模型", "策略", "2023H2", "2024", "2025(探索)", "2026(探索)"])
def exp_table():
    rows = []
    for _, r in EXP[EXP.scenario == "C2_explicit"].iterrows(): rows.append([r.model, VN[r.variant], f2(r.n_names, 0), f2(r.z_log_amt20), f2(r.z_vol20), f2(r.z_r5)])
    return md(rows, ["模型", "策略", "平均持仓名数", "z(log成交额20日)", "z(波动率20日)", "z(5日收益)"])
def fill_table():
    rows = []
    for s in ("C2_explicit", "C3_realistic"):
        for v in V3:
            x = FD[(FD.scenario == s) & (FD.variant == v)]; rows.append([s, VN[v], pc(x.buy_fill_rate.mean(), 1), pc(x.cash_frac.mean(), 1), f2(x.n_hold.mean(), 0), pc(x.max_weight.max(), 1), f2(x.pending_exit_mean.mean()), f2(x.rej_limit_up_locked_per_period.mean()), f2(x.rej_lot_too_small_per_period.mean())])
    return md(rows, ["场景", "策略", "买入成交率", "平均现金占比", "平均持仓数", "最大单票权重", "平均待卖(跌停/停牌)数", "涨停买不进/期", "手数不足拒单/期"])
mC = lambda s, v, c: mean_over(s, v, c)
n1 = int((NS[NS.family == "A1"].chosen == "A1_K150").sum()); n1t = int((NS.family == "A1").sum()); n2 = NS[NS.family == "A2"]; t10 = int(n2.chosen.str.contains("t1.0").sum()); dof = int(n2.chosen.str.contains("doff").sum())
def sig_cnt(s, c): x = PA[(PA.scenario == s) & (PA.comparison == c)]; return int((x.net_hac_t > 1.96).sum()), int((x.net_ci_lo > 0).sum()), float(x.net_diff_mean.mean())
fa_neg = lambda s, v: int((FA[(FA.scenario == s) & (FA.variant == v)].alpha_hac_t < -1.96).sum()); fa_pos = lambda s, v: int((FA[(FA.scenario == s) & (FA.variant == v)].alpha_hac_t > 1.96).sum())
S_sel = TV.groupby("scenario")[["selection_effect_S_adj", "selection_effect_R_adj", "turnover_effect_S_adj", "delta_net", "delta_cost_saving"]].mean()
S_t = TV.assign(sg=TV.selection_S_adj_hac_t.abs() > 1.96, rg=TV.selection_R_adj_hac_t > 1.96).groupby("scenario")[["sg", "rg"]].sum()
cal = CC.assign(gap=CC.achieved_turnover / CC.target_turnover - 1).groupby(CC.control.str[0]).gap.agg(["mean", "min", "max"]); lamS = TV.lambda_S.agg(["mean", "max"]); lamR = TV.lambda_R.agg(["min", "mean"])
rec_max = float(REC.maxabs.max()); opt = OP.assign(d=OP.best_fixed_eval_sharpe_hindsight - OP.nested_eval_sharpe).groupby("family").d.mean()
c2 = "C2_explicit"; c1 = "C1_flat"; c3 = "C3_realistic"; r2 = lambda v: FA[(FA.scenario == c2) & (FA.variant == v)].r2.mean()
rep = f"""# P2-B 成本感知组合化报告(A0 朴素 Top50 / A1 排名缓冲 / A2 成本感知优化)

> 生成时间:2026-10-19(Asia/Singapore, UTC+8)。目录 `v2\\p2_alpha_monetization\\`。**研究排名不是投资建议;分数不是概率或收益预测;无真实下单、无券商、无杠杆。** 2025–2026 段为已被前期阶段看过的数据的复用,**一律为探索性,不是盲测**。冻结的 P0/P1 输出、forward_registry 及其 manifest 未被改动。

## 0. 结论摘要(先说结论,再说证据)

1. **A0→A1→A2 的净收益改善几乎全部来自"成本/换手下降",而不是"选股更好"。** 显式成本口径(C2)下 6 模型平均:A0 毛年化 {pc(mC(c2,'A0','ann_gross'),1)} → 净年化 {pc(mC(c2,'A0','ann_net'),1)}(年化成本拖累 {pc(mC(c2,'A0','cost_drag_annual'),1)},单边换手 {f2(mC(c2,'A0','turnover_oneway_per_period'))}/期);A1(嵌套)净年化 {pc(mC(c2,'A1_nested','ann_net'),1)};A2(嵌套)净年化 {pc(mC(c2,'A2_nested','ann_net'),1)}(换手 {f2(mC(c2,'A2_nested','turnover_oneway_per_period'))}/期,成本拖累 {pc(mC(c2,'A2_nested','cost_drag_annual'),1)},但**毛年化只有 {pc(mC(c2,'A2_nested','ann_gross'),1)}**,比 A0 低)。flat 0.30% 口径(C1)平均净年化:A0 {pc(mC(c1,'A0','ann_net'),1)}、A1 {pc(mC(c1,'A1_nested','ann_net'),1)}、A2 {pc(mC(c1,'A2_nested','ann_net'),1)}。
2. **匹配换手对照把"换手节省"与"选股"分开。** 以"随机跳过"对照(S:与 A1/A2 同换手、仍用同一冻结分数,但换仓时机随机)为基准,C2 下选股效应均值 {S_sel.loc[c2,'selection_effect_S_adj']*100:.3f}%/期(Δ净均值 {S_sel.loc[c2,'delta_net']*100:.3f}%/期,其中成本节省 {S_sel.loc[c2,'delta_cost_saving']*100:.3f}%/期);12 个(模型×族)中选股效应 |t|>1.96 的有 {int(S_t.loc[c2,'sg'])} 个。"随机保留"对照(R,随机丢弃持仓)是更苛刻的稻草人,选股效应均值 {S_sel.loc[c2,'selection_effect_R_adj']*100:.3f}%/期,t>1.96 的 {int(S_t.loc[c2,'rg'])} 个——它说明"按排名保留"优于"随机保留",但不说明比"同换手的陈旧 Top50"更好。**结论:没有证据表明缓冲/优化改善了预测本身,改善是换手下降带来的成本节省。**
3. **因子调整后没有净 alpha。** 以等权基准为参照、对 P2-A 的 5 因子回归(C2,6 模型):净 alpha t<−1.96 的个数 A0 {fa_neg(c2,'A0')}、A1 {fa_neg(c2,'A1_nested')}、A2 {fa_neg(c2,'A2_nested')};t>1.96 的个数 A0 {fa_pos(c2,'A0')}、A1 {fa_pos(c2,'A1_nested')}、A2 {fa_pos(c2,'A2_nested')}。A2 的回归 R² 均值 {r2('A2_nested'):.2f}(A0 {r2('A0'):.2f}),说明其期收益与这些风格因子高度同向(含成本后的残差为负):持有约 {mC(c2,'A2_nested','n_hold'):.0f} 个名字,倾向低波动、成交额偏小(§8.3)。
4. **配对检验(C2)。** A2(嵌套)−A0 平均净差 {sig_cnt(c2,'A2_nested - A0')[2]*100:.3f}%/期,HAC t>1.96 的模型 {sig_cnt(c2,'A2_nested - A0')[0]}/6,block bootstrap 区间不含 0 的 {sig_cnt(c2,'A2_nested - A0')[1]}/6;A1(嵌套)−A0 平均 {sig_cnt(c2,'A1_nested - A0')[2]*100:.3f}%/期,t>1.96 的 {sig_cnt(c2,'A1_nested - A0')[0]}/6;A2−A1 平均 {sig_cnt(c2,'A2_nested - A1_nested')[2]*100:.3f}%/期,t>1.96 的 {sig_cnt(c2,'A2_nested - A1_nested')[0]}/6。差异方向一致为正,但幅度小、统计证据弱且模型间不一致。
5. **加上现实流动性/准入约束后,毛收益大幅缩水**(C3:排除科创板/北交所、候选与买入要求 20 日成交额≥1 亿、单日参与率≤5%):6 模型平均毛年化 A0 {pc(mC(c3,'A0','ann_gross'),1)}、A1 {pc(mC(c3,'A1_nested','ann_gross'),1)}、A2 {pc(mC(c3,'A2_nested','ann_gross'),1)};净年化 A0 {pc(mC(c3,'A0','ann_net'),1)}、A1 {pc(mC(c3,'A1_nested','ann_net'),1)}、A2 {pc(mC(c3,'A2_nested','ann_net'),1)};三种策略相对等权基准的平均超额均为负。这与 P2-A 的发现一致(amt20≥1e8 后多头腿≈0)。资金放大到 1 亿(C4)后 A0/A1 更差。**因此在显式成本和现实约束下,没有任何一个组合策略显示出可交易的净 alpha。**
6. **可信度与披露:** (a) A1 的嵌套选参在 {n1}/{n1t} 个(模型×场景×块)中选到网格边界 K_hold=150,A2 在 {t10}/{len(n2)} 个中选到 τ=1.0(网格上界),说明网格偏窄、"越少交易越好"的趋势没有被网格包住;(b) 显式口径中的半价差收缩系数 0.25 是在看到未收缩 Corwin–Schultz 估计中位数 0.43%(不合理)之后设定的,对全部策略统一,并在 0.5×/2× 下做了敏感性(§10),A0/A1 的净收益对价差假设非常敏感;(c) 嵌套评估是"按块调度的真实持仓续跑",选参依据是固定参数全程跑的反事实,二者有状态差异(§1.6);(d) 全部 2025–2026 数字为探索性。

## 1. 设计与实现

### 1.1 复用 P0 执行器、区分"决策"与"执行"
- 执行引擎 `b_sim.simulate_ext` 是对 P0 `p0lib.simulate` 状态机的**扩展副本**,默认参数下与 P0 **逐位相同**:信号日收盘 s 决策 → s+1 收盘按未复权收盘价成交 → 5 日节奏;新买入等权 `tv=nav/目标数`(受现金缩放);保留的名字不再平衡;买入遇停牌/涨停封死/无价/不足一手被拒绝且**不替补**;卖出遇停牌/跌停被阻塞并**逐日重试**;每边费用 `cost_rt/2`。A0 对账(`results\\b_reconcile_A0.csv`,{len(REC)} 项):`simulate_ext` 与 `p0lib.simulate` 差 0;与冻结的 P0/track-B 6 模型 c30 期收益最大绝对差 {rec_max:.1e};冻结 `in_top50` 集合对称差 0;等权基准与 P0 差约 1e-16。
- **决策只看信号日 s 收盘已知信息**:当日持仓快照、截至 s 的分数/因子/排名(`ctx` 只含 `k,s,d,units,cash,pend`,不含价格/成交量数组);**执行结果(涨跌停、停牌、部分成交、费用)只在 s+1 起的执行日发生**。执行成本、参与率上限和流动性过滤用的是 d−1 及更早的 ADV/波动率/价差。决策日志 `results\\b_logs\\decisions_*.csv.gz` 与订单日志 `orders_*.csv.gz` 分开存放。
- 新增的执行特性(默认关闭则与 P0 完全等价):再平衡模式(A2:按目标权重买卖,卖出先于买入,不加杠杆、现金≥0)、参与率上限(部分成交,卖出剩余部分逐日续卖、买入不追补)、最小成交额、显式成本。

### 1.2 三个策略
- **A0**:朴素 Top50 等权;每期取候选池内分数最高的 50 只,与 P0 baseline_fixed 一致。
- **A1(滞后/缓冲)**:买入线 rank≤50;**已持有名字只要今日(候选池内)排名≤K_hold 就保留**;目标始终**恰好 50 只**(池不足时取池大小)。**补仓规则**:被剔除/空出的名额由"当日排名最高且当前未持有"的名字按排名顺序补齐,不按现金改变只数、不留空位、不对保留名字加仓。K_hold=50 与 A0 完全相同(单元测试验证)。
- **A2(成本感知、风险约束优化,自写对偶求解器,无 cvxpy)**:最大化 Σ[α_j w_j − ½ a_j w_j² − τ·κ⁺_j(w_j−w0_j)⁺ − τ·κ⁻_j(w0_j−w_j)⁺],约束 0≤w≤cap、Σw≤1(无杠杆、仅多头)、可选 |X'w|≤δ(标准化 log 成交额20日/波动率20日/5日收益暴露);α=β̂·z,z 为候选池内分数的均匀秩标准化;**β̂ 为"相对收益标签对 z 的逐期斜率"的扩张均值,只用已成熟标签的期(m≤k−2)**,前 8 期用声明常数 β0=0.003、下限 0.0005;风险为对角阵(5·vol20²,风险厌恶 λ=25),κ 取执行所用成本模型的分边费率;候选=分数前 100 名+现有持仓;小于 0.2% 净值的调仓不执行。求解器与 SLSQP 在小例上对照(测试)。

### 1.3 成本口径(两套,互不叠加)
- **flat 0.30% 往返**:每边 0.15%(P0 约定),只有 `flat` 一个成本分量。
- **显式 OHLCV 成本(C2–C4)**,对每笔**实际成交额**逐边计算:佣金 {PR['explicit_cost']['commission_bp']}bp/边 + 过户费 {PR['explicit_cost']['transfer_bp']}bp/边 + **印花税仅卖出**({PR['explicit_cost']['stamp_bp_before_2023_08_28']:.0f}bp,2023-08-28 起 {PR['explicit_cost']['stamp_bp_from_2023_08_28']:.0f}bp)+ **半价差** = 0.25×(20 日 Corwin–Schultz 全价差的一半),下限为半个最小变动价位/价格,全价差上限 2% + **冲击** = 0.5×vol20×√(成交额/ADV20),上限 5%。信息集为 d−1。四类分量在订单日志中逐笔记录并求和为总费用(单元测试验证无重复计费)。**披露:0.25 的半价差收缩系数是在看到未收缩估计过高(中位 0.43%)后设定的,不是为了策略排序而调;§10 做了 0.5×/2× 敏感性。**

### 1.4 场景与开关(默认值及理由)
| 场景 | 成本 | 资金 | 参与率上限 | 最小成交额 | 科创板/北交所 | 说明 |
|---|---|---|---|---|---|---|
| C1_flat | flat 0.30% | 1000 万 | 关 | 关 | 允许 | 与 P0/A3 对账口径一致 |
| C2_explicit | 显式 | 1000 万 | 关 | 关 | 允许 | 与 P0 候选池一致,只换成本口径 |
| C3_realistic | 显式 | 1000 万 | 5% × 前日 ADV20 | 1 亿(候选与买入) | **排除** | 保守的零售账户默认:无科创/北交所权限、不碰小成交额 |
| C4_capacity_1e8 | 显式 | 1 亿 | 同 C3 | 同 C3 | 排除 | 容量检验 |

涨停买不进、跌停卖不出、停牌不成交在所有场景中始终生效(P0 规则)。C1/C2 允许科创/北交所是为了 A0 与 P0 可对账;这意味着 C1/C2 含有现实中零售账户可能无法交易的名字。

### 1.5 预先声明的参数网格与嵌套时间验证
协议在主运行前写入 `results\\b_protocol.json`(声明时间 {PR['declared_at']})。网格:A1 K_hold∈{{65,80,100,150}};A2 τ∈{{0.25,0.5,1.0}}×cap∈{{3%,5%}}×δ∈{{无,0.3}} 共 12 组。**扩张窗口嵌套验证**:前 52 期用默认参数(A1 K=80;A2 τ=0.5,cap=5%,δ无)做预热;之后每 26 期为一个检验块(k=52–77、78–103、104–129、130–155、156–181、182–206),块内参数取**块起点之前所有期**净夏普最大者。评估窗为 k≥52(共 155 期,2023 年中至 2026-09)。测试验证:把块起点及之后的收益篡改,选参结果不变。

### 1.6 嵌套设计的已知缺陷(诚实披露)
选参依据是"每个参数固定全程运行"的反事实收益序列,而最终评估是"按块调度、真实持仓续跑"的序列;换参数时组合从上一个参数的持仓状态过渡,会产生与固定参数序列不完全一致的换手/成本,因此嵌套结果并不严格等于任何一个固定参数的回测。§7 给出"嵌套 vs 事后最优固定参数"的差距(嵌套平均比事后最优低:A1 {opt.get('A1',np.nan):.3f}、A2 {opt.get('A2',np.nan):.3f} 个夏普单位,每期口径)。

### 1.7 换手 vs 预测的归因方法
- **直接拆分**:Δ净(X−A0) = Δ毛 + Δ成本节省(成本不复利,残差很小)。
- **匹配换手对照**:对 A1_nested 和 A2_nested 各自的评估期平均换手,构造两类用同一冻结分数的对照:**R(随机保留)**——每个仍在池中的持仓以概率 p 保留、与排名无关;**S(随机跳过)**——以概率 q 刷新为 Top50,否则整个组合不动。p/q 用二分法(8 步、单一校准种子)标定,20 个种子取平均。**校准偏差**:S 的单种子校准系统性偏低(实现换手比目标低 {abs(cal.loc['S','max'])*100:.0f}%~{abs(cal.loc['S','min'])*100:.0f}%,均值 {cal.loc['S','mean']*100:.0f}%),R 偏高(范围 {cal.loc['R','min']*100:.0f}%~{cal.loc['R','max']*100:.0f}%,均值 {cal.loc['R','mean']*100:.0f}%);因此另提供**精确换手校准版**:把对照的逐期收益与 A0 线性混合 `ret_adj = ret_ctrl + λ(ret_A0 − ret_ctrl)`(λ 由换手差决定,S 的 λ 均值 {lamS['mean']:.2f}、最大 {lamS['max']:.2f};R 的 λ 范围 {lamR['min']:.2f}~{lamR['mean']:.2f}),报告中的归因数字均用校准版,原始版保留在 CSV。线性混合近似本身是一个假设。
- **恒等式**:Δ净 = [net(对照)−net(A0)](换手效应)+[net(X)−net(对照)](选股效应)。选股效应另做 HAC t。

## 2. A0 对账与引擎检查
- A0 与 P0:`simulate_ext` vs `p0lib.simulate` 差 0;冻结 P0/track-B c30 收益最大绝对差 {rec_max:.1e};`in_top50` 对称差 0。
- 现金最小值在所有运行(含嵌套调度和对照)中均为正;求解器失败 0 次。
- 订单状态:FILLED、PARTIAL_FILLED(参与率上限)、REJECTED(停牌/涨停/不足一手/低于最小成交额)、BLOCKED_RETRY(跌停/停牌卖出)。

## 3. 结果:flat 0.30% 往返(C1,评估窗 k≥52,155 期,年化因子 252/5)
{perf_table(c1)}
## 4. 结果:显式 OHLCV 成本(C2)
{perf_table(c2)}
### 4.1 成本拖累分解(年化,占净值,单利)
{decomp_table()}
显式口径下 A0 的拖累以半价差({pc(mC(c2,'A0','drag_spread_annual'),1)})与冲击({pc(mC(c2,'A0','drag_impact_annual'),1)})为主,佣金+过户 {pc(mC(c2,'A0','drag_commission_annual')+mC(c2,'A0','drag_transfer_annual'),1)}、印花税 {pc(mC(c2,'A0','drag_stamp_annual'),1)}(6 模型均值)。显式总成本约为 flat 口径的 {mC(c2,'A0','cost_drag_annual')/mC(c1,'A0','cost_drag_annual'):.1f} 倍,直接决定"A0 是否还有净收益"。

### 4.2 分年度(HGB/LGBM,C2;净收益均值 / 相对等权超额均值,%/5日;2025–2026 为探索性)
{year_table()}
## 5. 因子调整净 alpha(超额=策略期收益−等权基准零成本收益;对 SMB/REV5/LVOL/MOM60/ILLIQ 回归,HAC lag 4)
### 5.1 C2 显式成本
{fa_table(c2)}
### 5.2 C1 flat 0.30%
{fa_table(c1)}
说明:因子收益为 P2-A 的无成本多空因子收益;策略净收益含成本,所以净 alpha 是"含成本后的残差收益"。A2 的 R² 显著更高(组合更分散、风格暴露更集中),毛 alpha 也为负,因此其净收益改善不能解释为 alpha。

## 6. 配对检验(评估窗 k≥52;嵌套选参的 A1/A2 vs A0;HAC lag 4;5 期 block bootstrap B=4000)
### 6.1 C2 显式成本
{paired_table(c2)}
### 6.2 C1 flat 0.30%
{paired_table(c1)}
## 7. 嵌套验证选中的参数
{nested_table()}
统计:A1 在 {n1}/{n1t} 个(模型×场景×块)中选到 K_hold=150(网格上界);A2 在 {t10}/{len(n2)} 个中选 τ=1.0(上界),{dof}/{len(n2)} 个选 δ=无(无暴露约束)。**选到网格边界意味着最优值可能在网格之外,这个"最优 K_hold/τ"不应被当作已被识别的参数。** 嵌套夏普低于事后最优固定参数(最后几栏对比),说明选参本身有成本;"事后最优"仅作乐观上界,不是可实现结果。

## 8. 换手 vs 预测归因
### 8.1 C2 显式成本(Δ 为每 5 日期的均值,相对 A0)
{attr_table(c2)}
### 8.2 C1 flat 0.30%
{attr_table(c1)}
### 8.3 风格暴露(C2,k≥52,决策时点加权的候选池内标准化 z)
{exp_table()}
A2 相对 A0 大幅降低波动率暴露(z 约 −0.5~−0.8,A0 约 +0.1~+0.3),成交额暴露在 HGB/LGBM 上略偏小;A1 与 A0 暴露接近。δ 暴露约束在嵌套中很少被选中({dof}/{len(n2)} 选"无"):δ=0.3 的组合毛收益显著更低(固定网格见 `b_grid_fixed_params.csv`)。

## 9. 现实约束与容量(C2→C3→C4)
{c34_table()}
成交/占用诊断(6 模型均值):
{fill_table()}
## 10. 成本假设敏感性(C2,HGB/LGBM,按相同调度重跑;净年化 ≈ 每期均值×50.4)
{stress_table()}
## 11. 局限与诚实说明
1. **这是对已冻结分数的"事后组合化"研究。** 嵌套验证只防止在整个评估期上调参;冻结分数本身、因子、候选池规则都经过前期阶段的多次观察,2025–2026 为复用数据,**探索性**,不是盲测。
2. **显式成本是假设,不是实盘成交数据。** 价差来自 OHLC 的 Corwin–Schultz 估计(并人为收缩 0.25),冲击为平方根模型,参数未校准到真实成交;显式口径下 A0 的净结论对价差假设高度敏感(§10)。1000 万 / 1 亿资金的容量结论取决于 5% 参与率与 ADV20 的定义。
3. **A2 是简化的:** 对角风险、候选限前 100 名+持仓、β̂ 用扩张均值斜率、暴露约束只有三列;没有行业/市值中性、没有协方差估计。A2 持仓约 100–170 只,手数取整与小额订单可能影响实际执行(已用 min_trade_frac 与手数规则处理,未模拟盘口)。
4. **嵌套设计见 §1.6;网格边界选择见 §7;** 匹配换手对照的校准偏差见 §1.7。R 对照是苛刻的稻草人,S 对照更合理但仍是"同换手的陈旧 Top50",都不是对"最佳可能的低换手选股"的度量。
5. **C1/C2 允许科创板/北交所**(为对账 P0);C3/C4 排除。若账户无相应权限,C1/C2 的 A0 毛收益含有不可交易部分。
6. **成交模型:** 收盘价成交、无盘口;涨停买不进/跌停卖不出/停牌按 P0 规则;未模拟集合竞价、日内约束、融资融券、分红税、最低佣金。
7. **统计功效低:** 155 个 5 日期、约 3 年,HAC t 与 block bootstrap 区间很宽;"不显著"不等于"没有效果"。
8. 本报告是研究分析,**不是投资建议**;分数不是概率、也不是收益预测。

## 12. 文件清单
- 脚本:`b_sim.py`(引擎/成本/决策器/求解器)、`b_data.py`、`b_protocol.py`、`b_lib.py`、`b_run.py`、`b_launch.py`、`b_bench.py`、`b_agg.py`、`b_exposure.py`、`b_figs.py`、`b_report.py`;`p2lib.py` 追加了 HAC OLS 与 alpha bootstrap。
- 结果:`results\\b_protocol.json`、`b_reconcile_A0.csv`、`b_ew_benchmark.csv`、`b_runs\\*.parquet|_extras.json`(24 个作业)、`b_performance.csv`、`b_paired.csv`、`b_turnover_vs_prediction.csv`、`b_factor_alpha.csv`、`b_nested_selection.csv`、`b_nested_optimism_check.csv`、`b_grid_fixed_params.csv`、`b_fill_diagnostics.csv`、`b_cost_stress.csv`、`b_controls_calibration.csv`、`b_exposures.csv`、`b_logs\\`(订单与决策日志)。
- 图:`figures\\b_fig1_nav.png`(净值)、`b_fig2_cost_decomposition.png`、`b_fig3_turnover_vs_net.png`、`b_fig4_attribution.png`、`b_fig5_paired_forest.png`、`b_fig6_factor_alpha_t.png`、`b_fig7_constraints_capacity.png`、`b_fig8_cost_stress.png`。
- 测试:`tests\\test_p2b.py`。
- 复现:`python b_protocol.py; python b_bench.py; python b_launch.py; python b_agg.py; python b_exposure.py; python b_figs.py; python b_report.py`。
"""
open(Q.REP / "p2b_cost_aware_report.md", "w", encoding="utf-8", newline="\n").write(rep); print(len(rep))