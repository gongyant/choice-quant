"""P1-A3: generate p1_dl_report.md from the final data files (numbers read from files; fixed P0 facts quoted from p0_audit_report.md)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent.parent; V2 = H.parent
rd = lambda p, **k: pd.read_csv(H / p, encoding="utf-8-sig", dtype={"code": str}, **k)
V = pd.read_csv(H / "latest_all_stocks_20261016_v2.csv", dtype=str, encoding="utf-8-sig", keep_default_na=False)
E = V[V.eligible_asof == "True"].copy()
for c in E.columns:
    if c.startswith(("rank_", "pctile_", "r5", "r20", "r60", "vol20", "amount20", "turnover20")) or c.endswith("_score") or c in ("model_agreement_pctile_std", "model_pctile_mean_5", "n_models_top50_of5"): E[c] = pd.to_numeric(E[c])
n = len(E); MN = {"hgb": "HGB", "lgbm": "LightGBM", "dl_mlp": "MLP", "dl_gru": "GRU", "dl_tf": "Transformer"}; five = list(MN)
cards = json.load(open(H / "scores" / "stock_cards_data_20261016_v2.json", encoding="utf-8")); reg = pd.read_csv(H / "forward_registry_20261016.csv", dtype={"code": str}, encoding="utf-8-sig")
def fl(s): s = s if isinstance(s, str) else ""; return "; ".join(x for x in s.split(";") if x and any(t in x for t in ["涨停", "跌停", "ST", "单源", "未解释", "无法核验", "不足"])) or "-"
def full(k, which):
    t = E.sort_values(f"rank_{k}", ascending=(which == "Top")).head(20); L = ["| 排名 | 代码 | 名称 | 行业(巨潮最新) | score | 近5日 | 近20日 | 20日成交额(亿) | 5模型Top50计数 | 质量/涨跌停标记 |", "|---:|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in t.itertuples(): L.append(f"| {int(getattr(r, 'rank_' + k))} | {r.code} | {r.name} | {r.industry} | {getattr(r, k + '_score'):.6f} | {r.r5:+.1%} | {r.r20:+.1%} | {r.amount20 / 1e8:.2f} | {int(r.n_models_top50_of5)} | {fl(r.quality_flags)} |")
    return "\n".join(L)
def brief(k, which):
    t = E.sort_values(f"rank_{k}", ascending=(which == "Top")).head(20); L = ["| 排名 | 代码 | 名称 | 行业 | score | 标记 |", "|---:|---|---|---|---:|---|"]
    for r in t.itertuples(): L.append(f"| {int(getattr(r, 'rank_' + k))} | {r.code} | {r.name} | {r.industry} | {getattr(r, k + '_score'):.4f} | {fl(r.quality_flags)} |")
    return "\n".join(L)
top = {k: list(E.sort_values(f"rank_{k}").head(20).code) for k in five}; bot = {k: list(E.sort_values(f"rank_{k}", ascending=False).head(20).code) for k in five}
def ov(d): return pd.DataFrame([[len(set(d[a]) & set(d[b])) for b in five] for a in five], index=[MN[a] for a in five], columns=[MN[a] for a in five]).to_markdown()
ge3 = rd("reports/multi_model_top50_20261016.csv"); mc = rd("reports/model_comparison_v2.csv"); cs = rd("reports/cost_sensitivity_all_models.csv"); by = rd("reports/by_year_all_models.csv"); pp = rd("reports/paired_tests_all_models.csv"); ss = rd("sector_summary_v2.csv"); r20 = rd("reports/recent20_summary_v2.csv"); rc = rd("reports/reconciliation_dl.csv")
six = ["HGB", "LGBM", "Ridge", "MLP", "GRU", "TF"]; mcs = mc.set_index("strategy")
def pct(x, d=2): return "-" if pd.isna(x) else f"{x * 100:.{d}f}%"
L = []; A = L.append
A("# P1 研究报告:五模型研究排名看板 + 模型比较(Choice A股 1500 样本)"); A("")
A(f"> **研究排名,不是买入/卖出建议。模型 score 是原始输出,不是概率,也不是收益率。** 数据截至信号日 **2026-10-16 收盘**(报告生成于 2026-10-19,Asia/Singapore);排名范围 = 信号日合格(eligible_asof)股票 **{n}** 只,来自 1500 只分层抽样样本(含 208 只已退市),**不是全市场**。")
A("> **诚实结论先说**:修复 P0 数据/成交问题后,HGB / LightGBM / Ridge / MLP / GRU / Transformer 六个模型在 0.30% 往返成本下的 Top50 策略,**没有任何一个相对同制度等权基准的净超额在统计上显著**(朴素 t 与 HAC(4) t 均在 ±1.7 以内,bootstrap 区间含 0);所有两两配对差异也不显著。Rank IC(≈0.07)为正说明排序里有一点微弱信息,但被成本基本吃掉。2025–2026 数据已多次用于模型比较,**不是盲测**;真正的前向验证从 2026-10-19 之后的新数据开始(见第 8 节与 `forward_registry_20261016.csv`)。"); A("")
A("## 1. 研究看板(截至 2026-10-16 收盘)"); A("")
A("- HGB / LightGBM:P0 冻结模型(同一套 11 个横截面分位特征;训练 2023-10-17~2026-09-22,标签最晚到期 2026-10-08)。MLP(3 种子均值)、GRU(3 种子均值,GPU 训练)、Transformer(**仅种子 42**):轨道B最新模型(训练截至 2026-06-16,验证截至 2026-09-22,标签最晚到期 2026-10-08)。所有训练标签到期日均早于信号日。")
A("- \"5模型Top50计数\" = 该股同时进入 HGB/LGBM/MLP/GRU/TF 各自 Top50 的模型个数;\"涨停封板\" = 最新日收盘封板(次日开盘大概率买不进);ST 名称的涨跌停按 5% 近似(无历史 ST 清单)。"); A("")
A("### 1.1 HGB Top20"); A(""); A(full("hgb", "Top")); A(""); A("### 1.2 LightGBM Top20"); A(""); A(full("lgbm", "Top")); A("")
for i, (k, t) in enumerate([("dl_mlp", "MLP(3种子均值)"), ("dl_gru", "GRU(3种子均值)"), ("dl_tf", "Transformer(单种子42)")]): A(f"### 1.{3 + i} {t} Top20"); A(""); A(brief(k, "Top")); A("")
A("### 1.6 Bottom20(得分最低;仅描述模型\"最不看好\",同样不是卖出/做空建议)"); A("")
for k in ["hgb", "lgbm"]: A(f"**{MN[k]} Bottom20**"); A(""); A(full(k, "Bottom")); A("")
for k in ["dl_mlp", "dl_gru", "dl_tf"]: A(f"**{MN[k]} Bottom20**"); A(""); A(brief(k, "Bottom")); A("")
A("### 1.7 五模型 Top20 / Bottom20 两两重合只数"); A(""); A("Top20:"); A(""); A(ov(top)); A(""); A("Bottom20:"); A(""); A(ov(bot)); A("")
rk = E[[f"rank_{k}" for k in five]].corr(method="spearman").round(3); rk.index = rk.columns = [MN[k] for k in five]; A("全体合格股票的排名 Spearman 相关:"); A(""); A(rk.to_markdown()); A("")
A("要点:HGB 与 LightGBM 高度一致(相关 0.93,Top20 重合 11 只);MLP 与树模型 Top20 几乎无重合(0~1 只);GRU 与其它模型排名相关仅 0.30~0.41;Transformer 居中。模型分歧很大本身说明**\"最新一天的排名\"并不稳定、不是共识**。"); A("")
A(f"### 1.8 至少 3 个模型同时进入 Top50 的股票(共 {len(ge3)} 只;仅描述模型一致性,**不是买入建议**)"); A(""); A("| 代码 | 名称 | 行业 | 模型数 | 入选模型 | HGB | LGBM | MLP | GRU | TF | 分歧(std) | 近5日 | 近20日 | 涨停封板 | 其它标记 |"); A("|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
for r in ge3.itertuples(): A(f"| {r.code} | {r.name} | {r.industry} | {r.n_models_top50_of5} | {r.models_in_top50} | {r.rank_hgb} | {r.rank_lgbm} | {r.rank_mlp} | {r.rank_gru} | {r.rank_tf} | {float(r.model_agreement_pctile_std):.3f} | {float(r.r5):+.1%} | {float(r.r20):+.1%} | {'**是**' if str(r.limit_up_locked_latest).startswith('是') else '否'} | {fl(r.quality_flags)} |")
A(""); A("说明:这些股票多为科创板/创业板/北交所小盘;**603668 天马科技最新日涨停封板**,即使排名靠前也大概率无法成交。模型输入同源(同一组量价信息),一致≠独立验证。\"分歧(std)\" = 五模型百分位的标准差,越小越一致,不是置信度。"); A("")
A("## 2. 三只个股研究卡片(完整版见 `reports\\stock_cards_20261016_v2.md`,图见 `figures\\card_*_v2.png`)"); A("")
A("| 个股 | 行业 | 近5/20/60日 | 状态 | HGB 名次/分位 | LGBM | Ridge | MLP | GRU | TF | 五模型分歧 |"); A("|---|---|---|---|---|---|---|---|---|---|---|")
for c, d in cards.items():
    f = lambda m: "**池外未打分**" if d["scores"].get(m) is None else f"第{d['ranks'][m]}名 / {d['pct'][m]:.0%}"
    A(f"| {c} {d['name']} | {d['industry']} | {d['r5']:+.1%} / {d['r20']:+.1%} / {d['r60']:+.1%} | {'池外打分' if not d['in_sample'] else '1500样本内'} | {f('HGB')} | {f('LGBM')} | {f('Ridge')} | {f('MLP')} | {f('GRU')} | {f('TF')} | {('%.3f' % d['agreement_std']) if d['agreement_std'] is not None else '不适用'} |")
A(""); A("- **600060 海信视像不在 1500 样本内**:HGB/LGBM/Ridge 为同一冻结模型的池外打分(特征分位数相对 2026-10-16 合格池计算,名次为插入后的名次);**MLP** 用轨道B保存的最新 MLP 检查点(3 种子均值)打分(同方法已在样本内股票上复现B的分数,误差 <1e-5);**GRU / Transformer 需要 60 日序列 + 全截面成交额 z 分数 + 检查点标准化参数,本轮没有为池外股票生成,故\"池外未打分\",不编造**。它没有任何历史 OOS 预测记录。")
A("- 所有卡片分位均为\"在合格池内的相对位置\",**不是上涨概率**。"); A("")
A("## 3. 行业表现(Rank IC:score 与 5 日相对收益的 Spearman,期均值;巨潮最新行业分类,**非历史时点**)"); A("")
ind = ss[ss.group_type.str.startswith("industry")]; pv = ind.pivot_table(index="group", columns="model", values="rank_ic_mean")[six]; ns_ = ind[ind.model == "HGB"].set_index("group").n_stocks; rel = ind[ind.model == "HGB"].set_index("group").ic_reliability
A("| 行业 | 样本股票数 | " + " | ".join(six) + " | 可靠性 |"); A("|---|---:|" + "---:|" * 6 + "---|")
for g in pv.sort_values("HGB", ascending=False).index: A(f"| {g} | {int(ns_[g])} | " + " | ".join(f"{pv.loc[g, m]:.3f}" for m in six) + f" | {'可用' if rel[g] == '可用' else '不可靠'} |")
A(""); A("- 各行业平均 Rank IC 几乎都为正(约 0.04~0.12),金融最弱(HGB 0.003,仅 28 只)、能源偏弱;\"未分类\"(多为已退市股)最高。分组差异多为噪声:12 个行业 × 6 个模型存在多重比较,**不能据此做行业配置**。")
ex = ss[ss.group_type.isin(["exchange", "board"])].pivot_table(index=["group_type", "group"], columns="model", values="rank_ic_mean")[six]; A(""); A("按交易所/板块:"); A(""); A("| 分组 | " + " | ".join(six) + " |"); A("|---|" + "---:|" * 6)
for (gt, g), r in ex.iterrows(): A(f"| {gt}:{g} | " + " | ".join(f"{r[m]:.3f}" for m in six) + " |")
A(""); A("北交所 Rank IC 最高(HGB 0.125)但股票少、涨跌停(30%)特殊;科创板最低(0.046)。以上是 2022-07~2026-09 的 OOS 重放,**非盲测**。完整表:`sector_summary_v2.csv`。"); A("")
A("## 4. 最近 20 期(2026-05-14 ~ 2026-09-28,约 100 个交易日;5 日调仓,Top50,往返 0.30%)"); A(""); A("| 模型 | IC均值 | RankIC均值 | RankIC>0 期数/20 | 净收益均值/期 | 20期累计净收益 | 同期等权累计 | 净超额均值/期 |"); A("|---|---:|---:|---:|---:|---:|---:|---:|")
for r in r20.itertuples(): A(f"| {r.model} | {r.IC_mean:.4f} | {r.RankIC_mean:.4f} | {r.RankIC_pos_periods} | {pct(r.net_ret_mean_per_period)} | {pct(r.net_cum)} | {pct(r.ew_cum)} | {pct(r.net_excess_mean)} |")
A(""); A("20 期太少,仅描述:这一段市场整体下跌(等权累计 -17%),所有模型净收益为负;GRU 最近 20 期 RankIC≈0、累计 -36.5% 最差,但这是单一窗口(GRU 种子波动本身很大),不能解读为模型失效或优劣。逐期明细 `reports\\recent20_metrics_v2.md`。"); A("")
A("## 5. 模型比较(P0 同一套 walk-forward:207 期,2022-07-01 ~ 2026-09-28;往返 0.30%;Top50,1000万本金,100 股整手)"); A("")
A("| 模型 | 年化 | 夏普 | 最大回撤 | RankIC | 净超额均值/期 | 朴素 t | HAC(4) t | Bootstrap 95% |"); A("|---|---:|---:|---:|---:|---:|---:|---:|---|")
for m in six + ["EqualWeight"]:
    r = mcs.loc[m]
    if m == "EqualWeight": A(f"| 同制度等权(含成本) | {pct(r.annualized_return)} | {r.sharpe:.2f} | {pct(r.max_drawdown, 1)} | - | - | - | - | - |")
    else: A(f"| {m} | {pct(r.annualized_return)} | {r.sharpe:.2f} | {pct(r.max_drawdown, 1)} | {r.RankIC:.4f} | {pct(r.net_ex_excess_mean)} | {r.net_ex_t:.2f} | {r.net_ex_hac4_t:.2f} | [{pct(r.net_ex_boot_lo)}, {pct(r.net_ex_boot_hi)}] |")
A(""); A("(MLP/GRU 为 3 种子均值分数,TF 为单种子 42;净超额 = 策略单期净收益 - 同制度含成本等权单期净收益;数据来自 `reports\\model_comparison_v2.csv` = 轨道B/P0 的 side_by_side_c30.csv。)"); A("")
A("种子明细:"); A(""); A("| 模型 | 年化 | 夏普 | 最大回撤 | RankIC | 净超额 t |"); A("|---|---:|---:|---:|---:|---:|")
for m in ["MLP_s42", "MLP_s43", "MLP_s44", "GRU_s42", "GRU_s43", "GRU_s44", "TF_s42"]: r = mcs.loc[m]; A(f"| {m} | {pct(r.annualized_return)} | {r.sharpe:.2f} | {pct(r.max_drawdown, 1)} | {r.RankIC:.4f} | {r.net_ex_t:.2f} |")
A(""); A("GRU 三个种子年化从 -2.8% 到 +4.5%,种子间差异大于模型间差异;MLP 种子 42 年化 4.4%,43/44 约 9.7%。**单种子的 Transformer 数字无法判断随机性。**"); A("")
A("### 5.1 成本敏感性(年化收益 / 净超额 t;往返成本 0 / 0.10% / 0.15% / 0.30% / 0.60%)"); A(""); cs["cost_roundtrip"] = cs.cost_roundtrip.astype(float); costs = sorted(cs.cost_roundtrip.unique()); A("| 模型 | " + " | ".join(f"{c * 100:.2f}%" for c in costs) + " |"); A("|---|" + "---:|" * len(costs))
for m in six + ["EqualWeight"]:
    g = cs[cs.strategy == m].set_index("cost_roundtrip"); A(f"| {m} | " + " | ".join((f"{pct(float(g.loc[c, 'annualized_return']), 1)} / {float(g.loc[c, 'net_ex_t']):.2f}" if m != "EqualWeight" else pct(float(g.loc[c, 'annualized_return']), 1)) for c in costs) + " |")
A(""); A("零成本下 HGB/MLP/GRU/TF 的净超额 t 约 1.6~2.3(毛信号微弱存在),但 0.30% 往返成本时全部降到 <1,0.60% 时全部为负:信号被交易成本基本吃掉。"); A("")
A("### 5.2 分年复利净收益(0.30%)与 RankIC"); A(""); yc = by[by.strategy.isin(six + ["EqualWeight"])].pivot_table(index="year", columns="strategy", values="compounded_net")[six + ["EqualWeight"]]; yi = by[by.strategy.isin(six)].pivot_table(index="year", columns="strategy", values="RankIC")[six]
A("复利净收益:"); A(""); A("| 年 | " + " | ".join(six + ["等权"]) + " |"); A("|---|" + "---:|" * 7)
for y, r in yc.iterrows(): A(f"| {y} | " + " | ".join(pct(v, 1) for v in r.values) + " |")
A(""); A("RankIC(期均值):"); A(""); A("| 年 | " + " | ".join(six) + " |"); A("|---|" + "---:|" * 6)
for y, r in yi.iterrows(): A(f"| {y} | " + " | ".join(f"{v:.4f}" for v in r.values) + " |")
A(""); A("2025 年(市场大涨)所有模型净收益大幅为正,2026 年(至 9 月)全部为负;年度间差异远大于模型间差异,收益主要来自市场/风格暴露而不是稳定的选股超额。"); A("")
A("### 5.3 配对检验(模型相减,逐期差)"); A(""); A("| 比较 | 均值差 | t | p | HAC(4) t | Bootstrap 95% | n |"); A("|---|---:|---:|---:|---:|---|---:|")
for r in pp.itertuples(): A(f"| {r.comparison} | {float(r.excess_mean):+.5f} | {float(r.t):.2f} | {float(r.p):.3f} | {float(r.hac4_t):.2f} | [{float(r.boot_lo):+.5f}, {float(r.boot_hi):+.5f}] | {r.n} |")
A(""); A(f"全部 {len(pp)} 个配对比较中,|HAC t| 最大为 {pp.hac4_t.astype(float).abs().max():.2f}(<2),**没有任何一对差异显著**:不能说 MLP/GRU/Transformer 优于或劣于 HGB/LightGBM。"); A("")
A("### 5.4 累计净值与对账"); A(""); A("图:`figures\\cumulative_net_models.png`(对数轴)。各 DL 模型逐期净收益已用同一模拟器从 OOS 预测文件独立重放,与轨道B `period_returns.csv` / 订单日志对账(最大绝对差):"); A(""); A("| 检查 | MLP | GRU | TF |"); A("|---|---:|---:|---:|")
for chk in rc.check.unique():
    g = rc[rc.check == chk].set_index("model").max_abs_diff.astype(float); A(f"| {chk} | " + " | ".join(f"{g.get(m, np.nan):.1e}" for m in ["MLP", "GRU", "TF"]) + " |")
A(""); A("DL 训练设计(预注册,无调参):MLP 11 维分位特征;GRU/Transformer 60 日序列(6 通道+mask);训练窗口 3 年、purge 12 个交易日、每 4 期再训练,验证集为窗口内最后 60 日并与训练段隔 10 日。最新日模型的训练曲线很短(GRU/Transformer 最佳 epoch 偏早,验证 MSE >1,几乎没学到可泛化的信息),见 `dl\\dl_models_report.md`。"); A("")
nflag = lambda s: int(V.quality_flags.str.contains(s, regex=False).sum()); A("## 6. 数据质量"); A("")
A("### 6.1 P0 修复摘要(详见 `p0\\p0_audit_report.md`)"); A("")
A("- P0-A 信号池无前视(eligible_asof_t 只用 t 日及以前信息);P0-B 统一交易日历(1400 个交易日)与持仓状态机(涨停买不进→留现金,不换股;跌停/停牌卖不出→顺延重试);P0-C 用不复权价成交、新浪乘法型复权因子构造收益;P0-E 标签/池/基准同一池;P0-F 涨跌停用不复权前收、除权参考价与板块比例。无前视单测 45/45 通过(含 4 个故意前视必须失败的负向测试)。")
A("- 旧报告 HGB \"年化 13.9%、t=2.65\" 不可信(毛收益口径,且腾讯加法式前复权伪影污染);修复后同口径 HGB 年化 7.9%、净超额 t=0.72。若仍用旧腾讯前复权数据,HGB 年化为 -7.6%——数据伪影会改变结论方向。")
A("- 3 只无效价股票(600188/600066/000408)根因:腾讯前复权为加法式,早期价格变成 ≤0;已改为不复权价 + 新浪乘法因子。"); A("")
A("### 6.2 异常事件与已知数据局限"); A("")
A("- 异常事件日志 `p0\\quality_event_log.csv`:**1272 条 / 266 只**(加法式前复权伪影 1086 + 前复权非正价 72 已修正;停牌复牌后可能无涨跌停 104 条,保留并标记;**修正后仍无法解释 10 条**,9 条为除权参考价 1 个最小价位的取整偏差,1 条康美药业 2021-12-15 -10.7% 需人工核对)。")
A(f"- **价格单源**:{nflag('价格单源')} 只样本股票没有独立第二来源;沪深其余股票新浪/腾讯不复权收盘价逐日一致,但**复权因子仅新浪单源**。")
A(f"- **成交量单位无法核验**:920196(仅 1 根 bar)、600317(仅 9 根 bar)按源默认单位并标记(只影响量比特征)。无换手率(腾讯源)的 {nflag('无换手率')} 只,turnover 特征按 0 处理。")
A("- **已退市股票占样本 13.9%**(208/1500,其中 2 只 600485、600677 无任何行情数据,在主表中无分数);总体退市占比约 3.6%,样本对退市股过采样;**无法获得完整历史退市/上市池**(北交所退市名单缺失),存在幸存者/抽样偏差。")
A(f"- **ST 为近似**:没有历史 ST 清单,仅用名称 + 主板收盘恰好 ±5% 限价的保守近似;最新 {nflag('ST/*ST名称')} 只名称含 ST;漏判率未量化(阻断项)。")
A(f"- 覆盖率 <95% 的股票 {int(V.quality_flags.str.contains('数据覆盖率').sum())} 只;含未解释异常 {nflag('含未解释异常')} 只;最新日涨停封板 {nflag('最新日涨停封板')} 只、跌停封板 {nflag('最新日跌停封板')} 只。")
A("- 首个 walk-forward 测试日为 2022-07-01;停牌股按主日历保留为缺口;行业为巨潮最新分类,**非历史时点**;同花顺(THS)行业/概念无成分数据,缺失。"); A("")
A("## 7. 风险与局限"); A("")
for t in ["**2025–2026 不是盲测**:这段数据已被用于模型比较、成本/参数讨论与诊断多次,所有\"OOS\"数字只是 walk-forward 重放。", "**幸存者/抽样偏差**:1500 只为分层抽样,含 13.9% 已退市股;不代表全市场,不能外推。", "**行业非历史时点**;ST 近似;复权因子单源;成交量单位 2 只无法核验。",
          "**Transformer 单种子**(训练耗时超过预注册 30 分钟线,未补 43/44),GRU 种子间年化差 7 个百分点以上:DL 数字随机性大;MLP/GRU 为 3 种子分数均值。", "**GPU/CPU 差异**:MLP 在 CPU、GRU/Transformer 在 GPU(RTX 5070 Ti,cuDNN deterministic)训练,与 CPU 不保证逐位一致;对账只验证\"分数→净收益\"链路,不验证训练可复现。",
          "**成本为粗略假设**:往返 0.30%,滑点内含;未建模冲击成本、做空、盘口深度;涨停买不进/跌停卖不出已建模但不含极端连续一字板场景。", "**无调参、无因子/模型消融、无行业/市值中性化**:结果可能主要来自小盘/高波动暴露。",
          "**score 未校准**:仅用于排序,不是概率、不是收益预测;不同模型的 score 数值不可互比(MLP/GRU 尺度远大于 HGB)。", "**多重比较**:6 个模型 × 多指标 × 多子样本;任何\"某行业/某模型更好\"都需要在前向数据上重新验证。", "研究排名≠买入建议;本报告不构成投资意见;未下任何真实订单。"]: A(f"- {t}")
A(""); A("## 8. 前向验证(从 2026-10-19 起的新数据)"); A("")
A(f"- 本版本预测已冻结:`forward_registry_20261016.csv`(6 个模型 × Top50 = {len(reg)} 行:asof 日期、模型、排名、代码、名称、score、来源表 sha256)和 `forward_registry_20261016_manifest.json`(22 个关键文件的 sha256:v2 主表、DL 分数、冻结模型、DL 检查点、p0lib/p1lib/dl 训练代码)。v2 主表 sha256 = `{reg.source_table_sha256.iloc[0]}`。")
A("- 评估规则(预先声明):信号 2026-10-16 收盘 → 2026-10-19 收盘买入 → 2026-10-26 收盘卖出的相对收益(减合格池等权);指标为 Rank IC、Top50 相对收益、扣 0.30% 后的净超额(报告 HAC/bootstrap);**不得在看到前向结果后改模型/参数/阈值**;之后每周新信号必须使用同一冻结版本,或以新版本号单独登记。")
A("- 至少需要 ≥30 个独立 5 日期(约 150 个交易日,2027-05 前后)才有初步统计功效;在此之前前向结果仅为描述。"); A("")
A("## 9. 结论(诚实版)"); A("")
for t in ["**没有模型有统计显著的扣费净超额。** 六个模型在 0.30% 往返成本下,净超额/期为 -0.21%~+0.12%,朴素 t 与 HAC t 均在 ±1.7 以内:HGB +0.10%(t 0.72)、LGBM +0.09%(0.64)、MLP +0.12%(0.85)、GRU +0.10%(0.52)、TF +0.07%(0.46)、Ridge -0.21%(-1.67);bootstrap 区间均含 0。",
          "**Rank IC ≈ 0.07 为正**(六个模型相近),说明量价特征对 5 日横截面排序有微弱信息;但 Top50 组合的毛超额太小,0.30% 成本下被吃掉,0.60% 时全部为负。", "**深度学习没有带来可证明的改进**:MLP/GRU/Transformer 与 HGB/LightGBM 的配对差全部不显著;GRU 不同种子年化相差 >7 个百分点,噪声比模型差异大。",
          f"最新日排名(第 1 节)模型间分歧很大,仅 {len(ge3)} 只股票被 ≥3 个模型同时放进 Top50,其中有涨停封板的情形;**这份看板应当作研究线索,而不是交易清单**。", "下一步建议:用前向数据验证冻结版本;之后再做因子消融、市值/行业中性化、更严格的成本模型,并扩展样本到全市场(含完整退市池)再下结论。"]: A(f"- {t}")
A(""); A("## 10. 文件与复现"); A("")
A("- 主表 `latest_all_stocks_20261016_v2.csv`(1500 行,A1 原表不改);`historical_predictions_all_models.parquet`(1,860,516 行,键 (signal_date, code, model_id) 唯一);`per_stock_summary_v2.csv`;`sector_summary_v2.csv`;`strategy_trades_v2.csv`;`reports\\{top_bottom_20261016_v2, recent20_metrics_v2, stock_cards_20261016_v2, reconciliation_dl}.md`;`stock_dashboard_v2.py` + `run_dashboard.ps1`(v1 保留为 `stock_dashboard.py` / `run_dashboard_v1.ps1`)。")
A("- 复现:`.venv\\Scripts\\python.exe a3\\a3_latest.py` → `a3\\a3_history.py` → `a3\\a3_cards.py` → `a3\\a3_report.py`;A1/A2 脚本与轨道B训练脚本见 `review_bundle.zip` 内关键代码;`README_bundle.md` 含审计提问清单。"); A("")
(H / "p1_dl_report.md").write_text("\n".join(L), encoding="utf-8"); print(len(L), "lines", (H / "p1_dl_report.md").stat().st_size)
