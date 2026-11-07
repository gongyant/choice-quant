import re
from pathlib import Path
H = Path(r"D:\choice_quant_starter\v2\p1_dl"); L = (H / "stock_dashboard.py").read_text(encoding="utf-8").split("\n")
def rep(a, b, new):  # replace 1-based inclusive lines a..b
    L[a - 1:b] = new.split("\n") if new != "" else []
def line(n): return L[n - 1]
# ---- page 5 text (176) ----
assert "已知局限" in line(176); L[175] = line(176).replace("模型 score 未校准。", "模型 score 未校准;DL 模型:TF 仅种子42,GRU 在 GPU 训练(CPU/GPU 数值会略有差异),未做调参/消融,600060 池外无 GRU/TF 分数。")
# ---- page 4 (153-167) ----
P4 = '''    mc = load_csv("reports/model_comparison_v2.csv"); cs = load_csv("reports/cost_sensitivity_all_models.csv"); by = load_csv("reports/by_year_all_models.csv"); pp = load_csv("reports/paired_tests_all_models.csv"); cum = load_csv("reports/cumulative_net_models.csv"); sm = load_csv("historical_predictions_summary_v2.csv")
    six = list(MODELS); st.subheader("六模型 walk-forward 对比(207期,2022-07-01~2026-09-28,Top50,往返成本0.30%;P0 重放与 DL 均已对账)")
    t = mc[mc.strategy.isin(six + ["EqualWeight"])].copy(); t["strategy"] = pd.Categorical(t.strategy, six + ["EqualWeight"]); t = t.sort_values("strategy")
    st.dataframe(t[["strategy", "annualized_return", "annualized_vol", "sharpe", "max_drawdown", "win_rate", "RankIC", "RankIC_IR", "net_mean_per_period", "net_ex_excess_mean", "net_ex_t", "net_ex_hac4_t", "net_ex_boot_lo", "net_ex_boot_hi"]].round(4), hide_index=True, width="stretch")
    st.caption("net_ex_* = 相对同制度含成本等权的单期净超额(均值、朴素t、HAC(4)t、bootstrap 95%区间)。**没有任何模型的净超额统计显著**;Rank IC 为正但扣费后 Top50 与 0 无法区分。MLP/GRU=3种子均值分数,TF=单种子42。2025-2026 非盲测。")
    with st.expander("各种子明细(MLP/GRU 3种子,TF 种子42)"): st.dataframe(mc[mc.strategy.str.contains("_s4")].round(4), hide_index=True, width="stretch")
    nav = cum.set_index("signal_date")[[c for c in cum.columns if c.endswith("_nav")]]; nav.columns = [c.replace("_nav", "") for c in nav.columns]; lg = st.checkbox("对数坐标", value=True, key="logy")
    f = px.line(nav, title="累计净值(0.30%往返成本;含2025-2026,非盲测)"); f.update_yaxes(type="log" if lg else "linear"); st.plotly_chart(f, width="stretch")
    sm["year"] = sm.signal_date.str[:4]; yr = sm.groupby(["model", "year"]).agg(RankIC=("RankIC", "mean"), 期数=("RankIC", "size")).reset_index(); yr["model"] = pd.Categorical(yr.model, six); yr = yr.sort_values(["year", "model"])
    a, b = st.columns(2); a.plotly_chart(px.bar(yr, x="year", y="RankIC", color="model", barmode="group", title="分年 Rank IC(期均值)"), width="stretch")
    byy = by[by.strategy.isin(six + ["EqualWeight"])]; b.plotly_chart(px.bar(byy, x="year", y="compounded_net", color="strategy", barmode="group", title="分年复利净收益(0.30%)"), width="stretch")
    st.dataframe(byy.pivot_table(index="year", columns="strategy", values="compounded_net")[[c for c in six + ["EqualWeight"] if c in byy.strategy.unique()]].round(4), width="stretch")
    st.subheader("成本敏感性(往返成本 0 / 0.10% / 0.15% / 0.30% / 0.60%)"); c2 = cs[cs.strategy.isin(six + ["EqualWeight"])]
    a, b = st.columns(2); a.plotly_chart(px.line(c2, x="cost_roundtrip", y="annualized_return", color="strategy", markers=True, title="年化收益 vs 往返成本"), width="stretch"); b.plotly_chart(px.line(c2[c2.strategy != "EqualWeight"], x="cost_roundtrip", y="net_ex_t", color="strategy", markers=True, title="净超额 t 值 vs 往返成本"), width="stretch")
    st.dataframe(c2.pivot_table(index="strategy", columns="cost_roundtrip", values="annualized_return").round(4), width="stretch")
    st.subheader("配对检验(模型相减;净收益/RankIC 每期差)"); st.dataframe(pp.round(4), hide_index=True, width="stretch"); st.caption("所有配对差异均不显著(见 p1_dl_report.md)。")
    roll = pd.concat([sm[sm.model == m].assign(roll=sm[sm.model == m].RankIC.rolling(20).mean()) for m in six]); st.plotly_chart(px.line(roll, x="signal_date", y="roll", color="model", title="Rank IC 滚动20期均值"), width="stretch")
    st.subheader("分行业/板块 Rank IC(sector_summary_v2.csv;巨潮最新分类,非历史时点;THS 无成分数据,缺失)"); ss = load_csv("sector_summary_v2.csv"); gt = st.selectbox("分组类型", sorted(ss.group_type.unique()), key="gt")
    pv = ss[ss.group_type == gt].pivot_table(index="group", columns="model", values="rank_ic_mean")[six]; st.dataframe(pv.round(4), width="stretch"); st.caption("分组样本量、IC 期数见下表;ic_reliability='不可靠' 的组请勿解读。"); st.dataframe(ss[ss.group_type == gt].drop(columns=["classification_note"]).round(4), hide_index=True, width="stretch")
    st.subheader("深度学习模型说明"); st.markdown("MLP(11 维横截面分位特征,3 种子均值)、GRU(60 日序列,3 种子均值,GPU 训练)、小型 Transformer(**仅种子 42**)。训练窗口/purge/重训频率与 P0 相同;验证集=窗口内最后 60 日;未调参、未做消融。最新日分数的训练/验证:GRU/TF 最佳 epoch 偏早(见 dl_models_report.md)。")'''
rep(153, 167, P4)
# ---- page 3 history: insert multi-model pctile chart before radio (135) ----
assert "历史样本外预测模型" in line(135)
ins = '''            allm = hist[hist.eligible_asof].sort_values("signal_date"); figm = px.line(allm, x="signal_date", y="pctile", color="model", title="六模型历史排名分位(各模型 walk-forward 样本外;100%=最高)", category_orders={"model": list(MODELS)}); figm.update_layout(height=340, yaxis=dict(range=[0, 1])); st.plotly_chart(figm, width="stretch")
            st.dataframe(allm.groupby("model").agg(信号日数=("pctile", "size"), 分位均值=("pctile", "mean"), 分位p10=("pctile", lambda s: s.quantile(.1)), 分位p90=("pctile", lambda s: s.quantile(.9)), 最近一次分位=("pctile", "last")).reindex(list(MODELS)).round(3), width="stretch")'''
L[134:134] = ins.split("\n")
# ---- page 3 latest tables (117, 122) ----
assert line(117).lstrip().startswith("if r.eligible_asof")
L[116] = '''        if r.eligible_asof: st.dataframe(pd.DataFrame({"模型": list(MODELS), "score(原始,非概率)": [r[f"{k}_score"] for k in MODELS.values()], "排名(1=最高)": [int(r[f"rank_{k}"]) for k in MODELS.values()], "合格股票中分位": [r[f"pctile_{k}"] for k in MODELS.values()]}), hide_index=True); st.write(f"五模型(HGB/LGBM/MLP/GRU/TF)分歧 = 百分位标准差 {r.model_agreement_pctile_std:.3f}(越小越一致,不是置信度);进入 Top50 的模型:{r.models_in_top50 or '无'}")'''
assert line(122).lstrip().startswith("st.dataframe(pd.DataFrame({\"模型\"")
L[121] = '''        st.dataframe(pd.DataFrame({"模型": list(MODELS), "score(原始,非概率)": [cd["scores"].get(m) for m in MODELS], "插入样本池后名次": [cd["ranks"].get(m) for m in MODELS], "分位": [cd["pct"].get(m) for m in MODELS]}).fillna("未打分(池外,GRU/TF 需序列输入)"), hide_index=True); st.write("说明:", ";".join(cd["flags"]))'''
# ---- page 2 (92-95) ----
assert "sk = st.selectbox" in line(92)
L[91] = line(92).replace('["rank_hgb", "rank_lgbm", "rank_ridge", "r5"', '["rank_hgb", "rank_lgbm", "rank_ridge", "rank_dl_mlp", "rank_dl_gru", "rank_dl_tf", "model_agreement_pctile_std", "n_models_top50_of5", "r5"')
L[92] = line(93).replace('"rank_hgb", "rank_lgbm", "rank_ridge", "hgb_score", "lgbm_score", "ridge_score", "quality_flags"]', '"rank_hgb", "rank_lgbm", "rank_ridge", "rank_dl_mlp", "rank_dl_gru", "rank_dl_tf", "hgb_score", "lgbm_score", "ridge_score", "dl_mlp_score", "dl_gru_score", "dl_tf_score", "model_agreement_pctile_std", "n_models_top50_of5", "models_in_top50", "quality_flags"]')
assert "dl_tf_score" in line(93)
L[93] = ""; L[94] = ""
# ---- page 1 (76-82) ----
P1 = '''    st.subheader("模型一致度(合格股票排名,六模型)"); keys = list(MODELS.values()); rk = e[[f"rank_{k}" for k in keys]].corr(method="spearman").round(3); rk.index = list(MODELS); rk.columns = list(MODELS); a, b = st.columns(2)
    a.write("排名 Spearman 相关"); a.dataframe(rk)
    top = {m: set(e.nsmallest(50, f"rank_{k}").code) for m, k in MODELS.items()}; ov = pd.DataFrame([[len(top[x] & top[y]) for y in MODELS] for x in MODELS], index=list(MODELS), columns=list(MODELS)); b.write("Top50 重合只数"); b.dataframe(ov)
    st.subheader("五模型分歧与 ≥3 个模型同时进入 Top50 的股票"); a, b = st.columns(2)
    a.plotly_chart(px.histogram(e, x="model_agreement_pctile_std", nbins=40, title="五模型(HGB/LGBM/MLP/GRU/TF)百分位标准差分布(越小越一致)"), width="stretch"); cnt = e.n_models_top50_of5.value_counts().sort_index().rename_axis("进入Top50的模型数").reset_index(name="只数"); b.write("进入 Top50 的模型数分布"); b.dataframe(cnt, hide_index=True)
    mm = load_csv("reports/multi_model_top50_20261016.csv"); st.dataframe(mm[["code", "name", "industry", "n_models_top50_of5", "models_in_top50", "rank_hgb", "rank_lgbm", "rank_mlp", "rank_gru", "rank_tf", "model_agreement_pctile_std", "limit_up_locked_latest"]], hide_index=True, width="stretch"); st.caption("仅为模型一致性描述,不是买入建议;模型输入高度相关,一致不等于独立验证。")
    st.subheader("最近20期实测(5日调仓,往返0.30%,Top50;OOS重放,非盲测)"); r20 = load_csv("reports/recent20_metrics_v2.csv"); sm = load_csv("reports/recent20_summary_v2.csv"); st.dataframe(sm.round(4), width="stretch")
    f1 = px.line(r20, x="signal_date", y="RankIC", color="model", markers=True, title="最近20期 Rank IC"); f2 = px.line(r20, x="signal_date", y="net_ret_c30", color="model", markers=True, title="最近20期 净收益(单期)")
    a, b = st.columns(2); a.plotly_chart(f1, width="stretch"); b.plotly_chart(f2, width="stretch"); st.caption("样本仅20期(约100个交易日),仅描述,不能据此判断模型优劣。")'''
assert "模型一致度" in line(76); rep(76, 82, P1)
# ---- header lines ----
L[38] = line(39).replace("stock_cards_data_20261016.json", "stock_cards_data_20261016_v2.json"); assert "_v2.json" in L[38]
L[24] = line(25).replace("historical_predictions.parquet", "historical_predictions_all_models.parquet"); assert "all_models" in L[24]
L[16] = line(17).replace("latest_all_stocks_20261016.csv", "latest_all_stocks_20261016_v2.csv"); assert "_v2.csv" in L[16]
L[17] = '''    for c in ["close_unadjusted", "r5", "r20", "r60", "vol20", "amount20", "turnover20", "model_agreement_pctile_std", "model_pctile_mean_5", "n_models_top50_of5"] + [f"{p}{k}" for k in ["hgb", "lgbm", "ridge", "dl_mlp", "dl_gru", "dl_tf"] for p in ["rank_", "pctile_"]] + [f"{k}_score" for k in ["hgb", "lgbm", "ridge", "dl_mlp", "dl_gru", "dl_tf"]]:'''
L[11] = '''ASOF = "2026-10-16"; MODELS = {"HGB": "hgb", "LGBM": "lgbm", "Ridge": "ridge", "MLP": "dl_mlp", "GRU": "dl_gru", "TF": "dl_tf"}; OUTSIDER = "600060"'''
L[0] = '"""P1 read-only research dashboard v2 (Streamlit + Plotly): HGB/LGBM/Ridge + MLP/GRU/Transformer. Data as of 2026-10-16. Research rankings only - NOT investment advice; model score is NOT a probability or a return.'
L[1] = 'Run:  .\\run_dashboard.ps1   (http://127.0.0.1:8501)   [v1 = stock_dashboard.py, kept unchanged]"""'
(H / "stock_dashboard_v2.py").write_text("\n".join(l for i, l in enumerate(L) if True), encoding="utf-8")
# run script + smoke
rd = (H / "run_dashboard.ps1").read_text(encoding="utf-8"); (H / "run_dashboard_v1.ps1").write_text(rd, encoding="utf-8") if not (H / "run_dashboard_v1.ps1").exists() else None
assert "stock_dashboard.py" in rd; (H / "run_dashboard.ps1").write_text(rd.replace("stock_dashboard.py", "stock_dashboard_v2.py").replace("Starts the read-only dashboard", "Starts the read-only dashboard v2 (6 models; v1 = run_dashboard_v1.ps1)"), encoding="utf-8")
sd = (H / "smoke_dashboard.py").read_text(encoding="utf-8"); sd2 = sd.replace("stock_dashboard.py", "stock_dashboard_v2.py").replace("latest_all_stocks_20261016.csv", "latest_all_stocks_20261016_v2.csv").replace("Headless smoke test of", "Headless smoke test (v2) of")
(H / "smoke_dashboard_v2.py").write_text(sd2, encoding="utf-8"); print("written", len(L))
