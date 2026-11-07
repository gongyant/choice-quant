# -*- coding: utf-8 -*-
"""P2 read-only research dashboard (Streamlit + Plotly): P2-A signal anatomy, P2-B cost-aware portfolios, P2-C Transformer ablations; entry to the P1 per-stock dashboard.
Run:  .\run_dashboard_p2.ps1   (http://127.0.0.1:8502).  Reads results\ / figures\ / reports\ only; writes nothing. NOT investment advice; scores are NOT probabilities; 2022-07..2026 results are exploratory."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
H = Path(__file__).resolve().parent; V2 = H.parent; R = H / "results"; FIG = H / "figures"; REP = H / "reports"; P1 = H.parent / "p1_dl"
st.set_page_config(page_title="P2 研究看板(只读)", layout="wide")
PAGES = ["概览", "P2-A 信号解剖", "P2-B 成本感知组合", "P2-C Transformer 消融", "P1 逐股页面入口", "报告与图", "研究选股器"]
ORDER = [PAGES[0], PAGES[6]] + PAGES[1:6]     # 研究选股器 shown second
MODELS = ["HGB", "LGBM", "Ridge", "MLP", "GRU", "TF"]
@st.cache_data(show_spinner=False)
def csv(name): return pd.read_csv(R / name, encoding="utf-8-sig")
@st.cache_data(show_spinner=False)
def bruns(model, scen): return pd.read_parquet(R / "b_runs" / f"{model}_{scen}.parquet", columns=["k", "variant", "ret", "gross_ret", "turnover_value", "signal_idx"])
@st.cache_data(show_spinner=False)
def cby(): return pd.read_parquet(R / "c_by_date.parquet")
@st.cache_data(show_spinner=False)
def scr(): return pd.read_csv(R / "research_screener_20261016.csv", dtype={"code": str}, encoding="utf-8-sig")
@st.cache_data(show_spinner=False)
def oos_hist(code):
    f = P1 / "historical_predictions_all_models.parquet"
    if not f.exists(): return None
    h = pd.read_parquet(f, columns=["signal_date", "code", "model", "rank", "pctile", "in_top50", "realized_ret_5d", "realized_rel_ret_5d"], filters=[("code", "==", code)])
    h["signal_date"] = pd.to_datetime(h.signal_date); return h.sort_values("signal_date")
@st.cache_data(show_spinner=False)
def prices(code):
    sys.path.insert(0, str(V2 / "p0")); import p0lib as L
    ru, fa = (P1 / "data_extra" / "raw", P1 / "data_extra" / "adj") if code == "600060" else (V2 / "p0" / "data" / "raw_unadj_parquet", V2 / "p0" / "data" / "adj_factors")
    if not (ru / f"{code}.parquet").exists(): return None
    d = pd.read_parquet(ru / f"{code}.parquet").sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True); d["date"] = pd.to_datetime(d.date)
    fac = pd.read_parquet(fa / f"{code}.parquet") if (fa / f"{code}.parquet").exists() else None; d["F"] = L.factor_on_dates(d.date, fac)
    for k in ["open", "high", "low", "close"]: d["a" + k] = d[k] / d.F
    return d
def banner(): st.caption("只读研究看板 · 不下单 · 不是投资建议 · 模型分数不是概率 · 2022-07 以后的样本外数据已被反复使用,所有结论均为探索性")
def ci_bar(d, x, y, lo, hi, color=None, title=""):
    fig = px.bar(d, x=x, y=y, color=color, barmode="group", title=title, error_y=d[hi] - d[y], error_y_minus=d[y] - d[lo]); fig.update_layout(height=420, margin=dict(t=40, b=10)); return fig
if "page" not in st.session_state: st.session_state["page"] = PAGES[0]
page = st.sidebar.radio("页面", ORDER, key="page"); st.sidebar.caption("数据:P0/P1 冻结于 2026-10-16;P2 结果文件见 results\\")
# ------------------------------------------------------------------ overview
if page == PAGES[0]:
    st.header("P2 研究看板(只读)"); banner()
    st.markdown("""
- **P2-A 信号解剖**:新 RankIC 口径(执行日可得前向收益)下 6 个模型的分组 IC、偏 IC、十分位与因子归因。
- **P2-B 成本感知组合**:A0 Top50 / A1 滞后带 / A2 成本感知优化在 C1 flat、C2 显式、C3 现实约束、C4 容量下的净值、成本分解、换手 vs 预测分解。
- **P2-C Transformer 消融**:种子方差、多周期头 / 排序损失 / 持续性正则 / purge 几何的单因素对比、匹配信息集的滞后堆叠 LightGBM。
- **P1 逐股页面**:沿用 P1 v2 看板(端口 8501),本看板只提供入口与只读的最新排名查询。
""")
    c = st.columns(3)
    try:
        sv = csv("c_seed_variance.csv"); t0 = sv[(sv.variant == "T0") & (sv.signals.astype(str) == "207")].iloc[0]
        c[0].metric("TF 三种子 RankIC(207 信号)", f"{t0.rankic_mean:.4f}", f"std {t0.rankic_std:.4f}")
    except Exception as e: c[0].write(f"P2-C 结果缺失:{e}")
    pf = csv("b_performance.csv"); x = pf[(pf.scenario == "C2_explicit") & (pf.window == "eval_k52+")]
    for i, v in enumerate(["A0", "A2_nested"]): c[i + 1].metric(f"C2 6模型均值 {v} 净年化", f"{x[x.variant == v].ann_net.mean()*100:.1f}%", f"毛 {x[x.variant == v].ann_gross.mean()*100:.1f}%", delta_color="off")
# ------------------------------------------------------------------ P2-A
elif page == PAGES[1]:
    st.header("P2-A 信号解剖"); banner()
    tab = st.tabs(["分组 IC", "偏 IC", "十分位", "因子归因", "新旧口径"])
    with tab[0]:
        g = csv("rankic_by_group.csv"); gv = st.selectbox("分组变量", list(g.group_var.unique()), key="a_gv"); d = g[g.group_var == gv]
        st.plotly_chart(ci_bar(d, "group", "mean", "ci_lo", "ci_hi", "model", f"RankIC by {gv}(95% block bootstrap)"), width="stretch"); st.dataframe(d.round(4), width="stretch", hide_index=True)
    with tab[1]:
        p = csv("partial_rankic.csv"); st.plotly_chart(ci_bar(p, "control_set", "mean", "ci_lo", "ci_hi", "model", "偏 RankIC(控制 as-of-t 特征后)"), width="stretch"); st.dataframe(p.round(4), width="stretch", hide_index=True)
    with tab[2]:
        dm = csv("decile_means.csv"); de = st.selectbox("口径", list(dm.definition.unique()), index=list(dm.definition.unique()).index("new") if "new" in set(dm.definition) else 0, key="a_def")
        ms = st.multiselect("模型", MODELS, default=["LGBM", "TF"], key="a_dm"); d = dm[(dm.definition == de) & dm.model.isin(ms)]
        st.plotly_chart(px.bar(d, x="decile", y="mean_rel_ret", color="model", barmode="group", title="十分位平均相对收益(5 日)"), width="stretch")
        st.dataframe(csv("decile_summary.csv").query("definition == @de").round(4), width="stretch", hide_index=True)
    with tab[3]:
        fa = csv("factor_attribution.csv"); st.dataframe(fa.round(4), width="stretch", hide_index=True)
        dep = st.selectbox("被解释变量", list(fa.dependent.unique()), key="a_dep"); sp = st.selectbox("模型设定", list(fa.spec.unique()), key="a_spec"); d = fa[(fa.dependent == dep) & (fa.spec == sp)]
        st.plotly_chart(ci_bar(d, "model", "alpha", "alpha_ci_lo", "alpha_ci_hi", None, f"因子调整 alpha:{dep} / {sp}"), width="stretch")
    with tab[4]: st.dataframe(csv("rankic_old_vs_new.csv").round(4), width="stretch", hide_index=True)
# ------------------------------------------------------------------ P2-B
elif page == PAGES[2]:
    st.header("P2-B 成本感知组合"); banner()
    tab = st.tabs(["净值", "成本分解", "换手 vs 预测", "因子 alpha", "配对检验"])
    with tab[0]:
        c1, c2 = st.columns(2); m = c1.selectbox("模型", MODELS, index=5, key="b_m"); s = c2.selectbox("情景", ["C1_flat", "C2_explicit", "C3_realistic", "C4_capacity_1e8"], index=1, key="b_s")
        d = bruns(m, s); ew = csv("b_ew_benchmark.csv"); fig = go.Figure()
        for v in ["A0", "A1_nested", "A2_nested"]:
            x = d[d.variant == v].sort_values("k"); dt = pd.to_datetime(ew.signal_date.values[x.k.values])
            fig.add_scatter(x=dt, y=(1 + x.ret).cumprod(), name=f"{v} 净"); fig.add_scatter(x=dt, y=(1 + x.gross_ret).cumprod(), name=f"{v} 毛", line=dict(dash="dot"))
        fig.add_scatter(x=pd.to_datetime(ew.signal_date), y=(1 + ew.ew_ret_c0).cumprod(), name="等权候选池(c0)", line=dict(color="black"))
        fig.add_vline(x=pd.to_datetime(ew.signal_date.iloc[52]), line_dash="dash"); fig.update_layout(height=480, title=f"{m} / {s}(虚线右侧为嵌套选择评估期 k≥52)"); st.plotly_chart(fig, width="stretch")
    with tab[1]:
        pf = csv("b_performance.csv"); x = pf[(pf.scenario == "C2_explicit") & (pf.window == "eval_k52+") & pf.variant.isin(["A0", "A1_nested", "A2_nested"])]
        cc = [c for c in x.columns if c.startswith("drag_") and c != "drag_flat_annual"]; y = x.melt(id_vars=["model", "variant"], value_vars=cc, var_name="成本项", value_name="年化拖累")
        st.plotly_chart(px.bar(y, x="model", y="年化拖累", color="成本项", facet_col="variant", title="C2 显式成本分解(评估期,年化)"), width="stretch")
        st.dataframe(x[["model", "variant", "ann_gross", "ann_net", "turnover_oneway_per_period", "cost_drag_annual"] + cc].round(4), width="stretch", hide_index=True)
    with tab[2]:
        tv = csv("b_turnover_vs_prediction.csv"); s = st.selectbox("情景", list(tv.scenario.unique()), index=1, key="b_tv_s"); d = tv[tv.scenario == s]
        y = d.melt(id_vars=["model", "family"], value_vars=["turnover_effect_S_adj", "selection_effect_S_adj"], var_name="分解", value_name="净收益/期")
        st.plotly_chart(px.bar(y, x="model", y="净收益/期", color="分解", facet_col="family", barmode="relative", title="净收益改善 = 换手效应 + 选择效应(相对 A0,S 控制,换手精确匹配)"), width="stretch")
        st.dataframe(d.round(5), width="stretch", hide_index=True)
    with tab[3]: st.dataframe(csv("b_factor_alpha.csv").round(4), width="stretch", hide_index=True)
    with tab[4]: st.dataframe(csv("b_paired.csv").round(5), width="stretch", hide_index=True)
# ------------------------------------------------------------------ P2-C
elif page == PAGES[3]:
    st.header("P2-C Transformer 受控消融"); banner()
    st.caption("T0=P1 配方(seed42 = P1 冻结分数);T0G=20 日 purge 几何;T1=排序损失;T2=多周期头 MSE;T3=多周期排序损失;T4=持续性正则;LAGLGBM=同 60×7 信息集的滞后堆叠 LightGBM。消融只在前 104 信号(前 26 个 retrain 块)上训练。")
    tab = st.tabs(["种子方差", "消融对比", "配对检验", "lag-LGBM vs TF", "事后诊断"])
    rn = csv("c_runs.csv"); sv = csv("c_seed_variance.csv"); pr = csv("c_paired.csv")
    with tab[0]:
        x = rn[(rn.window == "all") & rn.variant.isin(["T0", "LAGLGBM"])]; met = st.selectbox("指标", ["rankic", "stability", "top50_overlap", "turnover_c2", "ann_net_c1", "ann_net_c2", "alpha_c2_hac_t"], key="c_met")
        st.plotly_chart(px.strip(x, x="variant", y=met, color="seed", title=f"{met}:各种子(207 信号)"), width="stretch"); st.dataframe(sv.round(4), width="stretch", hide_index=True)
    with tab[1]:
        x = sv[sv.signals.astype(str) == "104"]; met = st.selectbox("指标", ["rankic", "stability", "top50_overlap", "turnover_c2", "ann_net_c1", "ann_net_c2", "alpha_c2_hac_t"], key="c_met2")
        fig = go.Figure(go.Bar(x=x.variant, y=x[f"{met}_mean"], error_y=dict(type="data", symmetric=False, array=x[f"{met}_max"] - x[f"{met}_mean"], arrayminus=x[f"{met}_mean"] - x[f"{met}_min"]), text=x.seeds)); fig.update_layout(title=f"{met}:种子均值(误差条=种子极差,前 104 信号)", height=430)
        st.plotly_chart(fig, width="stretch")
    with tab[2]:
        sc = st.selectbox("范围", sorted(pr.scope.unique()), index=sorted(pr.scope.unique()).index("seed-mean"), key="c_scope"); d = pr[pr.scope == sc]
        st.dataframe(d[["A", "B", "desc", "kind", "signals", "metric", "n", "diff_mean", "hac_t", "ci_lo", "ci_hi", "share_A_gt_B"]].round(5), width="stretch", hide_index=True)
    with tab[3]:
        b = cby(); fig = go.Figure()
        for v in ["T0", "LAGLGBM"]:
            g = b[b.variant == v].groupby("k")[["ic", "net_c2"]].mean(); fig.add_scatter(x=g.index, y=g.ic.cumsum(), name=f"{v} 累计 RankIC")
        fig.update_layout(title="累计 RankIC(种子均值)", height=420); st.plotly_chart(fig, width="stretch")
        st.dataframe(pr[(pr.A == "LAGLGBM") & (pr.B == "T0")][["signals", "scope", "metric", "diff_mean", "hac_t", "ci_lo", "ci_hi"]].round(5), width="stretch", hide_index=True)
    with tab[4]:
        st.dataframe(csv("c_posthoc.csv").round(4), width="stretch", hide_index=True); st.markdown("**ORACLE 合成信号(仅诊断,含未来信息,不是策略)**"); st.dataframe(csv("c_oracle_diag.csv").round(4), width="stretch", hide_index=True)
# ------------------------------------------------------------------ P1 entry
elif page == PAGES[4]:
    st.header("P1 逐股页面入口"); banner()
    st.markdown("P1 的逐股页面(K 线、历史预测、模型对比)保持在 P1 v2 看板中,未改动。启动:`D:\\choice_quant_starter\\v2\\p1_dl\\run_dashboard.ps1`,然后打开 [http://127.0.0.1:8501](http://127.0.0.1:8501)。")
    f = P1 / "latest_all_stocks_20261016_v2.csv"
    if f.exists():
        lat = pd.read_csv(f, dtype=str, encoding="utf-8-sig", keep_default_na=False); q = st.text_input("快速查询(代码或名称,只读,数据截至 2026-10-16)", key="q")
        cols = [c for c in ["code", "name", "eligible_asof", "rank_hgb", "rank_lgbm", "rank_ridge", "rank_dl_mlp", "rank_dl_gru", "rank_dl_tf", "n_models_top50_of5"] if c in lat.columns]
        d = lat[lat.code.str.contains(q) | lat.name.str.contains(q)] if q else lat.head(20); st.dataframe(d[cols].head(50), width="stretch", hide_index=True)
    else: st.warning("未找到 P1 最新排名文件")
# ------------------------------------------------------------------ reports
elif page == PAGES[5]:
    st.header("报告与图"); banner()
    reps = sorted(REP.glob("p2*_report.md")); r = st.selectbox("报告", [p.name for p in reps], key="rep")
    with st.expander("展开报告全文", expanded=False): st.markdown((REP / r).read_text(encoding="utf-8"))
    figs = sorted(FIG.glob("*.png")); f = st.selectbox("图", [p.name for p in figs], key="fig"); st.image(str(FIG / f), width="stretch")# ------------------------------------------------------------------ research screener (read-only)
elif page == PAGES[6]:
    st.header("研究选股器(截至 2026-10-16 收盘的冻结分数)")
    st.error("**研究排名,不是买入推荐。** 分数不是概率,也不是预期收益;研究权重只是 P2-B A2 规则的研究模拟(空仓起步、C2 显式成本、嵌套选出的最近一块参数)。"
             "**P2 结论:信号未被证明可在成本后变现**(A0/A1/A2 在 C2 下净收益与 alpha 均不显著,见 P2-B/P2-C 报告)。不构成投资建议;本页只读,不下单。")
    f = R / "research_screener_20261016.csv"
    if not f.exists(): st.warning("未找到 results\\research_screener_20261016.csv,请先运行 c_screener.py"); st.stop()
    S0 = scr(); meta = json.load(open(R / "research_screener_20261016_meta.json", encoding="utf-8"))
    st.caption(f"六模型分数 = P1 2026-10-16 冻结分数(TF 为 seed42 单种子);TF 三种子平均 = P2-C T0 seeds {meta['tf3_seeds']} 在最近一个 OOS 信号日 {meta['tf3_signal_date']} 的分位平均(P2-C 未保存模型,不重训,故不是 10-16 分数)。"
               "三因子基线 = 5日反转(-r5)、低波动(-vol20)、Amihud 非流动性 的池内分位及其平均。行业为巨潮最新分类,**非历史时点**。")
    c1, c2, c3, c4 = st.columns(4)
    only_e = c1.checkbox("只看 2026-10-16 可选池(eligible)", True, key="s_elig")
    inds = sorted(S0.industry_l1.dropna().astype(str).unique()); ind = c1.multiselect("行业(最新分类,非时点)", inds, key="s_ind")
    exs = sorted(S0.exchange.dropna().astype(str).unique()); ex = c2.multiselect("交易所", exs, key="s_ex")
    liq = c2.number_input("20日成交额中位数下限(亿元)", 0.0, 100.0, 0.0, 0.1, key="s_liq")
    nmin = c3.slider("至少进入 Top50 的模型数(6个)", 0, 6, 0, key="s_ntop"); smax = c3.slider("六模型分位标准差上限(越小越一致)", 0.0, 0.5, 0.5, 0.01, key="s_std")
    x_lu = c4.checkbox("排除 10-16 涨停封板", False, key="s_xlu"); x_st = c4.checkbox("排除现名含 ST", False, key="s_xst"); x_ac = c4.checkbox("排除科创/北交(需准入)", False, key="s_xac")
    d = S0.copy()
    if only_e: d = d[d.eligible_asof]
    if ind: d = d[d.industry_l1.astype(str).isin(ind)]
    if ex: d = d[d.exchange.astype(str).isin(ex)]
    if liq > 0: d = d[d.amt20.fillna(0) >= liq * 1e8]
    if nmin > 0: d = d[d.n_models_top50_of6.fillna(0) >= nmin]
    if smax < 0.5: d = d[d.pctile_std_6.fillna(1) <= smax]
    if x_lu: d = d[~d.limit_up_locked_t]
    if x_st: d = d[~d.st_name_current]
    if x_ac: d = d[~d.needs_special_access]
    SORT = {"六模型平均分位": "pctile_mean_6", "进入Top50模型数": "n_models_top50_of6", "TF三种子平均分位(OOS末日)": "tf3_pctile_mean", "三因子基线平均分位": "base3_pctile_mean", "A2研究权重(六模型平均)": "a2w_mean6",
            **{f"{m} 分位": f"pctile_{m}" for m in MODELS}, "20日成交额": "amt20", "20日收益": "r20", "20日波动率": "vol20"}
    s1, s2, s3 = st.columns(3); sk = s1.selectbox("排序", list(SORT), key="s_sort"); asc = s2.checkbox("升序", False, key="s_asc"); topn = s3.number_input("显示行数", 10, 1500, 100, 10, key="s_n")
    d = d.sort_values([SORT[sk], "code"], ascending=[asc, True], na_position="last")
    cols = {"code": "代码", "name": "名称", "exchange": "交易所", "board": "板块", "industry_l1": "行业(最新)", "n_models_top50_of6": "Top50模型数", "models_in_top50": "进入Top50的模型", "pctile_mean_6": "六模型平均分位", "pctile_std_6": "分位标准差",
            **{f"rank_{m}": f"{m}排名" for m in MODELS}, "tf3_pctile_mean": "TF三种子分位(OOS末日)", "base3_pctile_mean": "三因子基线分位", "r20": "20日收益", "vol20": "20日波动", "turn20": "20日换手", "amihud20": "Amihud", "amt20": "20日成交额",
            "a2w_mean6": "A2研究权重(6模型均)", "a2w_TF": "A2研究权重(TF)", "limit_up_locked_t": "涨停封板", "st_name_current": "现名ST", "needs_special_access": "需准入(科创/北交)", "low_liquidity_flag": "低流动性(<0.5亿)"}
    st.markdown(f"筛选后 **{len(d)}** 只(全表 {len(S0)} 只;可选池 {int(S0.eligible_asof.sum())} 只)")
    st.dataframe(d[list(cols)].head(int(topn)).rename(columns=cols).round(4), width="stretch", hide_index=True)
    st.caption("排名 1 = 该模型分数最高;分位 1 = 池内最高。A2 权重为研究模拟,未计实际下单约束之外的任何因素,不是持仓建议。")
    st.subheader("个股详情(只用已有数据)")
    opts = [f"{c} {n}" for c, n in zip(d.code, d.name)][:int(topn)]
    if not opts: st.info("没有满足筛选条件的股票"); st.stop()
    sel = st.selectbox("选择个股", opts, key="s_code"); code = sel.split(" ")[0]; row = S0[S0.code == code].iloc[0]
    m1, m2, m3, m4 = st.columns(4); m1.metric("六模型平均分位", f"{row.pctile_mean_6:.3f}" if pd.notna(row.pctile_mean_6) else "—"); m2.metric("Top50 模型数", f"{row.n_models_top50_of6:.0f}" if pd.notna(row.n_models_top50_of6) else "—")
    m3.metric("20日成交额(亿)", f"{row.amt20/1e8:.2f}" if pd.notna(row.amt20) else "—"); m4.metric("A2 研究权重(6模型均)", f"{row.a2w_mean6:.4f}" if pd.notna(row.a2w_mean6) else "—")
    st.dataframe(pd.DataFrame({m: [row[f"rank_{m}"], row[f"pctile_{m}"], row.get(f"a2w_{m}")] for m in MODELS}, index=["排名", "分位", "A2研究权重"]).round(4), width="stretch")
    h = oos_hist(code)
    if h is None or h.empty: st.info("该股无历史 OOS 预测记录")
    else:
        fig = px.line(h, x="signal_date", y="pctile", color="model", title="历史 OOS 分位(P1 walk-forward,1 = 最高)"); fig.update_layout(height=360, margin=dict(t=40, b=10)); st.plotly_chart(fig, width="stretch")
        r = h.dropna(subset=["realized_rel_ret_5d"]); hm = st.selectbox("模型(实际后续收益)", MODELS, index=5, key="s_hm"); r = r[r.model == hm]
        if len(r):
            fig = px.scatter(r, x="pctile", y="realized_rel_ret_5d", hover_data=["signal_date"], title=f"{hm}: OOS 分位 vs 实际后续 5 日相对收益(事后已实现)"); fig.update_layout(height=340, margin=dict(t=40, b=10)); st.plotly_chart(fig, width="stretch")
            st.caption(f"{hm} 该股 OOS 信号 {len(r)} 次;分位与实际相对收益 Spearman = {r.pctile.corr(r.realized_rel_ret_5d, method='spearman'):.3f}(单股样本小,仅作描述)")
            st.dataframe(r.sort_values("signal_date", ascending=False)[["signal_date", "rank", "pctile", "in_top50", "realized_ret_5d", "realized_rel_ret_5d"]].head(30).round(4), width="stretch", hide_index=True)
    px_ = prices(code)
    if px_ is None: st.info("无价格数据")
    else:
        q = px_.tail(500); fig = go.Figure([go.Candlestick(x=q.date, open=q.aopen, high=q.ahigh, low=q.alow, close=q.aclose, name="前复权(按P0复权因子)")])
        fig.add_scatter(x=q.date, y=q.aclose.rolling(20).mean(), name="MA20"); fig.add_scatter(x=q.date, y=q.aclose.rolling(60).mean(), name="MA60")
        fig.update_layout(height=420, xaxis_rangeslider_visible=False, margin=dict(t=30, b=10), title="价格(截至 2026-10-16,已有数据)"); st.plotly_chart(fig, width="stretch")
