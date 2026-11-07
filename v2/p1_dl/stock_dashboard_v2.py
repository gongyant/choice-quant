"""P1 read-only research dashboard v2 (Streamlit + Plotly): HGB/LGBM/Ridge + MLP/GRU/Transformer. Data as of 2026-10-16. Research rankings only - NOT investment advice; model score is NOT a probability or a return.
Run:  .\run_dashboard.ps1   (http://127.0.0.1:8501)   [v1 = stock_dashboard.py, kept unchanged]"""
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
H = Path(__file__).resolve().parent; V2 = H.parent
sys.path.insert(0, str(V2 / "p0")); import p0lib as L
ASOF = "2026-10-16"; MODELS = {"HGB": "hgb", "LGBM": "lgbm", "Ridge": "ridge", "MLP": "dl_mlp", "GRU": "dl_gru", "TF": "dl_tf"}; OUTSIDER = "600060"
st.set_page_config(page_title="Choice A股研究排名(只读)", layout="wide")
PAGES = ["① 市场全览", "② 股票查询", "③ 单股页", "④ 模型对比", "⑤ 风险与数据质量"]
@st.cache_data(show_spinner=False)
def load_latest():
    d = pd.read_csv(H / "latest_all_stocks_20261016_v2.csv", dtype=str, encoding="utf-8-sig", keep_default_na=False)
    for c in ["close_unadjusted", "r5", "r20", "r60", "vol20", "amount20", "turnover20", "model_agreement_pctile_std", "model_pctile_mean_5", "n_models_top50_of5"] + [f"{p}{k}" for k in ["hgb", "lgbm", "ridge", "dl_mlp", "dl_gru", "dl_tf"] for p in ["rank_", "pctile_"]] + [f"{k}_score" for k in ["hgb", "lgbm", "ridge", "dl_mlp", "dl_gru", "dl_tf"]]:
        if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
    d["eligible_asof"] = d.eligible_asof == "True"; d["reason_short"] = d.ineligible_reason.str.replace(r"\(.*", "", regex=True); return d
@st.cache_data(show_spinner=False)
def load_csv(rel, **kw): return pd.read_csv(H / rel, encoding="utf-8-sig", dtype={"code": str}, **kw)
@st.cache_data(show_spinner=False)
def load_hist_code(code):
    p = H / "historical_predictions_all_models.parquet"
    cols = ["signal_date", "code", "model", "model_id", "train_end_date", "eligible_asof", "score", "rank", "pctile", "in_top50", "buy_status", "realized_ret_5d", "realized_rel_ret_5d", "n_pool"]
    return pd.read_parquet(p, columns=cols, filters=[("code", "==", code)])
@st.cache_data(show_spinner=False)
def load_prices(code):
    ru, fa = (H / "data_extra" / "raw", H / "data_extra" / "adj") if code == OUTSIDER else (V2 / "p0" / "data" / "raw_unadj_parquet", V2 / "p0" / "data" / "adj_factors")
    f = ru / f"{code}.parquet"
    if not f.exists(): return None
    d = pd.read_parquet(f).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True); d["date"] = pd.to_datetime(d.date)
    fac = pd.read_parquet(fa / f"{code}.parquet") if (fa / f"{code}.parquet").exists() else None
    d["F"] = L.factor_on_dates(d.date, fac)
    for k in ["open", "high", "low", "close"]: d["a" + k] = d[k] / d.F
    d["ma20"] = d.aclose.rolling(20).mean(); d["ma60"] = d.aclose.rolling(60).mean(); return d
@st.cache_data(show_spinner=False)
def load_cards(): return json.load(open(H / "scores" / "stock_cards_data_20261016_v2.json", encoding="utf-8"))
@st.cache_data(show_spinner=False)
def load_events(): return pd.read_csv(V2 / "p0" / "quality_event_log.csv", dtype={"code": str}, usecols=["code", "name", "event_date", "cause", "action"])
@st.cache_data(show_spinner=False)
def load_dl_scores():
    out = {}
    for f in sorted(glob.glob(str(H / "scores" / "dl_*scores_20261016.csv"))):
        try:
            d = pd.read_csv(f, dtype={"code": str}); out[Path(f).name] = d
        except Exception: pass
    return out
@st.cache_data(show_spinner=False)
def p0_meta(): return json.load(open(H / "models" / "model_meta_2026-10-16.json", encoding="utf-8"))
def banner():
    st.title("Choice A股研究排名(本地只读)")
    st.warning(f"数据截至 **{ASOF}** 收盘(最新信号日)。**研究排名,非买入/卖出建议**;模型 score 为原始输出,**不是概率,也不是收益率**。样本为1500只分层抽样(含已退市),不是全市场;2025-2026数据已多次用于模型比较,不是盲测。")
def go_stock(code):
    lab = "600060 海信视像(池外,不在1500样本)" if code == OUTSIDER else f"{code} {LAT.set_index('code').loc[code, 'name']}"
    st.session_state["code"] = code; st.session_state["page"] = PAGES[2]; st.session_state["stk_filter"] = ""; st.session_state["stk_select"] = lab
LAT = load_latest()
st.sidebar.header("导航")
if "page" not in st.session_state: st.session_state["page"] = PAGES[0]
if "code" not in st.session_state: st.session_state["code"] = "600060"
page = st.sidebar.radio("页面", PAGES, key="page")
st.sidebar.markdown("**预设个股**")
for c, n in [("600060", "海信视像(池外)"), ("300487", "蓝晓科技"), ("000027", "深圳能源")]: st.sidebar.button(f"{c} {n}", key=f"quick_{c}", on_click=go_stock, args=(c,))
st.sidebar.caption("只读;不下单;不含未来价格预测。")
banner()
# ---------------------------------------------------------------- page 1
if page == PAGES[0]:
    e = LAT[LAT.eligible_asof]; c1, c2, c3, c4 = st.columns(4)
    c1.metric("主表行数(1500样本)", len(LAT)); c2.metric("信号日合格(eligible_asof)", len(e)); c3.metric("不合格", len(LAT) - len(e)); c4.metric("无行情数据", int((LAT.reason_short == "无行情数据").sum()))
    a, b = st.columns(2)
    with a:
        rs = LAT[~LAT.eligible_asof].reason_short.value_counts().rename_axis("原因").reset_index(name="只数"); st.subheader("不合格原因分布"); st.plotly_chart(px.bar(rs, x="只数", y="原因", orientation="h", text="只数"), width="stretch")
    with b:
        st.subheader("合格股票近20日收益分布"); st.plotly_chart(px.histogram(e, x="r20", nbins=60, labels={"r20": "近20日复权收益"}), width="stretch")
    st.subheader("模型一致度(合格股票排名,六模型)"); keys = list(MODELS.values()); rk = e[[f"rank_{k}" for k in keys]].corr(method="spearman").round(3); rk.index = list(MODELS); rk.columns = list(MODELS); a, b = st.columns(2)
    a.write("排名 Spearman 相关"); a.dataframe(rk)
    top = {m: set(e.nsmallest(50, f"rank_{k}").code) for m, k in MODELS.items()}; ov = pd.DataFrame([[len(top[x] & top[y]) for y in MODELS] for x in MODELS], index=list(MODELS), columns=list(MODELS)); b.write("Top50 重合只数"); b.dataframe(ov)
    st.subheader("五模型分歧与 ≥3 个模型同时进入 Top50 的股票"); a, b = st.columns(2)
    a.plotly_chart(px.histogram(e, x="model_agreement_pctile_std", nbins=40, title="五模型(HGB/LGBM/MLP/GRU/TF)百分位标准差分布(越小越一致)"), width="stretch"); cnt = e.n_models_top50_of5.value_counts().sort_index().rename_axis("进入Top50的模型数").reset_index(name="只数"); b.write("进入 Top50 的模型数分布"); b.dataframe(cnt, hide_index=True)
    mm = load_csv("reports/multi_model_top50_20261016.csv"); st.dataframe(mm[["code", "name", "industry", "n_models_top50_of5", "models_in_top50", "rank_hgb", "rank_lgbm", "rank_mlp", "rank_gru", "rank_tf", "model_agreement_pctile_std", "limit_up_locked_latest"]], hide_index=True, width="stretch"); st.caption("仅为模型一致性描述,不是买入建议;模型输入高度相关,一致不等于独立验证。")
    st.subheader("最近20期实测(5日调仓,往返0.30%,Top50;OOS重放,非盲测)"); r20 = load_csv("reports/recent20_metrics_v2.csv"); sm = load_csv("reports/recent20_summary_v2.csv"); st.dataframe(sm.round(4), width="stretch")
    f1 = px.line(r20, x="signal_date", y="RankIC", color="model", markers=True, title="最近20期 Rank IC"); f2 = px.line(r20, x="signal_date", y="net_ret_c30", color="model", markers=True, title="最近20期 净收益(单期)")
    a, b = st.columns(2); a.plotly_chart(f1, width="stretch"); b.plotly_chart(f2, width="stretch"); st.caption("样本仅20期(约100个交易日),仅描述,不能据此判断模型优劣。")
# ---------------------------------------------------------------- page 2
elif page == PAGES[1]:
    st.subheader("1500只样本股票(含已退市/无数据)可搜索排序表")
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1]); q = c1.text_input("搜索代码或名称", key="q"); inds = c2.multiselect("行业(巨潮最新分类)", sorted(LAT.industry_l1.unique()), key="ind"); brd = c3.multiselect("板块", sorted(LAT.board.unique()), key="brd")
    only = c4.checkbox("只看合格", value=True, key="only"); d = LAT.copy()
    if q: d = d[d.code.str.contains(q) | d.name.str.contains(q, case=False)]
    if inds: d = d[d.industry_l1.isin(inds)]
    if brd: d = d[d.board.isin(brd)]
    if only: d = d[d.eligible_asof]
    sk = st.selectbox("排序列", ["rank_hgb", "rank_lgbm", "rank_ridge", "rank_dl_mlp", "rank_dl_gru", "rank_dl_tf", "model_agreement_pctile_std", "n_models_top50_of5", "r5", "r20", "r60", "vol20", "amount20", "code"], key="sortcol"); asc = st.radio("顺序", ["升序(排名1在前)", "降序"], horizontal=True, key="asc") == "升序(排名1在前)"
    d = d.sort_values(sk, ascending=asc, na_position="last"); show = ["code", "name", "exchange", "board", "industry", "eligible_asof", "ineligible_reason", "close_unadjusted", "r5", "r20", "r60", "vol20", "amount20", "rank_hgb", "rank_lgbm", "rank_ridge", "rank_dl_mlp", "rank_dl_gru", "rank_dl_tf", "hgb_score", "lgbm_score", "ridge_score", "dl_mlp_score", "dl_gru_score", "dl_tf_score", "model_agreement_pctile_std", "n_models_top50_of5", "models_in_top50", "quality_flags"]


    st.caption(f"{len(d)} 行;排名 1 = 得分最高(仅研究排名);score 非概率/收益率;行业为最新分类,非历史时点。")
    st.dataframe(d[show], width="stretch", height=560, column_config={"r5": st.column_config.NumberColumn(format="%.2%"), "r20": st.column_config.NumberColumn(format="%.2%"), "r60": st.column_config.NumberColumn(format="%.2%"), "vol20": st.column_config.NumberColumn(format="%.2%"), "amount20": st.column_config.NumberColumn(format="%.0f")})
    st.download_button("导出当前筛选(CSV,UTF-8-SIG)", d[show].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), file_name="filtered_stocks_20261016.csv", mime="text/csv")
    pick = st.selectbox("在单股页打开", ["-"] + [f"{r.code} {r.name}" for r in d.head(300).itertuples()], key="pick")
    if pick != "-": st.button("打开单股页", on_click=go_stock, args=(pick.split()[0],), key="openbtn")
# ---------------------------------------------------------------- page 3
elif page == PAGES[2]:
    opts = {f"{r.code} {r.name}": r.code for r in LAT.itertuples()}; opts["600060 海信视像(池外,不在1500样本)"] = OUTSIDER; names = list(opts)
    flt = st.text_input("搜索代码/名称筛选下拉", key="stk_filter"); sel_names = [n for n in names if flt in n] or names
    cur = st.session_state.get("code", OUTSIDER); idx = next((i for i, n in enumerate(sel_names) if opts[n] == cur), 0)
    label = st.selectbox("股票", sel_names, key="stk_select", **({} if "stk_select" in st.session_state else {"index": idx})); code = opts[label]; st.session_state["code"] = code
    inside = code != OUTSIDER; cards = load_cards()
    if inside:
        r = LAT[LAT.code == code].iloc[0]; name, ind, board, status = r["name"], r.industry, r.board, r.status
    else:
        cd = cards[OUTSIDER]; name, ind, board, status = cd["name"], cd["industry"], cd["board"], "上市"
        st.info("**池外打分,无历史OOS记录**:600060 不在1500只样本内。下面的分数由同一冻结模型对其做池外打分(特征分位数相对 2026-10-16 合格样本池计算),没有 walk-forward 历史预测,也没有预测后实现结果。")
    st.header(f"{code} {name}"); st.caption(f"行业(巨潮最新分类,非历史时点):{ind};板块:{board};状态:{status}")
    if inside:
        c = st.columns(6); c[0].metric("最新收盘(不复权)", f"{r.close_unadjusted:.2f}" if pd.notna(r.close_unadjusted) else "-"); c[1].metric("近5日", f"{r.r5:+.2%}" if pd.notna(r.r5) else "-"); c[2].metric("近20日", f"{r.r20:+.2%}" if pd.notna(r.r20) else "-"); c[3].metric("近60日", f"{r.r60:+.2%}" if pd.notna(r.r60) else "-")
        c[4].metric("20日波动率", f"{r.vol20:.2%}" if pd.notna(r.vol20) else "-"); c[5].metric("20日成交额中位数(亿)", f"{r.amount20 / 1e8:.2f}" if pd.notna(r.amount20) else "-")
        if r.eligible_asof: st.dataframe(pd.DataFrame({"模型": list(MODELS), "score(原始,非概率)": [r[f"{k}_score"] for k in MODELS.values()], "排名(1=最高)": [int(r[f"rank_{k}"]) for k in MODELS.values()], "合格股票中分位": [r[f"pctile_{k}"] for k in MODELS.values()]}), hide_index=True); st.write(f"五模型(HGB/LGBM/MLP/GRU/TF)分歧 = 百分位标准差 {r.model_agreement_pctile_std:.3f}(越小越一致,不是置信度);进入 Top50 的模型:{r.models_in_top50 or '无'}")
        else: st.error(f"信号日不合格,无排名:{r.ineligible_reason}")
        st.write("数据质量标记:", r.quality_flags or "无")
    else:
        c = st.columns(6); c[0].metric("最新收盘(不复权)", f"{cd['close']:.2f}"); c[1].metric("近5日", f"{cd['r5']:+.2%}"); c[2].metric("近20日", f"{cd['r20']:+.2%}"); c[3].metric("近60日", f"{cd['r60']:+.2%}"); c[4].metric("20日波动率", f"{cd['vol20']:.2%}"); c[5].metric("20日成交额中位数(亿)", f"{cd['amt20'] / 1e8:.2f}")
        NS = "未打分(池外,GRU/TF需序列输入)"; fm_ = lambda v, f: NS if v is None else f.format(v); st.dataframe(pd.DataFrame({"模型": list(MODELS), "score(原始,非概率)": [fm_(cd["scores"].get(m), "{:.6f}") for m in MODELS], "插入样本池后名次": [fm_(cd["ranks"].get(m), "{:d}") for m in MODELS], "分位": [fm_(cd["pct"].get(m), "{:.1%}") for m in MODELS]}), hide_index=True); st.write("说明:", ";".join(cd["flags"]))
    px_ = load_prices(code)
    if px_ is None: st.warning("无价格数据(接口未取到)。")
    else:
        nshow = st.slider("显示最近N个交易日", 60, 1400, 400, 20, key="nshow"); v = px_.tail(nshow); fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=v.date, open=v.aopen, high=v.ahigh, low=v.alow, close=v.aclose, name="日K(复权)", increasing_line_color="#d62728", decreasing_line_color="#2ca02c"), row=1, col=1)
        fig.add_trace(go.Scatter(x=v.date, y=v.ma20, name="MA20", line=dict(width=1.2)), row=1, col=1); fig.add_trace(go.Scatter(x=v.date, y=v.ma60, name="MA60", line=dict(width=1.2)), row=1, col=1)
        fig.add_trace(go.Bar(x=v.date, y=v.amount / 1e8, name="成交额(亿元)", marker_color="#888"), row=2, col=1); fig.update_layout(height=620, xaxis_rangeslider_visible=False, title=f"{code} {name} 日K(复权价=不复权价/新浪复权因子,仅历史)", margin=dict(t=50, b=10)); st.plotly_chart(fig, width="stretch")
        st.caption(f"价格来源:{px_.source.iloc[0] if 'source' in px_ else '-'};最后交易日 {px_.date.iloc[-1].date()};最新不复权收盘 {px_.close.iloc[-1]:.2f}。本页不绘制任何未来价格预测。")
    if inside:
        hist = load_hist_code(code)
        if hist.empty or not hist.eligible_asof.any(): st.info("该股票在 walk-forward 评估区间(2022-07~2026-09)内没有合格的信号日,无历史预测记录。")
        else:
            allm = hist[hist.eligible_asof].sort_values("signal_date"); figm = px.line(allm, x="signal_date", y="pctile", color="model", title="六模型历史排名分位(各模型 walk-forward 样本外;100%=最高)", category_orders={"model": list(MODELS)}); figm.update_layout(height=340, yaxis=dict(range=[0, 1])); st.plotly_chart(figm, width="stretch")
            st.dataframe(allm.groupby("model").agg(信号日数=("pctile", "size"), 分位均值=("pctile", "mean"), 分位p10=("pctile", lambda s: s.quantile(.1)), 分位p90=("pctile", lambda s: s.quantile(.9)), 最近一次分位=("pctile", "last")).reindex(list(MODELS)).round(3), width="stretch")
            m = st.radio("历史样本外预测模型", list(MODELS), horizontal=True, key="hist_model"); h = hist[(hist.model == m) & hist.eligible_asof].sort_values("signal_date"); hr = h.dropna(subset=["realized_rel_ret_5d"])
            st.subheader(f"历史样本外预测(walk-forward,{m};每个点=当时已训练模型在信号日的输出)")
            fig = go.Figure(); fig.add_trace(go.Scatter(x=h.signal_date, y=h.pctile, mode="lines+markers", name="当日排名分位(100%=最高)")); fig.add_trace(go.Bar(x=hr.signal_date, y=hr.realized_rel_ret_5d, name="之后5日相对收益(右轴)", yaxis="y2", marker_color=np.where(hr.realized_rel_ret_5d >= 0, "#d62728", "#2ca02c"), opacity=0.5))
            fig.update_layout(height=380, yaxis=dict(title="排名分位", range=[0, 1]), yaxis2=dict(title="5日相对收益", overlaying="y", side="right", tickformat=".0%"), legend=dict(orientation="h")); st.plotly_chart(fig, width="stretch")
            a, b = st.columns(2)
            with a:
                st.plotly_chart(px.scatter(hr, x="pctile", y="realized_rel_ret_5d", hover_data=["signal_date"], labels={"pctile": "信号日排名分位", "realized_rel_ret_5d": "之后5日相对收益"}, title="预测分位 vs 实现相对收益(每点=一个信号日)"), width="stretch")
            with b:
                n = len(hr); sp = hr.pctile.corr(hr.realized_rel_ret_5d, method="spearman") if n >= 5 else np.nan
                st.metric("样本数(有已到期标签的信号日)", n); st.metric("Spearman(分位, 实现相对收益)", f"{sp:.3f}" if pd.notna(sp) else "-")
                if n < 30: st.error("样本<30:**不可靠**,不要解读。")
                else: st.warning("样本≥30,但单只股票的时间序列相关噪声很大,仅描述,不是稳定命中率。")
                t = h[h.in_top50]; tr = t.dropna(subset=["realized_rel_ret_5d"]); st.write(f"入选 Top50 {len(t)} 次;其中有实现结果 {len(tr)} 次,实现相对收益均值 {tr.realized_rel_ret_5d.mean():+.2%}" if len(tr) else f"入选 Top50 {len(t)} 次")
                if len(tr) < 30: st.caption("入选次数少于30,均值不可靠。")
            st.dataframe(h.tail(12)[["signal_date", "model_id", "train_end_date", "score", "rank", "pctile", "in_top50", "buy_status", "realized_ret_5d", "realized_rel_ret_5d"]].sort_values("signal_date", ascending=False), hide_index=True, width="stretch")
            st.caption("train_end_date 为该模型训练样本的最后信号日(其标签到期日更早于信号日;purge 12日)。预测不含任何未来信息;2025-2026 非盲测。")
# ---------------------------------------------------------------- page 4
elif page == PAGES[3]:
    mc = load_csv("reports/model_comparison_v2.csv"); cs = load_csv("reports/cost_sensitivity_all_models.csv"); by = load_csv("reports/by_year_all_models.csv"); pp = load_csv("reports/paired_tests_all_models.csv"); cum = load_csv("reports/cumulative_net_models.csv"); sm = load_csv("historical_predictions_summary_v2.csv")
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
    st.subheader("深度学习模型说明"); st.markdown("MLP(11 维横截面分位特征,3 种子均值)、GRU(60 日序列,3 种子均值,GPU 训练)、小型 Transformer(**仅种子 42**)。训练窗口/purge/重训频率与 P0 相同;验证集=窗口内最后 60 日;未调参、未做消融。最新日分数的训练/验证:GRU/TF 最佳 epoch 偏早(见 dl_models_report.md)。")
# ---------------------------------------------------------------- page 5
else:
    st.subheader("数据质量与风险"); QF = LAT.copy(); fl = QF.quality_flags.str.split(";").explode().str.replace(r"\d+", "N", regex=True); fl = fl[fl != ""].value_counts().rename_axis("标记(数字已归一)").reset_index(name="只数"); a, b = st.columns(2)
    a.write("quality_flags 汇总"); a.dataframe(fl, hide_index=True, width="stretch"); b.write("不合格原因"); b.dataframe(LAT[~LAT.eligible_asof].reason_short.value_counts().rename_axis("原因").reset_index(name="只数"), hide_index=True)
    kind = st.selectbox("查看名单", ["价格单源", "成交量单位无法核验", "覆盖率<95%", "ST/*ST名称", "最新日涨停封板", "最新日跌停封板", "无换手率", "含未解释异常", "价格数据缺失"], key="kind")
    d = QF[QF.quality_flags.str.contains(kind.replace("<", "<").replace("无换手率", "无换手率"), regex=False)]; st.write(f"{len(d)} 只"); st.dataframe(d[["code", "name", "exchange", "board", "eligible_asof", "quality_flags", "ineligible_reason"]], hide_index=True, width="stretch")
    ev = load_events(); st.subheader("P0 异常事件日志(quality_event_log.csv,1272条/266只)"); st.dataframe(ev.cause.value_counts().rename_axis("原因").reset_index(name="条数"), hide_index=True)
    top = ev.groupby(["code", "name"]).size().sort_values(ascending=False).head(20).reset_index(name="事件数"); st.write("事件最多的20只"); st.dataframe(top, hide_index=True)
    st.markdown("**已知局限**:无历史ST序列(ST 5%涨跌停仅近似);退市股持仓只能按最后收盘价估值;行业为最新分类;价格主要来自新浪/腾讯(部分单源);920196、600317 成交量单位无法核验;2025-2026 非盲测;模型 score 未校准;DL 模型:TF 仅种子42,GRU 在 GPU 训练(CPU/GPU 数值会略有差异),未做调参/消融,600060 池外无 GRU/TF 分数。")
    m = p0_meta(); st.caption(f"模型版本:{m['model_version']};训练区间 {m['train_first_date']}~{m['train_last_date']};训练行数 {m['n_train_rows']:,}")

