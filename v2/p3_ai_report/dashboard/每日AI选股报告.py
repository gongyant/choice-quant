# -*- coding: utf-8 -*-
"""每日AI选股报告(P3,只读)。启动: .\run_dashboard_p3.ps1  -> http://127.0.0.1:8503 。P2 仪表盘(8502)和 P1(8501)不受影响。"""
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import p3common as P
st.set_page_config(page_title="每日AI选股报告", layout="wide")
st.title("每日AI选股报告"); st.error("⚠️ " + P.DISCLAIMER)
o = P.ranking(); CT = P.calib(); q = P.quality(); sd = o.signal_date.iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("最新数据日期", sd); c2.metric("样本股 / 可打分", f"{len(o)} / {int(o.eligible_asof.sum())}")
t5 = CT[(CT.horizon == "5d") & (CT.bin == 10)].iloc[0]; c3.metric("最高组 5日跑赢>2% 频率 (基准)", f"{t5.freq_outperform:.1%} ({t5.base_outperform:.1%})"); c4.metric("可信度(原定义)", t5.credibility)
st.caption(f"数据质量:最后日期={sd} 的 {q.get(f'最后日期 = 2026-10-21', '—')} 只;规则 v2 sha256 {o.rules_sha256.iloc[0][:12]}…;最高组跑输>2% 频率 {t5.freq_underperform:.1%}(全体 {t5.base_underperform:.1%})。")
ver = st.radio("规则版本", ["v2(收紧趋势/防追高)", "v1(原规则)"], horizontal=True, key="ver")
col = "top5_v2" if ver.startswith("v2") else "top5"; srt = "sel" if col == "top5_v2" else "sel_score"
show = ["code", "name", "close_unadj", "chg_5d", "chg_20d", "trend", "composite_pctile", "pctile_HGB", "pctile_LGBM", "pctile_TF", "model_disagreement", "risk_level", "s5d_credibility"]
st.subheader("重点观察 Top5"); top = o[o[col]].sort_values(srt, ascending=False); st.dataframe(top[show + ["conclusion_v2" if col == "top5_v2" else "conclusion"]], hide_index=True, use_container_width=True)
if col == "top5_v2": st.markdown("**下跌观察名单(暂不买入)**"); st.dataframe(o[o.watch_v2].sort_values("sel", ascending=False)[show], hide_index=True, use_container_width=True)
st.subheader("高风险 / 回避名单"); st.dataframe(o[o.avoid5].sort_values("avoid_score", ascending=False)[show + ["st_or_delist_risk", "consec_limit_down", "avoid_score"]], hide_index=True, use_container_width=True)
st.subheader("搜索任意股票")
qq = st.text_input("代码或名称", value="", key="q")
cands = o[o.code.str.contains(qq) | o.name.astype(str).str.contains(qq)] if qq else o[o[col]]
if len(cands) == 0: st.warning("没有匹配的股票")
else:
    code = st.selectbox("选择", list(cands.code), format_func=lambda c: f"{c} {o.set_index('code').name[c]}", key="code"); r = o.set_index("code").loc[code]
    a, b, c = st.columns(3); a.metric("收盘", f"{r.close_unadj:.2f}"); b.metric("综合分位", "—" if pd.isna(r.composite_pctile) else f"{r.composite_pctile:.0%}"); c.metric("风险等级 / 趋势", f"{r.risk_level} / {r.trend}")
    st.write({"5日": f"{r.chg_5d:+.1%}", "20日": f"{r.chg_20d:+.1%}", "v1 结论": r.conclusion if isinstance(r.conclusion, str) else "", "v2 结论": r.conclusion_v2 if isinstance(r.conclusion_v2, str) else "", "可选池": bool(r.eligible_asof)})
    ci, fu = P.info()
    if code in ci: st.markdown(f"**主营**:{ci[code]['main']}  \n**近期公告**:" + (";".join(f"{x} {y}" for x, y in ci[code]["ann"]) or "未取得"))
    W = P.world(); j = list(W["codes"]).index(code); n = 120; cal = pd.to_datetime(W["cal"][-W["C"].shape[0]:])[-n:]
    C = W["C"][-n:, j]; A = W["ADJf"][-n:, j]; k = np.where(np.isfinite(C))[0]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, row_heights=[0.7, 0.3], vertical_spacing=0.08, subplot_titles=("K 线(前复权,对齐最新收盘)+ 未来情景区间", "历史 OOS 综合分位"))
    if len(k):
        f = (A / np.where(np.isfinite(C), C, np.nan)) * (C[k[-1]] / A[k[-1]])
        fig.add_trace(go.Candlestick(x=cal, open=W["O"][-n:, j] * f, high=W["Hh"][-n:, j] * f, low=W["Lo"][-n:, j] * f, close=C * f, name="K线", increasing_line_color="#d62728", decreasing_line_color="#2ca02c"), 1, 1)
        fig.add_trace(go.Scatter(x=cal, y=pd.Series(A * C[k[-1]] / A[k[-1]]).rolling(20, min_periods=15).mean(), name="MA20", line=dict(width=1)), 1, 1)
        if pd.notna(r.calib_bin):
            for hz, days, colr in [("5d", 7, "orange"), ("20d", 28, "purple")]:
                t = CT[(CT.horizon == hz) & (CT.bin == int(r.calib_bin))].iloc[0]; x = cal[-1] + pd.Timedelta(days=days); last = C[k[-1]]
                fig.add_trace(go.Scatter(x=[x, x], y=[last * (1 + t.abs_q10), last * (1 + t.abs_q90)], mode="lines", line=dict(width=10, color=colr), opacity=0.5, name=f"{hz} 10%-90% 历史经验区间"), 1, 1)
                fig.add_trace(go.Scatter(x=[x], y=[last * (1 + t.abs_q50)], mode="markers", marker=dict(color=colr), name=f"{hz} 中位"), 1, 1)
    hc = P.hist_comp(); hc = hc[hc.code == code]; fig.add_trace(go.Scatter(x=pd.to_datetime(hc.signal_date), y=hc.composite, mode="lines+markers", marker=dict(size=3), name="历史综合分位"), 2, 1)
    fig.update_layout(height=650, xaxis_rangeslider_visible=False); st.plotly_chart(fig, use_container_width=True)
    if pd.notna(r.calib_bin):
        st.caption("情景:" + ";".join(f"{hz} 跑赢/持平/跑输 = {t.freq_outperform:.0%}/{t.freq_flat:.0%}/{t.freq_underperform:.0%},可信度:{t.credibility}" for hz in ("5d", "20d") for t in [CT[(CT.horizon == hz) & (CT.bin == int(r.calib_bin))].iloc[0]]))
    else: st.caption("该股不在今日可打分池中,无情景区间。")