# -*- coding: utf-8 -*-
"""P4 Top5 量化选股仪表盘(只读结果文件,不训练、不连接券商、不下单)。"""
from pathlib import Path
import pandas as pd, numpy as np, streamlit as st, plotly.graph_objects as go
R = Path(__file__).resolve().parents[1] / "results"; FG = R.parent / "figures"
st.set_page_config(page_title="P4 Top5 量化选股", layout="wide")
st.title("P4 Top5 量化选股报告(探索性研究)")
st.warning("研究排名 ≠ 买入推荐。分数不是上涨概率或收益率。全部历史结果为探索性(2022-07~2026-09 已被反复使用)。本系统不连接券商、不下单。所有方案均未通过预先登记的买入标准:当前没有足够可信的买入机会。")
tab1, tab2, tab3 = st.tabs(["策略对比", "最新研究Top5", "个股走势与历史准确性"])
with tab1:
    U = pd.read_csv(R / "p4_comparison_all.csv", encoding="utf-8-sig").drop_duplicates(["strategy", "horizon", "period"])
    h = st.radio("持有期", [5, 20], horizontal=True, format_func=lambda x: f"{x}日"); per = st.selectbox("区间", U.period.unique())
    v = U[(U.horizon == h) & (U.period == per)][["strategy", "avg_holdings", "cum_gross", "cum_net", "cum_pool", "mean_excess_net", "t_excess", "beat_pool_share", "mdd_net", "avg_turnover", "total_fees", "rankic", "ndcg5", "passes_buy_criteria"]]
    st.dataframe(v.sort_values("cum_net", ascending=False).style.format({c: "{:.2%}" for c in ["cum_gross", "cum_net", "cum_pool", "mean_excess_net", "beat_pool_share", "mdd_net", "avg_turnover", "total_fees"]} | {"t_excess": "{:.2f}", "rankic": "{:.3f}", "ndcg5": "{:.3f}", "avg_holdings": "{:.2f}"}, na_rep="–"), use_container_width=True)
    f = FG / f"p4_equity_h{h}.png"
    if f.exists(): st.image(str(f))
with tab2:
    T = pd.read_csv(R / "p4_latest_top5_all.csv", dtype={"code": str}, encoding="utf-8-sig"); h2 = st.radio("期限 ", [5, 20], horizontal=True, format_func=lambda x: f"{x}日", key="h2")
    t = T[T.horizon == h2][["cand_rank", "code", "name", "board", "industry", "close_unadj", "select_score", "roe_ann_pct", "rev_yoy_pct", "np_yoy_pct", "debt_ratio_pct", "pe_ann", "pb", "r20", "dd60", "report_period", "notice_date", "conclusion"]]
    st.caption(f"数据截至 {T.signal_date.iloc[0]};方案 S4(E2 模型分 50% + 低波动 50% + 缓冲区),Top20→约束→Top5。")
    st.dataframe(t, use_container_width=True)
    cal = R / "s4_calibration.csv"
    if cal.exists(): st.subheader("S4 历史入选股的经验分布(不是概率)"); st.dataframe(pd.read_csv(cal, encoding="utf-8-sig"), use_container_width=True)
with tab3:
    T = pd.read_csv(R / "p4_latest_top5_all.csv", dtype={"code": str}, encoding="utf-8-sig"); opts = T.drop_duplicates("code"); code = st.selectbox("股票", opts.code, format_func=lambda c: f"{c} {opts.set_index('code').name[c]}")
    o = pd.read_csv(R / f"p4_ohlc_{code}.csv", encoding="utf-8-sig").dropna(subset=["close"])
    fig = go.Figure(go.Candlestick(x=o.date, open=o.open, high=o.high, low=o.low, close=o.close, increasing_line_color="red", decreasing_line_color="green")); fig.update_layout(height=420, xaxis_rangeslider_visible=False, title="真实日K线(未复权,近300交易日)")
    st.plotly_chart(fig, use_container_width=True)
    hh = pd.read_csv(R / f"p4_hist_{code}.csv", encoding="utf-8-sig")
    f2 = go.Figure(); f2.add_bar(x=hh.signal_date, y=hh.realized_ex5, name="实际5日超额(相对池)"); f2.add_scatter(x=hh.signal_date, y=hh.e2_pctile - 0.5, name="E2 模型分位-0.5(历史 OOS 预测)", yaxis="y2")
    f2.update_layout(height=380, yaxis2=dict(overlaying="y", side="right"), title="历史预测 vs 实际(207 个信号日)"); st.plotly_chart(f2, use_container_width=True)
    a = pd.read_csv(R / "p4_top5_stock_accuracy.csv", dtype={"code": str}, encoding="utf-8-sig"); st.subheader("该股历史准确性"); st.dataframe(a[a.code == code], use_container_width=True)
