# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import plotly.express as px
import p3common as P
st.set_page_config(page_title="历史预测 vs 实际", layout="wide")
st.title("历史预测与实际涨跌对比"); st.error("⚠️ " + P.DISCLAIMER)
S = P.res("hist_top5_summary.csv"); PR = P.res("hist_top5_by_period.csv"); Y = P.res("hist_bins_by_year.csv", dtype={"year": str}); PK = P.res("hist_top5_picks.csv", dtype={"code": str})
st.subheader("历史逐期 Top5 汇总(成本前/扣 0.30%/0.60% 往返)"); st.dataframe(S, hide_index=True, use_container_width=True)
cost = st.selectbox("成本", ["成本前", "扣 0.30%", "扣 0.60%"], key="cost"); cc = {"成本前": "ret5", "扣 0.30%": "ret5_net_30bp", "扣 0.60%": "ret5_net_60bp"}[cost]
d = PR.copy(); d["r"] = d[cc].where(d.n_picks > 0, 0.0); d = d.sort_values("signal_date"); d["累计"] = d.groupby("strategy").r.transform(lambda s: (1 + s).cumprod() - 1)
pool = d[d.strategy == "v1"].assign(strategy="全池平均(成本前)"); pool["累计"] = (1 + pool.pool_ret5).cumprod() - 1
st.plotly_chart(px.line(pd.concat([d, pool]), x="signal_date", y="累计", color="strategy", title=f"累计收益({cost};2025 起为探索期,非盲测)"), use_container_width=True)
hz = st.radio("期限", ["5d", "20d"], horizontal=True, key="hz")
st.plotly_chart(px.line(Y[Y.horizon == hz], x="bin", y="mean_rel", color="year", markers=True, title=f"{hz} 各分数组平均相对收益(分年度)"), use_container_width=True)
st.dataframe(Y[Y.horizon == hz].pivot(index="year", columns="bin", values="beat_median").round(3), use_container_width=True); st.caption("上表:各组跑赢当天池中位数的比例")
strat = st.selectbox("逐只明细", ["v1", "v2", "F3基线"], key="strat"); st.dataframe(PK[PK.strategy == strat].sort_values("signal_date", ascending=False), hide_index=True, use_container_width=True)