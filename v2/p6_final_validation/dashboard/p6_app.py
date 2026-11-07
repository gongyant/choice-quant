# -*- coding: utf-8 -*-
"""P6 日常投资研究看板(只读;不训练、不连接券商、不下单)。端口 8506。"""
from pathlib import Path
import json, numpy as np, pandas as pd, streamlit as st
B = Path(__file__).resolve().parents[1]; R = B / "results"
rd = lambda f, **k: pd.read_csv(f, encoding="utf-8-sig", **k) if Path(f).exists() else pd.DataFrame()
st.set_page_config(page_title="P6 Top5 研究", layout="wide")
st.title("P6 最佳 Top5 日常研究看板")
st.error("研究排名 ≠ 买入推荐。预先登记买入标准：**不通过**。不连接券商、不下单。")

TOP = rd(R / "latest_top5.csv", dtype={"code": str})
CUR = rd(R / "current_holdings.csv", dtype={"code": str})
GATE = rd(R / "buy_gate.csv")
SC = rd(R / "strategy_comparison.csv")
EQ = rd(R / "b1_equity_h5.csv")
A0 = rd(R / "eq_A0_h5.csv")
PER = rd(R / "b1_periods_h5.csv")
FV = rd(B / "forward_validation.csv", dtype={"code": str})
FW = rd(B / "forward_registry.csv", dtype={"code": str})
meta = {}
if (R / "b1_recon_meta.json").exists(): meta = json.load(open(R / "b1_recon_meta.json", encoding="utf-8"))

# ----- Homepage -----
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("本期研究 Top5（B1）")
    b1 = TOP[TOP.strategy.str.startswith("B1")] if len(TOP) else TOP
    if len(b1):
        st.caption(f"信号日 {b1.signal_date.iloc[0]} · 全部为「研究候选,未达到买入条件」")
        st.dataframe(b1[["research_rank", "code", "name", "close", "volp", "dd60", "r20", "conclusion"]], use_container_width=True, hide_index=True)
    else: st.warning("未取得 latest_top5")
with c2:
    st.subheader("当前模拟持仓")
    if len(CUR):
        st.caption(f"来源信号 {CUR.origin_signal.iloc[0]} · 买入日 {CUR.entry_date.iloc[0]} · 状态 {CUR.status.iloc[0]}")
        st.dataframe(CUR[["code", "name", "entry_open", "ref_close", "chg_pct", "planned_exit"]], use_container_width=True, hide_index=True)
    else: st.info("暂无模拟持仓")
with c3:
    st.subheader("买入条件")
    if len(GATE):
        st.dataframe(GATE, use_container_width=True, hide_index=True)
    st.metric("数据更新", str(b1.signal_date.iloc[0]) if len(b1) else "未取得")
    st.caption("行情快照来自新浪收盘;合格池成员沿用 P4 缓存截至日。")

st.subheader("最近一期真实核验")
if len(FV) and len(FV.dropna(how="all")):
    st.dataframe(FV.tail(20), use_container_width=True, hide_index=True)
else:
    st.info("尚无到期前向核验（10-22 的 5 日标签最早约 10-30 成熟）。见 forward_registry 待核验队列。")
    if len(FW): st.dataframe(FW.tail(15), use_container_width=True, hide_index=True)

st.subheader("B1 / B0 / 基准累计表现（5日净，探索性回测）")
if len(SC):
    show = SC[SC.horizon == 5][["strategy", "cum_net", "mdd", "turnover", "fees_sum", "t_hac_excess", "beat_pool"]]
    st.dataframe(show.style.format({"cum_net": "{:.1%}", "mdd": "{:.1%}", "turnover": "{:.2f}", "fees_sum": "{:.1%}", "beat_pool": "{:.1%}", "t_hac_excess": "{:.2f}"}, na_rep="–"), use_container_width=True)
if len(EQ):
    chart = EQ[["signal_date", "equity"]].rename(columns={"equity": "B1"})
    if len(A0): chart = chart.merge(A0[["signal_date", "equity"]].rename(columns={"equity": "A0_原始低波"}), on="signal_date", how="left")
    if len(PER):
        pool_eq = (1 + PER.pool_ret).cumprod() * 1_000_000
        chart = chart.merge(pd.DataFrame({"signal_date": PER.signal_date, "POOL_毛": pool_eq}), on="signal_date", how="left")
    st.line_chart(chart.set_index("signal_date"))

# ----- Secondary pages -----
tab1, tab2, tab3, tab4, tab5 = st.tabs(["个股卡片", "A0–A3对照", "基准与成本", "LV变体", "对账与规则"])
with tab1:
    if len(TOP):
        opts = TOP.drop_duplicates("code")
        code = st.selectbox("股票", opts.code, format_func=lambda c: f"{c} {opts.set_index('code').name[c]}")
        row = TOP[TOP.code == code].iloc[0]
        st.write(row.drop(labels=[c for c in row.index if c.startswith("fund_") or c == "similar_score_history"]).to_dict())
        st.write("财务", {k.replace("fund_", ""): row[k] for k in row.index if k.startswith("fund_")})
        st.write("相似评分历史", json.loads(row.similar_score_history) if isinstance(row.similar_score_history, str) else row.similar_score_history)
        hist = rd(R / "b1_trades_h5.csv", dtype={"code": str})
        if len(hist):
            hh = hist[hist.code == code]
            st.subheader("历史入选与实现收益")
            st.dataframe(hh[["signal_date", "status", "entry_date", "exit_date", "slot_ret_gross", "slot_ret_net"]], use_container_width=True, hide_index=True)
        wn = rd(R / "latest_why_not.csv", dtype={"code": str})
        if len(wn): st.subheader("为何未选其他候选"); st.dataframe(wn, use_container_width=True, hide_index=True)
with tab2:
    st.dataframe(rd(R / "strategy_comparison.csv"), use_container_width=True)
    st.dataframe(rd(R / "a0a3_paired_tests.csv"), use_container_width=True)
    st.json(json.load(open(R / "a0a3_attribution.json", encoding="utf-8")) if (R / "a0a3_attribution.json").exists() else {})
with tab3:
    st.dataframe(rd(R / "part3_benchmarks.csv"), use_container_width=True)
    st.dataframe(rd(R / "part3_cost_time_splits.csv"), use_container_width=True)
    if (R / "part3_tstat_explain.json").exists(): st.json(json.load(open(R / "part3_tstat_explain.json", encoding="utf-8")))
with tab4:
    st.caption("预先限定 LV20/LV60/残差波动；不做参数搜索。残差波动历史更高不代表未来更好。")
    st.dataframe(rd(R / "part4_lv_variants.csv"), use_container_width=True)
with tab5:
    st.write("对账摘要", meta.get("recon_summary", {}))
    st.write("持有期说明", meta.get("overlap_note", {}))
    st.dataframe(rd(R / "b1_reconciliation.csv").tail(10), use_container_width=True)
    rp = B / "selection_rules_p6_B1_v1.json"
    if rp.exists(): st.json(json.load(open(rp, encoding="utf-8")))
