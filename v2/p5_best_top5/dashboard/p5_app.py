# -*- coding: utf-8 -*-
"""P5 Top5 研究仪表盘(只读结果文件;不训练、不连接券商、不下单)。端口 8505;P4 页面(8504)保持不变。"""
from pathlib import Path
import json, pandas as pd, numpy as np, streamlit as st
B = Path(__file__).resolve().parents[1]; R = B / "results"
rd = lambda f, **k: pd.read_csv(f, encoding="utf-8-sig", **k) if Path(f).exists() else pd.DataFrame()
st.set_page_config(page_title="P5 Top5 研究", layout="wide"); st.title("P5 Top5 研究仪表盘(探索性)")
st.error("研究排名 ≠ 买入推荐。select_score 不是上涨概率或收益率。预先登记的买入标准:当前所有方案均【不通过】。本系统不连接券商、不下单。")
t1, t2, t3, t4, t5 = st.tabs(["B0–B3 回测对比", "最新 Top5(10-22)", "五只股票对比", "历史实际与准确率", "前向验证 B1 vs B3 / 买入标准"])
pct = lambda cols: {c: "{:.2%}" for c in cols}
with t1:
    S = rd(R / "a1v2_summary.csv"); h = st.radio("持有期", [5, 20], horizontal=True, format_func=lambda x: f"{x}日")
    st.caption("B0=原始低波Top5; B1=低波+约束+缓冲; B2=E2 LightGBM+约束+缓冲; B3=S4(E2 50%+低波 50%)+约束+缓冲。同一引擎,双边0.30%。并购退市按换股条款估值(见敏感性)。")
    st.dataframe(S[S.horizon == h].style.format(pct(["cum_gross", "cum_net", "cum_pool_gross", "mean_net", "mean_excess", "beat_pool", "win_abs", "mdd", "turnover", "fees_sum"]) | {"t_hac_excess": "{:.2f}"}, na_rep="–"), use_container_width=True)
    st.subheader("配对检验(每期净收益差)"); PT = rd(R / "a1v2_paired_tests.csv"); st.dataframe(PT[PT.horizon == h].style.format(pct(["mean_diff", "ci_lo", "ci_hi", "share_pos"]) | {"hac_t": "{:.2f}"}), use_container_width=True)
    st.subheader("分年度净累计"); Y = rd(R / "a1v2_by_year.csv"); st.dataframe(Y[Y.horizon == h].pivot_table(index="year", columns="strategy", values="cum_net").style.format("{:.1%}", na_rep="–"), use_container_width=True)
    st.subheader("并购退市估值敏感性"); st.dataframe(rd(R / "a1v2_merger_sensitivity.csv").query("horizon == @h"), use_container_width=True); st.dataframe(rd(R / "a1v2_merger_delist_valuation.csv"), use_container_width=True)
with t2:
    L = rd(R / "b4_latest_lists_20261022.csv", dtype={"code": str}); s = st.radio("方案", ["B1", "B3"], horizontal=True); h2 = st.radio("期限", [5, 20], horizontal=True, key="h2", format_func=lambda x: f"{x}日")
    v = L[(L.strategy == s) & (L.horizon == h2)]; st.caption(f"信号日 2026-10-22(收盘后);模型版本:{v.model_version.iloc[0] if len(v) else '未取得'};模拟成交=10-23 开盘。")
    st.dataframe(v[v.in_final5][["code", "name", "industry", "close_1022", "select_score", "vol20", "dd60", "r20", "sim_weight", "sim_shares_per_100k"]], use_container_width=True)
    with st.expander("Top20 候选"): st.dataframe(v, use_container_width=True)
with t3:
    C = rd(R / "b4_five_stock_compare.csv", dtype={"code": str}); st.caption("PE_TTM = 10-22 收盘价 / TTM EPS(H1 2026 + FY2025 − H1 2025);ROE 同时给出原始半年值与年化值;券商/保险的经营现金流含客户资金,不可与一般企业对比。")
    st.dataframe(C.set_index("name").T, use_container_width=True)
with t4:
    HO = rd(R / "a1v2_holdings.csv", dtype={"code": str}); PER = rd(R / "a1v2_periods.csv"); s4 = st.selectbox("方案", sorted(PER.strategy.unique())); h4 = st.radio("期限 ", [5, 20], horizontal=True, key="h4", format_func=lambda x: f"{x}日")
    p = PER[(PER.strategy == s4) & (PER.horizon == h4)].copy(); p["净超额"] = p.ret_net - p.pool_ret
    c1, c2, c3 = st.columns(3); c1.metric("跑赢池比例", f"{(p['净超额'] > 0).mean():.1%}"); c2.metric("绝对正收益期比例", f"{(p.ret_net > 0).mean():.1%}"); c3.metric("净累计", f"{np.prod(1 + p.ret_net) - 1:.1%}")
    st.line_chart(p.set_index("signal_date")[["ret_net", "pool_ret"]].add(1).cumprod())
    hh = HO[(HO.strategy == s4) & (HO.horizon == h4) & HO.status.isin(["FILLED", "MARKED_LAST_CLOSE_UNSOLD"])]
    acc = hh.groupby(["code", "name"]).agg(入选次数=("slot_ret_gross", "size"), 平均毛收益=("slot_ret_gross", "mean"), 上涨比例=("slot_ret_gross", lambda z: (z > 0).mean())).reset_index().sort_values("入选次数", ascending=False)
    st.subheader("个股历史入选与实际结果(毛收益,开盘到开盘)"); st.dataframe(acc, use_container_width=True)
with t5:
    st.subheader("前向登记(只追加)"); st.dataframe(rd(B / "forward_registry.csv", dtype={"code": str}), use_container_width=True)
    FV = rd(R / "b5_forward_by_period.csv")
    if len(FV): st.subheader("已到期前向结果 B1 vs B3"); st.dataframe(FV, use_container_width=True)
    else: st.info("尚无到期的前向结果(10-22 信号最早于 5 个交易日后到期;需先运行 run_weekly_verify_p5.ps1)。")
    st.subheader("预先登记的买入标准 p5_buy_criterion_v1"); cr = B / "buy_criterion_p5_v1.json"
    if cr.exists(): st.json(json.load(open(cr, encoding="utf-8")))
    G = rd(R / "b5_buy_gate_latest.csv") if (R / "b5_buy_gate_latest.csv").exists() else rd(R / "b5_buy_gate.csv"); st.dataframe(G, use_container_width=True)
