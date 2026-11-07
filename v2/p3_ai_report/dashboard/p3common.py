# -*- coding: utf-8 -*-
"""Shared read-only loaders for the P3 dashboard. Reads P3 outputs only; writes nothing; no broker connection."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import streamlit as st
H = Path(__file__).resolve().parents[1]; R = H / "results"; DATE = "20261021"
DISCLAIMER = "研究排名不是买入推荐。模型分数不是概率,也不是预期收益;情景区间是历史经验分布。P2 结论:信号未被证明在扣除交易成本后能赚钱。2025–2026 结果不是盲测。"
@st.cache_data(show_spinner=False)
def ranking(): return pd.read_csv(H / f"latest_research_ranking_{DATE}_v2.csv", dtype={"code": str}, encoding="utf-8-sig")
@st.cache_data(show_spinner=False)
def calib(): return pd.read_csv(R / "calibration_table.csv", encoding="utf-8-sig")
@st.cache_data(show_spinner=False)
def res(name, **kw): return pd.read_csv(R / name, encoding="utf-8-sig", **kw)
@st.cache_data(show_spinner=False)
def hist_comp(): return pd.read_parquet(R / "hist_composite.parquet")
@st.cache_data(show_spinner=False)
def world():
    z = np.load(H / "data" / f"world_{DATE}.npz"); return {k: z[k] for k in ["cal", "codes", "ADJf", "C", "O", "Hh", "Lo", "AMT"]}
@st.cache_data(show_spinner=False)
def info():
    ci = json.load(open(R / f"company_info_{DATE}_v2.json", encoding="utf-8"))
    f = R / f"fundamentals_{DATE}_v2.json"; fu = json.load(open(f, encoding="utf-8")) if f.exists() else {}
    return ci, fu
def quality(): return json.load(open(R / "data_quality_summary.json", encoding="utf-8"))