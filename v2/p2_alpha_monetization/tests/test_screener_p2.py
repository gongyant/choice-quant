# -*- coding: utf-8 -*-
"""Tests for the research screener export (results/research_screener_20261016.csv) and the dashboard page 研究选股器."""
import sys, re, json, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, pytest
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); R = HERE / "results"; P1 = HERE.parent / "p1_dl"
MODELS = ["HGB", "LGBM", "Ridge", "MLP", "GRU", "TF"]; K1 = {"HGB": "hgb", "LGBM": "lgbm", "Ridge": "ridge", "MLP": "dl_mlp", "GRU": "dl_gru", "TF": "dl_tf"}
@pytest.fixture(scope="module")
def S(): return pd.read_csv(R / "research_screener_20261016.csv", dtype={"code": str}, encoding="utf-8-sig")
@pytest.fixture(scope="module")
def LAT(): return pd.read_csv(P1 / "latest_all_stocks_20261016_v2.csv", dtype={"code": str}, encoding="utf-8-sig").set_index("code")
@pytest.fixture(scope="module")
def META(): return json.load(open(R / "research_screener_20261016_meta.json", encoding="utf-8"))
def test_real_codes_names_and_columns(S):
    assert len(S) >= 1490 and S.code.str.fullmatch(r"\d{6}").all() and S.code.is_unique and S.name.notna().all() and (S.name.astype(str).str.len() > 0).all()
    need = ["industry_l1", "exchange", "board", "n_models_top50_of6", "pctile_std_6", "tf3_pctile_mean", "base3_pctile_mean", "r20", "vol20", "turn20", "amihud20", "amt20",
            "limit_up_locked_t", "st_name_current", "low_liquidity_flag", "needs_special_access", "a2w_mean6"] + [f"{a}_{m}" for m in MODELS for a in ("rank", "pctile", "a2w")]
    assert not [c for c in need if c not in S.columns]
def test_frozen_scores_identical_to_p1(S, LAT):
    L = LAT.reindex(S.code); e = S.eligible_asof.values
    assert e.sum() == (LAT.eligible_asof.astype(str) == "True").sum()
    for m, k in K1.items():
        assert np.allclose(S[f"score_{m}"].values[e], L[f"{k}_score"].values[e]) and np.array_equal(S[f"rank_{m}"].values[e], L[f"rank_{k}"].values[e]), m
def test_agreement_columns(S):
    e = S[S.eligible_asof]; top = np.column_stack([e[f"rank_{m}"] <= 50 for m in MODELS]); P = e[[f"pctile_{m}" for m in MODELS]].values
    assert np.array_equal(top.sum(1), e.n_models_top50_of6.values) and np.allclose(P.std(1), e.pctile_std_6) and np.allclose(P.mean(1), e.pctile_mean_6)
    assert ((P >= 0) & (P <= 1)).all() and e.models_in_top50.fillna("").str.count(r"\w+").sum() == top.sum()
def test_baselines_and_exposures(S, LAT):
    e = S[S.eligible_asof]; L = LAT.reindex(e.code)
    assert np.allclose(e[["base_REV5_pctile", "base_LOWVOL_pctile", "base_ILLIQ_pctile"]].mean(1), e.base3_pctile_mean)
    assert np.allclose(e.r20.values, L.r20.values, atol=1e-6, equal_nan=True) and np.allclose(e.amt20.values, L.amount20.values, rtol=1e-6, equal_nan=True)
    assert np.corrcoef(e.base_REV5_pctile, -e.r5)[0, 1] > 0.85 and (e.amihud20 >= 0).all()
def test_tradability_flags(S):
    assert (S.st_name_current == S.name.str.upper().str.contains("ST")).all()
    assert (S.needs_special_access == (S.board.astype(str).str.contains("科创") | S.board.astype(str).str.contains("北交") | (S.exchange == "BJ"))).all()
    assert (S.low_liquidity_flag == ~(S.amt20.fillna(0) >= 5e7)).all() and S.limit_up_locked_t.dtype == bool
def test_tf_three_seed_average(S, META):
    assert META["tf3_seeds"] == [42, 43, 44] and META["tf3_signal_date"] < "2026-10-16" and (S.tf3_signal_date == META["tf3_signal_date"]).all()
    e = S[S.tf3_pctile_mean.notna()]; assert len(e) > 1000 and e.tf3_pctile_mean.between(0, 1).all()
def test_a2_research_weights(S, META):
    for m in MODELS:
        ex = json.load(open(R / "b_runs" / f"{m}_C2_explicit_extras.json", encoding="utf-8")); assert META["a2"][m]["params"] == ex["A2_schedule"][-1]
        cap = float(ex["A2_schedule"][-1].split("_")[2][1:]); w = S[f"a2w_{m}"]
        assert (w.fillna(0) >= 0).all() and w.sum() <= 1 + 1e-9 and w.max() <= cap + 1e-9 and (w > 0).sum() <= 100 and S.loc[w > 0, "eligible_asof"].all(), m
    assert "research simulation" in META["note"].lower() and S.research_note.str.contains("不是买入推荐").all()
def test_screener_does_not_touch_p0_p1():
    src = (HERE / "c_screener.py").read_text(encoding="utf-8"); assert "forward_registry" not in src.replace("nothing under p0/ or p1_dl/ or forward_registry is written", "")
    writes = re.findall(r"(to_csv|to_parquet|open\([^)]*\"w\")", src); assert len(writes) == 2 and "OUT" in src
    t0 = dt.datetime(2026, 10, 19, 13, 23).timestamp(); assert (P1 / "latest_all_stocks_20261016_v2.csv").stat().st_mtime < t0
    for f in (HERE.parent / "p0").glob("forward_registry*"): assert f.stat().st_mtime < t0
def test_dashboard_screener_page():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(HERE / "stock_dashboard_p2.py"), default_timeout=240); at.run(); at.sidebar.radio(key="page").set_value("研究选股器").run(); assert not at.exception
    top = at.error[0].value; assert "不是买入推荐" in top and "不是概率" in top and "未被证明可在成本后变现" in top
    assert any("非历史时点" in c.value for c in at.caption) and len(at.dataframe) >= 2
    at.selectbox(key="s_code").set_value(at.selectbox(key="s_code").options[0]).run(); assert not at.exception
    assert len(at.get("plotly_chart")) >= 3