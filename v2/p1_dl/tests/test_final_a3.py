"""P1-A3 final integration tests (read-only checks on delivered files)."""
import json, hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest
H = Path(__file__).resolve().parent.parent; V2 = H.parent
sys.path.insert(0, str(H))
rd = lambda p, **k: pd.read_csv(H / p, dtype=str, encoding="utf-8-sig", keep_default_na=False, **k)
BANNED = ("pred_excess_return", "p_outperform", "prob", "win_rate_pred", "expected_return")
@pytest.fixture(scope="module")
def v2(): return rd("latest_all_stocks_20261016_v2.csv")
@pytest.fixture(scope="module")
def a1(): return rd("latest_all_stocks_20261016.csv")
def test_v2_rows_unique_leading_zeros(v2):
    assert len(v2) == 1500 and v2.code.is_unique and v2.code.str.len().eq(6).all() and v2.code.str.match(r"^\d{6}$").all()
    assert (v2.code.str.startswith("0")).sum() > 100
def test_v2_no_probability_or_return_columns(v2):
    assert not [c for c in v2.columns if any(b in c.lower() for b in BANNED)]
    for c in ["dl_mlp_score", "dl_gru_score", "dl_tf_score", "rank_dl_mlp", "rank_dl_gru", "rank_dl_tf", "pctile_dl_mlp", "pctile_dl_gru", "pctile_dl_tf", "model_agreement_pctile_std"]: assert c in v2.columns
def test_v2_eligibility_identical_to_a1_and_a1_untouched(v2, a1):
    assert (v2.eligible_asof == a1.eligible_asof).all() and (v2.code == a1.code).all() and (v2.eligible_asof == "True").sum() == 1178
    for c in a1.columns:
        if c not in ("dl_mlp_score", "dl_gru_score", "rank_dl_gru"): assert (v2[c] == a1[c]).all(), c
    assert (a1.dl_mlp_score == "").all() and (a1.dl_gru_score == "").all()
    assert len((H / "latest_all_stocks_20261016.csv").read_text(encoding="utf-8-sig").splitlines()) == 1501
@pytest.mark.parametrize("k,col,src", [("mlp", "dl_mlp", "mlp_score_mean"), ("gru", "dl_gru", "gru_score_mean"), ("tf", "dl_tf", "tf_score_mean")])
def test_merged_dl_scores_equal_source(v2, k, col, src):
    s = pd.read_csv(H / "scores" / f"dl_{k}_scores_20261016.csv", dtype={"code": str}).set_index("code"); e = v2[v2.eligible_asof == "True"].set_index("code")
    assert np.allclose(pd.to_numeric(e[f"{col}_score"]).values, s.loc[e.index, src].values, rtol=0, atol=1e-15)
    assert (pd.to_numeric(e[f"rank_{col}"]).values == s.loc[e.index, "rank"].values).all()
    ine = v2[v2.eligible_asof != "True"]; assert (ine[f"{col}_score"] == "").all() and (ine[f"rank_{col}"] == "").all()
    assert set(v2.code) - set(s.index) == {"600485", "600677"}
def test_model_agreement_definition(v2):
    e = v2[v2.eligible_asof == "True"]; P = e[[f"pctile_{m}" for m in ["hgb", "lgbm", "dl_mlp", "dl_gru", "dl_tf"]]].astype(float)
    assert np.allclose(P.std(axis=1, ddof=0).values, e.model_agreement_pctile_std.astype(float).values, atol=1e-5)
    R = e[[f"rank_{m}" for m in ["hgb", "lgbm", "dl_mlp", "dl_gru", "dl_tf"]]].astype(float); assert ((R <= 50).sum(axis=1).values == e.n_models_top50_of5.astype(int).values).all()
def test_top_bottom_v2_and_multi_model_list(v2):
    t = pd.read_csv(H / "reports" / "top_bottom_20261016_v2.csv", dtype={"code": str}, encoding="utf-8-sig"); assert len(t) == 200 and t.groupby(["list", "model"]).size().eq(20).all()
    g = pd.read_csv(H / "reports" / "multi_model_top50_20261016.csv", dtype={"code": str}, encoding="utf-8-sig"); e = v2[v2.eligible_asof == "True"]
    assert set(g.code) == set(e.code[e.n_models_top50_of5.astype(int) >= 3]) and g.name.notna().all() and len(g) == 28
@pytest.fixture(scope="module")
def hp(): return pd.read_parquet(H / "historical_predictions_all_models.parquet", columns=["signal_date", "code", "model", "model_id", "eligible_asof", "score", "rank", "pctile", "in_top50", "realized_ret_5d", "realized_rel_ret_5d", "n_pool"])
def test_all_models_key_unique_and_models(hp):
    assert not hp.duplicated(["signal_date", "code", "model_id"]).any() and len(hp) == 1860516 and set(hp.model) == {"HGB", "LGBM", "Ridge", "MLP", "GRU", "TF"}
    assert hp.code.str.len().eq(6).all() and hp.groupby("model").size().eq(310086).all()
    assert hp.groupby(["model", "signal_date"]).in_top50.sum().eq(50).all()
def test_all_models_originals_unchanged_and_dl_scores_match_oos(hp):
    old = pd.read_parquet(H / "historical_predictions.parquet", columns=["signal_date", "code", "model", "model_id", "score", "rank"])
    a = hp[hp.model.isin(["HGB", "LGBM", "Ridge"])].reset_index(drop=True); assert len(a) == len(old) and (a.model_id.values == old.model_id.values).all() and np.allclose(a.score.fillna(0).values, old.score.fillna(0).values, atol=0)
    for k, M in [("mlp", "MLP"), ("gru", "GRU"), ("tf", "TF")]:
        o = pd.read_parquet(H / "scores" / f"dl_{k}_oos_predictions.parquet", columns=["signal_date", "code", f"{k}_score_mean", "eligible_asof", "rank_in_pool"]); h = hp[hp.model == M].merge(o, on=["signal_date", "code"], validate="1:1")
        e = h[h.eligible_asof_x]; assert np.allclose(e.score.values, e[f"{k}_score_mean"].values, atol=0) and (e["rank"].values == e.rank_in_pool.values.astype("float32")).all()
        assert np.allclose(e.pctile.values, (1 - (e["rank"] - 1) / (e.n_pool - 1)).values, atol=1e-6)
def test_dl_period_returns_reconciled():
    rc = pd.read_csv(H / "reports" / "reconciliation_dl.csv"); assert rc.max_abs_diff.max() < 1e-12 and set(rc.model) == {"MLP", "GRU", "TF"} and len(rc) >= 30
    cum = pd.read_csv(H / "reports" / "cumulative_net_models.csv")
    for k, M in [("mlp", "MLP"), ("gru", "GRU"), ("tf", "TF")]: pr = pd.read_csv(H / "dl" / "results" / k / "period_returns.csv"); assert np.abs(cum[M].values - pr[f"{M}_mean_ret_c30"].values).max() < 1e-15
def test_per_stock_and_sector_v2():
    ps = pd.read_csv(H / "per_stock_summary_v2.csv", dtype={"code": str}, encoding="utf-8-sig"); assert len(ps) == 1500 and ps.code.is_unique and ps.code.str.len().eq(6).all()
    for k in ["mlp", "gru", "tf"]:
        assert ((ps[f"n_pairs_{k}"] < 30) == ps[f"corr_reliability_{k}"].str.contains("不可靠")).all()
    ss = pd.read_csv(H / "sector_summary_v2.csv", encoding="utf-8-sig"); assert {"MLP", "GRU", "TF", "HGB"} <= set(ss.model) and len(ss) == 126
def test_recent20_v2():
    r = pd.read_csv(H / "reports" / "recent20_metrics_v2.csv"); assert set(r.model) == {"HGB", "LGBM", "Ridge", "MLP", "GRU", "TF"} and r.groupby("model").size().eq(20).all()
def test_forward_registry_frozen_hashes():
    r = pd.read_csv(H / "forward_registry_20261016.csv", dtype={"code": str}, encoding="utf-8-sig"); assert len(r) == 300 and r.groupby("model").code.nunique().eq(50).all() and r.code.str.len().eq(6).all()
    assert (r.source_table_sha256 == hashlib.sha256((H / "latest_all_stocks_20261016_v2.csv").read_bytes()).hexdigest()).all()
    m = json.load(open(H / "forward_registry_20261016_manifest.json", encoding="utf-8")); assert len(m["sha256"]) == 22
    for k, v in m["sha256"].items(): assert hashlib.sha256((V2 / k).read_bytes()).hexdigest() == v, k
def test_report_content_and_top_lists_match_table(v2):
    t = (H / "p1_dl_report.md").read_text(encoding="utf-8"); assert "研究排名" in t and "不是概率" in t and "池外未打分" in t and "没有任何一个" in t
    e = v2[v2.eligible_asof == "True"].assign(r=lambda d: d.rank_hgb.astype(int)).sort_values("r")
    for r in e.head(20).itertuples(): assert f"| {r.r} | {r.code} | {r.name} |" in t
    e = v2[v2.eligible_asof == "True"].assign(r=lambda d: d.rank_dl_gru.astype(int)).sort_values("r")
    for r in e.head(20).itertuples(): assert f"| {r.r} | {r.code} | {r.name} |" in t
    assert not [b for b in ("pred_excess_return_5d", "p_outperform_5d") if b in t]
def test_dashboard_v2_smoke():
    import smoke_dashboard_v2 as s
    res, secs = s.run_all(); assert res and all(x[1] for x in res), [x for x in res if not x[1]]
    ps = (H / "run_dashboard.ps1").read_text(encoding="utf-8"); assert "stock_dashboard_v2.py" in ps and (H / "run_dashboard_v1.ps1").exists()
