"""P1-B1 tests: MLP leakage / isolation / reproducibility. Synthetic fixtures come from the P0 test suite (read-only import); artifact tests skip if the artifact is not yet produced."""
import sys, copy
from pathlib import Path
import numpy as np, pandas as pd, pytest
HERE = Path(__file__).resolve().parent; P1 = HERE.parent; V2 = P1.parent
sys.path.insert(0, str(P1 / "dl")); sys.path.insert(0, str(V2 / "p0" / "leakage_unit_tests"))
import dl_common as C; import dl_mlp as M
from synth import make_synth, pipeline, tamper_future
L = C.L
SMALL = dict(max_epochs=3, patience=2)
@pytest.fixture(scope="module")
def world():
    X = make_synth(T=420, N=80, seed=3); P = pipeline(X); m_tr = P["elig"] & np.isfinite(P["yw"]); return X, P, m_tr
I = 300
def test_split_geometry_and_label_expiry(world):
    X, P, m_tr = world; sp = C.split_train_val(P["Xq"], P["yw"], m_tr, X.cal, I)
    assert len(sp["Xtr"]) > 3000 and len(sp["Xva"]) > 500
    assert sp["dtr"].max() + C.H + 1 < sp["dva"].min()                 # every training label has expired before the first validation date
    assert sp["dva"].max() + C.H + 1 < I - 5                           # validation labels expire before the P0 purge boundary (signal-5)
    assert sp["hi"] == I - 12 and sp["label_expiry_val_max"] == sp["hi"] + C.H + 1 < I
    assert sp["vs"] - sp["te"] == C.GAP_DAYS
def test_NEGATIVE_split_without_gap_is_detected(world):
    X, P, m_tr = world; sp = C.split_train_val(P["Xq"], P["yw"], m_tr, X.cal, I, gap=0)
    assert not (sp["dtr"].max() + C.H + 1 < sp["dva"].min())          # gap=0 violates the label-expiry rule; the check above would fail
def test_same_day_samples_never_in_train_and_validation(world):
    X, P, m_tr = world; sp = C.split_train_val(P["Xq"], P["yw"], m_tr, X.cal, I)
    assert set(sp["dtr"]).isdisjoint(set(sp["dva"])); assert sp["dtr"].max() < sp["dva"].min()
def test_train_part_is_prefix_of_P0_training_rows(world):
    X, P, m_tr = world; sp = C.split_train_val(P["Xq"], P["yw"], m_tr, X.cal, I); Xp, yp = L.train_rows(P["Xq"], P["yw"], m_tr, X.cal, I)
    np.testing.assert_array_equal(Xp[:len(sp["Xtr"])], sp["Xtr"]); np.testing.assert_array_equal(yp[:len(sp["ytr"])], sp["ytr"])
    assert len(Xp) == len(sp["Xtr"]) + len(sp["Xva"]) + int(sum(m_tr[d].sum() for d in range(sp["te"] + 1, sp["vs"])))   # train+gap+val == P0 window exactly
def test_scaler_uses_training_statistics_only(world):
    X, P, m_tr = world; sp = C.split_train_val(P["Xq"], P["yw"], m_tr, X.cal, I); mu, sd = M.fit_scaler(sp["Xtr"])
    np.testing.assert_allclose(mu, sp["Xtr"].mean(0), atol=1e-6)
    Xq2 = P["Xq"].copy(); yw2 = P["yw"].copy(); rng = np.random.default_rng(0)
    Xq2[sp["te"] + 1:] = rng.uniform(-0.5, 0.5, Xq2[sp["te"] + 1:].shape) * 5; yw2[sp["te"] + 1:] = 1.0     # destroy everything after the training part (val, gap, test)
    sp2 = C.split_train_val(Xq2, yw2, m_tr, X.cal, I); mu2, sd2 = M.fit_scaler(sp2["Xtr"])
    np.testing.assert_array_equal(mu, mu2); np.testing.assert_array_equal(sd, sd2); assert float(sp["ytr"].std()) == float(sp2["ytr"].std())
def test_model_parameters_unaffected_by_test_period_data(world):
    X, P, m_tr = world; sp = C.split_train_val(P["Xq"], P["yw"], m_tr, X.cal, I); a = M.fit_mlp(sp["Xtr"], sp["ytr"], sp["Xva"], sp["yva"], 42, SMALL)
    Xq2 = P["Xq"].copy(); yw2 = P["yw"].copy(); Xq2[sp["hi"] + 1:] = 9.0; yw2[sp["hi"] + 1:] = -9.0                  # destroy every row after the validation end (test period)
    sp2 = C.split_train_val(Xq2, yw2, m_tr, X.cal, I); b = M.fit_mlp(sp2["Xtr"], sp2["ytr"], sp2["Xva"], sp2["yva"], 42, SMALL)
    for k in a["state"]: assert torch_equal(a["state"][k], b["state"][k])
    assert a["best_epoch"] == b["best_epoch"]
def torch_equal(x, y):
    import torch; return bool(torch.equal(x, y))
def test_inference_on_latest_day_uses_no_future_fields(world):
    X, P, m_tr = world; i = X.C.shape[0] - 40
    Y = tamper_future(X, i); P2 = pipeline(Y)       # everything dated after i is destroyed (prices, volumes, factors, traded status)
    np.testing.assert_allclose(P["Xq"][i], P2["Xq"][i], atol=1e-12); np.testing.assert_array_equal(P["elig"][i], P2["elig"][i])
    mt2 = P2["elig"] & np.isfinite(P2["yw"]); sp = C.split_train_val(P["Xq"], P["yw"], m_tr, X.cal, i); sq = C.split_train_val(P2["Xq"], P2["yw"], mt2, X.cal, i)
    np.testing.assert_array_equal(sp["Xtr"], sq["Xtr"]); np.testing.assert_array_equal(sp["ytr"], sq["ytr"]); np.testing.assert_array_equal(sp["yva"], sq["yva"])   # labels used for training already matured by i
    ma = M.fit_mlp(sp["Xtr"], sp["ytr"], sp["Xva"], sp["yva"], 7, SMALL); mb = M.fit_mlp(sq["Xtr"], sq["ytr"], sq["Xva"], sq["yva"], 7, SMALL)
    pool = np.where(P["elig"][i])[0]; assert len(pool) > 20
    np.testing.assert_array_equal(M.predict(ma, P["Xq"][i][pool]), M.predict(mb, P2["Xq"][i][pool]))
def test_NEGATIVE_label_leak_is_detected(world):
    """If a model were trained on rows whose label expires AFTER the signal date, tampering with the future would change the model: the harness must be able to see it."""
    X, P, m_tr = world; i = X.C.shape[0] - 40; Y = tamper_future(X, i); P2 = pipeline(Y)
    d = i - 2                                            # label of day i-2 spans closes i-1..i+4 -> depends on the future
    assert not np.allclose(np.nan_to_num(P["yw"][d]), np.nan_to_num(P2["yw"][d]))
def test_seed_reproducible(world):
    X, P, m_tr = world; sp = C.split_train_val(P["Xq"], P["yw"], m_tr, X.cal, I)
    a = M.fit_mlp(sp["Xtr"], sp["ytr"], sp["Xva"], sp["yva"], 42, SMALL); b = M.fit_mlp(sp["Xtr"], sp["ytr"], sp["Xva"], sp["yva"], 42, SMALL); c = M.fit_mlp(sp["Xtr"], sp["ytr"], sp["Xva"], sp["yva"], 43, SMALL)
    assert all(torch_equal(a["state"][k], b["state"][k]) for k in a["state"]); assert not all(torch_equal(a["state"][k], c["state"][k]) for k in a["state"])
    np.testing.assert_array_equal(M.predict(a, sp["Xva"][:100]), M.predict(b, sp["Xva"][:100]))
# ---------------- artifact checks (skip until the artifacts exist) ----------------
SC = P1 / "scores"
def test_latest_scores_csv_schema_and_leading_zeros():
    f = SC / "dl_mlp_scores_20261016.csv"
    if not f.exists(): pytest.skip("latest scores not produced yet")
    df = pd.read_csv(f, dtype={"code": str}); assert len(df) == 1498 and df.code.is_unique and df.code.str.len().eq(6).all() and df.code.str.isdigit().all()
    assert any(c.startswith("0") for c in df.code); assert {"code", "mlp_score_mean", "rank", "eligible_asof"} <= set(df.columns)
    assert not any(("realized" in c) or ("ret_5d" in c) or ("label" in c and c != "max_label_expiry") for c in df.columns)
    assert (df.asof_date == "2026-10-16").all() and (pd.to_datetime(df.max_label_expiry) < pd.Timestamp("2026-10-16")).all() and (pd.to_datetime(df.train_end) < pd.to_datetime(df.val_end)).all()
    assert df.loc[~df.eligible_asof.astype(bool), "mlp_score_mean"].isna().all() and df["rank"].dropna().min() == 1
def test_oos_parquet_models_were_trained_before_prediction():
    f = SC / "dl_mlp_oos_predictions.parquet"
    if not f.exists(): pytest.skip("oos parquet not produced yet")
    d = pd.read_parquet(f); assert d.code.map(type).eq(str).all() and d.code.str.len().eq(6).all()
    sd = pd.to_datetime(d.signal_date)
    assert (pd.to_datetime(d.trained_asof) <= sd).all() and (pd.to_datetime(d.train_end) < pd.to_datetime(d.val_start)).all() and (pd.to_datetime(d.val_end) < sd).all()
    assert (pd.to_datetime(d.max_label_expiry) < sd).all()
    assert d.loc[~d.eligible_asof, "mlp_score_mean"].isna().all() and d.loc[d.eligible_asof, "mlp_score_mean"].notna().all()
    assert d.loc[~d.label_matured, "realized_ret_5d"].isna().all()
    assert d.groupby("signal_date").code.nunique().eq(1498).all()