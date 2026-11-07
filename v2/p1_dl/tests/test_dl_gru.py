"""P1-B2 tests: 60-day sequence model (GRU) - no future data in sequences/scores, mask semantics, train-only scaling, isolation, reproducibility, artifacts."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pytest, torch
HERE = Path(__file__).resolve().parent; P1 = HERE.parent; V2 = P1.parent
sys.path.insert(0, str(P1 / "dl")); sys.path.insert(0, str(V2 / "p0" / "leakage_unit_tests"))
import dl_common as C, dl_seq as Q
from synth import make_synth, tamper_future
L = C.L; I = 300
TINY = dict(hidden=8, max_epochs=2, patience=2, subsample=4, threads=2, batch=1024)
def world_of(X):
    L.finalize(X); L.limit_flags(X); return C.world_from(X)
@pytest.fixture(scope="module")
def ctx():
    X = make_synth(T=420, N=80, seed=3, suspensions=True); W = world_of(X); F, mask = Q.build_seq_features(W); return X, W, F, mask
def eq(a, b): np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-6, atol=1e-6, equal_nan=True)
def test_sequence_features_use_no_future_data(ctx):
    X, W, F, mask = ctx; t = 350; Y = tamper_future(X, t, how="garbage"); W2 = world_of(Y); F2, m2 = Q.build_seq_features(W2)
    eq(F[:t + 1], F2[:t + 1]); assert (mask[:t + 1] == m2[:t + 1]).all()           # every cell dated <= t identical although everything after t was destroyed
    Y = tamper_future(X, t, how="nan"); W3 = world_of(Y); F3, m3 = Q.build_seq_features(W3); eq(F[:t + 1], F3[:t + 1])
def test_NEGATIVE_window_ending_after_t_is_detected(ctx):
    X, W, F, mask = ctx; t = 350; W2 = world_of(tamper_future(X, t)); F2, _ = Q.build_seq_features(W2)
    assert not np.allclose(np.nan_to_num(F[t + 1]), np.nan_to_num(F2[t + 1]))      # a window that included day t+1 would differ -> the equality test above is able to see leakage
def test_mask_is_one_on_suspended_days_and_time_axis_not_compressed(ctx):
    X, W, F, mask = ctx; susp = np.argwhere(~X.traded & (np.arange(X.C.shape[0])[:, None] > 70)); assert len(susp) > 5
    mu, sd = Q.fit_norm(F, 0, 250); S = Q.apply_norm(F, mask, mu, sd)
    for d, j in susp[:20]: assert S[d, j, -1] == 1.0 and np.all(S[d, j, :-1] == 0.0)
    assert S.shape[:2] == X.C.shape                                                 # one row per master-calendar day: nothing removed
    d, j = susp[0]; dd = min(d + 3, X.C.shape[0] - 1); ar = torch.arange(-Q.SEQ_LEN + 1, 1)
    win = Q._windows(torch.tensor(S), torch.tensor([dd]), torch.tensor([j]), ar)[0].numpy(); np.testing.assert_array_equal(win, S[dd - Q.SEQ_LEN + 1:dd + 1, j])
    assert win[Q.SEQ_LEN - 1 - (dd - d), -1] == 1.0                                 # the suspended day sits at its true calendar offset inside the window
def test_scaler_uses_training_segment_only(ctx):
    X, W, F, mask = ctx; sp = Q.split_idx(W.m_tr, W.yw, W.cal, I); mu, sd = Q.fit_norm(F, sp["lo"], sp["te"])
    F2 = F.copy(); F2[sp["te"] + 1:] = np.random.default_rng(0).normal(5, 9, F2[sp["te"] + 1:].shape); mu2, sd2 = Q.fit_norm(F2, sp["lo"], sp["te"])
    np.testing.assert_array_equal(mu, mu2); np.testing.assert_array_equal(sd, sd2)
    a = max(0, sp["lo"] - Q.SEQ_LEN + 1); seg = F[a:sp["te"] + 1].reshape(-1, F.shape[2]); np.testing.assert_allclose(mu, np.nanmean(seg, 0), rtol=1e-5, atol=1e-7)
def test_split_idx_matches_B1_geometry_and_isolation(ctx):
    X, W, F, mask = ctx; sp = Q.split_idx(W.m_tr, W.yw, W.cal, I); b1 = C.split_train_val(W.Xq, W.yw, W.m_tr, W.cal, I)
    np.testing.assert_array_equal(sp["tr"][0], b1["dtr"]); np.testing.assert_array_equal(sp["tr"][2], b1["ytr"]); np.testing.assert_array_equal(sp["va"][0], b1["dva"]); np.testing.assert_array_equal(sp["va"][2], b1["yva"])
    assert set(sp["tr"][0]).isdisjoint(set(sp["va"][0])) and sp["tr"][0].max() + C.H + 1 < sp["va"][0].min() and sp["label_expiry_val_max"] < I - 5
    assert W.m_tr[sp["tr"][0], sp["tr"][1]].all() and W.elig[sp["va"][0], sp["va"][1]].all()    # only eligible_asof_t rows
def test_model_parameters_unaffected_by_test_period_data(ctx):
    X, W, F, mask = ctx; sp = Q.split_idx(W.m_tr, W.yw, W.cal, I); mu, sd = Q.fit_norm(F, sp["lo"], sp["te"]); S1 = torch.tensor(Q.apply_norm(F, mask, mu, sd))
    a = Q.fit_seq(S1, sp["tr"], sp["va"], 42, TINY)
    F2 = F.copy(); F2[sp["hi"] + 1:] = 7.0; m2 = mask.copy(); m2[sp["hi"] + 1:] = False; yw2 = W.yw.copy(); yw2[sp["hi"] + 1:] = -9.0
    sp2 = Q.split_idx(W.m_tr, yw2, W.cal, I); mu2, sd2 = Q.fit_norm(F2, sp2["lo"], sp2["te"]); S2 = torch.tensor(Q.apply_norm(F2, m2, mu2, sd2)); b = Q.fit_seq(S2, sp2["tr"], sp2["va"], 42, TINY)
    assert all(torch.equal(a["state"][k], b["state"][k]) for k in a["state"]) and a["best_epoch"] == b["best_epoch"]
def test_latest_day_score_independent_of_anything_after_t(ctx):
    X, W, F, mask = ctx; t = 350; sp = Q.split_idx(W.m_tr, W.yw, W.cal, t); mu, sd = Q.fit_norm(F, sp["lo"], sp["te"]); S = torch.tensor(Q.apply_norm(F, mask, mu, sd)); M = Q.fit_seq(S, sp["tr"], sp["va"], 5, TINY)
    Y = tamper_future(X, t); W2 = world_of(Y); F2, m2 = Q.build_seq_features(W2); sq = Q.split_idx(W2.m_tr, W2.yw, W2.cal, t)
    for k in range(3): np.testing.assert_allclose(sp["tr"][2], sq["tr"][2], atol=1e-12)                      # labels used for training had already matured by t
    mu2, sd2 = Q.fit_norm(F2, sq["lo"], sq["te"]); S2 = torch.tensor(Q.apply_norm(F2, m2, mu2, sd2)); M2 = Q.fit_seq(S2, sq["tr"], sq["va"], 5, TINY)
    pool = np.where(W.elig[t])[0]; assert len(pool) > 20 and (W.elig[t] == W2.elig[t]).all()
    np.testing.assert_allclose(Q.predict_seq(M, S, t, pool), Q.predict_seq(M2, S2, t, pool), rtol=1e-5, atol=1e-6)
def test_seed_reproducible_cpu(ctx):
    X, W, F, mask = ctx; sp = Q.split_idx(W.m_tr, W.yw, W.cal, I); mu, sd = Q.fit_norm(F, sp["lo"], sp["te"]); S = torch.tensor(Q.apply_norm(F, mask, mu, sd))
    a, b, c = [Q.fit_seq(S, sp["tr"], sp["va"], s, TINY, "cpu") for s in (42, 42, 43)]
    assert all(torch.equal(a["state"][k], b["state"][k]) for k in a["state"]) and not all(torch.equal(a["state"][k], c["state"][k]) for k in a["state"])
SC = P1 / "scores"
def test_latest_gru_scores_schema_and_leading_zeros():
    f = SC / "dl_gru_scores_20261016.csv"
    if not f.exists(): pytest.skip("not produced yet")
    df = pd.read_csv(f, dtype={"code": str}); assert len(df) == 1498 and df.code.is_unique and df.code.str.len().eq(6).all() and any(c.startswith("0") for c in df.code)
    assert {"code", "gru_score_mean", "rank", "eligible_asof"} <= set(df.columns) and not any("realized" in c or "ret_5d" in c for c in df.columns)
    assert (df.asof_date == "2026-10-16").all() and (pd.to_datetime(df.max_label_expiry) < pd.Timestamp("2026-10-16")).all() and df.loc[~df.eligible_asof.astype(bool), "gru_score_mean"].isna().all()
def test_gru_oos_parquet_models_trained_before_prediction():
    f = SC / "dl_gru_oos_predictions.parquet"
    if not f.exists(): pytest.skip("not produced yet")
    d = pd.read_parquet(f); sd_ = pd.to_datetime(d.signal_date); assert d.code.map(type).eq(str).all() and d.code.str.len().eq(6).all()
    assert (pd.to_datetime(d.trained_asof) <= sd_).all() and (pd.to_datetime(d.train_end) < pd.to_datetime(d.val_start)).all() and (pd.to_datetime(d.val_end) < sd_).all() and (pd.to_datetime(d.max_label_expiry) < sd_).all()
    assert d.loc[~d.eligible_asof, "gru_score_mean"].isna().all() and d.loc[d.eligible_asof, "gru_score_mean"].notna().all() and d.groupby("signal_date").code.nunique().eq(1498).all()