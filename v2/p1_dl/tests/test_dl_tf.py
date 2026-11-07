"""P1-B3 tests: small Transformer encoder - attention mask hides suspended days, no future leakage, train-only scaling, reproducibility, artifacts."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pytest, torch
HERE = Path(__file__).resolve().parent; P1 = HERE.parent; V2 = P1.parent
sys.path.insert(0, str(P1 / "dl")); sys.path.insert(0, str(V2 / "p0" / "leakage_unit_tests")); sys.path.insert(0, str(HERE))
import dl_common as C, dl_seq as Q
import test_dl_gru as G
from synth import make_synth, tamper_future
I = 300
TINY = dict(arch="tf", d_model=16, nhead=2, ff=32, layers=2, dropout=0.2, max_epochs=2, patience=2, subsample=4, threads=2, batch=1024)
@pytest.fixture(scope="module")
def ctx():
    X = make_synth(T=420, N=80, seed=3, suspensions=True); W = G.world_of(X); F, mask = Q.build_seq_features(W); return X, W, F, mask
def test_architecture_matches_preregistration():
    net = Q.make_net(7, {**Q.CFG, **TINY, "d_model": 64, "nhead": 4, "ff": 128, "layers": 2}); assert isinstance(net, Q.TFNet)
    assert len(net.enc.layers) == 2 and net.inp.out_features == 64 and net.enc.layers[0].self_attn.num_heads == 4 and net.enc.layers[0].linear1.out_features == 128 and net.pe.shape == (60, 64)
def test_attention_mask_hides_suspended_days(ctx):
    X, W, F, mask = ctx; sp = Q.split_idx(W.m_tr, W.yw, W.cal, I); mu, sd = Q.fit_norm(F, sp["lo"], sp["te"]); S = torch.tensor(Q.apply_norm(F, mask, mu, sd))
    torch.manual_seed(0); net = Q.make_net(7, {**Q.CFG, **TINY}).eval(); ar = torch.arange(-59, 1)
    cand = [(d, j) for d in range(150, 400) for j in range(80) if (not mask[d, j]) and mask[d - 59:d, j].any()]; assert len(cand) > 10
    d = torch.tensor([c[0] for c in cand[:32]]); j = torch.tensor([c[1] for c in cand[:32]]); x = Q._windows(S, d, j, ar).clone(); pad = x[:, :, -1] > 0.5; assert pad.any()
    with torch.no_grad():
        y0 = net(x); x2 = x.clone(); noise = torch.randn_like(x2[:, :, :-1]) * 50; x2[:, :, :-1] = torch.where(pad[:, :, None], noise, x2[:, :, :-1])    # garbage in masked cells only
        y1 = net(x2); assert torch.allclose(y0, y1, atol=1e-5)
        x3 = x.clone(); k = int((~pad[0]).nonzero()[0]); x3[0, k, 0] += 5.0; assert not torch.allclose(net(x3)[0], y0[0], atol=1e-6)             # changing a VALID day does change the output
def test_NEGATIVE_gru_is_not_mask_invariant_so_the_test_can_fail(ctx):
    X, W, F, mask = ctx; S = torch.tensor(Q.apply_norm(F, mask, *Q.fit_norm(F, 0, 250))); net = Q.make_net(7, {**Q.CFG, "arch": "gru", "hidden": 8}).eval(); ar = torch.arange(-59, 1)
    cand = [(d, j) for d in range(150, 400) for j in range(80) if (not mask[d, j]) and mask[d - 6:d, j].any()][:16]; assert len(cand) >= 4   # masked days shortly before t (a GRU forgets old ones)
    x = Q._windows(S, torch.tensor([c[0] for c in cand]), torch.tensor([c[1] for c in cand]), ar).clone(); pad = x[:, :, -1] > 0.5; x2 = x.clone(); x2[:, :, :-1] = torch.where(pad[:, :, None], torch.randn_like(x2[:, :, :-1]) * 50, x2[:, :, :-1])
    with torch.no_grad(): assert not torch.allclose(net(x), net(x2), atol=1e-5)
def test_scores_independent_of_anything_after_t(ctx):
    X, W, F, mask = ctx; t = 350; sp = Q.split_idx(W.m_tr, W.yw, W.cal, t); mu, sd = Q.fit_norm(F, sp["lo"], sp["te"]); S = torch.tensor(Q.apply_norm(F, mask, mu, sd)); M = Q.fit_seq(S, sp["tr"], sp["va"], 5, TINY)
    W2 = G.world_of(tamper_future(X, t)); F2, m2 = Q.build_seq_features(W2); sq = Q.split_idx(W2.m_tr, W2.yw, W2.cal, t); np.testing.assert_allclose(sp["tr"][2], sq["tr"][2], atol=1e-12)
    mu2, sd2 = Q.fit_norm(F2, sq["lo"], sq["te"]); S2 = torch.tensor(Q.apply_norm(F2, m2, mu2, sd2)); M2 = Q.fit_seq(S2, sq["tr"], sq["va"], 5, TINY); pool = np.where(W.elig[t])[0]; assert len(pool) > 20
    np.testing.assert_allclose(Q.predict_seq(M, S, t, pool), Q.predict_seq(M2, S2, t, pool), rtol=1e-5, atol=1e-6)
def test_scaler_and_parameters_use_training_data_only(ctx):
    X, W, F, mask = ctx; sp = Q.split_idx(W.m_tr, W.yw, W.cal, I); mu, sd = Q.fit_norm(F, sp["lo"], sp["te"]); S1 = torch.tensor(Q.apply_norm(F, mask, mu, sd)); a = Q.fit_seq(S1, sp["tr"], sp["va"], 42, TINY)
    F2 = F.copy(); F2[sp["hi"] + 1:] = 7.0; m2 = mask.copy(); m2[sp["hi"] + 1:] = False; yw2 = W.yw.copy(); yw2[sp["hi"] + 1:] = -9.0
    sp2 = Q.split_idx(W.m_tr, yw2, W.cal, I); mu2, sd2 = Q.fit_norm(F2, sp2["lo"], sp2["te"]); np.testing.assert_array_equal(mu, mu2); np.testing.assert_array_equal(sd, sd2)
    S2 = torch.tensor(Q.apply_norm(F2, m2, mu2, sd2)); b = Q.fit_seq(S2, sp2["tr"], sp2["va"], 42, TINY); assert all(torch.equal(a["state"][k], b["state"][k]) for k in a["state"])
def test_same_day_isolation_and_label_expiry(ctx):
    X, W, F, mask = ctx; sp = Q.split_idx(W.m_tr, W.yw, W.cal, I); assert set(sp["tr"][0]).isdisjoint(set(sp["va"][0])) and sp["tr"][0].max() + C.H + 1 < sp["va"][0].min() and sp["label_expiry_val_max"] < I - 5
def test_seed_reproducible_cpu(ctx):
    X, W, F, mask = ctx; sp = Q.split_idx(W.m_tr, W.yw, W.cal, I); S = torch.tensor(Q.apply_norm(F, mask, *Q.fit_norm(F, sp["lo"], sp["te"])))
    a, b, c = [Q.fit_seq(S, sp["tr"], sp["va"], s, TINY, "cpu") for s in (42, 42, 43)]
    assert all(torch.equal(a["state"][k], b["state"][k]) for k in a["state"]) and not all(torch.equal(a["state"][k], c["state"][k]) for k in a["state"])
SC = P1 / "scores"
def test_latest_tf_scores_schema_and_leading_zeros():
    f = SC / "dl_tf_scores_20261016.csv"
    if not f.exists(): pytest.skip("not produced yet")
    df = pd.read_csv(f, dtype={"code": str}); assert len(df) == 1498 and df.code.is_unique and df.code.str.len().eq(6).all() and any(c.startswith("0") for c in df.code)
    assert {"code", "tf_score_mean", "tf_s42", "rank", "eligible_asof"} <= set(df.columns) and not any("realized" in c or "ret_5d" in c for c in df.columns)
    assert (df.asof_date == "2026-10-16").all() and (pd.to_datetime(df.max_label_expiry) < pd.Timestamp("2026-10-16")).all() and df.loc[~df.eligible_asof.astype(bool), "tf_score_mean"].isna().all()
def test_tf_oos_parquet_models_trained_before_prediction():
    f = SC / "dl_tf_oos_predictions.parquet"
    if not f.exists(): pytest.skip("not produced yet")
    d = pd.read_parquet(f); sd_ = pd.to_datetime(d.signal_date); assert d.code.map(type).eq(str).all() and d.code.str.len().eq(6).all()
    assert (pd.to_datetime(d.trained_asof) <= sd_).all() and (pd.to_datetime(d.train_end) < pd.to_datetime(d.val_start)).all() and (pd.to_datetime(d.val_end) < sd_).all() and (pd.to_datetime(d.max_label_expiry) < sd_).all()
    assert d.loc[~d.eligible_asof, "tf_score_mean"].isna().all() and d.loc[d.eligible_asof, "tf_score_mean"].notna().all() and d.groupby("signal_date").code.nunique().eq(1498).all()