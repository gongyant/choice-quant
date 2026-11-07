# -*- coding: utf-8 -*-
"""P2-C tests: purge geometry & label-expiry margins (P1 and 20d), P1 recipe identity, rank loss, persistence penalty, lag-stacked LightGBM has no look-ahead,
metric correctness, resume logic, coverage / frozen reconciliation, protocol timing, outputs, frozen manifest."""
import sys, json, re, types, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, pytest, torch
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import p2lib as Q, c_tf as T, c_wf as W_, c_eval as E, c_lgbm as LG
RES = Q.RES
def fake_world(Tn=1200, N=5, seed=0):
    cal = pd.bdate_range("2018-01-01", periods=Tn); W = types.SimpleNamespace(cal=cal, m_tr=np.ones((Tn, N), bool)); W.m_tr[:T.SEQ_LEN - 1] = False
    Y = np.random.default_rng(seed).normal(size=(Tn, N, 4)); return W, Y
# ---------- geometry ----------
@pytest.mark.parametrize("geom,heads", [("p1", [5]), ("g20", [5]), ("g20", [1, 5, 10, 20])])
def test_split_expiry_margins(geom, heads):
    W, Y = fake_world()
    for i in (900, 1000, 1150):
        sp = T.split(W, Y[:, :, :len(heads)], heads, i, geom)
        for h in heads:
            assert sp["expiry_train"][h] < sp["vs"]                         # last training label matures before the first validation feature date
            assert sp["expiry_val"][h] <= i - 6                             # validation labels matured >= 6 rows before the signal date
        assert sp["hi"] < i and sp["te"] < sp["vs"] <= sp["hi"]
def test_g20_derivation_numbers():
    cal = pd.bdate_range("2018-01-01", periods=1200); i = 1000
    lo, te, vs, hi = T.geometry(cal, i, "p1"); assert (i - hi, vs - te) == (12, 10)
    lo, te, vs, hi = T.geometry(cal, i, "g20"); assert (i - hi, vs - te) == (27, 25)
    assert hi + 1 + 20 == i - 6 and te + 1 + 20 < vs                     # exactly P1's margins for the 20-day label; the 60-day lookback adds nothing (features are backward looking)
def test_p1_geometry_would_leak_20d():
    W, Y = fake_world()
    with pytest.raises(AssertionError): T.split(W, Y, [1, 5, 10, 20], 1000, "p1")
def test_saved_blocks_respect_margins():
    n = 0
    for p in sorted((T.P2 / "c_checkpoints").glob("*/s*_r*.npz")):
        z = np.load(p, allow_pickle=False); m = json.loads(str(z["meta"])); i = int(z["sig"][0])
        for h, e in m["expiry_train"].items(): assert e < m["vs"], p
        for h, e in m["expiry_val"].items(): assert e <= i - 6, p
        n += 1
    assert n > 100
def test_variant_geometry_declared():
    assert T.VARIANTS["T2"]["geom"] == T.VARIANTS["T3"]["geom"] == T.VARIANTS["T0G"]["geom"] == "g20"
    assert T.VARIANTS["T0"]["geom"] == T.VARIANTS["T1"]["geom"] == T.VARIANTS["T4"]["geom"] == "p1"
    for v in ("T0G", "T1", "T2", "T4"):
        d = {k for k in ("heads", "loss", "geom", "persist") if T.VARIANTS[v][k] != T.VARIANTS["T0"][k]}; assert len(d) == 1 or (v == "T2" and d == {"heads", "geom"})
    assert {k for k in ("heads", "loss", "geom", "persist") if T.VARIANTS["T2"][k] != T.VARIANTS["T0G"][k]} == {"heads"}
    assert {k for k in ("heads", "loss", "geom", "persist") if T.VARIANTS["T3"][k] != T.VARIANTS["T2"][k]} == {"loss"}
# ---------- recipe identity ----------
def test_mhnet_single_head_equals_p1_tfnet():
    cfg = dict(T.TF_CFG); torch.manual_seed(0); a = T.Q1.TFNet(7, cfg).eval(); torch.manual_seed(0); b = T.MHNet(7, cfg, 1).eval()
    for (k1, v1), (k2, v2) in zip(a.state_dict().items(), b.state_dict().items()): assert k1 == k2 and torch.equal(v1, v2)
    x = torch.randn(16, 60, 7); x[:, :, -1] = 0.0; x[:3, :10, -1] = 1.0
    with torch.no_grad(): assert torch.allclose(a(x).reshape(-1), b(x).reshape(-1), atol=1e-6)
def test_reproduction_of_p1_block0():
    r = json.load(open(RES / "c_repro_r000.json", encoding="utf-8")); s = json.dumps(r)
    vals = [float(v) for k, v in r.items() if k.startswith("max_abs")]; assert vals and max(vals) < 1e-4 and r["corr"] > 0.999999 and r["best_epoch_mine"] == r["best_epoch_p1"], s
# ---------- losses ----------
def test_rank_loss_orders():
    r = torch.linspace(0, 1, 50)[:, None]; good = T._rank_loss_day(r * 5, r); bad = T._rank_loss_day(-r * 5, r); assert float(good) < float(bad)
    s = torch.randn(50, 1); y = torch.randn(50, 1); r1 = torch.tensor(T._pct_rank(np.zeros(50, int), y.numpy())).float(); r2 = torch.tensor(T._pct_rank(np.zeros(50, int), (y ** 3 + 2).numpy())).float()
    assert torch.allclose(T._rank_loss_day(s, r1), T._rank_loss_day(s, r2))   # invariant to monotone label transforms
def test_pct_rank_within_day():
    d = np.array([0, 0, 0, 1, 1]); y = np.array([[3.], [1.], [2.], [10.], [-1.]]); R = T._pct_rank(d, y); assert np.allclose(R[:, 0], [1, 1 / 3, 2 / 3, 1, .5])
def test_persistence_penalty_not_detached():
    src = (HERE / "c_tf.py").read_text(encoding="utf-8"); line = [l for l in src.splitlines() if "pen =" in l][0]
    assert ".var()" in line and "detach" not in line.split("pen =")[1].split(";")[0]
    p = torch.randn(100); pl = p + 0.1 * torch.randn(100); f = lambda a, b: ((a - b) ** 2).mean() / (a.var() + 1e-8); assert abs(float(f(p, pl)) - float(f(10 * p, 10 * pl))) < 1e-4
# ---------- lag-stacked LightGBM: same information set, no look-ahead ----------
def test_lag_windows_no_lookahead():
    Tn, N = 200, 4; Z = np.zeros((Tn, N, 7), np.float32); Z[:, :, 0] = np.arange(Tn)[:, None]; d = np.array([100, 150, 199]); j = np.array([0, 1, 3])
    X = LG.windows(Z, d, j); assert X.shape == (3, 420); Xr = X.reshape(3, 60, 7)
    assert (Xr[:, :, 0].max(axis=1) == d).all() and (Xr[:, 0, 0] == d - 59).all()
    Z2 = Z.copy(); Z2[151:] = 999; assert np.array_equal(LG.windows(Z2, d[:2], j[:2]), X[:2])
def test_lag_blocks_use_tf_geometry():
    for p in sorted((T.P2 / "c_checkpoints" / "LAGLGBM").glob("s42_r*.npz"))[:52]:
        m = json.loads(str(np.load(p)["meta"])); q = T.P2 / "c_checkpoints" / "T0" / p.name.replace("s42", "s43")
        if q.exists(): m2 = json.loads(str(np.load(q)["meta"])); assert (m["lo"], m["te"], m["vs"], m["hi"], m["n_train"], m["n_val"]) == (m2["lo"], m2["te"], m2["vs"], m2["hi"], m2["n_train"], m2["n_val"])
# ---------- metrics ----------
def test_stability_and_overlap_identity():
    rng = np.random.default_rng(1); K, N = 6, 300; s = rng.normal(size=N); Sc = np.tile(s, (K, 1)); cand = np.ones((K, N), bool)
    st_, ov = E.stability_series(Sc, cand); assert np.allclose(st_[1:], 1) and np.allclose(ov[1:], 1)
    Sc2 = rng.normal(size=(K, N)); st2, ov2 = E.stability_series(Sc2, cand); assert abs(np.nanmean(st2)) < 0.15 and np.nanmean(ov2) < 0.5
def test_rankic_perfect():
    rng = np.random.default_rng(2); R_ = rng.normal(size=(3, 100)); cand = np.ones((3, 100), bool); assert np.allclose(E.rankic_series(R_, cand, R_), 1)
def test_ema_is_causal():
    rng = np.random.default_rng(3); Sc = rng.normal(size=(8, 50)); cand = np.ones((8, 50), bool); a = E.ema_scores(Sc, cand); Sc2 = Sc.copy(); Sc2[5:] = 0; b = E.ema_scores(Sc2, cand); assert np.allclose(a[:5], b[:5])
def test_paired_alpha_is_difference_of_alphas():
    rng = np.random.default_rng(4); F = rng.normal(size=(100, 5)); ya = rng.normal(size=100) * .01 + F @ np.ones(5) * .001; yb = rng.normal(size=100) * .01
    a, _, _ = Q.ols_hac(ya, F); b, _, _ = Q.ols_hac(yb, F); d, _, _ = Q.ols_hac(ya - yb, F); assert abs((a[0] - b[0]) - d[0]) < 1e-12
def test_tint():
    m, sd, lo, hi = E.tint([1.0, 2.0, 3.0]); assert m == 2 and abs(sd - 1) < 1e-12 and lo < 2 < hi
# ---------- resume ----------
def test_resume_job_done(tmp_path, monkeypatch):
    monkeypatch.setattr(W_, "CK", tmp_path); (tmp_path / "TX").mkdir()
    for r in range(25): (tmp_path / "TX" / f"s1_r{r:03d}.npz").write_bytes(b"x")
    assert not W_.job_done("TX", 1, 26); (tmp_path / "TX" / "s1_r025.npz").write_bytes(b"x"); assert W_.job_done("TX", 1, 26) and not W_.job_done("TX", 1, 0)
# ---------- outputs / protocol / reconciliation ----------
def test_protocol_declared_before_variant_results():
    pr = json.load(open(RES / "c_protocol.json", encoding="utf-8")); assert "compute_reduction" in pr
    ts = dt.datetime.strptime(pr["declared_at"].split(" (")[0], "%Y-%m-%d %H:%M").timestamp()   # local (Asia/Singapore) wall time, same clock as file mtimes
    first = min(p.stat().st_mtime for v in ("T1", "T2", "T3", "T0G", "T4") for p in (T.P2 / "c_checkpoints" / v).glob("*.npz"))
    assert ts < first
def test_coverage_and_frozen_reconciliation():
    cv = pd.read_csv(RES / "c_coverage.csv"); u = cv[cv.used]; assert len(u) >= 9 and (u.n_covered == u.n_needed).all() and (u.finite_share_of_pool > 0.99).all()
    rc = pd.read_csv(RES / "c_reconcile.csv"); d = rc["diff"].dropna(); assert len(d) == 4 and (d.abs() < 1e-9).all()
def test_runs_consistent_with_seed_variance():
    RN = pd.read_csv(RES / "c_runs.csv"); SV = pd.read_csv(RES / "c_seed_variance.csv"); r = RN[(RN.variant == "T0") & (RN.window == "all")]; s = SV[(SV.variant == "T0") & (SV.signals.astype(str) == "207")].iloc[0]
    assert abs(r.rankic.mean() - s.rankic_mean) < 1e-12 and s.n_seeds >= 3
def test_paired_same_signals():
    PR = pd.read_csv(RES / "c_paired.csv"); x = PR[PR.signals.astype(str) == "104"]; assert (x.n <= 104).all() and len(x) > 0
    assert {"T1", "T2", "T3", "T0G", "T4", "LAGLGBM"} <= set(PR.A)
def test_oracle_labelled():
    o = pd.read_csv(RES / "c_oracle_diag.csv"); assert o.label.str.contains("ORACLE").all() and o.label.str.contains("diagnostic").all()
def test_figures_nonempty():
    figs = sorted(Q.FIG.glob("c_fig*.png")); assert len(figs) >= 6 and all(f.stat().st_size > 5000 for f in figs)
def test_report_present():
    t = (Q.REP / "p2c_transformer_ablation_report.md").read_text(encoding="utf-8")
    for k in ("种子方差", "滞后堆叠", "purge", "探索性", "减配", "局限", "配对"): assert k in t, k
# ---------- frozen artefacts untouched ----------
def test_frozen_manifest_and_p1_untouched():
    man = json.load(open(Q.P1D / "forward_registry_20261016_manifest.json", encoding="utf-8"))
    for rel, h in man["sha256"].items(): assert Q.sha256(Q.V2 / rel.replace("\\", "/")) == h, rel
    t0 = dt.datetime(2026, 10, 19, 13, 23).timestamp()
    for p in list((Q.P1D / "dl").rglob("*")) + [Q.P1D / "scores" / "dl_tf_oos_predictions.parquet"]:
        if p.is_file() and "__pycache__" not in str(p): assert p.stat().st_mtime < t0, p