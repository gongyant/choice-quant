# -*- coding: utf-8 -*-
"""P2-A tests: no look-ahead, key uniqueness, reproducibility, group completeness, frozen files untouched."""
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd, pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import p2lib as Q
RES = Q.RES
@pytest.fixture(scope="module")
def panel(): return pd.read_parquet(RES / "p2a_panel.parquet")
# ---------- exec_forward on synthetic matrices ----------
def mk(T=40, N=4):
    ADJ = np.cumprod(1 + np.full((T, N), 0.01), axis=0) * (1 + np.arange(N))[None, :]
    traded = np.ones((T, N), bool); up = np.zeros((T, N), bool); dn = np.zeros((T, N), bool)
    return ADJ, traded, up, dn
def run(ADJ, traded, up, dn, i=10, **kw):
    return Q.exec_forward(ADJ, traded & ~up, traded & ~dn, [i], **kw)
def test_normal_entry_next_day_exit_plus5():
    ADJ, tr, up, dn = mk(); o = run(ADJ, tr, up, dn)
    assert (o["entry_idx"][0] == 11).all() and (o["exit_idx"][0] == 16).all()
    assert np.allclose(o["ret"][0], ADJ[16] / ADJ[11] - 1) and not o["forced"].any()
def test_limit_up_lock_delays_entry_not_replaced():
    ADJ, tr, up, dn = mk(); up[11, 1] = True; o = run(ADJ, tr, up, dn)
    assert o["entry_idx"][0, 1] == 12 and o["exit_idx"][0, 1] == 17 and o["entry_idx"][0, 0] == 11
def test_suspension_delays_entry_and_never_before_t_plus_1():
    ADJ, tr, up, dn = mk(); tr[11:13, 2] = False; o = run(ADJ, tr, up, dn)
    assert o["entry_idx"][0, 2] == 13 and (o["entry_idx"][0] >= 11).all()
def test_never_buyable_is_nan():
    ADJ, tr, up, dn = mk(); up[11:20, 3] = True; o = run(ADJ, tr, up, dn)
    assert o["entry_idx"][0, 3] == -1 and np.isnan(o["ret"][0, 3])
def test_limit_down_delays_exit_and_forced_flag():
    ADJ, tr, up, dn = mk(); dn[16, 0] = True; o = run(ADJ, tr, up, dn); assert o["exit_idx"][0, 0] == 17 and not o["forced"][0, 0]
    dn[16:40, 1] = True; o = run(ADJ, tr, up, dn); assert o["forced"][0, 1] and o["exit_idx"][0, 1] == 11 + 5 + 10
def test_future_prices_after_exit_do_not_change_return():
    ADJ, tr, up, dn = mk(); a = run(ADJ, tr, up, dn)["ret"].copy(); ADJ2 = ADJ.copy(); ADJ2[17:] *= 7; b = run(ADJ2, tr, up, dn)["ret"]
    assert np.allclose(a, b)
def test_tail_beyond_calendar_is_nan():
    ADJ, tr, up, dn = mk(T=20); o = Q.exec_forward(ADJ, tr & ~up, tr & ~dn, [15]); assert np.isnan(o["ret"]).all()
# ---------- as-of-t controls: truncation invariance (no look-ahead) ----------
def test_controls_use_only_data_up_to_t():
    rng = np.random.default_rng(1); T, N = 80, 6; AMT = rng.uniform(1e7, 1e8, (T, N)); TURN = rng.uniform(0.5, 5, (T, N)); r1 = rng.normal(0, .02, (T, N))
    full_t, full_a = Q.rolling_turn20(TURN), Q.amihud20(r1, AMT)
    for t in (30, 50, 70):
        assert np.allclose(Q.rolling_turn20(TURN[:t + 1])[t], full_t[t], equal_nan=True) and np.allclose(Q.amihud20(r1[:t + 1], AMT[:t + 1])[t], full_a[t], equal_nan=True)
        T2 = TURN.copy(); A2 = AMT.copy(); T2[t + 1:] *= 100; A2[t + 1:] *= 100
        assert np.allclose(Q.rolling_turn20(T2)[t], full_t[t], equal_nan=True) and np.allclose(Q.amihud20(r1, A2)[t], full_a[t], equal_nan=True)
def test_panel_controls_match_P0_features_asof_t(panel):
    """r5/r20/vol20 in the panel equal an independent recomputation from the p0lib matrices truncated at t (no later bar used)."""
    import p0lib as L, p1lib as P1
    X = L.build_data(); L.limit_flags(X); W = P1.world_from_X(X); cal = X.cal
    d = panel[panel.eligible].sample(2000, random_state=3); ci = cal.get_indexer(pd.to_datetime(d.signal_date)); cj = pd.Index(X.codes).get_indexer(d.code)
    for k in ["r5", "r20", "vol20"]: assert np.allclose(d[k].values, W.f[k].values[ci, cj], equal_nan=True)
    tr = X.traded[ci, cj]; Ft = W.ADJf[ci, cj] / W.ADJf[np.maximum(ci - 5, 0), cj] - 1      # uses rows <= t only
    assert np.allclose(np.where(tr, Ft, np.nan), d.r5.values, equal_nan=True)
def test_execution_only_defines_returns_not_pool(panel):
    e = panel[panel.eligible]
    assert (e.ret_new.isna()).sum() > 0 and (~e.buyable_t1).sum() > 0      # pool contains names that later turned out unbuyable -> pool not conditioned on the future
    meta = json.load(open(RES / "p2a_panel_meta.json", encoding="utf-8")); assert all(v == 0 for v in meta["elig_mismatch_vs_frozen"].values())
def test_entry_after_signal_and_hold_at_least_5(panel):
    f = panel[panel.entry_date.notna() & panel.exit_date.notna()]
    assert (pd.to_datetime(f.entry_date) > pd.to_datetime(f.signal_date)).all() and (f.hold_rows >= 5).all() and (f.entry_delay >= 0).all()
    assert (pd.to_datetime(f.exit_date) > pd.to_datetime(f.entry_date)).all()
# ---------- keys / codes ----------
def test_panel_key_unique_and_codes_6char(panel):
    assert not panel.duplicated(["signal_date", "code"]).any() and panel.code.map(lambda s: bool(re.fullmatch(r"\d{6}", s))).all() and panel.code.str.startswith("0").sum() > 0
def test_frozen_key_unique_and_scores_equal_panel(panel):
    hp = pd.read_parquet(Q.HP_ALL, columns=["signal_date", "code", "model", "score", "model_id"]); assert not hp.duplicated(["signal_date", "code", "model_id"]).any() and len(hp) == 1860516
    hp["signal_date"] = pd.to_datetime(hp.signal_date); m = hp[hp.model == "GRU"].merge(panel[["signal_date", "code", "score_GRU"]], on=["signal_date", "code"])
    assert len(m) == 310086 and np.allclose(m.score, m.score_GRU, equal_nan=True)
def test_frozen_files_untouched():
    man = json.load(open(Q.P1D / "forward_registry_20261016_manifest.json", encoding="utf-8")); assert len(man["sha256"]) == 22
    for rel, h in man["sha256"].items(): assert Q.sha256(Q.V2 / rel.replace("\\", "/")) == h, rel
# ---------- reproducibility ----------
def test_bootstrap_deterministic():
    x = np.random.default_rng(0).normal(0.05, 0.1, 207); assert Q.block_ci(x) == Q.block_ci(x)
    assert abs(Q.nw_t(x)[0] - __import__("p0lib").nw_t(x, 4)[0]) < 1e-12
def test_rankic_reproducible_from_panel(panel):
    e = panel[panel.eligible & panel.ret_new_rel.notna()]
    s = pd.Series({i: Q.spearman(x.score_HGB.values, x.ret_new_rel.values) for i, x in e.groupby("i")})
    T = pd.read_csv(RES / "rankic_old_vs_new.csv"); row = T[(T.definition == "new") & (T.model == "HGB")].iloc[0]
    assert abs(s.mean() - row["mean"]) < 1e-12 and abs(Q.summarize(s.values)["hac_t"] - row.hac_t) < 1e-9
    from scipy.stats import spearmanr; x = e[e.i == e.i.iloc[0]]; assert abs(spearmanr(x.score_HGB, x.ret_new_rel)[0] - Q.spearman(x.score_HGB.values, x.ret_new_rel.values)) < 1e-12
def test_oracle_signal_sanity_is_diagnostic_only(panel):
    """ORACLE DIAGNOSTIC ONLY (score = future return): must give IC=1; proves the IC pipeline, never used as a result."""
    x = panel[(panel.eligible) & panel.ret_new_rel.notna() & (panel.i == panel.i.iloc[0])]; assert abs(Q.spearman(x.ret_new_rel.values, x.ret_new_rel.values) - 1) < 1e-12
    assert not any("oracle" in c.lower() for c in pd.read_csv(RES / "rankic_old_vs_new.csv").model.unique())
def test_partial_spearman_removes_control():
    rng = np.random.default_rng(5); C = rng.normal(size=(600, 1)); a = C[:, 0] + .3 * rng.normal(size=600); b = C[:, 0] + .3 * rng.normal(size=600)
    assert Q.spearman(a, b) > 0.8 and abs(Q.partial_spearman(a, b, C)[0]) < 0.15
# ---------- group completeness ----------
def test_groups_partition_the_sample(panel):
    gc = pd.read_csv(RES / "group_completeness.csv"); assert (gc.n_sample == gc.n_in_groups + gc.n_missing).all()
    assert gc.set_index("group_var").loc[["g_liq", "g_vol", "g_amt_abs", "g_exch"], "n_missing"].sum() == 0
    assert set(panel.exchange.dropna().unique()) <= {"SH", "SZ", "BJ"}
    G = pd.read_csv(RES / "rankic_by_group.csv")
    for gv in ["流动性(amt20 日内三分位)", "波动率(vol20 日内三分位)"]:
        d = G[(G.group_var == gv) & (G.model == "HGB")]; assert len(d) == 3 and d.avg_n_per_date.sum() > 1100
    assert set(G.model.unique()) == set(Q.MODELS)
def test_all_six_models_in_main_tables():
    for f in ["rankic_old_vs_new.csv", "partial_rankic.csv", "decile_summary.csv", "subset_robustness.csv", "factor_attribution.csv", "baseline_paired_ic.csv"]:
        assert set(Q.MODELS) <= set(pd.read_csv(RES / f).model.unique()), f
def test_attribution_selfcheck_baseline_is_spanned():
    A = pd.read_csv(RES / "factor_attribution.csv"); r = A[(A.model == "3F(REV5+LOWVOL+ILLIQ)") & (A.spec == "5因子") & (A.dependent == "D10-D1多空价差")].iloc[0]
    assert r.r2 > 0.85 and abs(r.alpha_hac_t) < 2
# ---------- deliverables ----------
def test_figures_nonempty_and_cn_font():
    pngs = list(Q.FIG.glob("*.png")); assert len(pngs) >= 8 and all(p.stat().st_size > 20000 for p in pngs)
    import matplotlib.font_manager as fm; assert any("Microsoft YaHei" in f.name for f in fm.fontManager.ttflist) or Path(r"C:\Windows\Fonts\msyh.ttc").exists()
def test_report_utf8_no_bom_and_sections():
    b = (Q.REP / "p2a_alpha_anatomy_report.md").read_bytes(); assert not b.startswith(b"\xef\xbb\xbf"); t = b.decode("utf-8")
    for k in ["不是投资建议", "探索性", "旧口径", "新口径", "偏 RankIC", "三因子", "十分位", "局限", "P2-B"]: assert k in t, k
    assert "\ufffd" not in t and len(t) > 15000
def test_verify_files_clean():
    j = json.load(open(Q.P2 / "a_verify.json", encoding="utf-8")); assert j["inconsistencies"] == [] and (Q.P2 / "a_verify.md").stat().st_size > 1000