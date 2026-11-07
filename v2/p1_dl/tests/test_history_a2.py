"""P1-A2 tests: historical walk-forward predictions, trades, summaries, dashboard smoke."""
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd, pytest
H = Path(__file__).resolve().parents[1]; V2 = H.parent; sys.path.insert(0, str(H)); sys.path.insert(0, str(V2 / "p0"))
import p0lib as L
CAL = L.load_calendar(); POS = {str(d.date()): i for i, d in enumerate(CAL)}
@pytest.fixture(scope="module")
def hp(): return pd.read_parquet(H / "historical_predictions.parquet")
@pytest.fixture(scope="module")
def tr(): return pd.read_csv(H / "strategy_trades.csv", dtype={"code": str}, encoding="utf-8-sig")
def test_shape_keys_and_columns(hp):
    assert hp.signal_date.nunique() == 207 and hp.signal_date.min() == "2022-07-01" and hp.signal_date.max() == "2026-09-28"
    assert not hp.duplicated(["signal_date", "code", "model_id"]).any()          # merge key for dl_mlp / dl_gru later
    assert set(hp.model.unique()) == {"Ridge", "HGB", "LGBM"} and hp.model_id.nunique() == 156
    assert "pred_excess_return_5d" not in hp.columns and "p_outperform_5d" not in hp.columns
    assert hp.groupby(["model", "signal_date"]).size().nunique() == 1 and hp.groupby(["model", "signal_date"]).size().iloc[0] == 1498
def test_codes_leading_zeros_everywhere(hp, tr):
    for s in [hp.code, tr.code, pd.read_csv(H / "per_stock_summary.csv", dtype={"code": str}, encoding="utf-8-sig").code, pd.read_csv(H / "sector_summary.csv", encoding="utf-8-sig", dtype=str).iloc[:, 0]]:
        pass
    for s in [hp.code, tr.code]: assert (s.str.len() == 6).all() and s.str.startswith("0").sum() > 100
    ps = pd.read_csv(H / "per_stock_summary.csv", dtype={"code": str}, encoding="utf-8-sig"); assert len(ps) == 1500 and ps.code.is_unique and (ps.code.str.len() == 6).all() and "000027" in set(ps.code)
def test_training_ends_before_label_window_of_evaluation_date(hp):
    d = hp.drop_duplicates(["model_id", "signal_date"])
    i = d.signal_date.map(POS).values; te = d.train_end_date.map(POS).values; le = d.train_label_end_date.map(POS).values
    assert (te <= i - 12).all()                       # purge: last training signal at least 12 trading days before the evaluation signal
    assert (le == te + 6).all() and (le < i).all()    # training labels (t+6) expired strictly before the evaluation signal date
    assert (d.train_first_date < d.train_end_date).all()
    g = d.groupby("model_id").signal_date.agg(["min", "max", "nunique"]); assert (g["nunique"] <= 4).all()          # one model serves at most 4 consecutive periods
    first = d.groupby("model_id").signal_date.min(); te0 = d.groupby("model_id").train_end_date.first(); assert (te0.map(POS) == first.map(POS) - 12).all()   # trained on the block's first signal
def test_scores_only_for_eligible_and_masks(hp):
    ne = hp[~hp.eligible_asof]; assert ne.score.isna().all() and ne["rank"].isna().all() and ne.f_r5.isna().all() and ne.realized_rel_ret_5d.isna().all() and (~ne.in_top50).all()
    e = hp[hp.eligible_asof]; assert e.score.notna().all() and e.f_r5.notna().all()
    for (m, d), g in e.groupby(["model", "signal_date"]):
        if d in ("2022-07-01", "2024-03-01", "2026-09-28") or d[:7] == "2025-02":
            assert sorted(g["rank"].astype(int)) == list(range(1, len(g) + 1)) and g.in_top50.sum() == 50 and (g.n_pool == len(g)).all()
def test_eligible_rows_have_a_bar_on_signal_date_and_stale_flag(hp):
    e = hp[(hp.model == "HGB") & hp.eligible_asof].sample(400, random_state=1); raw = {}
    for r in e.itertuples():
        if r.code not in raw: raw[r.code] = set(pd.read_parquet(L.RU / f"{r.code}.parquet", columns=["date"]).date.dt.strftime("%Y-%m-%d"))
        assert r.signal_date in raw[r.code]                                  # eligible => traded bar exists at t (suspended days are never eligible)
    fl = hp[(hp.model == "HGB") & hp.label_uses_stale_price].head(60)
    for r in fl.itertuples():
        s = set(pd.read_parquet(L.RU / f"{r.code}.parquet", columns=["date"]).date.dt.strftime("%Y-%m-%d")); assert (r.order_date not in s) or (r.exit_date not in s)
def test_same_day_cross_stock_isolation_relative_return_uses_only_same_day_pool(hp):
    g = hp[(hp.model == "HGB") & hp.eligible_asof & hp.realized_ret_5d.notna()]
    for d in ["2022-07-01", "2023-05-29", "2025-01-17", "2026-09-28"]:
        x = g[g.signal_date == d]; assert abs((x.realized_ret_5d - x.realized_ret_5d.mean() - x.realized_rel_ret_5d).abs().max()) < 1e-12 and abs(x.realized_rel_ret_5d.mean()) < 1e-12
    a = hp[hp.eligible_asof].groupby(["signal_date", "code"]).score.nunique(); assert (a <= 3).all()        # one score per model per stock-day
    # score of a stock-day never depends on realized outcomes: realized columns of unmatured labels (none here) would be NULL; all 207 dates matured
    assert hp[hp.eligible_asof].groupby("signal_date").realized_rel_ret_5d.apply(lambda s: s.notna().mean()).min() > 0.9
def test_reconciliation_with_p0_period_returns_and_sets(hp):
    rc = pd.read_csv(H / "reports" / "reconciliation_p0.csv"); assert (rc.max_abs_diff < 1e-12).all() and len(rc) >= 30
    P0 = pd.read_csv(V2 / "p0" / "results" / "fixed_sina_adj" / "period_returns.csv"); sm = pd.read_csv(H / "historical_predictions_summary.csv", encoding="utf-8-sig")
    for m in ["Ridge", "HGB", "LGBM"]:
        s = sm[sm.model == m].sort_values("signal_date"); assert np.allclose(s.net_ret_c30.values, P0[f"{m}_ret_c30"].values, atol=1e-12, rtol=0) and np.allclose(s.RankIC.values, P0[f"{m}_RankIC"].values, atol=1e-12, rtol=0, equal_nan=True)
    sets = pickle.load(open(V2 / "p0" / "results" / "fixed_sina_adj" / "sets.pkl", "rb"))
    for m in ["HGB"]:
        t = hp[(hp.model == m) & hp.in_top50].groupby("signal_date").code.apply(set)
        for k, d in enumerate(sets["signal_dates"]): assert t[d] == set(sets["topk"][m][k])
def test_trades_consistent_with_p0_order_log(tr):
    o = pd.read_csv(V2 / "p0" / "results" / "fixed_sina_adj" / "orders_primary_cost30bps.csv", dtype={"code": str})
    for m in ["Ridge", "HGB", "LGBM"]:
        nf = int(((o.strategy == m) & (o.side == "BUY") & (o.status == "FILLED")).sum()); t = tr[tr.strategy == m]
        assert nf == int(t.trade_status.isin(["CLOSED", "OPEN_AT_END_OR_STUCK"]).sum()); assert int((t.trade_status == "BUY_REJECTED").sum()) == int(((o.strategy == m) & (o.side == "BUY") & (o.status == "REJECTED")).sum())
    c = tr[tr.trade_status == "CLOSED"]; net = (c.sell_value_div_adj - c.sell_fee) / (c.buy_value + c.buy_fee) - 1; assert np.allclose(net, c.net_ret) and (c.actual_sell_date >= c.actual_buy_date).all() and (c.actual_buy_date >= c.planned_buy_date).all()
    assert (c.buy_shares % 100 == 0).all() and np.allclose(c.buy_fee, c.buy_value * 0.0015)
def test_corporate_action_exdate_not_mistaken_for_limit_down():
    fac_codes = [p.stem for p in L.FA.glob("*.parquet")]; pick = []
    for c in fac_codes:
        f = pd.read_parquet(L.FA / f"{c}.parquet")
        if len(f) and ((f.date >= "2021-01-01") & (f.qfq_factor > 0)).sum() >= 2 and (L.RU / f"{c}.parquet").exists(): pick.append(c)
    X = L.build_data(codes=pick[:400]); L.limit_flags(X); C = X.C; Cf = pd.DataFrame(C).ffill().values; prev = np.vstack([np.full((1, C.shape[1]), np.nan), Cf[:-1]]); raw_ret = C / prev - 1
    exd = np.vstack([np.zeros((1, C.shape[1]), bool), np.diff(X.F, axis=0) != 0]) & X.has_row; big = exd & (raw_ret < -0.11)
    assert big.sum() >= 5, "need real ex-rights days with raw price drop > 11%"
    adj_ret = pd.DataFrame(X.ADJ_unadj_based).pct_change(fill_method=None).values; assert np.nanmax(np.abs(adj_ret[big])) < 0.31   # adjusted return across ex-date stays within board limits
    flagged = X.lim_dn & big
    assert flagged.sum() <= 0.02 * big.sum()                                               # raw -11%..-50% ex-rights drops are (almost) never read as limit-down
    for t, j in zip(*np.where(flagged)):                                                    # the rare flagged ones must be genuine ex-rights-reference limit-downs (or the documented 5% ST proxy: adjusted ret ~ -5%)
        rate = {"创业板": .2, "科创板": .2, "北交所": .3}.get(X.u.board.iloc[j], .1); assert adj_ret[t, j] <= -rate + 0.003 or abs(adj_ret[t, j] + 0.05) < 0.003
def test_dashboard_smoke():
    from smoke_dashboard import run_all
    res, _ = run_all(); assert all(r[1] for r in res), [r for r in res if not r[1]]
