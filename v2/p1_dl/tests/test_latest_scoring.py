"""P1-A1 tests: latest-day scoring has no future dependence, codes keep leading zeros, master table integrity."""
import sys, copy, json
from pathlib import Path
import numpy as np, pandas as pd, pytest
H = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(H))
import p1lib as P1
from p1lib import L
CSV = H / "latest_all_stocks_20261016.csv"
@pytest.fixture(scope="module")
def df(): return pd.read_csv(CSV, dtype=str, encoding="utf-8-sig", keep_default_na=False)
def test_n_rows_and_unique_codes(df):
    S = pd.read_csv(H.parent / "data" / "sample_1500.csv", dtype=str)
    assert len(df) == 1500 and df.code.is_unique and set(df.code) == set(S.code)
def test_codes_are_6_char_strings_with_leading_zeros(df):
    assert (df.code.str.len() == 6).all() and df.code.str.fullmatch(r"\d{6}").all()
    assert (df.code.str.startswith("0")).sum() > 100 and {"000027", "002829", "000006"} <= set(df.code)   # leading zeros survive the CSV round trip
def test_dates_and_required_columns(df):
    assert (df.asof_date == "2026-10-16").all() and df.last_trade_date.replace("", "2026-10-16").le("2026-10-16").all()
    need = "asof_date code name exchange board industry last_trade_date close_unadjusted r5 r20 r60 vol20 amount20 turnover20 quality_flags eligible_asof ineligible_reason hgb_score lgbm_score ridge_score dl_mlp_score dl_gru_score rank_hgb rank_lgbm rank_ridge rank_dl_gru model_version predicted_horizon".split()
    assert set(need) <= set(df.columns)
    assert "pred_excess_return_5d" not in df.columns and "p_outperform_5d" not in df.columns      # uncalibrated: no return/probability columns
def test_eligibility_ranks_and_reasons(df):
    e = df.eligible_asof == "True"; n = int(e.sum()); assert n > 1000
    assert (df.loc[~e, "ineligible_reason"] != "").all() and (df.loc[e, "ineligible_reason"] == "").all()
    for r in ["rank_hgb", "rank_lgbm", "rank_ridge"]:
        assert sorted(df.loc[e, r].astype(int)) == list(range(1, n + 1)); assert (df.loc[~e, r] == "").all()
    for s in ["hgb_score", "lgbm_score", "ridge_score"]: assert (df.loc[e, s] != "").all() and (df.loc[~e, s] == "").all()
    assert {"600485", "600677"} <= set(df.code[~e]) and (df.loc[df.code.isin(["600485", "600677"]), "ineligible_reason"].str.startswith("无行情数据")).all()
def test_training_labels_expire_before_signal_date():
    m = json.load(open(H / "models" / "model_meta_2026-10-16.json", encoding="utf-8"))
    assert m["latest_label_end_date_used"] < m["signal_date"] and m["train_last_date"] < m["latest_label_end_date_used"] and m["seed"] == 42
    cal = L.load_calendar(); assert cal[cal.get_loc(pd.Timestamp(m["train_last_date"])) + 6] == pd.Timestamp(m["latest_label_end_date_used"])
@pytest.fixture(scope="module")
def world():
    codes = [c for c in L.load_universe().code if (L.RU / f"{c}.parquet").exists()][::10]
    X = L.build_data(codes=codes); return X
def _scores(X, i):
    W = P1.world_from_X(X); fitted, meta = P1.train_frozen(W, i); pool, sc = P1.score_asof(W, i, fitted)
    return pool, sc, meta
def _tamper(X, lo, hi=None, seed=1):
    X2 = copy.deepcopy(X); rng = np.random.default_rng(seed)
    for k in ["O", "Hh", "L", "C", "V", "AMT", "TURN", "F"]:
        M = getattr(X2, k); sl = slice(lo, hi); M[sl] = M[sl] * rng.uniform(0.5, 2.0, size=M[sl].shape)
    return L.finalize(X2)
def test_scores_unchanged_when_data_after_signal_date_is_tampered(world):
    X = world; i = len(X.cal) - 1 - 15
    pool0, sc0, meta = _scores(X, i); assert len(pool0) > 30
    pool1, sc1, _ = _scores(_tamper(X, i + 1), i)
    assert np.array_equal(pool0, pool1)
    for n in P1.MODELS: assert np.allclose(sc0[n], sc1[n], rtol=0, atol=0), n
    assert meta["latest_label_end_date_used"] < meta["signal_date"]
def test_training_rows_never_include_dates_after_i_minus_12(world):
    """Purge: every training date d satisfies d <= i-12, so its label end d+6 <= i-6 < i (labels are matured before the signal date)."""
    X = world; i = len(X.cal) - 1 - 15; W = P1.world_from_X(X); m = W.elig & np.isfinite(W.yw); Xt, yt = L.train_rows(W.Xq, W.yw, m, W.cal, i)
    lo = int(W.cal.searchsorted(W.cal[i] - pd.Timedelta(days=365 * 3))); assert len(Xt) == sum(int(m[d].sum()) for d in range(lo, i - 12 + 1))
def test_negative_control_tampering_history_changes_scores(world):
    X = world; i = len(X.cal) - 1 - 15; pool0, sc0, _ = _scores(X, i)
    X2 = _tamper(X, i - 40, i + 1, seed=3); pool1, sc1, _ = _scores(X2, i)
    assert (not np.array_equal(pool0, pool1)) or any(not np.allclose(sc0[n], sc1[n]) for n in P1.MODELS)
def test_last_day_features_use_only_dates_up_to_asof(world):
    X = world; T = len(X.cal); W = P1.world_from_X(X); i = T - 1
    X2 = _tamper(X, T - 1, None, seed=9); W2 = P1.world_from_X(X2)
    for k in L.FEATS: assert np.allclose(W.f[k].values[T - 2], W2.f[k].values[T - 2], equal_nan=True), k      # the day before the tampered day is unaffected (features are trailing)
