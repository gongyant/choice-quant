# -*- coding: utf-8 -*-
import json, hashlib, os, stat
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parents[1]; D = "20261021"; REG = H / "predictions_registry" / f"pred_{D}.csv"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rk(): return pd.read_csv(H / f"latest_research_ranking_{D}.csv", dtype={"code": str}, encoding="utf-8-sig")
def test_rows_and_leading_zeros():
    R = rk(); assert len(R) == 1498 and R.code.is_unique and R.code.str.fullmatch(r"\d{6}").all()
    assert R.code.str.startswith("00").sum() > 0
    G = pd.read_csv(REG, dtype={"code": str}); assert len(G) == 1498 * 11 and G.code.str.fullmatch(r"\d{6}").all() and G.groupby("model").size().eq(1498).all()
def test_registry_immutable():
    m = [json.loads(l) for l in open(H / "predictions_registry" / "manifest.jsonl", encoding="utf-8")]; e = [x for x in m if x["file"] == REG.name]
    assert len(e) == 1 and e[0]["sha256"] == sha(REG) and e[0]["rows"] == 16478
    assert not os.stat(REG).st_mode & stat.S_IWRITE
def test_no_future_data():
    z = np.load(H / "data" / f"world_{D}.npz"); assert pd.Timestamp(str(z["cal"][-1])) == pd.Timestamp("2026-10-21")
    mi = json.load(open(H / "data" / "merge_info.json", encoding="utf-8")); assert max(mi["new_trade_dates"]) <= "2026-10-21"
    S = json.load(open(H / "results" / "calibration_summary.json", encoding="utf-8")); assert S["last"] <= "2026-09-28"
    G = pd.read_csv(REG, dtype={"code": str}); assert (G.signal_date.astype(str).str.replace("-", "") == D).all()
def test_rules_precede_results():
    r = H / "selection_rules.json"; J = json.load(open(r, encoding="utf-8"))
    for f in [REG, H / f"latest_research_ranking_{D}.csv", H / "results" / f"top5_{D}.csv", H / "reports" / f"daily_ai_report_{D}.md"]:
        assert r.stat().st_mtime < f.stat().st_mtime, f
    G = pd.read_csv(REG, nrows=1); assert J["written_at_local"] < G.generated_at[0]
def test_top5_avoid5_follow_rules():
    R = rk(); T = R[R.top5]; A = R[R.avoid5]; assert len(T) == 5 and len(A) == 5
    assert (T.amt20 >= 1e8).all() and (~T.st_or_delist_risk).all() and (T.n_models_ge70 >= 4).all() and (T.model_disagreement <= 0.15).all()
    assert (T.chg_5d <= 0.15).all() and (T.chg_20d <= 0.30).all() and (~((T.trend == "下跌") & (T.chg_5d < -0.08))).all()
    best = R[R.passes_top_filters].sel_score.nlargest(5).min(); assert np.isclose(T.sel_score.min(), best)
    bear = (A.composite_pctile <= 0.20) & (A[[f"pctile_{m}" for m in ["HGB","LGBM","MLP","GRU","TF"]]].le(0.30).sum(1) >= 4); assert (bear | A.st_or_delist_risk | (A.n_limit_down_5d >= 2)).all() and (~A.name.str.contains("四通股份")).all()
def test_figures_and_report():
    R = rk()
    for c in R[R.top5 | R.avoid5].code: assert (H / "figures" / f"cand_{c}_{D}.png").stat().st_size > 20000
    t = (H / "reports" / f"daily_ai_report_{D}.md").read_text(encoding="utf-8"); assert "不是买入推荐" in t and "未取得" in t and "预测可信度低" in t
def test_p0_untouched():
    p0 = H.parent / "p0" / "data"
    if p0.exists():
        assert all(f.stat().st_mtime < pd.Timestamp("2026-10-19").timestamp() for f in [f for f in p0.rglob("*.csv")][:3000])