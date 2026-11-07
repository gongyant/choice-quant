# -*- coding: utf-8 -*-
import json, hashlib, sys, subprocess, os
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parents[1]; D = "20261021"; PY = sys.executable
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def man(): return [json.loads(l) for l in open(H / "rules_manifest.jsonl", encoding="utf-8")]
def test_v2_rules_registered_before_results():
    m = [x for x in man() if x["file"] == "selection_rules_v2.json"]; assert len(m) == 1 and m[0]["sha256"] == sha(H / "selection_rules_v2.json")
    J = json.load(open(H / "selection_rules_v2.json", encoding="utf-8")); assert J["written_at_local"] == m[0]["registered_at"]
    t = pd.Timestamp(m[0]["registered_at"]).timestamp()
    for f in [H / f"latest_research_ranking_{D}_v2.csv", H / "results" / f"top5_v2_{D}.csv", H / "results" / "hist_top5_summary.csv", H / "reports" / f"daily_ai_report_{D}_v2.md", H / "reports" / "history_vs_actual_report.md"]:
        assert f.stat().st_mtime > t, f
def test_v1_untouched():
    assert sha(H / "selection_rules.json").upper().startswith("0B9CDEF5")
    t = pd.Timestamp(man()[0]["registered_at"]).timestamp()
    for f in [H / f"latest_research_ranking_{D}.csv", H / "results" / f"top5_{D}.csv", H / "results" / f"avoid5_{D}.csv", H / "reports" / f"daily_ai_report_{D}.md", H / "predictions_registry" / f"pred_{D}.csv"]:
        assert f.stat().st_mtime < t, f
    m = [json.loads(l) for l in open(H / "predictions_registry" / "manifest.jsonl", encoding="utf-8")]; assert [x for x in m if x["file"] == f"pred_{D}.csv"][0]["sha256"] == sha(H / "predictions_registry" / f"pred_{D}.csv")
def test_v2_selection_follows_rules():
    o = pd.read_csv(H / f"latest_research_ranking_{D}_v2.csv", dtype={"code": str}, encoding="utf-8-sig"); assert len(o) == 1498 and o.code.str.fullmatch(r"\d{6}").all()
    T = o[o.top5_v2]; assert 1 <= len(T) <= 5 and T.passes_top_filters.all()
    assert not (T.chg_20d > 0.20).any() and not ((T.chg_20d < -0.15) & (T.ma20_slope_5d < 0)).any() and not (T.n_lim_up_5d >= 2).any()
    if len(T) == 5: assert np.isclose(T.sel.min(), o[o.passes_v2].sel.nlargest(5).min())
    else: assert o.passes_v2.sum() == len(T)
    W = o[o.watch_v2]; assert W.T1_down.all() and W.passes_top_filters.all() and not W.top5_v2.any()
    assert (o.avoid5_v2 == o.avoid5).all()
    assert (o[o.top5_v2].code.isin(pd.read_csv(H / "results" / f"top5_v2_{D}.csv", dtype={"code": str}).code)).all()
def test_history_tables():
    PR = pd.read_csv(H / "results" / "hist_top5_by_period.csv", encoding="utf-8-sig"); assert PR.groupby("strategy").size().eq(207).all()
    assert PR.signal_date.max() <= "2026-09-28" and PR.signal_date.min() >= "2022-07-01"
    assert np.allclose(PR.ret5_net_30bp[PR.n_picks > 0], PR.ret5[PR.n_picks > 0] - 0.003) and (PR.ret5_net_30bp[PR.n_picks == 0] == 0).all() and (PR.n_picks <= 5).all()
    PK = pd.read_csv(H / "results" / "hist_top5_picks.csv", dtype={"code": str}, encoding="utf-8-sig"); assert PK.code.str.fullmatch(r"\d{6}").all()
    v2 = PK[PK.strategy == "v2"]; assert not (v2.r20 > 0.20).any()
    Y = pd.read_csv(H / "results" / "hist_bins_by_year.csv", encoding="utf-8-sig"); assert set(Y.bin) == set(range(1, 11))
def test_reports():
    h = (H / "reports" / "history_vs_actual_report.md").read_text(encoding="utf-8"); assert "不是盲测" in h and ("扣0.30%" in h or "扣 0.30%" in h)
    m = (H / "reports" / "model_reliability_summary.md").read_text(encoding="utf-8"); assert "不能单独依赖" in m and "P2" in m
    v = (H / "reports" / f"daily_ai_report_{D}_v2.md").read_text(encoding="utf-8"); assert v.splitlines()[0].startswith("# 每日 AI 研究报告(v2") and "一句话回答" in v and "v1 与 v2 对比" in v and "下跌观察名单" in v
    assert (H / "reports" / f"daily_ai_report_{D}.md").exists()
    for f in ["hist_cumulative.png", "hist_bins_by_year.png", "hist_examples.png"]: assert (H / "figures" / f).stat().st_size > 20000
def test_forward_verify_append_only():
    log = H / "verification" / "verify_log.jsonl"; n0 = sum(1 for _ in open(log, encoding="utf-8")) if log.exists() else 0
    s = H / "verification" / "forward_summary.csv"; before = sha(s) if s.exists() else None
    r = subprocess.run([PY, str(H / "p3_verify_forward.py")], capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"}); assert r.returncode == 0, r.stderr[-500:]
    assert sum(1 for _ in open(log, encoding="utf-8")) == n0 + 1
    if before is not None: assert sha(s) == before or s.stat().st_size > 0
def test_verify_price_matches_world():
    sys.path.insert(0, str(H)); import p3_verify_forward as V
    z = np.load(H / "data" / f"world_{D}.npz"); cal = pd.DatetimeIndex(z["cal_full"]); codes = ["000001", "600000", "300750"]; codes = [c for c in codes if c in list(z["codes"])]
    A = V.adj_close(codes, cal); W = z["ADJf_full"][:, [list(z["codes"]).index(c) for c in codes]]
    i, k = cal.get_loc(pd.Timestamp("2025-03-03")), cal.get_loc(pd.Timestamp("2025-03-10")); assert np.allclose(A[k] / A[i], W[k] / W[i], rtol=1e-4)
def test_dashboard_smoke():
    r = subprocess.run([PY, str(H / "smoke_dashboard_p3.py")], capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"}); assert r.returncode == 0, r.stdout[-800:]