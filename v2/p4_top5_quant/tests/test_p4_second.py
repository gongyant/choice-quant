# -*- coding: utf-8 -*-
"""P4 batch-2 tests: PIT fundamentals availability, rules registered before evaluation, selection constraints."""
import sys, json, hashlib, pickle, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, pytest
P4 = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(P4)); import p4lib as P
def test_rules_registered_before_eval():
    m = [json.loads(l) for l in open(P4 / "rules_manifest.jsonl", encoding="utf-8-sig")]; r = [x for x in m if x["file"] == "selection_rules_p4_v1.json"][0]
    assert hashlib.sha256((P4 / "selection_rules_p4_v1.json").read_bytes()).hexdigest() == r["sha256"]
    ev = P4 / "results" / "bt_summary_e2.csv"
    if ev.exists(): assert dt.datetime.fromisoformat(r["registered_at_local"]).timestamp() < ev.stat().st_mtime
def test_fundamentals_never_before_notice():
    F = np.load(P4 / "cache" / "fin_raw_mats.npz"); cal = pd.DatetimeIndex(np.load(P4 / "cache" / "flags.npz")["cal"])
    no = F["notice_ord"]; po = F["period_ord"]; day = np.array([d.toordinal() for d in cal])[:, None]
    m = np.isfinite(no); assert (no[m] < np.broadcast_to(day, no.shape)[m]).all()          # notice strictly before the day it is used
    assert (po[m] < no[m]).all()                                                            # period end before notice
def test_no_backfill_first_value():
    F = np.load(P4 / "cache" / "fin_raw_mats.npz"); po = F["period_ord"]
    d = np.diff(np.nan_to_num(po, nan=0), axis=0); assert (d[np.isfinite(po[1:]) & np.isfinite(po[:-1])] >= 0).all()   # reported period never moves backwards
def test_selection_industry_cap():
    for f in (P4 / "predictions_registry").glob("p4_pred_*_E2S*_h*.csv"):
        d = pd.read_csv(f, dtype={"code": str}); s = d[d.in_final5]
        assert len(s) <= 5 and s.industry.value_counts().max() <= 2 and not s.fails_vol_cap.any() and not s.fails_drawdown_cap.any()
