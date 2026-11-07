# -*- coding: utf-8 -*-
"""P4 batch-3 tests: StockMixer walk-forward label maturity and window causality."""
import json
from pathlib import Path
import numpy as np, pandas as pd, pytest
P4 = Path(__file__).resolve().parents[1]
def test_sm_meta_matured():
    fs = list((P4 / "preds").glob("SM*_meta.json"))
    if not fs: pytest.skip("no SM runs")
    for f in fs:
        m = json.load(open(f))
        for r in m["retrains"]: assert r["label_end_max"] < r["retrain_signal_date"]
        assert m["final_train"]["last"] < "2026-10-21" and m["W"] in (10, 20)
def test_sm_preds_label_end_before_signal():
    for f in (P4 / "preds").glob("SM*_h*.parquet"):
        d = pd.read_parquet(f, columns=["signal_date", "label_end_max"]); assert (d.label_end_max < d.signal_date).all(), f
def test_window_is_causal():
    src = (P4 / "p4_sm_train.py").read_text(encoding="utf-8")
    assert "Xg[d - W + 1:d + 1]" in src and "i - h - 2 + 1" in src     # window ends at d; training days end at i-h-2
