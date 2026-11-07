# -*- coding: utf-8 -*-
"""P3-B0: rebuild the merged world (no scoring, no registry write) and save a per-(signal_date, code) feature panel for history checks:
r5, r20, vol20, amt20, ami20, ma20 slope (5d), close vs MA20, limit-up/down locked at t and counts in last 5 days, traded_t. Same definitions as p3_rank/p3_score."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; V2 = H.parent; P1 = V2 / "p1_dl"; sys.path[:0] = [str(V2 / "p0"), str(V2), str(P1 / "dl"), str(P1)]
import p0lib as L
MG = H / "data" / "merged"; L.D, L.RU, L.FA, L.LAST_TD = MG, MG / "raw_unadj_parquet", MG / "adj_factors", pd.Timestamp("2026-10-21")
import dl_common as C
W = C.build_world(); X = W.X; cal = W.cal; f = W.f; codes = list(X.codes)
ma = pd.DataFrame(W.ADJf).rolling(20, min_periods=15).mean().values
r1 = f["r1"].values; amt = X.AMT; ami = ((np.abs(pd.DataFrame(r1)) / pd.DataFrame(amt).where(pd.DataFrame(amt) > 0)) * 1e8).replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=15).mean().values
LU = pd.DataFrame(X.lim_up.astype(float)).rolling(5, min_periods=1).sum().values; LD = pd.DataFrame(X.lim_dn.astype(float)).rolling(5, min_periods=1).sum().values
h = pd.read_parquet(P1 / "historical_predictions_all_models.parquet", columns=["signal_date"]); sd = sorted(set(h.signal_date)) + ["2026-10-21"]
rows = []
for s in sd:
    i = cal.get_loc(pd.Timestamp(s))
    rows.append(pd.DataFrame(dict(signal_date=s, code=codes, r5=f["r5"].values[i], r20=f["r20"].values[i], vol20=f["vol20"].values[i], amt20=f["amt20"].values[i], ami20=ami[i],
        ma20_slope_5d=ma[i] / ma[i - 5] - 1, close_vs_ma20=W.ADJf[i] / ma[i] - 1, lim_up_t=X.lim_up[i], lim_dn_t=X.lim_dn[i], n_lim_up_5d=LU[i], n_lim_dn_5d=LD[i], traded_t=X.traded[i], elig=W.elig[i])))
P = pd.concat(rows); P.to_parquet(H / "data" / "feature_panel_hist.parquet", index=False); print(P.shape, P.signal_date.nunique())