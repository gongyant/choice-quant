# -*- coding: utf-8 -*-
"""P3-4b SUPPLEMENTARY calibration diagnostics (added after the first results; NOT pre-registered; original definition kept unchanged).
Per composite decile: median / mean relative return, share beating the SAME-DAY POOL MEDIAN, std of relative return and tail shares (|rel| > x),
with moving-block bootstrap CIs over signal dates. Also saves per-(date, code) composite history for the candidate figures."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H)); import p3_calib as K
def main(date):
    D = K.load(date); rng = np.random.default_rng(7); rows = []
    for hz, rc, x, blk in [("5d", "realized_rel_ret_5d", K.X5, 4), ("20d", "rel_20d", K.X20, 8)]:
        d = D.dropna(subset=[rc]).copy(); d["beat_med"] = (d[rc] > d.groupby("signal_date")[rc].transform("median")).astype(float)
        dates = sorted(d.signal_date.unique()); nd = len(dates); g = d.groupby(["signal_date", "bin"])
        bm = g.beat_med.sum().unstack(fill_value=0).reindex(dates).values; mr = g[rc].sum().unstack(fill_value=0).reindex(dates).values; n = g.size().unstack(fill_value=0).reindex(dates).values
        bs_b, bs_m = [], []
        for _ in range(2000):
            st = rng.integers(0, nd - blk + 1, int(np.ceil(nd / blk))); idx = (st[:, None] + np.arange(blk)).ravel()[:nd]; nn = n[idx].sum(0)
            bs_b.append(bm[idx].sum(0) / nn); bs_m.append(mr[idx].sum(0) / nn - mr[idx].sum() / nn.sum())
        lb, hb = np.percentile(bs_b, [2.5, 97.5], axis=0); lm, hm = np.percentile(bs_m, [2.5, 97.5], axis=0)
        for b in range(1, 11):
            s = d[d.bin == b]
            rows.append(dict(horizon=hz, bin=b, median_rel=s[rc].median(), mean_rel=s[rc].mean(), mean_rel_minus_all=s[rc].mean() - d[rc].mean(), mean_rel_ci_lo=lm[b - 1], mean_rel_ci_hi=hm[b - 1],
                             share_beat_pool_median=s.beat_med.mean(), share_beat_median_ci_lo=lb[b - 1], share_beat_median_ci_hi=hb[b - 1], std_rel=s[rc].std(),
                             share_above_x=(s[rc] > x).mean(), share_below_minus_x=(s[rc] < -x).mean(), share_tails=(s[rc].abs() > x).mean(), threshold=x))
    S = pd.DataFrame(rows); S["note"] = "补充诊断(事后加,非预注册);原定义见 calibration_table.csv"; S.to_csv(H / "results" / "calibration_supplement.csv", index=False, encoding="utf-8-sig")
    D[["signal_date", "code", "composite", "bin", "realized_rel_ret_5d", "rel_20d"]].to_parquet(H / "results" / "hist_composite.parquet", index=False)
    print(S.round(4).to_string())
if __name__ == "__main__": main(sys.argv[1] if len(sys.argv) > 1 else "20261021")