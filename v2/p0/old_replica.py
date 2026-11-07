"""E0 helper: re-run the ORIGINAL (look-ahead) baseline.py logic unchanged, same data/params, but
 (1) write to p0/results/old_replica (never touches results/baseline), (2) save the per-period candidate pool and Top-K sets
 so that Jaccard overlap with the fixed pipeline can be computed. Every number produced here is INVALID_BEFORE_FIX (P0-A..F)."""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
V2 = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(V2))
import baseline as B
OUT = V2 / "p0" / "results" / "old_replica"; OUT.mkdir(parents=True, exist_ok=True)
H, FEATS = B.H, B.FEATS
def main(topk=50, fee=0.003, seed=42, retrain=4, first_test="2022-07-01"):
    panel = pd.read_parquet(V2 / "data" / "panel_1500.parquet")
    p = B.build(panel); dates = pd.DatetimeIndex(sorted(p.date.unique()))
    p[[f"q_{x}" for x in FEATS]] = B.rank_feats(p); qf = [f"q_{x}" for x in FEATS]
    tr_pool = p[p.eligible_t & p.fut_ret.notna()].copy()
    tr_pool["cs_mean"] = tr_pool.groupby("date").fut_ret.transform("mean"); tr_pool["y"] = tr_pool.fut_ret - tr_pool.cs_mean
    lo, hi = tr_pool.groupby("date").y.transform(lambda s: s.quantile(.01)), tr_pool.groupby("date").y.transform(lambda s: s.quantile(.99))
    tr_pool["y_w"] = tr_pool.y.clip(lo, hi)
    first = dates.searchsorted(pd.Timestamp(first_test)); sig_dates = dates[first:len(dates) - H - 1:H]
    models = B.make_models(seed); names = list(models)
    recs, holdings = [], {n: None for n in names + ["Momentum20", "Reversal5"]}; turn = {n: [] for n in holdings}; fitted = {}
    saved = []
    for ix, dt_ in enumerate(sig_dates):
        cutoff = dates[max(0, dates.get_loc(dt_) - H)]
        if ix % retrain == 0:
            tr = tr_pool[(tr_pool.end_date < cutoff) & (tr_pool.date >= dt_ - pd.Timedelta(days=3 * 365))]
            if len(tr) < 20000: continue
            for n in names:
                m = models[n](); m.fit(tr[qf].fillna(0), tr.y_w); fitted[n] = m
        if not fitted: continue
        elig_all = p[(p.date == dt_) & p.eligible_t]
        day = p[(p.date == dt_) & p.eligible_t & p.tradable].copy()
        if len(day) < max(100, topk * 2): continue
        day["cs_mean"] = day.fut_ret.mean()
        row = {"signal_date": str(dt_.date()), "n_eligible": int(len(day)), "universe_ew_gross": float(day.fut_ret.mean())}
        x = day[qf].fillna(0); scores = {n: fitted[n].predict(x) for n in names}
        scores["Momentum20"] = day.r20.values; scores["Reversal5"] = -day.r5.values
        rec = {"signal_date": str(dt_.date()), "old_candidate_pool": sorted(day.code), "old_eligible_t_pool": sorted(elig_all.code), "topk": {}}
        for n, sc in scores.items():
            day["_s"] = sc; top = day.nlargest(topk, "_s"); codes = set(top.code); prev = holdings[n]
            turn[n].append(1.0 if prev is None else 1 - len(codes & prev) / len(codes)); holdings[n] = codes
            row[f"{n}_gross"] = float(top.fut_ret.mean()); row[f"{n}_net"] = float(top.fut_ret.mean()) - fee * turn[n][-1]; row[f"{n}_turnover"] = turn[n][-1]
            y = day.fut_ret - day.cs_mean
            row[f"{n}_IC"] = float(np.corrcoef(sc, y)[0, 1]) if np.std(sc) > 0 else np.nan
            row[f"{n}_RankIC"] = float(spearmanr(sc, y).statistic) if np.std(sc) > 0 else np.nan
            rec["topk"][n] = sorted(codes)
        row["EqualWeight_net"] = row["universe_ew_gross"]; recs.append(row); saved.append(rec)
        if ix % 20 == 0: print("signal", ix, dt_.date(), flush=True)
    bt = pd.DataFrame(recs); bt.to_csv(OUT / "walk_forward_returns_replica.csv", index=False, encoding="utf-8-sig")
    pickle.dump(saved, open(OUT / "old_sets.pkl", "wb"))
    old = pd.read_csv(V2 / "results" / "baseline" / "walk_forward_returns.csv")
    num = [c for c in old.columns if c != "signal_date"]
    same_dates = bool((old.signal_date.values == bt.signal_date.values).all()) if len(old) == len(bt) else False
    diff = {c: float(np.nanmax(np.abs(old[c].values - bt[c].values))) for c in num} if same_dates else "date mismatch"
    json.dump({"rows_old": len(old), "rows_replica": len(bt), "same_dates": same_dates, "max_abs_diff_per_column": diff}, open(OUT / "replica_check.json", "w"), indent=1)
    print("replica check:", json.dumps({"rows": [len(old), len(bt)], "same_dates": same_dates, "maxdiff": (max(diff.values()) if same_dates else None)}))
if __name__ == "__main__": main()
