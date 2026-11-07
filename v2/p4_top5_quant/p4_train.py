# -*- coding: utf-8 -*-
"""P4-E1 walk-forward: python p4_train.py <E1reg|E1lrank|E1xendcg> <5|20>
Signal dates = the 207 P1/P3 OOS signal dates. Retrain every 4 signal dates on a rolling 3-year window of days whose label is FULLY matured
(d <= i-h-2; every 3rd day). Hyper-parameters: 4-point grid chosen ONCE on an inner validation split inside the first training window
(last 120 trading days, separated by an h+2 gap), then frozen for all later retrains. Also fits a final model at the last date (2026-10-21)."""
import sys, json, time, pickle, itertools
import numpy as np, pandas as pd, lightgbm as lgb
import p4lib as P
MODEL, h = sys.argv[1], int(sys.argv[2]); t0 = time.time(); C = P.P4 / "cache"; OUT = P.P4 / "preds"; OUT.mkdir(exist_ok=True)
FEAT = MODEL[:2]; Xq = np.load(C / f"Xq_{FEAT}.npy", mmap_mode="r"); fl = np.load(C / "flags.npz"); lab = pickle.load(open(C / "labels.pkl", "rb"))["lab"][h]
cal = pd.DatetimeIndex(fl["cal"]); codes = fl["codes"]; elig = fl["elig"]; T = len(cal)
sig = sorted(pd.read_parquet(P.V2 / "p1_dl" / "historical_predictions_all_models.parquet", columns=["signal_date"]).signal_date.unique())
sig_idx = [int(cal.get_loc(pd.Timestamp(s))) for s in sig]; assert len(sig_idx) == 207
OBJ = {"reg": "regression", "lrank": "lambdarank", "xendcg": "rank_xendcg"}[MODEL[2:]]
BASE = dict(objective=OBJ, learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, num_threads=24, verbose=-1, seed=42, deterministic=True, force_row_wise=True)
if OBJ == "lambdarank": BASE.update(lambdarank_truncation_level=30)
GRID = [dict(num_leaves=a, min_data_in_leaf=b) for a, b in itertools.product([15, 63], [200, 1000])]; NROUND = 300
def rows(days):
    Xs, ys, gs = [], [], []
    for d in days:
        m = elig[d] & np.isfinite(lab["exw"][d]) & np.isfinite(lab["gr"][d])
        if m.sum() < 50: continue
        Xs.append(np.asarray(Xq[d][m])); ys.append(lab["exw"][d][m] if OBJ == "regression" else lab["gr"][d][m]); gs.append(int(m.sum()))
    return np.concatenate(Xs), np.concatenate(ys), gs
def fit(days, prm):
    Xt, yt, g = rows(days); ds = lgb.Dataset(Xt, yt, group=None if OBJ == "regression" else g, free_raw_data=True)
    return lgb.train({**BASE, **prm}, ds, NROUND), len(yt)
def top_ex(bst, days, k=10):
    v = []
    for d in days:
        m = np.where(elig[d] & np.isfinite(lab["ex"][d]))[0]
        if len(m) < 50: continue
        s = bst.predict(np.asarray(Xq[d][m])); v.append(lab["ex"][d][m][np.argsort(-s)[:k]].mean())
    return float(np.mean(v))
# ---- one-time inner validation at the first signal date ----
i0 = sig_idx[0]; days0 = P.train_days(i0, h, cal); hi = days0[-1]
val = [d for d in days0 if d > hi - 120]; inner = [d for d in days0 if d <= hi - 120 - h - 2]
assert max(inner) + 1 + h < min(val)
sel = []
for prm in GRID:
    b, n = fit(inner, prm); sel.append(dict(**prm, n_inner=n, val_top10_excess=top_ex(b, val)))
sel = pd.DataFrame(sel); best = GRID[int(sel.val_top10_excess.values.argmax())]
print(MODEL, h, "grid", sel.to_dict("records"), "best", best, round(time.time() - t0), "s", flush=True)
# ---- walk-forward ----
parts, meta = [], []; bst = None
for k, i in enumerate(sig_idx):
    if k % 4 == 0:
        days = P.train_days(i, h, cal); assert P.label_end_index(days[-1], h) < i
        bst, n = fit(days, best); meta.append(dict(retrain_signal_date=str(cal[i].date()), train_first=str(cal[days[0]].date()), train_last=str(cal[days[-1]].date()),
                                                   label_end_max=str(cal[P.label_end_index(days[-1], h)].date()), n_rows=n, n_days=len(days)))
        if k % 20 == 0: print("retrain", k, cal[i].date(), n, round(time.time() - t0), "s", flush=True)
    m = np.where(elig[i])[0]; s = bst.predict(np.asarray(Xq[i][m]))
    parts.append(pd.DataFrame(dict(signal_date=str(cal[i].date()), code=codes[m], score=s, model=MODEL, horizon=h, retrain_signal_date=meta[-1]["retrain_signal_date"], label_end_max=meta[-1]["label_end_max"])))
pd.concat(parts).to_parquet(OUT / f"{MODEL}_h{h}.parquet", index=False)
# ---- final model at the last complete trading day ----
iL = T - 1; days = P.train_days(iL, h, cal); bF, n = fit(days, best); m = np.where(elig[iL])[0]
pd.DataFrame(dict(signal_date=str(cal[iL].date()), code=codes[m], score=bF.predict(np.asarray(Xq[iL][m])), model=MODEL, horizon=h,
                  train_first=str(cal[days[0]].date()), train_last=str(cal[days[-1]].date()), label_end_max=str(cal[P.label_end_index(days[-1], h)].date()))).to_parquet(OUT / f"{MODEL}_h{h}_latest.parquet", index=False)
bF.save_model(str(OUT / f"{MODEL}_h{h}_latest.txt"))
imp = pd.Series(bF.feature_importance("gain"), index=pickle.load(open(C / "featnames.pkl", "rb"))[FEAT]).sort_values(ascending=False)
json.dump(dict(model=MODEL, horizon=h, grid=sel.to_dict("records"), best=best, nround=NROUND, base=BASE, retrains=meta, final_train=dict(first=str(cal[days[0]].date()), last=str(cal[days[-1]].date()), n=n),
               top_gain=imp.head(15).round(1).to_dict(), secs=round(time.time() - t0)), open(OUT / f"{MODEL}_h{h}_meta.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("done", MODEL, h, round(time.time() - t0), "s", flush=True)
