# -*- coding: utf-8 -*-
"""P2-C lag-stacked LightGBM on the SAME information set as the Transformer: the identical normalised 60-day x (6 channels + mask) window, flattened (420 columns, oldest->newest);
same signals, same P1 split geometry / training-validation rows / 4-signal retrain blocks / pool; target = winsorised pool-demeaned 5d relative return / training std.
LightGBM hyper-parameters = the frozen P0 LGBM (num_leaves15, lr.05, min_child200, subsample.8, colsample.8, lambda1) + early stopping on the Transformer's validation segment (patience 20, max 400 trees)."""
import sys, os, json, time, argparse
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_tf as T, c_wf as W_
import lightgbm as lgb
Q1, L, C = T.Q1, T.L, T.C
CK = T.P2 / "c_checkpoints" / "LAGLGBM"
def windows(Z, d, j):
    ar = np.arange(-T.SEQ_LEN + 1, 1); return Z[(d[:, None] + ar[None, :]), j[:, None]].reshape(len(d), -1).astype(np.float32)      # [n, 60*7]
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[42]); ap.add_argument("--blocks", type=int, default=0); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--fixed", type=int, default=0, help="POST-HOC sensitivity: fixed number of trees, no early stopping"); a = ap.parse_args()
    global CK
    if a.fixed: CK = CK.parent / f"LAGLGBM_F{a.fixed}"
    CK.mkdir(parents=True, exist_ok=True); VN = CK.name
    W = C.build_world(); F, mask = Q1.build_seq_features(W); sig_all = W_.grid(W); Y = T.label_stack(W, [5]); nblk = (len(sig_all) + 3) // 4; t0 = time.time()
    for k in ("Xq", "f", "lab", "y"): setattr(W, k, None)
    for seed in a.seeds:
        for rr in range(a.blocks or nblk):
            bp = CK / f"s{seed}_r{rr:03d}.npz"
            if bp.exists(): continue
            ix = rr * 4; i = sig_all[ix]; sp = T.split(W, Y, [5], i, "p1"); mu, sd = Q1.fit_norm(F, sp["lo"], sp["te"]); Z = Q1.apply_norm(F, mask, mu, sd)
            ysc = float(np.std(sp["tr"][2][:, 0])); t1 = time.time()
            Xtr = windows(Z, sp["tr"][0], sp["tr"][1]); ytr = (sp["tr"][2][:, 0] / ysc).astype(np.float32); Xva = windows(Z, sp["va"][0], sp["va"][1]); yva = (sp["va"][2][:, 0] / ysc).astype(np.float32)
            m = lgb.LGBMRegressor(n_estimators=a.fixed or 400, num_leaves=15, learning_rate=0.05, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=a.threads)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[] if a.fixed else [lgb.early_stopping(20, verbose=False)]); del Xtr
            sigs = sig_all[ix:ix + 4]; sc = np.full((len(sigs), W.N, 1), np.nan, dtype=np.float32)
            for q, ii in enumerate(sigs):
                pool = np.where(W.elig[ii])[0]
                if len(pool) >= 100: sc[q, pool, 0] = m.predict(windows(Z, np.full(len(pool), ii), pool))
            meta = dict(variant=VN, seed=seed, retrain=rr, asof=str(W.cal[i].date()), n_train=int(len(ytr)), n_val=int(len(yva)), best_iteration=int(m.best_iteration_ or a.fixed or 0), seconds=time.time() - t1, heads=[5], expiry_train=sp["expiry_train"], expiry_val=sp["expiry_val"], vs=sp["vs"], hi=sp["hi"], te=sp["te"], lo=sp["lo"])
            np.savez(str(bp) + ".tmp.npz", scores=sc, sig=np.array(sigs), meta=json.dumps(meta)); os.replace(str(bp) + ".tmp.npz", bp)
            W_.wlog(f"{VN} s{seed}: block {rr+1}/{nblk} ({time.time()-t0:.0f}s, fit {time.time()-t1:.0f}s, iters {meta['best_iteration']}, n_train {len(ytr)})")
if __name__ == "__main__": main()