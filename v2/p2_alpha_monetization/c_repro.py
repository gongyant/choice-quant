# -*- coding: utf-8 -*-
"""Reproduction check: the P2-C trainer in the T0 configuration must reproduce P1's seed-42 retrain r000 (checkpoint predictions and the frozen OOS score parquet)."""
import sys, time, json
import numpy as np, pandas as pd, torch
sys.path.insert(0, "."); import c_tf as T, c_wf as W_
Q1, L, C = T.Q1, T.L, T.C
t0 = time.time(); W = C.build_world(); F, mask = Q1.build_seq_features(W); print("world", round(time.time() - t0)); sig = W_.grid(W); Y = T.label_stack(W, [5]); i = sig[0]
sp = T.split(W, Y, [5], i, "p1"); mu, sd = Q1.fit_norm(F, sp["lo"], sp["te"]); S = torch.tensor(Q1.apply_norm(F, mask, mu, sd), device="cuda")
P1ck = torch.load(T.V2 / "p1_dl" / "dl" / "checkpoints" / "tf" / "tf_s42_r000.pt", weights_only=False); print("P1 ckpt best_epoch", P1ck["best_epoch"], "n_train", P1ck["n_train"], "mine n_train", len(sp["tr"][0]))
t1 = time.time(); M = T.fit2(S, sp["tr"], sp["va"], 42, T.VARIANTS["T0"], device="cuda"); print("train s", round(time.time() - t1), "best_epoch", M["best_epoch"], "best_val", M["best_val"], "P1 best_val", P1ck["best_val"])
pool = np.where(W.elig[i])[0]; mine = T.predict2(M, S, i, pool)[:, 0]; ref = Q1.predict_seq(P1ck, S, i, pool, "cuda")
pq = pd.read_parquet(T.V2 / "p1_dl" / "scores" / "dl_tf_oos_predictions.parquet"); print(pq.columns.tolist()[:20])
d0 = str(W.cal[i].date()); sub = pq[(pq.signal_date == d0) & pq.eligible_asof]; col = [c for c in pq.columns if c in ("tf_s42",)][0]
codes = np.array(W.X.codes); frozen = sub.set_index("code")[col].reindex(codes[pool]).values
print("max|mine-P1ckpt|", np.abs(mine - ref).max(), " corr", np.corrcoef(mine, ref)[0, 1], " max|mine-frozen parquet|", np.nanmax(np.abs(mine - frozen)), " max|ckpt-frozen|", np.nanmax(np.abs(ref - frozen)), "scale", np.std(ref))
json.dump(dict(max_abs_mine_vs_p1ckpt=float(np.abs(mine - ref).max()), corr=float(np.corrcoef(mine, ref)[0, 1]), best_epoch_mine=M["best_epoch"], best_epoch_p1=P1ck["best_epoch"], best_val_mine=M["best_val"], best_val_p1=P1ck["best_val"], max_abs_mine_vs_frozen=float(np.nanmax(np.abs(mine - frozen))), n_train_mine=int(len(sp["tr"][0])), n_train_p1=int(P1ck["n_train"])), open("results/c_repro_r000.json", "w"), indent=1)