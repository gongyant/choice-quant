# -*- coding: utf-8 -*-
"""P2-C walk-forward driver. Same signal grid / 4-signal retrain blocks / pool rule as P1 (run_seq_wf.py). Resumable at retrain-block granularity; jobs are claimed with lock files so several worker processes can share one queue."""
import sys, os, json, time, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_tf as T
Q1, L, C = T.Q1, T.L, T.C
CK = T.P2 / "c_checkpoints"; NBLK = 52
# (variant, seed, n_blocks) ; n_blocks=0 -> all 52 retrain blocks. Compute reduction (declared before results): ablation variants use the first 26 retrain blocks (signals 0-103) only.
PRIORITY = [("T0", 43, 0), ("T0", 44, 0)] + [(v, s, 26) for s in (42, 43, 44) for v in ("T1", "T2", "T3", "T0G", "T4")]
def job_done(variant, seed, mb):
    return all((CK / variant / f"s{seed}_r{r:03d}.npz").exists() for r in range(mb or NBLK))
def wlog(msg, f="c_run.log"):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [pid {os.getpid()}] {msg}"; print(line, flush=True)
    with open(T.P2 / "results" / f, "a", encoding="utf-8") as fh: fh.write(line + "\n")
def grid(W, first_test="2022-07-01"):
    first = int(W.cal.searchsorted(pd.Timestamp(first_test))); return list(range(first, W.T - L.H - 1, L.H))
def block_path(variant, seed, rr): return CK / variant / f"s{seed}_r{rr:03d}.npz"
def run_job(W, F, mask, variant, seed, device, retrain=4, max_blocks=0):
    # max_blocks: only blocks < max_blocks are trained (0 = all)
    spec = T.VARIANTS[variant]; heads = spec["heads"]; (CK / variant).mkdir(parents=True, exist_ok=True); sig_all = grid(W); Y = T.label_stack(W, heads); nblk = (len(sig_all) + retrain - 1) // retrain; t0 = time.time(); tt = 0.0; ndone = 0
    for rr in range(nblk):
        if max_blocks and rr >= max_blocks: break
        bp = block_path(variant, seed, rr)
        if bp.exists(): continue
        ix = rr * retrain; i = sig_all[ix]; sp = T.split(W, Y, heads, i, spec["geom"])
        if len(sp["tr"][0]) + len(sp["va"][0]) < 20000: continue
        mu, sd = Q1.fit_norm(F, sp["lo"], sp["te"]); S = torch.tensor(Q1.apply_norm(F, mask, mu, sd), device=device); t1 = time.time()
        M = T.fit2(S, sp["tr"], sp["va"], seed, spec, device=device); tt += time.time() - t1
        sigs = sig_all[ix:ix + retrain]; sc = np.full((len(sigs), W.N, len(heads)), np.nan, dtype=np.float32)
        for k, ii in enumerate(sigs):
            pool = np.where(W.elig[ii])[0]
            if len(pool) >= 100: sc[k, pool] = T.predict2(M, S, ii, pool, device)
        meta = dict(variant=variant, seed=seed, retrain=rr, asof=str(W.cal[i].date()), lo=sp["lo"], te=sp["te"], vs=sp["vs"], hi=sp["hi"], n_train=M["n_train"], n_val=M["n_val"], best_epoch=M["best_epoch"], best_val=M["best_val"],
                    epochs_run=len(M["history"]), seconds=M["seconds"], expiry_train=sp["expiry_train"], expiry_val=sp["expiry_val"], heads=heads, history=M["history"])
        np.savez(str(bp) + ".tmp.npz", scores=sc, sig=np.array(sigs), meta=json.dumps(meta)); os.replace(str(bp) + ".tmp.npz", bp); ndone += 1
        if ndone % 4 == 0 or rr == nblk - 1: wlog(f"{variant} s{seed}: block {rr+1}/{nblk} done ({time.time()-t0:.0f}s, train {tt:.0f}s, best_ep {M['best_epoch']}, n_train {M['n_train']})")
    return all(block_path(variant, seed, r).exists() for r in range(max_blocks or nblk))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", default="auto"); ap.add_argument("--device", default="cuda"); ap.add_argument("--max-blocks", type=int, default=0); ap.add_argument("--threads", type=int, default=4); a = ap.parse_args()
    jobs = PRIORITY if a.jobs == "auto" else [(j.split(":")[0], int(j.split(":")[1]), int(j.split(":")[2]) if j.count(":") > 1 else 0) for j in a.jobs.split(",")]; T.TF_CFG["threads"] = a.threads; t0 = time.time()
    if all(job_done(*j) for j in jobs): print("nothing to do"); return
    W = C.build_world(); F, mask = Q1.build_seq_features(W)
    for k in ("Xq", "f", "lab", "y"): setattr(W, k, None)          # free memory: only elig / yw / m_tr / ADJf / cal are needed from here on
    W.X = None; import gc; gc.collect(); wlog(f"world built T={W.T} N={W.N} in {time.time()-t0:.0f}s; torch {torch.__version__} device {a.device}")
    for variant, seed, mb in jobs:
        lock = CK / variant / f"s{seed}.lock"; (CK / variant).mkdir(parents=True, exist_ok=True)
        if job_done(variant, seed, mb): continue
        try: fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
        except FileExistsError: continue
        try:
            wlog(f"START {variant} seed {seed} blocks={mb or NBLK}"); ok = run_job(W, F, mask, variant, seed, a.device, max_blocks=mb)
            if ok: wlog(f"DONE {variant} seed {seed} blocks={mb or NBLK} in {time.time()-t0:.0f}s (worker clock)")
        finally:
            try: os.remove(lock)
            except OSError: pass
if __name__ == "__main__": main()