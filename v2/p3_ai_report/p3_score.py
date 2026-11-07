# -*- coding: utf-8 -*-
"""P3-2: score the latest signal date with FROZEN models only (no training):
HGB/LGBM/Ridge = p1_dl/models/*_frozen_2026-10-16.joblib; MLP/GRU = p1_dl/dl/checkpoints/{mlp,gru}/*_latest_s{42,43,44}_r999.pt (3-seed mean, as P1);
TF = tf_latest_s42_r999.pt (seed 42 only, the only latest TF checkpoint). Simple factors REV5/REV20/LOWVOL/ILLIQ + 3-factor composite.
p0lib paths are redirected IN MEMORY to p3_ai_report/data/merged (P0 files are not touched). Before scoring the new date, the 2026-10-16 scores are
re-computed from the merged data and compared with P1's published scores (reproduction / no-drift check).
Writes predictions_registry/pred_<date>.csv (append-only; refuses to overwrite) + manifest.jsonl, and data/world_<date>.npz for later steps."""
import sys, json, hashlib, datetime as dt, os, stat
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; V2 = H.parent; P1 = V2 / "p1_dl"; DLD = P1 / "dl"
sys.path[:0] = [str(V2 / "p0"), str(V2), str(DLD), str(P1)]
import p0lib as L
MG = H / "data" / "merged"; END = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-10-21")
L.D, L.RU, L.FA, L.LAST_TD = MG, MG / "raw_unadj_parquet", MG / "adj_factors", END
import torch, joblib
import dl_common as C, dl_seq as Q, dl_mlp as M
torch.set_num_threads(8)
REG = (H / "data" / "dryrun_registry") if os.environ.get("P3_DRYRUN") else (H / "predictions_registry"); REG.mkdir(exist_ok=True)   # dry runs never touch the real registry
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pct(x, pool):
    out = np.full(len(x), np.nan); v = x[pool]; ok = np.isfinite(v); r = pd.Series(-v[ok]).rank(method="first").values; n = ok.sum()
    out[np.where(pool)[0][ok]] = 1 - (r - 1) / (n - 1); return out          # P1 convention: rank 1 -> 1.0, last -> 0.0
def main():
    t0 = dt.datetime.now(); W = C.build_world(); X = W.X; cal = W.cal; T, N = W.T, W.N; i = T - 1; assert cal[i] == END, (cal[i], END)
    i0 = int(cal.get_loc(pd.Timestamp("2026-10-16"))); codes = list(X.codes); print("world", T, N, "pool", int(W.elig[i].sum()), "pool@1016", int(W.elig[i0].sum()), flush=True)
    MOD = {}
    for n in ["HGB", "LGBM", "Ridge"]:
        f = P1 / "models" / f"{n}_frozen_2026-10-16.joblib"; MOD[n] = dict(kind="sk", files=[f], obj=joblib.load(f))
    for n, tag, seeds in [("MLP", "mlp", [42, 43, 44]), ("GRU", "gru", [42, 43, 44]), ("TF", "tf", [42])]:
        fs = [DLD / "checkpoints" / tag / f"{tag}_latest_s{s}_r999.pt" for s in seeds]; MOD[n] = dict(kind=tag, files=fs, obj=[torch.load(f, weights_only=False) for f in fs])
    F, mask = Q.build_seq_features(W); Scache = {}
    def score(n, d):
        pool = np.where(W.elig[d])[0]; out = np.full(N, np.nan); m = MOD[n]
        if m["kind"] == "sk": out[pool] = m["obj"].predict(W.Xq[d][pool])
        elif m["kind"] == "mlp": out[pool] = np.mean([M.predict(o, W.Xq[d][pool]) for o in m["obj"]], axis=0)
        else:
            v = []
            for o in m["obj"]:
                k = (o["mu"].tobytes(), o["sd"].tobytes())
                if k not in Scache: Scache[k] = torch.tensor(Q.apply_norm(F, mask, o["mu"], o["sd"]))
                v.append(Q.predict_seq(o, Scache[k], d, pool))
            out[pool] = np.mean(v, axis=0)
        return out
    # ---- reproduction check on 2026-10-16 vs P1 published scores ----
    lat = pd.read_csv(P1 / "latest_all_stocks_20261016_v2.csv", dtype={"code": str}, encoding="utf-8-sig").set_index("code").reindex(codes)
    rep = {}; colmap = {"HGB": "hgb_score", "LGBM": "lgbm_score", "Ridge": "ridge_score", "MLP": "dl_mlp_score", "GRU": "dl_gru_score", "TF": "dl_tf_score"}
    el1016 = (lat.eligible_asof.astype(str) == "True").values; rep["pool_1016_identical"] = bool((el1016 == W.elig[i0]).all())
    for n in MOD:
        s = score(n, i0); ref = pd.to_numeric(lat[colmap[n]], errors="coerce").values; ok = np.isfinite(ref) & np.isfinite(s)
        rep[n] = dict(n=int(ok.sum()), max_abs_diff=float(np.max(np.abs(s[ok] - ref[ok]))), spearman=float(pd.Series(s[ok]).corr(pd.Series(ref[ok]), method="spearman")))
    print("repro 10-16:", json.dumps(rep), flush=True)
    # ---- latest date ----
    S = {n: score(n, i) for n in MOD}; pool = W.elig[i]
    f = W.f; r1 = f["r1"].values; amt = X.AMT
    a = (np.abs(pd.DataFrame(r1)) / pd.DataFrame(amt).where(pd.DataFrame(amt) > 0)) * 1e8; ami = a.replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=15).mean().values
    FAC = {"REV5": -f["r5"].values[i], "REV20": -f["r20"].values[i], "LOWVOL": -f["vol20"].values[i], "ILLIQ": ami[i]}
    FAC = {k: np.where(pool, v, np.nan) for k, v in FAC.items()}
    P = {k: pct(v, pool) for k, v in {**S, **FAC}.items()}
    three = np.nanmean(np.column_stack([P["REV5"], P["LOWVOL"], P["ILLIQ"]]), axis=1); three = np.where(pool, three, np.nan); S["F3"] = three; P["F3"] = pct(three, pool)
    for k, v in FAC.items(): S[k] = v
    # ---- registry (append-only) ----
    date = END.strftime("%Y%m%d"); out = REG / f"pred_{date}.csv"
    if out.exists(): raise SystemExit(f"{out} already exists: registry is append-only; refusing to overwrite")
    mv = {n: (f"{n}_frozen_2026-10-16(P1-A1, train 2023-10-17..2026-09-22)" if MOD[n]["kind"] == "sk" else f"{MOD[n]['kind']}_latest_r999_s" + "+".join(p.stem.split('_s')[1].split('_')[0] for p in MOD[n]["files"]) + "(train_end 2026-06-16, val_end 2026-09-22)") for n in MOD}
    mv.update({"REV5": "factor:-r5", "REV20": "factor:-r20", "LOWVOL": "factor:-vol20", "ILLIQ": "factor:amihud20(P2 def)", "F3": "factor:mean pctile(REV5,LOWVOL,ILLIQ)"})
    ck = {n: ";".join(f"{p.name}:{sha(p)[:16]}" for p in MOD[n]["files"]) for n in MOD}
    gen = dt.datetime.now().isoformat(timespec="seconds") + "+08:00"; nm = pd.read_csv(V2 / "data" / "sample_1500.csv", dtype=str).set_index("code").name.reindex(codes).values
    rows = []
    for k in list(MOD) + ["REV5", "REV20", "LOWVOL", "ILLIQ", "F3"]:
        rk = np.where(np.isfinite(P[k]), pd.Series(-S[k]).where(pool & np.isfinite(S[k])).rank(method="first").values, np.nan)
        rows.append(pd.DataFrame(dict(generated_at=gen, signal_date=str(END.date()), code=codes, name=nm, model=k, model_version=mv[k], checkpoint_sha256=ck.get(k, ""), eligible_asof=pool, score=S[k], rank=rk, pctile=P[k])))
    R = pd.concat(rows, ignore_index=True); R["row_hash"] = [hashlib.sha256(f"{a}|{b}|{c}|{d!r}".encode()).hexdigest()[:16] for a, b, c, d in zip(R.signal_date, R.code, R.model, R.score)]
    R.to_csv(out, index=False, encoding="utf-8-sig", float_format="%.10g"); os.chmod(out, stat.S_IREAD)
    man = dict(file=out.name, sha256=sha(out), rows=len(R), n_codes=len(codes), n_eligible=int(pool.sum()), signal_date=str(END.date()), generated_at=gen, models=mv,
               scoring_script_sha256=sha(__file__), merge_info_sha256=sha(H / "data" / "merge_info.json"), repro_check_20261016=rep, note="append-only; never edit; new runs must use a new file name")
    with open(REG / "manifest.jsonl", "a", encoding="utf-8") as fh: fh.write(json.dumps(man, ensure_ascii=False) + "\n")
    # ---- world snapshot for later steps ----
    ma20 = pd.DataFrame(W.ADJf).rolling(20, min_periods=15).mean().values
    np.savez_compressed(H / "data" / f"world_{date}.npz", cal=np.array([str(d.date()) for d in cal]), codes=np.array(codes), ADJf=W.ADJf[-300:], C=X.C[-300:], O=X.O[-300:], Hh=X.Hh[-300:], Lo=X.L[-300:],
                        AMT=X.AMT[-300:], V=X.V[-300:], traded=X.traded[-300:], lim_up=X.lim_up[-300:], lim_dn=X.lim_dn[-300:], elig=W.elig[-300:], ma20=ma20[-300:], has_row=X.has_row[-300:],
                        r5=f["r5"].values[i], r20=f["r20"].values[i], r60=f["r60"].values[i], vol20=f["vol20"].values[i], amt20=f["amt20"].values[i], ami20=ami[i], turn20=pd.DataFrame(X.TURN).rolling(20, min_periods=15).mean().values[i],
                        ADJf_full=W.ADJf.astype(np.float32), elig_full=W.elig, cal_full=np.array([str(d.date()) for d in cal]))
    print("registry", out.name, len(R), "pool", int(pool.sum()), "sec", (dt.datetime.now() - t0).seconds, flush=True)
if __name__ == "__main__": main()