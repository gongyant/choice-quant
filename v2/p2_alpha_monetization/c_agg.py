# -*- coding: utf-8 -*-
"""P2-C aggregation driver -> results/c_*.csv. Run with .venv (CPU). Only COMPLETE jobs are used (T0/LAGLGBM: 207 signals; ablation variants: first 104 signals = 26 retrain blocks)."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as st
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_eval as E, p2lib as Q
R = E.R; N104 = 104; VARS_ABL = ["T1", "T2", "T3", "T0G", "T4"]; SEEDS = [42, 43, 44]
def main():
    t0 = time.time(); C = E.Ctx(); K, N = C.K, C.N; cov_rows = []; by = {}; heads_of = {}; raw = {}
    def need(var): return K if var in ("T0", "LAGLGBM") else N104
    # ---- assemble ----
    for var in ["T0", "LAGLGBM"] + VARS_ABL:
        for sd in SEEDS:
            n = need(var)
            if var == "T0" and sd == 42: Sc = C.D["scores"]["TF"][:, :, None].astype(np.float32); heads = [5]; cov = np.ones(K, bool)
            else: Sc, heads, cov = E.load_variant(var, sd, K, N, C.sig_pos, 52 if var in ("T0", "LAGLGBM") else 26)
            if Sc is None: cov_rows.append(dict(variant=var, seed=sd, n_covered=0, n_needed=n, used=False)); continue
            ncov = int(cov[:n].sum()); fin = np.mean([np.isfinite(Sc[k, :, heads.index(5)])[C.D["cand0"][k]].mean() for k in range(n) if cov[k] and C.D["cand0"][k].sum() >= 100]) if ncov else np.nan
            ok = ncov == n; cov_rows.append(dict(variant=var, seed=sd, n_covered=ncov, n_needed=n, used=ok, finite_share_of_pool=fin, heads=str(heads)))
            if not ok: continue
            raw[(var, sd)] = (Sc, heads)
    pd.DataFrame(cov_rows).to_csv(R / "c_coverage.csv", index=False, encoding="utf-8-sig"); print("assembled", sorted(raw), f"{time.time()-t0:.0f}s", flush=True)
    # ---- per-run metrics ----
    runs = []; ICh = {}
    for (var, sd), (Sc, heads) in sorted(raw.items()):
        n = need(var); h5 = heads.index(5); b = C.run_metrics(Sc[:, :, h5], n); b["xb"] = (b.k % 4 == 0); by[(var, sd)] = b
        if len(heads) > 1:
            for hi, h in enumerate(heads): ICh[(var, sd, h)] = float(np.nanmean(E.rankic_series(Sc[:n, :, hi].astype(float), C.D["cand0"][:n], C.RET[:n])))
        for win in ("all", "eval_k52+", "first104"):
            if win != "all" and n < K: continue
            bb = b if win == "all" else (b[b.k >= E.NB] if win == "eval_k52+" else b[b.k < N104]); s = C.summary(bb)
            s.update(variant=var, seed=sd, window=win, stab_within_block=float(np.nanmean(bb.stab[~bb.xb])), stab_cross_boundary=float(np.nanmean(bb.stab[bb.xb])), overlap_within_block=float(np.nanmean(bb.top50_overlap[~bb.xb])), overlap_cross_boundary=float(np.nanmean(bb.top50_overlap[bb.xb])))
            runs.append(s)
        print("metrics", var, sd, f"{time.time()-t0:.0f}s", flush=True)
    RN = pd.DataFrame(runs); first = ["variant", "seed", "window"]; RN = RN[first + [c for c in RN.columns if c not in first]]; RN.to_csv(R / "c_runs.csv", index=False, encoding="utf-8-sig")
    pd.concat([b.assign(variant=v, seed=s) for (v, s), b in by.items()]).to_parquet(R / "c_by_date.parquet", index=False)
    pd.DataFrame([dict(variant=v, seed=s, horizon=h, rankic_vs_5d_realised=x) for (v, s, h), x in ICh.items()]).to_csv(R / "c_head_rankic.csv", index=False, encoding="utf-8-sig")
    # ---- seed variance (window all; for 104-signal variants plus T0/LAGLGBM restricted to the same 104) ----
    mets = ["rankic", "rankic_hac_t", "stability", "top50_overlap", "turnover_c1", "turnover_c2", "ann_gross_c2", "ann_net_c1", "ann_net_c2", "net_c2_per_period", "alpha_c2", "alpha_c2_hac_t", "sharpe_net_c2"]
    rows = []
    def sub104(var, sd): return by[(var, sd)][by[(var, sd)].k < N104].reset_index(drop=True)
    for var in ["T0", "LAGLGBM"] + VARS_ABL:
        for lab, n in [("207", K), ("104", N104)]:
            if var in VARS_ABL and lab == "207": continue
            ss = [sd for sd in SEEDS if (var, sd) in by]
            if not ss: continue
            per = {sd: C.summary(by[(var, sd)] if lab == "207" else sub104(var, sd)) for sd in ss}
            r = dict(variant=var, signals=lab, seeds=",".join(map(str, ss)), n_seeds=len(ss))
            for m in mets:
                x = np.array([per[sd][m] for sd in ss]); mu, sdv, lo, hi = E.tint(x); r.update({f"{m}_mean": mu, f"{m}_std": sdv, f"{m}_min": float(x.min()), f"{m}_max": float(x.max()), f"{m}_ci_lo": lo, f"{m}_ci_hi": hi})
            cs = []
            for k in range(0, K if lab == "207" else N104):
                vs = [raw[(var, sd)][0][k, :, raw[(var, sd)][1].index(5)] for sd in ss]; m_ = C.D["cand0"][k] & np.all([np.isfinite(v) for v in vs], axis=0)
                if len(ss) > 1 and m_.sum() >= 30: cs.append(np.mean([Q.spearman(vs[a][m_], vs[b][m_]) for a in range(len(ss)) for b in range(a + 1, len(ss))]))
            r["cross_seed_rank_corr"] = float(np.mean(cs)) if cs else np.nan; rows.append(r)
    SV = pd.DataFrame(rows); SV.to_csv(R / "c_seed_variance.csv", index=False, encoding="utf-8-sig")
    # ---- paired comparisons ----
    PAIRS = [("T0G", "T0", "几何(purge)单因素: 20日标签purge vs P1", "isolating"), ("T1", "T0", "损失单因素: 逐日pairwise排序损失 vs MSE(5d头)", "isolating"), ("T2", "T0G", "输出头单因素: 多周期头(1/5/10/20, MSE) vs 单5d头, 同g20几何", "isolating"),
             ("T3", "T2", "损失单因素: 多周期排序损失 vs 多周期MSE, 同g20几何", "isolating"), ("T3", "T0G", "多周期头+排序损失(打包), 同g20几何", "bundle"), ("T4", "T0", "持续性正则 lambda=0.5 vs P1基线", "isolating"),
             ("T3", "T0", "全部改动打包 vs P1基线 (不可分解)", "bundle"), ("LAGLGBM", "T0", "匹配信息集: 滞后堆叠LightGBM vs Transformer", "model class"), ("T3", "T1", "多周期头(+g20几何) vs 单头, 同为排序损失 (几何也不同: 混杂)", "confounded")]
    prow = []
    def get(var, sd, n): b = by[(var, sd)]; return b[b.k < n].reset_index(drop=True)
    for A_, B_, desc, kind in PAIRS:
        for n, lab in [(N104, "104"), (K, "207")]:
            if lab == "207" and (A_ in VARS_ABL or B_ in VARS_ABL): continue
            ss = [sd for sd in SEEDS if (A_, sd) in by and (B_, sd) in by]
            if not ss: continue
            for sd in ss:
                for r in E.paired(get(A_, sd, n), get(B_, sd, n), ctx=C): prow.append(dict(A=A_, B=B_, desc=desc, kind=kind, signals=lab, scope=f"seed{sd}", n_seeds=1, **r))
            if len(ss) > 1:
                for r in E.paired(E.seed_mean_by([get(A_, sd, n) for sd in ss]), E.seed_mean_by([get(B_, sd, n) for sd in ss]), ctx=C): prow.append(dict(A=A_, B=B_, desc=desc, kind=kind, signals=lab, scope="seed-mean", n_seeds=len(ss), **r))
    PR = pd.DataFrame(prow); PR.to_csv(R / "c_paired.csv", index=False, encoding="utf-8-sig"); print("paired", f"{time.time()-t0:.0f}s", flush=True)
    # ---- post-hoc diagnostics (labelled): causal EMA smoothing of the baseline's scores; seed-ensemble ----
    post = []
    for sd in SEEDS:
        if ("T0", sd) not in raw: continue
        Sc = raw[("T0", sd)][0][:, :, 0]; Se = E.ema_scores(Sc, C.D["cand0"], 0.5); b = C.run_metrics(Se, K); s = C.summary(b); post.append(dict(diag="T0+EMA(0.5) [post-hoc]", seed=sd, **s))
        for r in E.paired(b, by[("T0", sd)], ctx=C): post.append(dict(diag="T0+EMA(0.5) minus T0 [post-hoc paired]", seed=sd, **r))
        by[("T0EMA", sd)] = b
    for var in ("T0", "LAGLGBM"):
        ss = [sd for sd in SEEDS if (var, sd) in raw]
        if len(ss) >= 2:
            Sc = np.nanmean(np.stack([np.where(C.D["cand0"][k][None, :], np.stack([E.rank_pct(raw[(var, sd)][0][k, :, 0].astype(float), C.D["cand0"][k] & np.isfinite(raw[(var, sd)][0][k, :, 0])) for sd in ss]), np.nan) for k in range(K)], axis=1), axis=0)
            b = C.run_metrics(Sc, K); post.append(dict(diag=f"{var} seed-ensemble (rank-average of {len(ss)} seeds) [post-hoc]", seed=-1, **C.summary(b)))
    Sf, hf, cf = E.load_variant("LAGLGBM_F200", 42, K, N, C.sig_pos, 52)
    if Sf is not None and cf.all():
        b = C.run_metrics(Sf[:, :, 0], K); by[("LAGLGBM_F200", 42)] = b; post.append(dict(diag="LAGLGBM fixed 200 trees, no early stopping [post-hoc sensitivity]", seed=42, **C.summary(b)))
        for ref in ("T0", "LAGLGBM"):
            for r in E.paired(b, by[(ref, 42)], ctx=C): post.append(dict(diag=f"LAGLGBM_F200 minus {ref} [post-hoc paired]", seed=42, **r))
    pd.DataFrame(post).to_csv(R / "c_posthoc.csv", index=False, encoding="utf-8-sig")
    # ---- oracle synthetic diagnostics (use FUTURE returns by construction; NOT a strategy; only shows how score persistence maps to turnover/cost at a given IC level) ----
    rng = np.random.default_rng(20261019); orows = []; cand = C.D["cand0"]; Z = np.zeros((K, N))
    for k in range(K):
        m = cand[k] & np.isfinite(C.RET[k])
        if m.sum() > 30: Z[k, m] = st.norm.ppf((st.rankdata(C.RET[k][m]) - 0.5) / m.sum())
    rho = 0.05
    for phi in (0.0, 0.5, 0.8, 0.95):
        e = rng.standard_normal(N); Sc = np.zeros((K, N))
        for k in range(K): e = phi * e + np.sqrt(1 - phi ** 2) * rng.standard_normal(N); Sc[k] = rho * Z[k] + np.sqrt(1 - rho ** 2) * e
        Sc = np.where(cand, Sc, np.nan); b = C.run_metrics(Sc, K); orows.append(dict(label="ORACLE-SYNTHETIC (diagnostic only, uses future returns)", rho_true_signal=rho, noise_ar1_phi=phi, **C.summary(b)))
    pd.DataFrame(orows).to_csv(R / "c_oracle_diag.csv", index=False, encoding="utf-8-sig")
    # ---- reconciliation with frozen P1/P2-A/P2-B ----
    rec = []; PFB = pd.read_csv(R / "b_performance.csv"); r0 = RN[(RN.variant == "T0") & (RN.seed == 42) & (RN.window == "all")].iloc[0]
    for scen, tag in (("C1_flat", "c1"), ("C2_explicit", "c2")):
        x = PFB[(PFB.model == "TF") & (PFB.scenario == scen) & (PFB.variant == "A0") & (PFB.window == "full207")].iloc[0]
        rec.append(dict(check=f"P2-B A0 TF {scen} full207: net per period", p2b=x.net_mean_per_period, p2c=r0[f"net_{tag}_per_period"], diff=r0[f"net_{tag}_per_period"] - x.net_mean_per_period))
        rec.append(dict(check=f"P2-B A0 TF {scen} full207: turnover/period", p2b=x.turnover_oneway_per_period, p2c=r0[f"turnover_{tag}"], diff=r0[f"turnover_{tag}"] - x.turnover_oneway_per_period))
    ic = pd.read_csv(R / "rankic_old_vs_new.csv"); rec.append(dict(check="P2-A RankIC TF (new definition)", p2b=np.nan, p2c=r0["rankic"], diff=np.nan, note="see rankic_old_vs_new.csv: " + str(ic[(ic.model == "TF")].head(3).to_dict("records")) if "model" in ic.columns else ""))
    pd.DataFrame(rec).to_csv(R / "c_reconcile.csv", index=False, encoding="utf-8-sig")
    # ---- training diagnostics from block metadata ----
    mr = []
    for var in ["T0", "LAGLGBM"] + VARS_ABL:
        for p in sorted((E.CK / var).glob("s*_r*.npz")):
            z = np.load(p, allow_pickle=False); m = json.loads(str(z["meta"])); sd = int(p.stem.split("_")[0][1:]); mr.append(dict(variant=var, seed=sd, retrain=m["retrain"], n_train=m["n_train"], n_val=m["n_val"], best_epoch=m.get("best_epoch", np.nan), best_iteration=m.get("best_iteration", np.nan), epochs_run=m.get("epochs_run", np.nan), best_val=m.get("best_val", np.nan), seconds=m["seconds"]))
    pd.DataFrame(mr).to_csv(R / "c_block_meta.csv", index=False, encoding="utf-8-sig")
    print("done", f"{time.time()-t0:.0f}s")
if __name__ == "__main__": main()