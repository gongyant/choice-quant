"""P1-B1 walk-forward for the PyTorch MLP. Same universe, same eligible_asof_t pool, same label, same 3y window / purge / retrain-every-4-periods / Top50 / simulator / cost
convention as P0 (p0/baseline_fixed.py). Only the model differs. Resumable: per-(seed,retrain) checkpoints in p1_dl/dl/checkpoints/."""
import sys, json, argparse, pickle, time, datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dl_common as C; import dl_mlp as M
L = C.L; DL = C.DL; P1 = C.P1
COSTS = [0.0, 0.001, 0.0015, 0.003, 0.006]
def cname(c): return f"c{int(round(c * 10000))}"
def ts(): return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def wlog(msg):
    with open(DL / "worklog_dl.md", "a", encoding="utf-8") as f: f.write(f"- [{ts()}] {msg}\n")
    print(msg, flush=True)
def get_model(W, i, seed, rr, tag, ckdir, tlog, curves):
    import torch
    ck = ckdir / f"{tag}_s{seed}_r{rr:03d}.pt"
    sp = C.split_train_val(W.Xq, W.yw, W.m_tr, W.cal, i)
    if ck.exists():
        m = torch.load(ck, weights_only=False); return m, sp
    m = M.fit_mlp(sp["Xtr"], sp["ytr"], sp["Xva"], sp["yva"], seed)
    torch.save(m, ck)
    tlog.append(dict(tag=tag, seed=seed, retrain=rr, trained_asof=str(W.cal[i].date()), n_train=m["n_train"], n_val=m["n_val"], best_epoch=m["best_epoch"], best_val_mse=m["best_val"],
                     epochs_run=len(m["history"]), last_train_loss=m["history"][-1]["train_loss"], seconds=m["seconds"]))
    for h in m["history"]: curves.append(dict(tag=tag, seed=seed, retrain=rr, **h))
    return m, sp
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="mlp"); ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--topk", type=int, default=50); ap.add_argument("--retrain", type=int, default=4); ap.add_argument("--first-test", default="2022-07-01")
    ap.add_argument("--max-signals", type=int, default=0); ap.add_argument("--no-latest", action="store_true"); ap.add_argument("--check-hgb", action="store_true")
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    out = DL / "results" / a.tag; out.mkdir(parents=True, exist_ok=True); ckdir = DL / "checkpoints" / a.tag; ckdir.mkdir(parents=True, exist_ok=True); t0 = time.time()
    import torch; torch.set_num_threads(M.CFG["threads"])
    wlog(f"run {a.tag}: torch {torch.__version__}, cuda_available={torch.cuda.is_available()}, device=cpu (deterministic), seeds={a.seeds}, max_signals={a.max_signals}")
    W = C.build_world(); X, cal, T, N = W.X, W.cal, W.T, W.N; wlog(f"world built: T={T} N={N} in {time.time()-t0:.0f}s")
    first = int(cal.searchsorted(pd.Timestamp(a.first_test))); sig_all = list(range(first, T - L.H - 1, L.H))
    isd = X.u.delist_date.notna().values   # DIAGNOSTIC ONLY (future information), never used for selection
    names = [f"MLP_s{s}" for s in a.seeds] + ["MLP_mean"]
    models = {}; tlog, curves = [], []; sig_used, pools, topk = [], [], {n: [] for n in names}; scores_store, rows_ic = [], []
    cur_meta = None; t_train = 0.0
    for ix, i in enumerate(sig_all):
        if a.max_signals and ix >= a.max_signals: break
        if ix % a.retrain == 0:
            rr = ix // a.retrain; spx = C.split_train_val(W.Xq, W.yw, W.m_tr, cal, i)
            if len(spx["Xtr"]) + len(spx["Xva"]) < 20000: continue     # same availability rule as P0 (>=20000 training rows)
            tt = time.time()
            for s in a.seeds: models[s], sp = get_model(W, i, s, rr, a.tag, ckdir, tlog, curves)
            t_train += time.time() - tt
            cur_meta = dict(model_id=f"{a.tag}_r{rr:03d}", trained_asof=str(cal[i].date()), train_window_start=str(cal[sp["lo"]].date()), train_end=str(cal[sp["te"]].date()), val_start=str(cal[sp["vs"]].date()),
                            val_end=str(cal[sp["hi"]].date()), max_label_expiry=str(cal[sp["hi"] + L.H + 1].date()), n_train=int(len(sp["Xtr"])), n_val=int(len(sp["Xva"])))
            assert sp["label_expiry_train_max"] < sp["vs"] and sp["label_expiry_val_max"] < i
            wlog(f"retrain r{rr:03d} at {cal[i].date()}: n_train={cur_meta['n_train']} n_val={cur_meta['n_val']} train_end={cur_meta['train_end']} val_end={cur_meta['val_end']} (max label expiry {cur_meta['max_label_expiry']} < signal date); cumulative train time {t_train:.0f}s")
        if not models: continue
        pool = np.where(W.elig[i])[0]
        if len(pool) < max(100, a.topk * 2): continue
        xs = W.Xq[i][pool]; sc = {f"MLP_s{s}": M.predict(models[s], xs) for s in a.seeds}; sc["MLP_mean"] = np.mean([sc[f"MLP_s{s}"] for s in a.seeds], axis=0)
        yy = W.y[i][pool]; fin = np.isfinite(yy); rec = {"signal_date": str(cal[i].date()), "n_pool": int(len(pool))}
        for n in names:
            s_ = np.nan_to_num(sc[n], nan=-1e9); top = pool[np.argsort(-s_, kind="stable")[:a.topk]]; topk[n].append(top); rec[f"{n}_delist_share"] = float(isd[top].mean())
            rec[f"{n}_IC"] = float(np.corrcoef(sc[n][fin], yy[fin])[0, 1]) if fin.sum() > 5 and np.std(sc[n][fin]) > 0 else np.nan
            rec[f"{n}_RankIC"] = float(spearmanr(sc[n][fin], yy[fin]).statistic) if fin.sum() > 5 and np.std(sc[n][fin]) > 0 else np.nan
        sig_used.append(i); pools.append(pool); rows_ic.append(rec); scores_store.append(dict(i=i, pool=pool, sc=sc, meta=dict(cur_meta)))
        if ix % 25 == 0: print("signal", ix, cal[i].date(), round(time.time() - t0), "s", flush=True)
    assert len(sig_used) == 0 or all(np.diff(sig_used) == L.H), "signals must be contiguous 5-day periods"
    # --- identity with P0 universe/pools ---
    p0 = pickle.load(open(L.P0 / "results" / "fixed_sina_adj" / "sets.pkl", "rb"))
    n_cmp = len(sig_used); same_sig = list(p0["signal_idx"][:n_cmp]) == list(sig_used)
    same_pool = all([X.codes[j] for j in pools[k]] == p0["pools"][k] for k in range(n_cmp)) if same_sig else False
    wlog(f"identity vs P0 fixed_sina_adj: signals identical={same_sig}, eligible pools identical for all {n_cmp} periods={same_pool}")
    assert same_sig and same_pool, "universe/pool differs from P0"
    if a.check_hgb:
        sys.path.insert(0, str(L.V2)); import baseline as B
        hg = B.make_models(42)["HGB"]; i0 = sig_used[0]; Xt, yt = L.train_rows(W.Xq, W.yw, W.m_tr, cal, i0); m = hg(); m.fit(Xt, yt)
        ok = []
        for k in range(min(n_cmp, a.retrain)):
            pool = pools[k]; s_ = m.predict(W.Xq[sig_used[k]][pool]); top = [X.codes[j] for j in pool[np.argsort(-np.nan_to_num(s_, nan=-1e9), kind="stable")[:a.topk]]]
            ok.append(top == p0["topk"]["HGB"][k])
        wlog(f"HGB reproduction check in this pipeline: Top50 identical to P0 for {sum(ok)}/{len(ok)} periods")
    # --- simulation (same engine) ---
    res = pd.DataFrame({"signal_date": [r["signal_date"] for r in rows_ic], "order_date": [str(cal[i + 1].date()) for i in sig_used], "n_pool": [r["n_pool"] for r in rows_ic]})
    logs = []; extras = {}; ts0 = time.time()
    for sname in names + ["EqualWeight"]:
        tg = pools if sname == "EqualWeight" else topk[sname]; cap = 1e9 if sname == "EqualWeight" else 1e7
        for c in COSTS:
            o, lg, ex = L.simulate(X, W.ADJf, sig_used, tg, c, capital=cap, log_name=("MLP_mean" if (abs(c - 0.003) < 1e-12 and sname == "MLP_mean") else None))
            res[f"{sname}_ret_{cname(c)}"] = o.ret.values
            if abs(c - 0.003) < 1e-12:
                for k in ["replaced", "turnover_value", "cash_frac_post", "n_hold", "n_pending_exit", "buy_attempts", "buy_filled"]: res[f"{sname}_{k}"] = o[k].values
                for k in [k for k in o.columns if k.startswith("rej_")]: res[f"{sname}_{k}"] = o[k].fillna(0).values
                extras[sname] = ex
                if len(lg): logs.append(lg)
    wlog(f"simulation done in {time.time()-ts0:.0f}s")
    ic = pd.DataFrame(rows_ic).drop(columns=["signal_date", "n_pool"]); res = pd.concat([res, ic], axis=1)
    p0r = pd.read_csv(L.P0 / "results" / "fixed_sina_adj" / "period_returns.csv")
    assert (p0r.signal_date.values[:len(res)] == res.signal_date.values).all()
    for c in ["c0", "c30", "c60"]:
        d = np.abs(res[f"EqualWeight_ret_{c}"].values - p0r[f"EqualWeight_ret_{c}"].values[:len(res)]).max(); wlog(f"EqualWeight {c}: max |diff| vs P0 file = {d:.2e}")
    res.to_csv(out / "period_returns.csv", index=False, encoding="utf-8-sig")
    if logs: pd.concat(logs).to_csv(out / "orders_MLP_mean_cost30bps.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tlog).to_csv(out / "train_log.csv", index=False); pd.DataFrame(curves).to_csv(out / "loss_curves.csv", index=False)
    # --- OOS prediction parquet ---
    parts = []; codes = np.array(X.codes, dtype=object); order_log = pd.concat(logs) if logs else pd.DataFrame()
    buy = order_log[order_log.side == "BUY"] if len(order_log) else order_log
    bmap = {(r.signal_date, r.code): (r.status, r.reason) for r in buy.itertuples()} if len(buy) else {}
    for k, st in enumerate(scores_store):
        i = st["i"]; pool = st["pool"]; sc = st["sc"]; df = pd.DataFrame({"signal_date": str(cal[i].date()), "code": codes}); m = st["meta"]
        for key in ["model_id", "trained_asof", "train_window_start", "train_end", "val_start", "val_end", "max_label_expiry"]: df[key] = m[key]
        elig = np.zeros(N, bool); elig[pool] = True; df["eligible_asof"] = elig
        for s in a.seeds + ["mean"]:
            col = np.full(N, np.nan); col[pool] = sc[f"MLP_s{s}"] if s != "mean" else sc["MLP_mean"]; df[f"mlp_s{s}" if s != "mean" else "mlp_score_mean"] = col
        rk = np.full(N, np.nan); order = np.argsort(-np.nan_to_num(sc["MLP_mean"], nan=-1e9), kind="stable"); rk[pool[order]] = np.arange(1, len(pool) + 1); df["rank_in_pool"] = rk
        df["in_top50"] = df.rank_in_pool <= a.topk
        matured = bool(i + L.H + 1 <= T - 1 and cal[i + L.H + 1] <= L.LAST_TD); lab = W.lab[i].copy(); df["label_matured"] = matured
        lab = lab if matured else np.full(N, np.nan); df["realized_ret_5d"] = lab; pm = np.nanmean(lab[pool]) if matured else np.nan
        df["pool_mean_ret_5d"] = pm; df["realized_excess_5d"] = lab - pm if matured else np.nan
        df["realized_label_expiry_date"] = str(cal[i + L.H + 1].date()); sig = df.signal_date.iloc[0]
        df["buy_status_cost30"] = [bmap.get((sig, c), ("", ""))[0] for c in codes]; df["buy_reject_reason"] = [bmap.get((sig, c), ("", ""))[1] for c in codes]
        parts.append(df)
    oos = pd.concat(parts, ignore_index=True); oos["code"] = oos["code"].astype(str)
    outp = P1 / "scores" / ("dl_mlp_oos_predictions.parquet" if not a.smoke else "dl_smoke_oos_predictions.parquet"); oos.to_parquet(outp, index=False)
    wlog(f"OOS predictions parquet: {outp} rows={len(oos)} (periods={len(parts)}, codes={N})")
    json.dump(dict(args=vars(a), n_periods=len(res), seconds_total=round(time.time() - t0), seconds_training=round(t_train), torch=torch.__version__, cuda_available=torch.cuda.is_available(), terminal_unsold=extras), open(out / "run_info.json", "w"), indent=1, ensure_ascii=False)
    if not a.no_latest: run_latest(W, a, ckdir, out)
    wlog(f"run {a.tag} finished in {time.time()-t0:.0f}s")
def run_latest(W, a, ckdir, out):
    """Models for the newest signal date (last bar 2026-10-16). Training uses ONLY samples whose labels have expired (P0 purge hi=i-12 => label expiry <= i-6) and validation inside that window."""
    i = W.T - 1; cal = W.cal; X = W.X; assert cal[i] == L.LAST_TD, cal[i]
    ms = {}; tl, cv = [], []
    for s in a.seeds: ms[s], sp = get_model(W, i, s, 999, a.tag + "_latest", ckdir, tl, cv)
    pool = np.where(W.elig[i])[0]; sc = {s: np.full(W.N, np.nan) for s in a.seeds}
    for s in a.seeds: sc[s][pool] = M.predict(ms[s], W.Xq[i][pool])
    mean = np.mean([sc[s] for s in a.seeds], axis=0); rk = np.full(W.N, np.nan); order = np.argsort(-np.nan_to_num(mean[pool], nan=-1e9), kind="stable"); rk[pool[order]] = np.arange(1, len(pool) + 1)
    df = pd.DataFrame({"code": pd.Series(X.codes, dtype=str), "mlp_score_mean": mean, **{f"mlp_s{s}": sc[s] for s in a.seeds}, "rank": rk, "eligible_asof": W.elig[i]})
    df["asof_date"] = str(cal[i].date()); df["model_id"] = f"{a.tag}_latest"; df["train_end"] = str(cal[sp["te"]].date()); df["val_end"] = str(cal[sp["hi"]].date()); df["max_label_expiry"] = str(cal[sp["hi"] + L.H + 1].date())
    assert df.code.str.len().eq(6).all() and df.code.is_unique
    p = P1 / "scores" / ("dl_mlp_scores_20261016.csv" if not a.smoke else "dl_smoke_scores_20261016.csv"); df.to_csv(p, index=False, encoding="utf-8-sig")
    pd.DataFrame(tl).to_csv(out / "train_log_latest.csv", index=False); pd.DataFrame(cv).to_csv(out / "loss_curves_latest.csv", index=False)
    wlog(f"latest scores {p}: rows={len(df)} eligible={int(df.eligible_asof.sum())} ranked={int(np.isfinite(df['rank']).sum())}; latest model train_end={df.train_end[0]} val_end={df.val_end[0]} max label expiry={df.max_label_expiry[0]} (< {cal[i].date()})")
if __name__ == "__main__": main()