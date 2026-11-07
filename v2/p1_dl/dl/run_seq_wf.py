"""P1-B2 walk-forward for the sequence model (GRU). Identical universe / eligible_asof_t pool / label / 3y window / purge / retrain-every-4 / Top50 / simulator / costs as P0 and B1.
Only the model input (60-day sequence) and network differ. Resumable via per-(seed,retrain) checkpoints."""
import sys, json, argparse, pickle, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dl_common as C, dl_seq as Q, run_mlp_wf as R
L = C.L; DL = C.DL; P1 = C.P1; wlog = R.wlog
def get_seq_model(W, F, mask, i, seed, rr, tag, ckdir, tlog, curves, device, cfg, S_cache):
    ck = ckdir / f"{tag}_s{seed}_r{rr:03d}.pt"; sp = S_cache["sp"]
    if ck.exists(): return torch.load(ck, weights_only=False)
    M = Q.fit_seq(S_cache["S"], sp["tr"], sp["va"], seed, cfg, device); M["mu"], M["sd"] = S_cache["mu"], S_cache["sd"]; torch.save(M, ck)
    tlog.append(dict(tag=tag, seed=seed, retrain=rr, trained_asof=str(W.cal[i].date()), n_train=M["n_train"], n_val=M["n_val"], best_epoch=M["best_epoch"], best_val_mse=M["best_val"], epochs_run=len(M["history"]),
                     last_train_loss=M["history"][-1]["train_loss"], seconds=M["seconds"], device=device, subsample=cfg.get("subsample", 1)))
    for h in M["history"]: curves.append(dict(tag=tag, seed=seed, retrain=rr, **h))
    return M
def prep(W, F, mask, i, device):
    sp = Q.split_idx(W.m_tr, W.yw, W.cal, i); mu, sd = Q.fit_norm(F, sp["lo"], sp["te"]); S = torch.tensor(Q.apply_norm(F, mask, mu, sd), device=device)
    assert sp["label_expiry_train_max"] < sp["vs"] and sp["label_expiry_val_max"] < i
    return dict(sp=sp, mu=mu, sd=sd, S=S)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="gru"); ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44]); ap.add_argument("--device", default="auto")
    ap.add_argument("--topk", type=int, default=50); ap.add_argument("--retrain", type=int, default=4); ap.add_argument("--first-test", default="2022-07-01"); ap.add_argument("--max-signals", type=int, default=0)
    ap.add_argument("--subsample", type=int, default=1); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--max-epochs", type=int, default=12); ap.add_argument("--no-latest", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--latest-only", action="store_true"); ap.add_argument("--arch", default="gru", choices=["gru", "tf"]); a = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device; cfg = dict(subsample=a.subsample, threads=a.threads, max_epochs=a.max_epochs, arch=a.arch)
    if a.arch == "tf": cfg.update(d_model=64, nhead=4, ff=128, layers=2)
    PFX, pf = ("GRU", "gru") if a.arch == "gru" else ("TF", "tf"); out = DL / "results" / a.tag; out.mkdir(parents=True, exist_ok=True); ckdir = DL / "checkpoints" / a.tag; ckdir.mkdir(parents=True, exist_ok=True); t0 = time.time()
    wlog(f"run {a.tag}: torch {torch.__version__}, device={device}, seeds={a.seeds}, subsample={a.subsample}, max_epochs={a.max_epochs}, max_signals={a.max_signals}, threads={a.threads}")
    W = C.build_world(); X, cal, T, N = W.X, W.cal, W.T, W.N; F, mask = Q.build_seq_features(W); wlog(f"world + sequence features built: T={T} N={N} channels={F.shape[2]} in {time.time()-t0:.0f}s")
    first = int(cal.searchsorted(pd.Timestamp(a.first_test))); sig_all = list(range(first, T - L.H - 1, L.H)); isd = X.u.delist_date.notna().values
    names = [f"{PFX}_s{s}" for s in a.seeds] + [f"{PFX}_mean"]; models = {}; tlog, curves = [], []; sig_used, pools, topk = [], [], {n: [] for n in names}; scores_store, rows_ic = [], []; cur_meta = None; t_train = 0.0; Sc = None
    if a.latest_only: run_latest(W, F, mask, a, ckdir, out, device, cfg, PFX, pf); return
    for ix, i in enumerate(sig_all):
        if a.max_signals and ix >= a.max_signals: break
        if ix % a.retrain == 0:
            rr = ix // a.retrain; Sc = prep(W, F, mask, i, device); sp = Sc["sp"]
            if len(sp["tr"][0]) + len(sp["va"][0]) < 20000: continue
            tt = time.time()
            for s in a.seeds: models[s] = get_seq_model(W, F, mask, i, s, rr, a.tag, ckdir, tlog, curves, device, cfg, Sc)
            t_train += time.time() - tt
            cur_meta = dict(model_id=f"{a.tag}_r{rr:03d}", trained_asof=str(cal[i].date()), train_window_start=str(cal[sp["lo"]].date()), train_end=str(cal[sp["te"]].date()), val_start=str(cal[sp["vs"]].date()),
                            val_end=str(cal[sp["hi"]].date()), max_label_expiry=str(cal[sp["hi"] + L.H + 1].date()), n_train=int(len(sp["tr"][0])), n_val=int(len(sp["va"][0])))
            for s in a.seeds: assert np.allclose(models[s]["mu"], Sc["mu"]) and np.allclose(models[s]["sd"], Sc["sd"])    # checkpoint stats == training-window stats
            wlog(f"retrain r{rr:03d} at {cal[i].date()}: n_train={cur_meta['n_train']} n_val={cur_meta['n_val']} train_end={cur_meta['train_end']} val_end={cur_meta['val_end']} (max label expiry {cur_meta['max_label_expiry']} < signal date); cumulative train time {t_train:.0f}s")
        if not models: continue
        pool = np.where(W.elig[i])[0]
        if len(pool) < max(100, a.topk * 2): continue
        sc = {f"{PFX}_s{s}": Q.predict_seq(models[s], Sc["S"], i, pool, device) for s in a.seeds}; sc[f"{PFX}_mean"] = np.mean([sc[f"{PFX}_s{s}"] for s in a.seeds], axis=0)
        yy = W.y[i][pool]; fin = np.isfinite(yy); rec = {"signal_date": str(cal[i].date()), "n_pool": int(len(pool))}
        for n in names:
            s_ = np.nan_to_num(sc[n], nan=-1e9); top = pool[np.argsort(-s_, kind="stable")[:a.topk]]; topk[n].append(top); rec[f"{n}_delist_share"] = float(isd[top].mean())
            rec[f"{n}_IC"] = float(np.corrcoef(sc[n][fin], yy[fin])[0, 1]) if fin.sum() > 5 and np.std(sc[n][fin]) > 0 else np.nan
            rec[f"{n}_RankIC"] = float(spearmanr(sc[n][fin], yy[fin]).statistic) if fin.sum() > 5 and np.std(sc[n][fin]) > 0 else np.nan
        sig_used.append(i); pools.append(pool); rows_ic.append(rec); scores_store.append(dict(i=i, pool=pool, sc=sc, meta=dict(cur_meta)))
        if ix % 25 == 0: print("signal", ix, cal[i].date(), round(time.time() - t0), "s", flush=True)
    assert len(sig_used) == 0 or all(np.diff(sig_used) == L.H)
    p0 = pickle.load(open(L.P0 / "results" / "fixed_sina_adj" / "sets.pkl", "rb")); n_cmp = len(sig_used); same_sig = list(p0["signal_idx"][:n_cmp]) == list(sig_used)
    same_pool = all([X.codes[j] for j in pools[k]] == p0["pools"][k] for k in range(n_cmp)) if same_sig else False
    wlog(f"identity vs P0 fixed_sina_adj: signals identical={same_sig}, eligible pools identical for all {n_cmp} periods={same_pool}"); assert same_sig and same_pool
    res = pd.DataFrame({"signal_date": [r["signal_date"] for r in rows_ic], "order_date": [str(cal[i + 1].date()) for i in sig_used], "n_pool": [r["n_pool"] for r in rows_ic]})
    logs = []; extras = {}; ts0 = time.time(); main_n = f"{PFX}_mean"
    for sname in names + ["EqualWeight"]:
        tg = pools if sname == "EqualWeight" else topk[sname]; cap = 1e9 if sname == "EqualWeight" else 1e7
        for c in R.COSTS:
            o, lg, ex = L.simulate(X, W.ADJf, sig_used, tg, c, capital=cap, log_name=(main_n if (abs(c - 0.003) < 1e-12 and sname == main_n) else None)); res[f"{sname}_ret_{R.cname(c)}"] = o.ret.values
            if abs(c - 0.003) < 1e-12:
                for k in ["replaced", "turnover_value", "cash_frac_post", "n_hold", "n_pending_exit", "buy_attempts", "buy_filled"]: res[f"{sname}_{k}"] = o[k].values
                for k in [k for k in o.columns if k.startswith("rej_")]: res[f"{sname}_{k}"] = o[k].fillna(0).values
                extras[sname] = ex
                if len(lg): logs.append(lg)
    wlog(f"simulation done in {time.time()-ts0:.0f}s"); res = pd.concat([res, pd.DataFrame(rows_ic).drop(columns=["signal_date", "n_pool"])], axis=1)
    p0r = pd.read_csv(L.P0 / "results" / "fixed_sina_adj" / "period_returns.csv"); assert (p0r.signal_date.values[:len(res)] == res.signal_date.values).all()
    for c in ["c0", "c30", "c60"]: wlog(f"EqualWeight {c}: max |diff| vs P0 file = {np.abs(res[f'EqualWeight_ret_{c}'].values - p0r[f'EqualWeight_ret_{c}'].values[:len(res)]).max():.2e}")
    res.to_csv(out / "period_returns.csv", index=False, encoding="utf-8-sig")
    if logs: pd.concat(logs).to_csv(out / f"orders_{main_n}_cost30bps.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tlog).to_csv(out / "train_log.csv", index=False); pd.DataFrame(curves).to_csv(out / "loss_curves.csv", index=False)
    parts = []; codes = np.array(X.codes, dtype=object); order_log = pd.concat(logs) if logs else pd.DataFrame(); buy = order_log[order_log.side == "BUY"] if len(order_log) else order_log
    bmap = {(r.signal_date, r.code): (r.status, r.reason) for r in buy.itertuples()} if len(buy) else {}
    for k, st in enumerate(scores_store):
        i = st["i"]; pool = st["pool"]; sc = st["sc"]; df = pd.DataFrame({"signal_date": str(cal[i].date()), "code": codes}); m = st["meta"]
        for key in ["model_id", "trained_asof", "train_window_start", "train_end", "val_start", "val_end", "max_label_expiry"]: df[key] = m[key]
        elig = np.zeros(N, bool); elig[pool] = True; df["eligible_asof"] = elig
        for s in a.seeds + ["mean"]:
            col = np.full(N, np.nan); col[pool] = sc[f"{PFX}_s{s}"] if s != "mean" else sc[main_n]; df[f"{pf}_s{s}" if s != "mean" else f"{pf}_score_mean"] = col
        rk = np.full(N, np.nan); order = np.argsort(-np.nan_to_num(sc[main_n], nan=-1e9), kind="stable"); rk[pool[order]] = np.arange(1, len(pool) + 1); df["rank_in_pool"] = rk; df["in_top50"] = df.rank_in_pool <= a.topk
        matured = bool(i + L.H + 1 <= T - 1 and cal[i + L.H + 1] <= L.LAST_TD); lab = W.lab[i].copy() if matured else np.full(N, np.nan); df["label_matured"] = matured; df["realized_ret_5d"] = lab
        pm = np.nanmean(lab[pool]) if matured else np.nan; df["pool_mean_ret_5d"] = pm; df["realized_excess_5d"] = lab - pm if matured else np.nan; df["realized_label_expiry_date"] = str(cal[i + L.H + 1].date()); sig = df.signal_date.iloc[0]
        df["buy_status_cost30"] = [bmap.get((sig, c), ("", ""))[0] for c in codes]; df["buy_reject_reason"] = [bmap.get((sig, c), ("", ""))[1] for c in codes]; parts.append(df)
    oos = pd.concat(parts, ignore_index=True); oos["code"] = oos["code"].astype(str)
    outp = P1 / "scores" / (f"dl_{pf}_oos_predictions.parquet" if not a.smoke else f"dl_smoke_{pf}_oos_predictions.parquet"); oos.to_parquet(outp, index=False)
    wlog(f"OOS predictions parquet: {outp} rows={len(oos)} (periods={len(parts)}, codes={N})")
    json.dump(dict(args=vars(a), device=device, n_periods=len(res), seconds_total=round(time.time() - t0), seconds_training=round(t_train), torch=torch.__version__, terminal_unsold=extras), open(out / "run_info.json", "w"), indent=1, ensure_ascii=False)
    if not a.no_latest: run_latest(W, F, mask, a, ckdir, out, device, cfg, PFX, pf)
    wlog(f"run {a.tag} finished in {time.time()-t0:.0f}s")
def run_latest(W, F, mask, a, ckdir, out, device, cfg, PFX, pf):
    i = W.T - 1; cal = W.cal; X = W.X; assert cal[i] == L.LAST_TD; Sc = prep(W, F, mask, i, device); sp = Sc["sp"]; ms = {}; tl, cv = [], []
    for s in a.seeds: ms[s] = get_seq_model(W, F, mask, i, s, 999, a.tag + "_latest", ckdir, tl, cv, device, cfg, Sc)
    pool = np.where(W.elig[i])[0]; sc = {s: np.full(W.N, np.nan) for s in a.seeds}
    for s in a.seeds: sc[s][pool] = Q.predict_seq(ms[s], Sc["S"], i, pool, device)
    mean = np.mean([sc[s] for s in a.seeds], axis=0); rk = np.full(W.N, np.nan); order = np.argsort(-np.nan_to_num(mean[pool], nan=-1e9), kind="stable"); rk[pool[order]] = np.arange(1, len(pool) + 1)
    df = pd.DataFrame({"code": pd.Series(X.codes, dtype=str), f"{pf}_score_mean": mean, **{f"{pf}_s{s}": sc[s] for s in a.seeds}, "rank": rk, "eligible_asof": W.elig[i]})
    df["asof_date"] = str(cal[i].date()); df["model_id"] = f"{a.tag}_latest"; df["train_end"] = str(cal[sp["te"]].date()); df["val_end"] = str(cal[sp["hi"]].date()); df["max_label_expiry"] = str(cal[sp["hi"] + L.H + 1].date())
    assert df.code.str.len().eq(6).all() and df.code.is_unique
    p = P1 / "scores" / (f"dl_{pf}_scores_20261016.csv" if not a.smoke else f"dl_smoke_{pf}_scores_20261016.csv"); df.to_csv(p, index=False, encoding="utf-8-sig")
    if tl: pd.DataFrame(tl).to_csv(out / "train_log_latest.csv", index=False); pd.DataFrame(cv).to_csv(out / "loss_curves_latest.csv", index=False)
    wlog(f"latest scores {p}: rows={len(df)} eligible={int(df.eligible_asof.sum())} ranked={int(np.isfinite(df['rank']).sum())}; train_end={df.train_end[0]} val_end={df.val_end[0]} max label expiry={df.max_label_expiry[0]} (< {cal[i].date()})")
if __name__ == "__main__": main()