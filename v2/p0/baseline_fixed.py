"""P0-fixed walk-forward baseline (replaces the look-ahead baseline.py logic; the old file is kept untouched).
Same model classes/hyper-parameters (imported from baseline.make_models), same Top-K=50, 5-day horizon, 3y rolling window, retrain every 4 periods,
purge of 5 trading days, first test 2022-07-01, seed 42.  What changes (see p0lib docstring): P0-A..F.
--price-source: sina_adj (corrected: raw/Sina factor) | tx_qfq_full | tx_qfq_rowfilter | tx_qfq_drop3  (E1 variants use the OLD Tencent-qfq series for features/returns
but the same raw-price execution engine, so differences isolate the data-correction effect).
"""
import sys, json, argparse, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
V2 = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(V2)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline as B          # only make_models is reused (identical hyper-parameters); nothing is executed/overwritten on import
import p0lib as L
COSTS = [0.0, 0.001, 0.003, 0.006]
def cname(c): return f"c{int(round(c * 10000))}"
def tx_qfq_matrix(X, mode):
    panel = pd.read_parquet(V2 / "data" / "panel_1500.parquet", columns=["code", "date", "close", "board"])
    piv = panel.pivot(index="date", columns="code", values="close").reindex(index=X.cal, columns=X.codes)
    M = piv.values.astype(float)
    if mode == "rowfilter":
        rate = {"创业板": .20, "科创板": .20, "北交所": .30}
        for j, c in enumerate(X.codes):
            col = pd.Series(M[:, j]); has = col.notna().values
            s = col.dropna(); ret = s.pct_change(); lim = rate.get(X.u.board.iloc[j], .10)
            ld = X.u.list_date.iloc[j]; ipo = np.arange(len(s)) < 5 if (pd.notna(ld) and ld >= L.WIN_START) else np.zeros(len(s), bool)
            bad = ((ret.abs() > lim + 0.02) & ~ipo) | ((s <= 0) | ~np.isfinite(s))
            s[bad.values] = np.nan; M[np.where(has)[0], j] = s.values
    M[~(M > 0)] = np.nan; M[~np.isfinite(M)] = np.nan
    return M
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="fixed_sina_adj"); ap.add_argument("--price-source", default="sina_adj")
    ap.add_argument("--topk", type=int, default=50); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--retrain", type=int, default=4)
    ap.add_argument("--first-test", default="2022-07-01"); ap.add_argument("--max-signals", type=int, default=0)
    a = ap.parse_args(); out = L.P0 / "results" / a.tag; out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    codes = None
    if a.price_source == "tx_qfq_drop3":
        u = L.load_universe(); codes = [c for c in u.code if (L.RU / f"{c}.parquet").exists() and c not in ("600188", "600066", "000408")]
    X = L.build_data(codes=codes); L.limit_flags(X)
    if a.price_source == "sina_adj": ADJ = X.ADJ_unadj_based
    else: ADJ = tx_qfq_matrix(X, "rowfilter" if a.price_source == "tx_qfq_rowfilter" else "full")
    f, elig, ADJf = L.compute_features(X, ADJ); lab, y, yw = L.make_labels(ADJf, elig)
    Xq = L.rank_features(f, elig); T, N = elig.shape; cal = X.cal
    first = cal.searchsorted(pd.Timestamp(a.first_test)); sig_all = list(range(first, T - L.H - 1, L.H))
    models = B.make_models(a.seed); names = list(models); strat = names + ["Momentum20", "Reversal5"]
    fitted = {}; sig_used, pools, topk_sets, scores_ic = [], [], {n: [] for n in strat}, []
    m_tr = elig & np.isfinite(yw); isd = (X.u.status.values != "上市")   # P0-D diagnostic only: delisted-in-sample flag (future information, NOT used for selection)
    for ix, i in enumerate(sig_all):
        if a.max_signals and ix >= a.max_signals: break
        if ix % a.retrain == 0:
            Xt, yt = L.train_rows(Xq, yw, m_tr, cal, i)   # label end (t+6) must be < signal-5  => purge
            if len(Xt) < 20000: continue
            for n in names:
                m = models[n](); m.fit(Xt, yt); fitted[n] = m
        if not fitted: continue
        pool = np.where(elig[i])[0]
        if len(pool) < max(100, a.topk * 2): continue
        xs = Xq[i][pool]; sc = {n: fitted[n].predict(xs) for n in names}
        sc["Momentum20"] = f["r20"].values[i][pool]; sc["Reversal5"] = -f["r5"].values[i][pool]
        yy = y[i][pool]; fin = np.isfinite(yy); rec = {"signal_date": str(cal[i].date()), "n_pool": int(len(pool))}
        lp = lab[i][pool]; rec["pool_delist_share"] = float(isd[pool].mean()); rec["pool_label_listed"] = float(np.nanmean(lp[~isd[pool]])); rec["pool_label_delisted"] = float(np.nanmean(lp[isd[pool]])) if isd[pool].any() else np.nan
        for n in strat:
            s = np.nan_to_num(sc[n], nan=-1e9); top = pool[np.argsort(-s, kind="stable")[:a.topk]]; topk_sets[n].append(top)
            lt = lab[i][top]; rec[f"{n}_delist_share"] = float(isd[top].mean()); rec[f"{n}_label_listed"] = float(np.nanmean(lt[~isd[top]])) if (~isd[top]).any() else np.nan; rec[f"{n}_label_delisted"] = float(np.nanmean(lt[isd[top]])) if isd[top].any() else np.nan
            if fin.sum() > 5 and np.std(sc[n][fin]) > 0:
                rec[f"{n}_IC"] = float(np.corrcoef(sc[n][fin], yy[fin])[0, 1]); rec[f"{n}_RankIC"] = float(spearmanr(sc[n][fin], yy[fin]).statistic)
            else: rec[f"{n}_IC"] = rec[f"{n}_RankIC"] = np.nan
        sig_used.append(i); pools.append(pool); scores_ic.append(rec)
        if ix % 25 == 0: print(a.tag, "signal", ix, cal[i].date(), round(time.time() - t0), "s", flush=True)
    assert all(np.diff(sig_used) == L.H), "signals must be contiguous 5-day periods"
    ic = pd.DataFrame(scores_ic); res = pd.DataFrame({"signal_date": ic.signal_date, "order_date": [str(cal[i + 1].date()) for i in sig_used], "n_pool": ic.n_pool})
    logs, extras = [], {}
    for sname in strat + ["EqualWeight"]:
        tg = pools if sname == "EqualWeight" else topk_sets[sname]; cap = 1e9 if sname == "EqualWeight" else 1e7
        for c in COSTS:
            o, lg, ex = L.simulate(X, ADJf, sig_used, tg, c, capital=cap, log_name=(sname if (c == 0.003 and sname != "EqualWeight") else None))
            res[f"{sname}_ret_{cname(c)}"] = o.ret.values
            if c == 0.003:
                for k in ["replaced", "turnover_value", "cash_frac_post", "n_hold", "n_pending_exit", "buy_attempts", "buy_filled"]: res[f"{sname}_{k}"] = o[k].values
                for k in [k for k in o.columns if k.startswith("rej_")]: res[f"{sname}_{k}"] = o[k].fillna(0).values
                extras[sname] = ex
                if len(lg): logs.append(lg)
    res = pd.concat([res, ic.drop(columns=["signal_date", "n_pool"])], axis=1)
    res.to_csv(out / "period_returns.csv", index=False, encoding="utf-8-sig")
    if logs: pd.concat(logs).to_csv(out / "orders_primary_cost30bps.csv", index=False, encoding="utf-8-sig")
    pickle.dump(dict(signal_idx=sig_used, signal_dates=[str(cal[i].date()) for i in sig_used], codes=X.codes, pools=[[X.codes[j] for j in p] for p in pools],
                     topk={n: [[X.codes[j] for j in t] for t in topk_sets[n]] for n in strat}), open(out / "sets.pkl", "wb"))
    json.dump(dict(args=vars(a), n_stocks=N, n_periods=len(res), seconds=round(time.time() - t0), terminal_unsold=extras, limit_up_bars=int(X.lim_up.sum()), limit_dn_bars=int(X.lim_dn.sum()),
                   avg_pool=float(res.n_pool.mean()), unit_unverified=int(X.meta.unit_unverified.sum()), sources=X.meta.source.value_counts().to_dict()), open(out / "run_info.json", "w"), ensure_ascii=False, indent=1)
    print("done", a.tag, round(time.time() - t0), "s; periods", len(res))
if __name__ == "__main__": main()
