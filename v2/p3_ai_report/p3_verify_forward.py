# -*- coding: utf-8 -*-
"""P3-5 forward verification (append-only). For every predictions_registry/pred_<date>.csv (sha checked against manifest.jsonl), once the merged prices
(data/merged, after p3_fetch.py + p3_merge.py) reach the horizon, compute per-stock realised returns with the P0 label convention:
  5d : adjusted close t+1 -> t+6 ; 20d : t+1 -> t+21 ; relative = minus the mean over the signal-day eligible pool.
Writes (append-only, never rewrites existing rows; a (signal_date, horizon) already present is skipped):
  verification/forward_stock_returns.csv  one row per (signal_date, horizon, code) with model scores/pctiles
  verification/forward_summary.csv        per (signal_date, horizon, model): RankIC, top-decile mean rel, Top5 v1/v2 mean ret, pool mean
  verification/verify_log.jsonl           every run (also when nothing has matured yet)
Usage: python p3_verify_forward.py   (no arguments; use run_weekly_verify.ps1 to fetch + merge + verify)."""
import sys, json, hashlib, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; REG = H / "predictions_registry"; MG = H / "data" / "merged"; VF = H / "verification"; VF.mkdir(exist_ok=True)
sys.path[:0] = [str(H.parent / "p0")]
import p0lib as L
L.RU, L.FA = MG / "raw_unadj_parquet", MG / "adj_factors"
HZ = {"5d": 6, "20d": 21}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def adj_close(codes, cal):
    M = np.full((len(cal), len(codes)), np.nan)
    for j, c in enumerate(codes):
        d = L.load_raw(c)
        if d is None: continue
        d = d.assign(date=pd.to_datetime(d.date)); d = d[d.volume.fillna(0) > 0]; fac = L.load_factor(c)
        if fac is not None: fac = fac.assign(date=pd.to_datetime(fac.date, errors="coerce")).dropna(subset=["date"])
        a = d.close.values * L.factor_on_dates(d.date, fac); s = pd.Series(a, index=d.date).reindex(cal); M[:, j] = s.ffill().values
    return M
def existing(f, keys):
    if not f.exists(): return set()
    e = pd.read_csv(f, dtype=str, encoding="utf-8-sig", usecols=keys); return set(map(tuple, e.values))
def append(f, df):
    df.to_csv(f, mode="a", header=not f.exists(), index=False, encoding="utf-8-sig")
def main():
    run = dict(run_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"), done=[], pending=[], errors=[])
    man = {json.loads(l)["file"]: json.loads(l) for l in open(REG / "manifest.jsonl", encoding="utf-8")}
    cal = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(MG / "trade_calendar.csv").iloc[:, 0])); run["data_last_date"] = str(cal[-1].date())
    have = existing(VF / "forward_summary.csv", ["signal_date", "horizon"])
    for f in sorted(REG.glob("pred_*.csv")):
        if f.name not in man or man[f.name]["sha256"] != sha(f): run["errors"].append(f"{f.name}: sha mismatch / not in manifest -> skipped"); continue
        P = pd.read_csv(f, dtype={"code": str}, encoding="utf-8-sig"); sd = pd.Timestamp(P.signal_date.iloc[0]); i = cal.get_loc(sd)
        todo = [h for h, k in HZ.items() if (str(sd.date()), h) not in have]
        ready = [h for h in todo if i + HZ[h] <= len(cal) - 1]
        for h in todo:
            if h not in ready: run["pending"].append(dict(file=f.name, horizon=h, needs_date_index=i + HZ[h], have_trading_days_after=len(cal) - 1 - i))
        if not ready: continue
        codes = sorted(P.code.unique()); A = adj_close(codes, cal); col = {c: j for j, c in enumerate(codes)}
        el = P[P.model == "HGB"].set_index("code").eligible_asof.astype(str).eq("True")
        rk = H / f"latest_research_ranking_{sd:%Y%m%d}_v2.csv"; picks = {}
        if rk.exists():
            o = pd.read_csv(rk, dtype={"code": str}, encoding="utf-8-sig"); picks = {"Top5_v1": set(o.code[o.top5]), "Top5_v2": set(o.code[o.top5_v2]), "Avoid5": set(o.code[o.avoid5])}
        for h in ready:
            k = HZ[h]; ret = pd.Series({c: A[i + k, col[c]] / A[i + 1, col[c]] - 1 for c in codes}); pool = ret[el.reindex(ret.index).fillna(False)]; rel = ret - pool.mean()
            W = P.pivot(index="code", columns="model", values="pctile"); S = pd.DataFrame(dict(signal_date=str(sd.date()), horizon=h, code=codes, ret=ret.reindex(codes).values, rel_to_pool=rel.reindex(codes).values,
                 eligible=el.reindex(codes).values, entry_date=str(cal[i + 1].date()), exit_date=str(cal[i + k].date()))).join(W.add_prefix("pctile_"), on="code")
            S["verified_at"] = run["run_at"]; append(VF / "forward_stock_returns.csv", S)
            rows = []; E = S[S.eligible.astype(bool)]
            for m in W.columns:
                ok = E[[f"pctile_{m}", "rel_to_pool"]].dropna(); top = ok[ok[f"pctile_{m}"] >= ok[f"pctile_{m}"].quantile(0.9)]
                rows.append(dict(signal_date=str(sd.date()), horizon=h, group=m, n=len(ok), rank_ic=ok[f"pctile_{m}"].corr(ok.rel_to_pool, method="spearman"), mean_rel=top.rel_to_pool.mean(), pool_mean_ret=pool.mean()))
            for g, cs in picks.items():
                x = S[S.code.isin(cs)]; rows.append(dict(signal_date=str(sd.date()), horizon=h, group=g, n=len(x), rank_ic=np.nan, mean_rel=x.rel_to_pool.mean(), mean_ret=x.ret.mean(), pool_mean_ret=pool.mean(), codes=" ".join(sorted(cs))))
            R = pd.DataFrame(rows); R["verified_at"] = run["run_at"]; R["data_last_date"] = run["data_last_date"]; append(VF / "forward_summary.csv", R); run["done"].append(dict(file=f.name, horizon=h))
    with open(VF / "verify_log.jsonl", "a", encoding="utf-8") as fh: fh.write(json.dumps(run, ensure_ascii=False) + "\n")
    print(json.dumps(run, ensure_ascii=False, indent=1))
if __name__ == "__main__": main()