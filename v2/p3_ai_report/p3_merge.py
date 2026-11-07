# -*- coding: utf-8 -*-
"""P3-1b: merge P0 raw (read-only, <=2026-10-16) + incremental bars (>2026-10-16) into p3_ai_report/data/merged/, harmonising units on the overlap
(2026-09-01..10-16) and writing data_quality_latest.{csv,md}. Original P0 files are never written."""
import sys, json, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; V2 = H.parent; P0D = V2 / "p0" / "data"; INC = H / "data" / "incr_raw"; FAN = H / "data" / "adj_factors_latest"
MG = H / "data" / "merged"; MRU, MFA = MG / "raw_unadj_parquet", MG / "adj_factors"
for p in (MRU, MFA): p.mkdir(parents=True, exist_ok=True)
OLD_END = pd.Timestamp("2026-10-16")
def snap(r):
    if not np.isfinite(r) or r <= 0: return np.nan
    k = np.round(np.log10(r)); return float(10 ** k) if abs(np.log10(r) - k) < 0.05 else np.nan
def main(END):
    END = pd.Timestamp(END); cal0 = pd.read_csv(P0D / "trade_calendar.csv", parse_dates=["date"]).date
    try:
        import akshare as ak; c = pd.to_datetime(ak.tool_trade_date_hist_sina().trade_date); new = c[(c > OLD_END) & (c <= END)].sort_values(); cal_src = "akshare.tool_trade_date_hist_sina"
    except Exception as e:
        new = None; cal_src = f"observed dates (calendar API failed: {type(e).__name__})"
    rows = []; obs = []
    codes = sorted(p.stem for p in (P0D / "raw_unadj_parquet").glob("*.parquet"))
    for c in codes:
        o = pd.read_parquet(P0D / "raw_unadj_parquet" / f"{c}.parquet").sort_values("date").drop_duplicates("date", keep="last")
        r = dict(code=c, p0_last_date=o.date.max().date(), orig_source=o.source.iloc[0], fetched=False)
        f = INC / f"{c}.parquet"
        if f.exists():
            n = pd.read_parquet(f).sort_values("date").drop_duplicates("date", keep="last"); r["fetched"] = True; r["fetched_source"] = n.fetched_source.iloc[0] if len(n) else ""
            ov = o.merge(n, on="date", suffixes=("", "_n")); ov = ov[ov.date >= "2026-09-01"]; r["n_overlap"] = len(ov)
            vm = 1.0; tmode = "as_fetched"
            if len(ov) >= 5:
                r["overlap_close_maxabs"] = float((ov.close - ov.close_n).abs().max()); r["overlap_amount_maxrel"] = float(((ov.amount - ov.amount_n).abs() / ov.amount.abs().clip(lower=1)).max())
                okv = (ov.volume > 0) & (ov.volume_n > 0); vm = snap(float(np.median(ov.volume[okv] / ov.volume_n[okv]))) if okv.sum() >= 3 else np.nan
                if not np.isfinite(vm): vm = 1.0; r["unit_warning"] = True
                if str(r.get("fetched_source", "")).startswith("tencent") and not r["orig_source"].startswith("tencent"):
                    okt = okv & (ov.turnover > 0); osh = float(np.median(ov.volume[okt] / ov.turnover[okt])) if okt.sum() >= 3 else np.nan; tmode = "volume/outstanding(from P0 Sina overlap)"
                    r["outstanding_est"] = osh
            r["volume_mult"] = vm
            nn = n[n.date > OLD_END].copy(); nn["volume"] = nn.volume * vm
            if tmode != "as_fetched": nn["turnover"] = nn.volume / r["outstanding_est"] if np.isfinite(r.get("outstanding_est", np.nan)) else np.nan
            r["turnover_mode"] = tmode; nn["source"] = r["orig_source"]; nn["p3_appended"] = True
            o["p3_appended"] = False; m = pd.concat([o, nn[[k for k in o.columns if k in nn.columns] + (["fetched_source"] if "fetched_source" in nn else [])]], ignore_index=True)
            obs += list(nn.date)
            fa = pd.read_parquet(FAN / f"{c}.parquet") if (FAN / f"{c}.parquet").exists() else None; r["factor_source"] = "latest_sina" if fa is not None and len(fa) else "p0_original"
        else:
            m = o.copy(); m["p3_appended"] = False; fa = None; r["factor_source"] = "p0_original"
        if fa is None or not len(fa): fa = pd.read_parquet(P0D / "adj_factors" / f"{c}.parquet") if (P0D / "adj_factors" / f"{c}.parquet").exists() else None
        m.to_parquet(MRU / f"{c}.parquet", index=False)
        if fa is not None: fa.to_parquet(MFA / f"{c}.parquet", index=False)
        r["last_date"] = m.date.max().date(); r["n_new_rows"] = int(m.p3_appended.sum()); rows.append(r)
    if new is None: vc = pd.Series(obs).value_counts(); new = pd.DatetimeIndex(sorted(vc[vc >= 300].index))
    cal = pd.DatetimeIndex(sorted(set(cal0) | set(pd.DatetimeIndex(new)))); pd.DataFrame({"date": cal}).to_csv(MG / "trade_calendar.csv", index=False)
    S = pd.DataFrame(rows); S.to_csv(H / "data" / "merge_log.csv", index=False, encoding="utf-8-sig")
    info = dict(END=str(END.date()), calendar_source=cal_src, new_trade_dates=[str(d.date()) for d in new], n_codes=len(codes), n_fetched=int(S.fetched.sum()), n_with_new_rows=int((S.n_new_rows > 0).sum()),
                overlap_close_maxabs_max=float(S.get("overlap_close_maxabs", pd.Series([np.nan])).max()), overlap_amount_maxrel_p99=float(S.get("overlap_amount_maxrel", pd.Series([np.nan])).quantile(0.99)),
                volume_mult_counts={str(k): int(v) for k, v in S.volume_mult.value_counts(dropna=False).items()} if "volume_mult" in S else {}, merged_at=dt.datetime.now().isoformat(timespec="seconds"))
    json.dump(info, open(H / "data" / "merge_info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(json.dumps(info, ensure_ascii=False))
if __name__ == "__main__": main(sys.argv[1] if len(sys.argv) > 1 else "2026-10-21")