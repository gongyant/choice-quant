"""For codes where Sina has NO factor table (e.g. 689009 CDR) derive a conservative factor table from dividend/bonus/transfer announcements (cash dividend per 10 shares, bonus+transfer per 10 shares).
Ex-reference price = (prev raw close - cash/10) / (1 + (bonus+transfer)/10). Validated against the raw OPEN of the ex-date (report). Rights issues are not covered (flagged as assumption)."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, akshare as ak
sys.path.insert(0, str(Path(__file__).resolve().parent)); import p0lib as L
rep = []
for p in sorted(L.FA.glob("*.parquet")):
    fac = pd.read_parquet(p)
    if len(fac) > 0 and not (fac.get("factor_status", pd.Series(["ok"])).iloc[0] != "ok"): continue
    c = p.stem; raw = L.load_raw(c); rs = raw.set_index("date")
    dv = ak.stock_history_dividend_detail(symbol=c, indicator="分红"); time.sleep(1); dv["ex"] = pd.to_datetime(dv["除权除息日"], errors="coerce"); dv = dv[dv.ex >= pd.Timestamp("2020-12-01")].sort_values("ex")
    ratios = []
    for _, r in dv.iterrows():
        if r.ex > L.LAST_TD or r.ex not in rs.index: continue
        prevP = rs.close[rs.index < r.ex].iloc[-1]; ref = (prevP - float(r["派息"]) / 10) / (1 + (float(r["送股"]) + float(r["转增"])) / 10)
        ratios.append((r.ex, ref / prevP)); rep.append(dict(code=c, ex_date=str(r.ex.date()), prev_close=prevP, announced_ref=ref, ratio=ref / prevP, raw_open=rs.open[r.ex], open_vs_ref=rs.open[r.ex] / ref - 1))
    f_after = 1.0; rows = []
    for ex, rt in sorted(ratios, reverse=True): rows.append((ex, f_after)); f_after = f_after / rt
    rows.append((pd.Timestamp("1900-01-01"), f_after)); t = pd.DataFrame(rows, columns=["date", "qfq_factor"]).sort_values("date"); t.insert(0, "code", c)
    t["source"] = "derived_from_sina_stock_history_dividend_detail (Sina qfq-factor not available)"; t["factor_status"] = "derived_from_announcements"; t["fetched_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    t.to_parquet(p, index=False); print(c, "derived", len(ratios), "events")
pd.DataFrame(rep).to_csv(L.P0 / "factor_fallback_report.csv", index=False, encoding="utf-8-sig"); print(pd.DataFrame(rep).to_string())
