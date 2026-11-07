"""Fetch raw+factor for pre-specified stocks that are NOT in the 1500 sample (600060). Same sources/format as p0/collect_unadj.py. Output only under p1_dl/data_extra."""
import sys, time
from pathlib import Path
import pandas as pd, akshare as ak
H = Path(__file__).resolve().parent; RAW = H/"data_extra"/"raw"; ADJ = H/"data_extra"/"adj"
for code, exch in [("600060", "SH")]:
    s = exch.lower()+code; src = "sina"; now = pd.Timestamp.now().isoformat(timespec="seconds")
    try: raw = ak.stock_zh_a_daily(symbol=s, start_date="20210101", end_date="20261016", adjust="")
    except Exception as e: print("sina raw failed", type(e).__name__, e); raw = ak.stock_zh_a_hist_tx(symbol=s, start_date="20210101", end_date="20261016", adjust=""); src = "tencent"
    time.sleep(1.0)
    fac = ak.stock_zh_a_daily(symbol=s, start_date="20210101", end_date="20261016", adjust="qfq-factor")
    raw["date"] = pd.to_datetime(raw["date"]); raw = raw.sort_values("date").drop_duplicates("date", keep="last")
    for c in raw.columns:
        if c != "date": raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw.insert(0, "code", code); raw["source"] = "sina.stock_zh_a_daily(adjust='')" if src == "sina" else "tencent.stock_zh_a_hist_tx(adjust='') [sina raw unavailable]"; raw["fetched_at"] = now
    raw.to_parquet(RAW/f"{code}.parquet", index=False)
    fac = fac.copy(); fac["date"] = pd.to_datetime(fac["date"]); fac["qfq_factor"] = pd.to_numeric(fac["qfq_factor"], errors="coerce")
    fac.insert(0, "code", code); fac["source"] = "sina.stock_zh_a_daily(adjust='qfq-factor')"; fac["factor_status"] = "ok"; fac["fetched_at"] = now
    fac.to_parquet(ADJ/f"{code}.parquet", index=False)
    # independent Tencent raw cross-check
    time.sleep(1.0)
    tx = ak.stock_zh_a_hist_tx(symbol=s, start_date="20210101", end_date="20261016", adjust=""); tx["date"] = pd.to_datetime(tx["date"])
    m = raw.merge(tx, on="date", suffixes=("", "_tx")); print(code, src, "rows", len(raw), raw.date.min().date(), raw.date.max().date(), "factor rows", len(fac), "tx rows", len(tx), "common", len(m), "max|close diff|", float((m.close-m.close_tx).abs().max()))
    tx.to_parquet(H/"data_extra"/f"{code}_tencent_raw.parquet", index=False)
    print(raw.tail(3).to_string()); print(fac.sort_values("date").tail(4).to_string())
