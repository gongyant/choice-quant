# -*- coding: utf-8 -*-
"""P3-1 incremental daily bars (UNADJUSTED, same sources as P0 collect_unadj.py) + Sina qfq-factor tables, for the 1498 P0 stocks.
Writes ONLY under p3_ai_report/data/: incr_raw/<code>.parquet (bars 2026-09-01..END, overlap used to verify vs P0 raw), adj_factors_latest/<code>.parquet, fetch_status.csv.
P0 raw files are read-only. Resumable; bounded retries; stops after 40 consecutive failures."""
import sys, time, threading, datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import socket; socket.setdefaulttimeout(12)   # akshare passes no timeout; hung connections through the local proxy were the bottleneck
import pandas as pd, akshare as ak
H = Path(__file__).resolve().parent; V2 = H.parent; P0R = V2 / "p0" / "data" / "raw_unadj_parquet"
OUT = H / "data" / "incr_raw"; FAO = H / "data" / "adj_factors_latest"; ST = H / "data" / f"fetch_status_{sys.argv[2] if len(sys.argv) > 2 else 0}.csv"
for p in (OUT, FAO): p.mkdir(parents=True, exist_ok=True)
START, END = "20260901", sys.argv[1] if len(sys.argv) > 1 else "20261021"
U = pd.read_csv(V2 / "data" / "sample_1500.csv", dtype=str).set_index("code")
lock = threading.Lock(); stop = threading.Event(); consec = {"n": 0}; rows = []
def retry(fn, n=4):
    last = None
    for a in range(n):
        try: return fn()
        except ValueError as e:
            if "not available" in str(e): raise
            last = e
        except Exception as e: last = e
        time.sleep(1.5 * (a + 1))
    raise last
def one(code):
    if stop.is_set(): return
    if (OUT / f"{code}.parquet").exists() and (FAO / f"{code}.parquet").exists(): return
    ex = U.loc[code, "exchange"]; sym = ex.lower() + code
    old = pd.read_parquet(P0R / f"{code}.parquet", columns=["source"]).source.iloc[0]; err = ""; src = old; n = 0; fst = "ok"
    try:
        if ex == "BJ":   # Tencent cannot serve BJ: Sina (P0 primary source) with retries
            raw = retry(lambda: ak.stock_zh_a_daily(symbol=sym, start_date=START, end_date=END, adjust="")); src = old
        else:            # P0's documented fallback source (Tencent raw). Sina raw was unreliable through the local proxy on 2026-10-21 (SSL EOF).
            raw = retry(lambda: ak.stock_zh_a_hist_tx(symbol=sym, start_date=START, end_date=END, adjust="")); src = "tencent.stock_zh_a_hist_tx(adjust='') [p3 incremental]"
        raw = raw.copy(); raw["date"] = pd.to_datetime(raw["date"]); raw["code"] = code; raw["fetched_source"] = src; raw["source"] = old; raw["fetched_at"] = dt.datetime.now().isoformat(timespec="seconds")
        for c in ["open", "close", "high", "low", "volume", "turnover", "amount"]:
            if c not in raw: raw[c] = float("nan")
        raw = raw[["code", "date", "open", "close", "high", "low", "volume", "turnover", "amount", "source", "fetched_source", "fetched_at"]]; n = len(raw)
        try: fac = retry(lambda: ak.stock_zh_a_daily(symbol=sym, start_date="20210101", end_date=END, adjust="qfq-factor"))
        except ValueError as e:
            if "not available" in str(e): fac = pd.DataFrame({"date": [], "qfq_factor": []}); fst = "sina_factor_not_available"
            else: raise
        fac = fac.copy(); fac["date"] = pd.to_datetime(fac["date"]); fac["qfq_factor"] = pd.to_numeric(fac["qfq_factor"], errors="coerce")
        raw.to_parquet(OUT / f"{code}.parquet", index=False); fac.to_parquet(FAO / f"{code}.parquet", index=False); status = "ok"
        with lock: consec["n"] = 0
    except Exception as e:
        status = "fail"; err = f"{type(e).__name__}: {str(e)[:160]}"
        with lock:
            consec["n"] += 1
            if consec["n"] >= 40: stop.set()
    with lock: rows.append(dict(code=code, exchange=ex, source=src, status=status, rows=n, factor_status=fst, error=err, fetched_at=dt.datetime.now().isoformat(timespec="seconds")))
    time.sleep(0.15)
if __name__ == "__main__":
    codes = sorted(p.stem for p in P0R.glob("*.parquet")); SH = int(sys.argv[2]) if len(sys.argv) > 2 else 0; NS = int(sys.argv[3]) if len(sys.argv) > 3 else 1; codes = codes[SH::NS]; t0 = time.time(); print("codes", len(codes), "END", END, flush=True)
    with ThreadPoolExecutor(1) as ex:
        for k, _ in enumerate(ex.map(one, codes)):
            if k % 50 == 0: print(f"{dt.datetime.now():%H:%M:%S} {k}/{len(codes)} done files={len(list(OUT.glob('*.parquet')))} consec_fail={consec['n']}", flush=True)
    old = pd.read_csv(ST, dtype={"code": str}) if ST.exists() else pd.DataFrame()
    st = pd.concat([old, pd.DataFrame(rows)]).drop_duplicates("code", keep="last") if rows else old; st.to_csv(ST, index=False, encoding="utf-8-sig")
    print("finished", round(time.time() - t0), "s; ok files", len(list(OUT.glob("*.parquet"))), "stopped" if stop.is_set() else "", flush=True)