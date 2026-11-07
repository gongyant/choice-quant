"""P0-C: download UNADJUSTED daily bars + adjustment-factor tables (resumable, throttled, cached).
Primary source : Sina  ak.stock_zh_a_daily(adjust="")            -> data/raw_unadj_parquet/<code>.parquet
Factor table   : Sina  ak.stock_zh_a_daily(adjust="qfq-factor")  -> data/adj_factors/<code>.parquet  (ex-date based corporate-action factors)
Cross-check    : Tencent ak.stock_zh_a_hist_tx(adjust="")        -> data/raw_unadj_tx/<code>.parquet (SH/SZ only, phase B)
Stops after N consecutive failures (no infinite retry); progress is the set of cached files + data/unadj_fetch_status.csv.
usage: python collect_unadj.py [--phase A|B] [--workers 2] [--pause 0.6]
"""
import sys, time, random, argparse, threading, datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, akshare as ak
V2 = Path(__file__).resolve().parents[1]; P0 = V2 / "p0"; D = P0 / "data"
RU = D / "raw_unadj_parquet"; FA = D / "adj_factors"; TX = D / "raw_unadj_tx"
for p in (RU, FA, TX): p.mkdir(parents=True, exist_ok=True)
START, END = "20210101", "20261016"
lock = threading.Lock(); stop = threading.Event(); consec = {"n": 0}; stat_rows = []
PRIORITY = ["600188", "600066", "000408", "600096", "000933", "601666", "689009", "600317", "920196"]
def sym(code, exch): return exch.lower() + code
def get_sina(code, exch):
    """Sina raw bars; if Sina cannot serve the symbol (JSONDecodeError seen for 000004/000005/689009/600317) fall back to Tencent raw (SH/SZ).
    Factor table: Sina qfq-factor. ValueError('...not available') = Sina has no factor table for the symbol -> empty table, flagged."""
    src = "sina"
    try:
        raw = ak.stock_zh_a_daily(symbol=sym(code, exch), start_date=START, end_date=END, adjust="")
    except Exception as e:
        if exch == "BJ": raise
        raw = ak.stock_zh_a_hist_tx(symbol=sym(code, exch), start_date=START, end_date=END, adjust="").rename(columns={"volume": "volume"})
        src = "tencent"
    fstat = "ok"
    try:
        fac = ak.stock_zh_a_daily(symbol=sym(code, exch), start_date=START, end_date=END, adjust="qfq-factor")
    except ValueError as e:
        if "not available" in str(e): fac = pd.DataFrame({"date": [], "qfq_factor": []}); fstat = "sina_factor_not_available"
        else: raise
    return raw, fac, src, fstat
def retry(fn, n=3):
    err = ""
    for a in range(n):
        try:
            r = fn()
            return r, ""
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:150]}"; time.sleep(min(15, 2 ** (a + 1) + random.random()))
    return None, err
def work(row, phase, pause, maxfail):
    if stop.is_set(): return
    code, exch = row.code, row.exchange
    if phase == "B" and (TX / f"{code}.parquet").exists(): return
    if phase == "A" and (RU / f"{code}.parquet").exists() and (FA / f"{code}.parquet").exists(): return   # another worker already cached it
    now = dt.datetime.now().isoformat(timespec="seconds"); st = dict(code=code, exchange=exch, phase=phase, status="ok", rows=0, error="", fetched_at=now)
    if phase == "A":
        res, err = retry(lambda: get_sina(code, exch))
        if res is None or res[0] is None or len(res[0]) == 0:
            st.update(status="fail", error=err or "empty")
        else:
            raw, fac, src, fstat = res
            raw["date"] = pd.to_datetime(raw["date"]); raw = raw.sort_values("date").drop_duplicates("date", keep="last")
            for c in raw.columns:
                if c != "date": raw[c] = pd.to_numeric(raw[c], errors="coerce")
            raw.insert(0, "code", code); raw["source"] = "sina.stock_zh_a_daily(adjust='')" if src == "sina" else "tencent.stock_zh_a_hist_tx(adjust='') [sina raw unavailable]"; raw["fetched_at"] = now
            raw.to_parquet(RU / f"{code}.parquet", index=False)
            fac = fac.copy(); fac["date"] = pd.to_datetime(fac["date"]); fac["qfq_factor"] = pd.to_numeric(fac["qfq_factor"], errors="coerce")
            fac.insert(0, "code", code); fac["source"] = "sina.stock_zh_a_daily(adjust='qfq-factor')"; fac["factor_status"] = fstat; fac["fetched_at"] = now
            fac.to_parquet(FA / f"{code}.parquet", index=False); st["rows"] = len(raw); st["error"] = f"raw_src={src};factor={fstat}"
    else:
        if exch == "BJ": return
        r, err = retry(lambda: ak.stock_zh_a_hist_tx(symbol=sym(code, exch), start_date=START, end_date=END, adjust=""))
        if r is None or len(r) == 0: st.update(status="fail", error=err or "empty")
        else:
            r = r.rename(columns={"volume": "volume_raw"}); r["date"] = pd.to_datetime(r["date"])
            for c in r.columns:
                if c != "date": r[c] = pd.to_numeric(r[c], errors="coerce")
            r.insert(0, "code", code); r["source"] = "tencent.stock_zh_a_hist_tx(adjust='')"; r["fetched_at"] = now
            r.to_parquet(TX / f"{code}.parquet", index=False); st["rows"] = len(r)
    with lock:
        stat_rows.append(st)
        if st["status"] == "fail":
            consec["n"] += 1
            if consec["n"] >= maxfail: stop.set()
        else: consec["n"] = 0
    time.sleep(pause + random.random() * 0.3)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="A"); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pause", type=float, default=0.6); ap.add_argument("--max-consec-fail", type=int, default=25); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    s = pd.read_csv(V2 / "data" / "sample_1500.csv", dtype={"code": str})
    have = {p.stem for p in (V2 / "data" / "raw_parquet").glob("*.parquet")}
    s = s[s.code.isin(have)].copy(); s["pri"] = s.code.map(lambda c: PRIORITY.index(c) if c in PRIORITY else 99); s = s.sort_values(["pri", "code"])
    cache = RU if a.phase == "A" else TX
    done = {p.stem for p in cache.glob("*.parquet")}
    if a.phase == "A": done &= {p.stem for p in FA.glob("*.parquet")}
    todo = s[~s.code.isin(done)]
    if a.reverse: todo = todo.iloc[::-1]
    if a.limit: todo = todo.head(a.limit)
    print(f"phase {a.phase}: universe {len(s)}, cached {len(done)}, todo {len(todo)}", flush=True)
    sf = D / "unadj_fetch_status.csv"
    def flush():
        with lock: new = stat_rows[:]; stat_rows.clear()
        if new: pd.DataFrame(new).to_csv(sf, mode="a", header=not sf.exists(), index=False, encoding="utf-8-sig"); print(dt.datetime.now().strftime("%H:%M:%S"), "status rows", len(new), flush=True)
    def fl():
        while True:
            time.sleep(45); flush()
    threading.Thread(target=fl, daemon=True).start()
    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(lambda r: work(r, a.phase, a.pause, a.max_consec_fail), todo.itertuples()))
    flush()
    allst = pd.read_csv(sf, dtype={"code": str}); last = allst[allst.phase == a.phase].drop_duplicates("code", keep="last")
    print("done. ok =", int((last.status == "ok").sum()), "fail =", int((last.status == "fail").sum()))
    if stop.is_set(): print("[ABORT] too many consecutive failures; progress saved", flush=True); sys.exit(3)
if __name__ == "__main__": main()
