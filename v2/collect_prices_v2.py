"""P1 步骤4：分批采集日线(Parquet)，限速、重试、断点续传、本地缓存。
SH/SZ: 腾讯 stock_zh_a_hist_tx(前复权)；BJ: 新浪 stock_zh_a_daily(前复权)，因腾讯接口不支持北交所。
不绕过登录/权限；连续失败过多则保存进度并中止，不无限重试。
用法: python collect_prices_v2.py sample_1500.csv [--workers 3] [--pause 1.0] [--limit N]
"""
import sys, time, random, argparse, threading, datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, akshare as ak
V2 = Path(__file__).resolve().parent; DATA = V2 / "data"; RAWP = DATA / "raw_parquet"; RAWP.mkdir(exist_ok=True)
START, END = "20210101", "20261016"
lock = threading.Lock(); stop = threading.Event(); consec = {"n": 0}
def fetch(code, exch):
    if exch == "BJ":
        d = ak.stock_zh_a_daily(symbol="bj" + code, start_date=START, end_date=END, adjust="qfq")
        d = d.rename(columns={"volume": "volume_raw"}); d["source"] = "sina.stock_zh_a_daily"; d["volume_unit"] = "股"
        d = d[["date", "open", "high", "low", "close", "volume_raw", "amount", "turnover", "outstanding_share", "source", "volume_unit"]]
    else:
        d = ak.stock_zh_a_hist_tx(symbol=exch.lower() + code, start_date=START, end_date=END, adjust="qfq")
        d = d.rename(columns={"volume": "volume_raw"}); d["source"] = "tencent.stock_zh_a_hist_tx"
        d["volume_unit"] = "未验证(见质量报告)"; d["outstanding_share"] = float("nan")
        d = d[["date", "open", "high", "low", "close", "volume_raw", "amount", "turnover", "outstanding_share", "source", "volume_unit"]]
    return d
def work(row, pause, statuses, args):
    if stop.is_set(): return
    code, exch = row.code, row.exchange
    target = RAWP / f"{code}.parquet"
    t0 = time.time(); err = ""; d = None
    for a in range(3):
        try:
            d = fetch(code, exch)
            if d is None or d.empty: raise ValueError("接口返回空表")
            break
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:150]}"; d = None
            time.sleep(min(20, 2 ** (a + 1) + random.random()))
    now = dt.datetime.now().isoformat(timespec="seconds")
    if d is not None:
        d["date"] = pd.to_datetime(d["date"]); d = d.sort_values("date").drop_duplicates("date", keep="last")
        for c in ["open", "high", "low", "close", "volume_raw", "amount", "turnover", "outstanding_share"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d.insert(0, "code", code); d["exchange"] = exch; d["adjust"] = "qfq"; d["fetched_at"] = now
        d.to_parquet(target, index=False)
        st = dict(code=code, exchange=exch, status="ok", rows=len(d), first=str(d.date.min().date()), last=str(d.date.max().date()),
                  error="", fetched_at=now, secs=round(time.time() - t0, 1))
        with lock: consec["n"] = 0
    else:
        st = dict(code=code, exchange=exch, status="fail", rows=0, first="", last="", error=err, fetched_at=now, secs=round(time.time() - t0, 1))
        with lock:
            consec["n"] += 1
            if consec["n"] >= args.max_consec_fail: stop.set()
    with lock: statuses.append(st)
    time.sleep(pause + random.random() * 0.4)
def main():
    p = argparse.ArgumentParser(); p.add_argument("sample"); p.add_argument("--workers", type=int, default=3)
    p.add_argument("--pause", type=float, default=1.0); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-consec-fail", type=int, default=25); a = p.parse_args()
    if a.pause < 0.5: p.error("pause不得小于0.5秒")
    s = pd.read_csv(DATA / a.sample, dtype={"code": str})
    stat_file = DATA / "fetch_status.csv"
    done = set()
    if stat_file.exists():
        old = pd.read_csv(stat_file, dtype={"code": str}); done = set(old[old.status == "ok"].code)
        done = {c for c in done if (RAWP / f"{c}.parquet").exists()}
    todo = s[~s.code.isin(done)]
    if a.limit: todo = todo.head(a.limit)
    print(f"样本{len(s)}，已完成缓存{len(done)}，本次待采{len(todo)}", flush=True)
    statuses = []
    def flusher():
        last = 0
        while True:
            time.sleep(60)
            with lock:
                new = statuses[:]; statuses.clear()
            if new:
                pd.DataFrame(new).to_csv(stat_file, mode="a", header=not stat_file.exists(), index=False, encoding="utf-8-sig")
                print(dt.datetime.now().strftime("%H:%M:%S"), "已写入状态", len(new), flush=True)
            if flusher.quit: break
    flusher.quit = False
    th = threading.Thread(target=flusher, daemon=True); th.start()
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda r: work(r, a.pause, statuses, a), todo.itertuples()))
    flusher.quit = True; time.sleep(1)
    with lock: new = statuses[:]; statuses.clear()
    if new: pd.DataFrame(new).to_csv(stat_file, mode="a", header=not stat_file.exists(), index=False, encoding="utf-8-sig")
    allst = pd.read_csv(stat_file, dtype={"code": str})
    print("完成。ok =", int((allst.drop_duplicates("code", keep="last").status == "ok").sum()), "fail =", int((allst.drop_duplicates("code", keep="last").status == "fail").sum()))
    if stop.is_set(): print("[中止] 连续失败过多，进度已保存；请检查网络/接口后再续跑", flush=True); sys.exit(3)
if __name__ == "__main__": main()