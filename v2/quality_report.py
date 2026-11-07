"""P1 步骤5：逐股票数据质量审计 + 合并面板(Parquet)。
输出: reports/quality_per_stock.csv, reports/quality_summary.json, reports/quality_report.md, data/panel_<n>.parquet
"""
import sys, json, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, akshare as ak
V2 = Path(__file__).resolve().parent; DATA = V2 / "data"; REP = V2 / "reports"; RAWP = DATA / "raw_parquet"
SAMPLE = sys.argv[1] if len(sys.argv) > 1 else "sample_1500.csv"
LAST_TD = pd.Timestamp("2026-10-16"); WIN_START = pd.Timestamp("2021-01-01")
cal = ak.tool_trade_date_hist_sina(); cal = pd.to_datetime(cal.trade_date)
cal = cal[(cal >= WIN_START) & (cal <= LAST_TD)].reset_index(drop=True)
s = pd.read_csv(DATA / SAMPLE, dtype={"code": str}, parse_dates=["list_date", "delist_date"])
st = pd.read_csv(DATA / "fetch_status.csv", dtype={"code": str}).drop_duplicates("code", keep="last").set_index("code")
def limit_pct(board, name, date):
    # 简化涨跌停幅度：创业板2020-08-24起20%，科创板20%，北交所30%，ST 5%(主板)，其余10%。ST按当前名称判断，非时点，仅用于异常标记。
    if board == "北交所": return 0.30
    if board == "科创板": return 0.20
    if board == "创业板": return 0.20 if date >= pd.Timestamp("2020-08-24") else 0.10
    return 0.10
parts, rows = [], []
for r in s.itertuples():
    f = RAWP / f"{r.code}.parquet"
    q = dict(code=r.code, name=r.name, exchange=r.exchange, board=r.board, status=r.status, industry_l1=r.industry_l1,
             sample_group=r.sample_group, fetched=f.exists())
    if not f.exists():
        q["fetch_status"] = st.status.get(r.code, "未采集"); q["error"] = st.error.get(r.code, ""); rows.append(q); continue
    d = pd.read_parquet(f).sort_values("date").reset_index(drop=True)
    q.update(rows=len(d), first=str(d.date.min().date()), last=str(d.date.max().date()), source=d.source.iloc[0])
    q["dup_dates"] = int(d.date.duplicated().sum())
    start = max(WIN_START, r.list_date) if pd.notna(r.list_date) else WIN_START
    end = min(LAST_TD, r.delist_date) if pd.notna(r.delist_date) else LAST_TD
    exp = cal[(cal >= start) & (cal <= end)]
    have = set(d.date)
    q["expected_days"] = int(len(exp)); q["missing_days_vs_calendar"] = int(sum(1 for x in exp if x not in have))
    q["coverage"] = round(1 - q["missing_days_vs_calendar"] / max(1, len(exp)), 4)
    q["late_start_days"] = int((d.date.min() - exp.min()).days) if len(exp) else np.nan
    q["stale_end"] = bool(d.date.max() < min(LAST_TD, end) - pd.Timedelta(days=5))
    px = d[["open", "high", "low", "close"]]
    q["nan_price"] = int(px.isna().any(axis=1).sum()); q["nonpos_price"] = int((px <= 0).any(axis=1).sum())
    hi_bad = d.high < d[["open", "close", "low"]].max(axis=1) - 1e-6; lo_bad = d.low > d[["open", "close", "high"]].min(axis=1) + 1e-6
    q["ohlc_inconsistent"] = int((hi_bad | lo_bad).sum())
    q["zero_volume_days"] = int((d.volume_raw.fillna(0) <= 0).sum())
    ret = d.close.pct_change()
    lim = np.array([limit_pct(r.board, r.name, x) for x in d.date])
    ipo = np.arange(len(d)) < 5 if (pd.notna(r.list_date) and r.list_date >= WIN_START) else np.zeros(len(d), bool)
    ext = (ret.abs() > lim + 0.02) & ~ipo
    q["extreme_returns"] = int(ext.sum()); q["max_abs_ret"] = round(float(ret.abs().max()), 4) if len(d) > 1 else np.nan
    # 成交量单位验证：逐年用 amount/(close*volume) 中位数反推(前复权会让早年比值偏离，所以只吸附到10的幂并给出容差)
    ok_rows = d[(d.volume_raw > 0) & (d.amount > 0) & (d.close > 0)].copy()
    ok_rows["ratio"] = ok_rows.amount / (ok_rows.close * ok_rows.volume_raw)
    yr_mult = {}
    for y, g in ok_rows.groupby(ok_rows.date.dt.year):
        if len(g) < 10: continue
        rr = float(g.ratio.median()); k = round(np.log10(rr))
        yr_mult[y] = 10.0 ** k if abs(np.log10(rr) - k) < 0.2 else np.nan
    known = {v for v in yr_mult.values() if v == v}
    tail = ok_rows.tail(60); ratio = float((tail.ratio).median()) if len(tail) >= 10 else np.nan
    q["amount_per_close_vol"] = round(ratio, 3) if ratio == ratio else np.nan
    q["volume_mult_by_year"] = json.dumps({int(k): (None if v != v else v) for k, v in yr_mult.items()})
    q["unit_changes_over_time"] = len(known) > 1
    mult = (list(known)[0] if len(known) == 1 else np.nan)
    q["volume_to_shares_multiplier"] = mult
    q["volume_unit_unverified_years"] = int(sum(1 for v in yr_mult.values() if v != v))
    q["turnover_median"] = round(float(d.turnover.median()), 5) if d.turnover.notna().any() else np.nan
    q["amount_median_cny"] = float(d.amount.median()) if d.amount.notna().any() else np.nan
    flags = []
    if q["dup_dates"]: flags.append("重复日期")
    if q["nan_price"] or q["nonpos_price"]: flags.append("无效价格")
    if q["ohlc_inconsistent"]: flags.append("OHLC不一致")
    if q["coverage"] < 0.95: flags.append("覆盖率<95%")
    if q["extreme_returns"]: flags.append("超涨跌幅限制收益")
    if q["stale_end"]: flags.append("数据提前结束(停牌/退市/断更)")
    if mult != mult: flags.append("成交量单位未能验证或随时间变化")
    if mult == mult and mult != 1.0 and mult != 100.0: flags.append("成交量单位异常(非股/手)")
    q["flags"] = "|".join(flags); q["fetch_status"] = "ok"; rows.append(q)
    d["volume_shares"] = d.volume_raw * (d.date.dt.year.map(yr_mult).fillna(mult) if len(known) else np.nan)
    d["board"] = r.board; d["industry_l1"] = r.industry_l1
    parts.append(d)
qd = pd.DataFrame(rows); qd.to_csv(REP / "quality_per_stock.csv", index=False, encoding="utf-8-sig")
panel = pd.concat(parts, ignore_index=True)
out = DATA / f"panel_{len(s)}.parquet"; panel.to_parquet(out, index=False)
ok = qd[qd.fetched == True]
summ = {"generated_at": dt.datetime.now().isoformat(timespec="seconds"), "sample_size": int(len(s)), "fetched_ok": int(len(ok)),
        "fetch_failed_or_missing": int(len(qd) - len(ok)), "panel_rows": int(len(panel)), "panel_codes": int(panel.code.nunique()),
        "date_min": str(panel.date.min().date()), "date_max": str(panel.date.max().date()), "last_trading_day_expected": str(LAST_TD.date()),
        "coverage_mean": round(float(ok.coverage.mean()), 4), "coverage_lt95": int((ok.coverage < 0.95).sum()),
        "stocks_with_dup_dates": int((ok.dup_dates > 0).sum()), "stocks_with_invalid_price": int(((ok.nan_price > 0) | (ok.nonpos_price > 0)).sum()),
        "stocks_with_ohlc_inconsistent": int((ok.ohlc_inconsistent > 0).sum()),
        "stocks_with_extreme_returns": int((ok.extreme_returns > 0).sum()), "stocks_stale_end": int(ok.stale_end.sum()),
        "volume_unit_unverified": int(ok.volume_to_shares_multiplier.isna().sum()), "unit_changes_over_time": int(ok.unit_changes_over_time.sum()),
        "volume_multiplier_counts": {str(k): int(v) for k, v in ok.groupby(ok.volume_to_shares_multiplier.fillna(-1)).size().items()},
        "volume_multiplier_by_exchange": {f"{e}|{m}": int(n) for (e, m), n in ok.groupby(["exchange", ok.volume_to_shares_multiplier.fillna(-1)]).size().items()},
        "parquet_dir_MB": round(sum(p.stat().st_size for p in RAWP.glob("*.parquet")) / 1e6, 1), "panel_MB": round(out.stat().st_size / 1e6, 1),
        "by_board": ok.board.value_counts().to_dict(), "flag_counts": pd.Series("|".join(ok["flags"].fillna("")).split("|")).replace("", np.nan).dropna().value_counts().to_dict()}
with open(REP / "quality_summary.json", "w", encoding="utf-8") as f: json.dump(summ, f, ensure_ascii=False, indent=2, default=int)
print(json.dumps(summ, ensure_ascii=False, indent=1, default=int))