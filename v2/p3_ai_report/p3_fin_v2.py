# -*- coding: utf-8 -*-
"""P3-6a: latest reported financial key numbers for the Top5 / avoid5 names from Sina Finance (akshare.stock_financial_abstract). Missing -> '未取得'."""
import sys, json, socket, datetime as dt
socket.setdefaulttimeout(20)
from pathlib import Path
import pandas as pd, akshare as ak
H = Path(__file__).resolve().parent; date = sys.argv[1] if len(sys.argv) > 1 else "20261021"
codes = sorted(set(pd.read_csv(H / "results" / f"top5_v2_{date}.csv", dtype={"code": str}).code) | set(pd.read_csv(H / "results" / f"watch_v2_{date}.csv", dtype={"code": str}).code)) + list(pd.read_csv(H / "results" / f"avoid5_v2_{date}.csv", dtype={"code": str}).code)  # v2
_unused = list(pd.read_csv(H / "results" / f"top5_{date}.csv", dtype={"code": str}).code) + list(pd.read_csv(H / "results" / f"avoid5_{date}.csv", dtype={"code": str}).code)
K = ["营业总收入", "归母净利润", "扣非净利润", "经营现金流量净额", "净资产收益率(ROE)", "基本每股收益"]
out = {}
for c in codes:
    r = dict(source="新浪财经 财务摘要(akshare.stock_financial_abstract)", url=f"https://vip.stock.finance.sina.com.cn/corp/go.php/vFD_FinanceSummary/stockid/{c}.phtml", fetched_at=dt.datetime.now().isoformat(timespec="seconds"))
    try:
        d = ak.stock_financial_abstract(symbol=c); d = d[d["选项"] == "常用指标"].drop_duplicates("指标").set_index("指标"); cols = [x for x in d.columns if str(x).isdigit()]; cols = sorted(cols, reverse=True)
        p = cols[0]; prev = str(int(p[:4]) - 1) + p[4:]; r["period"] = p
        for k in K:
            if k in d.index:
                v = d.loc[k, p]; r[k] = None if pd.isna(v) else float(v)
                if prev in d.columns and k in ("营业总收入", "归母净利润", "扣非净利润"):
                    pv = d.loc[k, prev]; r[k + "_去年同期"] = None if pd.isna(pv) else float(pv)
                    r[k + "_同比"] = (float(v) / float(pv) - 1) if (pd.notna(v) and pd.notna(pv) and float(pv) > 0) else None
    except Exception as e: r["error"] = f"未取得: {type(e).__name__} {str(e)[:100]}"
    out[c] = r; print(c, r.get("period"), r.get("营业总收入"), r.get("归母净利润"), r.get("error", ""), flush=True)
json.dump(out, open(H / "results" / f"fundamentals_{date}_v2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)