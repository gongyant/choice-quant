"""P0-C: cross-check the Sina adjustment-factor table against the dividend/bonus/transfer announcements (Sina stock_history_dividend_detail) for the priority + validation stocks."""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, akshare as ak
sys.path.insert(0, str(Path(__file__).resolve().parent)); import p0lib as L
codes = ["600188", "600066", "000408", "600096", "000933", "601666", "689009", "600317", "000004", "600048", "600325", "600604", "600649", "600895", "000002", "000560", "001979", "002244"]
u = L.load_universe().set_index("code"); val = pd.read_csv(L.V2 / "data" / "sample_validation.csv", dtype={"code": str}).set_index("code")
out, notes = [], []
for c in codes:
    exch = (u.exchange.get(c) if c in u.index else val.exchange.get(c, None)) or ("SH" if c.startswith(("6", "9")) else "SZ")
    try:
        raw = L.load_raw(c); src = "cache"
        if raw is None:
            raw = ak.stock_zh_a_hist_tx(symbol=exch.lower() + c, start_date="20210101", end_date="20261016", adjust="").rename(columns={"volume": "volume"}); raw["date"] = pd.to_datetime(raw.date); raw = raw.sort_values("date"); src = "tencent-live"; time.sleep(1)
        fac = L.load_factor(c)
        if fac is None:
            fac = ak.stock_zh_a_daily(symbol=exch.lower() + c, start_date="20210101", end_date="20261016", adjust="qfq-factor"); fac["date"] = pd.to_datetime(fac.date); fac["qfq_factor"] = fac.qfq_factor.astype(float); time.sleep(1)
        dv = ak.stock_history_dividend_detail(symbol=c, indicator="分红"); time.sleep(1)
    except Exception as e:
        notes.append(dict(code=c, error=f"{type(e).__name__}: {str(e)[:100]}")); continue
    dv = dv[pd.to_datetime(dv["除权除息日"], errors="coerce") >= pd.Timestamp("2021-01-01")].copy(); dv["ex"] = pd.to_datetime(dv["除权除息日"]); fac = fac.sort_values("date").reset_index(drop=True); rs = raw.set_index("date").close
    for _, r in dv.iterrows():
        prevP = rs[rs.index < r.ex].iloc[-1] if (rs.index < r.ex).any() else np.nan
        ref = (prevP - float(r["派息"]) / 10) / (1 + (float(r["送股"]) + float(r["转增"])) / 10) if np.isfinite(prevP) else np.nan
        exp = ref / prevP if np.isfinite(prevP) else np.nan
        f1 = fac[fac.date == r.ex]; f0 = fac[fac.date < r.ex]
        obs = (f1.qfq_factor.iloc[0] / f0.qfq_factor.iloc[-1]) if (len(f1) and len(f0)) else np.nan
        out.append(dict(code=c, ex_date=str(r.ex.date()), bonus_per10=float(r["送股"]), transfer_per10=float(r["转增"]), cash_div_per10=float(r["派息"]), progress=r["进度"], prev_raw_close=prevP,
                        expected_ratio_from_announcement=exp, factor_ratio_in_table=obs, abs_diff=abs(exp - obs) if np.isfinite(exp) and np.isfinite(obs) else np.nan, factor_present_on_ex_date=bool(len(f1))))
o = pd.DataFrame(out); o.to_csv(L.P0 / "corporate_action_crosscheck.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(notes).to_csv(L.P0 / "corporate_action_crosscheck_errors.csv", index=False)
print(len(o), "ex-dates checked; abs diff median", o.abs_diff.median(), "max", o.abs_diff.max(), "n>0.01:", int((o.abs_diff > 0.01).sum()), "missing factor:", int((~o.factor_present_on_ex_date).sum()), "errors:", notes)
print(o.sort_values("abs_diff", ascending=False).head(8).to_string())
