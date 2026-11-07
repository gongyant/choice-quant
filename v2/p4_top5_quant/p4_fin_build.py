# -*- coding: utf-8 -*-
"""P4-E2 step 1: point-in-time fundamentals panel from Eastmoney datacenter (业绩报表 RPT_LICO_FN_CPD + 资产负债表 RPT_DMSK_FN_BALANCE), fetched
2026-10-21 into p4_top5_quant/fin_raw. A report value becomes usable on the FIRST TRADING DAY STRICTLY AFTER its NOTICE_DATE (first announcement date).
Missing NOTICE_DATE -> conservative statutory deadline (Q1 04-30, H1 08-31, Q3 10-31, annual 04-30 next year). Values never written before availability.
Limitation: Eastmoney stores the latest version of each period's numbers (restatements may replace the originally announced values)."""
import json, pickle, glob
import numpy as np, pandas as pd
import p4lib as P
C = P.P4 / "cache"; FR = P.P4 / "fin_raw"
X = pickle.load(open(C / "world.pkl", "rb")); cal = X.cal; codes = list(X.codes); col = {c: j for j, c in enumerate(codes)}; T, N = len(cal), len(codes)
def stat_lag(p):
    p = pd.Timestamp(p); m = p.month
    return {3: pd.Timestamp(p.year, 4, 30), 6: pd.Timestamp(p.year, 8, 31), 9: pd.Timestamp(p.year, 10, 31), 12: pd.Timestamp(p.year + 1, 4, 30)}[m]
def load(kind, dcol):
    fs = sorted(glob.glob(str(FR / f"{kind}_*.csv.gz"))); d = pd.concat([pd.read_csv(f, dtype={"SECURITY_CODE": str}) for f in fs])
    d = d[d.SECURITY_CODE.isin(col)].copy(); d["period"] = pd.to_datetime(d[dcol].str[:10]); d["notice"] = pd.to_datetime(d.NOTICE_DATE.str[:10], errors="coerce")
    d["lag_used"] = d.notice.isna(); d.loc[d.lag_used, "notice"] = d.loc[d.lag_used, "period"].map(stat_lag)
    return d.sort_values(["SECURITY_CODE", "period", "notice"]).drop_duplicates(["SECURITY_CODE", "period"], keep="first")
Y = load("yjbb", "REPORTDATE"); Z = load("zcfz", "REPORT_DATE")
Y = Y.merge(Z[["SECURITY_CODE", "period", "DEBT_ASSET_RATIO", "notice"]].rename(columns={"notice": "notice_bs"}), on=["SECURITY_CODE", "period"], how="left")
q = Y.period.dt.month // 3; Y["eps_ann"] = Y.BASIC_EPS * 4 / q; Y["roe_ann"] = Y.WEIGHTAVG_ROE * 4 / q; Y["ocf_ps_ann"] = Y.MGJYXJJE * 4 / q
FIN = {"roe_ann": "roe_ann", "rev_yoy": "YSTZ", "np_yoy": "SJLTZ", "gross_margin": "XSMLL", "ocf_to_bps": None, "accrual": None, "debt_ratio": "DEBT_ASSET_RATIO", "ep": None, "bp": None}
Y["ocf_to_bps"] = Y.ocf_ps_ann / Y.BPS.where(Y.BPS > 0); Y["accrual"] = (Y.eps_ann - Y.ocf_ps_ann) / Y.BPS.where(Y.BPS > 0)
M = {k: np.full((T, N), np.nan) for k in ["roe_ann", "rev_yoy", "np_yoy", "gross_margin", "ocf_to_bps", "accrual", "debt_ratio", "eps_ann", "bps", "period_ord", "notice_ord"]}
calv = cal.values
for c, g in Y.groupby("SECURITY_CODE"):
    j = col[c]; g = g.sort_values(["notice", "period"]); best = None
    for r in g.itertuples():
        i = int(np.searchsorted(calv, np.datetime64(r.notice), side="right"))   # first trading day strictly after notice
        if i >= T or (best is not None and r.period <= best): continue
        best = r.period
        vals = dict(roe_ann=r.roe_ann, rev_yoy=r.YSTZ, np_yoy=r.SJLTZ, gross_margin=r.XSMLL, ocf_to_bps=r.ocf_to_bps, accrual=r.accrual, eps_ann=r.eps_ann, bps=r.BPS, period_ord=r.period.toordinal(), notice_ord=r.notice.toordinal(),
                    debt_ratio=r.DEBT_ASSET_RATIO if (pd.notna(r.notice_bs) and r.notice_bs <= r.notice) else np.nan)
        for k, v in vals.items(): M[k][i:, j] = v     # forward-filled until overwritten by a newer period
# debt ratio announced later than yjbb (rare) -> written from its own notice
for c, g in Y[Y.notice_bs.notna() & (Y.notice_bs > Y.notice)].groupby("SECURITY_CODE"):
    j = col[c]
    for r in g.itertuples():
        i = int(np.searchsorted(calv, np.datetime64(r.notice_bs), side="right"))
        if i < T: M["debt_ratio"][i:, j] = np.where(M["period_ord"][i:, j] == r.period.toordinal(), r.DEBT_ASSET_RATIO, M["debt_ratio"][i:, j])
Cu = X.C; M["ep"] = M["eps_ann"] / Cu; M["bp"] = M["bps"] / Cu
for k in ["rev_yoy", "np_yoy"]: M[k] = np.clip(M[k], -300, 300)
stale = (cal.values.astype("datetime64[D]").astype(int)[:, None] + 719163 - M["notice_ord"]) > 400   # report older than ~13 months -> drop
names = ["roe_ann", "rev_yoy", "np_yoy", "gross_margin", "ocf_to_bps", "accrual", "debt_ratio", "ep", "bp"]
for k in names: M[k][stale] = np.nan
fl = np.load(C / "flags.npz"); elig = fl["elig"]
f = {k: pd.DataFrame(M[k]) for k in names}; Xf = P.rank_feats(f, elig, names)
E1 = np.load(C / "Xq_E1.npy", mmap_mode="r"); np.save(C / "Xq_E2.npy", np.concatenate([np.asarray(E1), Xf], axis=2))
fn = pickle.load(open(C / "featnames.pkl", "rb")); fn["E2"] = fn["E1"] + names; pickle.dump(fn, open(C / "featnames.pkl", "wb"))
np.savez_compressed(C / "fin_raw_mats.npz", **{k: M[k].astype(np.float32) for k in M})
cov = []
for d in ["2022-07-01", "2023-07-03", "2024-07-01", "2025-07-01", "2026-09-28", "2026-10-21"]:
    i = int(cal.searchsorted(pd.Timestamp(d))); m = elig[i]
    cov.append(dict(date=str(cal[i].date()), pool=int(m.sum()), **{f"cov_{k}": float(np.isfinite(M[k][i, m]).mean()) for k in names}))
cov = pd.DataFrame(cov); cov.to_csv(P.P4 / "results" / "fin_coverage.csv", index=False, encoding="utf-8-sig")
info = dict(rows_yjbb=len(Y), codes_with_data=int(Y.SECURITY_CODE.nunique()), n_universe=N, share_statutory_lag=float(Y.lag_used.mean()), periods=sorted(Y.period.dt.strftime("%Y-%m-%d").unique().tolist()),
            median_notice_lag_days=float((Y.notice - Y.period).dt.days.median()))
json.dump(info, open(C / "fin_info.json", "w"), indent=1); print(info); print(cov.round(3).to_string())
