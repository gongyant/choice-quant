# -*- coding: utf-8 -*-
"""Build the shared P2-A panel (signal_date x code) from frozen predictions + P0 matrices. Read-only wrt P0/P1."""
import json, time
import numpy as np, pandas as pd
import p2lib as Q
t0 = time.time()
X, W = Q.build_world()
hp = pd.read_parquet(Q.HP_ALL, columns=["signal_date", "code", "model", "eligible_asof", "score", "pctile", "model_id"])
hp["signal_date"] = pd.to_datetime(hp.signal_date)
sig = sorted(hp.signal_date.unique()); cal = X.cal; ci = cal.get_indexer(pd.DatetimeIndex(sig)); assert (ci >= 0).all()
codes = np.array(X.codes); N = len(codes); K = len(sig)
assert set(hp.code.unique()) == set(codes), "code set mismatch"
f = W.f
turn20 = Q.rolling_turn20(X.TURN); ami = Q.amihud20(f["r1"].values, X.AMT)
buyable = X.traded & ~X.lim_up; sellable = X.traded & ~X.lim_dn
fw = Q.exec_forward(W.ADJf, buyable, sellable, ci)
lab, y, yw = W.lab, W.y, W.yw
T = len(cal)
def at(M): return np.asarray(M)[ci]
rows = {}
rows["i"] = np.repeat(ci, N); rows["signal_date"] = np.repeat(np.array(sig), N); rows["code"] = np.tile(codes, K)
rows["eligible"] = at(W.elig).ravel()
for k, M in dict(amt20=f["amt20"].values, log_amt20=f["log_amt20"].values, r1=f["r1"].values, r5=f["r5"].values, r20=f["r20"].values, r60=f["r60"].values, vol20=f["vol20"].values,
                 turn20=turn20, amihud20=ami).items(): rows[k] = at(M).ravel()
rows["lim_up_t"] = at(X.lim_up).ravel(); rows["lim_dn_t"] = at(X.lim_dn).ravel()
rows["ret_old"] = at(lab).ravel(); rows["ret_old_rel"] = at(y).ravel()   # t+1 close -> t+6 close (P0 label), rel = minus eligible-pool mean
nxt = np.minimum(ci + 1, T - 1)
rows["buyable_t1"] = buyable[nxt].ravel() & np.repeat(ci + 1 < T, N)
E, Xi = fw["entry_idx"], fw["exit_idx"]
cal_ns = cal.values
rows["entry_date"] = np.where(E >= 0, cal_ns[np.maximum(E, 0)], np.datetime64("NaT")).ravel()
rows["exit_date"] = np.where(Xi >= 0, cal_ns[np.maximum(Xi, 0)], np.datetime64("NaT")).ravel()
rows["entry_delay"] = np.where(E >= 0, E - (ci[:, None] + 1), -1).ravel()
rows["hold_rows"] = np.where((E >= 0) & (Xi >= 0), Xi - E, -1).ravel()
rows["exit_forced"] = fw["forced"].ravel(); rows["ret_new"] = fw["ret"].ravel()
P = pd.DataFrame(rows)
# new-definition relative return: minus the mean over the eligible pool members with a realised executable return (same date)
m = P.eligible & P.ret_new.notna()
P["ret_new_rel"] = np.nan; P.loc[m, "ret_new_rel"] = P.ret_new[m] - P[m].groupby("i").ret_new.transform("mean")
# static (non point-in-time!) attributes from universe_master (cninfo latest)
u = pd.read_csv(Q.V2 / "data" / "universe_master.csv", dtype=str).set_index("code")
for c in ["exchange", "board", "industry_l1"]: P[c] = P.code.map(u[c])
P["industry_l1_note"] = "cninfo_latest_not_point_in_time"
P["is_st_name_current"] = P.code.map(u.is_st_name).astype(str).str.lower().isin(["true", "1", "1.0"])
P["delisted_in_sample"] = P.code.map(u.status).eq("已终止上市")
# scores
for mname in Q.MODELS:
    h = hp[hp.model == mname].set_index(["signal_date", "code"])
    key = pd.MultiIndex.from_arrays([P.signal_date, P.code])
    P["score_" + mname] = h.score.reindex(key).values
    P["elig_" + mname] = h.eligible_asof.reindex(key).values
chk = {m: int((P["elig_" + m].astype(bool) != P.eligible).sum()) for m in Q.MODELS}
meta = dict(n_rows=len(P), n_dates=K, n_codes=N, eligible_rows=int(P.eligible.sum()), elig_mismatch_vs_frozen=chk,
            ret_new_eligible_finite=int(m.sum()), ret_old_eligible_finite=int((P.eligible & P.ret_old_rel.notna()).sum()),
            share_entry_delayed=float((P.entry_delay[P.eligible & (P.entry_delay >= 0)] > 0).mean()),
            share_not_buyable_t1_eligible=float((~P.buyable_t1[P.eligible]).mean()),
            share_exit_forced=float(P.exit_forced[m].mean()), built_seconds=round(time.time() - t0, 1))
P.drop(columns=["industry_l1_note"]).to_parquet(Q.RES / "p2a_panel.parquet", index=False)
(Q.RES / "p2a_panel_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(meta, ensure_ascii=False, indent=1))