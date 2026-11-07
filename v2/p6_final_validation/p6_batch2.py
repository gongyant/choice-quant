# -*- coding: utf-8 -*-
"""P6 Parts 3-7: stricter benchmarks + cost/time splits; LV20/LV60/resid-vol; 10-23 snapshot;
forward registry freeze; buy-gate; outputs for dashboard. CPU only. READ-ONLY on P0-P5."""
import json, hashlib, os, stat, datetime as dt, glob
from pathlib import Path
import numpy as np, pandas as pd
import p6_common as Q
independent_backtest = Q.independent_backtest; COST_RT=Q.COST_RT; C2=Q.C2; K=Q.K; INIT=Q.INIT
t0 = __import__("time").time(); R = Q.P6 / "results"; REG = Q.P6 / "predictions_registry"; REG.mkdir(exist_ok=True)
W = Q.load(); X = W.X; P = Q.P; cal = W.cal; codes = W.codes; col = W.col
rows5 = Q.rows_for(W, 5); rows20 = Q.rows_for(W, 20)
Oa = np.where(X.traded, X.O / X.F, np.nan)
r1 = pd.DataFrame(np.where(X.traded, X.C / X.F, np.nan)).ffill().pct_change(fill_method=None)

# ---------- Part3: benchmarks ----------
def ew_rets(mask_fn, h, rows, cost=True):
    """Equal-weight portfolio of names selected by mask_fn(i)->bool array; same entry/exit as B1; optional 0.30% RT on full turnover each period (conservative: assume 100% turnover)."""
    si = [W.sig_idx[q] for q in rows]; out = []
    for i in si:
        e, x = i + 1, min(i + 1 + h, len(cal) - 1)
        m = mask_fn(i) & X.traded[e] & ~W.olu[e]
        jj = np.where(m)[0]
        if len(jj) == 0: out.append(0.0); continue
        r = Oa[x, jj] / Oa[e, jj] - 1; r = r[np.isfinite(r)]
        g = float(np.mean(r)) if len(r) else 0.0
        out.append((1 + g) * (1 - COST_RT) - 1 if cost else g)  # full RT each period (no buffer for EW benches)
    return np.array(out)

def risk_matched_top5(h, rows):
    """Each period: among filt names, pick 5 with vol closest to the B1 portfolio's mean vol that period (B1 picks under A3 rules). Same costs/buffer as B1? Use plain equal-weight rebalance with full cost for fairness as 'risk-matched EW5'."""
    b1pk = Q.select_both(W, W.lowvol, rows)
    out_g = []
    for q, i in zip(rows, [W.sig_idx[q] for q in rows]):
        e, x = i + 1, min(i + 1 + h, len(cal) - 1)
        bp = b1pk[list(rows).index(q)]
        target = np.nanmean(W.vol[i, bp]) if len(bp) else np.nanmedian(W.vol[i, Q.filt(W, i)])
        f = Q.filt(W, i) & np.isfinite(W.vol[i]); cand = np.where(f)[0]
        if len(cand) < 5: out_g.append(0.0); continue
        dist = np.abs(W.vol[i, cand] - target); pick = cand[np.argsort(dist)[:5]]
        r = Oa[x, pick] / Oa[e, pick] - 1; g = float(np.nanmean(r)); out_g.append((1 + g) * (1 - COST_RT) - 1)
    return np.array(out_g)

# Build B1 period returns from batch1 output
B1p = pd.read_csv(R / "b1_periods_h5.csv", encoding="utf-8-sig")
B1p20 = pd.read_csv(R / "b1_periods_h20.csv", encoding="utf-8-sig")
IDX = pd.read_csv(Q.P6 / "index_daily.csv")
# Map each signal period to index open-to-open approx via close[exit]/close[entry]-1 (close proxy; disclosed)
def index_period_rets(h, rows, colname):
    si = [W.sig_idx[q] for q in rows]; ix = IDX.set_index("date")[colname]
    out = []
    for i in si:
        e, x = i + 1, min(i + 1 + h, len(cal) - 1)
        d0, d1 = str(cal[e].date()), str(cal[x].date())
        if d0 in ix.index and d1 in ix.index and ix.loc[d0] > 0: out.append(float(ix.loc[d1] / ix.loc[d0] - 1))
        else: out.append(np.nan)
    return np.array(out)

bench = {}
for h, rows, b1df in [(5, rows5, B1p), (20, rows20, B1p20)]:
    pool = b1df.pool_ret.values  # gross pool from engine
    pool_net = (1 + pool) * (1 - COST_RT) - 1  # apply same RT for fair net comparison (disclosed: pool originally gross)
    lv_ew = ew_rets(lambda i: Q.filt(W, i) & (W.volp[i] <= 0.20), h, rows, cost=True)
    # low-vol EW of bottom quintile
    sse = index_period_rets(h, rows, "SSE"); csi = index_period_rets(h, rows, "CSI300")
    rm = risk_matched_top5(h, rows)
    b1 = b1df.ret_net.values
    def stats(x, name):
        x = np.asarray(x, float); m = np.isfinite(x); y = x[m]
        return dict(strategy=name, horizon=h, n=int(m.sum()), cum_net=float(np.prod(1+y)-1), mean=float(y.mean()), mdd=float(Q.mdd(y)),
                    t_vs_0=float(Q.hac_t(y, 4 if h==5 else 2)))
    rows_out = [stats(b1, "B1"), stats(pool, "POOL_EW_gross"), stats(pool_net, "POOL_EW_net0.30%"), stats(lv_ew, "LOWVOL_EW_bottom20pct_net"),
                stats(sse, "SSE_index_close_proxy"), stats(csi, "CSI300_index_close_proxy"), stats(rm, "RISKMATCH_vol_Top5_net")]
    # excess B1 vs each
    for name, ref in [("vs_POOL_gross", pool), ("vs_POOL_net", pool_net), ("vs_LV_EW", lv_ew), ("vs_CSI300", csi), ("vs_RISKMATCH", rm)]:
        ex = b1 - ref; lo, hi = Q.block_ci(ex[np.isfinite(ex)], 8 if h==5 else 4)
        rows_out.append(dict(strategy=f"B1_excess_{name}", horizon=h, n=int(np.isfinite(ex).sum()), mean_excess=float(np.nanmean(ex)),
                             t_hac=float(Q.hac_t(ex[np.isfinite(ex)], 4 if h==5 else 2)), ci_lo=float(lo), ci_hi=float(hi),
                             beat=(ex>0).mean(), cum_b1=float(np.prod(1+b1)-1), cum_ref=float(np.prod(1+ref[np.isfinite(ref)])-1) if np.isfinite(ref).any() else np.nan))
    bench[h] = rows_out
    # year splits for B1
    b1df = b1df.copy(); b1df["year"] = b1df.signal_date.str[:4]; b1df["pool_net"] = pool_net; b1df["lv_ew"] = lv_ew; b1df["csi"] = csi
BENCH = pd.DataFrame([r for h in bench for r in bench[h]])
BENCH.to_csv(R / "part3_benchmarks.csv", index=False, encoding="utf-8-sig")

# Cost sensitivity (fixed params: only vary cost)
cost_rows = []
pk = Q.select_both(W, W.lowvol, rows5)
for c in [0.0, 0.001, 0.002, 0.003, 0.005, 0.008]:
    # quick: scale fees from engine gross by replacing net
    # Re-run independent with monkeypatched C2 — simpler recompute from trades of A3
    pass
# Use A3 trades gross and re-apply costs
A3tr = pd.read_csv(R / "a0a3_trades.csv", encoding="utf-8-sig", dtype={"code": str})
A3tr = A3tr[(A3tr.strategy == "A3_缓冲+约束(=B1)") & (A3tr.horizon == 5)]
A3per = pd.read_csv(R / "a0a3_periods.csv", encoding="utf-8-sig")
A3per5 = A3per[(A3per.strategy == "A3_缓冲+约束(=B1)") & (A3per.horizon == 5)].copy()
# From holdings-like trades: recompute period net at alternate costs
for c_rt in [0.0, 0.001, 0.002, 0.003, 0.005, 0.008]:
    c2 = c_rt / 2; nets = []
    for sd, g in A3tr.groupby("signal_date"):
        slot_nets = []
        for r in g.itertuples():
            if str(r.status).startswith("SKIPPED"): slot_nets.append(0.0); continue
            cb = 0.0 if r.retained_from_prev in (True, "True", "true", 1) else c2
            # keep_into_next column may exist
            keep = getattr(r, "keep_into_next", False) in (True, "True", "true", 1)
            cs = 0.0 if keep else c2
            slot_nets.append((1 + r.slot_ret_gross) * (1 - cb) * (1 - cs) - 1)
        # pad to K
        while len(slot_nets) < K: slot_nets.append(0.0)
        nets.append(sum(slot_nets[:K]) / K)
    nets = np.array(nets); cost_rows.append(dict(cost_rt=c_rt, horizon=5, cum_net=float(np.prod(1+nets)-1), mean=float(nets.mean()), mdd=float(Q.mdd(nets)), t_excess_vs_pool=float(Q.hac_t(nets - A3per5.pool_ret.values, 4))))
# Time splits (fixed A3 params): first half / second half / by year already in a0a3_by_year
half = len(A3per5) // 2
for label, sl in [("first_half", A3per5.iloc[:half]), ("second_half", A3per5.iloc[half:]), ("pre_2025", A3per5[A3per5.signal_date < "2025-01-01"]), ("from_2025", A3per5[A3per5.signal_date >= "2025-01-01"])]:
    e = sl.ret_net - sl.pool_ret
    cost_rows.append(dict(split=label, horizon=5, n=len(sl), cum_net=float(np.prod(1+sl.ret_net)-1), cum_pool=float(np.prod(1+sl.pool_ret)-1),
                          mean_excess=float(e.mean()), t_hac=float(Q.hac_t(e.values, 4)), mdd=float(Q.mdd(sl.ret_net)), turnover=float(sl.turnover.mean())))
pd.DataFrame(cost_rows).to_csv(R / "part3_cost_time_splits.csv", index=False, encoding="utf-8-sig")

# Explain t≈2 vs pool t≈0.19
expl = dict(
    note="B1相对A0(原始低波)的配对差主要来自费用节约,两序列高度相关,差的波动小→t≈2;相对股票池,B1绝对路径波动仍大且均值超额接近0→t≈0.19。",
    corr_b1_a0=float(np.corrcoef(
        A3per5.ret_net.values,
        A3per[(A3per.strategy=="A0_无缓冲无约束")&(A3per.horizon==5)].ret_net.values)[0,1]),
    std_b1_minus_a0=float((A3per5.ret_net.values - A3per[(A3per.strategy=="A0_无缓冲无约束")&(A3per.horizon==5)].ret_net.values).std()),
    std_b1_minus_pool=float((A3per5.ret_net - A3per5.pool_ret).std()),
    mean_b1_minus_a0=float((A3per5.ret_net.values - A3per[(A3per.strategy=="A0_无缓冲无约束")&(A3per.horizon==5)].ret_net.values).mean()),
    mean_b1_minus_pool=float((A3per5.ret_net - A3per5.pool_ret).mean()),
)
json.dump(expl, open(R / "part3_tstat_explain.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------- Part4: LV20 / LV60 / residual vol ----------
vol20 = W.vol
vol60 = r1.rolling(60, min_periods=40).std().values
# residual vol: residual of r1 vs pool EW market
mkt = r1.where(pd.DataFrame(W.elig)).mean(1)
# residual = r - beta*mkt; resid vol = std of residual
beta = W.beta  # already 60d beta
resid = r1.values - beta * mkt.values[:, None]
resid_vol = pd.DataFrame(resid).rolling(20, min_periods=15).std().values
def lv_matrix(vol):
    lv = np.full((len(W.sig), W.N), np.nan)
    for q, i in enumerate(W.sig_idx):
        m = W.elig[i]; lv[q, m] = -vol[i, m]
    return lv
LV = {"LV20": W.lowvol, "LV60": lv_matrix(vol60), "RESID_VOL20": lv_matrix(resid_vol)}
lv_sum = []
for name, M in LV.items():
    for h, rows in [(5, rows5), (20, rows20)]:
        pr, tr, eq = independent_backtest(W, Q.select_both(W, M, rows), rows, h, f"B1rules_{name}", apply_mergers=True)
        e = pr.ret_net - pr.pool_ret
        lv_sum.append(dict(feature=name, horizon=h, cum_net=float(np.prod(1+pr.ret_net)-1), cum_gross=float(np.prod(1+pr.ret_gross)-1),
                           mdd=float(Q.mdd(pr.ret_net)), turnover=float(pr.turnover.mean()), fees=float(pr.fees.sum()),
                           mean_excess=float(e.mean()), t_hac=float(Q.hac_t(e.values, 4 if h==5 else 2)), beat_pool=float((e>0).mean())))
        pr.to_csv(R / f"part4_{name}_h{h}_periods.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(lv_sum).to_csv(R / "part4_lv_variants.csv", index=False, encoding="utf-8-sig")

print("PART3", BENCH[BENCH.horizon==5][["strategy","cum_net","mean","mdd","t_vs_0","mean_excess","t_hac"]].round(4).to_string())
print("EXPLAIN", expl)
print("COST/SPLIT", pd.read_csv(R/"part3_cost_time_splits.csv").round(4).to_string())
print("PART4", pd.DataFrame(lv_sum).round(4).to_string())
print("secs", round(__import__("time").time()-t0))
