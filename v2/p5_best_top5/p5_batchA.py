# -*- coding: utf-8 -*-
"""P5 Batch A: (1) B0-B3 identical engine + paired B3-B1 test; (2) E0/E1 drawdown forensics + backtest bug checks; (3) financial audit.
B0 = eligible-pool equal weight (gross, benchmark); B1 = low-vol Top5 (plain tradable filter); B2 = low-vol through the SAME Top20->constraints->buffer module
as S4 (isolates the model's contribution); B3 = S4 (E2reg 50% + low-vol 50% + constraints + buffer). EXPLORATORY."""
import json, glob, time
import numpy as np, pandas as pd
import p5_common as Q
t0 = time.time(); R = Q.P5 / "results"; R.mkdir(exist_ok=True); W = Q.load(); P = Q.P
E2 = {h: Q.mat(W, pd.read_parquet(Q.PR / f"E2reg_h{h}.parquet")) for h in [5, 20]}
E0 = Q.mat(W, W.H[W.H.model == "LGBM"]); E1 = {h: Q.mat(W, pd.read_parquet(Q.PR / f"E1reg_h{h}.parquet")) for h in [5, 20]}
E1L = {h: Q.mat(W, pd.read_parquet(Q.PR / f"E1lrank_h{h}.parquet")) for h in [5, 20]}
per, hold = [], []
for h in [5, 20]:
    rows = Q.rows_for(W, h)
    for nm, pk in [("B1_低波动Top5", Q.plain(W, W.lowvol, rows)), ("B2_低波动+约束+缓冲", Q.select(W, W.lowvol, rows, True, False)), ("B3_S4", Q.select(W, E2[h], rows, True, True)),
                   ("E0_LGBM原模型", Q.plain(W, E0, rows)), ("E1_LGBM回归", Q.plain(W, E1[h], rows)), ("E1_LambdaRank", Q.plain(W, E1L[h], rows))]:
        pr, ho = Q.run(W, pk, rows, h, nm); per.append(pr); hold.append(ho)
PER = pd.concat(per); HO = pd.concat(hold); PER["year"] = PER.signal_date.str[:4]
PER.to_csv(R / "a1_periods.csv", index=False, encoding="utf-8-sig"); HO.to_csv(R / "a1_holdings.csv", index=False, encoding="utf-8-sig")
def summ(a):
    e = a.ret_net - a.pool_ret
    return dict(n=len(a), avg_hold=a.n_held.mean(), cum_gross=np.prod(1 + a.ret_gross) - 1, cum_net=np.prod(1 + a.ret_net) - 1, cum_pool=np.prod(1 + a.pool_ret) - 1, mean_excess=e.mean(),
                t_hac=Q.hac_t(e, 4), beat=(e > 0).mean(), mdd=P.mdd(a.ret_net), mdd_pool=P.mdd(a.pool_ret), turnover=a.turnover.mean(), fees=a.fees.sum())
S, Y = [], []
for (nm, h), a in PER.groupby(["strategy", "horizon"]):
    S.append(dict(strategy=nm, horizon=h, **summ(a)))
    for y, b in a.groupby("year"): Y.append(dict(strategy=nm, horizon=h, year=y, **summ(b)))
for h in [5, 20]:
    a = PER[(PER.strategy == "B1_低波动Top5") & (PER.horizon == h)]; S.append(dict(strategy="B0_股票池等权(未扣费)", horizon=h, n=len(a), cum_gross=np.prod(1 + a.pool_ret) - 1, cum_net=np.nan, mdd=P.mdd(a.pool_ret)))
    for y, b in a.groupby("year"): Y.append(dict(strategy="B0_股票池等权(未扣费)", horizon=h, year=y, cum_gross=np.prod(1 + b.pool_ret) - 1, mdd=P.mdd(b.pool_ret)))
S = pd.DataFrame(S); Y = pd.DataFrame(Y); S.to_csv(R / "a1_summary.csv", index=False, encoding="utf-8-sig"); Y.to_csv(R / "a1_by_year.csv", index=False, encoding="utf-8-sig")
PT = []
for h, blk, lag in [(5, 8, 4), (20, 4, 2)]:
    for a_, b_ in [("B3_S4", "B1_低波动Top5"), ("B3_S4", "B2_低波动+约束+缓冲"), ("B2_低波动+约束+缓冲", "B1_低波动Top5")]:
        x = PER[(PER.strategy == a_) & (PER.horizon == h)].set_index("signal_date").ret_net - PER[(PER.strategy == b_) & (PER.horizon == h)].set_index("signal_date").ret_net
        lo, hi = Q.block_ci(x.values, blk); PT.append(dict(pair=f"{a_} - {b_}", horizon=h, n=len(x), mean_diff=x.mean(), hac_t=Q.hac_t(x.values, lag), ci_lo=lo, ci_hi=hi, share_pos=(x > 0).mean(), block=blk))
PT = pd.DataFrame(PT); PT.to_csv(R / "a1_paired_tests.csv", index=False, encoding="utf-8-sig")
# ---------------- (2) forensics ----------------
F = []
for nm in ["E0_LGBM原模型", "E1_LGBM回归", "E1_LambdaRank", "B1_低波动Top5", "B3_S4"]:
    hh = HO[(HO.strategy == nm) & (HO.horizon == 5) & HO.status.isin(["FILLED", "MARKED_LAST_CLOSE_UNSOLD"])].copy()
    q = {s: k for k, s in enumerate(W.sig)}; ii = hh.signal_date.map(lambda s: W.sig_idx[q[s]]).values; jj = hh.code.map(W.col).values
    lim = np.array([W.X.lim_up[i - 4:i + 1, j].sum() + W.X.lim_dn[i - 4:i + 1, j].sum() for i, j in zip(ii, jj)])
    sz = np.array([(np.nan_to_num(W.fmcap[i][W.elig[i]]) < W.fmcap[i, j]).mean() for i, j in zip(ii, jj)])
    F.append(dict(strategy=nm, n=len(hh), vol_pctile=np.nanmean(W.volp[ii, jj]), beta=np.nanmean(W.beta[ii, jj]), dd60=np.nanmean(W.dd60[ii, jj]), r20=np.nanmean(W.r20[ii, jj]),
                  size_pctile=np.nanmean(sz), limit_moves_5d=lim.mean(), share_limit_5d=(lim > 0).mean(), slot_ret_mean=hh.slot_ret_gross.mean(), slot_ret_p05=hh.slot_ret_gross.quantile(.05),
                  worst_slot=hh.slot_ret_gross.min(), n_marked_unsold=int((hh.status == "MARKED_LAST_CLOSE_UNSOLD").sum())))
F = pd.DataFrame(F); poolx = dict(strategy="(池均值参照)", vol_pctile=0.5, size_pctile=0.5, beta=float(np.nanmean([np.nanmean(W.beta[i][W.elig[i]]) for i in W.sig_idx])),
                                   dd60=float(np.nanmean([np.nanmean(W.dd60[i][W.elig[i]]) for i in W.sig_idx])), r20=float(np.nanmean([np.nanmean(W.r20[i][W.elig[i]]) for i in W.sig_idx])))
F = pd.concat([F, pd.DataFrame([poolx])]); F.to_csv(R / "a2_exposures.csv", index=False, encoding="utf-8-sig")
worstP = PER[PER.strategy.isin(["E0_LGBM原模型", "E1_LGBM回归"]) & (PER.horizon == 5)].nsmallest(20, "ret_net")[["strategy", "signal_date", "ret_net", "pool_ret", "n_held"]]
worstP.to_csv(R / "a2_worst_periods.csv", index=False, encoding="utf-8-sig")
worstS = HO[HO.strategy.isin(["E0_LGBM原模型", "E1_LGBM回归"]) & (HO.horizon == 5)].nsmallest(30, "slot_ret_gross")[["strategy", "signal_date", "code", "name", "status", "entry_date", "exit_date", "exit_delay_days", "slot_ret_gross"]]
worstS.to_csv(R / "a2_worst_stocks.csv", index=False, encoding="utf-8-sig")
# lambdarank label / NDCG check
gr5 = W.LAB[5]["gr"]; ex5 = W.LAB[5]["ex"]; chk = []
for q, i in enumerate(W.sig_idx[::10]):
    m = np.isfinite(gr5[i]); g = gr5[i][m]; e = ex5[i][m]
    chk.append(dict(signal_date=W.sig[q * 10], n=int(m.sum()), grade_counts_min=int(np.bincount(g.astype(int), minlength=10).min()), grade_counts_max=int(np.bincount(g.astype(int), minlength=10).max()),
                    top_grade_is_best=bool(e[g == 9].min() >= e[g == 8].max() - 1e-12), monotone=bool(np.all(np.diff([e[g == k].mean() for k in range(10)]) > 0))))
LC = pd.DataFrame(chk)
bot = []
for nm, M in [("E0_LGBM原模型", E0), ("E1_LGBM回归", E1[5]), ("E1_LambdaRank", E1L[5])]:
    pk = Q.plain(W, M, Q.rows_for(W, 5)); b9 = [];  b0 = []
    for q, p in enumerate(pk):
        i = W.sig_idx[q]; g = gr5[i][p]; g = g[np.isfinite(g)]; b9.append((g == 9).mean() if len(g) else np.nan); b0.append((g == 0).mean() if len(g) else np.nan)
    bot.append(dict(strategy=nm, top5_in_top_decile=np.nanmean(b9), top5_in_bottom_decile=np.nanmean(b0)))
LC.to_csv(R / "a2_label_check.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(bot).to_csv(R / "a2_top5_decile_mix.csv", index=False, encoding="utf-8-sig")
# backtest bug checks
bugs = {}
bugs["signal_spacing_5d"] = sorted(set(np.diff(W.sig_idx).tolist())); bugs["signal_spacing_20d"] = sorted(set(np.diff(W.sig_idx[::4]).tolist()))
bugs["overlap_note"] = "5日每期 entry=i+1 开盘, exit=i+6 开盘 = 下一期 entry;20日每4个信号(间隔20交易日)。持有期互不重叠,连乘复利合法。仅卖出顺延的仓位会跨入下一期。"
d = HO[HO.status.isin(["FILLED", "MARKED_LAST_CLOSE_UNSOLD"])]
bugs["delayed_exits"] = d.groupby(["strategy", "horizon"]).exit_delay_days.apply(lambda s: int((s > 0).sum())).pipe(lambda z: {f"{a}|h{b}": int(v) for (a, b), v in z.items()})
mk = d[d.status == "MARKED_LAST_CLOSE_UNSOLD"]; bugs["marked_unsold_rows"] = len(mk); bugs["marked_unsold_detail"] = mk[["strategy", "horizon", "signal_date", "code", "name", "slot_ret_gross"]].to_dict("records")
# delisting stress: marked positions written down to -100%
st = []
for (nm, h), a in PER.groupby(["strategy", "horizon"]):
    m = mk[(mk.strategy == nm) & (mk.horizon == h)]; adj = a.set_index("signal_date").ret_net.copy()
    for r in m.itertuples(): adj[r.signal_date] -= (1 + r.slot_ret_gross) / 5
    st.append(dict(strategy=nm, horizon=h, cum_net=np.prod(1 + a.ret_net) - 1, cum_net_delist_minus100=np.prod(1 + adj) - 1, n_marked=len(m)))
pd.DataFrame(st).to_csv(R / "a2_delist_stress.csv", index=False, encoding="utf-8-sig")
# benchmark survivorship: pool_bench drops names with no bar at exit (nanmean). Recompute with those names held to their next tradable open.
Oa = np.where(W.X.traded, W.X.O / W.X.F, np.nan); T = Oa.shape[0]; pb = []
for h in [5, 20]:
    for q in Q.rows_for(W, h):
        i = W.sig_idx[q]; e, x = i + 1, min(i + 1 + h, T - 1); m = np.where(W.elig[i] & W.X.traded[e] & ~W.olu[e])[0]
        r1 = Oa[x, m] / Oa[e, m] - 1; miss = ~np.isfinite(r1); r2 = r1.copy()
        for k in np.where(miss)[0]:
            j = m[k]; nx = np.where(W.X.traded[x:, j])[0]
            r2[k] = (Oa[x + nx[0], j] if len(nx) else W.A[-1, j]) / Oa[e, j] - 1
        pb.append(dict(horizon=h, signal_date=W.sig[q], pool_orig=np.nanmean(r1), pool_fixed=np.mean(r2), n_missing=int(miss.sum())))
PB = pd.DataFrame(pb); PB.to_csv(R / "a2_pool_bench_fix.csv", index=False, encoding="utf-8-sig")
bugs["pool_bench_missing_exit_share"] = float(PB.n_missing.sum() / max(1, len(PB))); bugs["pool_bench_mean_diff"] = PB.groupby("horizon").apply(lambda g: float((g.pool_fixed - g.pool_orig).mean())).to_dict()
json.dump(bugs, open(R / "a2_bug_checks.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
# ---------------- (3) financial audit ----------------
FR = Q.P4D / "fin_raw"; y = pd.concat([pd.read_csv(f, dtype={"SECURITY_CODE": str}) for f in sorted(glob.glob(str(FR / "yjbb_*.csv.gz")))])
y = y[y.SECURITY_CODE.isin(W.col)].copy(); y["period"] = pd.to_datetime(y.REPORTDATE.str[:10]); y["notice"] = pd.to_datetime(y.NOTICE_DATE.str[:10]); y = y.sort_values("notice").drop_duplicates(["SECURITY_CODE", "period"], keep="first")
y["q"] = y.period.dt.month // 3; y["eps_ann"] = y.BASIC_EPS * 4 / y.q
k = y.set_index(["SECURITY_CODE", "period"]).BASIC_EPS
def ttm(r):
    if r.q == 4: return r.BASIC_EPS
    pa = (r.SECURITY_CODE, pd.Timestamp(r.period.year - 1, 12, 31)); ps = (r.SECURITY_CODE, pd.Timestamp(r.period.year - 1, r.period.month, r.period.day))
    return r.BASIC_EPS + k.get(pa, np.nan) - k.get(ps, np.nan)
asof = pd.Timestamp("2026-10-21"); last = y[y.notice < asof].sort_values(["period"]).groupby("SECURITY_CODE").tail(1).copy()
last["eps_ttm"] = last.apply(ttm, axis=1); close = pd.Series(W.X.C[-1], index=W.codes); last["close"] = last.SECURITY_CODE.map(close)
last["pe_ann"] = last.close / last.eps_ann.where(last.eps_ann > 0); last["pe_ttm"] = last.close / last.eps_ttm.where(last.eps_ttm > 0)
last["roe_raw"] = last.WEIGHTAVG_ROE; last["roe_ann"] = last.WEIGHTAVG_ROE * 4 / last.q
last["pe_rel_diff"] = last.pe_ann / last.pe_ttm - 1
q3 = y[y.period == "2026-09-30"]
fa = dict(asof="2026-10-21", n_latest=len(last), latest_period_counts=last.period.dt.strftime("%Y-%m-%d").value_counts().to_dict(),
          pe_ann_vs_ttm_median_abs_rel_diff=float(last.pe_rel_diff.abs().median()), pe_share_abs_diff_gt_20pct=float((last.pe_rel_diff.abs() > .2).mean()),
          pe_sign_disagree=int(((last.eps_ann > 0) != (last.eps_ttm > 0)).sum()),
          q3_2026_rows_total_in_fetch=len(q3), q3_2026_notice_le_1020=int((q3.notice <= "2026-10-20").sum()), q3_2026_notice_eq_1021=int((q3.notice == "2026-10-21").sum()), q3_2026_notice_gt_1021=int((q3.notice > "2026-10-21").sum()),
          q3_rule="财报在公告日之后的第一个交易日才可用:10-21 当日只能使用公告日 <= 10-20 的报告;公告日 10-21 的报告不进入 10-21 的特征(检查 fin_raw_mats.notice_ord)",
          nan_handling_E2="rank_feats: 缺失值在横截面排序后置为 0(即中位数)——视为'中性/未知',不是'差';LightGBM 未见到 NaN。")
Fm = np.load(C := Q.C / "fin_raw_mats.npz"); no = Fm["notice_ord"][-1]; fa["max_notice_used_on_1021"] = str(pd.Timestamp.fromordinal(int(np.nanmax(no))).date()); fa["n_used_with_notice_ge_1021"] = int((no >= pd.Timestamp("2026-10-21").toordinal()).sum())
fa["contaminated"] = fa["n_used_with_notice_ge_1021"] > 0
json.dump(fa, open(R / "a3_fin_audit.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
last[["SECURITY_CODE", "SECURITY_NAME_ABBR", "period", "notice", "close", "BASIC_EPS", "eps_ann", "eps_ttm", "pe_ann", "pe_ttm", "roe_raw", "roe_ann"]].to_csv(R / "a3_pe_roe_latest.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 300)
print(S.round(4).to_string()); print(PT.round(4).to_string()); print(Y[Y.horizon == 5].pivot_table(index="year", columns="strategy", values="cum_net").round(3).to_string())
print(Y[Y.horizon == 5].pivot_table(index="year", columns="strategy", values="cum_gross").round(3)["B0_股票池等权(未扣费)"].to_string())
print(F.round(3).to_string()); print(pd.DataFrame(bot).round(3)); print(LC.to_string()); print(worstP.head(8).to_string())
print(worstS.head(10).to_string()); print(pd.DataFrame(st).round(3).to_string()); print({k: v for k, v in bugs.items() if k != "marked_unsold_detail"}); print(fa); print("secs", round(time.time() - t0))
