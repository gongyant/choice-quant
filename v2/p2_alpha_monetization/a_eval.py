# -*- coding: utf-8 -*-
"""P2-A main evaluation (items 2-10). Reads results/p2a_panel.parquet (frozen scores + as-of-t controls + forward returns). No model retraining."""
import json, time, sys
import numpy as np, pandas as pd
from scipy import stats as st
import p2lib as Q
t0 = time.time(); R = Q.RES
P = pd.read_parquet(R / "p2a_panel.parquet"); P["signal_date"] = pd.to_datetime(P.signal_date)
Pe = P[P.eligible].copy().reset_index(drop=True)          # point-in-time eligible_asof_t pool (known at t close)
MOD = Q.MODELS; SC = {m: "score_" + m for m in MOD}
g = Pe.groupby("i")
# ---- as-of-t cross-sectional groups / baselines (computed within the eligible pool at t; no execution-day information) ----
def tercile(col, labels):
    r = g[col].rank(method="first"); n = g[col].transform("count"); q = np.ceil(r / n * 3).clip(1, 3)
    return q.map({1: labels[0], 2: labels[1], 3: labels[2]}).where(Pe[col].notna())
Pe["g_liq"] = tercile("amt20", ["L1低流动性", "L2中", "L3高流动性"]); Pe["g_vol"] = tercile("vol20", ["V1低波动", "V2中", "V3高波动"])
a = Pe.amt20
Pe["g_amt_abs"] = pd.cut(a, [0, 5e7, 1e8, 3e8, 1e15], labels=["A1[2e7,5e7)", "A2[5e7,1e8)", "A3[1e8,3e8)", "A4>=3e8"], right=False)
Pe["g_exch"] = Pe.exchange; Pe["g_board"] = Pe.board; Pe["g_ind"] = Pe.industry_l1
def prk(x): return x.groupby(Pe.i).rank(pct=True)
Pe["rk_r5"] = prk(-Pe.r5); Pe["rk_r20"] = prk(-Pe.r20); Pe["rk_lowvol"] = prk(-Pe.vol20); Pe["rk_illiq"] = prk(Pe.amihud20); Pe["rk_small"] = prk(-Pe.log_amt20)
BASE = {"REV5": -Pe.r5, "REV20": -Pe.r20, "LOWVOL": -Pe.vol20, "ILLIQ(Amihud)": Pe.amihud20,
        "3F(REV5+LOWVOL+ILLIQ)": (Pe.rk_r5 + Pe.rk_lowvol + Pe.rk_illiq) / 3, "3F-size(REV5+LOWVOL+SMALL)": (Pe.rk_r5 + Pe.rk_lowvol + Pe.rk_small) / 3}
for k, v in BASE.items(): Pe["score_" + k] = v
ALLS = MOD + list(BASE.keys())
DEF = {"old_all": ("ret_old_rel", None), "old_exec_t1": ("ret_old_rel", "buyable_t1"), "new": ("ret_new_rel", None)}
def sample(defn):
    col, flt = DEF[defn]; m = Pe[col].notna()
    if flt: m &= Pe[flt]
    return m
SMP = {d: sample(d) for d in DEF}
dates = sorted(Pe.i.unique()); date_of = Pe.groupby("i").signal_date.first()
def ic_series(mask, score, ret, minn=8):
    """per-date Spearman RankIC (index = i)."""
    d = Pe[mask]; out = {}
    for i, x in d.groupby("i"):
        if len(x) >= minn: out[i] = Q.spearman(x[score].values, x[ret].values)
    return pd.Series(out)
def row(prefix, s): return {**prefix, **Q.summarize(s.values)}
ic_dates = []   # long table of per-date IC

# ============ 2/10: RankIC old vs new definition, six models + baselines ============
rows = []
for d in DEF:
    for m in ALLS:
        s = ic_series(SMP[d], "score_" + m, DEF[d][0]); rows.append(row(dict(definition=d, model=m, avg_n=float(Pe[SMP[d]].groupby("i").size().mean())), s))
        ic_dates.append(pd.DataFrame(dict(i=s.index, signal_date=date_of.loc[s.index].values, definition=d, model=m, ic=s.values)))
T1 = pd.DataFrame(rows); T1["p_bonf6"] = np.where(T1.model.isin(MOD), np.minimum(1, T1.hac_p * 6), np.nan)
T1.to_csv(R / "rankic_old_vs_new.csv", index=False, encoding="utf-8-sig")
ICD = pd.concat(ic_dates, ignore_index=True); ICD.to_csv(R / "rankic_by_date.csv", index=False, encoding="utf-8-sig")
# paired old - new
rows = []
for m in MOD:
    o = ICD[(ICD.model == m) & (ICD.definition == "old_all")].set_index("i").ic; n = ICD[(ICD.model == m) & (ICD.definition == "new")].set_index("i").ic
    dd = (o - n).dropna(); rows.append(row(dict(model=m, comparison="old_all - new"), dd))
pd.DataFrame(rows).to_csv(R / "rankic_old_minus_new_paired.csv", index=False, encoding="utf-8-sig")
print("sec2 done", round(time.time() - t0)); sys.stdout.flush()

# ---- decile helper ----
def deciles(mask, score, ret):
    """returns array [n_dates,10] of mean ret per within-date score decile (ordinal ranks; decile 10 = highest score) and dates index."""
    d = Pe[mask]; out = []; idx = []
    for i, x in d.groupby("i"):
        n = len(x)
        if n < 50: continue
        o = st.rankdata(x[score].values, method="ordinal"); dec = (((o - 1) * 10) // n).astype(int)
        r = x[ret].values; out.append(np.bincount(dec, weights=r, minlength=10) / np.bincount(dec, minlength=10)); idx.append(i)
    return np.array(out), np.array(idx)
DC = {}
for d in ["old_all", "new"]:
    for m in ALLS: DC[(d, m)] = deciles(SMP[d], "score_" + m, DEF[d][0])
rows_s, rows_d, rows_by = [], [], []
for (d, m), (A, idx) in DC.items():
    mu = A.mean(axis=0); rho_agg = st.spearmanr(np.arange(10), mu)[0]; rho_dates = np.array([st.spearmanr(np.arange(10), A[k])[0] for k in range(len(A))])
    steps = int((np.diff(mu) > 0).sum()); spread = A[:, 9] - A[:, 0]
    rows_s.append(row(dict(definition=d, model=m, spearman_decile_vs_mean=float(rho_agg), mean_date_spearman=float(np.nanmean(rho_dates)), steps_up_of_9=steps, D1_mean=float(mu[0]), D10_mean=float(mu[9]), D10_minus_pool=float(np.mean(A[:, 9])), **{f"D{k+1}": float(mu[k]) for k in range(10)}), pd.Series(spread)))
    sp = pd.DataFrame(dict(i=idx, spread=spread)); sp["signal_date"] = date_of.loc[idx].values; sp["year"] = sp.signal_date.dt.year; sp["half"] = sp.year.astype(str) + np.where(sp.signal_date.dt.month <= 6, "H1", "H2")
    for lvl in ["year", "half"]:
        for k, x in sp.groupby(lvl):
            s = x.spread.values; tt = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 and s.std() > 0 else np.nan
            rows_by.append(dict(definition=d, model=m, level=lvl, period=k, n_dates=len(s), spread_mean=float(s.mean()), naive_t=float(tt), pos_share=float((s > 0).mean())))
    for k in range(10): rows_d.append(dict(definition=d, model=m, decile=k + 1, mean_rel_ret=float(mu[k]), t_hac=Q.nw_t(A[:, k])[0], n_dates=len(A)))
    if d == "new": pd.DataFrame(dict(i=idx, signal_date=date_of.loc[idx].values, model=m, spread=spread)).to_csv(R / f"_tmp_spread_{m.replace('/', '_').replace('(', '').replace(')', '').replace('+', '_')}.csv", index=False)
DS = pd.DataFrame(rows_s); DS.to_csv(R / "decile_summary.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(rows_d).to_csv(R / "decile_means.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(rows_by).to_csv(R / "decile_spread_by_period.csv", index=False, encoding="utf-8-sig")
import glob, os
for f in glob.glob(str(R / "_tmp_spread_*.csv")): os.remove(f)
print("sec7 done", round(time.time() - t0)); sys.stdout.flush()

# ============ 4: group RankIC (new definition; groups as-of t; industry = cninfo LATEST, not point-in-time) ============
rows = []
for gv, label in [("g_liq", "流动性(amt20 日内三分位)"), ("g_amt_abs", "流动性(amt20 绝对档)"), ("g_vol", "波动率(vol20 日内三分位)"), ("g_exch", "交易所"), ("g_board", "板块"), ("g_ind", "行业(cninfo最新,非时点)")]:
    for gr in sorted(Pe.loc[SMP["new"], gv].dropna().astype(str).unique()):
        mask = SMP["new"] & (Pe[gv].astype(str) == gr)
        cnt = Pe[mask].groupby("i").size(); okd = cnt[cnt >= 15].index
        for m in MOD:
            s = ic_series(mask & Pe.i.isin(okd), "score_" + m, "ret_new_rel", minn=15)
            r = row(dict(group_var=label, group=gr, model=m, avg_n_per_date=float(cnt.mean()), dates_with_n_ge_15=int(len(okd))), s)
            if len(s) < 30: r.update(hac_t=np.nan, hac_p=np.nan, ci_lo=np.nan, ci_hi=np.nan); r["note"] = "日数<30,不报告t/CI"
            rows.append(r)
G = pd.DataFrame(rows); G.to_csv(R / "rankic_by_group.csv", index=False, encoding="utf-8-sig")
# completeness: groups partition the sample
comp = []
for gv in ["g_liq", "g_amt_abs", "g_vol", "g_exch", "g_board", "g_ind"]:
    comp.append(dict(group_var=gv, n_sample=int(SMP["new"].sum()), n_in_groups=int(Pe.loc[SMP["new"], gv].notna().sum()), n_missing=int(Pe.loc[SMP["new"], gv].isna().sum())))
pd.DataFrame(comp).to_csv(R / "group_completeness.csv", index=False, encoding="utf-8-sig")
print("sec4 done", round(time.time() - t0)); sys.stdout.flush()

# ============ 5: daily partial RankIC ============
CS = {"S1: r5,r20": ["r5", "r20"], "S2: +vol20": ["r5", "r20", "vol20"], "S3: +turn20": ["r5", "r20", "vol20", "turn20"], "S4: +amihud20(四项全控)": ["r5", "r20", "vol20", "turn20", "amihud20"],
      "S5: S4+log_amt20": ["r5", "r20", "vol20", "turn20", "amihud20", "log_amt20"], "S3F: r5,vol20,amihud20(三因子分量)": ["r5", "vol20", "amihud20"]}
rows = []; pic_dates = []
sm = Pe[SMP["new"]]; grp = {i: x for i, x in sm.groupby("i")}
for sname, cols in CS.items():
    for m in MOD:
        pc, fm, raw, nn = [], [], [], []; ii = []
        for i, x in grp.items():
            msk = x[cols].notna().all(axis=1).values
            if msk.sum() < 100: continue
            xs = x[msk]; p_, b_, n_ = Q.partial_spearman(xs["score_" + m].values, xs.ret_new_rel.values, xs[cols].values)
            pc.append(p_); fm.append(b_); nn.append(n_); raw.append(Q.spearman(xs["score_" + m].values, xs.ret_new_rel.values)); ii.append(i)
        pc, fm, raw = np.array(pc), np.array(fm), np.array(raw)
        r = row(dict(control_set=sname, model=m, avg_n=float(np.mean(nn))), pd.Series(pc)); r["raw_ic_same_sample_mean"] = float(np.nanmean(raw)); r["partial_over_raw"] = r["mean"] / r["raw_ic_same_sample_mean"]
        r["fm_slope_mean"] = float(np.nanmean(fm)); r["fm_slope_hac_t"] = Q.nw_t(fm)[0]; rows.append(r)
        pic_dates.append(pd.DataFrame(dict(i=ii, signal_date=date_of.loc[ii].values, control_set=sname, model=m, partial_ic=pc, raw_ic=raw)))
PIC = pd.DataFrame(rows); PIC.to_csv(R / "partial_rankic.csv", index=False, encoding="utf-8-sig")
pd.concat(pic_dates).to_csv(R / "partial_rankic_by_date.csv", index=False, encoding="utf-8-sig")
# same for baselines (does reversal survive the same controls? shows what the controls absorb)
rows = []
for sname in ["S1: r5,r20", "S4: +amihud20(四项全控)"]:
    cols = CS[sname]
    for m in ["REV5", "REV20", "3F(REV5+LOWVOL+ILLIQ)"]:
        pc = []
        for i, x in grp.items():
            msk = x[cols].notna().all(axis=1).values
            if msk.sum() < 100: continue
            xs = x[msk]; pc.append(Q.partial_spearman(xs["score_" + m].values, xs.ret_new_rel.values, xs[cols].values)[0])
        rows.append(row(dict(control_set=sname, model=m), pd.Series(pc)))
pd.DataFrame(rows).to_csv(R / "partial_rankic_baselines.csv", index=False, encoding="utf-8-sig")
print("sec5 done", round(time.time() - t0)); sys.stdout.flush()

# ============ 6: baseline comparison (paired per-date differences, new definition) + score correlation ============
rows = []
INew = ICD[ICD.definition == "new"].pivot(index="i", columns="model", values="ic")
for m in MOD:
    for b in BASE:
        dd = (INew[m] - INew[b]).dropna(); rows.append(row(dict(model=m, baseline=b, model_mean_ic=float(INew[m].mean()), baseline_mean_ic=float(INew[b].mean())), dd))
pd.DataFrame(rows).to_csv(R / "baseline_paired_ic.csv", index=False, encoding="utf-8-sig")
cr = []
for m in MOD:
    for b in BASE:
        s = [Q.spearman(x["score_" + m].values, x["score_" + b].values) for i, x in grp.items()]
        cr.append(dict(model=m, baseline=b, mean_rank_corr=float(np.nanmean(s))))
pd.DataFrame(cr).to_csv(R / "score_corr_with_baselines.csv", index=False, encoding="utf-8-sig")
print("sec6 done", round(time.time() - t0)); sys.stdout.flush()

# ============ 8: subsets (new definition): exclude low liquidity / limit-up lock at t / ST-name / delisted-in-sample ============
SUB = {"全体 eligible(基准)": pd.Series(True, index=Pe.index),
       "剔除低流动性(amt20 日内最低三分位)": Pe.g_liq != "L1低流动性",
       "剔除 amt20<1e8 (5倍入池门槛)": Pe.amt20 >= 1e8,
       "剔除 t 日收盘涨停封板": ~Pe.lim_up_t,
       "剔除名称含ST(现名,非时点)": ~Pe.is_st_name_current,
       "剔除退市样本股(用到未来退市信息,仅诊断)": ~Pe.delisted_in_sample,
       "组合干净集:非低流动性+非t日涨停+非ST名": (Pe.g_liq != "L1低流动性") & ~Pe.lim_up_t & ~Pe.is_st_name_current,
       "仅低流动性三分位(对照)": Pe.g_liq == "L1低流动性", "仅高流动性三分位": Pe.g_liq == "L3高流动性"}
rows = []
for sn, sm_ in SUB.items():
    mask = SMP["new"] & sm_
    for m in MOD:
        s = ic_series(mask, "score_" + m, "ret_new_rel"); A, idx = deciles(mask, "score_" + m, "ret_new_rel"); sp = pd.Series(A[:, 9] - A[:, 0])
        r = row(dict(subset=sn, model=m, avg_n=float(Pe[mask].groupby("i").size().mean())), s)
        r.update({"spread_mean": float(sp.mean()), "spread_hac_t": Q.nw_t(sp.values)[0], "spread_ci_lo": Q.block_ci(sp.values)[0], "spread_ci_hi": Q.block_ci(sp.values)[1], "D10_mean": float(A[:, 9].mean()), "D1_mean": float(A[:, 0].mean())})
        rows.append(r)
pd.DataFrame(rows).to_csv(R / "subset_robustness.csv", index=False, encoding="utf-8-sig")
# execution diagnostics by score decile (share not buyable etc.; old-definition phantom return)
rows = []
for m in MOD:
    d = Pe[SMP["old_all"]]
    for i, x in d.groupby("i"):
        pass
    Pm = Pe[Pe.ret_old_rel.notna()].copy(); o = Pm.groupby("i")["score_" + m].rank(method="first"); n = Pm.groupby("i")["score_" + m].transform("count"); Pm["dec"] = ((o - 1) * 10 // n).astype(int) + 1
    gg = Pm.groupby("dec").agg(share_not_buyable_t1=("buyable_t1", lambda s: 1 - s.mean()), share_lim_up_at_t=("lim_up_t", "mean"), share_st_name=("is_st_name_current", "mean"), share_delisted_stock=("delisted_in_sample", "mean"),
                              share_entry_delayed=("entry_delay", lambda s: (s > 0).mean()), share_never_executable=("ret_new", lambda s: s.isna().mean()), mean_log_amt20=("log_amt20", "mean"))
    gg["model"] = m; rows.append(gg.reset_index())
pd.concat(rows).to_csv(R / "execution_by_decile.csv", index=False, encoding="utf-8-sig")
print("sec8 done", round(time.time() - t0)); sys.stdout.flush()

# ============ 9: factor-adjusted attribution ============
def fac_returns():
    out = {}
    d = Pe[SMP["new"]]
    spec = {"SMB(小-大,amt20)": ("amt20", -1), "REV5(低r5-高r5)": ("r5", -1), "LVOL(低-高vol20)": ("vol20", -1), "MOM60(高-低r60)": ("r60", 1), "ILLIQ(高-低Amihud)": ("amihud20", 1)}
    for k, (c, sgn) in spec.items():
        v = []
        for i, x in d.groupby("i"):
            x = x[x[c].notna()]; n = len(x); r = x[c].rank(method="first").values / n
            hi, lo = (r > 0.7), (r <= 0.3)
            v.append((i, (x.ret_new_rel.values[hi].mean() - x.ret_new_rel.values[lo].mean()) * sgn))
        out[k] = pd.Series(dict(v))
    return pd.DataFrame(out)
FR = fac_returns(); FR.to_csv(R / "factor_returns_by_date.csv", encoding="utf-8-sig")
def ols_hac(y, Xf, lag=Q.HAC_LAG):
    y = np.asarray(y, float); Z = np.column_stack([np.ones(len(y)), Xf]); n, k = Z.shape
    beta = np.linalg.solve(Z.T @ Z, Z.T @ y); e = y - Z @ beta; U = Z * e[:, None]; S = U.T @ U
    for l in range(1, lag + 1): w = 1 - l / (lag + 1); G = U[l:].T @ U[:-l]; S += w * (G + G.T)
    Ai = np.linalg.inv(Z.T @ Z); cov = Ai @ S @ Ai; se = np.sqrt(np.diag(cov)); r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return beta, beta / se, r2
def alpha_ci(y, Xf, B=2000, block=Q.BLOCK, seed=Q.SEED_BOOT):
    n = len(y); rng = np.random.default_rng(seed); nb = int(np.ceil(n / block)); s = rng.integers(0, n - block + 1, size=(B, nb)); idx = (s[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    Z = np.column_stack([np.ones(n), Xf]); al = np.empty(B)
    for b in range(B):
        zb = Z[idx[b]]; al[b] = np.linalg.lstsq(zb, y[idx[b]], rcond=None)[0][0]
    return float(np.percentile(al, 2.5)), float(np.percentile(al, 97.5))
rows = []
spreads = {}
for m in ALLS:
    A, idx = DC[("new", m)]; spreads[m] = (pd.Series(A[:, 9] - A[:, 0], index=idx), pd.Series(A[:, 9] - A[:, :].mean(axis=1), index=idx))
for m in ALLS:
    for tag, y in zip(["D10-D1多空价差", "D10相对池均值(多头超额)"], spreads[m]):
        for fname, cols in [("5因子", list(FR.columns)), ("3因子(REV5,LVOL,ILLIQ)", ["REV5(低r5-高r5)", "LVOL(低-高vol20)", "ILLIQ(高-低Amihud)"]), ("无因子(均值)", [])]:
            dfm = pd.concat([y.rename("y"), FR[cols]], axis=1).dropna() if cols else y.rename("y").to_frame()
            yy = dfm.y.values; Xf = dfm[cols].values if cols else np.zeros((len(yy), 0))
            beta, t, r2 = ols_hac(yy, Xf); lo, hi = alpha_ci(yy, Xf)
            r = dict(model=m, dependent=tag, spec=fname, n_dates=len(yy), alpha=float(beta[0]), alpha_hac_t=float(t[0]), alpha_ci_lo=lo, alpha_ci_hi=hi, r2=float(r2) if cols else 0.0)
            for j, c in enumerate(cols): r["beta_" + c] = float(beta[j + 1]); r["t_" + c] = float(t[j + 1])
            rows.append(r)
pd.DataFrame(rows).to_csv(R / "factor_attribution.csv", index=False, encoding="utf-8-sig")
print("sec9 done", round(time.time() - t0)); sys.stdout.flush()
# 24-sample summary of headline numbers for the report
json.dump(dict(sample=dict(n_eligible=int(len(Pe)), n_new=int(SMP["new"].sum()), n_old=int(SMP["old_all"].sum()), n_old_exec=int(SMP["old_exec_t1"].sum()), n_dates=int(len(dates))), elapsed=round(time.time() - t0)), open(R / "eval_meta.json", "w", encoding="utf-8"), ensure_ascii=False)
print("all done", round(time.time() - t0))