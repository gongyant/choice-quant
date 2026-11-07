# -*- coding: utf-8 -*-
"""P4 batch 1 evaluation: same universe / signal dates / execution / costs for P3 v2 (frozen), E0 (original P1 LGBM), simple factors and E1 models.
All results EXPLORATORY (2022-07..2026-09 has been used repeatedly). Writes results/*.csv."""
import json, pickle, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import p4lib as P
t0 = time.time(); C = P.P4 / "cache"; R = P.P4 / "results"; R.mkdir(exist_ok=True); PR = P.P4 / "preds"
fl = np.load(C / "flags.npz"); LAB = pickle.load(open(C / "labels.pkl", "rb"))["lab"]; X = pickle.load(open(C / "world.pkl", "rb"))
cal = pd.DatetimeIndex(fl["cal"]); codes = list(fl["codes"]); col = {c: j for j, c in enumerate(codes)}; elig, amt20, olu, old, st = fl["elig"], fl["amt20"], fl["olu"], fl["old"], fl["st"]
names = X.u.name.values; N = len(codes)
H = pd.read_parquet(P.V2 / "p1_dl" / "historical_predictions_all_models.parquet", columns=["signal_date", "code", "model", "score", "eligible_asof"]); H = H[H.eligible_asof]
sig = sorted(H.signal_date.unique()); sig_idx = [int(cal.get_loc(pd.Timestamp(s))) for s in sig]; DESIGN_END = "2024-12-31"
def mat(df, col_="score"):
    M = np.full((len(sig), N), np.nan); k = {s: q for q, s in enumerate(sig)}; r = df.signal_date.map(k).values; c = df.code.map(col).values; ok = ~pd.isna(c)
    M[r[ok].astype(int), c[ok].astype(int)] = df[col_].values[ok]; return M
S = {}
w = H[H.model.isin(["HGB", "LGBM", "MLP", "GRU", "TF"])].pivot_table(index=["signal_date", "code"], columns="model", values="score").dropna()
comp = w.groupby(level=0).rank(pct=True).mean(1).rename("score").reset_index()
S["E0_LGBM原模型"] = {5: mat(H[H.model == "LGBM"]), 20: mat(H[H.model == "LGBM"])}
S["P3综合分(5模型均值,无规则)"] = {5: mat(comp), 20: mat(comp)}
Xq = np.load(C / "Xq_E1.npy", mmap_mode="r"); fn = pickle.load(open(C / "featnames.pkl", "rb"))["E1"]
def fmat(expr):
    M = np.full((len(sig), N), np.nan)
    for q, i in enumerate(sig_idx): M[q, elig[i]] = expr(i)[elig[i]]
    return M
iv, ir5, ia = fn.index("vol20"), fn.index("r5"), fn.index("amihud20")
lowvol = fmat(lambda i: -np.asarray(Xq[i][:, iv])); f3 = fmat(lambda i: (-np.asarray(Xq[i][:, ir5]) - np.asarray(Xq[i][:, iv]) + np.asarray(Xq[i][:, ia])) / 3)
S["因子_低波动"] = {5: lowvol, 20: lowvol}; S["因子_F3(反转+低波+非流动)"] = {5: f3, 20: f3}
for m, nm in [("E1reg", "E1_LGBM回归(增强因子)"), ("E1lrank", "E1_LambdaRank"), ("E1xendcg", "E1_rank_xendcg")]:
    S[nm] = {h: mat(pd.read_parquet(PR / f"{m}_h{h}.parquet")) for h in [5, 20] if (PR / f"{m}_h{h}.parquet").exists()}
p3 = pd.read_csv(P.P3 / "results" / "hist_top5_picks.csv", dtype={"code": str}, encoding="utf-8-sig"); p3 = p3[p3.strategy == "v2"].sort_values(["signal_date", "rank_in_pick"])
p3picks = {s: [col[c] for c in g.code] for s, g in p3.groupby("signal_date")}
def filt(i): return elig[i] & X.traded[i] & ~X.lim_up[i] & (np.nan_to_num(amt20[i]) >= 1e8) & ~st[i]
def picks_from(M, rows, K):
    out = []
    for q in rows:
        i = sig_idx[q]; s = np.where(filt(i), M[q], np.nan); ok = np.where(np.isfinite(s))[0]; out.append(ok[np.argsort(-s[ok], kind="stable")][:K])
    return out
per_all, hold_all, summ, icrows = [], [], [], []
for h in [5, 20]:
    rows = list(range(0, len(sig), 1 if h == 5 else 4)); si = [sig_idx[q] for q in rows]; bench = P.pool_bench(X, si, h, elig, olu)
    ex, gr = LAB[h]["ex"], LAB[h]["gr"]
    strat = [("P3_v2冻结基线", None)] + [(k, v.get(h)) for k, v in S.items() if v.get(h) is not None]
    for nm, M in strat:
        # ranking quality on the full eligible pool (all 207 dates)
        if M is not None:
            ics, nd = [], []
            for q, i in enumerate(sig_idx):
                m = elig[i] & np.isfinite(M[q]) & np.isfinite(ex[i])
                if m.sum() < 50: continue
                ics.append(spearmanr(M[q][m], ex[i][m]).statistic); nd.append(P.ndcg_at(M[q][m], gr[i][m], 5))
            ics = np.array(ics); icrows.append(dict(strategy=nm, horizon=h, rankic_mean=ics.mean(), rankic_ir=ics.mean() / ics.std(), rankic_pos=(ics > 0).mean(), ndcg5=np.nanmean(nd), n_dates=len(ics)))
        for K in ([5] if M is None else [5, 10]):
            pk = [p3picks.get(sig[q], []) for q in rows] if M is None else picks_from(M, rows, K)
            pr, ho = P.backtest(X, si, pk, h, K, 0.003, skip=st, olu=olu, old=old, names=names, strategy=nm)
            pr["horizon"], pr["K"], pr["pool_ret"] = h, K, bench; ho["horizon"], ho["K"] = h, K
            per_all.append(pr); hold_all.append(ho)
            for per_nm, msk in [("全期", np.ones(len(pr), bool)), ("设计期2022-07~2024-12", pr.signal_date <= DESIGN_END), ("探索期2025-01~2026-09", pr.signal_date > DESIGN_END)]:
                a = pr[msk]
                summ.append(dict(strategy=nm, horizon=h, K=K, period=per_nm, n_periods=len(a), avg_holdings=a.n_held.mean(), mean_gross=a.ret_gross.mean(), mean_net=a.ret_net.mean(),
                                 cum_gross=np.prod(1 + a.ret_gross) - 1, cum_net=np.prod(1 + a.ret_net) - 1, pool_mean=a.pool_ret.mean(), cum_pool=np.prod(1 + a.pool_ret) - 1,
                                 mean_excess_net=(a.ret_net - a.pool_ret).mean(), beat_pool_share=(a.ret_net > a.pool_ret).mean(), mdd_net=P.mdd(a.ret_net), mdd_pool=P.mdd(a.pool_ret),
                                 avg_turnover=a.turnover.mean(), total_fees=a.fees.sum(), t_excess=(a.ret_net - a.pool_ret).mean() / ((a.ret_net - a.pool_ret).std() / np.sqrt(len(a)))))
PER = pd.concat(per_all); HO = pd.concat(hold_all); SU = pd.DataFrame(summ); IC = pd.DataFrame(icrows)
PER.to_csv(R / "bt_periods_e0_e1.csv", index=False, encoding="utf-8-sig"); HO.to_csv(R / "bt_holdings_e0_e1.csv", index=False, encoding="utf-8-sig")
SU.to_csv(R / "bt_summary_e0_e1.csv", index=False, encoding="utf-8-sig"); IC.to_csv(R / "rank_quality_e0_e1.csv", index=False, encoding="utf-8-sig")
# ---- why positive RankIC but weak Top5 (h=5, E0 and P3 composite) ----
ex5 = LAB[5]["ex"]; D = []
for nm in ["E0_LGBM原模型", "P3综合分(5模型均值,无规则)", "E1_LambdaRank"]:
    if nm not in S or 5 not in S[nm]: continue
    M = S[nm][5]
    for q, i in enumerate(sig_idx):
        m = np.where(elig[i] & np.isfinite(M[q]) & np.isfinite(ex5[i]))[0]
        if len(m) < 100: continue
        o = m[np.argsort(-M[q][m], kind="stable")]; e = ex5[i]; n = len(o)
        row = dict(strategy=nm, signal_date=sig[q], n_pool=n, top5=e[o[:5]].mean(), top10=e[o[:10]].mean(), top50=e[o[:50]].mean(), top_decile=e[o[:n // 10]].mean(), bottom_decile=e[o[-(n // 10):]].mean(),
                   top5_sd_within=e[o[:5]].std(), pool_sd=e[m].std(), top_decile_sd_within=e[o[:n // 10]].std(),
                   top5_tradable_filtered=np.nanmean(e[[j for j in o if filt(i)[j]][:5]]))
        for b in range(10): row[f"dec{b+1}"] = e[o[b * n // 10:(b + 1) * n // 10]].mean()
        D.append(row)
D = pd.DataFrame(D); D.to_csv(R / "why_ic_vs_top5_by_date.csv", index=False, encoding="utf-8-sig")
W = D.groupby("strategy").agg(**{c: (c, "mean") for c in ["top5", "top10", "top50", "top_decile", "bottom_decile", "top5_tradable_filtered", "top5_sd_within", "top_decile_sd_within", "pool_sd"] + [f"dec{b+1}" for b in range(10)]},
                              top5_sd_across=("top5", "std"), top_decile_sd_across=("top_decile", "std"), top5_pos_share=("top5", lambda x: (x > 0).mean()), top_decile_pos_share=("top_decile", lambda x: (x > 0).mean())).reset_index()
W.to_csv(R / "why_ic_vs_top5_summary.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(SU[SU.period == "全期"].drop(columns=["period"]).round(4).to_string()); print(IC.round(4).to_string()); print(W.round(4).T.to_string()); print("secs", round(time.time() - t0))
