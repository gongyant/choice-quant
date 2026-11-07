# -*- coding: utf-8 -*-
"""P3-4: empirical scenario calibration from HISTORICAL OOS walk-forward predictions (p1_dl/historical_predictions_all_models.parquet, 207 signal dates 2022-07..2026-09).
composite = mean of per-date pool percentiles of HGB/LGBM/MLP/GRU/TF (TF = seed 42 only, same as the latest scoring). Deciles per date (10 = highest).
5d target = P0 label realised (t+1 close -> t+6 close) relative to the same-day pool mean; 20d target computed independently from adjusted prices:
t+1 close -> t+21 close, minus the same-day pool mean. Classes vs fixed thresholds (5d +-2%, 20d +-5%). Block bootstrap over signal dates;
20d also on a non-overlapping subsample (every 4th signal date). These are EMPIRICAL RELATIVE FREQUENCIES, not probabilities."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; V2 = H.parent; OUT = H / "results"; OUT.mkdir(exist_ok=True)
RULES = json.load(open(H / "selection_rules.json", encoding="utf-8")); X5, X20 = RULES["calibration"]["thresholds"]["5d"], RULES["calibration"]["thresholds"]["20d"]
FIVE = ["HGB", "LGBM", "MLP", "GRU", "TF"]; NB = 10; B = 2000; RNG = np.random.default_rng(20261021)
def load(date):
    h = pd.read_parquet(V2 / "p1_dl" / "historical_predictions_all_models.parquet", columns=["signal_date", "code", "model", "score", "eligible_asof", "realized_ret_5d", "realized_rel_ret_5d"])
    h = h[h.eligible_asof & h.model.isin(FIVE)]
    w = h.pivot_table(index=["signal_date", "code"], columns="model", values="score"); w = w.dropna()
    pc = w.groupby(level=0).rank(pct=True); comp = pc.mean(axis=1).rename("composite")
    base = h[h.model == "HGB"].set_index(["signal_date", "code"])[["realized_ret_5d", "realized_rel_ret_5d"]]
    D = pd.concat([comp, base], axis=1, join="inner").reset_index()
    D["bin"] = D.groupby("signal_date").composite.transform(lambda s: np.ceil(s.rank(method="first") / len(s) * NB).clip(1, NB)).astype(int)
    z = np.load(H / "data" / f"world_{date}.npz"); cal = pd.DatetimeIndex(z["cal_full"]); A = z["ADJf_full"].astype(float); E = z["elig_full"]; codes = list(z["codes"]); col = {c: j for j, c in enumerate(codes)}
    r20 = {}
    for sd, g in D.groupby("signal_date"):
        i = cal.get_loc(pd.Timestamp(sd))
        if i + 21 > len(cal) - 1: continue
        j = np.array([col[c] for c in g.code]); ret = A[i + 21, j] / A[i + 1, j] - 1
        r20[sd] = pd.Series(ret, index=g.index)
    D["ret_20d"] = pd.concat(r20.values()) if r20 else np.nan
    D["rel_20d"] = D.ret_20d - D.groupby("signal_date").ret_20d.transform("mean")
    D["year"] = D.signal_date.str[:4]; return D
def cls(rel, x): return np.select([rel > x, rel < -x], [1, -1], 0)
def boot_diff(D, rcol, x, block):
    """per-date sums -> moving-block bootstrap over the date sequence. Returns CIs of bin freq_out, (bin - all) for freq_out / freq_under / mean rel."""
    d = D.dropna(subset=[rcol]).copy(); d["out"] = (d[rcol] > x).astype(float); d["und"] = (d[rcol] < -x).astype(float); dates = sorted(d.signal_date.unique()); nd = len(dates)
    g = d.groupby(["signal_date", "bin"]); agg = {k: g[c].sum().unstack(fill_value=0).reindex(dates).values.astype(float) for k, c in [("o", "out"), ("u", "und"), ("r", rcol)]}
    Nn = g.size().unstack(fill_value=0).reindex(dates).values.astype(float)
    nb = int(np.ceil(nd / block)); fb, do, du, dr = [], [], [], []
    for _ in range(B):
        st = RNG.integers(0, nd - block + 1, nb); idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:nd]; n = Nn[idx].sum(0)
        o, u, r = (agg[k][idx].sum(0) for k in "our"); fb.append(o / n); do.append(o / n - o.sum() / n.sum()); du.append(u / n - u.sum() / n.sum()); dr.append(r / n - r.sum() / n.sum())
    P = lambda a: np.percentile(np.array(a), [2.5, 97.5], axis=0)
    return P(fb), P(do), nd, P(du), P(dr)
def table(D, hz):
    rcol, acol, x, block = ("realized_rel_ret_5d", "realized_ret_5d", X5, 4) if hz == "5d" else ("rel_20d", "ret_20d", X20, 8)
    d = D.dropna(subset=[rcol]); c = cls(d[rcol].values, x); d = d.assign(c=c)
    base = dict(out=(d.c == 1).mean(), flat=(d.c == 0).mean(), under=(d.c == -1).mean())
    fci, dci, nd, uci, rci = boot_diff(D, rcol, x, block); rows = []; mrel_all = d[rcol].mean()
    for b in range(1, NB + 1):
        g = d[d.bin == b]; q = g[acol].quantile([0.1, 0.5, 0.9]).values
        rows.append(dict(horizon=hz, bin=b, n_obs=len(g), n_dates=nd, threshold=x, freq_outperform=(g.c == 1).mean(), freq_flat=(g.c == 0).mean(), freq_underperform=(g.c == -1).mean(),
                         freq_out_ci_lo=fci[0, b - 1], freq_out_ci_hi=fci[1, b - 1], diff_vs_base=(g.c == 1).mean() - base["out"], diff_ci_lo=dci[0, b - 1], diff_ci_hi=dci[1, b - 1],
                         base_outperform=base["out"], base_flat=base["flat"], base_underperform=base["under"], mean_rel=g[rcol].mean(), abs_q10=q[0], abs_q50=q[1], abs_q90=q[2],
                         under_minus_base=(g.c == -1).mean() - base["under"], under_diff_ci_lo=uci[0, b - 1], under_diff_ci_hi=uci[1, b - 1], mean_rel_minus_all=g[rcol].mean() - mrel_all, mean_rel_diff_ci_lo=rci[0, b - 1], mean_rel_diff_ci_hi=rci[1, b - 1], rel_q10=g[rcol].quantile(0.1), rel_q50=g[rcol].quantile(0.5), rel_q90=g[rcol].quantile(0.9), block_len_signals=block))
    T = pd.DataFrame(rows)
    sig = lambda lo, hi: (lo > 0) | (hi < 0)
    T["supplement"] = np.where(sig(T.under_diff_ci_lo, T.under_diff_ci_hi) | sig(T.mean_rel_diff_ci_lo, T.mean_rel_diff_ci_hi), "补充(事后加列,非预注册):跑输频率或平均相对收益与全样本差异显著", "补充:跑输频率与平均相对收益差异也不显著")
    T["credibility"] = np.where((T.diff_ci_lo > 0) | (T.diff_ci_hi < 0), "有历史统计支持(仍可能失效)", "预测可信度低"); return T
def nonoverlap20(D):
    d = D.dropna(subset=["rel_20d"]); dates = sorted(d.signal_date.unique()); keep = set(dates[::4]); d = d[d.signal_date.isin(keep)]; d = d.assign(out=d.rel_20d > X20)
    base = d.out.mean(); cnt = d.groupby(["signal_date", "bin"]).out.agg(["sum", "count"]).unstack(fill_value=0); S, Nn = cnt["sum"].values, cnt["count"].values; nd = len(S); diffs = []
    for _ in range(B):
        idx = RNG.integers(0, nd, nd); s, n = S[idx].sum(0), Nn[idx].sum(0); diffs.append(s / n - s.sum() / n.sum())
    lo, hi = np.percentile(np.array(diffs), [2.5, 97.5], axis=0); f = d.groupby("bin").out.mean()
    return pd.DataFrame(dict(horizon="20d_nonoverlap", bin=range(1, NB + 1), n_dates=nd, freq_outperform=f.values, base_outperform=base, diff_vs_base=f.values - base, diff_ci_lo=lo, diff_ci_hi=hi))
def main(date):
    D = load(date); T5, T20 = table(D, "5d"), table(D, "20d"); NO = nonoverlap20(D)
    T20 = T20.merge(NO[["bin", "diff_vs_base", "diff_ci_lo", "diff_ci_hi"]].rename(columns=lambda c: c if c == "bin" else "nonoverlap_" + c), on="bin")
    T20["credibility"] = np.where(((T20.diff_ci_lo > 0) & (T20.nonoverlap_diff_ci_lo > 0)) | ((T20.diff_ci_hi < 0) & (T20.nonoverlap_diff_ci_hi < 0)), "有历史统计支持(仍可能失效)", "预测可信度低")
    T = pd.concat([T5, T20], ignore_index=True); T.to_csv(OUT / "calibration_table.csv", index=False, encoding="utf-8-sig")
    Y = []
    for hz, rc, x in [("5d", "realized_rel_ret_5d", X5), ("20d", "rel_20d", X20)]:
        d = D.dropna(subset=[rc]); g = d.assign(out=d[rc] > x).groupby(["year", "bin"]).out.mean().unstack(); g["horizon"] = hz; g["base"] = d.assign(out=d[rc] > x).groupby("year").out.mean(); g["n_dates"] = d.groupby("year").signal_date.nunique(); Y.append(g.reset_index())
    Y = pd.concat(Y); Y.to_csv(OUT / "calibration_by_year.csv", index=False, encoding="utf-8-sig")
    ic = D.groupby("signal_date").apply(lambda g: g.composite.corr(g.realized_rel_ret_5d, method="spearman")); ic20 = D.dropna(subset=["rel_20d"]).groupby("signal_date").apply(lambda g: g.composite.corr(g.rel_20d, method="spearman"))
    summ = dict(n_signal_dates=int(D.signal_date.nunique()), first=D.signal_date.min(), last=D.signal_date.max(), n_obs=len(D), n_dates_20d=int(D.dropna(subset=["rel_20d"]).signal_date.nunique()),
                composite_rankic_5d_mean=float(ic.mean()), composite_rankic_5d_pos_share=float((ic > 0).mean()), composite_rankic_20d_mean=float(ic20.mean()), composite_rankic_20d_pos_share=float((ic20 > 0).mean()),
                thresholds=dict(d5=X5, d20=X20), note="经验频率 = 历史上该组股票相对池均值 跑赢/持平/跑输 的比例;不是概率,不保证未来;2022-07 以后样本已被 P0/P1/P2 反复使用,属探索性")
    json.dump(summ, open(OUT / "calibration_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str); print(json.dumps(summ, ensure_ascii=False, default=str))
    print(T[["horizon", "bin", "n_obs", "freq_outperform", "freq_underperform", "diff_vs_base", "diff_ci_lo", "diff_ci_hi", "abs_q10", "abs_q50", "abs_q90", "credibility"]].round(4).to_string())
if __name__ == "__main__": main(sys.argv[1] if len(sys.argv) > 1 else "20261021")