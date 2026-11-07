# -*- coding: utf-8 -*-
"""P3-B1: apply selection_rules_v2.json (registered in rules_manifest.jsonl BEFORE this script ran) to (a) the latest date and (b) all 207 historical OOS signal dates.
Never modifies v1 outputs or the registry. Outputs *_v2 files + results/hist_* tables."""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; V2 = H.parent; R = H / "results"; sys.path.insert(0, str(H)); import p3_calib as K
FIVE = ["HGB", "LGBM", "MLP", "GRU", "TF"]; DATE = "20261021"; DESIGN_END = "2024-12-31"; COST = [0.0, 0.003, 0.006]
def check_rules():
    m = [json.loads(l) for l in open(H / "rules_manifest.jsonl", encoding="utf-8")][-1]; assert m["file"] == "selection_rules_v2.json"
    assert hashlib.sha256((H / "selection_rules_v2.json").read_bytes()).hexdigest() == m["sha256"]; return m
def trend(r20, vol, slope, cvm):
    b = np.maximum(0.05, np.nan_to_num(vol) * np.sqrt(20)); up = (r20 > b) & (slope > 0) & (cvm > 0); dn = (r20 < -b) & (slope < 0) & (cvm < 0)
    return np.where(up, "上涨", np.where(dn, "下跌", "震荡"))
def rule_masks(d, st=None):
    """d needs: elig, amt20, lim_up_t, lim_dn_t, traded_t, n_lim_dn_5d, n_lim_up_5d, n_ge70, disagree, r5, r20, trend, ma20_slope_5d"""
    base = d.elig & (d.amt20 >= 1e8) & ~d.lim_up_t & d.traded_t & ~d.lim_dn_t & (d.n_lim_dn_5d < 2) & (d.n_ge70 >= 4) & (d.disagree <= 0.15) & (d.r5 <= 0.15) & ~((d.trend == "下跌") & (d.r5 < -0.08))
    if st is not None: base &= ~st
    v1 = base & (d.r20 <= 0.30); T1 = (d.r20 < -0.15) & (d.ma20_slope_5d < 0); T2 = d.r20 > 0.20; T3 = d.n_lim_up_5d >= 2
    return base, v1, v1 & ~T1 & ~T2 & ~T3, v1 & T1 & ~T2 & ~T3, T1, T2, T3
def pick(d, mask, k=5): x = d[mask].sort_values(["sel", "amt20"], ascending=False); return x.head(k)
def latest(m):
    o = pd.read_csv(H / f"latest_research_ranking_{DATE}.csv", dtype={"code": str}, encoding="utf-8-sig")
    z = np.load(H / "data" / f"world_{DATE}.npz"); nlu = pd.Series(z["lim_up"][-5:].sum(0), index=list(z["codes"])); o["n_lim_up_5d"] = o.code.map(nlu)
    d = pd.DataFrame(dict(elig=o.eligible_asof, amt20=o.amt20, lim_up_t=o.limit_up_locked_t, lim_dn_t=o.limit_down_locked_t, traded_t=~o.suspended_t, n_lim_dn_5d=o.n_limit_down_5d, n_lim_up_5d=o.n_lim_up_5d,
                          n_ge70=o.n_models_ge70, disagree=o.model_disagreement, r5=o.chg_5d, r20=o.chg_20d, trend=o.trend, ma20_slope_5d=o.ma20_slope_5d)).fillna({"amt20": 0})
    base, v1, v2, watch, T1, T2, T3 = rule_masks(d, st=o.st_or_delist_risk); assert (v1 == o.passes_top_filters).all(), "v1 reproduction failed"
    o["sel"] = o.composite_pctile - 0.10 * o.vol20_pctile - 0.05 * (o.chg_20d > 0.15); o["passes_v2"] = v2; o["T1_down"], o["T2_chase20"], o["T3_limitups"] = T1, T2, T3
    top = pick(o, v2); wl = pick(o, watch); o["top5_v2"] = o.code.isin(top.code); o["watch_v2"] = o.code.isin(wl.code); o["avoid5_v2"] = o.avoid5
    o["conclusion_v2"] = np.where(o.top5_v2, "继续观察", np.where(o.watch_v2, "暂不买入(下跌观察)", np.where(o.avoid5_v2, "暂不买入(回避)", "")))
    o["rules_version"] = "p3-rules-v2"; o["rules_sha256"] = m["sha256"]
    o.to_csv(H / f"latest_research_ranking_{DATE}_v2.csv", index=False, encoding="utf-8-sig"); top.to_csv(R / f"top5_v2_{DATE}.csv", index=False, encoding="utf-8-sig")
    wl.to_csv(R / f"watch_v2_{DATE}.csv", index=False, encoding="utf-8-sig"); o[o.avoid5_v2].to_csv(R / f"avoid5_v2_{DATE}.csv", index=False, encoding="utf-8-sig")
    print("latest v1 pass", int(v1.sum()), "v2 pass", int(v2.sum()), "T1", int((v1 & T1).sum()), "T2", int((v1 & T2).sum()), "T3", int((v1 & T3).sum()))
    print(top[["code", "name", "close_unadj", "chg_5d", "chg_20d", "trend", "composite_pctile", "sel", "risk_level"]].round(3).to_string()); print(wl[["code", "name", "chg_20d", "trend"]].to_string())
def history():
    h = pd.read_parquet(V2 / "p1_dl" / "historical_predictions_all_models.parquet", columns=["signal_date", "code", "model", "score", "eligible_asof", "realized_ret_5d", "realized_rel_ret_5d"])
    h = h[h.eligible_asof & h.model.isin(FIVE)]; w = h.pivot_table(index=["signal_date", "code"], columns="model", values="score").dropna(); pc = w.groupby(level=0).rank(pct=True)
    d = pd.DataFrame(dict(composite=pc.mean(1), disagree=pc.std(1, ddof=0), n_ge70=(pc >= 0.70).sum(1))).reset_index()
    D = K.load(DATE)[["signal_date", "code", "realized_ret_5d", "realized_rel_ret_5d", "ret_20d", "rel_20d", "bin"]]; d = d.merge(D, on=["signal_date", "code"])
    P = pd.read_parquet(H / "data" / "feature_panel_hist.parquet"); d = d.merge(P, on=["signal_date", "code"], how="left")
    d["elig"] = True; d["amt20"] = d.amt20.fillna(0); d["traded_t"] = d.traded_t.fillna(False).astype(bool)
    for c in ["lim_up_t", "lim_dn_t"]: d[c] = d[c].fillna(False).astype(bool)
    d["vol_p"] = d.groupby("signal_date").vol20.rank(pct=True); d["trend"] = trend(d.r20.values, d.vol20.values, d.ma20_slope_5d.values, d.close_vs_ma20.values)
    d["sel"] = d.composite - 0.10 * d.vol_p - 0.05 * (d.r20 > 0.15)
    # F3 baseline (same definition as registry: mean pctile of -r5, -vol20, amihud), among the same liquid pool (amt20>=1e8, tradable, not limit-locked)
    for c, s in [("p_rev5", -d.r5), ("p_lowvol", -d.vol20), ("p_illiq", d.ami20)]: d[c] = s.groupby(d.signal_date).rank(pct=True)
    d["F3"] = d[["p_rev5", "p_lowvol", "p_illiq"]].mean(1)
    base, v1, v2, watch, T1, T2, T3 = rule_masks(d); d["v1"], d["v2"], d["T1"], d["T2"], d["T3"], d["base"] = v1, v2, T1, T2, T3, base
    liq = d.amt20 >= 1e8
    d["period"] = np.where(d.signal_date <= DESIGN_END, "设计期 2022-07~2024-12", "探索期 2025-01~2026-09(非盲测)")
    # constraint checks: among v1-passing names, excluded vs kept
    rows = []
    for per, g in [*d.groupby("period"), ("全期", d)]:
        g = g[g.v1]
        for nm, msk in [("T1 下跌(r20<-15%且MA20下行)", g.T1), ("T2 20日涨幅>20%", g.T2), ("T3 5日内涨停>=2", g.T3)]:
            for side, s in [("被排除", g[msk]), ("保留", g[~msk])]:
                for hz, c, x in [("5d", "realized_rel_ret_5d", 0.02), ("20d", "rel_20d", 0.05)]:
                    v = s[c].dropna(); rows.append(dict(period=per, constraint=nm, side=side, horizon=hz, n=len(v), under_share=(v < -x).mean(), mean_rel=v.mean(), median_rel=v.median(), out_share=(v > x).mean()))
    pd.DataFrame(rows).to_csv(R / "hist_v2_constraint_check.csv", index=False, encoding="utf-8-sig")
    # per-period top5 under v1 / v2 / F3 baseline / pool
    pool = d.groupby("signal_date").agg(pool_ret5=("realized_ret_5d", "mean"), pool_med5=("realized_ret_5d", "median"), pool_ret20=("ret_20d", "mean"))
    picks = []; per = []
    for sd, g in d.groupby("signal_date"):
        for nm, x in [("v1", pick(g, g.v1)), ("v2", pick(g, g.v2)), ("F3基线", g[liq.loc[g.index] & g.traded_t & ~g.lim_up_t].nlargest(5, "F3"))]:
            picks.append(x.assign(strategy=nm, rank_in_pick=np.arange(1, len(x) + 1))[["signal_date", "strategy", "rank_in_pick", "code", "composite", "sel", "F3", "r5", "r20", "trend", "realized_ret_5d", "realized_rel_ret_5d", "ret_20d", "rel_20d"]])
            per.append(dict(signal_date=sd, strategy=nm, n_picks=len(x), ret5=x.realized_ret_5d.mean() if len(x) else 0.0, rel5=x.realized_rel_ret_5d.mean() if len(x) else np.nan, ret20=x.ret_20d.mean() if len(x) else np.nan,
                            beat_med_share=(x.realized_ret_5d > pool.loc[sd, "pool_med5"]).mean() if len(x) else np.nan))
    PK = pd.concat(picks); PK.to_csv(R / "hist_top5_picks.csv", index=False, encoding="utf-8-sig")
    PR = pd.DataFrame(per).merge(pool, left_on="signal_date", right_index=True); PR["beat_pool"] = PR.ret5 > PR.pool_ret5
    for c in COST[1:]: PR[f"ret5_net_{int(c*1e4)}bp"] = np.where(PR.n_picks > 0, PR.ret5 - c, 0.0)
    PR["period"] = np.where(PR.signal_date <= DESIGN_END, "设计期", "探索期"); PR.to_csv(R / "hist_top5_by_period.csv", index=False, encoding="utf-8-sig")
    gap = pd.Series(sorted(d.signal_date.unique())); print("signal spacing days (median)", pd.to_datetime(gap).diff().dt.days.median())
    S = []
    for (st, pe), g in [*PR.groupby(["strategy", "period"]), *[((s, "全期"), g) for s, g in PR.groupby("strategy")]]:
        a = g[g.n_picks > 0]; row = dict(strategy=st, period=pe, n_periods=len(g), n_periods_with_picks=len(a), avg_picks=g.n_picks.mean(), mean_ret5=a.ret5.mean(), mean_pool5=g.pool_ret5.mean(), mean_excess5=(a.ret5 - a.pool_ret5).mean(),
                 hit_beat_pool=a.beat_pool.mean(), pick_beat_median_share=a.beat_med_share.mean(), mean_ret20=a.ret20.mean(), mean_pool20=a.pool_ret20.mean(),
                 cum_gross=np.prod(1 + g.where(g.n_picks > 0).ret5.fillna(0)) - 1, cum_pool=np.prod(1 + g.pool_ret5) - 1)
        for c in COST[1:]: row[f"cum_net_{int(c*1e4)}bp"] = np.prod(1 + g[f"ret5_net_{int(c*1e4)}bp"]) - 1
        S.append(row)
    S = pd.DataFrame(S); S.to_csv(R / "hist_top5_summary.csv", index=False, encoding="utf-8-sig"); print(S.round(4).to_string())
    # bins by year (5d / 20d): mean, median, beat-same-day-median share, underperform share
    Y = []
    for hz, c, x in [("5d", "realized_rel_ret_5d", 0.02), ("20d", "rel_20d", 0.05)]:
        e = d.dropna(subset=[c]).copy(); e["bm"] = e[c] > e.groupby("signal_date")[c].transform("median"); e["year"] = e.signal_date.str[:4]
        for (y, b), g in e.groupby(["year", "bin"]): Y.append(dict(horizon=hz, year=y, bin=b, n=len(g), mean_rel=g[c].mean(), median_rel=g[c].median(), beat_median=g.bm.mean(), under_share=(g[c] < -x).mean()))
    pd.DataFrame(Y).to_csv(R / "hist_bins_by_year.csv", index=False, encoding="utf-8-sig")
    print(pd.read_csv(R / "hist_v2_constraint_check.csv", encoding="utf-8-sig").query("horizon=='5d'").round(4).to_string())
if __name__ == "__main__":
    m = check_rules(); latest(m); history()