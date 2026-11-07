# -*- coding: utf-8 -*-
"""P3-3/5: per-stock latest research table (1498 rows) + Top5 focus / 5 avoid, strictly following selection_rules.json (written before this step).
Inputs: predictions_registry/pred_<date>.csv (frozen-model scores), data/world_<date>.npz, results/calibration_table.csv, P1 latest table (names/board/industry/status).
Output: latest_research_ranking_<date>.csv, results/top5_<date>.csv, results/avoid5_<date>.csv. Research only, not investment advice."""
import sys, json, os
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; V2 = H.parent; OUT = H / "results"
FIVE = ["HGB", "LGBM", "MLP", "GRU", "TF"]
def main(date):
    RULES = json.load(open(H / "selection_rules.json", encoding="utf-8"))
    P = pd.read_csv(H / ("data/dryrun_registry" if os.environ.get("P3_DRYRUN") else "predictions_registry") / f"pred_{date}.csv", dtype={"code": str}, encoding="utf-8-sig")
    z = np.load(H / "data" / f"world_{date}.npz"); codes = list(z["codes"]); cal = pd.DatetimeIndex(z["cal"]); n = len(codes)
    lat = pd.read_csv(V2 / "p1_dl" / "latest_all_stocks_20261016_v2.csv", dtype={"code": str}, encoding="utf-8-sig").set_index("code")
    U = pd.read_csv(V2 / "data" / "sample_1500.csv", dtype=str).set_index("code")
    o = pd.DataFrame(dict(code=codes)); o["name"] = U.name.reindex(codes).values
    CN = H / "data" / "current_names.csv"      # current short name (Tencent quote, fetched today) -> the '现名含 ST' rule uses the CURRENT name
    if CN.exists():
        cn = pd.read_csv(CN, dtype={"code": str}, encoding="utf-8-sig").set_index("code").name_now; o["name_sample"] = o.name; o["name"] = [cn.get(c, n) for c, n in zip(codes, o.name)]
    for c in ["exchange", "board", "industry_l1", "status", "list_date", "delist_date"]: o[c] = lat[c].reindex(codes).values if c in lat else U[c].reindex(codes).values
    o["signal_date"] = str(cal[-1].date())
    A, C, HR, TR = z["ADJf"], z["C"], z["has_row"], z["traded"]; T = A.shape[0]
    last = np.array([np.where(HR[:, j])[0][-1] if HR[:, j].any() else -1 for j in range(n)])
    o["last_trade_date"] = [str(cal[-T:][k].date()) if k >= 0 else "" for k in last]; o["close_unadj"] = [C[k, j] if k >= 0 else np.nan for j, k in enumerate(last)]
    o["data_stale"] = last < T - 1; o["suspended_t"] = ~TR[-1]
    for h in (5, 20, 60): o[f"chg_{h}d"] = A[-1] / A[-1 - h] - 1          # adjusted, forward-filled (suspended days carry the last price)
    o["vol20"], o["amt20"], o["amihud20"], o["turn20"] = z["vol20"], z["amt20"], z["ami20"], z["turn20"]
    ma = z["ma20"]; o["ma20_slope_5d"] = ma[-1] / ma[-6] - 1; o["close_vs_ma20"] = A[-1] / ma[-1] - 1
    band = np.maximum(0.05, np.nan_to_num(o.vol20.values, nan=0.0) * np.sqrt(20)); r20 = o.chg_20d.values
    up = (r20 > band) & (o.ma20_slope_5d > 0) & (o.close_vs_ma20 > 0); dn = (r20 < -band) & (o.ma20_slope_5d < 0) & (o.close_vs_ma20 < 0)
    o["trend_band"] = band; o["trend"] = np.where(up, "上涨", np.where(dn, "下跌", "震荡")); o.loc[~np.isfinite(r20), "trend"] = "数据不足"
    pv = P.pivot(index="code", columns="model", values="pctile").reindex(codes); rv = P.pivot(index="code", columns="model", values="rank").reindex(codes); sv = P.pivot(index="code", columns="model", values="score").reindex(codes)
    o["eligible_asof"] = P[P.model == "HGB"].set_index("code").eligible_asof.reindex(codes).astype(str).eq("True").values
    for m in FIVE + ["Ridge", "REV5", "REV20", "LOWVOL", "ILLIQ", "F3"]: o[f"pctile_{m}"] = pv[m].values; o[f"rank_{m}"] = rv[m].values
    for m in FIVE + ["Ridge"]: o[f"score_{m}"] = sv[m].values
    PM = o[[f"pctile_{m}" for m in FIVE]].values
    o["composite_pctile"] = np.where(o.eligible_asof, PM.mean(1), np.nan); o["model_disagreement"] = np.where(o.eligible_asof, PM.std(1), np.nan)
    o["n_models_ge70"] = np.where(o.eligible_asof, (PM >= 0.70).sum(1), np.nan); o["n_models_le30"] = np.where(o.eligible_asof, (PM <= 0.30).sum(1), np.nan)
    o["n_top50_of5"] = np.where(o.eligible_asof, (o[[f"rank_{m}" for m in FIVE]].values <= 50).sum(1), np.nan)
    el = o.eligible_asof.values; vp = np.full(n, np.nan); vp[el] = pd.Series(o.vol20.values[el]).rank(pct=True).values; o["vol20_pctile"] = vp
    # flags
    LU, LD = z["lim_up"], z["lim_dn"]
    o["limit_up_locked_t"] = LU[-1]; o["limit_down_locked_t"] = LD[-1]; o["n_limit_down_5d"] = LD[-5:].sum(0); o["consec_limit_down"] = o.n_limit_down_5d >= 2
    o["high_vol"] = o.vol20_pctile >= 0.80; o["low_liquidity"] = ~(np.nan_to_num(o.amt20.values) >= 1e8)
    o["st_or_delist_risk"] = o.name.astype(str).str.contains("ST") | o.status.astype(str).str.contains("终止|暂停")
    o["needs_access"] = o.board.astype(str).str.contains("科创|北交") | (o.exchange.astype(str) == "BJ")
    hi = o.st_or_delist_risk | o.consec_limit_down | (o.high_vol & o.low_liquidity); mid = o.high_vol | o.low_liquidity | o.limit_down_locked_t
    o["risk_level"] = np.where(hi, "高", np.where(mid, "中", "低"))
    # calibration mapping (today's decile of the composite)
    bins = np.full(n, np.nan); bins[el] = np.ceil(pd.Series(o.composite_pctile.values[el]).rank(method="first").values / el.sum() * 10).clip(1, 10); o["calib_bin"] = bins
    CT = pd.read_csv(OUT / "calibration_table.csv", encoding="utf-8-sig")
    for hz in ("5d", "20d"):
        t = CT[CT.horizon == hz].set_index("bin")
        for a, b in [("freq_outperform", "out"), ("freq_flat", "flat"), ("freq_underperform", "under"), ("abs_q10", "abs_q10"), ("abs_q50", "abs_q50"), ("abs_q90", "abs_q90"), ("credibility", "credibility"), ("diff_vs_base", "out_minus_base")]:
            o[f"s{hz}_{b}"] = o.calib_bin.map(t[a])
    # ---- Top5 (rules fixed beforehand) ----
    f = (o.eligible_asof & (o.amt20 >= 1e8) & ~o.st_or_delist_risk & ~o.limit_up_locked_t & ~o.suspended_t & ~o.limit_down_locked_t & ~o.consec_limit_down
         & (o.n_models_ge70 >= 4) & (o.model_disagreement <= 0.15) & (o.chg_5d <= 0.15) & (o.chg_20d <= 0.30) & ~((o.trend == "下跌") & (o.chg_5d < -0.08)))
    o["passes_top_filters"] = f; o["sel_score"] = np.where(f, o.composite_pctile - 0.10 * o.vol20_pctile - 0.05 * (o.chg_20d > 0.15), np.nan)
    top = o[f].sort_values(["sel_score", "amt20"], ascending=[False, False]).head(5); o["top5"] = o.code.isin(top.code)
    # ---- avoid 5 ----
    alive = pd.Series(last >= T - 20, index=o.index)      # traded within the last 20 trading days (long-dead codes are reported in data quality instead)
    cand = alive & ((o.eligible_asof & (o.composite_pctile <= 0.20) & (o.n_models_le30 >= 4)) | o.st_or_delist_risk | o.consec_limit_down)
    cand &= ~o.status.astype(str).str.contains("终止")       # already delisted stocks cannot be traded at all -> listed separately in data quality, not in the avoid list
    o["avoid_score"] = np.where(cand, (1 - o.composite_pctile.fillna(0.5)) + 0.5 * o.high_vol + 0.5 * o.low_liquidity + 1.0 * o.st_or_delist_risk + 1.0 * o.consec_limit_down + 0.5 * o.limit_down_locked_t, np.nan)
    av = o[cand].sort_values(["avoid_score", "code"], ascending=[False, True]).head(5); o["avoid5"] = o.code.isin(av.code)
    o["conclusion"] = np.where(o.top5, "继续观察", np.where(o.avoid5, "暂不买入(回避)", ""))
    o["note"] = "研究排名/历史经验频率,不是买入推荐;分数不是概率或收益"
    o = o.sort_values(["eligible_asof", "composite_pctile"], ascending=[False, False])
    o.to_csv(H / f"latest_research_ranking_{date}.csv", index=False, encoding="utf-8-sig"); top.to_csv(OUT / f"top5_{date}.csv", index=False, encoding="utf-8-sig"); av.to_csv(OUT / f"avoid5_{date}.csv", index=False, encoding="utf-8-sig")
    print("rows", len(o), "eligible", int(o.eligible_asof.sum()), "pass top filters", int(f.sum()), "avoid cands", int(cand.sum()))
    k = ["code", "name", "close_unadj", "chg_5d", "chg_20d", "trend", "composite_pctile", "model_disagreement", "n_models_ge70", "amt20", "vol20_pctile", "risk_level", "calib_bin", "s5d_out", "s5d_credibility", "s20d_out", "s20d_credibility"]
    print(o[o.top5][k + ["sel_score"]].round(4).to_string()); print(o[o.avoid5][k + ["avoid_score", "st_or_delist_risk", "consec_limit_down"]].round(4).to_string())
if __name__ == "__main__": main(sys.argv[1] if len(sys.argv) > 1 else "20261021")