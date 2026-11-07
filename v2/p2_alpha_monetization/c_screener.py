# -*- coding: utf-8 -*-
"""Research screener as of 2026-10-16 -> results/research_screener_20261016.csv (+ _columns.md, _meta.json).
Uses ONLY frozen artefacts: P1 latest six-model scores (latest_all_stocks_20261016_v2.csv), P0 data up to 2026-10-16, P2-B protocol / nested A2 schedules, P2-C OOS block scores.
No retraining; nothing under p0/ or p1_dl/ or forward_registry is written. RESEARCH RANKING ONLY - not a buy recommendation; scores are not probabilities or returns;
P2 found that the signal has NOT been shown to be monetisable after costs. The A2 weights are a research simulation (empty starting book, C2 cost model, as-of inputs)."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as Q, b_data as BD, b_sim as S, b_lib as BL, c_eval as E
R = Q.RES; P1 = Q.P1D; ASOF = "2026-10-16"; OUT = R / "research_screener_20261016.csv"
MOD = {"HGB": "hgb", "LGBM": "lgbm", "Ridge": "ridge", "MLP": "dl_mlp", "GRU": "dl_gru", "TF": "dl_tf"}
LOW_LIQ = 5e7      # default low-liquidity flag threshold (20-day median turnover, RMB); the dashboard lets the user change it
class AsofCost:
    """cost model evaluated with the latest available inputs: the execution day after 2026-10-16 is not in the data, so rates use day min(d, T-1)."""
    def __init__(self, inner, T): self.inner, self.T = inner, T
    def rate(self, side, d, cs, nominal): return self.inner.rate(side, min(int(d), self.T - 1), cs, nominal)
def parse_a2(v):     # "A2_t0.5_c0.05_doff" -> ("A2", 0.5, 0.05, None)
    _, t, c, d = v.split("_"); return ("A2", float(t[1:]), float(c[1:]), None if d == "doff" else float(d[1:]))
def pct_in_pool(score, pool):
    out = np.full(len(score), np.nan); x = score[pool]; ok = np.isfinite(x); r = pd.Series(x[ok]).rank(method="average", ascending=True).values
    out[np.where(pool)[0][ok]] = (r - 0.5) / ok.sum(); return out     # 1 = best (highest score)
def main():
    t0 = time.time(); D = BD.build(); X, W = D["X"], D["W"]; T, N = len(X.cal), D["N"]; i = T - 1; assert str(X.cal[i].date()) == ASOF, X.cal[i]
    codes = list(X.codes); lat = pd.read_csv(P1 / "latest_all_stocks_20261016_v2.csv", dtype={"code": str}, encoding="utf-8-sig").set_index("code").reindex(codes)
    elig = (lat.eligible_asof.astype(str) == "True").values; assert (elig == W.elig[i]).all(), "P1 latest eligibility differs from P0 world"
    o = pd.DataFrame(dict(code=codes, name=lat.name.values, exchange=lat.exchange.values, board=lat.board.values, industry_l1=lat.industry_l1.values, industry_l2=lat.industry_l2.values, industry_source="巨潮最新分类(非历史时点)",
                          status=lat.status.values, close_unadjusted=lat.close_unadjusted.values, eligible_asof=elig, ineligible_reason=lat.ineligible_reason.values))
    # ---- six frozen model scores: rank (1 = best) and percentile within the eligible pool ----
    P = {}
    for m, k in MOD.items():
        sc = pd.to_numeric(lat[f"{k}_score"], errors="coerce").values; o[f"score_{m}"] = sc          # P1 published ranks/percentiles (1 = best; ties as published by P1)
        o[f"rank_{m}"] = pd.to_numeric(lat[f"rank_{k}"], errors="coerce").values; p = pd.to_numeric(lat[f"pctile_{k}"], errors="coerce").values; o[f"pctile_{m}"] = p; P[m] = p
        r2 = pd.Series(-sc).where(elig).rank(method="min").values; assert np.nanmax(np.abs(r2[elig] - o[f"rank_{m}"].values[elig])) <= 5, m      # consistent up to ties of the CSV-rounded scores
    PM = np.column_stack([P[m] for m in MOD]); top = np.column_stack([o[f"rank_{m}"].values <= 50 for m in MOD])
    o["n_models_top50_of6"] = np.where(elig, top.sum(1), np.nan); o["models_in_top50"] = ["|".join(m for m, t in zip(MOD, r) if t) if e else "" for r, e in zip(top, elig)]
    o["pctile_mean_6"] = np.where(elig, np.nanmean(PM, 1), np.nan); o["pctile_std_6"] = np.where(elig, np.nanstd(PM, 1), np.nan)
    o["n_models_top50_of5_p1"] = pd.to_numeric(lat.n_models_top50_of5, errors="coerce").values; o["pctile_std_5_p1"] = pd.to_numeric(lat.model_agreement_pctile_std, errors="coerce").values
    # ---- Transformer three seeds: the LATEST OOS walk-forward signal date (the 2026-10-16 scores exist only for seed 42; P2-C did not save models, no retraining) ----
    k_last = D["K"] - 1; i_last = int(D["sig"][k_last]); pool_l = D["cand0"][k_last]; seeds = {42: D["scores"]["TF"][k_last]}
    for sd in (43, 44):
        A, h, cov = E.load_variant("T0", sd, D["K"], N, {int(s): kk for kk, s in enumerate(D["sig"])}, 52)
        if A is not None and cov[k_last]: seeds[sd] = A[k_last, :, 0]
    tfp = np.column_stack([pct_in_pool(seeds[s].astype(float), pool_l) for s in seeds]); o["tf3_signal_date"] = str(X.cal[i_last].date()); o["tf3_seeds"] = ",".join(map(str, seeds))
    o["tf3_pctile_mean"] = np.nanmean(tfp, 1); o["tf3_pctile_std"] = np.nanstd(tfp, 1); o["tf3_rank_of_mean"] = pd.Series(-o.tf3_pctile_mean.where(pool_l)).rank(method="min").values
    o["tf_s42_pctile_latest_oos"] = tfp[:, 0]
    # ---- P2-A as-of style exposures and baselines (as of 2026-10-16) ----
    f = W.f; turn20 = Q.rolling_turn20(X.TURN); ami = Q.amihud20(f["r1"].values, X.AMT)
    sty = dict(r5=f["r5"].values[i], r20=f["r20"].values[i], r60=f["r60"].values[i], vol20=f["vol20"].values[i], turn20=turn20[i], amihud20=ami[i], amt20=f["amt20"].values[i])
    for k, v in sty.items(): o[k] = v
    for k, v in sty.items():
        if k in ("r20", "vol20", "turn20", "amihud20", "amt20"): o[f"{k}_pctile"] = pct_in_pool(np.asarray(v, float), elig)
    base = {"REV5": -sty["r5"], "LOWVOL": -sty["vol20"], "ILLIQ": sty["amihud20"]}
    for k, v in base.items(): o[f"base_{k}_pctile"] = pct_in_pool(np.asarray(v, float), elig)
    o["base3_pctile_mean"] = o[[f"base_{k}_pctile" for k in base]].mean(axis=1, skipna=False)
    # ---- tradability flags (known at the 2026-10-16 close) ----
    o["limit_up_locked_t"] = X.lim_up[i]; o["limit_down_locked_t"] = X.lim_dn[i]; o["traded_t"] = X.traded[i]
    o["st_name_current"] = o.name.astype(str).str.contains("ST", case=False); o["star_board"] = o.board.astype(str).str.contains("科创"); o["bj_exchange"] = o.board.astype(str).str.contains("北交") | (o.exchange.astype(str) == "BJ")
    o["needs_special_access"] = o.star_board | o.bj_exchange; o["low_liquidity_flag"] = ~(np.nan_to_num(sty["amt20"], nan=0) >= LOW_LIQ)
    # ---- P2-B A2 research weights (latest nested parameters, C2 explicit cost, empty starting book) ----
    PRO = BL.PROTO; scen = PRO["scenarios"]["C2_explicit"]; cost = AsofCost(BL.make_cost(D, scen), T); fixed = {k: v for k, v in PRO["A2_fixed"].items() if k in ("lam", "M", "beta0", "beta_floor", "min_obs")}
    sig_ext = np.append(D["sig"], i); cand_ext = np.vstack([D["cand0"], elig[None, :]]); meta = {}
    for m, k in MOD.items():
        ex = json.load(open(R / "b_runs" / f"{m}_C2_explicit_extras.json", encoding="utf-8")); th = parse_a2(ex["A2_schedule"][-1])
        sc_ext = np.vstack([D["scores"][m], o[f"score_{m}"].values[None, :]])
        dec = S.A2Decider(sc_ext, cand_ext, sig_ext, D["ADJf"], W, X, cost, D["y"], tau=th[1], cap=th[2], delta=th[3], capital=scen["capital"], **fixed)
        res = dec(len(sig_ext) - 1, dict(k=len(sig_ext) - 1, s=i, d=i, units=np.zeros(N), cash=float(scen["capital"]), pend=None))
        w = np.zeros(N); w[res["tgt"]] = res["w"]; o[f"a2w_{m}"] = np.where(elig, w, np.nan)
        meta[m] = dict(params=ex["A2_schedule"][-1], **{a: (float(b) if isinstance(b, (int, float, np.floating)) else b) for a, b in res["meta"].items() if np.isscalar(b)})
    o["a2w_mean6"] = o[[f"a2w_{m}" for m in MOD]].mean(axis=1); o["research_note"] = "研究排名/研究模拟权重,不是买入推荐;分数不是概率或收益"
    o = o.sort_values(["eligible_asof", "pctile_mean_6"], ascending=[False, False]); o.to_csv(OUT, index=False, encoding="utf-8-sig")
    json.dump(dict(asof=ASOF, n_rows=len(o), n_eligible=int(elig.sum()), tf3_signal_date=str(X.cal[i_last].date()), tf3_seeds=list(seeds), low_liq_threshold=LOW_LIQ, a2=meta, seconds=round(time.time() - t0, 1),
                   note="A2 weights: research simulation from an empty book with the nested-selected parameters of the last block and the C2 explicit cost model (cost inputs as of 2026-10-16). Not a recommendation."),
              open(R / "research_screener_20261016_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("rows", len(o), "eligible", int(elig.sum()), "tf3", list(seeds), X.cal[i_last].date(), "a2", {m: (v["params"], v.get("n_names"), round(v.get("sum_w", 0), 3)) for m, v in meta.items()}, f"{time.time()-t0:.0f}s")
if __name__ == "__main__": main()