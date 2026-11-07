"""P1-A1: frozen HGB/LGBM/Ridge scores for the latest signal date (2026-10-16) for EVERY stock of the original 1500-stock sample."""
import sys, json, time, joblib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1lib as P1
from p1lib import L, H, V2
t0 = time.time(); ASOF = pd.Timestamp("2026-10-16")
X = L.build_data(); L.limit_flags(X); W = P1.world_from_X(X); cal = W.cal; T = len(cal); i = T - 1
assert cal[i] == ASOF, cal[i]
print("data built", round(time.time() - t0), "s; stocks", len(X.codes), "pool at asof", int(W.elig[i].sum()), flush=True)
t1 = time.time(); fitted, meta = P1.train_frozen(W, i, seed=42); meta["train_seconds"] = round(time.time() - t1, 1)
pool, sc = P1.score_asof(W, i, fitted)
lib_sha = L.sha256(V2/"p0"/"p0lib.py"); meta["p0lib_sha256"] = lib_sha; meta["baseline_py_sha256"] = L.sha256(V2/"baseline.py")
meta["p1lib_sha256"] = P1.file_sha(H/"p1lib.py"); meta["score_latest_sha256"] = P1.file_sha(Path(__file__)); meta["sample_sha256"] = L.sha256(V2/"data"/"sample_1500.csv")
meta["model_version"] = f"p1A1-{ASOF.date()}-HGB_LGBM_Ridge-seed42-train{meta['train_first_date']}_{meta['train_last_date']}-p0lib{lib_sha[:8]}"
meta["n_pool_asof"] = int(len(pool))
(H/"models").mkdir(exist_ok=True)
for n in P1.MODELS: joblib.dump(fitted[n], H/"models"/f"{n}_frozen_{ASOF.date()}.joblib")
json.dump(meta, open(H/"models"/f"model_meta_{ASOF.date()}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("trained", meta["model_version"], "rows", meta["n_train_rows"], "sec", meta["train_seconds"], flush=True)
# ---------- master table: all 1500 original sample stocks ----------
S = pd.read_csv(V2/"data"/"sample_1500.csv", dtype=str)
UM = pd.read_csv(V2/"data"/"universe_master.csv", dtype=str).set_index("code")
IND = pd.read_csv(V2/"data"/"industry_cninfo.csv", dtype=str).set_index("code")
EV = pd.read_csv(V2/"p0"/"quality_event_log.csv", dtype={"code": str}); COV = pd.read_csv(V2/"p0"/"coverage_classification.csv", dtype={"code": str}).set_index("code")
QPS = pd.read_csv(V2/"reports"/"quality_per_stock.csv", dtype={"code": str}).set_index("code")
cause_counts = EV.groupby(["code", "cause"]).size().unstack(fill_value=0)
col = {c: j for j, c in enumerate(X.codes)}; has_tx = {p.stem for p in L.TXD.glob("*.parquet")}
amt20 = W.f["amt20"].values; turn20 = pd.DataFrame(X.TURN).rolling(20, min_periods=15).mean().values
rank = {n: np.full(len(X.codes), np.nan) for n in P1.MODELS}; score = {n: np.full(len(X.codes), np.nan) for n in P1.MODELS}; pct = {n: np.full(len(X.codes), np.nan) for n in P1.MODELS}
for n in P1.MODELS:
    s = sc[n]; r = pd.Series(-s).rank(method="first").values; score[n][pool] = s; rank[n][pool] = r; pct[n][pool] = 1 - (r - 1) / (len(pool) - 1)
rows = []
for k, r in S.iterrows():
    c = r.code; um = UM.loc[c]; ind = IND.loc[c] if c in IND.index else None; flags = []; reason = []
    d = dict(asof_date=str(ASOF.date()), code=c, name=um["name"], exchange=um["exchange"], board=um["board"],
             industry=(f"{um['industry_l1']}/{um['industry_l2']}" if isinstance(um["industry_l1"], str) else ""), industry_l1=um["industry_l1"], industry_l2=um["industry_l2"],
             industry_source="巨潮行业分类标准(最新生效,非历史时点分类)", sample_group=r["sample_group"], list_date=um["list_date"], delist_date=um["delist_date"], status=um["status"])
    if c not in col:
        d.update(last_trade_date="", close_unadjusted=np.nan, eligible_asof=False, ineligible_reason="无行情数据(新浪/腾讯接口返回空表;样本内已终止上市股票,价格数据未取到)",
                 quality_flags="价格数据缺失;已终止上市" if r.status == "已终止上市" else "价格数据缺失")
        rows.append(d); continue
    j = col[c]; hb = np.where(X.has_row[:, j])[0]; lb = int(hb[-1]); d["last_trade_date"] = str(cal[lb].date())
    d["close_unadjusted"] = float(X.C[lb, j]); tr_last = bool(X.traded[i, j])
    for k_, nm in [("r5", "r5"), ("r20", "r20"), ("r60", "r60"), ("vol20", "vol20")]: d[nm] = float(W.f[k_].values[i, j]) if tr_last else np.nan
    d["amount20"] = float(amt20[i, j]) if np.isfinite(amt20[i, j]) else np.nan; d["turnover20"] = float(turn20[i, j]) if np.isfinite(turn20[i, j]) else np.nan
    elig = bool(W.elig[i, j]); d["eligible_asof"] = elig
    if not elig:
        dd = pd.Timestamp(um["delist_date"]) if isinstance(um["delist_date"], str) else None
        if lb < i and dd is not None and dd <= ASOF: reason.append(f"已退市(终止上市日{dd.date()},最后成交日{cal[lb].date()})")
        elif lb < i: reason.append(f"最新日无成交(停牌或数据缺失,最后成交日{cal[lb].date()})")
        elif not tr_last: reason.append("最新日无有效成交(成交量为0)")
        if tr_last:
            if not np.isfinite(W.f["r60"].values[i, j]) or not np.isfinite(W.f["vol20"].values[i, j]): reason.append("历史不足(不足60个交易日或20日有效收益不足)")
            if not (np.isfinite(amt20[i, j]) and amt20[i, j] >= 2e7): reason.append("流动性不足(20日成交额中位数<2000万元)")
            if not reason: reason.append("数据无效")
        elif not reason: reason.append("数据无效")
    d["ineligible_reason"] = ";".join(reason)
    # quality flags (as-of-t-visible or static data facts)
    if um["is_st_name"] == "True": flags.append("ST/*ST名称(涨跌停5%仅近似)")
    if um["status"] == "已终止上市": flags.append("已终止上市股票(样本全部纳入)")
    if r["exchange"] == "BJ" or "tencent" in X.meta.set_index("code").loc[c, "source"]: flags.append("价格单源(无独立第二来源)")
    if bool(X.meta.set_index("code").loc[c, "unit_unverified"]): flags.append("成交量单位无法核验")
    if "tencent" in X.meta.set_index("code").loc[c, "source"] or not np.isfinite(turn20[i, j]): flags.append("无换手率(腾讯源),turnover特征按0处理") if not np.isfinite(turn20[i, j]) else None
    if X.meta.set_index("code").loc[c, "factor_status"].startswith("empty"): flags.append("复权因子表为空(无公司行动或新浪无表)")
    if c in cause_counts.index:
        cc = cause_counts.loc[c]; a = int(cc.get("additive_qfq_artifact", 0) + cc.get("qfq_negative_or_zero_artifact", 0)) if "qfq_negative_or_zero_artifact" in cc.index else int(cc.get("additive_qfq_artifact", 0))
        un = int(sum(v for kk, v in cc.items() if "unexplained" in kk or "无法" in kk)); n_all = int(cc.sum())
        flags.append(f"P0异常事件{n_all}次(已修正/已记录,见quality_event_log)")
        if un: flags.append(f"含{un}次未解释异常")
    if c in COV.index and (float(COV.loc[c, "coverage"]) < 0.95): flags.append(f"数据覆盖率{float(COV.loc[c, 'coverage']):.0%}(<95%)")
    if X.lim_up[i, j]: flags.append("最新日涨停封板")
    if X.lim_dn[i, j]: flags.append("最新日跌停封板")
    if tr_last and (np.isfinite(W.f["r60"].values[i, j]) is False): pass
    if int(X.has_row[:, j].sum()) < 120: flags.append("上市/数据不足120个交易日")
    d["quality_flags"] = ";".join(flags)
    for n in P1.MODELS: d[{"HGB": "hgb", "LGBM": "lgbm", "Ridge": "ridge"}[n] + "_score"] = float(score[n][j]) if elig else np.nan
    for n in P1.MODELS: d["rank_" + {"HGB": "hgb", "LGBM": "lgbm", "Ridge": "ridge"}[n]] = (int(rank[n][j]) if elig else np.nan); d["pctile_" + {"HGB": "hgb", "LGBM": "lgbm", "Ridge": "ridge"}[n]] = (float(pct[n][j]) if elig else np.nan)
    rows.append(d)
df = pd.DataFrame(rows)
for c in ["dl_mlp_score", "dl_gru_score", "rank_dl_gru"]: df[c] = np.nan
df["model_version"] = meta["model_version"]; df["predicted_horizon"] = "5个交易日相对收益排序"
cols = ["asof_date", "code", "name", "exchange", "board", "industry", "last_trade_date", "close_unadjusted", "r5", "r20", "r60", "vol20", "amount20", "turnover20", "quality_flags", "eligible_asof", "ineligible_reason",
        "hgb_score", "lgbm_score", "ridge_score", "dl_mlp_score", "dl_gru_score", "rank_hgb", "rank_lgbm", "rank_ridge", "rank_dl_gru", "pctile_hgb", "pctile_lgbm", "pctile_ridge", "model_version", "predicted_horizon",
        "industry_l1", "industry_l2", "industry_source", "sample_group", "status", "list_date", "delist_date"]
df = df[cols].sort_values("code").reset_index(drop=True)
for c in ["rank_hgb", "rank_lgbm", "rank_ridge", "rank_dl_gru"]: df[c] = df[c].astype("Int64")
out = H/"latest_all_stocks_20261016.csv"; df.to_csv(out, index=False, encoding="utf-8-sig")
# also save the 11 rank-features of the latest day for audit
fe = pd.DataFrame(W.Xq[i][pool], columns=L.FEATS); fe.insert(0, "code", [X.codes[j] for j in pool]); (H/"scores").mkdir(exist_ok=True); fe.to_csv(H/"scores"/"latest_rank_features_20261016.csv", index=False)
json.dump(meta, open(H/"scores"/"score_run_info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", out, len(df), "elig", int(df.eligible_asof.sum()), round(time.time() - t0), "s")
print(df.loc[~df.eligible_asof, "ineligible_reason"].str.replace(r"\(.*", "", regex=True).value_counts().to_string())
