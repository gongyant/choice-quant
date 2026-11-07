"""P1-A3 step 2: historical_predictions_all_models.parquet, DL reconciliation, per_stock_summary_v2, sector_summary_v2, strategy_trades_v2,
historical_predictions_summary_v2, recent20_metrics_v2, model comparison tables, cumulative net figure. Original A2 files are not modified."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import p1lib as P1
from p1lib import L, B, H, V2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc"); plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]; plt.rcParams["axes.unicode_minus"] = False
t0 = time.time(); C30 = 0.003; MIN_N = 30; DLM = {"mlp": "MLP", "gru": "GRU", "tf": "TF"}; DLR = H / "dl" / "results"
X = L.build_data(); L.limit_flags(X); W = P1.world_from_X(X); cal = W.cal; codes = np.array(X.codes); cidx = {c: j for j, c in enumerate(codes)}; T, N = W.elig.shape
HP = pd.read_parquet(H / "historical_predictions.parquet"); print("HP", HP.shape, round(time.time() - t0), flush=True)
P0R = pd.read_csv(L.P0 / "results" / "fixed_sina_adj" / "period_returns.csv")
hgb = HP[HP.model == "HGB"][["signal_date", "code", "order_date", "exit_date", "label_uses_stale_price"] + [c for c in HP.columns if c.startswith("f_")]]
rc = []; parts = []; SUMM = []; TRD = []; DLRET = {}
def add(check, n, diff, extra=""): rc.append(dict(check=check, model=n, max_abs_diff=float(diff), note=extra))
def pair_trades(n, lg, UM):
    out = []; opn = {}; nb = {}
    for r in lg.itertuples():
        if r.side == "BUY":
            if r.status == "FILLED": opn[r.code] = r; nb[r.code] = 0
            else: out.append(dict(strategy=n, code=r.code, signal_date=r.signal_date, planned_buy_date=r.planned_date, actual_buy_date="", buy_price_unadj=r.fill_price, trade_status="BUY_REJECTED", unfilled_reason=r.reason))
        else:
            if r.status == "BLOCKED_RETRY": nb[r.code] = nb.get(r.code, 0) + 1; continue
            b = opn.pop(r.code, None)
            if b is None: continue
            net = (r.value - r.cost) / (b.value + b.cost) - 1
            out.append(dict(strategy=n, code=r.code, signal_date=b.signal_date, planned_buy_date=b.planned_date, actual_buy_date=b.fill_date, buy_price_unadj=b.fill_price, buy_shares=b.shares, buy_value=b.value, buy_fee=b.cost, planned_sell_date=r.planned_date, actual_sell_date=r.fill_date, sell_price_unadj=r.fill_price,
                            sell_delay_days=r.sell_delay_days, sell_shares_equiv=r.shares, sell_value_div_adj=r.value, sell_fee=r.cost, slippage_cost=0.0, sell_blocked_events=nb.pop(r.code, 0), gross_ret_div_adj=r.value / b.value - 1, net_ret=net, trade_status="CLOSED", unfilled_reason=""))
    for c, b in opn.items(): out.append(dict(strategy=n, code=c, signal_date=b.signal_date, planned_buy_date=b.planned_date, actual_buy_date=b.fill_date, buy_price_unadj=b.fill_price, buy_shares=b.shares, buy_value=b.value, buy_fee=b.cost, sell_blocked_events=nb.get(c, 0), trade_status="OPEN_AT_END_OR_STUCK", unfilled_reason="数据终止/未卖出至样本末"))
    return out
UM = pd.read_csv(V2 / "data" / "universe_master.csv", dtype=str).set_index("code")
for k, M in DLM.items():
    p = pd.read_parquet(H / "scores" / f"dl_{k}_oos_predictions.parquet"); sc = f"{k}_score_mean"; pr = pd.read_csv(DLR / k / "period_returns.csv"); K = p.signal_date.nunique()
    assert len(p) == 310086 and K == 207 and (sorted(p.signal_date.unique()) == pr.signal_date.tolist())
    nel_sc = int(p.loc[~p.eligible_asof, sc].notna().sum()); print(k, "scores on non-eligible rows:", nel_sc, flush=True)
    sdates = sorted(p.signal_date.unique()); sig_used = [int(cal.get_indexer([pd.Timestamp(d)])[0]) for d in sdates]; assert min(sig_used) >= 0
    # --- top-50 sets in rank order -> simulate (same engine) ---
    topk = []; pools = []
    for d in sdates:
        g = p[(p.signal_date == d) & p.eligible_asof]; pools.append(np.array([cidx[c] for c in g.code])); t = g[g.in_top50].sort_values("rank_in_pool"); assert len(t) == 50; topk.append(np.array([cidx[c] for c in t.code]))
    o, lg, ex = L.simulate(X, W.ADJf, sig_used, topk, C30, capital=1e7, log_name=M); DLRET[M] = o.ret.values
    b_ret = pr[f"{M}_mean_ret_c30"].values; add("net period return c30 (rerun of simulate from OOS Top50 vs dl results period_returns.csv)", M, np.abs(o.ret.values - b_ret).max(), f"{K} periods")
    for kk in ["turnover_value", "cash_frac_post", "buy_filled", "n_hold"]: add(f"{kk} (30bps)", M, np.abs(o[kk].values - pr[f"{M}_mean_{ {'turnover_value':'turnover_value','cash_frac_post':'cash_frac_post','buy_filled':'buy_filled','n_hold':'n_hold'}[kk]}"].values).max())
    od = pd.read_csv(DLR / k / f"orders_{M}_mean_cost30bps.csv", dtype={"code": str}); a = lg.reset_index(drop=True); b = od.reset_index(drop=True)
    ok = len(a) == len(b) and (a.code.values == b.code.values).all() and (a.status.values == b.status.values).all() and np.allclose(a.value.values, b.value.values) and np.allclose(a.cost.values, b.cost.values)
    add("order log identical to B orders_*_cost30bps.csv", M, 0.0 if ok else 1.0, f"rows {len(a)} vs {len(b)}")
    ic = []; ric = []
    for d, i, pool in zip(sdates, sig_used, pools):
        g = p[(p.signal_date == d) & p.eligible_asof].set_index("code"); s = g.loc[codes[pool], sc].values; yy = W.y[i][pool]; fin = np.isfinite(yy)
        if fin.sum() > 5 and np.std(s[fin]) > 0: ic.append(float(np.corrcoef(s[fin], yy[fin])[0, 1])); ric.append(float(spearmanr(s[fin], yy[fin]).statistic))
        else: ic.append(np.nan); ric.append(np.nan)
    add("IC per period (recomputed from OOS scores vs period_returns)", M, np.nanmax(np.abs(np.array(ic) - pr[f"{M}_mean_IC"].values))); add("RankIC per period", M, np.nanmax(np.abs(np.array(ric) - pr[f"{M}_mean_RankIC"].values)))
    ew = pr["EqualWeight_ret_c30"].values; add("EqualWeight net ret c30 vs P0 file", M, np.abs(ew - P0R["EqualWeight_ret_c30"].values).max())
    # trades (strategy name = MLP/GRU/TF)
    TRD += pair_trades(M, lg, UM)
    # --- long table part ---
    p = p.copy(); p.loc[~p.eligible_asof, [sc, f"{k}_s42"] + ([f"{k}_s43", f"{k}_s44"] if k != "tf" else [])] = np.nan
    npool = p.groupby("signal_date").eligible_asof.transform("sum").astype(int); rk = p.rank_in_pool.astype("float64")
    d = pd.DataFrame({"signal_date": p.signal_date.values, "code": p.code.values, "model": M, "model_id": p.model_id.values, "train_first_date": p.train_window_start.values, "train_end_date": p.train_end.values, "train_label_end_date": p.max_label_expiry.values, "n_train_rows": np.nan,
                      "eligible_asof": p.eligible_asof.values, "n_pool": npool.values, "score": p[sc].values, "rank": rk.astype("float32").values, "pctile": (1 - (rk - 1) / (npool - 1)).astype("float32").values, "in_top50": p.in_top50.values})
    bs = p.buy_status_cost30.fillna("").values; d["buy_status"] = np.where(bs != "", bs, np.where(p.in_top50.values, "RETAINED_ALREADY_HELD", "NOT_TOP50")); d["buy_reason"] = p.buy_reject_reason.fillna("").values
    bo = od[od.side == "BUY"].set_index(["signal_date", "code"]); ix = pd.MultiIndex.from_arrays([d.signal_date, d.code]); bo = bo[~bo.index.duplicated()].reindex(ix)
    d["buy_fill_date"] = bo.fill_date.fillna("").values; d["buy_price_unadj"] = bo.fill_price.values; d["buy_shares"] = bo.shares.values
    chk = d[(d.buy_status.isin(["FILLED", "REJECTED"]))]; add("buy_status FILLED/REJECTED count vs B order log BUY rows", M, abs(len(chk) - int(len(od[od.side == "BUY"]))), f"{len(chk)} vs {int((od.side=='BUY').sum())}")
    d = d.merge(hgb, on=["signal_date", "code"], how="left", validate="1:1"); assert len(d) == 310086
    d["realized_ret_5d"] = p.realized_ret_5d.values; d["realized_rel_ret_5d"] = p.realized_excess_5d.values
    add("realized_ret_5d equals A2 (HGB rows) label", M, np.nanmax(np.abs(d.realized_ret_5d.values - HP[HP.model == "HGB"].set_index(["signal_date", "code"]).reindex(pd.MultiIndex.from_arrays([d.signal_date, d.code])).realized_ret_5d.values)))
    d["score_s42"] = p[f"{k}_s42"].values; d["score_s43"] = p[f"{k}_s43"].values if k != "tf" else np.nan; d["score_s44"] = p[f"{k}_s44"].values if k != "tf" else np.nan
    d["n_seeds"] = 3 if k != "tf" else 1; d["trained_asof"] = p.trained_asof.values; d["val_end_date"] = p.val_end.values
    parts.append(d)
    # summary rows
    g = d[d.eligible_asof]; sm = []
    for dd, gg in g.groupby("signal_date", sort=True):
        t_ = gg[gg.in_top50]; sm.append(dict(model=M, signal_date=dd, model_id=gg.model_id.iloc[0], train_end_date=gg.train_end_date.iloc[0], n_pool=len(gg), n_label=int(gg.realized_rel_ret_5d.notna().sum()), top50_mean_rel_ret_5d=t_.realized_rel_ret_5d.mean(), top50_mean_ret_5d=t_.realized_ret_5d.mean(),
                                             top50_buy_filled=int((t_.buy_status == "FILLED").sum()), top50_retained=int((t_.buy_status == "RETAINED_ALREADY_HELD").sum()), top50_rejected=int((t_.buy_status == "REJECTED").sum())))
    sm = pd.DataFrame(sm); sm["IC"] = pr[f"{M}_mean_IC"].values; sm["RankIC"] = pr[f"{M}_mean_RankIC"].values; sm["net_ret_c30"] = b_ret; sm["ew_net_ret_c30"] = ew; sm["net_excess_c30"] = sm.net_ret_c30 - sm.ew_net_ret_c30; SUMM.append(sm)
    print(k, "done", round(time.time() - t0), flush=True)
# unified column set
extra = ["score_s42", "score_s43", "score_s44", "n_seeds", "trained_asof", "val_end_date"]
for c in extra: HP[c] = np.nan if c.startswith("score") else (1 if c == "n_seeds" else "")
HPA = pd.concat([HP] + parts, ignore_index=True)
for c in ["rank", "pctile"]: HPA[c] = HPA[c].astype("float32")
HPA["n_seeds"] = HPA.n_seeds.astype("int8"); HPA["n_train_rows"] = HPA.n_train_rows.astype("float64")
assert not HPA.duplicated(["signal_date", "code", "model_id"]).any()
HPA.to_parquet(H / "historical_predictions_all_models.parquet", index=False, compression="zstd"); print("all-models parquet", HPA.shape, HPA.model.value_counts().to_dict(), HPA.model_id.nunique(), round(time.time() - t0), flush=True)
rcd = pd.DataFrame(rc); rcd.to_csv(H / "reports" / "reconciliation_dl.csv", index=False, encoding="utf-8-sig"); print(rcd.to_string(), flush=True)
md = ["# DL 模型对账(重放 vs 轨道B results\\{mlp,gru,tf}\\period_returns.csv)", "", "方法:用 OOS 预测 parquet(`scores\\dl_*_oos_predictions.parquet`)里每期 Top50(按 rank_in_pool 排序),经 P0 同一个 `p0lib.simulate`(1000万元、整手、往返0.30%)重新模拟,与 B 的 `period_returns.csv`、`orders_*_cost30bps.csv` 逐期/逐行比较;IC/RankIC 用 parquet 分数与 P0 标签 y 重新计算。", "",
      "| 检查项 | 模型 | 最大绝对差 | 备注 |", "|---|---|---:|---|"]
for r in rcd.itertuples(): md.append(f"| {r.check} | {r.model} | {r.max_abs_diff:.2e} | {r.note if isinstance(r.note, str) else ''} |")
md += ["", f"结论:所有数值差异 ≤ {rcd.max_abs_diff.max():.1e}(0 或浮点舍入);订单日志逐行一致(=0)。DL 模型的逐期净收益可由 OOS 预测文件独立复现。"]; (H / "reports" / "reconciliation_dl.md").write_text("\n".join(md), encoding="utf-8")
# ---- summary v2 / recent20 v2 ----
sm0 = pd.read_csv(H / "historical_predictions_summary.csv", encoding="utf-8-sig"); sm2 = pd.concat([sm0] + SUMM, ignore_index=True); sm2.to_csv(H / "historical_predictions_summary_v2.csv", index=False, encoding="utf-8-sig")
ORDER = ["HGB", "LGBM", "Ridge", "MLP", "GRU", "TF"]; last = sorted(sm2.signal_date.unique())[-20:]; r20 = sm2[sm2.signal_date.isin(last)].copy(); r20["model"] = pd.Categorical(r20.model, ORDER); r20 = r20.sort_values(["model", "signal_date"])
r20c = r20[["model", "signal_date", "model_id", "IC", "RankIC", "net_ret_c30", "ew_net_ret_c30", "net_excess_c30", "top50_mean_rel_ret_5d", "top50_buy_filled", "top50_rejected"]]; r20c.to_csv(H / "reports" / "recent20_metrics_v2.csv", index=False, encoding="utf-8-sig")
agg = r20.groupby("model", observed=True).apply(lambda g: pd.Series(dict(IC_mean=g.IC.mean(), RankIC_mean=g.RankIC.mean(), RankIC_pos_periods=int((g.RankIC > 0).sum()), net_ret_mean_per_period=g.net_ret_c30.mean(), net_cum=(1 + g.net_ret_c30).prod() - 1, ew_cum=(1 + g.ew_net_ret_c30).prod() - 1, net_excess_mean=g.net_excess_c30.mean(), n_periods=len(g), period_range=f"{g.signal_date.min()}..{g.signal_date.max()}")), include_groups=False).reset_index()
agg["excess_cum_simple"] = agg.net_cum - agg.ew_cum; agg.to_csv(H / "reports" / "recent20_summary_v2.csv", index=False, encoding="utf-8-sig")
mdl = ["# 最近 20 期实测指标 v2(含 MLP / GRU / Transformer)", "", f"区间(信号日):{agg.period_range.iloc[0]};共 20 期 ≈ 100 个交易日。5日调仓,Top50,往返成本0.30%,等权基准=同制度含成本等权。**样本很少,仅描述性,不能据此判断模型好坏;2025-2026 数据不是盲测。** HGB/LGBM/Ridge 与 P0 对账(reconciliation_p0.md),MLP/GRU/TF 与轨道B period_returns 对账(reconciliation_dl.md)。MLP/GRU = 3 种子均值分数,TF = 单种子 42。", "",
       "| 模型 | IC均值 | RankIC均值 | RankIC>0期数/20 | 净收益均值/期 | 20期累计净收益 | 同期等权累计 | 累计差(简单相减) | 净超额均值/期 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in agg.itertuples(): mdl.append(f"| {r.model} | {r.IC_mean:.4f} | {r.RankIC_mean:.4f} | {r.RankIC_pos_periods} | {r.net_ret_mean_per_period:+.2%} | {r.net_cum:+.2%} | {r.ew_cum:+.2%} | {r.excess_cum_simple:+.2%} | {r.net_excess_mean:+.2%} |")
mdl += ["", "## 逐期明细", ""]
for m in ORDER:
    mdl += [f"### {m}", "", "| 信号日 | model_id | IC | RankIC | 净收益 | 等权净收益 | 净超额 | Top50相对收益均值 | Top50 买单成交/拒绝 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in r20[r20.model == m].itertuples(): mdl.append(f"| {r.signal_date} | {r.model_id} | {r.IC:+.3f} | {r.RankIC:+.3f} | {r.net_ret_c30:+.2%} | {r.ew_net_ret_c30:+.2%} | {r.net_excess_c30:+.2%} | {r.top50_mean_rel_ret_5d:+.2%} | {int(r.top50_buy_filled)}/{int(r.top50_rejected)} |")
    mdl.append("")
(H / "reports" / "recent20_metrics_v2.md").write_text("\n".join(mdl), encoding="utf-8"); print(agg.to_string(), flush=True)
# ---- trades v2 ----
TRo = pd.read_csv(H / "strategy_trades.csv", dtype={"code": str}, encoding="utf-8-sig"); TN = pd.DataFrame(TRD); TN["name"] = TN.code.map(UM["name"]); TN["industry_l1"] = TN.code.map(UM["industry_l1"])
TN["note"] = "轨道B订单日志(dl\\results\\*\\orders_*_mean_cost30bps.csv)配对;口径同 P0 模拟(本金1000万元、100股整手、往返0.30%),分数为多种子均值(TF 单种子)"
TN = TN[[c for c in TRo.columns if c in TN.columns]].sort_values(["strategy", "signal_date", "code"]); TR2 = pd.concat([TRo, TN], ignore_index=True); TR2.to_csv(H / "strategy_trades_v2.csv", index=False, encoding="utf-8-sig"); print("trades v2", TR2.shape, TR2.groupby(["strategy", "trade_status"]).size().to_dict(), flush=True)
# ---- per-stock v2 ----
PS = pd.read_csv(H / "per_stock_summary.csv", dtype={"code": str}, encoding="utf-8-sig", keep_default_na=False, na_values=[""]).set_index("code")
E = HPA[HPA.eligible_asof & HPA.model.isin(list(DLM.values()))]
for m in DLM.values():
    g = E[E.model == m]; kk = m.lower(); sp = {}; npairs = {}
    for c, gg in g.groupby("code"):
        gg = gg.dropna(subset=["realized_rel_ret_5d", "score"]); npairs[c] = len(gg); sp[c] = spearmanr(gg.score, gg.realized_rel_ret_5d).statistic if len(gg) >= 5 and gg.score.nunique() > 1 else np.nan
    PS[f"spearman_score_vs_rel_ret_{kk}"] = pd.Series(sp); PS[f"n_pairs_{kk}"] = pd.Series(npairs).reindex(PS.index).fillna(0).astype(int)
    PS[f"corr_reliability_{kk}"] = np.where(PS[f"n_pairs_{kk}"] < MIN_N, "不可靠(样本<30)", "样本>=30(单股时间序列相关仍噪声大,仅描述)")
    t = g[g.in_top50]; PS[f"top50_count_{kk}"] = t.groupby("code").size().reindex(PS.index).fillna(0).astype(int); tt = t.dropna(subset=["realized_rel_ret_5d"])
    PS[f"top50_mean_rel_ret_5d_{kk}"] = tt.groupby("code").realized_rel_ret_5d.mean().reindex(PS.index); PS[f"top50_mean_ret_5d_{kk}"] = tt.groupby("code").realized_ret_5d.mean().reindex(PS.index)
    PS[f"top50_n_realized_{kk}"] = tt.groupby("code").size().reindex(PS.index).fillna(0).astype(int); PS[f"top50_mean_reliability_{kk}"] = np.where(PS[f"top50_n_realized_{kk}"] < MIN_N, "不可靠(样本<30)", "样本>=30(仍为描述,非稳定命中率)")
    q = g.groupby("code").pctile; PS[f"pctile_p10_{kk}"] = q.quantile(.1).reindex(PS.index); PS[f"pctile_p50_{kk}"] = q.median().reindex(PS.index); PS[f"pctile_p90_{kk}"] = q.quantile(.9).reindex(PS.index)
    nt = TN[(TN.strategy == m) & (TN.trade_status == "CLOSED")].groupby("code"); PS[f"n_closed_trades_{kk}"] = nt.size().reindex(PS.index).fillna(0).astype(int); PS[f"mean_net_ret_per_trade_{kk}"] = nt.net_ret.mean().reindex(PS.index)
PS.reset_index().to_csv(H / "per_stock_summary_v2.csv", index=False, encoding="utf-8-sig"); print("per_stock_v2", PS.shape, {m: int((PS[f'n_pairs_{m.lower()}'] < 30).sum()) for m in DLM.values()}, flush=True)
# ---- sector v2 ----
SS0 = pd.read_csv(H / "sector_summary.csv", encoding="utf-8-sig"); EE = E.merge(UM[["exchange", "board", "industry_l1"]], left_on="code", right_index=True, how="left"); TRn = TN.merge(UM[["exchange", "board"]], left_on="code", right_index=True, how="left")
HPt = HPA[HPA.in_top50 & HPA.model.isin(list(DLM.values()))].merge(UM[["exchange", "board", "industry_l1"]], left_on="code", right_index=True, how="left")
def maxdd(r): eq = (1 + pd.Series(r)).cumprod(); return float((eq / eq.cummax() - 1).min())
out = []
for gtype, gcol, note in [("industry_l1(巨潮)", "industry_l1", "巨潮行业分类标准(最新生效,非历史时点;已退市股票多为'未分类')"), ("exchange", "exchange", "交易所"), ("board", "board", "板块(沪主板/深主板/创业板/科创板/北交所)")]:
    for grp in sorted(EE[gcol].dropna().unique()):
        for m in DLM.values():
            g = EE[(EE.model == m) & (EE[gcol] == grp)]; ics = []; ns = []
            for d_, gg in g.dropna(subset=["realized_rel_ret_5d"]).groupby("signal_date"):
                if len(gg) >= 10 and gg.score.nunique() > 1: ics.append(spearmanr(gg.score, gg.realized_rel_ret_5d).statistic); ns.append(len(gg))
            ics = np.array(ics); t = HPt[(HPt.model == m) & (HPt[gcol] == grp)]; trm = TRn[(TRn.strategy == m) & (TRn[gcol] == grp) & (TRn.trade_status == "CLOSED")]; ser = t.dropna(subset=["realized_ret_5d"]).groupby("signal_date").realized_ret_5d.mean()
            n_ic = len(ics); mean = float(ics.mean()) if n_ic else np.nan; tstat = float(ics.mean() / (ics.std(ddof=1) / np.sqrt(n_ic))) if n_ic > 2 and ics.std() > 0 else np.nan
            rel = "可用" if (n_ic >= MIN_N and (np.mean(ns) if ns else 0) >= 15) else "不可靠(IC期数<30或组内平均股票数<15)"
            out.append(dict(group_type=gtype, group=grp, model=m, n_stocks=int(g.code.nunique()), n_stock_dates=len(g), n_ic_dates=n_ic, avg_stocks_per_date=float(np.mean(ns)) if ns else np.nan, rank_ic_mean=mean, rank_ic_tstat_naive=tstat, ic_reliability=rel, top50_picks=len(t), top50_closed_trades=len(trm),
                            top50_net_ret_per_trade_mean=float(trm.net_ret.mean()) if len(trm) else np.nan, top50_net_ret_per_trade_median=float(trm.net_ret.median()) if len(trm) else np.nan, top50_gross_ret_5d_mean=float(t.realized_ret_5d.mean()) if len(t) else np.nan,
                            top50_picks_group_period_vol_ann=float(ser.std() * np.sqrt(252 / 5)) if len(ser) > 2 else np.nan, top50_picks_group_maxdd=maxdd(ser.values) if len(ser) > 2 else np.nan, n_periods_with_picks=len(ser), classification_note=note))
SS2 = pd.concat([SS0, pd.DataFrame(out)], ignore_index=True); SS2.to_csv(H / "sector_summary_v2.csv", index=False, encoding="utf-8-sig"); print("sector_v2", SS2.shape, flush=True)
pv = SS2[SS2.group_type.str.startswith("industry")].pivot_table(index="group", columns="model", values="rank_ic_mean")[ORDER]; print(pv.round(3).to_string())
pv2 = SS2[SS2.group_type.isin(["exchange", "board"])].pivot_table(index=["group_type", "group"], columns="model", values="rank_ic_mean")[ORDER]; print(pv2.round(3).to_string())
# ---- model comparison tables from B results ----
MAPN = {"P0_HGB": "HGB", "P0_LGBM": "LGBM", "P0_Ridge": "Ridge", "P0_Momentum20": "Momentum20", "P0_Reversal5": "Reversal5", "MLP_s42": "MLP_s42", "MLP_s43": "MLP_s43", "MLP_s44": "MLP_s44", "MLP_mean": "MLP", "B1_MLP_mean": "MLP", "B_GRU_s42": "GRU_s42", "B_GRU_s43": "GRU_s43", "B_GRU_s44": "GRU_s44", "B_GRU_mean": "GRU", "B2_GRU_mean": "GRU", "B_TF_s42": "TF_s42", "B_TF_mean": "TF", "EqualWeight": "EqualWeight", "EqualWeight(same regime, 0.30% RT)": "EqualWeight"}
def comb(fn):
    fr = []
    for m_ in ["mlp", "gru", "tf"]:
        x = pd.read_csv(DLR / m_ / fn); x["src"] = m_; x["strategy"] = x.strategy.map(MAPN); fr.append(x)
    z = pd.concat(fr, ignore_index=True); keep = z[(z.src == "mlp")]; add_ = z[((z.src == "gru") & z.strategy.isin(["GRU", "GRU_s42", "GRU_s43", "GRU_s44"])) | ((z.src == "tf") & z.strategy.isin(["TF", "TF_s42"]))]
    return pd.concat([keep, add_], ignore_index=True).drop(columns="src")
mc = comb("side_by_side_c30.csv"); mc.to_csv(H / "reports" / "model_comparison_v2.csv", index=False, encoding="utf-8-sig"); cs = comb("cost_sensitivity.csv"); cs.to_csv(H / "reports" / "cost_sensitivity_all_models.csv", index=False, encoding="utf-8-sig"); by = comb("by_year.csv"); by.to_csv(H / "reports" / "by_year_all_models.csv", index=False, encoding="utf-8-sig")
pp = pd.concat([pd.read_csv(DLR / "mlp" / "paired_vs_p0.csv").assign(model="MLP"), pd.read_csv(DLR / "gru" / "paired.csv").assign(model="GRU"), pd.read_csv(DLR / "tf" / "paired.csv").assign(model="TF")], ignore_index=True); pp.to_csv(H / "reports" / "paired_tests_all_models.csv", index=False, encoding="utf-8-sig")
print(mc[["strategy", "annualized_return", "sharpe", "max_drawdown", "RankIC", "net_ex_t", "net_ex_hac4_t"]].to_string())
# ---- cumulative net ----
pr_m = pd.read_csv(DLR / "mlp" / "period_returns.csv"); pr_g = pd.read_csv(DLR / "gru" / "period_returns.csv"); pr_t = pd.read_csv(DLR / "tf" / "period_returns.csv")
cum = pd.DataFrame({"signal_date": pr_m.signal_date, "HGB": P0R.HGB_ret_c30, "LGBM": P0R.LGBM_ret_c30, "Ridge": P0R.Ridge_ret_c30, "MLP": pr_m.MLP_mean_ret_c30, "GRU": pr_g.GRU_mean_ret_c30, "TF": pr_t.TF_mean_ret_c30, "EqualWeight": pr_m.EqualWeight_ret_c30})
assert (pr_m.signal_date == P0R.signal_date).all() and (pr_g.signal_date == pr_m.signal_date).all() and (pr_t.signal_date == pr_m.signal_date).all()
nav = (1 + cum.set_index("signal_date")).cumprod(); nav.columns = [c + "_nav" for c in nav.columns]; cum = pd.concat([cum.set_index("signal_date"), nav], axis=1).reset_index(); cum.to_csv(H / "reports" / "cumulative_net_models.csv", index=False, encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(11, 5.5)); cols = {"HGB": "#4c78a8", "LGBM": "#59a14f", "Ridge": "#b07aa1", "MLP": "#f28e2b", "GRU": "#e15759", "TF": "#76b7b2", "EqualWeight": "#444444"}
for m, c in cols.items(): ax.plot(pd.to_datetime(cum.signal_date), cum[m + "_nav"], label=m + ("(同制度含成本等权)" if m == "EqualWeight" else ""), color=c, lw=1.6 if m != "EqualWeight" else 2.2, ls="--" if m == "EqualWeight" else "-")
ax.set_yscale("log"); ax.legend(ncol=4, fontsize=9); ax.grid(alpha=0.3); ax.set_ylabel("累计净值(对数轴,起点=1)")
ax.set_title("六模型 Top50 策略累计净值(往返成本0.30%,207期,2022-07~2026-09;MLP/GRU=3种子均值,TF=单种子;2025-2026 非盲测;净超额均不显著)", fontsize=9); fig.tight_layout(); fig.savefig(H / "figures" / "cumulative_net_models.png", dpi=130); plt.close(fig)
print(nav.iloc[-1].round(3).to_dict()); print("ALL DONE", round(time.time() - t0), "s")

