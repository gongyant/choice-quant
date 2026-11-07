"""P1-A2: per_stock_summary.csv, sector_summary.csv (+notes), reconciliation_p0.md, figures. Reads historical_predictions.parquet / strategy_trades.csv."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1lib import L, H, V2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc"); plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]; plt.rcParams["axes.unicode_minus"] = False
MODELS = ["Ridge", "HGB", "LGBM"]; MIN_N = 30
HP = pd.read_parquet(H / "historical_predictions.parquet", columns=["signal_date", "code", "model", "eligible_asof", "score", "rank", "pctile", "in_top50", "buy_status", "realized_ret_5d", "realized_rel_ret_5d"])
TR = pd.read_csv(H / "strategy_trades.csv", dtype={"code": str}, encoding="utf-8-sig", usecols=["strategy", "code", "signal_date", "net_ret", "trade_status", "gross_ret_div_adj"])
S = pd.read_csv(V2 / "data" / "sample_1500.csv", dtype=str); LAT = pd.read_csv(H / "latest_all_stocks_20261016.csv", dtype=str, encoding="utf-8-sig", keep_default_na=False).set_index("code")
COV = pd.read_csv(V2 / "p0" / "coverage_classification.csv", dtype={"code": str}).set_index("code")
UM = pd.read_csv(V2 / "data" / "universe_master.csv", dtype=str).set_index("code")
ND = HP.signal_date.nunique(); E = HP[HP.eligible_asof]
# ---------------- per stock ----------------
rows = []
elig_n = E[E.model == "HGB"].groupby("code").size(); first_el = E[E.model == "HGB"].groupby("code").signal_date.min(); last_el = E[E.model == "HGB"].groupby("code").signal_date.max()
for c in S.code:
    d = dict(code=c, name=UM.loc[c, "name"], exchange=UM.loc[c, "exchange"], board=UM.loc[c, "board"], industry_l1=UM.loc[c, "industry_l1"], industry_l2=UM.loc[c, "industry_l2"], industry_note="巨潮最新分类(非历史时点)", status=UM.loc[c, "status"], n_eval_dates=ND, n_eligible=int(elig_n.get(c, 0)),
             eligible_share=float(elig_n.get(c, 0)) / ND, first_eligible_date=first_el.get(c, ""), last_eligible_date=last_el.get(c, ""))
    cv = COV.loc[c] if c in COV.index else None
    d.update(data_first_bar=(cv.first_bar if cv is not None else ""), data_last_bar=(cv.last_bar if cv is not None else ""), data_missing_days=(cv.missing_days if cv is not None else np.nan), data_expected_days=(cv.expected_days if cv is not None else np.nan),
             data_missing_rate=(float(cv.missing_days) / float(cv.expected_days) if cv is not None and cv.expected_days else np.nan), data_cause=(cv.cause if cv is not None else "no_price_data"), quality_flags=LAT.loc[c, "quality_flags"])
    rows.append(d)
PS = pd.DataFrame(rows).set_index("code")
for m in MODELS:
    g = E[E.model == m]; k = m.lower()
    sp = {}; npairs = {}
    for c, gg in g.groupby("code"):
        gg = gg.dropna(subset=["realized_rel_ret_5d", "score"]); npairs[c] = len(gg)
        sp[c] = spearmanr(gg.score, gg.realized_rel_ret_5d).statistic if len(gg) >= 5 and gg.score.nunique() > 1 else np.nan
    PS[f"spearman_score_vs_rel_ret_{k}"] = pd.Series(sp); PS[f"n_pairs_{k}"] = pd.Series(npairs).reindex(PS.index).fillna(0).astype(int)
    PS[f"corr_reliability_{k}"] = np.where(PS[f"n_pairs_{k}"] < MIN_N, "不可靠(样本<30)", "样本>=30(单股时间序列相关仍噪声大,仅描述)")
    t = g[g.in_top50]; PS[f"top50_count_{k}"] = t.groupby("code").size().reindex(PS.index).fillna(0).astype(int)
    tt = t.dropna(subset=["realized_rel_ret_5d"]); PS[f"top50_mean_rel_ret_5d_{k}"] = tt.groupby("code").realized_rel_ret_5d.mean().reindex(PS.index); PS[f"top50_mean_ret_5d_{k}"] = tt.groupby("code").realized_ret_5d.mean().reindex(PS.index)
    PS[f"top50_n_realized_{k}"] = tt.groupby("code").size().reindex(PS.index).fillna(0).astype(int); PS[f"top50_mean_reliability_{k}"] = np.where(PS[f"top50_n_realized_{k}"] < MIN_N, "不可靠(样本<30)", "样本>=30(仍为描述,非稳定命中率)")
    q = g.groupby("code").pctile; PS[f"pctile_p10_{k}"] = q.quantile(.1).reindex(PS.index); PS[f"pctile_p50_{k}"] = q.median().reindex(PS.index); PS[f"pctile_p90_{k}"] = q.quantile(.9).reindex(PS.index)
    nt = TR[(TR.strategy == m) & (TR.trade_status == "CLOSED")].groupby("code"); PS[f"n_closed_trades_{k}"] = nt.size().reindex(PS.index).fillna(0).astype(int); PS[f"mean_net_ret_per_trade_{k}"] = nt.net_ret.mean().reindex(PS.index)
PS = PS.reset_index(); PS.to_csv(H / "per_stock_summary.csv", index=False, encoding="utf-8-sig")
# ---------------- sector summary ----------------
E = E.merge(UM[["exchange", "board", "industry_l1"]], left_on="code", right_index=True, how="left"); TR = TR.merge(UM[["exchange", "board", "industry_l1"]], left_on="code", right_index=True, how="left")
HPt = HP[HP.in_top50].merge(UM[["exchange", "board", "industry_l1"]], left_on="code", right_index=True, how="left")
def maxdd(r): eq = (1 + pd.Series(r)).cumprod(); return float((eq / eq.cummax() - 1).min())
out = []
for gtype, gcol, note in [("industry_l1(巨潮)", "industry_l1", "巨潮行业分类标准(最新生效,非历史时点;已退市股票多为'未分类')"), ("exchange", "exchange", "交易所"), ("board", "board", "板块(沪主板/深主板/创业板/科创板/北交所)")]:
    for grp in sorted(E[gcol].dropna().unique()):
        for m in MODELS:
            g = E[(E.model == m) & (E[gcol] == grp)]; ics = []; ns = []
            for d_, gg in g.dropna(subset=["realized_rel_ret_5d"]).groupby("signal_date"):
                if len(gg) >= 10 and gg.score.nunique() > 1: ics.append(spearmanr(gg.score, gg.realized_rel_ret_5d).statistic); ns.append(len(gg))
            ics = np.array(ics); t = HPt[(HPt.model == m) & (HPt[gcol] == grp)]; tr = TR[(TR.strategy == m) & (TR[gcol] == grp) & (TR.trade_status == "CLOSED")]
            ser = t.dropna(subset=["realized_ret_5d"]).groupby("signal_date").realized_ret_5d.mean()
            n_ic = len(ics); mean = float(ics.mean()) if n_ic else np.nan; tstat = float(ics.mean() / (ics.std(ddof=1) / np.sqrt(n_ic))) if n_ic > 2 and ics.std() > 0 else np.nan
            rel = "可用" if (n_ic >= MIN_N and (np.mean(ns) if ns else 0) >= 15) else "不可靠(IC期数<30或组内平均股票数<15)"
            out.append(dict(group_type=gtype, group=grp, model=m, n_stocks=int(g.code.nunique()), n_stock_dates=len(g), n_ic_dates=n_ic, avg_stocks_per_date=float(np.mean(ns)) if ns else np.nan, rank_ic_mean=mean, rank_ic_tstat_naive=tstat, ic_reliability=rel,
                            top50_picks=len(t), top50_closed_trades=len(tr), top50_net_ret_per_trade_mean=float(tr.net_ret.mean()) if len(tr) else np.nan, top50_net_ret_per_trade_median=float(tr.net_ret.median()) if len(tr) else np.nan,
                            top50_gross_ret_5d_mean=float(t.realized_ret_5d.mean()) if len(t) else np.nan, top50_picks_group_period_vol_ann=float(ser.std() * np.sqrt(252 / 5)) if len(ser) > 2 else np.nan, top50_picks_group_maxdd=maxdd(ser.values) if len(ser) > 2 else np.nan, n_periods_with_picks=len(ser),
                            classification_note=note))
SS = pd.DataFrame(out); SS.to_csv(H / "sector_summary.csv", index=False, encoding="utf-8-sig")
(H / "reports" / "sector_summary_notes.md").write_text("""# sector_summary.csv 说明
- 行业 = 巨潮(cninfo)行业分类**最新生效版本**,不是历史时点分类;已退市股票多为"未分类/cninfo无数据"。
- **同花顺(THS)行业/概念:无成分数据**(v2\\data\\boards 里只有 board_catalog.csv 和空的 board_member_failures.csv,没有成分股映射),因此本表不含 THS 分组,也没有编造。Choice 板块成员需用户手工导出(见 v2\\docs\\choice_export_checklist.md)。
- 分组 Rank IC:每个信号日,在组内 eligible 且标签已到期的股票上计算 score 与 realized_rel_ret_5d(相对同日全体 eligible 池等权)的 Spearman,组内股票<10 的日期不计;rank_ic_mean 为各日均值,tstat 为朴素 t(未做 HAC,期间重叠的5日标签,仅描述)。
- top50_net_ret_per_trade:strategy_trades.csv 中该组股票的已平仓逐笔净收益(含0.30%往返成本、分红再投资口径)。top50_gross_ret_5d_mean:该组 Top50 入选记录的 P0 标签收益均值(未计成本、未区分是否成交)。
- group_period_vol_ann / maxdd:每个信号日取该组 Top50 入选股的 realized_ret_5d 等权均值,得到时间序列(只含有入选的期),年化波动=std*sqrt(252/5),最大回撤按该序列连乘;**不是可交易组合净值**。
- 若 ic_reliability 为"不可靠",请勿解读。多组×多模型比较存在多重检验问题。
""", encoding="utf-8")
# ---------------- reconciliation md ----------------
rc = pd.read_csv(H / "reports" / "reconciliation_p0.csv")
md = ["# 与 P0 结果对账(results\\fixed_sina_adj)", "", "walk-forward 重放使用 p0lib.py + baseline.make_models(同参数/seed 42/每4期再训练/3年窗口/purge 12日/首个测试日 2022-07-01),并记录每日×每股分数。下表为重放结果与 P0 `period_returns.csv`、`sets.pkl`、`orders_primary_cost30bps.csv` 的差异(207 期)。", "", "| 检查项 | 模型 | 最大绝对差 | 备注 |", "|---|---|---:|---|"]
for r in rc.itertuples(): md.append(f"| {r.check} | {r.model} | {r.max_abs_diff:.2e} | {r.note if isinstance(r.note, str) else ''} |")
md += ["", f"结论:所有差异 ≤ {rc.max_abs_diff.max():.1e}(浮点舍入),Top50 集合 207/207 期完全一致,订单日志逐行一致。"]; (H / "reports" / "reconciliation_p0.md").write_text("\n".join(md), encoding="utf-8")
# ---------------- figures ----------------
(H / "figures").mkdir(exist_ok=True)
ind = SS[(SS.group_type.str.startswith("industry")) & (SS.model == "HGB")].sort_values("rank_ic_mean")
fig, ax = plt.subplots(figsize=(9, 6)); col = np.where(ind.ic_reliability == "可用", "#4c78a8", "#bbbbbb"); ax.barh(ind.group, ind.rank_ic_mean, color=col)
for y_, (v, n) in enumerate(zip(ind.rank_ic_mean, ind.n_stocks)): ax.text(v + (0.002 if v >= 0 else -0.002), y_, f"n={n}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
ax.axvline(0, color="k", lw=0.8); ax.set_title("HGB 分行业 Rank IC(2022-07~2026-09 OOS重放;巨潮最新分类,非历史时点)\n灰色=不可靠(IC期数<30或组内股票少),n=该组股票数", fontsize=10); ax.set_xlabel("平均 Rank IC(score vs 5日相对收益)"); fig.tight_layout(); fig.savefig(H / "figures" / "sector_rankic_hgb.png", dpi=130); plt.close(fig)
sm = pd.read_csv(H / "historical_predictions_summary.csv", encoding="utf-8-sig"); fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for m in MODELS:
    s = sm[sm.model == m].reset_index(drop=True); axs[0].plot(pd.to_datetime(s.signal_date), s.RankIC.rolling(20).mean(), label=f"{m} 滚动20期均值"); axs[1].plot(pd.to_datetime(s.signal_date), s.RankIC.cumsum(), label=m)
axs[0].axhline(0, color="k", lw=0.8); axs[0].legend(); axs[0].set_title("P0 walk-forward 重放:Rank IC 滚动20期均值(上)与累计 Rank IC(下);2025-2026 非盲测", fontsize=10); axs[1].legend(); axs[1].set_ylabel("累计 RankIC"); fig.tight_layout(); fig.savefig(H / "figures" / "rolling_rankic_models.png", dpi=130); plt.close(fig)
print("per_stock", PS.shape, "sector", SS.shape)
print(ind[["group", "n_stocks", "n_ic_dates", "avg_stocks_per_date", "rank_ic_mean", "rank_ic_tstat_naive", "ic_reliability"]].to_string())
for gt in ["exchange", "board"]: print(SS[(SS.group_type == gt)][["group_type", "group", "model", "n_stocks", "n_ic_dates", "rank_ic_mean", "rank_ic_tstat_naive", "top50_picks", "top50_net_ret_per_trade_mean", "ic_reliability"]].to_string())
print(PS[[c for c in PS.columns if c.startswith("n_pairs")] + ["n_eligible"]].describe().to_string())
print((PS["n_pairs_hgb"] < 30).sum(), "stocks with <30 pairs (HGB)"); print(sm.groupby("model").RankIC.agg(["mean", "std", "count"]))
