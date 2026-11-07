# -*- coding: utf-8 -*-
"""P4 final: merge all comparison tables, S4 empirical calibration, per-stock history for the latest Top5, equity-curve figure. No training."""
import pickle, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import p4lib as P
C = P.P4 / "cache"; R = P.P4 / "results"; FG = P.P4 / "figures"; FG.mkdir(exist_ok=True); PR = P.P4 / "preds"; REG = P.P4 / "predictions_registry"
fl = np.load(C / "flags.npz"); LAB = pickle.load(open(C / "labels.pkl", "rb"))["lab"]; X = pickle.load(open(C / "world.pkl", "rb"))
cal = pd.DatetimeIndex(fl["cal"]); codes = list(fl["codes"]); col = {c: j for j, c in enumerate(codes)}; elig = fl["elig"]
# ---- 1. unified comparison ----
a = pd.read_csv(R / "bt_summary_e0_e1.csv"); a = a[a.K == 5].drop(columns="K"); b = pd.read_csv(R / "bt_summary_e2.csv"); c = pd.read_csv(R / "bt_summary_sm.csv")
U = pd.concat([a, b[~b.strategy.isin(a.strategy)], c[~c.strategy.isin(set(a.strategy) | set(b.strategy))]], ignore_index=True)
q1 = pd.read_csv(R / "rank_quality_e0_e1.csv")[["strategy", "horizon", "rankic_mean", "ndcg5"]].rename(columns={"rankic_mean": "rankic"})
H = pd.read_parquet(P.V2 / "p1_dl" / "historical_predictions_all_models.parquet", columns=["signal_date", "code", "model", "score", "eligible_asof"]); H = H[H.eligible_asof]
sig = sorted(H.signal_date.unique()); sig_idx = [int(cal.get_loc(pd.Timestamp(s))) for s in sig]
def mat(df):
    M = np.full((len(sig), len(codes)), np.nan); k = {s: i for i, s in enumerate(sig)}; r = df.signal_date.map(k).values; cc = df.code.map(col).values; ok = ~pd.isna(cc) & ~pd.isna(r)
    M[r[ok].astype(int), cc[ok].astype(int)] = df.score.values[ok]; return M
from scipy.stats import spearmanr
extra = []
for nm, src in [("P1_Transformer原预测", "TF"), ("E2_LGBM回归(量价+基本面)", "E2reg"), ("E2_LambdaRank(对照)", "E2lrank"), ("SM_StockMixer(E2)", "SME2"), ("SM_StockMixer(仅量价)", "SME1")]:
    for h in [5, 20]:
        M = mat(H[H.model == "TF"]) if src == "TF" else mat(pd.read_parquet(PR / f"{src}_h{h}.parquet")); ex, gr = LAB[h]["ex"], LAB[h]["gr"]; ic, nd = [], []
        for qq, i in enumerate(sig_idx):
            m = elig[i] & np.isfinite(M[qq]) & np.isfinite(ex[i])
            if m.sum() > 50: ic.append(spearmanr(M[qq][m], ex[i][m]).statistic); nd.append(P.ndcg_at(M[qq][m], gr[i][m], 5))
        extra.append(dict(strategy=nm, horizon=h, rankic=np.mean(ic), ndcg5=np.nanmean(nd)))
Q = pd.concat([q1, pd.DataFrame(extra)]); U = U.merge(Q, on=["strategy", "horizon"], how="left")
G = pd.concat([pd.read_csv(R / "buy_gate_e2.csv"), pd.read_csv(R / "buy_gate_sm.csv")]).drop_duplicates(["strategy", "horizon"])[["strategy", "horizon", "passes_buy_criteria"]]
U = U.merge(G, on=["strategy", "horizon"], how="left"); U["passes_buy_criteria"] = U.passes_buy_criteria.fillna(False)
U.to_csv(R / "p4_comparison_all.csv", index=False, encoding="utf-8-sig")
# ---- 2. S4 empirical calibration (picked names, slot return vs same-period pool) ----
ho = pd.read_csv(R / "bt_holdings_e2.csv", dtype={"code": str}); pe = pd.read_csv(R / "bt_periods_e2.csv")
s4 = ho[(ho.strategy == "S4_E2+低波混合缓冲") & ho.status.isin(["FILLED", "MARKED_LAST_CLOSE_UNSOLD"])].merge(pe[pe.strategy == "S4_E2+低波混合缓冲"][["signal_date", "horizon", "pool_ret"]], on=["signal_date", "horizon"])
s4["rel"] = s4.slot_ret_gross - s4.pool_ret; cal_rows = []
for h, x in [(5, .02), (20, .05)]:
    g = s4[s4.horizon == h]; r = g.rel; a_ = g.slot_ret_gross
    cal_rows.append(dict(horizon=h, n_picks=len(g), n_periods=g.signal_date.nunique(), threshold=x, freq_outperform=(r > x).mean(), freq_flat=((r >= -x) & (r <= x)).mean(), freq_underperform=(r < -x).mean(),
                         rel_q10=r.quantile(.1), rel_q50=r.median(), rel_q90=r.quantile(.9), abs_q10=a_.quantile(.1), abs_q50=a_.median(), abs_q90=a_.quantile(.9), abs_up_share=(a_ > 0).mean(),
                         note="S4 历史入选股的经验分布(单一分箱:入选/未入选),非概率预测;设计期已被反复使用"))
CAL = pd.DataFrame(cal_rows); CAL.to_csv(R / "s4_calibration.csv", index=False, encoding="utf-8-sig")
# ---- 3. per-stock history for the latest Top5 (both horizons) ----
top = pd.concat([pd.read_csv(f, dtype={"code": str}).assign(src=f.name) for f in sorted(REG.glob("p4_pred_*_E2S4_h*.csv"))]); top = top[top.in_final5]
top.to_csv(R / "p4_latest_top5_all.csv", index=False, encoding="utf-8-sig")
E5 = mat(pd.read_parquet(PR / "E2reg_h5.parquet")); ex5 = LAB[5]["ex"]; acc = []; A = np.where(X.traded, X.C / X.F, np.nan)
for cd in top.code.unique():
    j = col[cd]; rows = []
    for qq, i in enumerate(sig_idx):
        if not np.isfinite(E5[qq, j]): continue
        m = np.isfinite(E5[qq]); pc = (E5[qq][m] < E5[qq, j]).mean()
        rows.append(dict(signal_date=sig[qq], e2_pctile=pc, realized_ex5=ex5[i, j], s4_picked=bool(((s4.code == cd) & (s4.signal_date == sig[qq]) & (s4.horizon == 5)).any())))
    d = pd.DataFrame(rows); d.to_csv(R / f"p4_hist_{cd}.csv", index=False, encoding="utf-8-sig")
    hi = d[d.e2_pctile >= .8].dropna(); pk = s4[(s4.code == cd) & (s4.horizon == 5)]
    acc.append(dict(code=cd, name=X.u.name.values[j], n_dates=len(d), n_high_pred=len(hi), high_pred_beat_pool=(hi.realized_ex5 > 0).mean() if len(hi) else np.nan,
                    high_pred_mean_ex5=hi.realized_ex5.mean() if len(hi) else np.nan, corr_pred_vs_ex5=d[["e2_pctile", "realized_ex5"]].corr().iloc[0, 1] if d.realized_ex5.notna().sum() > 5 else np.nan,
                    s4_times_picked=len(pk), s4_pick_mean_rel=(pk.slot_ret_gross - pk.pool_ret).mean() if len(pk) else np.nan))
    k = np.arange(len(cal) - 300, len(cal))
    pd.DataFrame(dict(date=[str(cal[t].date()) for t in k], open=X.O[k, j], high=X.Hh[k, j], low=X.L[k, j], close=X.C[k, j], amount=X.AMT[k, j])).to_csv(R / f"p4_ohlc_{cd}.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(acc).to_csv(R / "p4_top5_stock_accuracy.csv", index=False, encoding="utf-8-sig")
# ---- 4. equity curves ----
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
per = pd.concat([pd.read_csv(R / "bt_periods_e2.csv"), pd.read_csv(R / "bt_periods_sm.csv")]).drop_duplicates(["strategy", "horizon", "signal_date"])
for h in [5, 20]:
    fig, ax = plt.subplots(figsize=(10, 5)); g = per[per.horizon == h]
    for s in ["P3_v2冻结基线", "因子_低波动", "E2_LGBM回归(量价+基本面)", "S4_E2+低波混合缓冲", "SM_S4_混合缓冲", "P1_Transformer原预测"]:
        x = g[g.strategy == s].sort_values("signal_date"); ax.plot(pd.to_datetime(x.signal_date), np.cumprod(1 + x.ret_net), label=s)
    x = g[g.strategy == "P3_v2冻结基线"].sort_values("signal_date"); ax.plot(pd.to_datetime(x.signal_date), np.cumprod(1 + x.pool_ret), "k--", label="股票池等权(未扣费)")
    ax.set_title(f"P4 扣费后净值 {h}日持有(探索性)"); ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FG / f"p4_equity_h{h}.png", dpi=110); plt.close(fig)
pd.set_option("display.width", 300); print(CAL.round(4).to_string()); print(pd.DataFrame(acc).round(3).to_string())
print(U[U.period == "全期"][["strategy", "horizon", "cum_gross", "cum_net", "rankic", "ndcg5"]].round(3).to_string())
