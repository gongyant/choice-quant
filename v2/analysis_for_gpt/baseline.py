"""P3/P4 初步基线：大样本面板上的滚动样本外比较(只读研究，不下单)。
信号：交易日t收盘后；建仓：t+1收盘价；平仓：t+6收盘价(5个交易日持有)。
标签：5日未来收益减去当日可交易股票池的横截面均值(相对收益)。
训练：仅使用 future_end_date < 当前信号日 的样本(标签已全部实现)，并额外留出5日隔离带，避免重叠标签泄漏。
模型：Ridge(横截面分位数特征)、HistGradientBoosting、LightGBM；基线：20日动量Top-K、反转(r5)、等权全池。
可交易过滤(仅用信号时点及执行日的实际行情)：t日20日成交额中位数>=2000万元；t+1无成交且非涨停收盘；t+6有成交。
成本：简化往返0.30%(佣金+印花税+滑点粗估)，--fee 可调；这不是完整A股交易仿真(T+1/100股/涨跌停排队未模拟)。
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
V2 = Path(__file__).resolve().parent; DATA = V2 / "data"; OUT = V2 / "results" / "baseline"; OUT.mkdir(parents=True, exist_ok=True)
H = 5
FEATS = ["r1", "r5", "r20", "r60", "vol20", "ma_gap5", "ma_gap20", "volume_ratio5", "turnover5", "log_amt20", "range20"]
def limit_pct(board):
    return {"北交所": 0.30, "科创板": 0.20, "创业板": 0.20}.get(board, 0.10)
def build(panel):
    parts = []
    for code, g in panel.groupby("code", sort=False):
        f = g.sort_values("date").copy(); c = f.close
        f["r1"] = c.pct_change(1, fill_method=None); f["r5"] = c.pct_change(5, fill_method=None)
        f["r20"] = c.pct_change(20, fill_method=None); f["r60"] = c.pct_change(60, fill_method=None)
        f["vol20"] = f.r1.rolling(20).std(); f["ma_gap5"] = c / c.rolling(5).mean() - 1; f["ma_gap20"] = c / c.rolling(20).mean() - 1
        vs = f.volume_shares.where(f.volume_shares > 0)
        f["volume_ratio5"] = vs / vs.rolling(5).mean(); f["turnover5"] = f.turnover.rolling(5).mean()
        f["amt20"] = f.amount.rolling(20).median(); f["log_amt20"] = np.log1p(f.amt20)
        f["range20"] = ((f.high - f.low) / c).rolling(20).mean()
        f["buy_px"] = c.shift(-1); f["exit_px"] = c.shift(-(H + 1)); f["end_date"] = f.date.shift(-(H + 1))
        f["buy_vol"] = f.volume_raw.shift(-1); f["exit_vol"] = f.volume_raw.shift(-(H + 1))
        f["buy_ret"] = c.shift(-1) / c - 1
        f["fut_ret"] = f.exit_px / f.buy_px - 1
        lim = limit_pct(f.board.iloc[0]); f["buy_limit_up"] = f.buy_ret >= lim - 0.003
        parts.append(f)
    p = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    p["tradable"] = (p.amt20 >= 2e7) & (p.buy_vol > 0) & (~p.buy_limit_up) & (p.exit_vol > 0)
    # 信号时点可见的过滤(用于训练/评分股票池)：只用t日及以前信息
    p["eligible_t"] = (p.amt20 >= 2e7) & p.r60.notna() & p.vol20.notna()
    return p.sort_values(["date", "code"]).reset_index(drop=True)
def rank_feats(df):
    return df.groupby("date")[FEATS].rank(pct=True) - 0.5
def make_models(seed):
    return {
        "Ridge": lambda: Ridge(alpha=50.0),
        "HGB": lambda: HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, max_depth=4, min_samples_leaf=200, learning_rate=0.05, l2_regularization=1.0, random_state=seed),
        "LGBM": lambda: lgb.LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=4)}
def perf(r, h=H):
    r = pd.Series(r).dropna().astype(float)
    if len(r) < 3: return {"periods": int(len(r))}
    eq = (1 + r).cumprod(); ppy = 252 / h
    ann = float(eq.iloc[-1] ** (ppy / len(r)) - 1); vol = float(r.std() * np.sqrt(ppy))
    mdd = float((eq / eq.cummax() - 1).min())
    return {"periods": int(len(r)), "total_return": round(float(eq.iloc[-1] - 1), 4), "annualized_return": round(ann, 4), "annualized_vol": round(vol, 4),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(ppy)), 3) if r.std() > 0 else None, "max_drawdown": round(mdd, 4),
            "calmar": round(ann / abs(mdd), 3) if mdd < 0 else None, "win_rate": round(float((r > 0).mean()), 4)}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("panel", nargs="?", default="panel_1500.parquet")
    ap.add_argument("--topk", type=int, default=50); ap.add_argument("--fee", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--retrain", type=int, default=4); ap.add_argument("--first-test", default="2022-07-01")
    a = ap.parse_args()
    panel = pd.read_parquet(DATA / a.panel)
    p = build(panel); dates = pd.DatetimeIndex(sorted(p.date.unique()))
    p[[f"q_{x}" for x in FEATS]] = rank_feats(p)
    qf = [f"q_{x}" for x in FEATS]
    tr_pool = p[p.eligible_t & p.fut_ret.notna()].copy()
    tr_pool["cs_mean"] = tr_pool.groupby("date").fut_ret.transform("mean")
    tr_pool["y"] = tr_pool.fut_ret - tr_pool.cs_mean
    lo, hi = tr_pool.groupby("date").y.transform(lambda s: s.quantile(.01)), tr_pool.groupby("date").y.transform(lambda s: s.quantile(.99))
    tr_pool["y_w"] = tr_pool.y.clip(lo, hi)
    first = dates.searchsorted(pd.Timestamp(a.first_test)); sig_dates = dates[first:len(dates) - H - 1:H]
    models = make_models(a.seed); names = list(models) 
    recs, ics, holdings = [], [], {n: None for n in names + ["Momentum20", "Reversal5"]}
    turn = {n: [] for n in holdings}; fitted = {}; last_fit = None
    for ix, dt_ in enumerate(sig_dates):
        # 留出隔离带：只用 future_end_date < dt_ - 5个交易日(日历上通过dates位置)
        cutoff = dates[max(0, dates.get_loc(dt_) - H)]
        if ix % a.retrain == 0:
            tr = tr_pool[(tr_pool.end_date < cutoff) & (tr_pool.date >= dt_ - pd.Timedelta(days=3 * 365))]
            if len(tr) < 20000: continue
            for n in names:
                m = models[n](); m.fit(tr[qf].fillna(0), tr.y_w); fitted[n] = m
            last_fit = dict(train_rows=int(len(tr)), train_end=str(tr.date.max().date()), asof=str(dt_.date()))
        if not fitted: continue
        day = p[(p.date == dt_) & p.eligible_t & p.tradable].copy()
        if len(day) < max(100, a.topk * 2): continue
        day["cs_mean"] = day.fut_ret.mean()
        row = {"signal_date": str(dt_.date()), "n_eligible": int(len(day)), "universe_ew_gross": float(day.fut_ret.mean())}
        x = day[qf].fillna(0)
        scores = {n: fitted[n].predict(x) for n in names}
        scores["Momentum20"] = day.r20.values; scores["Reversal5"] = -day.r5.values
        for n, sc in scores.items():
            day["_s"] = sc
            top = day.nlargest(a.topk, "_s")
            codes = set(top.code)
            prev = holdings[n]
            turn[n].append(1.0 if prev is None else 1 - len(codes & prev) / len(codes)); holdings[n] = codes
            row[f"{n}_gross"] = float(top.fut_ret.mean()); row[f"{n}_net"] = float(top.fut_ret.mean()) - a.fee * turn[n][-1] - 0.0  # 仅对换仓部分计成本
            row[f"{n}_turnover"] = turn[n][-1]
            y = day.fut_ret - day.cs_mean
            row[f"{n}_IC"] = float(np.corrcoef(sc, y)[0, 1]) if np.std(sc) > 0 else np.nan
            row[f"{n}_RankIC"] = float(spearmanr(sc, y).statistic) if np.std(sc) > 0 else np.nan
        row["EqualWeight_net"] = row["universe_ew_gross"]  # 基准：等权全池，不计成本(被动基准)
        recs.append(row)
    bt = pd.DataFrame(recs); bt.to_csv(OUT / "walk_forward_returns.csv", index=False, encoding="utf-8-sig")
    res = {"config": vars(a), "last_fit": last_fit, "first_signal": bt.signal_date.iloc[0], "last_signal": bt.signal_date.iloc[-1], "periods": int(len(bt)),
           "avg_eligible_per_signal": round(float(bt.n_eligible.mean()), 1), "strategies": {}}
    for n in names + ["Momentum20", "Reversal5"]:
        d = perf(bt[f"{n}_net"]); d["gross_mean_per_period"] = round(float(bt[f"{n}_gross"].mean()), 5)
        d["excess_vs_equal_weight_per_period"] = round(float((bt[f"{n}_gross"] - bt.universe_ew_gross).mean()), 5)
        d["IC_mean"] = round(float(bt[f"{n}_IC"].mean()), 4); d["RankIC_mean"] = round(float(bt[f"{n}_RankIC"].mean()), 4)
        sd = bt[f"{n}_RankIC"].std(); d["RankIC_IR"] = round(float(bt[f"{n}_RankIC"].mean() / sd), 3) if sd > 0 else None
        d["avg_turnover"] = round(float(bt[f"{n}_turnover"].mean()), 3)
        t_ex = (bt[f"{n}_gross"] - bt.universe_ew_gross); d["excess_tstat"] = round(float(t_ex.mean() / (t_ex.std() / np.sqrt(len(t_ex)))), 2)
        by = {}
        for y, g in bt.groupby(bt.signal_date.str[:4]): by[y] = {"periods": int(len(g)), "mean_net_per_period": round(float(g[f"{n}_net"].mean()), 5), "RankIC": round(float(g[f"{n}_RankIC"].mean()), 4)}
        d["by_year"] = by; res["strategies"][n] = d
    res["strategies"]["EqualWeight"] = perf(bt.EqualWeight_net)
    res["limitations"] = ["股票池=1500只分层样本，不是全市场；在市股票来自当前清单，仅含2021年后退市股，仍有幸存者偏差", "前复权价格由数据源在下载时点计算，非逐日点时复权", "行业为最新分类，未做行业中性", "未模拟T+1卖出约束外的涨跌停排队、100股整数手、容量与冲击成本；成本为粗估", "ST/退市风险标记未按时点处理", "买入价=t+1收盘价(需盘尾成交假设)"]
    with open(OUT / "backtest_metrics.json", "w", encoding="utf-8") as f: json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in res["strategies"].items()}, ensure_ascii=False, indent=1)[:6000])
    # 最新研究排序(描述性，非买入建议)：用最后一次训练的模型对最新交易日打分
    last = p[p.date == dates.max()]; last = last[last.eligible_t].copy()
    if len(last) and "LGBM" in fitted:
        last["lgbm_score"] = fitted["LGBM"].predict(last[qf].fillna(0)); last["ridge_score"] = fitted["Ridge"].predict(last[qf].fillna(0))
        cols = ["date", "code", "board", "industry_l1", "close", "r5", "r20", "r60", "vol20", "amt20", "lgbm_score", "ridge_score"]
        last.sort_values("lgbm_score", ascending=False)[cols].head(50).to_csv(OUT / "latest_research_ranking_DRAFT.csv", index=False, encoding="utf-8-sig")
if __name__ == "__main__": main()