"""P1-A2: replay the P0-fixed walk-forward (baseline_fixed.py logic, identical calls) while RECORDING every day x stock model output.
Outputs: historical_predictions.parquet (long: signal_date x code x model_id), strategy_trades.csv, reports/reconciliation_p0.{csv,md},
         reports/recent20_metrics.{csv,md}, historical_predictions_sample.csv, historical_predictions_summary.csv
Nothing under p0/ is modified; the P0 files are only read for reconciliation."""
import sys, json, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1lib as P1
from p1lib import L, B, H, V2
t0 = time.time(); TAG = "fixed_sina_adj"; TOPK = 50; SEED = 42; RETRAIN = 4; FIRST = "2022-07-01"; MODELS = ["Ridge", "HGB", "LGBM"]; C30 = 0.003
X = L.build_data(); L.limit_flags(X); W = P1.world_from_X(X); f, elig, ADJf, lab, y, yw, Xq, cal = W.f, W.elig, W.ADJf, W.lab, W.y, W.yw, W.Xq, W.cal
T, N = elig.shape; codes = np.array(X.codes); first = cal.searchsorted(pd.Timestamp(FIRST)); sig_all = list(range(first, T - L.H - 1, L.H))
models = B.make_models(SEED); m_tr = elig & np.isfinite(yw); fitted = {}; cur = None
sig_used, pools, topk_sets, ic_rows = [], [], {n: [] for n in MODELS}, []
S = {n: [] for n in MODELS}; RK = {n: [] for n in MODELS}; PC = {n: [] for n in MODELS}; META = []
for ix, i in enumerate(sig_all):
    if ix % RETRAIN == 0:
        Xt, yt = L.train_rows(Xq, yw, m_tr, cal, i)
        if len(Xt) < 20000: continue
        for n in MODELS: m = models[n](); m.fit(Xt, yt); fitted[n] = m
        lo = int(cal.searchsorted(cal[i] - pd.Timedelta(days=365 * 3)))
        cur = dict(block=ix // RETRAIN, retrain_signal_date=str(cal[i].date()), train_first_date=str(cal[lo].date()), train_end_date=str(cal[i - 12].date()), train_label_end_date=str(cal[i - 12 + L.H + 1].date()), n_train_rows=int(len(Xt)))
        print("retrain", ix, cal[i].date(), len(Xt), round(time.time() - t0), "s", flush=True)
    if not fitted: continue
    pool = np.where(elig[i])[0]
    if len(pool) < max(100, TOPK * 2): continue
    xs = Xq[i][pool]; yy = y[i][pool]; fin = np.isfinite(yy); rec = {}
    for n in MODELS:
        s = fitted[n].predict(xs); order = np.argsort(-np.nan_to_num(s, nan=-1e9), kind="stable"); topk_sets[n].append(pool[order[:TOPK]])
        r = np.empty(len(pool)); r[order] = np.arange(1, len(pool) + 1); S[n].append(s); RK[n].append(r); PC[n].append(1 - (r - 1) / (len(pool) - 1))
        if fin.sum() > 5 and np.std(s[fin]) > 0: rec[f"{n}_IC"] = float(np.corrcoef(s[fin], yy[fin])[0, 1]); rec[f"{n}_RankIC"] = float(spearmanr(s[fin], yy[fin]).statistic)
        else: rec[f"{n}_IC"] = rec[f"{n}_RankIC"] = np.nan
    sig_used.append(i); pools.append(pool); ic_rows.append(rec); META.append(dict(cur))
assert all(np.diff(sig_used) == L.H); K = len(sig_used); print("signals", K, "walk-forward done", round(time.time() - t0), "s", flush=True)
# ---------------- simulate (same engine, 30bps) ----------------
SIM = {}; LOGS = {}
for n in MODELS:
    o, lg, ex = L.simulate(X, ADJf, sig_used, topk_sets[n], C30, capital=1e7, log_name=n); SIM[n] = o; LOGS[n] = lg
oE, _, _ = L.simulate(X, ADJf, sig_used, pools, C30, capital=1e9); SIM["EqualWeight"] = oE
print("simulated", round(time.time() - t0), "s", flush=True)
# ---------------- reconciliation vs P0 files ----------------
P0R = pd.read_csv(L.P0 / "results" / TAG / "period_returns.csv"); P0O = pd.read_csv(L.P0 / "results" / TAG / "orders_primary_cost30bps.csv", dtype={"code": str}); P0S = pickle.load(open(L.P0 / "results" / TAG / "sets.pkl", "rb"))
rc = []
def add(check, n, diff, extra=""): rc.append(dict(check=check, model=n, max_abs_diff=float(diff), n=int(K), note=extra))
add("signal dates identical", "-", 0.0 if [str(cal[i].date()) for i in sig_used] == P0S["signal_dates"] else 1.0, f"{K} periods {cal[sig_used[0]].date()}..{cal[sig_used[-1]].date()}")
add("n_pool", "-", np.abs(np.array([len(p) for p in pools]) - P0R.n_pool.values).max())
for n in MODELS:
    add("net period return c30 (simulate rerun vs period_returns.csv)", n, np.abs(SIM[n].ret.values - P0R[f"{n}_ret_c30"].values).max())
    same = sum(set(codes[a]) == set(b) for a, b in zip(topk_sets[n], P0S["topk"][n])); add("Top50 set identical to sets.pkl (periods differing)", n, K - same, f"{same}/{K} identical")
    add("RankIC per period", n, np.nanmax(np.abs(np.array([r[f"{n}_RankIC"] for r in ic_rows]) - P0R[f"{n}_RankIC"].values)))
    add("IC per period", n, np.nanmax(np.abs(np.array([r[f"{n}_IC"] for r in ic_rows]) - P0R[f"{n}_IC"].values)))
    for k_ in ["turnover_value", "cash_frac_post", "buy_filled", "n_hold"]: add(f"{k_} (30bps)", n, np.abs(SIM[n][k_].values - P0R[f"{n}_{k_}"].values).max())
    a = LOGS[n].reset_index(drop=True); b = P0O[P0O.strategy == n].reset_index(drop=True)
    ok = len(a) == len(b) and (a.code.values == b.code.values).all() and (a.status.values == b.status.values).all() and np.allclose(a.value.values, b.value.values) and np.allclose(a.cost.values, b.cost.values)
    add("order log identical to orders_primary_cost30bps.csv", n, 0.0 if ok else 1.0, f"rows {len(a)} vs {len(b)}")
add("net period return c30", "EqualWeight", np.abs(SIM["EqualWeight"].ret.values - P0R["EqualWeight_ret_c30"].values).max())
rcd = pd.DataFrame(rc); (H / "reports").mkdir(exist_ok=True); rcd.to_csv(H / "reports" / "reconciliation_p0.csv", index=False, encoding="utf-8-sig")
print(rcd.to_string(), flush=True)
# ---------------- long historical predictions ----------------
rows_sig = np.array(sig_used); di = np.repeat(rows_sig, N); cj = np.tile(np.arange(N), K); ki = np.repeat(np.arange(K), N)
FE = Xq[di, cj, :].astype(np.float32); el = elig[di, cj]; FE[~el] = np.nan
lab_r = lab[di, cj]; y_r = np.where(el, y[di, cj], np.nan); has = X.has_row
stale = ~(has[np.minimum(di + 1, T - 1), cj] & has[np.minimum(di + L.H + 1, T - 1), cj])
nbar = np.array([len(p) for p in pools])
blog = {}
for n in MODELS:
    for r in LOGS[n].itertuples():
        if r.side == "BUY": blog[(n, r.signal_date, r.code)] = (r.status, r.reason if isinstance(r.reason, str) else "", r.fill_date if isinstance(r.fill_date, str) else "", r.fill_price, r.shares)
parts = []; sdates = np.array([str(cal[i].date()) for i in sig_used]); odates = np.array([str(cal[i + 1].date()) for i in sig_used]); edates = np.array([str(cal[i + L.H + 1].date()) for i in sig_used])
for n in MODELS:
    sc = np.full((K, N), np.nan); rk = np.full((K, N), np.nan); pc = np.full((K, N), np.nan); top = np.zeros((K, N), bool)
    for k in range(K): sc[k, pools[k]] = S[n][k]; rk[k, pools[k]] = RK[n][k]; pc[k, pools[k]] = PC[n][k]; top[k, topk_sets[n][k]] = True
    score = sc[ki, cj]; tp = top[ki, cj]
    bs = np.full(len(di), "NOT_TOP50", dtype=object); br = np.full(len(di), "", dtype=object); bf = np.full(len(di), "", dtype=object); bp = np.full(len(di), np.nan); bsh = np.full(len(di), np.nan)
    for idx in np.where(tp)[0]:
        e = blog.get((n, sdates[ki[idx]], codes[cj[idx]]))
        if e is None: bs[idx] = "RETAINED_ALREADY_HELD"
        else: bs[idx], br[idx], bf[idx], bp[idx], bsh[idx] = e
    mid = np.array([f"{n}_P0wf_s{SEED}_r{META[k]['block']:03d}" for k in range(K)], dtype=object)[ki]
    d = pd.DataFrame({"signal_date": sdates[ki], "code": codes[cj], "model": n, "model_id": mid, "train_first_date": np.array([m["train_first_date"] for m in META], dtype=object)[ki], "train_end_date": np.array([m["train_end_date"] for m in META], dtype=object)[ki],
                      "train_label_end_date": np.array([m["train_label_end_date"] for m in META], dtype=object)[ki], "n_train_rows": np.array([m["n_train_rows"] for m in META])[ki], "eligible_asof": el, "n_pool": nbar[ki],
                      "score": score, "rank": rk[ki, cj], "pctile": pc[ki, cj], "in_top50": tp, "buy_status": bs, "buy_reason": br, "buy_fill_date": bf, "buy_price_unadj": bp, "buy_shares": bsh,
                      "order_date": odates[ki], "exit_date": edates[ki], "realized_ret_5d": lab_r, "realized_rel_ret_5d": y_r, "label_uses_stale_price": stale & np.isfinite(lab_r)})
    for j_, k_ in enumerate(L.FEATS): d["f_" + k_] = FE[:, j_]
    parts.append(d)
HP = pd.concat(parts, ignore_index=True); HP["rank"] = HP["rank"].astype("float32"); HP["score"] = HP.score.astype("float64"); HP["pctile"] = HP.pctile.astype("float32")
HP.to_parquet(H / "historical_predictions.parquet", index=False, compression="zstd"); print("parquet", HP.shape, round(time.time() - t0), "s", flush=True)
HP[HP.signal_date.isin(sdates[-3:])].to_csv(H / "historical_predictions_sample.csv", index=False, encoding="utf-8-sig")
# summary per model x date
g = HP[HP.eligible_asof].copy()
sm = []
for (m, d_), gg in g.groupby(["model", "signal_date"], sort=True):
    t_ = gg[gg.in_top50]; sm.append(dict(model=m, signal_date=d_, model_id=gg.model_id.iloc[0], train_end_date=gg.train_end_date.iloc[0], n_pool=len(gg), n_label=int(gg.realized_rel_ret_5d.notna().sum()), top50_mean_rel_ret_5d=t_.realized_rel_ret_5d.mean(), top50_mean_ret_5d=t_.realized_ret_5d.mean(),
                                         top50_buy_filled=int((t_.buy_status == "FILLED").sum()), top50_retained=int((t_.buy_status == "RETAINED_ALREADY_HELD").sum()), top50_rejected=int((t_.buy_status == "REJECTED").sum())))
sm = pd.DataFrame(sm); ic = pd.DataFrame(ic_rows); ic["signal_date"] = sdates
for n in MODELS:
    mp = ic.set_index("signal_date"); sm.loc[sm.model == n, "IC"] = sm[sm.model == n].signal_date.map(mp[f"{n}_IC"]).values; sm.loc[sm.model == n, "RankIC"] = sm[sm.model == n].signal_date.map(mp[f"{n}_RankIC"]).values
    sm.loc[sm.model == n, "net_ret_c30"] = SIM[n].ret.values; sm.loc[sm.model == n, "ew_net_ret_c30"] = SIM["EqualWeight"].ret.values
sm["net_excess_c30"] = sm.net_ret_c30 - sm.ew_net_ret_c30; sm.to_csv(H / "historical_predictions_summary.csv", index=False, encoding="utf-8-sig")
# ---------------- recent 20 periods ----------------
r20 = sm.sort_values(["model", "signal_date"]).groupby("model").tail(20)[["model", "signal_date", "model_id", "IC", "RankIC", "net_ret_c30", "ew_net_ret_c30", "net_excess_c30", "top50_mean_rel_ret_5d", "top50_buy_filled", "top50_rejected"]]
r20.to_csv(H / "reports" / "recent20_metrics.csv", index=False, encoding="utf-8-sig")
agg = r20.groupby("model").agg(IC_mean=("IC", "mean"), RankIC_mean=("RankIC", "mean"), RankIC_pos_periods=("RankIC", lambda s: int((s > 0).sum())), net_ret_mean_per_period=("net_ret_c30", "mean"), net_cum=("net_ret_c30", lambda s: float((1 + s).prod() - 1)),
                               ew_cum=("ew_net_ret_c30", lambda s: float((1 + s).prod() - 1)), net_excess_mean=("net_excess_c30", "mean")).reset_index()
agg["n_periods"] = 20; agg["period_range"] = f"{r20.signal_date.min()}..{r20.signal_date.max()}"; agg["excess_cum_simple"] = agg.net_cum - agg.ew_cum
agg.to_csv(H / "reports" / "recent20_summary.csv", index=False, encoding="utf-8-sig")
md = ["# 最近 20 期实测指标(5日调仓,往返成本0.30%,Top50,等权基准为同制度含成本等权)", "", f"区间(信号日):{agg.period_range.iloc[0]};共 20 期 ≈ 100 个交易日。**样本很少,仅描述性,不能据此判断模型好坏。** 数据来自 P0 walk-forward 重放(与 P0 period_returns.csv 对账一致,见 reconciliation_p0.md)。2025-2026 数据已多次用于模型比较,不是盲测。", "",
      "| 模型 | IC均值 | RankIC均值 | RankIC>0期数/20 | 净收益均值/期 | 20期累计净收益 | 同期等权累计 | 累计差(简单相减) | 净超额均值/期 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in agg.itertuples(): md.append(f"| {r.model} | {r.IC_mean:.4f} | {r.RankIC_mean:.4f} | {r.RankIC_pos_periods} | {r.net_ret_mean_per_period:+.2%} | {r.net_cum:+.2%} | {r.ew_cum:+.2%} | {r.excess_cum_simple:+.2%} | {r.net_excess_mean:+.2%} |")
md += ["", "## 逐期明细", ""]
for m in MODELS:
    md += [f"### {m}", "", "| 信号日 | model_id | IC | RankIC | 净收益 | 等权净收益 | 净超额 | Top50相对收益均值 | Top50 买单成交/拒绝 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in r20[r20.model == m].itertuples(): md.append(f"| {r.signal_date} | {r.model_id} | {r.IC:+.3f} | {r.RankIC:+.3f} | {r.net_ret_c30:+.2%} | {r.ew_net_ret_c30:+.2%} | {r.net_excess_c30:+.2%} | {r.top50_mean_rel_ret_5d:+.2%} | {r.top50_buy_filled}/{r.top50_rejected} |")
    md.append("")
(H / "reports" / "recent20_metrics.md").write_text("\n".join(md), encoding="utf-8")
# ---------------- strategy trades ----------------
UM = pd.read_csv(V2 / "data" / "universe_master.csv", dtype=str).set_index("code"); out = []
for n in MODELS:
    lg = LOGS[n]; opn = {}; nb = {}
    for r in lg.itertuples():
        if r.side == "BUY":
            if r.status == "FILLED": opn[r.code] = r; nb[r.code] = 0
            else: out.append(dict(strategy=n, code=r.code, signal_date=r.signal_date, planned_buy_date=r.planned_date, actual_buy_date="", buy_price_unadj=r.fill_price, trade_status="BUY_REJECTED", unfilled_reason=r.reason))
        else:
            if r.status == "BLOCKED_RETRY":
                nb[r.code] = nb.get(r.code, 0) + 1; continue
            b = opn.pop(r.code, None)
            if b is None: continue
            bf, sf = b.cost, r.cost; net = (r.value - sf) / (b.value + bf) - 1
            out.append(dict(strategy=n, code=r.code, signal_date=b.signal_date, planned_buy_date=b.planned_date, actual_buy_date=b.fill_date, buy_price_unadj=b.fill_price, buy_shares=b.shares, buy_value=b.value, buy_fee=bf,
                            planned_sell_date=r.planned_date, actual_sell_date=r.fill_date, sell_price_unadj=r.fill_price, sell_delay_days=r.sell_delay_days, sell_shares_equiv=r.shares, sell_value_div_adj=r.value, sell_fee=sf, slippage_cost=0.0,
                            sell_blocked_events=nb.pop(r.code, 0), gross_ret_div_adj=r.value / b.value - 1, net_ret=net, trade_status="CLOSED", unfilled_reason=""))
    for c, b in opn.items(): out.append(dict(strategy=n, code=c, signal_date=b.signal_date, planned_buy_date=b.planned_date, actual_buy_date=b.fill_date, buy_price_unadj=b.fill_price, buy_shares=b.shares, buy_value=b.value, buy_fee=b.cost, sell_blocked_events=nb.get(c, 0), trade_status="OPEN_AT_END_OR_STUCK", unfilled_reason="数据终止/未卖出至样本末"))
TR = pd.DataFrame(out); TR["name"] = TR.code.map(UM["name"]); TR["industry_l1"] = TR.code.map(UM["industry_l1"])
TR["note"] = "数量按P0模拟:本金1000万元、Top50等权目标市值=NAV/50、100股整手;往返成本0.30%(每边0.15%)已含佣金+印花税+滑点假设,slippage_cost不单列;卖出价值按不复权价*复权因子口径(含分红再投资);买不到不换股"
cols = ["strategy", "signal_date", "code", "name", "industry_l1", "planned_buy_date", "actual_buy_date", "buy_price_unadj", "buy_shares", "buy_value", "buy_fee", "planned_sell_date", "actual_sell_date", "sell_price_unadj", "sell_delay_days", "sell_shares_equiv", "sell_value_div_adj", "sell_fee", "slippage_cost",
        "sell_blocked_events", "gross_ret_div_adj", "net_ret", "trade_status", "unfilled_reason", "note"]
TR = TR[cols].sort_values(["strategy", "signal_date", "code"]); TR.to_csv(H / "strategy_trades.csv", index=False, encoding="utf-8-sig")
print("trades", TR.shape, TR.trade_status.value_counts().to_dict(), round(time.time() - t0), "s")
json.dump(dict(n_signals=K, first_signal=sdates[0], last_signal=sdates[-1], n_model_ids=int(HP.model_id.nunique()), rows=int(len(HP)), retrain_blocks=int(len({m['block'] for m in META})), seconds=round(time.time() - t0)), open(H / "scores" / "history_run_info.json", "w"), indent=1)
