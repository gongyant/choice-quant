# -*- coding: utf-8 -*-
"""P5 Batch B: (4) five-stock comparison (TTM valuation, cash flow, announcements, 5d/20d risk) + 10-22 one-day update and B1/B3 lists
(5d and 20d, separate files, sha256-registered in p5 registry; 10-21 P4 registries untouched); (5) forward registry + pre-registered buy criterion.
Sources: P4 cache (read-only), Sina hq snapshot of 2026-10-22 (bars_20261022_sina.csv, fetched 15:35 UTC+8 after the close), Eastmoney yjbb/zcfz (P4 fin_raw),
Eastmoney announcement list (ann_5stocks.csv). Research ranking only, not advice; not connected to any broker."""
import json, glob, hashlib, os, stat, datetime as dt
import numpy as np, pandas as pd
import p5_common as Q
R = Q.P5 / "results"; DRY = os.environ.get("P5_DRYRUN") == "1"; REG = Q.P5 / ("_dry_registry" if DRY else "predictions_registry"); REG.mkdir(exist_ok=True); W = Q.load(); P = Q.P; X = W.X
D1 = "2026-10-22"; i0 = len(W.cal) - 1; assert str(W.cal[i0].date()) == "2026-10-21"
SB = pd.read_csv(Q.P5 / "bars_20261022_sina.csv", dtype={"code": str}, encoding="utf-8-sig").set_index("code").reindex(W.codes)
tr1 = (SB.date == D1).values & (SB.open.fillna(0) > 0).values
ret1 = np.where(tr1, SB.close.values / SB.prev_close.values - 1, np.nan)       # prev_close is exchange ex-right adjusted -> adjusted 1-day return
A0 = W.A[-1]; A1 = np.where(tr1, A0 * (1 + ret1), A0); Aext = np.vstack([W.A, A1])
r1 = pd.DataFrame(Aext).pct_change(fill_method=None); r1.iloc[-1] = np.where(tr1, ret1, np.nan)
vol1 = r1.rolling(20, min_periods=15).std().values[-1]
dd60_1 = (Aext[-1] / pd.DataFrame(Aext).rolling(60, min_periods=40).max().values[-1] - 1); r20_1 = Aext[-1] / Aext[-21] - 1; r5_1 = Aext[-1] / Aext[-6] - 1
lim_pct = np.where(np.isin(W.board, ["创业板", "科创板"]), 0.20, np.where(W.board == "北交所", 0.30, 0.10))
cur_st = (X.u.is_st_name.fillna(False) | X.u.name_has_delist_mark.fillna(False)).values.astype(bool); lim_pct = np.where(cur_st & ~np.isin(W.board, ["创业板", "科创板"]), 0.05, lim_pct)
limup1 = tr1 & (SB.close.values >= np.round(SB.prev_close.values * (1 + lim_pct), 2) - 0.001)
elig1 = W.elig[-1] & tr1                                                         # universe membership as of 10-21 (P4 rule), must trade on 10-22
filt1 = elig1 & ~limup1 & (np.nan_to_num(W.amt20[-1]) >= 1e8) & ~W.st[-1] & ~cur_st   # amt20 = 10-21 value (approximation, disclosed)
volp1 = pd.Series(np.where(elig1, vol1, np.nan)).rank(pct=True).values
cons1 = (np.nan_to_num(volp1, nan=1) <= .8) & (np.nan_to_num(dd60_1, nan=-1) > -.35) & (np.nan_to_num(r20_1, nan=-1) > -.25)
def pick(score, prev=()):
    s = np.where(filt1 & np.isfinite(score), score, np.nan); ok = np.where(np.isfinite(s))[0]; order = ok[np.argsort(-s[ok], kind="stable")]; rank = {j: r for r, j in enumerate(order)}; out, cnt = [], {}
    def add(j):
        if len(out) < 5 and cons1[j] and cnt.get(W.ind[j], 0) < 2 and j not in out: out.append(j); cnt[W.ind[j]] = cnt.get(W.ind[j], 0) + 1
    for j in prev:
        if j in rank and rank[j] < 40: add(j)
    for j in order[:20]: add(j)
    return out, order, s
lowv = np.where(filt1, -vol1, np.nan)
def pctm(v): m = filt1 & np.isfinite(v); o = np.full(len(v), np.nan); o[m] = pd.Series(v[m]).rank(pct=True).values; return o
# ---- fundamentals as of 10-22 (notice strictly before 10-22 for the signal; display uses same) ----
FR = Q.P4D / "fin_raw"
y = pd.concat([pd.read_csv(f, dtype={"SECURITY_CODE": str}) for f in sorted(glob.glob(str(FR / "yjbb_*.csv.gz")))]); y["period"] = pd.to_datetime(y.REPORTDATE.str[:10]); y["notice"] = pd.to_datetime(y.NOTICE_DATE.str[:10])
y = y.sort_values("notice").drop_duplicates(["SECURITY_CODE", "period"], keep="first"); y["q"] = y.period.dt.month // 3
K = y.set_index(["SECURITY_CODE", "period"])
def prior(c, p, f, back):
    k = (c, pd.Timestamp(p.year - back, p.month, p.day) if not (p.month == 12 and back == 0) else p); return K[f].get(k, np.nan)
def ttm(c, p, f):
    if p.month == 12: return K[f].get((c, p), np.nan)
    return K[f].get((c, p), np.nan) + K[f].get((c, pd.Timestamp(p.year - 1, 12, 31)), np.nan) - K[f].get((c, pd.Timestamp(p.year - 1, p.month, p.day)), np.nan)
def fund(c, asof=pd.Timestamp(D1)):
    a = y[(y.SECURITY_CODE == c) & (y.notice < asof)].sort_values("period")
    if not len(a): return dict(report_period="未取得")
    r = a.iloc[-1]; p = r.period; q = r.q; cl = SB.close.get(c, np.nan)
    eps_ttm = ttm(c, p, "BASIC_EPS"); np_ttm = ttm(c, p, "PARENT_NETPROFIT"); ocfps_ttm = ttm(c, p, "MGJYXJJE")
    return dict(report_period=str(p.date()), notice_date=str(r.notice.date()), revenue_ytd_yi=r.TOTAL_OPERATE_INCOME / 1e8, rev_yoy_pct=r.YSTZ, netprofit_ytd_yi=r.PARENT_NETPROFIT / 1e8, np_yoy_pct=r.SJLTZ,
                netprofit_ttm_yi=np_ttm / 1e8 if np.isfinite(np_ttm) else "未取得", eps_ytd=r.BASIC_EPS, eps_ttm=eps_ttm if np.isfinite(eps_ttm) else "未取得", eps_annualized=r.BASIC_EPS * 4 / q,
                roe_raw_ytd_pct=r.WEIGHTAVG_ROE, roe_annualized_pct=r.WEIGHTAVG_ROE * 4 / q, bps=r.BPS, ocf_per_share_ytd=r.MGJYXJJE, ocf_per_share_ttm=ocfps_ttm if np.isfinite(ocfps_ttm) else "未取得",
                ocf_to_eps_ytd=r.MGJYXJJE / r.BASIC_EPS if r.BASIC_EPS else np.nan, close_1022=cl, pe_ttm=cl / eps_ttm if np.isfinite(eps_ttm) and eps_ttm > 0 else "未取得(TTM EPS≤0或缺)",
                pe_annualized=cl / (r.BASIC_EPS * 4 / q) if r.BASIC_EPS > 0 else "未取得", pb=cl / r.BPS if r.BPS > 0 else "未取得", dividend_plan=r.ASSIGNDSCRPT, industry_em=r.PUBLISHNAME)
# ---- (4) five-stock comparison ----
FIVE = ["601066", "601318", "300015", "600515", "000581"]; AN = pd.read_csv(Q.P5 / "ann_5stocks.csv", dtype={"code": str}, encoding="utf-8-sig")
KEY = ["业绩", "预增", "预减", "收购", "股权", "冻结", "辞任", "回购", "利润分配", "权益分派", "董事长", "关联交易", "季度"]
Oa = np.where(X.traded, X.O / X.F, np.nan); cmp = []
for c in FIVE:
    j = W.col[c]; s = pd.Series(Aext[:, j]); h5 = (s.shift(-5) / s - 1).iloc[-500:-5]; h20 = (s.shift(-20) / s - 1).iloc[-500:-20]
    a = AN[AN.code == c]; maj = a[a.title.str.contains("|".join(KEY))].head(6)
    cmp.append(dict(code=c, name=W.names[j], board=W.board[j], industry_l2=W.ind[j], close_1021=X.C[i0, j], chg_1022_pct=ret1[j] * 100,
                    **fund(c), vol20_daily_pct=vol1[j] * 100, vol_pctile_pool=volp1[j], beta60=W.beta[-1, j], dd60_pct=dd60_1[j] * 100, r5_pct=r5_1[j] * 100, r20_pct=r20_1[j] * 100,
                    hist_5d_p05_pct=h5.quantile(.05) * 100, hist_5d_worst_pct=h5.min() * 100, hist_5d_up_share=(h5 > 0).mean(), hist_20d_p05_pct=h20.quantile(.05) * 100, hist_20d_worst_pct=h20.min() * 100, hist_20d_up_share=(h20 > 0).mean(),
                    major_announcements=" | ".join(f"{r.notice_date} {r.title}" for r in maj.itertuples()) or "未取得", latest_announcement=f"{a.notice_date.iloc[0]} {a.title.iloc[0]}" if len(a) else "未取得",
                    q3_2026_report_by_1022="已披露" if ((y.SECURITY_CODE == c) & (y.period == "2026-09-30") & (y.notice <= D1)).any() else "未取得(截至抓取时未披露)"))
CMP = pd.DataFrame(cmp); CMP.to_csv(R / "b4_five_stock_compare.csv", index=False, encoding="utf-8-sig")
# ---- 10-22 lists ----
E2L = {h: pd.read_parquet(Q.PR / f"E2reg_h{h}_latest.parquet") for h in [5, 20]}
man = REG / "manifest.jsonl"; regs = []
RULES = dict(version="p5_rules_v1", engine="p4lib.backtest", filters="elig(10-21)&traded 10-22&not close limit-up&amt20(10-21)>=1e8&not ST proxy&not current ST", constraints="Top20 -> volp<=0.8, dd60>-35%, r20>-25%, <=2 per industry_l2",
             buffer="keep previous simulated holdings while rank<40 (first live date: no previous holdings)", B1="score = -vol20 (low-vol)", B3="score = 0.5*pct(E2reg) + 0.5*pct(low-vol)",
             E2_note="E2reg model = P4 frozen final model; its score is the 10-21 latest prediction (features as of 10-21, labels matured). 10-22 features NOT rebuilt -> E2 half is one day stale (disclosed).")
rules_path = REG / "_dry_rules.json" if DRY else Q.P5 / "selection_rules_p5_v1.json"
if not rules_path.exists(): json.dump(RULES, open(rules_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
rules_sha = hashlib.sha256(rules_path.read_bytes()).hexdigest()
pool_f = REG / "p5_pool_20261022.csv"
if not pool_f.exists(): pd.DataFrame(dict(code=[W.codes[j] for j in np.where(elig1)[0]])).to_csv(pool_f, index=False, encoding="utf-8-sig"); regs.append(pool_f)
lists = []
for strat in ["B1", "B3"]:
    for h in [5, 20]:
        if strat == "B1": sc = lowv; mv = "lowvol_v1(无模型)"
        else:
            e2 = np.full(W.N, np.nan); lt = E2L[h]; e2[lt.code.map(W.col).values] = lt.score.values; sc = 0.5 * pctm(e2) + 0.5 * pctm(lowv); mv = f"E2reg_h{h}_latest (P4 frozen, sha in P4 preds meta) + lowvol"
        pk, order, s = pick(sc); rows = []
        for rk, j in enumerate(order[:20]):
            rows.append(dict(cand_rank=rk + 1, code=W.codes[j], name=W.names[j], board=W.board[j], industry=W.ind[j], close_1022=SB.close.iloc[j], select_score=s[j], in_final5=j in pk, vol20=vol1[j], volp=volp1[j], dd60=dd60_1[j], r20=r20_1[j],
                             sim_weight=0.2 if j in pk else 0.0, sim_shares_per_100k=int(20000 / SB.close.iloc[j] // 100 * 100) if j in pk else 0))
        d = pd.DataFrame(rows); d["strategy"] = strat; d["horizon"] = h; d["signal_date"] = D1; d["model_version"] = mv; d["rules_sha256"] = rules_sha
        d["note"] = "研究排名,不是买入推荐;select_score 不是概率/收益率;模拟持仓按 10-22 收盘价参考,实际模拟成交价=次日(10-23)开盘"
        out = REG / f"p5_pred_20261022_{strat}_h{h}.csv"
        if not out.exists(): d.to_csv(out, index=False, encoding="utf-8-sig"); (None if DRY else os.chmod(out, stat.S_IREAD)); regs.append(out)
        lists.append(pd.read_csv(out, dtype={"code": str}, encoding="utf-8-sig"))
done = set()
if man.exists(): done = {json.loads(l)["file"] for l in open(man, encoding="utf-8-sig")}
with open(man, "a", encoding="utf-8") as fh:
    for f in regs:
        if f.name not in done: fh.write(json.dumps(dict(file=f.name, sha256=hashlib.sha256(f.read_bytes()).hexdigest(), registered_at=dt.datetime.now().isoformat(timespec="seconds"), rules_sha256=rules_sha), ensure_ascii=False) + "\n")
L = pd.concat(lists); L.to_csv(R / "b4_latest_lists_20261022.csv", index=False, encoding="utf-8-sig")
# forward registry (append-only)
FW = REG / "_dry_forward.csv" if DRY else Q.P5 / "forward_registry.csv"; have = set(map(tuple, pd.read_csv(FW, dtype=str)[["signal_date", "strategy", "horizon"]].values)) if FW.exists() else set(); fw = []
for d in lists:
    k = (D1, d.strategy.iloc[0], str(d.horizon.iloc[0]))
    if k in have: continue
    f5 = d[d.in_final5]
    for r in f5.itertuples(): fw.append(dict(signal_date=D1, strategy=r.strategy, horizon=r.horizon, model_version=r.model_version, rules_sha256=r.rules_sha256, code=r.code, name=r.name, sim_weight=r.sim_weight,
                                             sim_shares_per_100k=r.sim_shares_per_100k, ref_close=r.close_1022, planned_entry="2026-10-23 开盘", planned_exit=f"entry+{r.horizon}交易日 开盘", registry_file=f"p5_pred_20261022_{r.strategy}_h{r.horizon}.csv"))
if fw: pd.DataFrame(fw).to_csv(FW, mode="a", header=not FW.exists(), index=False, encoding="utf-8-sig")
# ---- (5) pre-registered evidence-based buy criterion ----
CR = dict(version="p5_buy_criterion_v1", registered_at=dt.datetime.now().isoformat(timespec="seconds"),
          rule="一个策略在某期限下只有同时满足 C1-C6 才算'有证据支持的买入候选';否则一律'不通过:不构成买入依据'。",
          C1="回测(2022-07~2026-09, 扣0.30%双边成本)每期净超额(相对合格池)均值的 HAC t >= 2.0",
          C2="移动块自助 95% CI 下限 > 0(块长 5日=8, 20日=4)", C3="5 个年度中 >= 4 个年度净超额累计 > 0", C4="净值最大回撤 > -25%",
          C5="前向登记(10-22 起,只追加)已到期期数: 5日 >= 12 期, 20日 >= 6 期", C6="前向已到期期的平均净超额 > 0 且跑赢池比例 >= 50%",
          note="C1-C4 只看已有回测,样本已被反复使用(探索性)。即使全部通过,也只是'证据支持',不是投资建议,不连接券商不下单。")
cr_path = REG / "_dry_crit.json" if DRY else Q.P5 / "buy_criterion_p5_v1.json"
if not cr_path.exists(): json.dump(CR, open(cr_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
PER = pd.read_csv(R / "a1v2_periods.csv", encoding="utf-8-sig"); PER["ex"] = PER.ret_net - PER.pool_ret; G = []
nfw = pd.read_csv(Q.P5 / "forward_verify.csv", encoding="utf-8-sig") if (Q.P5 / "forward_verify.csv").exists() else pd.DataFrame(columns=["strategy", "horizon", "signal_date", "rel"])
for (nm, h), a in PER.groupby(["strategy", "horizon"]):
    e = a.ex.values; blk, lag = (8, 4) if h == 5 else (4, 2); lo, hi = Q.block_ci(e, blk); yrs = a.groupby("year").apply(lambda b: np.prod(1 + b.ret_net) / np.prod(1 + b.pool_ret) - 1)
    key = nm.split("_")[0]; fv = nfw[(nfw.strategy == key) & (nfw.horizon == h)] if len(nfw) else nfw; nper = fv.signal_date.nunique() if len(fv) else 0
    g = dict(strategy=nm, horizon=h, t_hac=Q.hac_t(e, lag), ci_lo=lo, ci_hi=hi, years_pos=int((yrs > 0).sum()), mdd=P.mdd(a.ret_net), fwd_matured_periods=nper)
    g.update(C1=g["t_hac"] >= 2, C2=lo > 0, C3=g["years_pos"] >= 4, C4=g["mdd"] > -.25, C5=nper >= (12 if h == 5 else 6), C6=False if nper == 0 else bool(fv.rel.mean() > 0 and (fv.rel > 0).mean() >= .5))
    g["PASS"] = all(g[k] for k in ["C1", "C2", "C3", "C4", "C5", "C6"]); g["结论"] = "通过(证据支持,仍非投资建议)" if g["PASS"] else "不通过:不构成买入依据"; G.append(g)
G = pd.DataFrame(G); G.to_csv(R / "b5_buy_gate.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 60)
print(CMP.T.to_string()); print(L[L.in_final5][["strategy", "horizon", "code", "name", "industry", "close_1022", "select_score", "vol20", "dd60", "r20"]].to_string()); print(G.round(3).to_string())
print("traded_1022", int(tr1.sum()), "elig1", int(elig1.sum()), "filt1", int(filt1.sum()), "limup1", int(limup1.sum()), "regs", [f.name for f in regs]); print(open(man, encoding="utf-8-sig").read())
