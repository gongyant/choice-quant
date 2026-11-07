# -*- coding: utf-8 -*-
"""P4 batch 3 (StockMixer) — derived from p4_e2_eval.py; P4 batch 2: E2 (price/volume + PIT fundamentals) vs P3 v2 / E0 / E1 / low-vol, plus the pre-registered Top20->Top5 selection module.
Same engine (p4lib.backtest), same dates, costs. EXPLORATORY. Requires selection_rules_p4_v1.json registered in rules_manifest.jsonl."""
import json, pickle, hashlib, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import p4lib as P
t0 = time.time(); C = P.P4 / "cache"; R = P.P4 / "results"; PR = P.P4 / "preds"
m = [json.loads(l) for l in open(P.P4 / "rules_manifest.jsonl", encoding="utf-8-sig")][-1]
assert hashlib.sha256((P.P4 / "selection_rules_p4_v1.json").read_bytes()).hexdigest() == m["sha256"], "rules changed after registration"
fl = np.load(C / "flags.npz"); LAB = pickle.load(open(C / "labels.pkl", "rb"))["lab"]; X = pickle.load(open(C / "world.pkl", "rb"))
cal = pd.DatetimeIndex(fl["cal"]); codes = list(fl["codes"]); col = {c: j for j, c in enumerate(codes)}; elig, amt20, olu, old, st = fl["elig"], fl["amt20"], fl["olu"], fl["old"], fl["st"]
names = X.u.name.values; ind = X.u.industry_l2.fillna("NA").values; board = X.u.board.values; N = len(codes)
H = pd.read_parquet(P.V2 / "p1_dl" / "historical_predictions_all_models.parquet", columns=["signal_date", "code", "model", "score", "eligible_asof"]); H = H[H.eligible_asof]
sig = sorted(H.signal_date.unique()); sig_idx = [int(cal.get_loc(pd.Timestamp(s))) for s in sig]; DESIGN_END = "2024-12-31"
def mat(df):
    M = np.full((len(sig), N), np.nan); k = {s: q for q, s in enumerate(sig)}; r = df.signal_date.map(k).values; c = df.code.map(col).values; ok = ~pd.isna(c) & ~pd.isna(r)
    M[r[ok].astype(int), c[ok].astype(int)] = df.score.values[ok]; return M
A = pd.DataFrame(np.where(X.traded, X.C / X.F, np.nan)).ffill(); dd60 = (A / A.rolling(60, min_periods=40).max() - 1).values; r20 = (A / A.shift(20) - 1).values
vol = pd.DataFrame(np.where(X.traded, X.C / X.F, np.nan)).ffill().pct_change(fill_method=None).rolling(20, min_periods=15).std().values
volp = pd.DataFrame(np.where(elig, vol, np.nan)).rank(axis=1, pct=True).values
lowvol = np.full((len(sig), N), np.nan)
for q, i in enumerate(sig_idx): lowvol[q, elig[i]] = -vol[i, elig[i]]
S = {"因子_低波动": {5: lowvol, 20: lowvol}, "P1_Transformer原预测": {5: mat(H[H.model == "TF"]), 20: mat(H[H.model == "TF"])}}
for mm, nm in [("E2reg", "E2_LGBM回归(量价+基本面)"), ("SME2", "SM_StockMixer(E2)"), ("SME1", "SM_StockMixer(仅量价)")]:
    S[nm] = {h: mat(pd.read_parquet(PR / f"{mm}_h{h}.parquet")) for h in [5, 20] if (PR / f"{mm}_h{h}.parquet").exists()}
p3 = pd.read_csv(P.P3 / "results" / "hist_top5_picks.csv", dtype={"code": str}, encoding="utf-8-sig"); p3 = p3[p3.strategy == "v2"].sort_values(["signal_date", "rank_in_pick"])
p3picks = {s: [col[c] for c in g.code] for s, g in p3.groupby("signal_date")}
def filt(i): return elig[i] & X.traded[i] & ~X.lim_up[i] & (np.nan_to_num(amt20[i]) >= 1e8) & ~st[i]
def cons(i): return (np.nan_to_num(volp[i], nan=1) <= 0.80) & (np.nan_to_num(dd60[i], nan=-1) > -0.35) & (np.nan_to_num(r20[i], nan=-1) > -0.25)
def pct(v, m): out = np.full(len(v), np.nan); out[m] = pd.Series(v[m]).rank(pct=True).values; return out
def plain(M, rows, K=5):
    out = []
    for q in rows:
        i = sig_idx[q]; s = np.where(filt(i), M[q], np.nan); ok = np.where(np.isfinite(s))[0]; out.append(ok[np.argsort(-s[ok], kind="stable")][:K])
    return out
def select(M, rows, buffer, blend, K=5, NB=40):
    out, prev, flags = [], [], []
    for q in rows:
        i = sig_idx[q]; f = filt(i) & np.isfinite(M[q]); s = M[q].copy()
        if blend: s = 0.5 * pct(M[q], f) + 0.5 * pct(lowvol[q], f & np.isfinite(lowvol[q]))
        s = np.where(f, s, np.nan); ok = np.where(np.isfinite(s))[0]; order = ok[np.argsort(-s[ok], kind="stable")]; rank = {j: r for r, j in enumerate(order)}
        ok_c = cons(i); pick, cnt = [], {}
        def add(j):
            if len(pick) < K and ok_c[j] and cnt.get(ind[j], 0) < 2 and j not in pick: pick.append(j); cnt[ind[j]] = cnt.get(ind[j], 0) + 1
        if buffer:
            for j in prev:
                if j in rank and rank[j] < NB: add(j)
        for j in order[:20]: add(j)
        out.append(np.array(pick, int)); prev = pick
    return out
per_all, hold_all, summ = [], [], []
for h in [5, 20]:
    rows = list(range(0, len(sig), 1 if h == 5 else 4)); si = [sig_idx[q] for q in rows]; bench = P.pool_bench(X, si, h, elig, olu)
    strat = [("P3_v2冻结基线", [p3picks.get(sig[q], []) for q in rows])] + [(k, plain(v[h], rows)) for k, v in S.items() if h in v]
    strat += [("S4_E2+低波混合缓冲", select(S["E2_LGBM回归(量价+基本面)"][h], rows, True, True))]
    if h in S["SM_StockMixer(E2)"]:
        M2 = S["SM_StockMixer(E2)"][h]; strat += [("SM_S3_混合固定", select(M2, rows, False, True)), ("SM_S4_混合缓冲", select(M2, rows, True, True))]
    for nm, pk in strat:
        pr, ho = P.backtest(X, si, pk, h, 5, 0.003, skip=st, olu=olu, old=old, names=names, strategy=nm)
        pr["horizon"], pr["pool_ret"] = h, bench; ho["horizon"] = h; ho["board"] = ho.code.map(lambda c: board[col[c]]); ho["industry"] = ho.code.map(lambda c: ind[col[c]])
        per_all.append(pr); hold_all.append(ho); fh = ho[ho.status.isin(["FILLED", "MARKED_LAST_CLOSE_UNSOLD"])]
        for per_nm, msk in [("全期", np.ones(len(pr), bool)), ("设计期2022-07~2024-12", pr.signal_date <= DESIGN_END), ("探索期2025-01~2026-09", pr.signal_date > DESIGN_END)]:
            a = pr[msk]; e = a.ret_net - a.pool_ret
            summ.append(dict(strategy=nm, horizon=h, period=per_nm, n_periods=len(a), avg_holdings=a.n_held.mean(), cum_gross=np.prod(1 + a.ret_gross) - 1, cum_net=np.prod(1 + a.ret_net) - 1,
                             cum_pool=np.prod(1 + a.pool_ret) - 1, mean_net=a.ret_net.mean(), mean_excess_net=e.mean(), t_excess=e.mean() / (e.std() / np.sqrt(len(a))), beat_pool_share=(e > 0).mean(),
                             mdd_net=P.mdd(a.ret_net), avg_turnover=a.turnover.mean(), total_fees=a.fees.sum(), star_chinext_share=float(fh.board.isin(["科创板", "创业板"]).mean()) if per_nm == "全期" else np.nan))
PER = pd.concat(per_all); HO = pd.concat(hold_all); SU = pd.DataFrame(summ)
# buy gate per strategy/horizon (pre-registered)
g = []
for (nm, h), a in SU.groupby(["strategy", "horizon"]):
    a = a.set_index("period"); ref = SU[(SU.horizon == h) & (SU.period == "全期")].set_index("strategy").cum_net
    g.append(dict(strategy=nm, horizon=h, c1_t_ge_2=a.loc["全期", "t_excess"] >= 2, c2_both_sub_pos=bool((a.mean_excess_net > 0).all()),
                  c3_beats_p3_lowvol_s4=bool(a.loc["全期", "cum_net"] > max(ref["P3_v2冻结基线"], ref["因子_低波动"], ref["S4_E2+低波混合缓冲"]))))
G = pd.DataFrame(g); G["passes_buy_criteria"] = G.c1_t_ge_2 & G.c2_both_sub_pos & G.c3_beats_p3_lowvol_s4
# ranking quality
ic = []
for nm in [k for k in S if k != "因子_低波动"]:
    for h in [hh for hh in [5, 20] if hh in S[nm]]:
        M = S[nm][h]; ex = LAB[h]["ex"]; v = []
        for q, i in enumerate(sig_idx):
            mm = elig[i] & np.isfinite(M[q]) & np.isfinite(ex[i])
            if mm.sum() > 50: v.append(spearmanr(M[q][mm], ex[i][mm]).statistic)
        ic.append(dict(strategy=nm, horizon=h, rankic=np.mean(v), rankic_ir=np.mean(v) / np.std(v)))
PER.to_csv(R / "bt_periods_sm.csv", index=False, encoding="utf-8-sig"); HO.to_csv(R / "bt_holdings_sm.csv", index=False, encoding="utf-8-sig")
SU.to_csv(R / "bt_summary_sm.csv", index=False, encoding="utf-8-sig"); G.to_csv(R / "buy_gate_sm.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(ic).to_csv(R / "rank_quality_sm.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(SU.round(4).to_string()); print(G.to_string()); print(pd.DataFrame(ic).round(4)); print("secs", round(time.time() - t0))
