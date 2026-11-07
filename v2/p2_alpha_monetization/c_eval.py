# -*- coding: utf-8 -*-
"""P2-C evaluation: assemble per-block checkpoints into [207,N] score matrices and compute, for every (variant, seed): RankIC, rank stability, Top50 overlap, A0 turnover / net (C1 flat, C2 explicit),
factor alpha, seed variance, paired HAC / block-bootstrap comparisons. T0 seed 42 = the FROZEN P1 scores (D['scores']['TF']); everything else is read from c_checkpoints/."""
import sys, json, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as st
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as Q, p0lib as L, b_data as BD, b_sim as S, b_lib as BL
R = Q.RES; CK = Q.P2 / "c_checkpoints"; PPY = 252 / 5; NB = 52
PROTO_C = json.load(open(R / "c_protocol.json", encoding="utf-8"))
def load_variant(variant, seed, n_sig, N, sig_pos, nblocks=52):
    """-> scores [n_sig,N,nh] (NaN where missing), list of head horizons, set of covered signal rows"""
    A = None; heads = None; cov = np.zeros(n_sig, bool)
    for rr in range(nblocks):
        p = CK / variant / f"s{seed}_r{rr:03d}.npz"
        if not p.exists(): continue
        z = np.load(p, allow_pickle=False); meta = json.loads(str(z["meta"])); sc = z["scores"]; sig = z["sig"]
        if A is None: heads = meta["heads"]; A = np.full((n_sig, N, sc.shape[2]), np.nan, np.float32)
        for q, i in enumerate(sig): A[sig_pos[int(i)]] = sc[q]; cov[sig_pos[int(i)]] = True
    return A, heads, cov
def rank_pct(v, m):
    out = np.full(len(v), np.nan); x = v[m]; out[m] = (st.rankdata(x) - 0.5) / len(x); return out
def stability_series(Sc, cand):
    """Spearman of scores between adjacent signal dates over names that are candidates and finite at both; Top50 overlap (A0 ranking over candidates)."""
    K = len(Sc); stab = np.full(K, np.nan); ov = np.full(K, np.nan)
    tops = []
    for k in range(K):
        pool, ordd = S.rank_order(Sc[k], cand[k]) if np.isfinite(Sc[k][cand[k]]).any() else (np.array([], int), np.array([], int)); tops.append(set(int(j) for j in ordd[:50]) if len(ordd) >= 50 else None)
    for k in range(1, K):
        m = cand[k] & cand[k - 1] & np.isfinite(Sc[k]) & np.isfinite(Sc[k - 1])
        if m.sum() >= 30: stab[k] = Q.spearman(Sc[k][m], Sc[k - 1][m])
        if tops[k] is not None and tops[k - 1] is not None: ov[k] = len(tops[k] & tops[k - 1]) / 50
    return stab, ov
def rankic_series(Sc, cand, RET):
    K = len(Sc); ic = np.full(K, np.nan)
    for k in range(K):
        m = cand[k] & np.isfinite(Sc[k]) & np.isfinite(RET[k])
        if m.sum() >= 8: ic[k] = Q.spearman(Sc[k][m], RET[k][m])
    return ic
def ema_scores(Sc, cand, a=0.5):
    """post-hoc diagnostic: causal EMA of within-day percentile scores (uses only dates <= k)."""
    out = np.full_like(Sc, np.nan, dtype=np.float64); prev = None
    for k in range(len(Sc)):
        m = cand[k] & np.isfinite(Sc[k])
        if not m.any(): prev = None; continue
        r = rank_pct(Sc[k].astype(float), m); cur = r.copy()
        if prev is not None:
            mm = m & np.isfinite(prev); cur[mm] = a * r[mm] + (1 - a) * prev[mm]
        out[k] = cur; prev = cur
    return out

class Ctx:
    def __init__(self):
        self.D = BD.build(); D = self.D; self.sig = D["sig"]; self.K = D["K"]; self.N = D["N"]; self.sig_pos = {int(i): k for k, i in enumerate(self.sig)}
        P = pd.read_parquet(R / "p2a_panel.parquet", columns=["i", "ret_new_rel"]).sort_values("i", kind="stable"); self.RET = P.ret_new_rel.values.reshape(self.K, self.N)
        self.EW = pd.read_csv(R / "b_ew_benchmark.csv"); FR = pd.read_csv(R / "factor_returns_by_date.csv", index_col=0, encoding="utf-8-sig"); FR.columns = [c.split("(")[0] for c in FR.columns]; self.FR = FR; self.fcols = list(FR.columns)
        self.scen = {s: BL.PROTO["scenarios"][s] for s in ("C1_flat", "C2_explicit")}
        self.costs = {s: BL.make_cost(D, self.scen[s]) for s in self.scen}
    def sim(self, Sc, n, scen):
        D = self.D; sc = self.scen[scen]; cand = BD.cand_mask(D, dict(no_star=sc["no_star"], no_bj=sc["no_bj"], min_adv=sc["min_adv"]))[:n]; eng = dict(capital=sc["capital"], pcap=sc["pcap"], ADV=D["ADV"], min_adv=sc["min_adv"])
        o, _, _ = S.simulate_ext(D["X"], D["ADJf"], list(self.sig[:n]), S.A0Decider(Sc[:n], cand, self.sig[:n]), self.costs[scen], rebalance=False, name="A0", keep_logs=False, **eng); o = o.copy(); o["k"] = np.arange(len(o)); return o
    def run_metrics(self, Sc, n, ph=1, label=None):
        """Sc [>=n,N] scores of the primary head. Returns per-date DataFrame (k = 0..n-1) and dict of per-run summaries."""
        D = self.D; cand = D["cand0"][:n]; Sc = Sc[:n].astype(np.float64)
        ic = rankic_series(Sc, cand, self.RET[:n]); stab, ov = stability_series(Sc, cand)
        by = pd.DataFrame(dict(k=np.arange(n), signal_idx=self.sig[:n], ic=ic, stab=stab, top50_overlap=ov))
        for scen in ("C1_flat", "C2_explicit"):
            o = self.sim(Sc, n, scen); tag = "c1" if scen.startswith("C1") else "c2"
            by[f"net_{tag}"] = o.ret.values; by[f"gross_{tag}"] = o.gross_ret.values; by[f"turn_{tag}"] = o.turnover_value.values; by[f"cost_{tag}"] = (o.cost_total / o.nav_pre).values
        return by
    def summary(self, by, win="all"):
        b = by if win == "all" else by[by.k >= NB]; n = len(b); out = dict(n_signals=n)
        s = Q.summarize(b.ic.values); out.update(rankic=s["mean"], rankic_hac_t=s["hac_t"], rankic_ci_lo=s["ci_lo"], rankic_ci_hi=s["ci_hi"], rankic_pos_share=s["pos_share"])
        out.update(stability=float(np.nanmean(b.stab)), top50_overlap=float(np.nanmean(b.top50_overlap)))
        for tag in ("c1", "c2"):
            p = L.perf(b[f"net_{tag}"].values); pg = L.perf(b[f"gross_{tag}"].values)
            out.update({f"net_{tag}_per_period": float(b[f"net_{tag}"].mean()), f"ann_net_{tag}": p["annualized_return"], f"ann_gross_{tag}": pg["annualized_return"], f"sharpe_net_{tag}": p["sharpe"], f"turnover_{tag}": float(b[f"turn_{tag}"].mean()), f"cost_{tag}_per_period": float(b[f"cost_{tag}"].mean())})
            ew = self.EW.ew_ret_c0.values[b.k.values]; ex0 = b[f"net_{tag}"].values - ew; Fm = self.FR.loc[b.signal_idx.values, self.fcols].values
            beta, t, r2 = Q.ols_hac(ex0, Fm); out.update({f"alpha_{tag}": float(beta[0]), f"alpha_{tag}_hac_t": float(t[0]), f"excess_{tag}_mean": float(ex0.mean())})
            if tag == "c2": lo, hi = Q.alpha_ci(ex0, Fm, B_=1000); out.update(alpha_c2_ci_lo=lo, alpha_c2_ci_hi=hi, alpha_c2_r2=float(r2), **{"beta_" + c: float(beta[j + 1]) for j, c in enumerate(self.fcols)}, **{"tbeta_" + c: float(t[j + 1]) for j, c in enumerate(self.fcols)})
        return out

def paired(byA, byB, cols=("ic", "stab", "top50_overlap", "net_c1", "net_c2", "gross_c2", "turn_c2", "cost_c2"), ctx=None, label=""):
    """A - B on identical signal rows. Returns rows with mean diff, HAC t, block-bootstrap CI, share of dates A>B."""
    m = byA.merge(byB, on="k", suffixes=("_a", "_b")); rows = []
    for c in cols:
        d = (m[c + "_a"] - m[c + "_b"]).values; d = d[np.isfinite(d)]; s = Q.summarize(d)
        rows.append(dict(metric=c, n=len(d), diff_mean=s["mean"], hac_t=s["hac_t"], hac_p=s["hac_p"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"], share_A_gt_B=float((d > 0).mean())))
    if ctx is not None:      # alpha difference = intercept of the net-return difference regressed on the factors (OLS is linear)
        for tag in ("c1", "c2"):
            d = (m[f"net_{tag}_a"] - m[f"net_{tag}_b"]).values; Fm = ctx.FR.loc[m.signal_idx_a.values, ctx.fcols].values; beta, t, _ = Q.ols_hac(d, Fm); lo, hi = Q.alpha_ci(d, Fm, B_=1000)
            rows.append(dict(metric=f"alpha_{tag}", n=len(d), diff_mean=float(beta[0]), hac_t=float(t[0]), hac_p=2 * st.t.sf(abs(float(t[0])), len(d) - 6), ci_lo=lo, ci_hi=hi, share_A_gt_B=np.nan))
    return rows
def seed_mean_by(bys):
    """equal-weight average of per-date series across seeds (a 'seed-averaged' strategy/metric path)."""
    cols = [c for c in bys[0].columns if c not in ("k", "signal_idx")]; out = bys[0][["k", "signal_idx"]].copy()
    for c in cols: out[c] = np.nanmean(np.column_stack([b[c].values for b in bys]), axis=1)
    return out
def tint(x, conf=0.95):
    x = np.asarray(x, float); n = len(x); m = x.mean(); sd = x.std(ddof=1) if n > 1 else np.nan
    if n < 2: return m, sd, np.nan, np.nan
    h = st.t.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n); return m, sd, m - h, m + h