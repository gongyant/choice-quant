# -*- coding: utf-8 -*-
"""EW benchmark through the P0 simulator (cost 0 and 0.30%) + A0 reconciliation of all six models to the P0 / track-B files (flat 0.30%, constraints off)."""
import json, numpy as np, pandas as pd
import p2lib as Q, b_sim as S, b_data as BD, p0lib as L
D = BD.build(); X, sig, ADJf, K = D["X"], D["sig"], D["ADJf"], D["K"]; cand = BD.cand_mask(D, {})
pools = [np.where(cand[k])[0] for k in range(K)]
ew = {}
for c in (0.0, 0.003):
    o, _, _ = L.simulate(X, ADJf, list(sig), pools, c, capital=1e9); ew[c] = o.ret.values
EW = pd.DataFrame(dict(k=np.arange(K), signal_idx=sig, signal_date=[str(X.cal[i].date()) for i in sig], ew_ret_c0=ew[0.0], ew_ret_c30=ew[0.003])); EW.to_csv(Q.RES / "b_ew_benchmark.csv", index=False)
p0 = pd.read_csv(Q.V2 / "p0" / "results" / "fixed_sina_adj" / "period_returns.csv")
rows = [dict(item="EW c0 vs P0 file", maxabs=float(np.abs(ew[0.0] - p0.EqualWeight_ret_c0.values).max())), dict(item="EW c30 vs P0 file", maxabs=float(np.abs(ew[0.003] - p0.EqualWeight_ret_c30.values).max()))]
ref = {"HGB": p0.HGB_ret_c30.values, "LGBM": p0.LGBM_ret_c30.values, "Ridge": p0.Ridge_ret_c30.values}
for m, f in [("MLP", "mlp"), ("GRU", "gru"), ("TF", "tf")]: ref[m] = pd.read_csv(Q.P1D / "dl" / "results" / f / "period_returns.csv")[f"{m}_mean_ret_c30"].values
for m in Q.MODELS:
    o, lg, ex = S.simulate_ext(X, ADJf, list(sig), S.A0Decider(D["scores"][m], cand, sig), S.FlatCost(0.003), name="A0")
    tg = [S.rank_order(D["scores"][m][k], cand[k])[1][:50] for k in range(K)]; o0, _, _ = L.simulate(X, ADJf, list(sig), tg, 0.003, capital=1e7)
    rows.append(dict(item=f"A0 {m}: simulate_ext vs p0lib.simulate (same targets)", maxabs=float(np.abs(o.ret.values - o0.ret.values).max())))
    rows.append(dict(item=f"A0 {m}: simulate_ext vs frozen P0/track-B period_returns c30", maxabs=float(np.abs(o.ret.values - ref[m]).max())))
    hp = pd.read_parquet(Q.HP_ALL, columns=["signal_date", "code", "model", "in_top50"]); h = hp[(hp.model == m) & hp.in_top50]; got = set(zip(pd.to_datetime(h.signal_date), h.code)); mine = set((pd.Timestamp(X.cal[sig[k]]), X.codes[j]) for k in range(K) for j in tg[k])
    rows.append(dict(item=f"A0 {m}: Top50 set == frozen in_top50 flags (symmetric diff)", maxabs=float(len(got ^ mine))))
R = pd.DataFrame(rows); R.to_csv(Q.RES / "b_reconcile_A0.csv", index=False); print(R.to_string())