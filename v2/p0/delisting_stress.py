"""Delisting stress (E2-type, minimal): re-simulate the SAME selections (saved sets.pkl) with stuck/delisted positions written down by 0/50/100% (30bps RT cost). Models are not retrained."""
import sys, pickle, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import p0lib as L
S5 = ["Ridge", "HGB", "LGBM", "Momentum20", "Reversal5"]
X = L.build_data(); L.limit_flags(X); f, elig, ADJf = L.compute_features(X, X.ADJ_unadj_based)
st = pickle.load(open(L.P0 / "results/fixed_sina_adj/sets.pkl", "rb")); idx = {c: j for j, c in enumerate(X.codes)}
sig = st["signal_idx"]; pools = [np.array([idx[c] for c in p]) for p in st["pools"]]; tops = {n: [np.array([idx[c] for c in t]) for t in st["topk"][n]] for n in S5}
rows = []; sim = {}
for h in (0.0, 0.5, 1.0):
    for n in S5 + ["EqualWeight"]:
        o, _, ex = L.simulate(X, ADJf, sig, pools if n == "EqualWeight" else tops[n], 0.003, capital=1e9 if n == "EqualWeight" else 1e7, haircut=h); sim[(h, n)] = o.ret.values
    for n in S5 + ["EqualWeight"]:
        r = dict(stuck_writedown=h, strategy=n, terminal_unsold=None); r.update(L.perf(sim[(h, n)]))
        if n != "EqualWeight":
            ex_ = sim[(h, n)] - sim[(h, "EqualWeight")]; t, p = L.tstat(ex_); t4, p4 = L.nw_t(ex_, 4); lo, hi = L.block_boot(ex_); r.update(net_ex_mean=float(ex_.mean()), net_ex_t=t, net_ex_hac4_t=t4, boot_lo=lo, boot_hi=hi)
        rows.append(r)
d = pd.DataFrame(rows); d.to_csv(L.P0 / "delisting_stress.csv", index=False, encoding="utf-8-sig"); pd.set_option("display.width", 250); print(d.round(4).to_string())
