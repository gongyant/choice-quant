# -*- coding: utf-8 -*-
"""Decision-time style exposure of A0/A1/A2 (weighted pool-standardised z of log_amt20, vol20, r5 at the signal close) from the saved decision logs."""
import numpy as np, pandas as pd
import p2lib as Q, b_data as BD, b_sim as S
D = BD.build(); X, sig, W = D["X"], D["sig"], D["W"]; cidx = {c: i for i, c in enumerate(X.codes)}; rows = []
for model in ("HGB", "LGBM", "GRU"):
    for scen in ("C1_flat", "C2_explicit"):
        cand = BD.cand_mask(D, {}) if scen in ("C1_flat", "C2_explicit") else None; sc = D["scores"][model]
        w_by = {"A0": None}
        for fam in ("A1", "A2"):
            dd = pd.read_csv(Q.RES / "b_logs" / f"decisions_{model}_{scen}_{fam}_nested.csv.gz", dtype={"code": str}); w_by[fam + "_nested"] = dd
        for v, dd in w_by.items():
            acc = {c: [] for c in ("log_amt20", "vol20", "r5")}; nn = []
            for k in range(52, D["K"]):
                s = int(sig[k]); pool = np.where(cand[k])[0]
                if v == "A0": _, o = S.rank_order(sc[k], cand[k]); j = o[:50]; w = np.full(len(j), 1 / len(j))
                else:
                    g = dd[(dd.k == k) & (dd.target_weight > 0)]; j = np.array([cidx[c] for c in g.code]); w = g.target_weight.values; w = w / w.sum()
                nn.append(len(j))
                for c in acc:
                    f = W.f[c].values[s]; fp = f[pool]; mu, sd = np.nanmean(fp), np.nanstd(fp); z = (f[j] - mu) / sd if sd > 0 else np.zeros(len(j)); acc[c].append(float(np.nansum(w * z)))
            rows.append(dict(model=model, scenario=scen, variant=v, n_names=float(np.mean(nn)), **{f"z_{c}": float(np.mean(a)) for c, a in acc.items()}))
pd.DataFrame(rows).to_csv(Q.RES / "b_exposures.csv", index=False, encoding="utf-8-sig"); print(pd.DataFrame(rows).round(3).to_string())