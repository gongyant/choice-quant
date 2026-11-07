# -*- coding: utf-8 -*-
"""P4-E0/E1 step 1: build the point-in-time panel cache (features ranked per day over the eligible pool, labels, flags). Reads P0/P3 data only."""
import json, time, pickle
import numpy as np, pandas as pd
import p4lib as P
t0 = time.time(); C = P.P4 / "cache"; C.mkdir(exist_ok=True)
X = P.load_world(); print("world", X.C.shape, X.cal[0].date(), X.cal[-1].date(), round(time.time() - t0), "s", flush=True)
f, elig, ADJf, amt20 = P.features(X); print("features", len(f), round(time.time() - t0), "s", flush=True)
E1 = [k for k in f if k != "amt20"]; E0 = list(P.FE0)
Xq = P.rank_feats(f, elig, E1); np.save(C / "Xq_E1.npy", Xq)
olu, old = P.open_flags(X); st = P.st_proxy(X)
lab = {}
for h in [5, 20]:
    l, ex, exw, gr = P.labels(X, elig, h); lab[h] = dict(ex=ex.astype(np.float32), exw=exw.astype(np.float32), gr=gr.astype(np.float32), raw=l.astype(np.float32))
pickle.dump(dict(lab=lab), open(C / "labels.pkl", "wb"), protocol=4)
np.savez_compressed(C / "flags.npz", elig=elig, amt20=amt20, olu=olu, old=old, st=st, lim_up=X.lim_up, lim_dn=X.lim_dn, traded=X.traded, cal=np.array([str(d.date()) for d in X.cal]), codes=np.array(X.codes))
pickle.dump(dict(E0=E0, E1=E1), open(C / "featnames.pkl", "wb"))
pickle.dump(X, open(C / "world.pkl", "wb"), protocol=4)
u = X.u; cur_st = (u.is_st_name.fillna(False) | u.name_has_delist_mark.fillna(False)).values; i = len(X.cal) - 1; m = elig[i]
info = dict(T=len(X.cal), N=len(X.codes), first=str(X.cal[0].date()), last=str(X.cal[-1].date()), n_feat_E0=len(E0), n_feat_E1=len(E1), pool_last=int(m.sum()),
            st_proxy_vs_current_name_last_day=dict(current_st_in_pool=int((cur_st & m).sum()), proxy_flag_in_pool=int((st[i] & m).sum()), both=int((st[i] & cur_st & m).sum())),
            st_proxy_share_hist=float(st[elig].mean()), open_limit_up_share=float(olu[X.traded].mean()), secs=round(time.time() - t0))
json.dump(info, open(C / "build_info.json", "w"), indent=1); print(info)
