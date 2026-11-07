import time, numpy as np, pandas as pd, sys
sys.path.insert(0,'.')
import p0lib as L
t=time.time(); X=L.build_data(); print('built',X.C.shape,round(time.time()-t,1)); print(X.meta.source.value_counts().to_dict(), X.meta.unit_unverified.sum())
print(X.meta[['code','amount_ratio_median','unit_mult_inferred','unit_mult_used','factor_status','n_factor_changes_in_window']].head(12).to_string())
up,dn=L.limit_flags(X); print('limit up/down bars',up.sum(),dn.sum(),'traded bars',X.traded.sum(), 'nolimit ipo',X.nolimit_ipo.sum())
f,elig,ADJf=L.compute_features(X,X.ADJ_unadj_based); print('elig per day mean',elig.sum(1).mean(), 'time',round(time.time()-t,1))
lab,y,yw=L.make_labels(ADJf,elig); print('labels', np.isfinite(y).sum())
# simulate smoke test: pick random targets
T=len(X.cal); sig=list(range(600,T-7,5)); rng=np.random.default_rng(0)
tg=[rng.choice(np.where(elig[i])[0],size=min(10,elig[i].sum()),replace=False) for i in sig]
out,lg,ex=L.simulate(X,ADJf,sig,tg,0.003,capital=1e7,log_name='rand')
print(out.head(3).T); print(out.ret.describe()); print(lg.status.value_counts().to_dict(), lg.reason.value_counts().to_dict(), ex)
print(L.perf(out.ret))
