# -*- coding: utf-8 -*-
"""P2-B data world: P0 matrices + frozen scores (via the P2-A panel, which was verified equal to the frozen parquet) + OHLCV cost inputs."""
import numpy as np, pandas as pd
import p2lib as Q, b_sim as S
import p0lib as L
def ffill(M): return pd.DataFrame(M).ffill().values
def build():
    X, W = Q.build_world(); cal = X.cal; N = len(X.codes)
    P = pd.read_parquet(Q.RES / "p2a_panel.parquet", columns=["i", "code", "signal_date", "eligible"] + ["score_" + m for m in Q.MODELS])
    sig = np.array(sorted(P.i.unique())); K = len(sig); assert (P.groupby("i").size() == N).all()
    P = P.sort_values(["i"], kind="stable"); assert (P.code.values.reshape(K, N)[0] == np.array(X.codes)).all()
    scores = {m: P["score_" + m].values.reshape(K, N) for m in Q.MODELS}; cand0 = P.eligible.values.reshape(K, N)
    assert (cand0 == W.elig[sig]).all()
    HS = S.corwin_schultz_half_spread(X.Hh, X.L, X.C)
    VOL = np.where(np.isfinite(ffill(W.f["vol20"].values)), ffill(W.f["vol20"].values), 0.03); ADV = ffill(W.f["amt20"].values); ADV = np.where(np.isfinite(ADV), ADV, 2e7)
    HS = np.where(np.isfinite(HS), HS, 0.0005)
    board = X.u.board.values; star = board == "科创板"; bj = board == "北交所"
    return dict(X=X, W=W, cal=cal, sig=sig, K=K, N=N, scores=scores, cand0=cand0, HS=HS, VOL=VOL, ADV=ADV, star=star, bj=bj, ADJf=W.ADJf, y=W.y)
def cand_mask(D, scen):
    """Candidate pool known at s: eligible_asof_t (+ optional realistic constraints known at s)."""
    c = D["cand0"].copy()
    if scen.get("no_star"): c &= ~D["star"][None, :]
    if scen.get("no_bj"): c &= ~D["bj"][None, :]
    if scen.get("min_adv"): c &= (D["W"].f["amt20"].values[D["sig"]] >= scen["min_adv"])
    return c