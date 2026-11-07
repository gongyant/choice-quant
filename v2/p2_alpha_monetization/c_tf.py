# -*- coding: utf-8 -*-
"""P2-C trainer: a controlled-variant copy of the P1 Transformer recipe. P1 modules (dl_common, dl_seq) are imported READ-ONLY; nothing under p1_dl/ or p0/ is modified.
Factors that can be switched, one at a time:  geometry (P1 purge vs. 20-day-label purge), heads (5d vs 1/5/10/20d), loss (MSE vs. cross-sectional pairwise rank loss), persistence penalty.
Everything else (7-channel 60-day input, mask, normalisation, TF architecture, AdamW/lr/wd/batch/clip/epochs/patience, seed handling) is identical to P1."""
import sys, time, copy
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
P2 = Path(__file__).resolve().parent; V2 = P2.parent
for p in (V2 / "p1_dl" / "dl", V2 / "p0", V2): sys.path.insert(0, str(p))
import dl_common as C, dl_seq as Q1
L = C.L; SEQ_LEN = Q1.SEQ_LEN
TF_CFG = dict(arch="tf", d_model=64, nhead=4, ff=128, layers=2, hidden=64, dropout=0.2, lr=1e-3, weight_decay=1e-4, batch=2048, max_epochs=12, patience=3, clip=1.0, threads=4, subsample=1)
VARIANTS = {
    "T0":  dict(heads=[5], loss="mse", geom="p1", persist=0.0, desc="P1 baseline recipe (seed 42 = frozen P1 scores)"),
    "T0G": dict(heads=[5], loss="mse", geom="g20", persist=0.0, desc="baseline with the 20-day-label purge geometry (control for the geometry change)"),
    "T1":  dict(heads=[5], loss="rank", geom="p1", persist=0.0, desc="5d head, pairwise rank loss instead of MSE (only the loss changes; day-grouped batches are implied)"),
    "T2":  dict(heads=[1, 5, 10, 20], loss="mse", geom="g20", persist=0.0, desc="multi-horizon heads, MSE per head, 20d purge"),
    "T3":  dict(heads=[1, 5, 10, 20], loss="rank", geom="g20", persist=0.0, desc="multi-horizon heads, pairwise rank loss per head, 20d purge"),
    "T4":  dict(heads=[5], loss="mse", geom="p1", persist=0.5, desc="baseline + score-persistence penalty (lag 5 trading days)"),
}
EXTRA = {"p1": 0, "g20": 15}      # 20 - 5 extra days at BOTH the val-end boundary and the train/val gap (see report derivation)
GAP_P1, HI_P1, VAL_DAYS = 10, 12, 60
def geometry(cal, i, geom, years=3):
    ex = EXTRA[geom]; lo = int(cal.searchsorted(cal[i] - pd.Timedelta(days=365 * years))); hi = i - HI_P1 - ex; vs = hi - VAL_DAYS + 1; te = vs - GAP_P1 - ex
    return lo, te, vs, hi
def label_stack(W, heads):
    out = []
    for h in heads:
        if h == L.H: out.append(W.yw)
        else: out.append(L.make_labels(W.ADJf, W.elig, h=h)[2])
    return np.stack(out, axis=2).astype(np.float64)
def collect(m_tr, Y, a, b):
    D, J, Ys = [], [], []
    for d in range(max(a, 0), b + 1):
        j = np.nonzero(m_tr[d])[0]
        if len(j): D.append(np.full(len(j), d)); J.append(j); Ys.append(Y[d][j])
    return (np.concatenate(D), np.concatenate(J), np.concatenate(Ys)) if D else (np.zeros(0, int), np.zeros(0, int), np.zeros((0, Y.shape[2])))
def split(W, Y, heads, i, geom):
    lo, te, vs, hi = geometry(W.cal, i, geom); m_tr = W.m_tr
    tr, va = collect(m_tr, Y, lo, te), collect(m_tr, Y, vs, hi)
    if len(tr[0]):
        for h in heads:
            assert int(tr[0].max()) + 1 + h < vs, ("train label expiry reaches validation", h)            # label of row d ends at close of d+1+h
            assert int(va[0].max()) + 1 + h <= i - 6, ("validation label not matured with margin 6", h)  # P1 margin: label end <= i-6
    return dict(tr=tr, va=va, lo=lo, te=te, vs=vs, hi=hi, expiry_train={h: int(tr[0].max()) + 1 + h for h in heads}, expiry_val={h: int(va[0].max()) + 1 + h for h in heads})
class MHNet(Q1.TFNet):
    """P1 TFNet; for >1 horizon the head is Linear(d, n_heads). With 1 head the module (and RNG consumption at init) is IDENTICAL to P1."""
    def __init__(self, d_in, cfg, nh=1):
        super().__init__(d_in, cfg)
        if nh > 1: self.head = nn.Linear(cfg.get("d_model", 64), nh)
    def forward(self, x):
        pad = x[:, :, -1] > 0.5; h = self.inp(x) + self.pe[:x.shape[1]]; h = self.enc(h, src_key_padding_mask=pad); return self.head(self.drop(h[:, -1]))     # [B, nh]
def _windows(S, d, j, ar): return S[(d[:, None] + ar[None, :]), j[:, None]]
def _pct_rank(d, Y):
    R = np.full(Y.shape, np.nan)
    for h in range(Y.shape[1]):
        R[:, h] = pd.Series(Y[:, h]).groupby(d).rank(pct=True, method="average").values
    return R
def _rank_loss_day(s, r):
    ls = []
    for h in range(s.shape[1]):
        m = torch.isfinite(r[:, h])
        if int(m.sum()) < 10: continue
        sh, rh = s[m, h], r[m, h]; w = torch.clamp(rh[:, None] - rh[None, :], min=0); ds = sh[:, None] - sh[None, :]
        ls.append((Fn.softplus(-ds) * w).sum() / w.sum().clamp_min(1e-9))
    return torch.stack(ls).mean()
def _day_slices(d):
    u, st = np.unique(d, return_index=True); en = np.append(st[1:], len(d)); return list(zip(st.tolist(), en.tolist()))
def fit2(S, tr, va, seed, spec, cfg=None, device="cuda"):
    """tr/va = (d, j, Y[n,nh]) with Y the winsorised pool-demeaned labels. S: normalised tensor on device. Returns model dict."""
    cfg = {**TF_CFG, **(cfg or {})}; t0 = time.time(); torch.set_num_threads(cfg["threads"]); torch.manual_seed(seed); np.random.seed(seed)
    if device == "cuda": torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    nh = len(spec["heads"]); pi = spec["heads"].index(L.H); ar = torch.arange(-SEQ_LEN + 1, 1, device=device)
    dtr, jtr = [torch.as_tensor(a, device=device) for a in tr[:2]]; dva, jva = [torch.as_tensor(a, device=device) for a in va[:2]]; assert int(dtr.min()) >= SEQ_LEN - 1 and int(dva.min()) >= SEQ_LEN - 1
    ysc = np.nanstd(tr[2], axis=0); ysc = np.where(ysc > 0, ysc, 1.0); ytr = torch.as_tensor(tr[2] / ysc, device=device).float(); yva = torch.as_tensor(va[2] / ysc, device=device).float()
    net = MHNet(S.shape[2], cfg, nh).to(device); opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]); mse = nn.MSELoss()
    g = torch.Generator().manual_seed(seed); best, best_ep, best_state, bad, hist = np.inf, -1, None, 0, []; n_all = len(dtr); rank = spec["loss"] == "rank"; lam = spec.get("persist", 0.0)
    if rank:
        rtr = torch.as_tensor(_pct_rank(tr[0], tr[2]), device=device).float(); rva = torch.as_tensor(_pct_rank(va[0], va[2]), device=device).float(); sl_tr, sl_va = _day_slices(tr[0]), _day_slices(va[0]); ND = 6
    def mse_heads(p, y):
        if nh == 1: return mse(p[:, 0], y[:, 0])                              # exactly P1's loss
        m = torch.isfinite(y); e = torch.where(m, (p - y) ** 2, torch.zeros_like(p)); return (e.sum(0) / m.sum(0).clamp_min(1)).mean()
    for ep in range(cfg["max_epochs"]):
        net.train(); tl, nb = 0.0, 0
        if not rank:
            perm = torch.arange(n_all, device=device)[torch.randperm(n_all, generator=g).to(device)]
            for a in range(0, len(perm), cfg["batch"]):
                idx = perm[a:a + cfg["batch"]]; opt.zero_grad(); p = net(_windows(S, dtr[idx], jtr[idx], ar)); loss = mse_heads(p, ytr[idx])
                if lam > 0:                                                    # persistence: same stock, 5 trading days earlier (the previous signal date) - relative squared change of the primary score
                    dl = dtr[idx] - 5; ok = dl >= SEQ_LEN - 1
                    if bool(ok.any()):
                        ok = ok & (S[dl.clamp_min(0), jtr[idx], -1] < 0.5); k = torch.nonzero(ok).squeeze(1)
                        if len(k) > 8:
                            pl = net(_windows(S, dl[k], jtr[idx][k], ar)); pen = ((p[k, pi] - pl[:, pi]) ** 2).mean() / (p[k, pi].var() + 1e-8); loss = loss + lam * pen      # scale-invariant (variance NOT detached, so shrinking the score scale cannot lower the penalty)
                loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), cfg["clip"]); opt.step(); tl += float(loss) * len(idx); nb += len(idx)
        else:
            dperm = torch.randperm(len(sl_tr), generator=g).tolist()
            for a in range(0, len(dperm), ND):
                grp = [sl_tr[q] for q in dperm[a:a + ND]]; idx = torch.cat([torch.arange(s0, s1, device=device) for s0, s1 in grp]); opt.zero_grad(); p = net(_windows(S, dtr[idx], jtr[idx], ar)); off = 0; ls = []
                for s0, s1 in grp: n = s1 - s0; ls.append(_rank_loss_day(p[off:off + n], rtr[s0:s1])); off += n
                loss = torch.stack(ls).mean(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), cfg["clip"]); opt.step(); tl += float(loss) * len(idx); nb += len(idx)
        net.eval(); vs_, vn = 0.0, 0
        with torch.no_grad():
            if not rank:
                for a in range(0, len(dva), 8192):
                    sl = slice(a, a + 8192); p = net(_windows(S, dva[sl], jva[sl], ar)); n = p.shape[0]
                    if nh == 1: vs_ += float(((p[:, 0] - yva[sl, 0]) ** 2).sum()); vn += n
                    else: vs_ += float(mse_heads(p, yva[sl])) * n; vn += n
            else:
                for s0, s1 in sl_va: p = net(_windows(S, dva[s0:s1], jva[s0:s1], ar)); vs_ += float(_rank_loss_day(p, rva[s0:s1])); vn += 1
        vl = vs_ / vn; hist.append(dict(epoch=ep + 1, train_loss=tl / nb, val_loss=vl, seconds=time.time() - t0))
        if vl < best - 1e-6: best, best_ep, best_state, bad = vl, ep + 1, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]: break
    return dict(state={k: v.cpu() for k, v in best_state.items()}, ysc=ysc, cfg=cfg, spec=spec, d_in=int(S.shape[2]), nh=nh, best_epoch=best_ep, best_val=best, history=hist, seconds=time.time() - t0, n_train=int(n_all), n_val=int(len(dva)), seed=int(seed))
def predict2(M, S, d, j, device="cuda"):
    net = MHNet(M["d_in"], M["cfg"], M["nh"]); net.load_state_dict(M["state"]); net = net.to(device).eval(); ar = torch.arange(-SEQ_LEN + 1, 1, device=device)
    jt = torch.as_tensor(np.asarray(j), device=device); dt = torch.full((len(jt),), int(d), device=device, dtype=torch.long)
    with torch.no_grad(): return net(_windows(S, dt, jt, ar)).cpu().numpy().astype(np.float64)       # [n, nh]