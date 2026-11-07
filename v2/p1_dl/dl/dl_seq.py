"""P1-B2: 60-day sequence models (GRU now; Transformer later) for cross-sectional 5-day relative-return ranking.

PRE-REGISTERED CHOICES (written to worklog_dl.md BEFORE the full run; no post-hoc tuning):
  sequence : last SEQ_LEN=60 rows of the MASTER trading calendar ending at signal day t (suspended / not-listed / missing days are KEPT as rows, value 0 + mask=1; the time axis is never compressed)
  channels : 6 value channels + 1 mask channel
             r1        adjusted 1-day return (raw close / Sina factor ratio; P0 feature r1)
             range     (high - low) / ex-rights reference previous close (p0lib X.ref)
             oc        close / open - 1 (raw prices, same day)
             logvr     log( volume_t / mean(volume_{t-20..t-1}) ), min 10 obs   (trailing, as-of)
             logamt_cs cross-sectional z-score (same day, over stocks with a bar) of log1p(turnover amount)
             logturn   log1p(turnover rate); missing (e.g. Tencent-fallback stocks) -> 0 (no extra flag; the trade mask is the only mask channel)
             mask      1 if no valid bar that day (suspended / not listed / no data), else 0
  scaling  : per-channel mean/std fitted ONLY on cells dated [lo-59 .. te] of each retrain's TRAINING segment (te = last training date), then clip to [-5, 5]; NaN -> 0; masked cells -> 0.
  model    : GRU(7 -> 64, 1 layer) -> last hidden state -> dropout 0.2 -> Linear(64,1)
  target/loss/early stopping/validation/purge/seeds identical to B1 (yw / training std, MSE; validation = last 60 days of the 3y window; 10 day gap; patience 3; best epoch restored)
  optim    : AdamW lr 1e-3, wd 1e-4, batch 2048, grad-clip 1.0, max 12 epochs
The score is a model-native ranking score, NOT a probability and NOT a % return.
"""
import time, copy
import numpy as np, pandas as pd
import torch, torch.nn as nn
SEQ_LEN = 60; CH = ["r1", "range", "oc", "logvr", "logamt_cs", "logturn"]
CFG = dict(arch="gru", hidden=64, layers=1, dropout=0.2, lr=1e-3, weight_decay=1e-4, batch=2048, max_epochs=12, patience=3, clip=1.0, threads=8, subsample=1)
def build_seq_features(W):
    """Raw per-cell channels [T,N,6] (NaN where undefined) and mask [T,N] (True = no valid bar). Uses only data dated <= the cell's own day."""
    X = W.X; tr = X.traded; T, N = tr.shape
    with np.errstate(all="ignore"):
        r1 = W.f["r1"].values.astype(float); rng_ = (X.Hh - X.L) / X.ref; oc = X.C / X.O - 1
        V = np.where(X.V > 0, X.V, np.nan); Vm = pd.DataFrame(V).shift(1).rolling(20, min_periods=10).mean().values; logvr = np.log(V / Vm)
        la = np.where(tr, np.log1p(np.where(X.AMT > 0, X.AMT, np.nan)), np.nan); cnt = np.isfinite(la).sum(1, keepdims=True)
        m = np.nanmean(np.where(cnt >= 5, la, np.nan), axis=1, keepdims=True); s = np.nanstd(np.where(cnt >= 5, la, np.nan), axis=1, keepdims=True); logamt = (la - m) / np.where(s > 0, s, np.nan)
        lt = np.log1p(np.where(tr & (X.TURN >= 0), X.TURN, np.nan))
    F = np.stack([r1, rng_, oc, logvr, logamt, lt], axis=2); F[~np.isfinite(F)] = np.nan; F[~tr] = np.nan
    return F.astype(np.float32), (~tr)
def fit_norm(F, lo, te):
    """Channel statistics from the training segment only: cells dated [lo-SEQ_LEN+1 .. te]."""
    a = max(0, lo - SEQ_LEN + 1); seg = F[a:te + 1].reshape(-1, F.shape[2])
    mu = np.nanmean(seg, axis=0); sd = np.nanstd(seg, axis=0); sd = np.where(sd > 1e-8, sd, 1.0); return mu.astype(np.float32), sd.astype(np.float32)
def apply_norm(F, mask, mu, sd):
    Z = np.clip((F - mu) / sd, -5, 5); Z = np.nan_to_num(Z, nan=0.0); Z[mask] = 0.0
    return np.concatenate([Z, mask[..., None].astype(np.float32)], axis=2).astype(np.float32)
def split_idx(m_tr, yw, cal, i, years=3, val_days=60, gap=10, H=5):
    """Same geometry as dl_common.split_train_val, but returns (date idx, stock idx, label) instead of feature rows."""
    lo = int(cal.searchsorted(cal[i] - pd.Timedelta(days=365 * years))); hi = i - 12; vs = hi - val_days + 1; te = vs - gap
    def col(a, b):
        D, J, Y = [], [], []
        for d in range(max(a, 0), b + 1):
            j = np.nonzero(m_tr[d])[0]
            if len(j): D.append(np.full(len(j), d)); J.append(j); Y.append(yw[d][j])
        return (np.concatenate(D), np.concatenate(J), np.concatenate(Y)) if D else (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    tr, va = col(lo, te), col(vs, hi)
    return dict(tr=tr, va=va, lo=lo, te=te, vs=vs, hi=hi, label_expiry_train_max=int(tr[0].max()) + H + 1 if len(tr[0]) else -1, label_expiry_val_max=int(va[0].max()) + H + 1 if len(va[0]) else -1)
class SeqNet(nn.Module):
    def __init__(self, d_in, cfg):
        super().__init__(); h = cfg["hidden"]; self.rnn = nn.GRU(d_in, h, num_layers=cfg["layers"], batch_first=True, dropout=cfg["dropout"] if cfg["layers"] > 1 else 0.0)
        self.drop = nn.Dropout(cfg["dropout"]); self.head = nn.Linear(h, 1)
    def forward(self, x):
        o, _ = self.rnn(x); return self.head(self.drop(o[:, -1])).squeeze(-1)
class TFNet(nn.Module):
    """B3: small Transformer encoder. Linear(7->d_model) + sinusoidal positional encoding, 2 pre-LN layers, 4 heads, FFN 128, dropout 0.2, src_key_padding_mask = mask channel
    (suspended / missing days are never attended to). Pooling (pre-registered): the representation at the LAST time step t, which is always a valid bar for eligible_asof_t rows."""
    def __init__(self, d_in, cfg):
        super().__init__(); d = cfg.get("d_model", 64); self.inp = nn.Linear(d_in, d)
        pos = torch.arange(SEQ_LEN).unsqueeze(1).float(); div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d)); pe = torch.zeros(SEQ_LEN, d); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d, cfg.get("nhead", 4), cfg.get("ff", 128), cfg["dropout"], activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg["layers"], norm=nn.LayerNorm(d), enable_nested_tensor=False); self.drop = nn.Dropout(cfg["dropout"]); self.head = nn.Linear(d, 1)
    def forward(self, x):
        pad = x[:, :, -1] > 0.5; h = self.inp(x) + self.pe[:x.shape[1]]; h = self.enc(h, src_key_padding_mask=pad)
        return self.head(self.drop(h[:, -1])).squeeze(-1)
def make_net(d_in, cfg):
    return TFNet(d_in, cfg) if cfg.get("arch", "gru") == "tf" else SeqNet(d_in, cfg)
def _windows(S, d, j, ar):
    """S [T,N,C] tensor, d/j index tensors [B] -> [B,SEQ_LEN,C] windows ending at day d (inclusive)."""
    return S[(d[:, None] + ar[None, :]), j[:, None]]
def fit_seq(S, tr, va, seed, cfg=None, device="cpu", verbose=False):
    """S: normalized tensor [T,N,7] ALREADY on `device`, built with training-segment statistics. tr/va = (d, j, y) with va strictly later than tr."""
    cfg = {**CFG, **(cfg or {})}; t0 = time.time(); torch.set_num_threads(cfg["threads"]); torch.manual_seed(seed); np.random.seed(seed)
    if device == "cuda": torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    ar = torch.arange(-SEQ_LEN + 1, 1, device=device); dtr, jtr, ytr = [torch.as_tensor(a, device=device) for a in tr]; dva, jva, yva = [torch.as_tensor(a, device=device) for a in va]
    assert int(dtr.min()) >= SEQ_LEN - 1 and int(dva.min()) >= SEQ_LEN - 1
    ysc = float(ytr.float().std()) or 1.0; ytr = (ytr / ysc).float(); yva = (yva / ysc).float()
    net = make_net(S.shape[2], cfg).to(device); opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]); lossf = nn.MSELoss()
    g = torch.Generator().manual_seed(seed); best, best_ep, best_state, bad, hist = np.inf, -1, None, 0, []; sub = cfg["subsample"]; n_all = len(dtr)
    for ep in range(cfg["max_epochs"]):
        if sub > 1: keep = torch.nonzero(((dtr + jtr) % sub) == (ep % sub)).squeeze(1)       # epoch-rotating uniform subsample (training only)
        else: keep = torch.arange(n_all, device=device)
        perm = keep[torch.randperm(len(keep), generator=g).to(device)]; net.train(); tl, nb = 0.0, 0
        for a in range(0, len(perm), cfg["batch"]):
            idx = perm[a:a + cfg["batch"]]; opt.zero_grad(); loss = lossf(net(_windows(S, dtr[idx], jtr[idx], ar)), ytr[idx]); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg["clip"]); opt.step(); tl += float(loss) * len(idx); nb += len(idx)
        net.eval(); vs_, vn = 0.0, 0
        with torch.no_grad():
            for a in range(0, len(dva), 8192):
                sl = slice(a, a + 8192); p = net(_windows(S, dva[sl], jva[sl], ar)); vs_ += float(((p - yva[sl]) ** 2).sum()); vn += len(p)
        vl = vs_ / vn; hist.append(dict(epoch=ep + 1, train_loss=tl / nb, val_loss=vl, seconds=time.time() - t0))
        if verbose: print(ep + 1, tl / nb, vl, round(time.time() - t0), flush=True)
        if vl < best - 1e-6: best, best_ep, best_state, bad = vl, ep + 1, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]: break
    return dict(state={k: v.cpu() for k, v in best_state.items()}, ysc=ysc, cfg=cfg, d_in=int(S.shape[2]), best_epoch=best_ep, best_val=best, history=hist, seconds=time.time() - t0,
                n_train=int(n_all), n_val=int(len(dva)), seed=int(seed), device=device)
def predict_seq(M, S, d, j, device="cpu"):
    """Scores for stocks j at day index d (scalar) using normalized tensor S (same statistics as the model's training)."""
    net = make_net(M["d_in"], M["cfg"]); net.load_state_dict(M["state"]); net = net.to(device).eval(); ar = torch.arange(-SEQ_LEN + 1, 1, device=device)
    jt = torch.as_tensor(np.asarray(j), device=device); dt = torch.full((len(jt),), int(d), device=device, dtype=torch.long)
    with torch.no_grad(): return net(_windows(S, dt, jt, ar)).cpu().numpy().astype(np.float64)