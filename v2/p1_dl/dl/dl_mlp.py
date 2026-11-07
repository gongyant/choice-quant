"""P1-B1: small PyTorch MLP for cross-sectional 5-day relative-return ranking.

PRE-REGISTERED CHOICES (written to worklog_dl.md BEFORE the full run; not tuned after seeing test results):
  input   : the 11 P0 cross-sectional rank features (rank_features: per-day pct rank - 0.5, NaN->0), then z-scored with mean/std fitted on the TRAINING part only
  target  : P0's yw (5-day label t+1 close -> t+6 close, de-meaned over the SAME eligible_asof_t pool, winsorised 1/99%) divided by the training-part std. Loss = MSE.
  network : 11 -> 64 -> 32 -> 1, GELU, dropout 0.2
  optim   : AdamW lr 1e-3, weight decay 1e-4, batch 4096, max 30 epochs, early stopping patience 3 on validation MSE, best-epoch weights restored
  seeds   : 42, 43, 44 (model init + shuffling); the ensemble score = arithmetic mean of the raw per-seed scores (all seeds share the same target scaling)
  device  : CPU (deterministic; model is tiny). CUDA availability is only recorded.
The score is a model-native ranking score. It is NOT a probability and NOT a return forecast in percent.
"""
import time, copy
import numpy as np
import torch, torch.nn as nn
CFG = dict(hidden=(64, 32), dropout=0.2, lr=1e-3, weight_decay=1e-4, batch=4096, max_epochs=30, patience=3, threads=4)
class MLP(nn.Module):
    def __init__(self, d_in, hidden=(64, 32), dropout=0.2):
        super().__init__(); layers, d = [], d_in
        for h in hidden: layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]; d = h
        layers += [nn.Linear(d, 1)]; self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)
def fit_scaler(Xtr):
    """Statistics from the training part ONLY."""
    mu = Xtr.mean(0); sd = Xtr.std(0); sd = np.where(sd > 1e-8, sd, 1.0); return mu.astype(np.float32), sd.astype(np.float32)
def fit_mlp(Xtr, ytr, Xva, yva, seed, cfg=None, device="cpu", verbose=False):
    """Train on (Xtr,ytr); early-stop on (Xva,yva) which must be LATER than Xtr (caller guarantees purge). Returns a self-contained state dict."""
    cfg = {**CFG, **(cfg or {})}; t0 = time.time(); torch.set_num_threads(cfg["threads"]); torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = fit_scaler(Xtr); ysc = float(ytr.std()) or 1.0
    f = lambda A: torch.tensor((A - mu) / sd, dtype=torch.float32, device=device)
    xt, xv = f(Xtr), f(Xva); yt = torch.tensor(ytr / ysc, dtype=torch.float32, device=device); yv = torch.tensor(yva / ysc, dtype=torch.float32, device=device)
    net = MLP(Xtr.shape[1], cfg["hidden"], cfg["dropout"]).to(device); opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    g = torch.Generator().manual_seed(seed); lossf = nn.MSELoss(); best, best_ep, best_state, bad, hist = np.inf, -1, None, 0, []
    n = len(xt)
    for ep in range(cfg["max_epochs"]):
        net.train(); perm = torch.randperm(n, generator=g).to(device); tl, nb = 0.0, 0
        for a in range(0, n, cfg["batch"]):
            idx = perm[a:a + cfg["batch"]]; opt.zero_grad(); loss = lossf(net(xt[idx]), yt[idx]); loss.backward(); opt.step(); tl += float(loss) * len(idx); nb += len(idx)
        net.eval()
        with torch.no_grad(): vl = float(lossf(net(xv), yv))
        hist.append(dict(epoch=ep + 1, train_loss=tl / nb, val_loss=vl))
        if verbose: print(ep + 1, tl / nb, vl)
        if vl < best - 1e-6: best, best_ep, best_state, bad = vl, ep + 1, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]: break
    return dict(state={k: v.cpu() for k, v in best_state.items()}, mu=mu, sd=sd, ysc=ysc, cfg=cfg, d_in=int(Xtr.shape[1]), best_epoch=best_ep, best_val=best, history=hist,
                seconds=time.time() - t0, n_train=int(len(Xtr)), n_val=int(len(Xva)), seed=int(seed))
def predict(M, X):
    net = MLP(M["d_in"], M["cfg"]["hidden"], M["cfg"]["dropout"]); net.load_state_dict(M["state"]); net.eval()
    with torch.no_grad(): return net(torch.tensor((X - M["mu"]) / M["sd"], dtype=torch.float32)).numpy().astype(np.float64)