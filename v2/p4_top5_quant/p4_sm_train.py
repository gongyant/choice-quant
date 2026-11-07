# -*- coding: utf-8 -*-
"""P4 batch 3: compact StockMixer (indicator mixing + temporal mixing + stock mixing) on the E2 (or E1 price/volume-only) information set.
python p4_sm_train.py <E2|E1> <5|20>  (run in .venv_cuda; ONE GPU job at a time)
Same schedule as LGBM: 207 signal dates, retrain every 4 signals, rolling 3y window of days with FULLY matured labels (d <= i-h-2), every day used.
Config fixed: D=64, m=20 market nodes, lr 1e-3, 8 epochs, loss = MSE + pairwise ranking; only the lookback window W in {10, 20} is chosen ONCE on the
inner validation split of the first window (last 120 days, h+2 gap), metric = mean top-10 excess. Final model at 2026-10-21 also fitted."""
import sys, json, time, pickle
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
FEAT, h = sys.argv[1], int(sys.argv[2]); P4 = Path(__file__).resolve().parent; C = P4 / "cache"; OUT = P4 / "preds"; t0 = time.time()
torch.manual_seed(42); np.random.seed(42); dev = "cuda"
fl = np.load(C / "flags.npz"); cal = pd.DatetimeIndex(fl["cal"]); codes = fl["codes"]; elig = fl["elig"]; T, N = elig.shape
lab = pickle.load(open(C / "labels.pkl", "rb"))["lab"][h]; Xall = np.load(C / "Xq_E2.npy", mmap_mode="r")
nf = 50 if FEAT == "E1" else Xall.shape[2]; Xg = torch.tensor(np.asarray(Xall[:, :, :nf]), device=dev)        # T x N x F (ranked, -0.5..0.5)
Y = torch.tensor(np.nan_to_num(lab["exw"]), device=dev); EX = lab["ex"]; Mk = torch.tensor(elig & np.isfinite(lab["exw"]), device=dev); EL = torch.tensor(elig, device=dev)
sig = sorted(pd.read_parquet(P4.parent / "p1_dl" / "historical_predictions_all_models.parquet", columns=["signal_date"]).signal_date.unique()); sig_idx = [int(cal.get_loc(pd.Timestamp(s))) for s in sig]
def train_days(i, years=3):
    lo = int(cal.searchsorted(cal[i] - pd.Timedelta(days=365 * years))); return list(range(max(lo, 30), i - h - 2 + 1))
class SM(nn.Module):
    def __init__(s, W, F, D=64, m=20):
        super().__init__(); s.ln1 = nn.LayerNorm(F); s.ind = nn.Sequential(nn.Linear(F, 2 * F), nn.GELU(), nn.Linear(2 * F, F))
        s.ln2 = nn.LayerNorm(F); s.tim = nn.Sequential(nn.Linear(W, W), nn.GELU(), nn.Linear(W, W)); s.emb = nn.Linear(W * F, D)
        s.A1 = nn.Linear(N, m, bias=False); s.A2 = nn.Linear(m, N, bias=False); s.out = nn.Linear(2 * D, 1)
    def forward(s, x, mask):                                    # x: N x W x F
        x = x + s.ind(s.ln1(x)); x = x + s.tim(s.ln2(x).transpose(1, 2)).transpose(1, 2)
        e = torch.nn.functional.gelu(s.emb(x.flatten(1))) * mask[:, None]          # N x D
        z = s.A2(torch.nn.functional.gelu(s.A1(e.T))).T                             # stock mixing via m market nodes
        return s.out(torch.cat([e, z], 1)).squeeze(1)
def win(d, W): return Xg[d - W + 1:d + 1].permute(1, 0, 2)
def loss_fn(p, y, m):
    p, y = p[m], y[m]; mse = ((p - y) ** 2).mean(); dp = p[:, None] - p[None, :]; dy = y[:, None] - y[None, :]
    return mse + torch.relu(-dp * dy).mean()
def fit(days, W, epochs=8):
    torch.manual_seed(42); net = SM(W, nf).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
    for ep in range(epochs):
        for d in np.random.permutation(days):
            m = Mk[d]
            if m.sum() < 50: continue
            opt.zero_grad(); l = loss_fn(net(win(d, W), EL[d].float()), Y[d], m); l.backward(); opt.step()
    return net.eval()
@torch.no_grad()
def score(net, d, W): return net(win(d, W), EL[d].float()).cpu().numpy()
def top_ex(net, days, W):
    v = []
    for d in days:
        m = np.where(elig[d] & np.isfinite(EX[d]))[0]
        if len(m) < 50: continue
        s = score(net, d, W)[m]; v.append(EX[d][m][np.argsort(-s)[:10]].mean())
    return float(np.mean(v))
i0 = sig_idx[0]; d0 = train_days(i0); hi = d0[-1]; val = [d for d in d0 if d > hi - 120]; inner = [d for d in d0 if d <= hi - 120 - h - 2]
grid = []
for W in [10, 20]:
    inn = [d for d in inner if d >= W]; grid.append(dict(W=W, val_top10_excess=top_ex(fit(inn, W), val, W)))
W = max(grid, key=lambda g: g["val_top10_excess"])["W"]; print(FEAT, h, grid, "W", W, round(time.time() - t0), "s", flush=True)
parts, meta = [], []
for k, i in enumerate(sig_idx):
    if k % 4 == 0:
        days = [d for d in train_days(i) if d >= W]; assert days[-1] + 1 + h < i; net = fit(days, W)
        meta.append(dict(retrain_signal_date=str(cal[i].date()), train_first=str(cal[days[0]].date()), train_last=str(cal[days[-1]].date()), label_end_max=str(cal[days[-1] + 1 + h].date()), n_days=len(days)))
        if k % 20 == 0: print("retrain", k, cal[i].date(), round(time.time() - t0), "s", flush=True)
    m = np.where(elig[i])[0]; s = score(net, i, W)[m]
    parts.append(pd.DataFrame(dict(signal_date=str(cal[i].date()), code=codes[m], score=s, model=f"SM{FEAT}", horizon=h, retrain_signal_date=meta[-1]["retrain_signal_date"], label_end_max=meta[-1]["label_end_max"])))
pd.concat(parts).to_parquet(OUT / f"SM{FEAT}_h{h}.parquet", index=False)
iL = T - 1; days = [d for d in train_days(iL) if d >= W]; net = fit(days, W); m = np.where(elig[iL])[0]
pd.DataFrame(dict(signal_date=str(cal[iL].date()), code=codes[m], score=score(net, iL, W)[m], model=f"SM{FEAT}", horizon=h, label_end_max=str(cal[days[-1] + 1 + h].date()))).to_parquet(OUT / f"SM{FEAT}_h{h}_latest.parquet", index=False)
torch.save(net.state_dict(), OUT / f"SM{FEAT}_h{h}_latest.pt")
json.dump(dict(model=f"SM{FEAT}", horizon=h, n_feat=nf, grid=grid, W=W, fixed=dict(D=64, m=20, lr=1e-3, epochs=8, loss="MSE+pairwise"), retrains=meta,
               final_train=dict(first=str(cal[days[0]].date()), last=str(cal[days[-1]].date())), secs=round(time.time() - t0)), open(OUT / f"SM{FEAT}_h{h}_meta.json", "w"), indent=1)
print("done", FEAT, h, round(time.time() - t0), "s", flush=True)
