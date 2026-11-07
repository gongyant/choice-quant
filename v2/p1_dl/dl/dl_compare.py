"""P1-B1 comparison: MLP (3 seeds + ensemble) vs FROZEN P0 HGB / LightGBM / Ridge / equal weight, same periods, same simulator, same costs.
P0 numbers come from p0/results/fixed_sina_adj/period_returns.csv (read-only). Because P0 only simulated 0/0.10/0.30/0.60% round trip, the P0 HGB/LGBM Top50 sets (p0 sets.pkl) are
re-simulated here with the identical engine at 0.15%; the same re-simulation at 0.30% is checked against the P0 file to prove the harness is identical."""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dl_common as C
L = C.L; DL = C.DL; P1 = C.P1
TAG = sys.argv[1] if len(sys.argv) > 1 else "mlp"; OUT = DL / "results" / TAG
mlp = pd.read_csv(OUT / "period_returns.csv"); p0 = pd.read_csv(L.P0 / "results" / "fixed_sina_adj" / "period_returns.csv")
assert (mlp.signal_date.values == p0.signal_date.values[:len(mlp)]).all(); p0 = p0.iloc[:len(mlp)].reset_index(drop=True)
assert len(mlp) == len(p0) == 207, (len(mlp), len(p0))
# --- re-simulate P0 HGB/LGBM/Ridge/Mom/Rev Top50 at 0.15% with identical engine (+ 0.30% self-check) ---
W = C.build_world(); X = W.X; idx = {c: j for j, c in enumerate(X.codes)}
sets = pickle.load(open(L.P0 / "results" / "fixed_sina_adj" / "sets.pkl", "rb")); sig = sets["signal_idx"]
assert list(sig) == [int(W.cal.searchsorted(pd.Timestamp(d))) for d in mlp.signal_date]
resim = {}
for s in ["HGB", "LGBM", "Ridge", "Momentum20", "Reversal5"]:
    tg = [np.array([idx[c] for c in t]) for t in sets["topk"][s]]
    for c in [0.0015, 0.003]:
        o, _, _ = L.simulate(X, W.ADJf, sig, tg, c, capital=1e7); resim[(s, c)] = o.ret.values
    d = np.abs(resim[(s, 0.003)] - p0[f"{s}_ret_c30"].values).max(); print(f"harness self-check {s}@0.30%: max |re-sim - P0 file| = {d:.2e}")
    p0[f"{s}_ret_c15"] = resim[(s, 0.0015)]; p0[f"{s}_selfcheck_maxdiff_c30"] = d
pools = [np.array([idx[c] for c in t]) for t in sets["pools"]]
o, _, _ = L.simulate(X, W.ADJf, sig, pools, 0.0015, capital=1e9); p0["EqualWeight_ret_c15"] = o.ret.values
assert np.allclose(p0["EqualWeight_ret_c15"].values, mlp["EqualWeight_ret_c15"].values, atol=1e-12)
selfcheck = {s: float(p0[f"{s}_selfcheck_maxdiff_c30"].iloc[0]) for s in ["HGB", "LGBM", "Ridge", "Momentum20", "Reversal5"]}
def sb(ex):
    ex = pd.Series(ex).dropna().values; t, p = L.tstat(ex); t4, p4 = L.nw_t(ex, 4); lo, hi = L.block_boot(ex)
    return dict(excess_mean=float(ex.mean()), t=t, p=p, hac4_t=t4, hac4_p=p4, boot_lo=lo, boot_hi=hi, n=len(ex))
SRC = {}
for s in ["HGB", "LGBM", "Ridge", "Momentum20", "Reversal5"]: SRC[f"P0_{s}"] = (p0, s)
for s in ["MLP_s42", "MLP_s43", "MLP_s44", "MLP_mean"]: SRC[s] = (mlp, s)
ew = {c: mlp[f"EqualWeight_ret_{c}"] for c in ["c0", "c10", "c15", "c30", "c60"]}
def row(name, df, s, c="c30"):
    net, gro = df[f"{s}_ret_{c}"], df[f"{s}_ret_c0"]; r = dict(strategy=name, cost_roundtrip=int(c[1:]) / 10000)
    r.update({k: v for k, v in L.perf(net).items() if k != "periods"}); r["net_mean_per_period"] = float(net.mean()); r["gross_mean_per_period"] = float(gro.mean())
    if f"{s}_RankIC" in df: r.update(IC=float(df[f"{s}_IC"].mean()), RankIC=float(df[f"{s}_RankIC"].mean()), RankIC_IR=float(df[f"{s}_RankIC"].mean() / df[f"{s}_RankIC"].std()), RankIC_t=L.tstat(df[f"{s}_RankIC"].dropna())[0])
    if f"{s}_replaced" in df and c == "c30":
        r["turnover_replaced_frac"] = float(df[f"{s}_replaced"].mean()); r["turnover_traded_value"] = float(df[f"{s}_turnover_value"].mean()); r["buy_fill_rate"] = float(df[f"{s}_buy_filled"].sum() / max(1, df[f"{s}_buy_attempts"].sum()))
        r["cash_frac"] = float(df[f"{s}_cash_frac_post"].mean())
    r.update({"net_ex_" + k: v for k, v in sb(net - ew[c]).items()}); g = sb(gro - ew["c0"]); r["gross_excess_mean"] = g["excess_mean"]; r["gross_excess_t"] = g["t"]; return r
rows = [row(n, d, s) for n, (d, s) in SRC.items()]
e = dict(strategy="EqualWeight(same regime, 0.30% RT)", cost_roundtrip=0.003); e.update({k: v for k, v in L.perf(mlp["EqualWeight_ret_c30"]).items() if k != "periods"}); e["net_mean_per_period"] = float(mlp["EqualWeight_ret_c30"].mean()); rows.append(e)
side = pd.DataFrame(rows); side.to_csv(OUT / "side_by_side_c30.csv", index=False, encoding="utf-8-sig")
# cost sensitivity
cs = []
for n, (d, s) in SRC.items():
    for c in ["c0", "c10", "c15", "c30", "c60"]:
        if f"{s}_ret_{c}" not in d: continue
        r = row(n, d, s, c); cs.append({k: r[k] for k in ["strategy", "cost_roundtrip", "annualized_return", "sharpe", "max_drawdown", "net_mean_per_period", "net_ex_excess_mean", "net_ex_t", "net_ex_hac4_t", "net_ex_boot_lo", "net_ex_boot_hi"]})
for c in ["c0", "c10", "c15", "c30", "c60"]:
    r = dict(strategy="EqualWeight", cost_roundtrip=int(c[1:]) / 10000); r.update({k: v for k, v in L.perf(ew[c]).items() if k in ("annualized_return", "sharpe", "max_drawdown")}); r["net_mean_per_period"] = float(ew[c].mean()); cs.append(r)
pd.DataFrame(cs).to_csv(OUT / "cost_sensitivity.csv", index=False, encoding="utf-8-sig")
# by year (0.30%)
by = []
for n, (d, s) in SRC.items():
    for y, g in d.groupby(d.signal_date.str[:4]):
        gi = g.index; by.append(dict(year=y, strategy=n, periods=len(g), compounded_net=float((1 + g[f"{s}_ret_c30"]).prod() - 1), mean_net=float(g[f"{s}_ret_c30"].mean()), RankIC=float(g[f"{s}_RankIC"].mean()),
                                    net_excess_mean=float((g[f"{s}_ret_c30"] - ew["c30"].loc[gi]).mean())))
for y, g in mlp.groupby(mlp.signal_date.str[:4]): by.append(dict(year=y, strategy="EqualWeight", periods=len(g), compounded_net=float((1 + g["EqualWeight_ret_c30"]).prod() - 1), mean_net=float(g["EqualWeight_ret_c30"].mean())))
pd.DataFrame(by).to_csv(OUT / "by_year.csv", index=False, encoding="utf-8-sig")
# paired MLP_mean vs HGB / LGBM (net 0.30%, RankIC)
pr = []
for ref in ["HGB", "LGBM"]:
    d = mlp["MLP_mean_ret_c30"] - p0[f"{ref}_ret_c30"]; r = dict(comparison=f"MLP_mean minus P0 {ref}: net return per period (0.30% RT)", **sb(d))
    di = mlp["MLP_mean_RankIC"] - p0[f"{ref}_RankIC"]; r2 = dict(comparison=f"MLP_mean minus P0 {ref}: RankIC per period", **sb(di)); pr += [r, r2]
pd.DataFrame(pr).to_csv(OUT / "paired_vs_p0.csv", index=False, encoding="utf-8-sig")
# seed distribution
sd = side[side.strategy.str.startswith("MLP")][["strategy", "annualized_return", "sharpe", "max_drawdown", "RankIC", "net_ex_excess_mean", "net_ex_t", "net_ex_hac4_t", "net_ex_boot_lo", "net_ex_boot_hi", "turnover_replaced_frac", "buy_fill_rate"]]
sd.to_csv(OUT / "seed_distribution.csv", index=False, encoding="utf-8-sig")
json.dump(dict(harness_selfcheck_maxdiff_c30=selfcheck, note="P0 file vs re-simulation of the P0 Top50 sets with this harness; 0 => identical engine/cost convention"), open(OUT / "harness_selfcheck.json", "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(side[["strategy", "annualized_return", "sharpe", "max_drawdown", "RankIC", "IC", "RankIC_IR", "net_ex_excess_mean", "net_ex_t", "net_ex_hac4_t", "net_ex_boot_lo", "net_ex_boot_hi", "turnover_replaced_frac", "buy_fill_rate"]].round(4).to_string())
print(pd.DataFrame(cs).round(4).to_string()); print(pd.DataFrame(by).round(4).to_string()); print(pd.DataFrame(pr).round(4).to_string())