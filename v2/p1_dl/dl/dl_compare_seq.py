"""B2/B3 comparison: sequence model (GRU / Transformer) vs B1 MLP vs frozen P0 HGB/LGBM vs same-regime equal weight. Usage: python dl_compare_seq.py <seq_tag> <PREFIX e.g. GRU>"""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dl_common as C
L = C.L; DL = C.DL
TAG, PFX = sys.argv[1], sys.argv[2]; OUT = DL / "results" / TAG
seq = pd.read_csv(OUT / "period_returns.csv"); mlp = pd.read_csv(DL / "results" / "mlp" / "period_returns.csv"); p0 = pd.read_csv(L.P0 / "results" / "fixed_sina_adj" / "period_returns.csv")
assert (seq.signal_date.values == mlp.signal_date.values).all() and (seq.signal_date.values == p0.signal_date.values[:len(seq)]).all(); p0 = p0.iloc[:len(seq)].reset_index(drop=True)
W = C.build_world(); X = W.X; idx = {c: j for j, c in enumerate(X.codes)}; sets = pickle.load(open(L.P0 / "results" / "fixed_sina_adj" / "sets.pkl", "rb")); sig = sets["signal_idx"]; selfcheck = {}
for s in ["HGB", "LGBM"]:
    tg = [np.array([idx[c] for c in t]) for t in sets["topk"][s]]
    for c in [0.0015, 0.003]:
        o, _, _ = L.simulate(X, W.ADJf, sig, tg, c, capital=1e7); p0[f"{s}_ret_{'c15' if c < 0.002 else 'c30r'}"] = o.ret.values
    selfcheck[s] = float(np.abs(p0[f"{s}_ret_c30r"].values - p0[f"{s}_ret_c30"].values).max())
assert np.allclose(seq["EqualWeight_ret_c30"], p0["EqualWeight_ret_c30"], atol=1e-12) and np.allclose(seq["EqualWeight_ret_c30"], mlp["EqualWeight_ret_c30"], atol=1e-12)
def sb(ex):
    ex = pd.Series(ex).dropna().values; t, p = L.tstat(ex); t4, p4 = L.nw_t(ex, 4); lo, hi = L.block_boot(ex)
    return dict(excess_mean=float(ex.mean()), t=t, p=p, hac4_t=t4, hac4_p=p4, boot_lo=lo, boot_hi=hi, n=len(ex))
seeds = [c.split("_ret_c0")[0] for c in seq.columns if c.startswith(f"{PFX}_s") and c.endswith("_ret_c0")]
SRC = {"P0_HGB": (p0, "HGB"), "P0_LGBM": (p0, "LGBM"), "B1_MLP_mean": (mlp, "MLP_mean")}
gru = None
if PFX != "GRU" and (DL / "results" / "gru" / "period_returns.csv").exists():
    gru = pd.read_csv(DL / "results" / "gru" / "period_returns.csv"); assert (gru.signal_date.values == seq.signal_date.values).all(); SRC["B2_GRU_mean"] = (gru, "GRU_mean")
for s in seeds: SRC[f"B_{s}"] = (seq, s)
SRC[f"B_{PFX}_mean"] = (seq, f"{PFX}_mean")
ew = {c: seq[f"EqualWeight_ret_{c}"] for c in ["c0", "c10", "c15", "c30", "c60"]}
def get(df, s, c):
    return df[f"{s}_ret_{c}"] if f"{s}_ret_{c}" in df else None
def row(name, df, s, c="c30"):
    net, gro = get(df, s, c), get(df, s, "c0"); r = dict(strategy=name, cost_roundtrip=int(c[1:]) / 10000); r.update({k: v for k, v in L.perf(net).items() if k != "periods"}); r["net_mean_per_period"] = float(net.mean()); r["gross_mean_per_period"] = float(gro.mean())
    r.update(IC=float(df[f"{s}_IC"].mean()), RankIC=float(df[f"{s}_RankIC"].mean()), RankIC_IR=float(df[f"{s}_RankIC"].mean() / df[f"{s}_RankIC"].std()), RankIC_t=L.tstat(df[f"{s}_RankIC"].dropna())[0])
    if f"{s}_replaced" in df and c == "c30": r.update(turnover_replaced_frac=float(df[f"{s}_replaced"].mean()), turnover_traded_value=float(df[f"{s}_turnover_value"].mean()), buy_fill_rate=float(df[f"{s}_buy_filled"].sum() / max(1, df[f"{s}_buy_attempts"].sum())), cash_frac=float(df[f"{s}_cash_frac_post"].mean()))
    r.update({"net_ex_" + k: v for k, v in sb(net - ew[c]).items()}); g = sb(gro - ew["c0"]); r["gross_excess_mean"] = g["excess_mean"]; r["gross_excess_t"] = g["t"]; return r
rows = [row(n, d, s) for n, (d, s) in SRC.items()]; e = dict(strategy="EqualWeight(same regime, 0.30% RT)", cost_roundtrip=0.003); e.update({k: v for k, v in L.perf(ew["c30"]).items() if k != "periods"}); e["net_mean_per_period"] = float(ew["c30"].mean()); rows.append(e)
side = pd.DataFrame(rows); side.to_csv(OUT / "side_by_side_c30.csv", index=False, encoding="utf-8-sig")
cs = []
for n, (d, s) in SRC.items():
    for c in ["c0", "c10", "c15", "c30", "c60"]:
        if get(d, s, c) is None: continue
        r = row(n, d, s, c); cs.append({k: r[k] for k in ["strategy", "cost_roundtrip", "annualized_return", "sharpe", "max_drawdown", "net_mean_per_period", "net_ex_excess_mean", "net_ex_t", "net_ex_hac4_t", "net_ex_boot_lo", "net_ex_boot_hi"]})
for c in ["c0", "c10", "c15", "c30", "c60"]:
    r = dict(strategy="EqualWeight", cost_roundtrip=int(c[1:]) / 10000); r.update({k: v for k, v in L.perf(ew[c]).items() if k in ("annualized_return", "sharpe", "max_drawdown")}); r["net_mean_per_period"] = float(ew[c].mean()); cs.append(r)
pd.DataFrame(cs).to_csv(OUT / "cost_sensitivity.csv", index=False, encoding="utf-8-sig")
by = []
for n, (d, s) in SRC.items():
    for y, g in d.groupby(d.signal_date.str[:4]): by.append(dict(year=y, strategy=n, periods=len(g), compounded_net=float((1 + g[f"{s}_ret_c30"]).prod() - 1), mean_net=float(g[f"{s}_ret_c30"].mean()), RankIC=float(g[f"{s}_RankIC"].mean()), net_excess_mean=float((g[f"{s}_ret_c30"] - ew["c30"].loc[g.index]).mean())))
for y, g in seq.groupby(seq.signal_date.str[:4]): by.append(dict(year=y, strategy="EqualWeight", periods=len(g), compounded_net=float((1 + g["EqualWeight_ret_c30"]).prod() - 1), mean_net=float(g["EqualWeight_ret_c30"].mean())))
pd.DataFrame(by).to_csv(OUT / "by_year.csv", index=False, encoding="utf-8-sig")
main_s = f"{PFX}_mean"; pr = []
for ref, (d, s) in [("P0 HGB", (p0, "HGB")), ("P0 LGBM", (p0, "LGBM")), ("B1 MLP_mean", (mlp, "MLP_mean"))] + ([("B2 GRU_mean", (gru, "GRU_mean"))] if gru is not None else []):
    pr.append(dict(comparison=f"{main_s} minus {ref}: net return per period (0.30% RT)", **sb(seq[f"{main_s}_ret_c30"] - d[f"{s}_ret_c30"]))); pr.append(dict(comparison=f"{main_s} minus {ref}: RankIC per period", **sb(seq[f"{main_s}_RankIC"] - d[f"{s}_RankIC"])))
pd.DataFrame(pr).to_csv(OUT / "paired.csv", index=False, encoding="utf-8-sig")
json.dump(dict(harness_selfcheck_maxdiff_c30=selfcheck), open(OUT / "harness_selfcheck.json", "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(side[["strategy", "annualized_return", "sharpe", "max_drawdown", "RankIC", "IC", "RankIC_IR", "net_ex_excess_mean", "net_ex_t", "net_ex_hac4_t", "net_ex_boot_lo", "net_ex_boot_hi", "turnover_replaced_frac", "buy_fill_rate", "gross_excess_t"]].round(4).to_string())
print(pd.DataFrame(cs).round(4).to_string()); print(pd.DataFrame(by).round(4).to_string()); print(pd.DataFrame(pr).round(4).to_string())