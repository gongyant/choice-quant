# -*- coding: utf-8 -*-
"""P5 forward verification (APPEND-ONLY). For every sha-checked p5 registry list whose horizon has matured in the P3 merged data
(updated by the weekly P3 fetch+merge; read here only), compute t+1 open -> t+1+h open adjusted return of each final-5 name, the same for the registered
eligible pool, and net relative = ret - 0.30% - pool. Appends to forward_verify.csv; existing rows are never rewritten. Then re-evaluates the pre-registered
buy criterion (p5_batchB gate logic) into results/b5_buy_gate_latest.csv. Research only; no broker, no orders."""
import json, hashlib, datetime as dt
import numpy as np, pandas as pd
import p5_common as Q
P = Q.P; REG = Q.P5 / "predictions_registry"; OUT = Q.P5 / "forward_verify.csv"; LOG = Q.P5 / "logs" / "verify_log.jsonl"
man = {json.loads(l)["file"]: json.loads(l) for l in open(REG / "manifest.jsonl", encoding="utf-8-sig")}
calf = pd.read_csv(P.MG / "trade_calendar.csv"); end = str(pd.to_datetime(calf.iloc[:, 0]).max().date())
X = P.load_world(end); cal = X.cal; col = {c: j for j, c in enumerate(X.codes)}; Oa = np.where(X.traded, X.O / X.F, np.nan)
done = set(map(tuple, pd.read_csv(OUT, dtype=str)[["file", "horizon"]].values)) if OUT.exists() else set(); new = []
for f in sorted(REG.glob("p5_pred_*_B*_h*.csv")):
    assert hashlib.sha256(f.read_bytes()).hexdigest() == man[f.name]["sha256"], f"sha mismatch {f.name}"
    d = pd.read_csv(f, dtype={"code": str}, encoding="utf-8-sig"); sd = str(d.signal_date.iloc[0]); h = int(d.horizon.iloc[0])
    if (f.name, str(h)) in done or pd.Timestamp(sd) not in cal: continue
    i = int(cal.get_loc(pd.Timestamp(sd)))
    if i + 1 + h >= len(cal): continue
    e, x = i + 1, i + 1 + h; pf = REG / f"p5_pool_{sd.replace('-', '')}.csv"
    pool = [col[c] for c in pd.read_csv(pf, dtype={"code": str}, encoding="utf-8-sig").code if c in col]
    pr = np.nanmean(Oa[x, pool] / Oa[e, pool] - 1)
    for r in d[d.in_final5].itertuples():
        j = col.get(r.code); ret = Oa[x, j] / Oa[e, j] - 1 if j is not None and X.traded[e, j] else np.nan
        new.append(dict(file=f.name, strategy=r.strategy, model_version=r.model_version, signal_date=sd, horizon=h, code=r.code, name=r.name, entry_date=str(cal[e].date()), exit_date=str(cal[x].date()),
                        ret_open_to_open=ret, ret_net=ret - 0.003, pool_ret=pr, rel=ret - 0.003 - pr, verified_at=dt.datetime.now().isoformat(timespec="seconds"),
                        note="" if np.isfinite(ret) else "未取得(停牌/无价),未计入"))
if new: pd.DataFrame(new).to_csv(OUT, mode="a", header=not OUT.exists(), index=False, encoding="utf-8-sig")
LOG.parent.mkdir(exist_ok=True); open(LOG, "a", encoding="utf-8").write(json.dumps(dict(run_at=dt.datetime.now().isoformat(timespec="seconds"), data_end=end, new_rows=len(new)), ensure_ascii=False) + "\n")
print("data_end", end, "new rows", len(new))
if OUT.exists():
    v = pd.read_csv(OUT, encoding="utf-8-sig"); g = v.groupby(["strategy", "horizon", "signal_date"]).agg(n=("code", "size"), port_net=("ret_net", "mean"), pool=("pool_ret", "first"), rel=("rel", "mean")).reset_index()
    g.to_csv(Q.P5 / "results" / "b5_forward_by_period.csv", index=False, encoding="utf-8-sig"); print(g.round(4).to_string())
    s = g.groupby(["strategy", "horizon"]).agg(periods=("rel", "size"), mean_rel=("rel", "mean"), beat=("rel", lambda z: (z > 0).mean())).reset_index()
    s["C5_enough_periods"] = s.periods >= np.where(s.horizon == 5, 12, 6); s["C6_fwd_ok"] = (s.mean_rel > 0) & (s.beat >= .5)
    B = pd.read_csv(Q.P5 / "results" / "b5_buy_gate.csv", encoding="utf-8-sig")
    B["key"] = B.strategy.str.split("_").str[0]; B = B.merge(s.rename(columns={"strategy": "key"}), on=["key", "horizon"], how="left")
    B["PASS_latest"] = B.C1 & B.C2 & B.C3 & B.C4 & B.C5_enough_periods.fillna(False) & B.C6_fwd_ok.fillna(False)
    B["结论_latest"] = np.where(B.PASS_latest, "通过(证据支持,仍非投资建议)", "不通过:不构成买入依据"); B.to_csv(Q.P5 / "results" / "b5_buy_gate_latest.csv", index=False, encoding="utf-8-sig"); print(B[["strategy", "horizon", "periods", "mean_rel", "PASS_latest"]].to_string())
