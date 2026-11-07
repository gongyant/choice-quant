# -*- coding: utf-8 -*-
"""P6 forward verification APPEND-ONLY. Matures p5 10-22 and p6 post-10-22 registries when P3 merged data advances."""
import json, hashlib, datetime as dt
import numpy as np, pandas as pd
import p6_common as Q
P = Q.P; REG = Q.P6 / "predictions_registry"; OUT = Q.P6 / "forward_validation.csv"; LOG = Q.P6 / "logs" / "verify_log.jsonl"
man = {}
if (REG / "manifest.jsonl").exists():
    for l in open(REG / "manifest.jsonl", encoding="utf-8-sig"): r = json.loads(l); man[r["file"]] = r
# Also verify P5 10-22 files (frozen) without modifying them
p5reg = Q.P5D / "predictions_registry"
calf = pd.read_csv(P.MG / "trade_calendar.csv"); end = str(pd.to_datetime(calf.iloc[:, 0]).max().date())
X = P.load_world(end); cal = X.cal; col = {c: j for j, c in enumerate(X.codes)}; Oa = np.where(X.traded, X.O / X.F, np.nan)
done = set()
if OUT.exists():
    old = pd.read_csv(OUT, dtype=str, encoding="utf-8-sig")
    if len(old): done = set(map(tuple, old[["registry_file", "horizon", "code"]].values)) if "registry_file" in old.columns else set()
new = []
files = list(REG.glob("p6_pred_*_B*_h*.csv")) + list(p5reg.glob("p5_pred_20261022_B*_h*.csv"))
for f in sorted(files):
    if f.name in man: assert hashlib.sha256(f.read_bytes()).hexdigest() == man[f.name]["sha256"], f.name
    d = pd.read_csv(f, dtype={"code": str}, encoding="utf-8-sig"); sd = str(d.signal_date.iloc[0]); h = int(d.horizon.iloc[0])
    if pd.Timestamp(sd) not in cal: continue
    i = int(cal.get_loc(pd.Timestamp(sd)))
    if i + 1 + h >= len(cal): continue
    e, x = i + 1, i + 1 + h
    # pool: p6 pool or p5 pool or elig
    pf = REG / f"p6_pool_{sd.replace('-','')}.csv"
    if not pf.exists(): pf = p5reg / f"p5_pool_{sd.replace('-','')}.csv"
    pool = [col[c] for c in (pd.read_csv(pf, dtype={"code": str}, encoding="utf-8-sig").code if pf.exists() else []) if c in col]
    if not pool: pool = list(np.where(X.traded[i])[0])
    pr = float(np.nanmean(Oa[x, pool] / Oa[e, pool] - 1))
    for r in d[d.in_final5 == True].itertuples():
        key = (f.name, str(h), r.code)
        if key in done: continue
        j = col.get(r.code); ret = float(Oa[x, j] / Oa[e, j] - 1) if j is not None and X.traded[e, j] else np.nan
        new.append(dict(signal_date=sd, strategy=r.strategy, horizon=h, code=r.code, name=r.name, entry_date=str(cal[e].date()), exit_date=str(cal[x].date()),
                        ret_open_to_open=ret, ret_net=(ret - 0.003) if np.isfinite(ret) else np.nan, pool_ret=pr,
                        rel=(ret - 0.003 - pr) if np.isfinite(ret) else np.nan, verified_at=dt.datetime.now().isoformat(timespec="seconds"),
                        registry_file=f.name, note="" if np.isfinite(ret) else "未取得"))
if new: pd.DataFrame(new).to_csv(OUT, mode="a", header=not OUT.exists() or OUT.stat().st_size == 0, index=False, encoding="utf-8-sig")
LOG.parent.mkdir(exist_ok=True); open(LOG, "a", encoding="utf-8").write(json.dumps(dict(run_at=dt.datetime.now().isoformat(timespec="seconds"), data_end=end, new_rows=len(new)), ensure_ascii=False) + "\n")
print("data_end", end, "new_rows", len(new))
