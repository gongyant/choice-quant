# -*- coding: utf-8 -*-
"""P4 forward verification (append-only): S4 (E2S4 h5/h20) and the low-vol Top5. For each registered file (sha checked vs manifest) whose horizon has
matured in the merged data (P3 pipeline: p3_fetch + p3_merge), compute t+1 open -> t+1+h open adjusted returns of the final Top5 and of the signal-day
eligible pool (= codes in the E1reg full-pool registry of that date), and APPEND to p3_ai_report/verification/p4_forward_verify.csv. Never rewrites rows."""
import json, hashlib, datetime as dt
import numpy as np, pandas as pd
import p4lib as P
REG = P.P4 / "predictions_registry"; VF = P.P3 / "verification"; VF.mkdir(exist_ok=True); OUT = VF / "p4_forward_verify.csv"; LOG = VF / "p4_verify_log.jsonl"
man = {}
for l in open(REG / "manifest.jsonl", encoding="utf-8-sig"): r = json.loads(l); man[r["file"]] = r
calf = pd.read_csv(P.MG / "trade_calendar.csv"); end = str(pd.to_datetime(calf.iloc[:, 0]).max().date())
X = P.load_world(end); cal = X.cal; col = {c: j for j, c in enumerate(X.codes)}; Oa = np.where(X.traded, X.O / X.F, np.nan)
done = set(map(tuple, pd.read_csv(OUT, dtype=str)[["file", "horizon"]].values)) if OUT.exists() else set(); new = []
for f in sorted(REG.glob("p4_pred_*_E2S4_h*.csv")) + sorted(REG.glob("p4_pred_*_LOWVOL_top5.csv")):
    assert hashlib.sha256(f.read_bytes()).hexdigest() == man[f.name]["sha256"], f"sha mismatch {f.name}"
    d = pd.read_csv(f, dtype={"code": str}); sd = str(d.signal_date.iloc[0]); i = int(cal.get_loc(pd.Timestamp(sd)))
    hs = [int(f.stem.split("_h")[-1])] if "_h" in f.stem else [5, 20]
    for h in hs:
        if (f.name, str(h)) in done or i + 1 + h >= len(cal): continue
        e, x = i + 1, i + 1 + h; poolf = REG / f"p4_pred_{sd.replace('-', '')}_E1reg_h5.csv"
        pool = [col[c] for c in pd.read_csv(poolf, dtype={"code": str}).code if c in col] if poolf.exists() else list(np.where(X.traded[i])[0])
        pr = np.nanmean(Oa[x, pool] / Oa[e, pool] - 1); picks = d[d.in_final5]
        for r in picks.itertuples():
            j = col[r.code]; ret = Oa[x, j] / Oa[e, j] - 1 if X.traded[e, j] else np.nan
            new.append(dict(file=f.name, strategy="S4" if "E2S4" in f.name else "LOWVOL", signal_date=sd, horizon=h, code=r.code, name=r.name, entry_date=str(cal[e].date()), exit_date=str(cal[x].date()),
                            ret_open_to_open=ret, pool_ret=pr, rel=ret - pr, verified_at=dt.datetime.now().isoformat(timespec="seconds")))
if new: pd.DataFrame(new).to_csv(OUT, mode="a", header=not OUT.exists(), index=False, encoding="utf-8-sig")
open(LOG, "a", encoding="utf-8").write(json.dumps(dict(run_at=dt.datetime.now().isoformat(timespec="seconds"), data_end=end, new_rows=len(new)), ensure_ascii=False) + "\n")
print("data_end", end, "new rows", len(new))
if OUT.exists():
    v = pd.read_csv(OUT); print(v.groupby(["strategy", "signal_date", "horizon"]).agg(n=("code", "size"), mean_ret=("ret_open_to_open", "mean"), pool=("pool_ret", "first"), hit=("rel", lambda s: (s > 0).mean())).round(4).to_string())
