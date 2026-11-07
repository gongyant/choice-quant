# -*- coding: utf-8 -*-
import subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import p2lib as Q
jobs = []
for scen in ["C1_flat", "C2_explicit", "C3_realistic", "C4_capacity_1e8"]:
    for m in Q.MODELS:
        extra = []
        if m in ("HGB", "LGBM") and scen == "C2_explicit": extra.append("--stress")
        if m in ("HGB", "LGBM", "GRU") or scen in ("C1_flat", "C2_explicit"): extra.append("--logs")
        if (Q.RES / 'b_runs' / f'{m}_{scen}.parquet').exists() and (Q.RES / 'b_runs' / f'{m}_{scen}_extras.json').exists(): continue
        jobs.append([sys.executable, "b_run.py", "--model", m, "--scen", scen] + extra)
def go(cmd):
    t0 = time.time(); log = Q.RES / "b_runs" / f"log_{cmd[3]}_{cmd[5]}.txt"; log.parent.mkdir(exist_ok=True)
    with open(log, "w", encoding="utf-8") as f: r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(Q.P2))
    return cmd[3], cmd[5], r.returncode, round(time.time() - t0)
with ThreadPoolExecutor(max_workers=3) as ex:
    for r in ex.map(go, jobs): print(r, flush=True)
print("ALL DONE", flush=True)