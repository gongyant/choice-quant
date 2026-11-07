# -*- coding: utf-8 -*-
"""Keeps N c_wf.py workers alive (restarts a worker that died, e.g. on a transient Windows commit-memory error) until every job in c_wf.PRIORITY is complete. Jobs resume from block checkpoints."""
import subprocess, sys, time, os
from pathlib import Path
import c_wf as W
n = int(sys.argv[1]) if len(sys.argv) > 1 else 2; P2 = Path(__file__).resolve().parent; procs = []; k = 0
def log(m):
    with open(P2 / "results" / "c_super.log", "a", encoding="utf-8") as f: f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}\n")
while True:
    procs = [p for p in procs if p.poll() is None]
    for q in W.CK.glob("*/*.lock"):          # stale locks of dead workers
        try:
            pid = int(q.read_text() or 0)
            if pid and subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout.count(str(pid)) == 0: q.unlink(); log(f"removed stale lock {q.name}")
        except Exception: pass
    if all(W.job_done(*j) for j in W.PRIORITY): log("ALL JOBS DONE"); break
    if len(procs) < n:
        k += 1; out = open(P2 / "results" / "c_worker_logs" / f"sup_w{k}.out", "w"); err = open(P2 / "results" / "c_worker_logs" / f"sup_w{k}.err", "w")
        procs.append(subprocess.Popen([sys.executable, "c_wf.py", "--jobs", "auto"], cwd=str(P2), stdout=out, stderr=err)); log(f"spawned worker {k}"); time.sleep(75)
    time.sleep(20)
