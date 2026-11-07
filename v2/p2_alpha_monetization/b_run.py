# -*- coding: utf-8 -*-
import sys, json, argparse
from pathlib import Path
import pandas as pd
import p2lib as Q, b_lib as BL, b_data as BD
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--scen", required=True); ap.add_argument("--stress", action="store_true"); ap.add_argument("--logs", action="store_true"); a = ap.parse_args()
out = Q.RES / "b_runs"; out.mkdir(exist_ok=True, parents=True)
allp, ex = BL.run_model_scen(a.model, a.scen, with_stress=a.stress, log_dir=(Q.RES / "b_logs") if a.logs else None)
allp.to_parquet(out / f"{a.model}_{a.scen}.parquet", index=False); (out / f"{a.model}_{a.scen}_extras.json").write_text(json.dumps(ex, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("done", a.model, a.scen, ex["seconds"], "s", {k: round(v["param"],3) for k, v in ex["controls"].items()}, [x["chosen"] for x in ex["selection"]["A2"]], [x["chosen"] for x in ex["selection"]["A1"]])