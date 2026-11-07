"""Step 0: verify P0 inputs (paths, sha256, rows, date ranges) against p0/audit_manifest.json before anything is scored."""
import sys, json, glob, hashlib
from pathlib import Path
import pandas as pd
H = Path(__file__).resolve().parent; V2 = H.parent; sys.path.insert(0, str(V2/"p0"))
import p0lib as L
def dir_hash(files):
    h = hashlib.sha256()
    for f in sorted(files): h.update(Path(f).name.encode()); h.update(L.sha256(f).encode())
    return h.hexdigest()
man = json.load(open(V2/"p0"/"audit_manifest.json", encoding="utf-8")); out = {"checks": []}
def chk(name, ok, detail): out["checks"].append(dict(name=name, ok=bool(ok), detail=detail)); print("OK " if ok else "FAIL", name, detail)
for k, f in [("sample_1500.csv", V2/"data/sample_1500.csv"), ("trade_calendar.csv", L.D/"trade_calendar.csv")]:
    s = L.sha256(f); chk("sha256 "+k, s == man["source_files"][k], s)
for k, d, key in [("raw_unadj_parquet", L.RU, "raw_unadj_parquet(Sina primary/Tencent fallback)"), ("adj_factors", L.FA, "adj_factors(Sina qfq-factor)"), ("raw_unadj_tx", L.TXD, "raw_unadj_tx(Tencent raw cross-check)")]:
    fl = glob.glob(str(d/"*.parquet")); h = dir_hash(fl); chk("tree sha256 "+k, h == man["data_dirs"][key]["tree_sha256"] and len(fl) == man["data_dirs"][key]["files"], f"{len(fl)} files {h[:16]}")
for k in ["p0lib.py", "baseline_fixed.py"]:
    s = L.sha256(V2/"p0"/k); chk("sha256 "+k, s == man["scripts"][k]["sha256"], s)
chk("sha256 baseline.py (old, untouched)", L.sha256(V2/"baseline.py") == man["old_scripts_untouched"]["baseline.py"], "")
u = pd.read_csv(V2/"data/sample_1500.csv", dtype=str); chk("sample rows", len(u) == 1500 and u.code.is_unique and (u.code.str.len() == 6).all(), f"{len(u)} rows, unique={u.code.is_unique}")
cal = L.load_calendar(); chk("calendar", len(cal) == 1400 and cal[0] == pd.Timestamp("2021-01-04") and cal[-1] == pd.Timestamp("2026-10-16"), f"{len(cal)} days {cal[0].date()}..{cal[-1].date()}")
rows, mn, mx, bad = [], [], [], 0
for c in u.code:
    f = L.RU/f"{c}.parquet"
    if not f.exists(): continue
    d = pd.read_parquet(f, columns=["date", "close"]); rows.append(len(d)); mn.append(d.date.min()); mx.append(d.date.max()); bad += int((d.close <= 0).sum())
chk("raw_unadj coverage", len(rows) == 1498, f"{len(rows)} files; rows total {sum(rows)}; first {min(mn).date()} last {max(mx).date()}; non-positive closes {bad}")
out["missing_codes"] = sorted(set(u.code) - {p.stem for p in L.RU.glob("*.parquet")})
out["all_ok"] = all(c["ok"] for c in out["checks"]); json.dump(out, open(H/"p0_input_verification.json", "w"), ensure_ascii=False, indent=1)
print("ALL OK" if out["all_ok"] else "SOME FAILED", out["missing_codes"])
