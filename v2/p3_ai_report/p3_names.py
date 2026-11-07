# -*- coding: utf-8 -*-
"""Current security short names (for the '现名含 ST' rule) from Tencent quote API qt.gtimg.cn (batch). Writes data/current_names.csv."""
import socket, time, datetime as dt
socket.setdefaulttimeout(15)
from pathlib import Path
import pandas as pd, requests
H = Path(__file__).resolve().parent; V2 = H.parent
U = pd.read_csv(V2 / "data" / "sample_1500.csv", dtype=str); syms = [(r.exchange.lower() + r.code, r.code) for r in U.itertuples()]
rows = []
for k in range(0, len(syms), 60):
    part = syms[k:k + 60]
    for a in range(3):
        try:
            t = requests.get("https://qt.gtimg.cn/q=" + ",".join(s for s, _ in part), timeout=15).content.decode("gbk", errors="replace"); break
        except Exception as e: t = ""; time.sleep(2)
    for line in t.split(";"):
        if "~" not in line: continue
        f = line.split('"')[1].split("~") if '"' in line else []
        if len(f) > 3 and f[2]: rows.append(dict(code=f[2], name_now=f[1].replace(" ", ""), quote_time=f[30] if len(f) > 30 else ""))
    time.sleep(0.2)
r = pd.DataFrame(rows).drop_duplicates("code"); r["source"] = "https://qt.gtimg.cn/q=<symbols>"; r["fetched_at"] = dt.datetime.now().isoformat(timespec="seconds")
r.to_csv(H / "data" / "current_names.csv", index=False, encoding="utf-8-sig"); print(len(r), "of", len(syms))
m = U.set_index("code").name; r = r.set_index("code"); ch = [(c, m[c], r.name_now[c]) for c in r.index if c in m.index and m[c] != r.name_now[c]]; print("changed", len(ch)); print(ch[:40])