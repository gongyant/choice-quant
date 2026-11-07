from pathlib import Path
p=Path('p0_data_audit.py'); s=p.read_text(encoding='utf-8')
s=s.replace('''    f = L.TXD / f"{code}.parquet"; return pd.read_parquet(f).set_index("date") if f.exists() else None''','''    f = L.TXD / f"{code}.parquet"
    if not f.exists(): return None
    d = pd.read_parquet(f); d = d.rename(columns={"volume_raw": "volume"}) if "volume_raw" in d.columns else d
    return d.set_index("date")''')
s=s.replace('''rows = []; Cf = L.ffill_mat(X.C)''','''rows = []; Cf = L.ffill_mat(X.C); oldg = {c: g.sort_values("date").reset_index(drop=True) for c, g in old.groupby("code")}''')
s=s.replace('''    g = old[old.code == c].sort_values("date").reset_index(drop=True)
    if len(g) < 2: continue''','''    g = oldg.get(c)
    if g is None or len(g) < 2: continue''')
p.write_text(s,encoding='utf-8')
