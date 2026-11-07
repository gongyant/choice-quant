from pathlib import Path
p=Path('collect_unadj.py'); s=p.read_text(encoding='utf-8')
s=s.replace('''    if phase == "A" and (RU''','''    if phase == "B" and (TX / f"{code}.parquet").exists(): return
    if phase == "A" and (RU''')
p.write_text(s,encoding='utf-8')
