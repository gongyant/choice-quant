from pathlib import Path
p=Path('collect_unadj.py'); s=p.read_text(encoding='utf-8')
s=s.replace('''    if stop.is_set(): return
    code, exch = row.code, row.exchange''','''    if stop.is_set(): return
    code, exch = row.code, row.exchange
    if phase == "A" and (RU / f"{code}.parquet").exists() and (FA / f"{code}.parquet").exists(): return   # another worker already cached it''')
s=s.replace('''ap.add_argument("--limit", type=int, default=0)''','''ap.add_argument("--limit", type=int, default=0); ap.add_argument("--reverse", action="store_true")''')
s=s.replace('''    if a.limit: todo = todo.head(a.limit)''','''    if a.reverse: todo = todo.iloc[::-1]
    if a.limit: todo = todo.head(a.limit)''')
p.write_text(s,encoding='utf-8')
