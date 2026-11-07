from pathlib import Path
p=Path('p0lib.py'); s=p.read_text(encoding='utf-8-sig')
s=s.replace('''(fac.factor_status.iloc[0] if "factor_status" in fac else "ok")''','''("empty_table(sina_not_available_or_no_actions)" if len(fac) == 0 else "ok")''')
p.write_text(s,encoding='utf-8')
