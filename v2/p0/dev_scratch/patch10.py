from pathlib import Path
p=Path('p0_data_audit.py'); s=p.read_text(encoding='utf-8')
old='''        if pd.notna(dd) and stale and (dd - cal[last_t]).days > 5: cause = "delisted_stock: data ends >5d before delist date (suspension/delisting-arrangement gap or source)"
        elif pd.notna(dd) and stale: cause = "delisted_stock: data ends at delisting (normal)"'''
new='''        gapd = (dd - cal[last_t]).days if pd.notna(dd) else None
        if pd.notna(dd) and stale and gapd > 70: cause = "delisted_stock: data ends >70d before delist date (check: merger/halt/source gap)"
        elif pd.notna(dd) and stale and gapd > 10: cause = "delisted_stock: last bar 10-70d before termination date (consistent with delisting-arrangement period / halt; normal)"
        elif pd.notna(dd) and stale: cause = "delisted_stock: data ends at delisting (normal)"'''
assert old in s; s=s.replace(old,new); p.write_text(s,encoding='utf-8')
