from pathlib import Path
p=Path('p0/p0_data_audit.py'); s=p.read_text(encoding='utf-8')
old='''    if f0 is not None and len(exp): cov_tx = 1 - len([d for d in exp if d not in set(f0.index)]) / len(exp)'''
new='''    if f0 is not None and len(exp):
        txs = set(f0.index); cov_tx = 1 - len([d for d in exp if d not in txs]) / len(exp)'''
assert old in s; s=s.replace(old,new)
old2='''        s = d.set_index("date"); com = s.index.intersection(f0.index); com = [x for x in com if s.volume[x] > 0 and f0.volume[x] > 0]'''
new2='''        s = d.set_index("date"); com = s.index.intersection(f0.index); com = [x for x in com if s.volume[x] > 0 and f0.volume[x] > 0]
        s = s.loc[com]; f0 = f0.loc[com]'''
assert old2 in s; s=s.replace(old2,new2); p.write_text(s,encoding='utf-8')
