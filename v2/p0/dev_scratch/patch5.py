from pathlib import Path
p=Path('p0lib.py'); s=p.read_text(encoding='utf-8')
s=s.replace("nolim = near_ipo[None, :] & (k < width)","nolim = near_ipo[None, :] & (k >= 0) & (k < width)")
p.write_text(s,encoding='utf-8')
