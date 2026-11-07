from pathlib import Path
p=Path('collect_unadj.py'); s=p.read_text(encoding='utf-8-sig')
old_start=s.index('def get_sina'); old_end=s.index('def retry')
new='''def get_sina(code, exch):
    """Sina raw bars; if Sina cannot serve the symbol (JSONDecodeError seen for 000004/000005/689009/600317) fall back to Tencent raw (SH/SZ).
    Factor table: Sina qfq-factor. ValueError('...not available') = Sina has no factor table for the symbol -> empty table, flagged."""
    src = "sina"
    try:
        raw = ak.stock_zh_a_daily(symbol=sym(code, exch), start_date=START, end_date=END, adjust="")
    except Exception as e:
        if exch == "BJ": raise
        raw = ak.stock_zh_a_hist_tx(symbol=sym(code, exch), start_date=START, end_date=END, adjust="").rename(columns={"volume": "volume"})
        src = "tencent"
    fstat = "ok"
    try:
        fac = ak.stock_zh_a_daily(symbol=sym(code, exch), start_date=START, end_date=END, adjust="qfq-factor")
    except ValueError as e:
        if "not available" in str(e): fac = pd.DataFrame({"date": [], "qfq_factor": []}); fstat = "sina_factor_not_available"
        else: raise
    return raw, fac, src, fstat
'''
s=s[:old_start]+new+s[old_end:]
s=s.replace('''            raw, fac = res
''','''            raw, fac, src, fstat = res
''')
s=s.replace('''raw["source"] = "sina.stock_zh_a_daily(adjust='')"; raw["fetched_at"] = now''','''raw["source"] = "sina.stock_zh_a_daily(adjust='')" if src == "sina" else "tencent.stock_zh_a_hist_tx(adjust='') [sina raw unavailable]"; raw["fetched_at"] = now''')
s=s.replace('''fac["source"] = "sina.stock_zh_a_daily(adjust='qfq-factor')"; fac["fetched_at"] = now''','''fac["source"] = "sina.stock_zh_a_daily(adjust='qfq-factor')"; fac["factor_status"] = fstat; fac["fetched_at"] = now''')
s=s.replace('''st["rows"] = len(raw)''','''st["rows"] = len(raw); st["error"] = f"raw_src={src};factor={fstat}"''')
p.write_text(s,encoding='utf-8')
