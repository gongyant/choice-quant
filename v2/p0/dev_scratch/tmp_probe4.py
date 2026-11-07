import akshare as ak, traceback
for s in ['sz000004','sh689009']:
    for adj in ['qfq','hfq']:
        try:
            d=ak.stock_zh_a_daily(symbol=s,start_date='20210101',end_date='20261016',adjust=adj); print(s,adj,'OK',d.shape)
        except Exception as e: print(s,adj,'ERR',type(e).__name__,str(e)[:80])
    try:
        d=ak.stock_zh_a_hist_tx(symbol=s,start_date='20210101',end_date='20261016',adjust=''); print(s,'tx raw OK',d.shape, d.tail(2).to_string())
    except Exception as e: print(s,'tx raw ERR',type(e).__name__,str(e)[:80])
    try:
        d=ak.stock_zh_a_daily(symbol=s,start_date='20240101',end_date='20241231',adjust=''); print(s,'narrow range OK',d.shape)
    except Exception as e: print(s,'narrow ERR',type(e).__name__,str(e)[:80])
