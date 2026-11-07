import pandas as pd, numpy as np, akshare as ak, time, pickle
pd.set_option('display.width',250); pd.set_option('display.max_columns',50)
out={}
for code,sym in [('600188','sh600188'),('600066','sh600066'),('000408','sz000408')]:
    r={}
    for adj in ['','qfq']:
        try: r['tx_'+(adj or 'raw')]=ak.stock_zh_a_hist_tx(symbol=sym,start_date='20210101',end_date='20261016',adjust=adj)
        except Exception as e: print(code,'tx',adj,'ERR',str(e)[:150])
        time.sleep(1)
    for adj in ['','qfq','qfq-factor','hfq-factor']:
        try: r['sina_'+(adj or 'raw')]=ak.stock_zh_a_daily(symbol=sym,start_date='20210101',end_date='20261016',adjust=adj)
        except Exception as e: print(code,'sina',adj,'ERR',str(e)[:150])
        time.sleep(1)
    out[code]=r
    for k,v in r.items(): print(code,k,v.shape, list(v.columns)[:9])
pickle.dump(out,open('p0/data/probe3.pkl','wb'))
