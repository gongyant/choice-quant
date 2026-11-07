import pandas as pd, numpy as np, pickle, akshare as ak
pd.set_option('display.width',250); pd.set_option('display.max_columns',50)
out=pickle.load(open('p0/data/probe3.pkl','rb'))
r=out['600188']; sr=r['sina_raw'].copy(); sr['date']=pd.to_datetime(sr.date); sr=sr.sort_values('date'); sr['ret']=sr.close.pct_change(); sr['prev']=sr.close.shift()
print(sr[sr.ret.abs()>0.11].to_string())
f=r['sina_qfq-factor'].copy(); f['date']=pd.to_datetime(f.date); f['qfq_factor']=f.qfq_factor.astype(float)
f=f.sort_values('date'); sq=r['sina_qfq'].copy(); sq['date']=pd.to_datetime(sq.date)
m=sr[['date','close']].merge(sq[['date','close']],on='date',suffixes=('','_q'))
fs=f.set_index('date').qfq_factor
m['f']=[fs[:d].iloc[-1] if len(fs[:d]) else np.nan for d in m.date]
m['mine']=m.close/m.f
print('formula check max abs diff vs sina qfq:',(m.mine-m.close_q).abs().max())
# tencent qfq - raw offset
tq=r['tx_qfq'].copy(); tq['date']=pd.to_datetime(tq.date); mm=m.merge(tq[['date','close']].rename(columns={'close':'txq'}),on='date'); mm['off']=mm.txq-mm.close
print(mm.groupby('off').size().head(12))
try:
    dv=ak.stock_history_dividend_detail(symbol='600188',indicator='分红'); print(dv.tail(12).to_string())
except Exception as e: print('div ERR',str(e)[:200])
