import pandas as pd, numpy as np, pickle
pd.set_option('display.width',250); pd.set_option('display.max_columns',50)
out=pickle.load(open('p0/data/probe3.pkl','rb'))
for code,r in out.items():
    print('=====',code)
    tr,tq,sr,sq=r['tx_raw'],r['tx_qfq'],r['sina_raw'],r['sina_qfq']
    for d in (tr,tq,sr,sq): d['date']=pd.to_datetime(d['date'])
    m=tr[['date','close']].merge(tq[['date','close']],on='date',suffixes=('_txraw','_txqfq')).merge(sr[['date','close']].rename(columns={'close':'close_sraw'}),on='date').merge(sq[['date','close']].rename(columns={'close':'close_sqfq'}),on='date')
    print(m.head(3).to_string()); print(m.tail(2).to_string())
    print('raw tx vs sina raw maxabs diff', (m.close_txraw-m.close_sraw).abs().max())
    print('tx qfq <=0:', (tq[['open','high','low','close']]<=0).any(axis=1).sum(), ' sina qfq<=0:', (sq[['open','high','low','close']]<=0).any(axis=1).sum())
    print('sina factors'); print(r['sina_qfq-factor'].head(40).to_string())
    for nm,d in [('tx_raw',tr),('tx_qfq',tq),('sina_raw',sr),('sina_qfq',sq)]:
        ret=d.sort_values('date').close.pct_change(); print(nm,'max|ret|',ret.abs().max(), 'n>0.25',(ret.abs()>0.25).sum(), 'n>0.11',(ret.abs()>0.11).sum())
