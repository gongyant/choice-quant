import pandas as pd, numpy as np, json
pd.set_option('display.width',250); pd.set_option('display.max_columns',50)
bt=pd.read_csv('results/baseline/walk_forward_returns.csv'); print(bt.shape, bt.columns.tolist()[:12])
for n in ['Ridge','HGB','LGBM','Momentum20','Reversal5']:
    ex=bt[n+'_net']-bt['EqualWeight_net']; t=ex.mean()/(ex.std()/np.sqrt(len(ex)))
    print(n, 'net-ex mean',round(ex.mean(),6),'t',round(t,3))
m=json.load(open('results/baseline/backtest_metrics.json',encoding='utf-8')); print({k:(v.get('annualized_return'),v.get('excess_tstat')) for k,v in m['strategies'].items()})
q=pd.read_csv('reports/quality_per_stock.csv',dtype={'code':str}); print(q.columns.tolist()); print(q.extreme_returns.sum(), (q.extreme_returns>0).sum())
print(q[q.code.isin(['600188','600066','000408','600096','000933','601666','920196','600317','689009'])][['code','name','rows','first','last','extreme_returns','max_abs_ret','nonpos_price','flags','volume_mult_by_year']].to_string())
p=pd.read_parquet('data/panel_1500.parquet'); print(p.shape, p.columns.tolist()); print(p.date.min(),p.date.max())
print(p.board.unique())
