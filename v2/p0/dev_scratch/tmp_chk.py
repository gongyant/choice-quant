import pandas as pd, numpy as np, sys
sys.path.insert(0,'.'); import p0lib as L
o=pd.read_csv('corporate_action_crosscheck.csv',dtype={'code':str}); print(o[~o.factor_present_on_ex_date][['code','ex_date','bonus_per10','transfer_per10','cash_div_per10','prev_raw_close']].to_string())
r=L.load_raw('600188').set_index('date')
for d in ['2021-07-23','2022-07-14','2023-07-17']:
    i=r.index.get_loc(pd.Timestamp(d)); row=r.iloc[i]; prev=r.close.iloc[i-1]
    print(d,'prev',prev,'open',row.open,'high',row.high,'low',row.low,'close',row.close)
