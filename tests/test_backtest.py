import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from model_backtest import make_panel,backtest,train_model,latest_candidates

def synthetic(seed=42,n=540,num=12):
    rng=np.random.default_rng(seed)
    dates=pd.bdate_range("2023-01-02",periods=n)
    parts=[]
    for i in range(num):
        returns=rng.normal(0,0.014,n)+0.00015*(i-6)
        close=18*np.exp(np.cumsum(returns))
        parts.append(pd.DataFrame({"code":f"{i+1:06d}","date":dates,"close":close,
             "open":close*(1+rng.normal(0,.004,n)),"high":close*1.025,"low":close*.975,
             "volume":rng.integers(500_000,4_000_000,n),"turnover":rng.uniform(.1,4,n)}))
    return pd.concat(parts,ignore_index=True)

def test_end_to_end():
    panel=make_panel(synthetic())
    last=panel.date.max()
    assert panel[panel.date==last].future_return.isna().all()
    assert panel[panel.date==last].future_end_date.isna().all()
    model,count=train_model(panel,last)
    assert count>=450
    result=latest_candidates(panel,model)
    assert len(result)==12
    bt,perf=backtest(panel,test_days=80,k=3)
    assert len(bt)>5
    assert all(np.isfinite(bt[["ML_net","momentum_net","equal_universe_net"]].to_numpy().ravel()))
    assert "ML_net" in perf

if __name__=="__main__":test_end_to_end();print("PASS")
