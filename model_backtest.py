"""最小化A股日频 ML 回测研究；只读CSV，不下单，不承诺预测准确。

信号收盘 t 生成；模拟 t+1 收盘建仓，t+6 收盘卖出（5个交易日持有）。
没有逐笔、涨跌停、停牌和流动性成交仿真，不可作为实盘收益承诺。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT=Path(__file__).resolve().parent
RAW=ROOT / "data" / "raw"
OUT=ROOT / "exports"
OUT.mkdir(exist_ok=True,parents=True)
HORIZON=5
FEATURES=["r1","r5","r20","r60","vol20","ma_gap5","ma_gap20","volume_ratio5","turnover5"]


def load_prices(folder: Path=RAW) -> pd.DataFrame:
    rows=[]
    for file in sorted(folder.glob("[0-9][0-9][0-9][0-9][0-9][0-9].csv")):
        f=pd.read_csv(file,dtype={"股票代码":str})
        rename={"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","成交额":"amount","换手率":"turnover"}
        f=f.rename(columns=rename)
        must={"date","close","high","low","open","volume"}
        if not must.issubset(f):
            raise ValueError(f"{file}: 缺失列 {must-set(f.columns)}")
        f["code"]=file.stem
        f["date"]=pd.to_datetime(f["date"],errors="coerce")
        for c in ["open","close","high","low","volume","turnover"]:
            if c not in f:f[c]=np.nan
            f[c]=pd.to_numeric(f[c],errors="coerce")
        f=f.dropna(subset=["date","close"]).sort_values("date").drop_duplicates("date",keep="last")
        if len(f)<180:
            print(f"[跳过] {file.stem}: 只有 {len(f)}条，至少需要180条用于基础特征")
            continue
        rows.append(f[["date","code","open","high","low","close","volume","turnover"]])
    if not rows:raise ValueError("没有可用行情！先运行 collect.py 或放入Choice格式已转换的CSV")
    return pd.concat(rows,ignore_index=True).sort_values(["code","date"]).reset_index(drop=True)


def make_panel(prices: pd.DataFrame,horizon=HORIZON) -> pd.DataFrame:
    parts=[]
    for code, g in prices.groupby("code",sort=False):
        f=g.copy().sort_values("date")
        c=f["close"]
        f["r1"]=c.pct_change(1,fill_method=None)
        f["r5"]=c.pct_change(5,fill_method=None)
        f["r20"]=c.pct_change(20,fill_method=None)
        f["r60"]=c.pct_change(60,fill_method=None)
        f["vol20"]=f["r1"].rolling(20).std()
        f["ma_gap5"]=c/c.rolling(5).mean()-1
        f["ma_gap20"]=c/c.rolling(20).mean()-1
        f["volume_ratio5"]=f["volume"]/f["volume"].rolling(5).mean().replace(0,np.nan)
        f["turnover5"]=f["turnover"].rolling(5).mean()
        f["future_buy_price"]=c.shift(-1)
        f["future_exit_price"]=c.shift(-(horizon+1))
        f["future_end_date"]=f["date"].shift(-(horizon+1))
        f["future_return"]=f["future_exit_price"]/f["future_buy_price"]-1
        f.replace([np.inf,-np.inf],np.nan,inplace=True)
        parts.append(f)
    panel=pd.concat(parts,ignore_index=True)
    panel["cross_mean_future_return"]=panel.groupby("date")["future_return"].transform("mean")
    panel["target_rel"]=panel["future_return"]-panel["cross_mean_future_return"]
    return panel.sort_values(["date","code"]).reset_index(drop=True)


def train_model(panel: pd.DataFrame, asof: pd.Timestamp):
    # future_end_date<asof: 严格屏蔽在当时尚未产生的真实收益标签
    hist=panel[(panel["future_end_date"]<asof) & panel["target_rel"].notna()].copy()
    hist=hist[hist["date"]>=asof-pd.Timedelta(days=4*365)]
    if len(hist)<450:raise ValueError(f"{asof.date()} 训练样本太少: {len(hist)}，请获取更长历史或扩充股票池")
    model=make_pipeline(SimpleImputer(strategy="median",add_indicator=False),HistGradientBoostingRegressor(
        max_iter=90,max_leaf_nodes=9,max_depth=3,min_samples_leaf=30,learning_rate=0.045,l2_regularization=0.4,random_state=42))
    model.fit(hist[FEATURES],hist["target_rel"])
    return model,len(hist)


def score_group(model,df: pd.DataFrame) -> pd.DataFrame:
    group=df.copy()
    # 至少60个历史交易日，降低新股/特征缺失影响
    group=group[group["r60"].notna() & group["r20"].notna() & group["vol20"].notna()]
    if group.empty:return group
    group["ml_score"]=model.predict(group[FEATURES])
    return group.sort_values("ml_score",ascending=False)


def portfolio_return(df: pd.DataFrame,score_col: str|None, k=3, fee=0.0025) -> tuple[float,list[str]]:
    if df.empty:return float("nan"),[]
    if score_col:
        chosen=df.sort_values(score_col,ascending=False).head(k)
    else:
        chosen=df
    return float(chosen["future_return"].mean()-fee),chosen["code"].to_list()


def metrics(series: pd.Series,horizon=HORIZON) -> dict:
    r=series.dropna().astype(float)
    if len(r)==0:return {"periods":0}
    equity=(1+r).cumprod()
    return {"periods":int(len(r)),"total_return":round(float(equity.iloc[-1]-1),5),
        "annualized_return":round(float(equity.iloc[-1]**(252/(len(r)*horizon))-1),5),
        "max_drawdown":round(float((equity/equity.cummax()-1).min()),5),
        "win_rate":round(float((r>0).mean()),5)}


def backtest(panel: pd.DataFrame,test_days=160,k=3,fee=0.0025,retrain_every=4) -> tuple[pd.DataFrame,dict]:
    dates=pd.DatetimeIndex(sorted(panel.date.unique()))
    if len(dates)<330:raise ValueError(f"只有{len(dates)}个交易日；建议下载至少两年以上的行情")
    start=max(160,len(dates)-test_days-HORIZON-1)
    test_signal_dates=dates[start:len(dates)-HORIZON-1:HORIZON]
    records=[]
    model=None
    for ix,dt in enumerate(test_signal_dates):
        # 每期重新拟合耗时；每4期(约20日)滚动重训，其余期使用已经训练的模型
        if ix%retrain_every==0 or model is None:
            model,rows=train_model(panel,dt)
        subset=panel[(panel.date==dt)&panel.future_return.notna()].copy()
        scored=score_group(model,subset)
        if len(scored)<min(3,k):continue
        ml,ml_codes=portfolio_return(scored,"ml_score",k,fee)
        momentum,mom_codes=portfolio_return(scored,"r20",k,fee)
        equal,eq_codes=portfolio_return(scored,None,k,fee)
        records.append({"signal_date":dt.strftime("%Y-%m-%d"),"execution":"信号次一交易日收盘价买入、五个交易日后收盘价卖出",
            "ML_net":ml,"momentum_net":momentum,"equal_universe_net":equal,"ML_codes":"|".join(ml_codes),
            "momentum_codes":"|".join(mom_codes),"candidates_count":len(scored)})
    if not records:raise ValueError("没有有效回测期：历史数据太短或股票池有效股票太少")
    bt=pd.DataFrame(records)
    perf={name:metrics(bt[name]) for name in ["ML_net","momentum_net","equal_universe_net"]}
    perf["backtest_first_signal"]=str(bt.signal_date.iloc[0])
    perf["backtest_last_signal"]=str(bt.signal_date.iloc[-1])
    perf["fee_roundtrip_assumption"]=fee
    perf["important_limitations"]=["使用当前静态股票池，存在幸存者偏差", "前复权历史数据可能存在非时点数据问题",
        "未模拟涨跌停、停牌、滑点变化、最小交易单位、成交能力", "成交采用次日收盘价，实盘执行可能不同", "模型只学习历史相对收益，不保证未来表现"]
    return bt,perf


def latest_candidates(panel:pd.DataFrame, model) -> pd.DataFrame:
    date=panel.date.max()
    scored=score_group(model,panel[panel.date==date])
    if scored.empty:return scored
    cols=["date","code","close","r5","r20","vol20","volume_ratio5","ml_score"]
    return scored[cols]


def main():
    p=argparse.ArgumentParser(description="基于A股历史日线的相对收益排序研究：无自动交易")
    p.add_argument("--raw",default="data/raw")
    p.add_argument("--test-days",type=int,default=160)
    p.add_argument("--top-k",type=int,default=3)
    p.add_argument("--fee-roundtrip",type=float,default=0.0025,help="含佣金/税费/滑点的简化往返成本假设")
    a=p.parse_args()
    if a.top_k<1: p.error("top-k必须 >=1")
    prices=load_prices(ROOT/a.raw)
    panel=make_panel(prices)
    universe_count=panel.code.nunique()
    print(f"数据: {len(panel):,}行，{universe_count}只，{panel.date.min().date()}—{panel.date.max().date()}")
    if universe_count<20:
        print("[重要警告] 当前股票池少于20只，结果仅用于教学；截图里9只房地产相关股属于严重行业集中。")
    back,perf=backtest(panel,test_days=a.test_days,k=a.top_k,fee=a.fee_roundtrip)
    back.to_csv(OUT/"walk_forward_returns.csv",index=False,encoding="utf-8-sig")
    with open(OUT/"backtest_metrics.json","w",encoding="utf-8") as f:json.dump(perf,f,ensure_ascii=False,indent=2)
    model,train_rows=train_model(panel,panel.date.max())
    latest=latest_candidates(panel,model)
    latest.to_csv(OUT/"research_candidates.csv",index=False,encoding="utf-8-sig")
    print("\n回测统计：\n"+json.dumps(perf,ensure_ascii=False,indent=2))
    print("\n最新研究排序(绝非买入建议)：\n"+latest.head(10).to_string(index=False))
    print("\n输出:",OUT)

if __name__=="__main__":main()
