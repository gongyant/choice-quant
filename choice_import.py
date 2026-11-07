"""可选：将Choice终端合法导出的日线CSV文件统一到本项目格式。
需要至少 日期/代码/开盘/最高/最低/收盘/成交量 六列；列名可中英文。
不尝试登录、自动化破解终端或使用非公开API。
"""
import argparse
from pathlib import Path
import pandas as pd

RENAMES={"交易日期":"日期","date":"日期","证券代码":"股票代码","代码":"股票代码","code":"股票代码",
 "open":"开盘","high":"最高","low":"最低","close":"收盘","volume":"成交量","turnover":"换手率",
 "开盘价":"开盘","最高价":"最高","最低价":"最低","收盘价":"收盘","成交量(股)":"成交量"}
REQUIRED={"日期","股票代码","开盘","最高","最低","收盘","成交量"}

def normalize(infile:Path,out_dir:Path):
    df=pd.read_excel(infile,dtype=str) if infile.suffix.lower() in {".xlsx",".xls"} else pd.read_csv(infile,dtype=str,encoding="utf-8-sig")
    df=df.rename(columns=RENAMES)
    if not REQUIRED.issubset(df.columns):raise ValueError(f"文件缺少{REQUIRED-set(df.columns)}; 现有字段={list(df.columns)}")
    df["股票代码"]=df["股票代码"].astype(str).str.extract(r"(\d{6})",expand=False)
    df["日期"]=pd.to_datetime(df["日期"],errors="coerce").dt.strftime("%Y-%m-%d")
    df=df.dropna(subset=["日期","股票代码"])
    out_dir.mkdir(parents=True,exist_ok=True)
    for code,g in df.groupby("股票代码"):
        file=out_dir/f"{code}.csv"
        # 先导出中性格式；若Choice本身导出的是未复权，需要在导出阶段核对公司行为及口径
        g.sort_values("日期").to_csv(file,index=False,encoding="utf-8-sig")
        print("已转换:",file,len(g),"行")
    print("[注意] Choice导出文件的复权类型和成交量单位必须与AKShare数据保持一致，不可混用不同调整口径。")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--out",default="data/raw")
    a=p.parse_args()
    normalize(Path(a.input),Path(a.out))
