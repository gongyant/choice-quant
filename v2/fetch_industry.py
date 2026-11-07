"""P1 步骤2：用巨潮(cninfo)公开接口取每只证券的行业分类(巨潮行业分类标准)，断点续传，4线程限速。
注意：取的是最新生效分类，不是历史时点分类，只用于抽样分层和展示，不能当作历史有效行业。
"""
import time, random, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, akshare as ak
V2 = Path(__file__).resolve().parent; DATA = V2 / "data"
out = DATA / "industry_cninfo.csv"
u = pd.read_csv(DATA / "universe_master_base.csv", dtype={"code": str})
u = u[~u.is_b_share]
u["delist_date"] = pd.to_datetime(u.delist_date, errors="coerce")
u = u[(u.status == "上市") | (u.delist_date >= "2021-01-01")]
done = set()
if out.exists(): done = set(pd.read_csv(out, dtype={"code": str}).code)
codes = [c for c in u.code if c not in done]
print("待取", len(codes), "已有", len(done), flush=True)
lock = threading.Lock(); rows = []; cnt = {"n": 0, "f": 0}
def one(c):
    ok = False; rec = None
    for a in range(3):
        try:
            d = ak.stock_industry_change_cninfo(symbol=c, start_date="19900101", end_date="20261016")
            d = d[d["分类标准"] == "巨潮行业分类标准"].sort_values("变更日期")
            if len(d):
                r = d.iloc[-1]
                rec = dict(code=c, industry_l1=r["行业门类"], industry_l2=r["行业大类"], industry_l3=r["行业次类"], industry_code=r["行业编码"],
                           industry_change_date=str(r["变更日期"])[:10], industry_source="巨潮行业分类标准(最新生效)")
            else: rec = dict(code=c, industry_l1=None, industry_source="cninfo无巨潮分类记录")
            ok = True; break
        except Exception:
            time.sleep(2 ** a + random.random())
    if not ok: rec = dict(code=c, industry_l1=None, industry_source="cninfo请求失败")
    with lock:
        rows.append(rec); cnt["n"] += 1; cnt["f"] += (not ok)
        if cnt["n"] % 100 == 0:
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False, encoding="utf-8-sig"); rows.clear()
            print(cnt["n"], "/", len(codes), "fails", cnt["f"], flush=True)
    time.sleep(0.7)
with ThreadPoolExecutor(3) as ex: list(ex.map(one, codes))
if rows: pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False, encoding="utf-8-sig")
print("done", cnt, flush=True)