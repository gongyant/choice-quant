"""P2：东方财富公开板块(行业/概念)目录与当日成分快照 + 同花顺板块目录(无成分接口)。
这些是供应商公开口径的“当前快照”，不是Choice板块，也不是历史成分：historical_valid=False，不得用于过去时点的训练特征。
动态板块(昨日连板/涨停等)按名称标记 is_dynamic=True。限速、重试、断点续传。
"""
import time, random, datetime as dt, re
from pathlib import Path
import pandas as pd, akshare as ak
V2 = Path(__file__).resolve().parent; B = V2 / "data" / "boards"; B.mkdir(parents=True, exist_ok=True)
NOW = dt.datetime.now().isoformat(timespec="seconds"); TODAY = dt.date.today().isoformat()
DYN = re.compile(r"昨日|今日|涨停|跌停|炸板|连板|首板|龙虎榜|百日新高|近期|创月新高|创年新高|融资融券|预盈|预增|预减|预亏|ST板块|B股|AB股|AH股|中盘|大盘|小盘")
def retry(fn, *a, n=3, **k):
    for i in range(n):
        try:
            r = fn(*a, **k)
            if r is None or len(r) == 0: raise ValueError("empty")
            return r
        except Exception:
            if i + 1 == n: raise
            time.sleep(3 * (i + 1) + random.random())
cat = []
for kind, f in [("industry", ak.stock_board_industry_name_em), ("concept", ak.stock_board_concept_name_em)]:
    try: d = retry(f)
    except Exception as e:
        print('东财目录失败(接口被拒/502/代理断开):', kind, type(e).__name__, flush=True); continue
    for r in d.itertuples(index=False):
        cat.append(dict(source="eastmoney.akshare_public", board_type="东财行业" if kind == "industry" else "东财概念", board_code=str(r[2]) if len(r) > 2 else "",
                        board_name=str(r[1]), captured_at=NOW, is_dynamic=bool(DYN.search(str(r[1]))), historical_valid=False, member_count_reported=int(r[-1]) if False else None))
for kind, f in [("industry", ak.stock_board_industry_summary_ths), ("concept", ak.stock_board_concept_name_ths)]:
    try:
        d = retry(f)
        name_col = "板块" if "板块" in d.columns else ("name" if "name" in d.columns else d.columns[0])
        code_col = "code" if "code" in d.columns else None
        for r in d.to_dict("records"):
            cat.append(dict(source="10jqka.akshare_public", board_type="同花顺行业" if kind == "industry" else "同花顺概念", board_code=str(r.get(code_col, "")) if code_col else "",
                            board_name=str(r[name_col]), captured_at=NOW, is_dynamic=bool(DYN.search(str(r[name_col]))), historical_valid=False, member_count_reported=None))
    except Exception as e:
        print("THS目录失败", kind, type(e).__name__)
catdf = pd.DataFrame(cat); catdf.to_csv(B / "board_catalog.csv", index=False, encoding="utf-8-sig")
print("目录条目", len(catdf), catdf.groupby("board_type").size().to_dict(), flush=True)
# 成分快照(仅东财)
out = B / "board_memberships.csv"
done = set(pd.read_csv(out, dtype=str).board_name + "|" + pd.read_csv(out, dtype=str).board_type) if out.exists() else set()
em = catdf[catdf.source.str.startswith('eastmoney')] if len(catdf) else catdf
fails = []; rows = []
for i, r in enumerate(em.itertuples(), 1):
    key = r.board_name + "|" + r.board_type
    if key in done: continue
    f = ak.stock_board_industry_cons_em if r.board_type == "东财行业" else ak.stock_board_concept_cons_em
    try:
        m = retry(f, symbol=r.board_name)
        code_col = [c for c in m.columns if "代码" in str(c)][0]; name_col = [c for c in m.columns if "名称" in str(c)][0]
        for x in m[[code_col, name_col]].itertuples(index=False):
            rows.append(dict(board_type=r.board_type, board_code=r.board_code, board_name=r.board_name, stock_code=str(x[0]).zfill(6), stock_name=x[1],
                             captured_at=NOW, source=r.source, is_dynamic=r.is_dynamic, effective_from="", effective_to="", historical_valid=False))
        if not len(m): fails.append(key)
    except Exception as e:
        fails.append(key + "|" + type(e).__name__)
    if len(rows) >= 2000 or i == len(em):
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False, encoding="utf-8-sig"); rows = []
    if i % 25 == 0: print(i, "/", len(em), "fails", len(fails), flush=True)
    time.sleep(1.3 + random.random() * 0.4)
if rows: pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False, encoding="utf-8-sig")
pd.Series(fails).to_csv(B / "board_member_failures.csv", index=False, header=["failed"], encoding="utf-8-sig")
print("done fails", len(fails), flush=True)