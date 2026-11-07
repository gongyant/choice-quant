"""P1 步骤1：构建并审计A股证券主表（只用公开接口，只读）。
输出 v2/data/universe_master.csv|parquet 与 v2/reports/universe_audit.json
"""
from __future__ import annotations
import json, time, random, datetime as dt
from pathlib import Path
import pandas as pd
import akshare as ak

V2 = Path(__file__).resolve().parent
DATA = V2 / "data"; REP = V2 / "reports"
DATA.mkdir(exist_ok=True); REP.mkdir(exist_ok=True)
NOW = dt.datetime.now().isoformat(timespec="seconds")

def retry(fn, *a, n=3, **k):
    for i in range(n):
        try:
            r = fn(*a, **k)
            if r is None or len(r) == 0: raise ValueError("empty")
            return r
        except Exception as e:
            if i + 1 == n: raise
            time.sleep(2 ** i + random.random())

def board_of(code: str, exch: str) -> str:
    if exch == "SH":
        if code.startswith("688") or code.startswith("689"): return "科创板"
        if code.startswith("900"): return "沪B股"
        if code.startswith(("600", "601", "603", "605")): return "沪主板"
        return "沪其他"
    if exch == "SZ":
        if code.startswith("200"): return "深B股"
        if code.startswith(("300", "301")): return "创业板"
        if code.startswith("002") or code.startswith("003") and False: return "深主板(原中小板)"
        if code.startswith(("000", "001", "003")): return "深主板"
        return "深其他"
    if exch == "BJ": return "北交所"
    return "未知"

def main():
    rows = []
    def add(df, code_col, name_col, list_col, exch, extra=None, status="上市", delist_col=None):
        for _, r in df.iterrows():
            code = str(r[code_col]).zfill(6)
            rows.append(dict(code=code, name=str(r[name_col]).strip(), exchange=exch,
                list_date=str(r[list_col])[:10] if list_col else None,
                delist_date=str(r[delist_col])[:10] if delist_col else None,
                status=status, board=board_of(code, exch),
                list_industry=(str(r[extra]).strip() if extra else None)))
    sh_main = retry(ak.stock_info_sh_name_code, symbol="主板A股")
    sh_star = retry(ak.stock_info_sh_name_code, symbol="科创板")
    sh_b = retry(ak.stock_info_sh_name_code, symbol="主板B股")
    sz_a = retry(ak.stock_info_sz_name_code, symbol="A股列表")
    sz_b = retry(ak.stock_info_sz_name_code, symbol="B股列表")
    bj = retry(ak.stock_info_bj_name_code)
    sh_del = retry(ak.stock_info_sh_delist)
    sz_del = retry(ak.stock_info_sz_delist, symbol="终止上市公司")
    for df in (sh_main, sh_star, sh_b): add(df, df.columns[0], df.columns[1], df.columns[5], "SH")
    add(sz_a, sz_a.columns[1], sz_a.columns[2], sz_a.columns[3], "SZ", extra=sz_a.columns[6])
    add(sz_b, sz_b.columns[1], sz_b.columns[2], sz_b.columns[3], "SZ", extra=sz_b.columns[6])
    add(bj, bj.columns[0], bj.columns[1], bj.columns[4], "BJ", extra=bj.columns[5])
    add(sh_del, sh_del.columns[0], sh_del.columns[1], sh_del.columns[2], "SH", status="已终止上市", delist_col=sh_del.columns[3])
    add(sz_del, sz_del.columns[0], sz_del.columns[1], sz_del.columns[2], "SZ", status="已终止上市", delist_col=sz_del.columns[3])
    u = pd.DataFrame(rows)
    u["is_b_share"] = u.board.isin(["沪B股", "深B股"])
    u["sec_type"] = "股票"
    u["is_st_name"] = u.name.str.contains(r"ST", case=False, regex=True)
    u["name_has_delist_mark"] = u.name.str.contains("退")
    u = u.drop_duplicates(["code", "exchange"], keep="first").reset_index(drop=True)
    # 交叉核对：AKShare 的 A 股代码名称总表
    audit = {"built_at": NOW}
    try:
        cn = retry(ak.stock_info_a_code_name)
        cn_codes = set(cn.iloc[:, 0].astype(str).str.zfill(6))
        listed_codes = set(u[(u.status == "上市")].code)
        audit["a_code_name_count"] = len(cn_codes)
        audit["in_a_code_name_not_in_master"] = sorted(cn_codes - listed_codes)[:50]
        audit["in_master_listed_not_in_a_code_name"] = sorted(listed_codes - cn_codes - set(u[u.is_b_share].code))[:50]
    except Exception as e:
        audit["a_code_name_error"] = str(e)[:200]
    # 实时快照(新浪)：最近交易日成交额，仅用于抽样分层，不作为特征
    try:
        sp = retry(ak.stock_zh_a_spot)
        sp = sp.rename(columns={sp.columns[0]: "sym", sp.columns[1]: "spot_name", sp.columns[2]: "spot_price"})
        amt_col = [c for c in sp.columns if str(c).startswith("成交额")][0]
        sp["code"] = sp.sym.str[-6:]
        sp = sp[["code", "spot_price", amt_col]].rename(columns={amt_col: "spot_amount"})
        sp = sp.drop_duplicates("code")
        sp.to_csv(DATA / f"spot_snapshot_{dt.date.today().isoformat()}.csv", index=False, encoding="utf-8-sig")
        u = u.merge(sp, on="code", how="left")
        audit["spot_rows"] = int(len(sp)); audit["spot_time"] = NOW
    except Exception as e:
        audit["spot_error"] = str(e)[:200]
    u["built_at"] = NOW
    u.to_csv(DATA / "universe_master_base.csv", index=False, encoding="utf-8-sig")
    audit["counts_by_board_status"] = {f"{b}|{s}": int(n) for (b, s), n in u.groupby(["board", "status"]).size().items()}
    audit["total_rows"] = int(len(u))
    audit["note"] = "清单为当前时点快照；退市名单只含官方交易所退市公告所列证券，仍不是完整历史股票池。ETF/基金/债券不在这些交易所股票清单中，因此未纳入。"
    with open(REP / "universe_audit.json", "w", encoding="utf-8") as f: json.dump(audit, f, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()