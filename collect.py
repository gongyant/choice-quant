"""从AKShare公开接口拉取A股日行情和Choice概念/行业板块目录，支持本地缓存。

不要用于绕过Choice认证、反爬措施或未授权复制付费数据库。
接口可变；运行前自行核查数据服务许可和最新文档。
"""
from __future__ import annotations
import argparse
import random
import time
from pathlib import Path
from datetime import date
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def fetch_with_retry(fn, *args, retries=3, **kwargs):
    for attempt in range(retries):
        try:
            result = fn(*args, **kwargs)
            if result is None or result.empty:
                raise ValueError("接口返回空表")
            return result
        except Exception as exc:
            print(f"  [警告] 第{attempt+1}/{retries}次失败: {exc}")
            if attempt + 1 < retries:
                time.sleep(min(15, 2 ** attempt + random.random()))
    raise RuntimeError("接口请求失败，已重试；请检查网络、接口或访问许可")


def codes_from_csv(path: Path) -> list[str]:
    df = pd.read_csv(path, dtype={"code": str})
    if "code" not in df:
        raise ValueError("universe.csv必须有code列")
    return sorted({str(x).zfill(6) for x in df.code.dropna() if str(x).strip()})


def collect_prices(codes: list[str], start: str, end: str, pause: float, force: bool) -> tuple[int,int]:
    import akshare as ak
    ok = 0
    failed = 0
    for i, code in enumerate(codes, 1):
        target = RAW / f"{code}.csv"
        if target.exists() and not force:
            saved = pd.read_csv(target)
            if not saved.empty and str(saved['日期'].iloc[0]).replace('-', '') <= start and str(saved['日期'].iloc[-1]).replace('-', '') >= end:
                print(f"[{i}/{len(codes)}] {code}: 缓存命中")
                ok += 1
                continue
        print(f"[{i}/{len(codes)}] {code}: 正在拉取 {start}—{end}")
        try:
            df = fetch_with_retry(ak.stock_zh_a_hist, symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
            need = {"日期", "开盘", "收盘", "最高", "最低", "成交量"}
            if not need.issubset(df.columns):
                raise ValueError(f"缺少必需列: {need - set(df.columns)}，实际列={list(df.columns)}")
            df = df.sort_values("日期").drop_duplicates("日期")
            df.to_csv(target, index=False, encoding="utf-8-sig")
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"  [失败] {code}: {exc}")
        time.sleep(max(0, pause))
    return ok, failed


def collect_boards(pause: float, max_boards: int) -> None:
    import akshare as ak
    out_dir = ROOT / "data" / "boards"
    out_dir.mkdir(exist_ok=True, parents=True)
    stamp = date.today().isoformat()
    for kind, list_func, member_func in [
        ("industry", ak.stock_board_industry_name_em, ak.stock_board_industry_cons_em),
        ("concept", ak.stock_board_concept_name_em, ak.stock_board_concept_cons_em),
    ]:
        try:
            board_df = fetch_with_retry(list_func)
            board_df["captured_date"] = stamp
            board_df.to_csv(out_dir / f"{kind}_board_list_{stamp}.csv", index=False, encoding="utf-8-sig")
            if "板块名称" not in board_df.columns:
                print(f"[提示] {kind}目录已保存，但无板块名称列，跳过成员采集")
                continue
            # 默认只抓少量板块；全量抓取必须由用户显式 --max-boards 0 开启，并检查服务条款
            subset = board_df if max_boards == 0 else board_df.head(max_boards)
            combined = []
            for board in subset["板块名称"].dropna().astype(str).unique():
                try:
                    members = fetch_with_retry(member_func, symbol=board)
                    members["board_name"] = board
                    members["board_type"] = kind
                    members["captured_date"] = stamp
                    combined.append(members)
                except Exception as exc:
                    print(f"  [失败] {kind}板块 {board}: {exc}")
                time.sleep(max(0, pause))
            if combined:
                pd.concat(combined, ignore_index=True).to_csv(out_dir / f"{kind}_members_snapshot_{stamp}.csv", index=False, encoding="utf-8-sig")
        except Exception as exc:
            print(f"[失败] {kind}板块目录: {exc}")


def main():
    p = argparse.ArgumentParser(description="合法公开接口采集行情；不涉及账户登录或实盘交易")
    p.add_argument("--universe", default="universe.csv")
    p.add_argument("--start", default="20220101")
    p.add_argument("--end", default=date.today().strftime("%Y%m%d"))
    p.add_argument("--pause", type=float, default=1.2, help="请求节奏，默认1.2秒")
    p.add_argument("--force", action="store_true", help="覆盖本地数据（可能重复下载）")
    p.add_argument("--boards", action="store_true", help="额外获取板块目录及部分成分股")
    p.add_argument("--max-boards", type=int, default=5, help="各类前N个板块，0为全量，请确认授权与限速")
    a = p.parse_args()
    if a.start > a.end:
        p.error("start不得晚于end")
    if a.pause < 0.5:
        p.error("pause不得小于0.5秒，以减少公开服务压力")
    codes = codes_from_csv(ROOT / a.universe)
    print(f"股票池: {len(codes)}只；注意此示例房地产占比很高，无法代表整个A股市场")
    ok, failed = collect_prices(codes, a.start, a.end, a.pause, a.force)
    if a.boards:
        collect_boards(a.pause, a.max_boards)
    print(f"下载完成：成功/缓存={ok}，失败={failed}；输出={RAW}")
    if failed: raise SystemExit(2)

if __name__ == "__main__":
    main()
