# -*- coding: utf-8 -*-
"""P3-1c: data quality of the merged daily data as of the latest trading day -> data_quality_latest.csv / .md"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; V2 = H.parent
def main(date):
    z = np.load(H / "data" / f"world_{date}.npz"); codes = list(z["codes"]); cal = pd.DatetimeIndex(z["cal"]); T = len(cal); END = cal[-1]
    U = pd.read_csv(V2 / "data" / "sample_1500.csv", dtype=str).set_index("code"); ML = pd.read_csv(H / "data" / "merge_log.csv", dtype={"code": str}, encoding="utf-8-sig").set_index("code")
    info = json.load(open(H / "data" / "merge_info.json", encoding="utf-8"))
    HR, TR, V, LU, LD = z["has_row"], z["traded"], z["V"], z["lim_up"], z["lim_dn"]; cal = cal[-HR.shape[0]:]; T = len(cal)   # npz matrices hold the last 300 calendar rows
    o = pd.DataFrame(dict(code=codes, name=U.name.reindex(codes).values, status=U.status.reindex(codes).values))
    last = np.array([np.where(HR[:, j])[0][-1] if HR[:, j].any() else -1 for j in range(len(codes))])
    o["last_date"] = [str(cal[k].date()) if k >= 0 else "" for k in last]; o["stale"] = last < T - 1; o["suspended_or_no_bar_t"] = ~TR[-1]
    o["st_name"] = o.name.str.contains("ST"); o["delisted_status"] = o.status.astype(str).str.contains("终止")
    o["limit_up_t"] = LU[-1]; o["limit_down_t"] = LD[-1]
    med = np.nanmedian(np.where(V[-21:-1] > 0, V[-21:-1], np.nan), axis=0)
    with np.errstate(all="ignore"): ratio = V[-1] / med
    o["vol_ratio_t_vs_20d_median"] = ratio; o["volume_spike_gt5x"] = TR[-1] & (ratio > 5); o["volume_drop_lt0.2x"] = TR[-1] & (ratio < 0.2)
    o["new_rows_appended"] = ML.n_new_rows.reindex(codes).fillna(0).astype(int).values; o["fetched"] = ML.fetched.reindex(codes).astype(str).eq("True").values
    o["overlap_close_maxabs"] = ML.get("overlap_close_maxabs").reindex(codes).values if "overlap_close_maxabs" in ML else np.nan
    o["volume_mult"] = ML.get("volume_mult").reindex(codes).values if "volume_mult" in ML else np.nan
    o["factor_source"] = ML.factor_source.reindex(codes).values
    o.to_csv(H / "data_quality_latest.csv", index=False, encoding="utf-8-sig")
    live = ~o.delisted_status
    st = {"股票总数": len(o), "已终止上市(样本内历史退市)": int(o.delisted_status.sum()), "本次成功增量采集": int(o.fetched.sum()), "采集失败/未取到": int((~o.fetched).sum()),
          f"最后日期 = {END.date()}": int((~o.stale).sum()), f"过期(最后日期 < {END.date()},不含已退市)": int((o.stale & live).sum()), f"{END.date()} 停牌/无成交(不含已退市)": int((o.suspended_or_no_bar_t & live).sum()),
          "现名含 ST": int(o.st_name.sum()), f"{END.date()} 涨停封板": int(o.limit_up_t.sum()), f"{END.date()} 跌停封板": int(o.limit_down_t.sum()),
          "成交量 > 20日中位数 5 倍": int(o.volume_spike_gt5x.sum()), "成交量 < 20日中位数 0.2 倍": int(o["volume_drop_lt0.2x"].sum()),
          "重叠期(09-01~10-16)收盘价与 P0 不一致(>0.005元)": int((o.overlap_close_maxabs > 0.005).sum()), "成交量单位换算 x100(腾讯手->股)": int((o.volume_mult == 100).sum()),
          "使用最新新浪复权因子": int((o.factor_source == "latest_sina").sum())}
    L = [f"# 数据质量检查(截至 {END.date()})", "", f"- 新增交易日:{', '.join(info['new_trade_dates'])}(日历来源:{info['calendar_source']})", "- 口径:原始(不复权)价 + 新浪 qfq 复权因子,与 P0 相同;P0 原始文件只读,合并结果在 `data/merged/`。",
         "- 来源:新浪原始日线经本机代理频繁 SSL 中断,SH/SZ 改用 P0 已记录的备用源腾讯(`stock_zh_a_hist_tx`),北交所仍用新浪;与 P0 重叠期逐日核对收盘价/成交额,并用重叠期把腾讯成交量(手)换算为股、换手率按 成交量/流通股本 重算。", "",
         "| 项目 | 数量 |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in st.items()]
    bad = o[(o.stale & live) | (o.overlap_close_maxabs > 0.005)].head(40)
    L += ["", "## 过期或重叠期不一致的股票(前 40)", "", "| 代码 | 名称 | 最后日期 | 停牌 | 重叠期最大收盘差 |", "|---|---|---|---|---:|"] + [f"| {r.code} | {r.name} | {r.last_date} | {r.suspended_or_no_bar_t} | {r.overlap_close_maxabs} |" for r in bad.itertuples()]
    sp = o[o.volume_spike_gt5x].sort_values("vol_ratio_t_vs_20d_median", ascending=False).head(15)
    L += ["", "## 成交量异常放大(前 15)", "", "| 代码 | 名称 | 当日/20日中位数 |", "|---|---|---:|"] + [f"| {r.code} | {r.name} | {r.vol_ratio_t_vs_20d_median:.1f} |" for r in sp.itertuples()]
    (H / "data_quality_latest.md").write_text("\n".join(L) + "\n", encoding="utf-8"); json.dump(st, open(H / "results" / "data_quality_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(json.dumps(st, ensure_ascii=False))
if __name__ == "__main__": main(sys.argv[1] if len(sys.argv) > 1 else "20261021")