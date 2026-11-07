# -*- coding: utf-8 -*-
"""Timing diagnostic: why does the new definition change RankIC? Decompose by entry/exit execution status (all statuses are execution-day facts, used only to DEFINE realised returns)."""
import numpy as np, pandas as pd
import p2lib as Q
P = pd.read_parquet(Q.RES / "p2a_panel.parquet"); Pe = P[P.eligible & P.ret_new_rel.notna()].copy()
Pe["exit_regular"] = (Pe.hold_rows == 5) & ~Pe.exit_forced
Pe["exit_delayed"] = (Pe.hold_rows > 5) & ~Pe.exit_forced
cls = {"entry正常&exit正常(与旧口径同一收益)": (Pe.entry_delay == 0) & Pe.exit_regular, "entry被推迟(t+1涨停封板/停牌)": Pe.entry_delay > 0, "exit被推迟(到期日跌停封板/停牌)": (Pe.entry_delay == 0) & Pe.exit_delayed, "强制退出(超10日仍不可卖)": Pe.exit_forced}
rows = []
for k, msk in cls.items():
    d = Pe[msk]
    for m in Q.MODELS:
        ics = pd.Series({i: Q.spearman(x["score_" + m].values, x.ret_new_rel.values) for i, x in d.groupby("i") if len(x) >= 8})
        rows.append(dict(segment=k, model=m, n_rows=len(d), share=len(d) / len(Pe), mean_ret_new=float(d.ret_new.mean()), mean_ret_old=float(d.ret_old.mean()), mean_rank_pctile=float(d["score_" + m].rank(pct=True).mean()),
                         pooled_spearman=Q.spearman(d["score_" + m].values, d.ret_new_rel.values)))
T = pd.DataFrame(rows); T.to_csv(Q.RES / "timing_diagnostic.csv", index=False, encoding="utf-8-sig")
# per date IC contributions: clean-timing-only IC
clean = cls["entry正常&exit正常(与旧口径同一收益)"]
rows = []
for m in Q.MODELS:
    d = Pe[clean]; s = pd.Series({i: Q.spearman(x["score_" + m].values, x.ret_new_rel.values) for i, x in d.groupby("i") if len(x) >= 8})
    rows.append({"model": m, "sample": "仅正常执行行(entry/exit均无推迟)", **Q.summarize(s.values)})
pd.DataFrame(rows).to_csv(Q.RES / "rankic_clean_timing_rows.csv", index=False, encoding="utf-8-sig")
print(T[T.model == "HGB"].round(4).to_string()); print(pd.DataFrame(rows)[["model", "mean", "hac_t"]].round(4))
print("exit_delayed share", Pe.exit_delayed.mean(), "forced", Pe.exit_forced.mean(), "entry delayed", (Pe.entry_delay > 0).mean())