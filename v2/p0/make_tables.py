"""Render before/after markdown tables from comparative_metrics.csv / e1 / cost sensitivity."""
import pandas as pd, numpy as np, sys
from pathlib import Path
P0 = Path(__file__).resolve().parent
cm = pd.read_csv(P0 / "comparative_metrics.csv")
def f(x, d=3, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return "-"
    return f"{x*100:.1f}%" if pct else f"{x:.{d}f}"
def table(df, title):
    h = "| 版本 | 策略 | 年化 | 夏普 | 最大回撤 | RankIC | 净超额均值/期 | 常规t | HAC4 t | Bootstrap 95%区间(净超额/期) | 换手(替换比例) | 买单成交率 |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    for _, r in df.iterrows():
        ci = f"[{r.net_ex_boot_lo*100:.2f}%, {r.net_ex_boot_hi*100:.2f}%]" if pd.notna(r.get("net_ex_boot_lo")) else "-"
        h += f"| {r.version} | {r.strategy} | {f(r.annualized_return,pct=True)} | {f(r.sharpe,2)} | {f(r.max_drawdown,pct=True)} | {f(r.RankIC,4)} | {f(r.net_ex_excess_mean*100 if pd.notna(r.get('net_ex_excess_mean')) else np.nan,3)}% | {f(r.net_ex_t,2)} | {f(r.net_ex_hac4_t,2)} | {ci} | {f(r.turnover_replaced,2)} | {f(r.buy_fill_rate,3)} |\n"
    return f"### {title}\n\n" + h + "\n"
order = ["Ridge", "HGB", "LGBM", "Momentum20", "Reversal5", "EqualWeight"]
txt = ""
for v, title in [("OLD", "旧版(INVALID_BEFORE_FIX，仅作对照)"), ("FIXED", "修复版(30bps往返成本，同制度等权基准)")]:
    d = cm[cm.version.str.startswith(v)].copy(); d["o"] = d.strategy.map(order.index); txt += table(d.sort_values("o"), title)
(P0 / "before_after_table.md").write_text(txt, encoding="utf-8"); print(txt)
