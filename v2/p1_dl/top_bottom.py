"""Top20 / Bottom20 research ranking (HGB and LGBM) for 2026-10-16, from latest_all_stocks_20261016.csv."""
import pandas as pd, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent
d = pd.read_csv(H/"latest_all_stocks_20261016.csv", dtype={"code": str}, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
e = d[d.eligible_asof.astype(str) == "True"].copy(); n = len(e)
def tab(model, which):
    s, r = f"{model}_score", f"rank_{model}"; t = e.sort_values(r, ascending=(which == "Top")).head(20)
    o = pd.DataFrame({"list": which, "model": model.upper(), "rank": t[r].astype(int).values, "code": t.code.values, "name": t.name.values, "industry": t.industry.values, "score": t[s].values,
                      "r5": t.r5.values, "r20": t.r20.values, "vol20": t.vol20.values, "amount20_yi": (t.amount20 / 1e8).values, "quality_flags": t.quality_flags.fillna("").values})
    return o
tabs = {(m, w): tab(m, w) for m in ["hgb", "lgbm"] for w in ["Top", "Bottom"]}
pd.concat(tabs.values()).to_csv(H/"reports"/"top_bottom_20261016.csv", index=False, encoding="utf-8-sig")
def md(t):
    L = ["| 排名 | 代码 | 名称 | 行业(巨潮最新) | score | 近5日 | 近20日 | 20日波动率 | 20日成交额(亿) | 质量标记 |", "|---:|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in t.itertuples():
        L.append(f"| {r.rank} | {r.code} | {r.name} | {r.industry} | {r.score:.6f} | {r.r5:+.1%} | {r.r20:+.1%} | {r.vol20:.2%} | {r.amount20_yi:.2f} | {r.quality_flags or '-'} |")
    return "\n".join(L)
ov_top = sorted(set(tabs[("hgb", "Top")].code) & set(tabs[("lgbm", "Top")].code)); ov_bot = sorted(set(tabs[("hgb", "Bottom")].code) & set(tabs[("lgbm", "Bottom")].code))
nm = dict(zip(d.code, d.name))
txt = [f"# 研究排名 Top20 / Bottom20(截至信号日 2026-10-16 收盘)", "",
       "> **仅为研究排名,不是买入/卖出建议。** score 是模型原始输出(对\"未来5个交易日去均值相对收益\"的回归拟合值),**未经校准,不是上涨概率,也不是收益率**。排名 1 = 模型认为未来5日相对表现最强。",
       f"> 排名范围:信号日合格股票 {n} 只(eligible_asof:当日有成交、已上市、20日成交额中位数≥2000万元、历史≥60日),来源样本为 1500 只分层抽样样本(含 208 只已退市,其中 2 只无数据),**不是全市场**。",
       f"> 模型:冻结的 HGB / LightGBM(P0 同参数,seed 42),训练区间 2023-10-17~2026-09-22,标签到期日最晚 2026-10-08,均早于信号日。行业为巨潮最新分类(非历史时点)。", "",
       f"两模型 Top20 重合 **{len(ov_top)}** 只:{', '.join(f'{c}{nm[c]}' for c in ov_top) or '无'};Bottom20 重合 **{len(ov_bot)}** 只:{', '.join(f'{c}{nm[c]}' for c in ov_bot) or '无'}。", ""]
for m, M in [("hgb", "HGB"), ("lgbm", "LightGBM")]:
    for w, cn in [("Top", "Top20(得分最高)"), ("Bottom", "Bottom20(得分最低)")]: txt += [f"## {M} {cn}", "", md(tabs[(m, w)]), ""]
rho = e[["rank_hgb", "rank_lgbm", "rank_ridge"]].astype(float).corr(method="spearman"); txt += ["## 三个模型排名的 Spearman 相关(合格股票)", "", rho.round(3).to_markdown(), ""]
(H/"reports"/"top_bottom_20261016.md").write_text("\n".join(txt), encoding="utf-8")
print("overlap top", len(ov_top), ov_top, "bottom", len(ov_bot), ov_bot); print(rho.round(3))

