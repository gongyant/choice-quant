"""P1-A3 step 1: latest_all_stocks_20261016_v2.csv (A1 table + DL scores + model agreement) and reports/top_bottom_20261016_v2.{md,csv}. Original A1 files are NOT modified."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent.parent
rd = lambda p: pd.read_csv(p, dtype=str, encoding="utf-8-sig", keep_default_na=False)
A1 = rd(H / "latest_all_stocks_20261016.csv"); assert len(A1) == 1500
DLF = {"mlp": ("dl_mlp", "mlp_score_mean", "mlp_latest"), "gru": ("dl_gru", "gru_score_mean", "gru_latest"), "tf": ("dl_tf", "tf_score_mean", "tf_latest")}
V = A1.copy(); info = {}
for k, (nm, sc, mid) in DLF.items():
    d = pd.read_csv(H / "scores" / f"dl_{k}_scores_20261016.csv", dtype={"code": str}); assert d.model_id.eq(mid).all() and d.code.is_unique
    d = d.set_index("code"); el_a1 = set(A1.code[A1.eligible_asof == "True"]); el_dl = set(d.index[d.eligible_asof.astype(str) == "True"])
    assert el_a1 == el_dl, (k, len(el_a1 ^ el_dl)); assert d["rank"].notna().sum() == len(el_a1) and set(d.index[d["rank"].notna()]) == el_a1
    sc_s = V.code.map(d[sc]); rk = V.code.map(d["rank"]); V[f"{nm}_score"] = np.where(V.code.isin(el_a1), sc_s.map(lambda x: "" if pd.isna(x) else repr(float(x))), "")
    # all codes present in DL file get their raw score even if ineligible? -> A1 convention: scores only for eligible rows
    V[f"rank_{nm}"] = np.where(V.code.isin(el_a1), rk.map(lambda x: "" if pd.isna(x) else str(int(x))), "")
    n = len(el_a1); r = pd.to_numeric(V[f"rank_{nm}"]); V[f"pctile_{nm}"] = (1 - (r - 1) / (n - 1)).map(lambda x: "" if pd.isna(x) else repr(float(x)))
    # rank 1 must be the highest score
    e = V[V.eligible_asof == "True"]; s = pd.to_numeric(e[f"{nm}_score"]); assert (pd.to_numeric(e[f"rank_{nm}"]) == s.rank(ascending=False, method="min")).all(), k
    info[k] = dict(file=f"scores/dl_{k}_scores_20261016.csv", n_file_rows=len(d), n_eligible=len(el_dl), missing_codes_vs_sample=sorted(set(A1.code) - set(d.index)), model_id=mid, train_end=str(d.train_end.iloc[0]), val_end=str(d.val_end.iloc[0]), max_label_expiry=str(d.max_label_expiry.iloc[0]))
# model agreement
five = ["hgb", "lgbm", "dl_mlp", "dl_gru", "dl_tf"]
P = V[[f"pctile_{m}" for m in five]].apply(pd.to_numeric, errors="coerce"); R = V[[f"rank_{m}" for m in five]].apply(pd.to_numeric, errors="coerce")
V["model_agreement_pctile_std"] = P.std(axis=1, ddof=0).map(lambda x: "" if pd.isna(x) else repr(round(float(x), 6)))
V["model_pctile_mean_5"] = P.mean(axis=1).map(lambda x: "" if pd.isna(x) else repr(round(float(x), 6)))
V["n_models_top50_of5"] = np.where(V.eligible_asof == "True", (R <= 50).sum(axis=1).astype(str), "")
V["models_in_top50"] = ["" if el != "True" else "+".join(n.upper().replace("DL_", "") for n, x in zip(five, row) if x <= 50) for el, row in zip(V.eligible_asof, R.values)]
V["model_version_dl"] = "mlp_latest(3seeds mean)|gru_latest(3seeds mean)|tf_latest(seed42 only); train_end 2026-06-16; val_end 2026-09-22; max_label_expiry 2026-10-08"
V["agreement_note"] = "model_agreement_pctile_std=5个模型(HGB,LGBM,MLP,GRU,TF)百分位的总体标准差;越小=各模型排名越一致,越大=分歧越大;不是置信度/概率;Ridge 不参与"
# column order: A1 columns then new
newc = [c for c in V.columns if c not in A1.columns]; assert not any(("pred_excess" in c) or ("p_outperform" in c) or ("prob" in c.lower()) for c in V.columns)
cols = list(A1.columns) + ["rank_dl_mlp", "rank_dl_tf", "pctile_dl_mlp", "pctile_dl_gru", "pctile_dl_tf", "dl_tf_score"]  # placeholders (dedup below)
order = []
for c in list(A1.columns) + [c for c in V.columns if c not in A1.columns]:
    if c not in order: order.append(c)
V = V[order]; V.to_csv(H / "latest_all_stocks_20261016_v2.csv", index=False, encoding="utf-8-sig")
# ---- verify ----
chk = rd(H / "latest_all_stocks_20261016_v2.csv"); assert len(chk) == 1500 and chk.code.is_unique and chk.code.str.len().eq(6).all() and (chk.eligible_asof == A1.eligible_asof).all() and (chk.eligible_asof == "True").sum() == 1178
print("v2 table OK rows", len(chk), "cols", chk.shape[1], "eligible", (chk.eligible_asof == "True").sum()); print(json.dumps(info, ensure_ascii=False))
json.dump(info, open(H / "scores" / "latest_v2_build_info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
# ---- top / bottom v2 ----
e = chk[chk.eligible_asof == "True"].copy(); n = len(e)
for c in e.columns:
    if c.startswith(("rank_", "pctile_", "r5", "r20", "r60", "vol20", "amount20")) or c.endswith("_score") or c in ("model_agreement_pctile_std", "model_pctile_mean_5", "n_models_top50_of5"): e[c] = pd.to_numeric(e[c])
MN = {"hgb": "HGB", "lgbm": "LightGBM", "dl_mlp": "MLP", "dl_gru": "GRU", "dl_tf": "Transformer(单种子42)"}
def tab(m, which):
    s, r = f"{m}_score", f"rank_{m}"; t = e.sort_values(r, ascending=(which == "Top")).head(20)
    return pd.DataFrame({"list": which, "model": MN[m], "rank": t[r].astype(int).values, "code": t.code.values, "name": t.name.values, "industry": t.industry.values, "score": t[s].values, "r5": t.r5.values, "r20": t.r20.values, "vol20": t.vol20.values, "amount20_yi": (t.amount20 / 1e8).values,
                         "n_models_top50_of5": t.n_models_top50_of5.astype(int).values, "quality_flags": t.quality_flags.values})
tabs = {(m, w): tab(m, w) for m in five for w in ["Top", "Bottom"]}
pd.concat(tabs.values()).to_csv(H / "reports" / "top_bottom_20261016_v2.csv", index=False, encoding="utf-8-sig")
def md(t):
    L = ["| 排名 | 代码 | 名称 | 行业(巨潮最新) | score | 近5日 | 近20日 | 20日波动率 | 20日成交额(亿) | 5模型Top50计数 | 质量标记 |", "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in t.itertuples():
        fl = "; ".join(x for x in r.quality_flags.split(";") if x and ("涨停" in x or "跌停" in x or "ST" in x or "单源" in x or "未解释" in x or "无法核验" in x)) or "-"
        L.append(f"| {r.rank} | {r.code} | {r.name} | {r.industry} | {r.score:.6f} | {r.r5:+.1%} | {r.r20:+.1%} | {r.vol20:.2%} | {r.amount20_yi:.2f} | {r.n_models_top50_of5} | {fl} |")
    return "\n".join(L)
top20 = {m: list(tabs[(m, "Top")].code) for m in five}; bot20 = {m: list(tabs[(m, "Bottom")].code) for m in five}
def ovm(d):
    M = pd.DataFrame(index=[MN[a].split("(")[0] for a in five], columns=[MN[a].split("(")[0] for a in five], dtype=int)
    for a in five:
        for b in five: M.loc[MN[a].split("(")[0], MN[b].split("(")[0]] = len(set(d[a]) & set(d[b]))
    return M
nm = dict(zip(e.code, e.name))
ge3 = e[e.n_models_top50_of5 >= 3].sort_values(["n_models_top50_of5", "model_pctile_mean_5"], ascending=[False, False])
ge3_t = pd.DataFrame({"code": ge3.code, "name": ge3.name, "industry": ge3.industry, "n_models_top50_of5": ge3.n_models_top50_of5.astype(int), "models_in_top50": ge3.models_in_top50, "rank_hgb": ge3.rank_hgb.astype(int), "rank_lgbm": ge3.rank_lgbm.astype(int), "rank_mlp": ge3.rank_dl_mlp.astype(int), "rank_gru": ge3.rank_dl_gru.astype(int), "rank_tf": ge3.rank_dl_tf.astype(int),
                      "model_agreement_pctile_std": ge3.model_agreement_pctile_std, "r5": ge3.r5, "r20": ge3.r20, "limit_up_locked_latest": ge3.quality_flags.str.contains("最新日涨停封板").map({True: "是(最新日涨停封板,次日大概率无法买入)", False: "否"}), "quality_flags": ge3.quality_flags})
ge3_t.to_csv(H / "reports" / "multi_model_top50_20261016.csv", index=False, encoding="utf-8-sig")
txt = ["# 研究排名 Top20 / Bottom20 —— 五模型版(截至信号日 2026-10-16 收盘)", "",
       "> **仅为研究排名,不是买入/卖出建议。** score 是模型原始输出,**未校准,不是上涨概率,也不是收益率**。排名 1 = 模型认为未来5日相对表现最强。",
       f"> 排名范围:信号日合格股票 {n} 只(来自 1500 只分层抽样样本,**非全市场**)。HGB/LightGBM = P0 冻结模型(训练 2023-10-17~2026-09-22);MLP(3种子均值)、GRU(3种子均值,GPU训练)、Transformer(**仅种子42**)= 轨道B 最新模型(训练截至 2026-06-16,验证截至 2026-09-22,标签最晚到期 2026-10-08,均早于信号日)。",
       "> 各模型在合格池内的排名互相独立计算;\"5模型Top50计数\" = 该股同时进入 HGB/LGBM/MLP/GRU/TF 各自 Top50 的模型个数。涨停封板=最新日(2026-10-16)收盘封板,次日开盘大概率买不到。", ""]
for m in five:
    for w, cn in [("Top", "Top20(得分最高)"), ("Bottom", "Bottom20(得分最低)")]: txt += [f"## {MN[m]} {cn}", "", md(tabs[(m, w)]), ""]
txt += ["## 五模型 Top20 两两重合数(对角线=20)", "", ovm(top20).to_markdown(), "", "## 五模型 Bottom20 两两重合数", "", ovm(bot20).to_markdown(), ""]
rk = e[[f"rank_{m}" for m in five]].astype(float); rk.columns = [MN[m].split("(")[0] for m in five]; txt += ["## 五模型排名的 Spearman 相关(合格股票)", "", rk.corr(method="spearman").round(3).to_markdown(), ""]
txt += [f"## 至少 3 个模型同时进入 Top50 的股票(共 {len(ge3_t)} 只;仅为模型一致性描述,**不是买入建议**)", "",
        "| 代码 | 名称 | 行业 | 模型数 | 入选模型 | HGB | LGBM | MLP | GRU | TF | 分歧(std) | 近5日 | 近20日 | 最新日涨停封板 |", "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
for r in ge3_t.itertuples(): txt.append(f"| {r.code} | {r.name} | {r.industry} | {r.n_models_top50_of5} | {r.models_in_top50} | {r.rank_hgb} | {r.rank_lgbm} | {r.rank_mlp} | {r.rank_gru} | {r.rank_tf} | {float(r.model_agreement_pctile_std):.3f} | {r.r5:+.1%} | {r.r20:+.1%} | {r.limit_up_locked_latest} |")
txt += ["", "说明:model_agreement_pctile_std 为 5 个模型百分位的标准差(越小越一致),只描述模型之间的分歧,**不是置信度**;模型彼此输入(同一组量价因子/序列)高度相关,一致不等于独立验证。"]
(H / "reports" / "top_bottom_20261016_v2.md").write_text("\n".join(txt), encoding="utf-8")
print(ovm(top20)); print(ovm(bot20)); print(rk.corr(method="spearman").round(3)); print("n>=3:", len(ge3_t)); print(ge3_t[["code", "name", "n_models_top50_of5", "models_in_top50", "limit_up_locked_latest"]].to_string())
for m in five: print(MN[m]); print(tabs[(m, "Top")][["rank", "code", "name", "industry", "score"]].to_string(index=False))

