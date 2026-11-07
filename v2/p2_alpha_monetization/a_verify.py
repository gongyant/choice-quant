# -*- coding: utf-8 -*-
"""P2-A item 1: verify frozen prediction files (read-only). Writes a_verify.json / a_verify.md"""
import json, re
import numpy as np, pandas as pd
import p2lib as Q
out = {"checks": [], "inconsistencies": []}
def chk(name, ok, detail=""):
    out["checks"].append(dict(name=name, ok=bool(ok), detail=str(detail)))
    if not ok: out["inconsistencies"].append(f"{name}: {detail}")
hp = pd.read_parquet(Q.HP_ALL)
P = pd.read_parquet(Q.RES / "p2a_panel.parquet")
out["files"] = {"historical_predictions_all_models.parquet": dict(path=str(Q.HP_ALL), bytes=Q.HP_ALL.stat().st_size, sha256=Q.sha256(Q.HP_ALL), rows=len(hp), columns=list(hp.columns))}
chk("总行数=1,860,516", len(hp) == 1860516, len(hp))
chk("键(signal_date,code,model_id)唯一", not hp.duplicated(["signal_date", "code", "model_id"]).any())
chk("键(signal_date,code,model)唯一", not hp.duplicated(["signal_date", "code", "model"]).any())
chk("模型集合=6", sorted(hp.model.unique()) == sorted(Q.MODELS), sorted(hp.model.unique()))
rows_pm = hp.groupby("model").size().to_dict(); out["rows_per_model"] = rows_pm
chk("每模型 310,086 行", all(v == 310086 for v in rows_pm.values()), rows_pm)
chk("代码均为6位字符串(前导零保留)", hp.code.map(lambda s: isinstance(s, str) and re.fullmatch(r"\d{6}", s) is not None).all())
lead0 = int(hp.drop_duplicates("code").code.str.startswith("0").sum()); out["codes_unique"] = int(hp.code.nunique()); out["codes_with_leading_zero"] = lead0
chk("以0开头代码存在且未丢零", lead0 > 400, lead0)
sd = {m: sorted(g.signal_date.unique()) for m, g in hp.groupby("model")}
chk("六模型 signal_date 集合相同且=207", all(sd[m] == sd["HGB"] for m in sd) and len(sd["HGB"]) == 207, len(sd["HGB"]))
out["signal_range"] = [sd["HGB"][0], sd["HGB"][-1]]
spacing = pd.Series(pd.to_datetime(sd["HGB"])).diff().dt.days.dropna(); out["signal_spacing_days"] = dict(min=int(spacing.min()), max=int(spacing.max()))
# signal spacing in trading days using the P0 calendar
X, W = Q.build_world(); cal = X.cal
ci = cal.get_indexer(pd.to_datetime(sd["HGB"])); chk("signal_date 都在 P0 主日历内", (ci >= 0).all())
chk("相邻信号相隔 5 个交易日(5 日标签不重叠)", set(np.diff(ci)) == {5}, sorted(set(np.diff(ci))))
cnt = hp.groupby("model").eligible_asof.sum().to_dict(); out["eligible_rows_per_model"] = {k: int(v) for k, v in cnt.items()}
chk("六模型 eligible 行数一致", len(set(cnt.values())) == 1, cnt)
nn = hp[hp.eligible_asof].groupby("model").score.apply(lambda s: int(s.isna().sum())).to_dict()
chk("eligible 行 score 无缺失(全部六模型)", sum(nn.values()) == 0, nn)
nn2 = hp[~hp.eligible_asof].groupby("model").score.apply(lambda s: int(s.notna().sum())).to_dict(); out["scores_on_non_eligible_rows"] = nn2
# training metadata
meta = []
for mn, g in hp.groupby("model"):
    for mid, h in g.groupby("model_id"):
        f = lambda c: h[c].iloc[0]
        first = h.signal_date.min()
        meta.append(dict(model=mn, model_id=mid, first_signal=first, n_sig=h.signal_date.nunique(), trained_asof=f("trained_asof"), train_first=f("train_first_date"), train_end=f("train_end_date"), val_end=f("val_end_date"), label_end=f("train_label_end_date"),
                         const=all(h[c].nunique(dropna=False) == 1 for c in ["trained_asof", "train_first_date", "train_end_date", "val_end_date", "train_label_end_date", "n_train_rows"])))
M = pd.DataFrame(meta); M.to_csv(Q.RES / "a_verify_train_meta.csv", index=False, encoding="utf-8-sig")
chk("每个 model_id 内训练元数据恒定", M.const.all())
chk("每模型 52 个 model_id(每块 4 期再训练)", (M.groupby("model").size() == 52).all(), M.groupby("model").size().to_dict())
d = lambda c: pd.to_datetime(M[c].replace("", np.nan))
M["gap_label_to_signal"] = (pd.to_datetime(M.first_signal) - d("label_end")).dt.days
chk("train_label_end_date(训练标签到期) < 该块首个 signal_date(全部模型)", (M.gap_label_to_signal > 0).all(), f"min gap days={int(M.gap_label_to_signal.min())}")
chk("train_end_date <= train_label_end_date(全部)", (d("train_end") <= d("label_end")).all())
tr = M[M.model.isin(["MLP", "GRU", "TF"])]
dd = lambda c: pd.to_datetime(tr[c].replace("", np.nan))
chk("DL: train_end < val_end <= label_end(max_label_expiry)", ((pd.to_datetime(tr.train_end) < pd.to_datetime(tr.val_end)) & (pd.to_datetime(tr.val_end) <= pd.to_datetime(tr.label_end))).all())
chk("DL: trained_asof == 该块首个 signal_date", (tr.trained_asof == tr.first_signal).all(), int((tr.trained_asof != tr.first_signal).sum()))
th = M[M.model.isin(["HGB", "LGBM", "Ridge"])]
chk("树/线性: 无 trained_asof/val_end 字段(表中为空,A2 训练无验证集)", (th.trained_asof.fillna("") == "").all() and (th.val_end.fillna("") == "").all())
out["train_meta_note"] = "HGB/LGBM/Ridge: train_end_date=该块最后训练信号日(label 到期 +8 日历日左右 < 首个信号);DL: train_end 早于树模型约 3 个月(留出 val 块用于早停),val_end 与树模型 train_end 同日,train_label_end=max_label_expiry=val_end+6 交易日"
# train_end of DL vs HGB at same block
M["blk"] = M.sort_values("first_signal").groupby("model").cumcount(); gm = M.pivot(index="blk", columns="model", values="train_end")
chk("DL 训练窗口结束日整体早于同块 HGB(留出验证期)", (pd.to_datetime(gm["GRU"]) < pd.to_datetime(gm["HGB"])).all())
chk("DL val_end == 同块 HGB train_end", (M[M.model == "GRU"].sort_values("first_signal").val_end.values == M[M.model == "HGB"].sort_values("first_signal").train_end.values).all())
# label definition: realised columns vs independent recomputation from P0 matrices
hh = hp[hp.eligible_asof][["signal_date", "code", "model", "realized_ret_5d", "realized_rel_ret_5d", "order_date", "exit_date", "label_uses_stale_price"]].copy(); hh["signal_date"] = pd.to_datetime(hh.signal_date)
Pk = P[P.eligible][["signal_date", "code", "ret_old", "ret_old_rel", "i"]]
mg = hh.merge(Pk, on=["signal_date", "code"], how="left")
for mn in Q.MODELS:
    g = mg[mg.model == mn]; a, b = g.realized_ret_5d.values.astype(float), g.ret_old.values
    both = np.isfinite(a) & np.isfinite(b); mism_nan = int((np.isfinite(a) != np.isfinite(b)).sum())
    mx = float(np.abs(a[both] - b[both]).max()); a2, b2 = g.realized_rel_ret_5d.values.astype(float), g.ret_old_rel.values; both2 = np.isfinite(a2) & np.isfinite(b2)
    out.setdefault("label_recompute", {})[mn] = dict(max_abs_diff_ret=mx, max_abs_diff_rel=float(np.abs(a2[both2] - b2[both2]).max()), nan_pattern_mismatch=mism_nan, n=int(len(g)))
    chk(f"{mn}: realized_ret_5d / rel 与独立重算(t+1收盘→t+6收盘)一致", mx < 1e-9 and np.abs(a2[both2] - b2[both2]).max() < 1e-9 and mism_nan == 0, out["label_recompute"][mn])
g = mg[mg.model == "HGB"]; iv = g.i.values.astype(int)
chk("order_date=cal[i+1],exit_date=cal[i+6]", (pd.to_datetime(g.order_date).values == cal.values[iv + 1]).all() and (pd.to_datetime(g.exit_date).values == cal.values[np.minimum(iv + 6, len(cal) - 1)]).all())
out["label_definition"] = "realized_ret_5d = 复权收盘(t+1) -> 复权收盘(t+6) 的收益(P0 主日历行号 i+1 → i+6;复权用 Sina 前复权因子,停牌日用最近已知价前向填充,label_uses_stale_price 标记);realized_rel_ret_5d = 减去当日 eligible 池均值。无论 t+1 是否涨停封板/停牌,都按此口径计入(未做可成交过滤)。"
out["stale_price_label_rows"] = {mn: int(hp[(hp.model == mn) & hp.eligible_asof].label_uses_stale_price.sum()) for mn in Q.MODELS}
ntr = hp.groupby("model").n_train_rows.apply(lambda s: int(s.notna().sum())).to_dict(); out["n_train_rows_nonnull_per_model"] = ntr
out["note_n_train_rows"] = "n_train_rows 仅 HGB/LGBM/Ridge 记录;MLP/GRU/TF 在总表中为空(DL 训练行数未写入冻结表,属记录缺失,不影响分数)"
# DL oos parquets
dl = {}
for k, mn in [("mlp", "MLP"), ("gru", "GRU"), ("tf", "TF")]:
    p = Q.P1D / "scores" / f"dl_{k}_oos_predictions.parquet"; d0 = pd.read_parquet(p)
    info = dict(path=str(p), bytes=p.stat().st_size, rows=len(d0), key_unique=not d0.duplicated(["signal_date", "code"]).any(), n_dates=int(d0.signal_date.nunique()), n_codes=int(d0.code.nunique()),
                code_6char=bool(d0.code.map(lambda s: re.fullmatch(r"\d{6}", s) is not None).all()), has_cols={c: (c in d0.columns) for c in ["trained_asof", "train_end", "val_end", "max_label_expiry", "realized_ret_5d", "realized_excess_5d"]})
    d0["signal_date"] = pd.to_datetime(d0.signal_date)
    sc = [c for c in d0.columns if c.endswith("score_mean")][0]
    mm = d0.merge(P[["signal_date", "code", "score_" + mn, "eligible"]], on=["signal_date", "code"], how="outer", indicator=True)
    both = mm[mm._merge == "both"]; msk = both.eligible & both[sc].notna() & both["score_" + mn].notna()
    info["score_equals_all_models_table_max_abs_diff"] = float(np.abs(both[sc][msk] - both["score_" + mn][msk]).max()); info["merge_both"] = int((mm._merge == "both").sum())
    info["eligible_score_nan_in_dl_file"] = int(both[both.eligible][sc].isna().sum())
    dl[mn] = info
    chk(f"{mn}: OOS parquet 行数 310,086 / 键唯一 / 代码6位 / 与总表分数一致", info["rows"] == 310086 and info["key_unique"] and info["code_6char"] and info["score_equals_all_models_table_max_abs_diff"] < 1e-12 and info["merge_both"] == 310086, info)
out["dl_oos_files"] = dl
# coverage and common sample
Pe = P[P.eligible]
sc_cols = ["score_" + m for m in Q.MODELS]
common = Pe[sc_cols].notna().all(axis=1)
out["coverage"] = dict(eligible_rows=int(len(Pe)), common_all_six_scores=int(common.sum()), common_with_new_def_return=int((common & Pe.ret_new_rel.notna()).sum()),
                       per_model_scores_eligible={m: int(Pe["score_" + m].notna().sum()) for m in Q.MODELS})
chk("GRU/TF 与 HGB 的 OOS 覆盖(日期×代码)在冻结文件中完全相同 → 共同样本=全部 eligible 行", common.sum() == len(Pe), out["coverage"])
out["coverage_note"] = "冻结总表里六模型覆盖相同(207 个信号日 × 1498 码,eligible 238,781 行)。差异不在覆盖而在训练窗口:DL 的 train_end 比 HGB 早约 3 个月(早停验证块),且 DL 为 3 seeds(Transformer 仅 seed 42)取均值;共同样本比较仍按 (signal_date,code) 交集进行。"
# registry untouched
man = json.loads((Q.P1D / "forward_registry_20261016_manifest.json").read_text(encoding="utf-8"))
bad = []; nfiles = 0
for rel, h in man["sha256"].items():
    nfiles += 1; pth = Q.V2 / rel.replace("\\", "/")
    if not pth.exists(): bad.append("MISSING " + rel)
    elif Q.sha256(pth) != h: bad.append("MISMATCH " + rel)
out["forward_registry_manifest"] = dict(files_checked=nfiles, mismatches=bad, registry_file_sha256=Q.sha256(Q.P1D / "forward_registry_20261016.csv"))
chk("forward_registry manifest sha256 全部匹配(未被改动)", nfiles > 0 and not bad, dict(n=nfiles, bad=bad))
out["verdict"] = "全部通过" if not out["inconsistencies"] else f"{len(out['inconsistencies'])} 项不一致"
(Q.P2 / "a_verify.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
md = ["# P2-A 冻结预测文件核验 (a_verify)", "", f"- 结论:**{out['verdict']}**", f"- 文件:`{Q.HP_ALL}`,{len(hp):,} 行,sha256 `{out['files']['historical_predictions_all_models.parquet']['sha256']}`", "",
      "## 检查项", "", "| 检查 | 结果 | 细节 |", "|---|---|---|"]
for c in out["checks"]: md.append(f"| {c['name']} | {'通过' if c['ok'] else '**不一致**'} | {c['detail'][:160].replace('|','/')} |")
md += ["", "## 标签定义", "", out["label_definition"], "", "## 训练元数据说明", "", out["train_meta_note"], "", "## 覆盖说明", "", out["coverage_note"], "", "## 记录缺口(非不一致)", "", "- " + out["note_n_train_rows"], "",
       "## 不一致记录", ""] + ([f"- {x}" for x in out["inconsistencies"]] or ["- 无"])
(Q.P2 / "a_verify.md").write_text("\n".join(md), encoding="utf-8")
print(out["verdict"]); print(json.dumps(out["inconsistencies"], ensure_ascii=False, indent=1)); print(json.dumps(out["coverage"])); print(json.dumps(out["label_recompute"]))