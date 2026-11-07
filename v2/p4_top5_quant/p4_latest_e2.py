# -*- coding: utf-8 -*-
"""P4 batch 2 latest: E2reg final model (matured labels only) at the last complete trading day -> Top20 -> pre-registered constraints -> Top5.
python p4_latest_e2.py <S1|S2|S3|S4> . Registers to predictions_registry (read-only, sha256). Research ranking, not advice."""
import sys, json, hashlib, pickle, datetime as dt, os, stat
import numpy as np, pandas as pd
import p4lib as P
VAR = sys.argv[1]; C = P.P4 / "cache"; REG = P.P4 / "predictions_registry"
fl = np.load(C / "flags.npz"); X = pickle.load(open(C / "world.pkl", "rb")); F = np.load(C / "fin_raw_mats.npz"); cal = list(fl["cal"]); i = len(cal) - 1; D = cal[i].replace("-", "")
codes = list(fl["codes"]); col = {c: j for j, c in enumerate(codes)}; u = X.u; ind = u.industry_l2.fillna("NA").values; elig = fl["elig"]
G = pd.read_csv(P.P4 / "results" / "buy_gate_e2.csv", encoding="utf-8-sig")
A = pd.DataFrame(np.where(X.traded, X.C / X.F, np.nan)).ffill(); dd60 = (A / A.rolling(60, min_periods=40).max() - 1).values[i]; r = lambda k: (A.iloc[i] / A.iloc[i - k] - 1).values
vol = A.pct_change(fill_method=None).rolling(20, min_periods=15).std().values[i]; volp = pd.Series(np.where(elig[i], vol, np.nan)).rank(pct=True).values
cur_st = (u.is_st_name.fillna(False) | u.name_has_delist_mark.fillna(False)).values.astype(bool)
filt = elig[i] & X.traded[i] & ~X.lim_up[i] & (np.nan_to_num(fl["amt20"][i]) >= 1e8) & ~fl["st"][i] & ~cur_st
cons = (np.nan_to_num(volp, nan=1) <= .8) & (np.nan_to_num(dd60, nan=-1) > -.35) & (np.nan_to_num(r(20), nan=-1) > -.25)
VN = {"S1": "S1_E2固定Top5(约束)", "S2": "S2_E2缓冲区", "S3": "S3_E2+低波混合固定", "S4": "S4_E2+低波混合缓冲"}[VAR]; blend = VAR in ("S3", "S4")
OUT = []
for h in [5, 20]:
    out = REG / f"p4_pred_{D}_E2{VAR}_h{h}.csv"; assert not out.exists(), out
    lt = pd.read_parquet(P.P4 / "preds" / f"E2reg_h{h}_latest.parquet"); assert lt.label_end_max.max() < cal[i]
    s = np.full(len(codes), np.nan); s[lt.code.map(col).values] = lt.score.values; m = filt & np.isfinite(s)
    if blend: s2 = np.full(len(s), np.nan); s2[m] = 0.5 * pd.Series(s[m]).rank(pct=True).values + 0.5 * pd.Series(-vol[m]).rank(pct=True).values
    else: s2 = np.where(m, s, np.nan)
    ok = np.where(np.isfinite(s2))[0]; order = ok[np.argsort(-s2[ok], kind="stable")]; pick, cnt = [], {}
    for j in order[:20]:   # buffer variants: no simulated holdings exist yet on the first live date -> same as fixed
        if len(pick) < 5 and cons[j] and cnt.get(ind[j], 0) < 2: pick.append(j); cnt[ind[j]] = cnt.get(ind[j], 0) + 1
    gate = bool(G[(G.strategy == VN) & (G.horizon == h)].passes_buy_criteria.iloc[0])
    rows = []
    for rk, j in enumerate(order[:20]):
        ep, bp = F["ep"][i, j], F["bp"][i, j]
        rows.append(dict(cand_rank=rk + 1, code=codes[j], name=u.name.values[j], board=u.board.values[j], industry=ind[j], close_unadj=X.C[i, j], model_score=s[j], select_score=s2[j],
                         in_final5=j in pick, meets_buy_criteria=gate and j in pick, star_or_chinext=u.board.values[j] in ("科创板", "创业板"),
                         fails_vol_cap=volp[j] > .8, fails_drawdown_cap=not (dd60[j] > -.35 and r(20)[j] > -.25), r5=r(5)[j], r20=r(20)[j], r60=r(60)[j], vol20=vol[j], dd60=dd60[j],
                         report_period=dt.date.fromordinal(int(F["period_ord"][i, j])).isoformat() if np.isfinite(F["period_ord"][i, j]) else "未取得",
                         notice_date=dt.date.fromordinal(int(F["notice_ord"][i, j])).isoformat() if np.isfinite(F["notice_ord"][i, j]) else "未取得",
                         roe_ann_pct=F["roe_ann"][i, j], rev_yoy_pct=F["rev_yoy"][i, j], np_yoy_pct=F["np_yoy"][i, j], gross_margin_pct=F["gross_margin"][i, j], debt_ratio_pct=F["debt_ratio"][i, j],
                         ocf_to_bps=F["ocf_to_bps"][i, j], pe_ann=(1 / ep) if ep > 0 else np.nan, pb=(1 / bp) if bp > 0 else np.nan, loss_making=ep <= 0))
    d = pd.DataFrame(rows); d["conclusion"] = np.where(d.in_final5, np.where(d.meets_buy_criteria, "满足预设买入标准(仍非投资建议)", "研究前五·未满足买入标准·建议观察"), "")
    d["variant"] = VN; d["horizon"] = h; d["signal_date"] = cal[i]; d["note"] = "研究排名不是买入推荐;分数不是概率/收益率;PE/PB 用最近一期年化 EPS/BPS 与当日收盘价估算"
    d.to_csv(out, index=False, encoding="utf-8-sig"); sh = hashlib.sha256(out.read_bytes()).hexdigest(); os.chmod(out, stat.S_IREAD)
    rec = dict(file=out.name, sha256=sh, model="E2reg", variant=VN, horizon=h, signal_date=cal[i], buy_gate_passed=gate, rules="selection_rules_p4_v1.json", written_at_local=dt.datetime.now().isoformat(timespec="seconds"), exploratory=True)
    open(REG / "manifest.jsonl", "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n"); print(json.dumps(rec, ensure_ascii=False))
    print(d[d.in_final5].round(3).to_string())
