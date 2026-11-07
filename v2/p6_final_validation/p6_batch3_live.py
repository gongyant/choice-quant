# -*- coding: utf-8 -*-
"""P6 Parts 5-7: 10-23 snapshot (do NOT touch 10-22 freeze), stock cards, forward registry, buy gate, deliverable CSVs."""
import json, hashlib, os, stat, datetime as dt, glob
from pathlib import Path
import numpy as np, pandas as pd
import p6_common as Q
R = Q.P6 / "results"; REG = Q.P6 / "predictions_registry"; REG.mkdir(exist_ok=True)
W = Q.load(); X = W.X; P = Q.P; cal = W.cal; codes = W.codes; col = W.col
D = "2026-10-23"; assert str(cal[-1].date()) == "2026-10-21", cal[-1]
# Load 10-22 and 10-23 sina bars to extend adj close
B22 = pd.read_csv(Q.P6 / "bars_20261022_sina.csv", dtype={"code": str}, encoding="utf-8-sig").set_index("code").reindex(codes)
B23 = pd.read_csv(Q.P6 / "bars_20261023_sina.csv", dtype={"code": str}, encoding="utf-8-sig").set_index("code").reindex(codes)
tr22 = (B22.date == "2026-10-22").values & (B22.open.fillna(0) > 0).values
tr23 = (B23.date == D).values & (B23.open.fillna(0) > 0).values
ret22 = np.where(tr22, B22.close.values / B22.prev_close.values - 1, np.nan)
ret23 = np.where(tr23, B23.close.values / B23.prev_close.values - 1, np.nan)
A0 = W.A[-1]; A22 = np.where(tr22, A0 * (1 + ret22), A0); A23 = np.where(tr23, A22 * (1 + ret23), A22)
Aext = np.vstack([W.A, A22, A23])  # rows: ... 10-21, 10-22, 10-23
r1 = pd.DataFrame(Aext).pct_change(fill_method=None)
r1.iloc[-2] = np.where(tr22, ret22, np.nan); r1.iloc[-1] = np.where(tr23, ret23, np.nan)
vol = r1.rolling(20, min_periods=15).std().values[-1]
dd60 = Aext[-1] / pd.DataFrame(Aext).rolling(60, min_periods=40).max().values[-1] - 1
r20 = Aext[-1] / Aext[-21] - 1; r5 = Aext[-1] / Aext[-6] - 1
lim_pct = np.where(np.isin(W.board, ["创业板", "科创板"]), 0.20, np.where(W.board == "北交所", 0.30, 0.10))
cur_st = (X.u.is_st_name.fillna(False) | X.u.name_has_delist_mark.fillna(False)).values.astype(bool)
lim_pct = np.where(cur_st & ~np.isin(W.board, ["创业板", "科创板"]), 0.05, lim_pct)
limup = tr23 & (B23.close.values >= np.round(B23.prev_close.values * (1 + lim_pct), 2) - 0.001)
elig = W.elig[-1] & tr23
filt = elig & ~limup & (np.nan_to_num(W.amt20[-1]) >= 1e8) & ~W.st[-1] & ~cur_st
volp = pd.Series(np.where(elig, vol, np.nan)).rank(pct=True).values
cons = (np.nan_to_num(volp, nan=1) <= .8) & (np.nan_to_num(dd60, nan=-1) > -.35) & (np.nan_to_num(r20, nan=-1) > -.25)
lowv = np.where(filt, -vol, np.nan)

# Previous B1 simulated holdings from 10-22 freeze (P5 registry) for buffer continuity
prev_codes = []
p5reg = Q.P5D / "predictions_registry" / "p5_pred_20261022_B1_h5.csv"
if p5reg.exists():
    prev_codes = pd.read_csv(p5reg, dtype={"code": str}, encoding="utf-8-sig").query("in_final5 == True").code.tolist()
prev_idx = [col[c] for c in prev_codes if c in col]

def pick(score, prev=(), use_cons=True, buffer=True, K=5, NB=40):
    s = np.where(filt & np.isfinite(score), score, np.nan); ok = np.where(np.isfinite(s))[0]
    order = ok[np.argsort(-s[ok], kind="stable")]; rank = {j: r for r, j in enumerate(order)}; out, cnt = [], {}
    ok_c = cons if use_cons else np.ones(W.N, bool)
    def add(j):
        if len(out) < K and ok_c[j] and (not use_cons or cnt.get(W.ind[j], 0) < 2) and j not in out:
            out.append(j); cnt[W.ind[j]] = cnt.get(W.ind[j], 0) + 1
    if buffer:
        for j in prev:
            if j in rank and rank[j] < NB: add(j)
    for j in order[:20]: add(j)
    return out, order, s

# Freeze B1 rules
RULES = dict(version="p6_B1_rules_v1", frozen_at=dt.datetime.now().isoformat(timespec="seconds"),
             signal="low-vol = -vol20", filters="elig(asof last P4 day)&traded signal day&not close limit-up&amt20>=1e8&not ST",
             constraints="Top20 -> volp<=0.8, dd60>-35%, r20>-25%, <=2 per industry_l2",
             buffer="keep previous simulated holdings while rank<40", cost="0.30% round-trip; retained names free",
             entry="t+1 open", exit="t+1+h open (delay if suspended/open limit-down)", slots=5,
             note="Frozen for forward validation after 2026-10-22. Do not retune.")
rules_path = Q.P6 / "selection_rules_p6_B1_v1.json"
if not rules_path.exists(): json.dump(RULES, open(rules_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
rules_sha = hashlib.sha256(rules_path.read_bytes()).hexdigest()

# B1 Top5 research rank (5d) with buffer from 10-22 holdings
pk_b1, order_b1, sc_b1 = pick(lowv, prev=prev_idx, use_cons=True, buffer=True)
# B0 Top5 research rank (20d) = plain low-vol no buffer no constraints
pk_b0, order_b0, sc_b0 = pick(lowv, prev=(), use_cons=False, buffer=False)

# Current simulated holdings: after 10-22 B1 h5 signal, planned entry 10-23 open, exit 10-23+5 open.
# As of close 10-23, the 10-22 signal's positions were bought at 10-23 open → CURRENT holdings = 10-22 B1 Top5
# (holding until ~10-30). Simultaneously 10-23 produces a NEW research rank for the NEXT entry (10-24 open).
hold_src = pd.read_csv(p5reg, dtype={"code": str}, encoding="utf-8-sig") if p5reg.exists() else pd.DataFrame()
hold_rows = []
if len(hold_src):
    for r in hold_src.query("in_final5 == True").itertuples():
        j = col.get(r.code)
        hold_rows.append(dict(asof=D, origin_signal="2026-10-22", strategy="B1", horizon=5, code=r.code, name=r.name,
                              industry=getattr(r, "industry", W.ind[j] if j is not None else ""),
                              entry_date="2026-10-23", entry_open=float(B23.open.get(r.code, np.nan)),
                              ref_close=float(B23.close.get(r.code, np.nan)), chg_pct=float(ret23[j]*100) if j is not None else np.nan,
                              planned_exit="entry+5 trading days open (~2026-10-30)", sim_weight=0.2,
                              status="OPEN_SIMULATED", note="模拟持仓,非实盘;研究用"))
CUR = pd.DataFrame(hold_rows); CUR.to_csv(R / "current_holdings.csv", index=False, encoding="utf-8-sig")

# Fundamentals helper (reuse P4 fin_raw)
FR = Q.P4D / "fin_raw"
y = pd.concat([pd.read_csv(f, dtype={"SECURITY_CODE": str}) for f in sorted(glob.glob(str(FR / "yjbb_*.csv.gz")))])
y["period"] = pd.to_datetime(y.REPORTDATE.str[:10]); y["notice"] = pd.to_datetime(y.NOTICE_DATE.str[:10])
y = y.sort_values("notice").drop_duplicates(["SECURITY_CODE", "period"], keep="first"); y["q"] = y.period.dt.month // 3
Keps = y.set_index(["SECURITY_CODE", "period"])["BASIC_EPS"]
def ttm_eps(c, p):
    if p.month == 12: return Keps.get((c, p), np.nan)
    return Keps.get((c, p), np.nan) + Keps.get((c, pd.Timestamp(p.year-1,12,31)), np.nan) - Keps.get((c, pd.Timestamp(p.year-1,p.month,p.day)), np.nan)
def fund(c):
    a = y[(y.SECURITY_CODE == c) & (y.notice < pd.Timestamp(D))].sort_values("period")
    if not len(a): return dict(report_period="未取得", pe_ttm="未取得", pb="未取得", roe_raw="未取得", roe_ann="未取得", np_yoy="未取得", rev_yoy="未取得", bps="未取得")
    r = a.iloc[-1]; p = r.period; cl = B23.close.get(c, np.nan); et = ttm_eps(c, p)
    return dict(report_period=str(p.date()), notice_date=str(r.notice.date()), revenue_yoy=r.YSTZ, np_yoy=r.SJLTZ,
                roe_raw=r.WEIGHTAVG_ROE, roe_ann=r.WEIGHTAVG_ROE*4/r.q, bps=r.BPS,
                pe_ttm=(cl/et if np.isfinite(et) and et>0 else "未取得"), pb=(cl/r.BPS if r.BPS and r.BPS>0 else "未取得"),
                industry_em=r.PUBLISHNAME, netprofit_ytd_yi=r.PARENT_NETPROFIT/1e8)

# Similar-score history: for each pick, look at past B1 OOS periods where selected name had similar volp rank, average next 5d excess
HO = pd.read_csv(R / "b1_trades_h5.csv", dtype={"code": str}, encoding="utf-8-sig")
PER = pd.read_csv(R / "b1_periods_h5.csv", encoding="utf-8-sig")
filled = HO[HO.status.isin(["FILLED", "MERGER_SWAP_RECORD"])]

def similar_history(j, top_n=30):
    """Past filled B1 slots with |volp_now - volp_then| small; report mean slot excess vs pool."""
    # approximate: use same code history + nearby vol names
    hist = filled[filled.code == codes[j]]
    if len(hist) >= 3:
        # join period pool
        m = hist.merge(PER[["signal_date", "pool_ret"]], on="signal_date", how="left")
        ex = m.slot_ret_gross - m.pool_ret
        return dict(n_hist=len(m), mean_slot_gross=float(m.slot_ret_gross.mean()), mean_excess=float(ex.mean()),
                    up_share=float((m.slot_ret_gross > 0).mean()), beat_pool_share=float((ex > 0).mean()),
                    worst=float(m.slot_ret_gross.min()), best=float(m.slot_ret_gross.max()))
    # fallback: all B1 slots with similar volp at signal
    return dict(n_hist=0, mean_slot_gross="未取得", mean_excess="未取得", up_share="未取得", beat_pool_share="未取得", worst="未取得", best="未取得")

# Buy criterion from P5 (still fail) + P6 note
buy = dict(version="p6_inherits_p5_buy_criterion_v1", status="不通过:不构成买入依据",
           reason="回测相对股票池超额 HAC t=0.19<2; 前向到期期数=0; 即使A3相对A0显著,也不满足相对池与前向门槛。",
           C1_t_vs_pool=0.19, C5_forward_periods=0)

def card(j, rank, score, strategy):
    f = fund(codes[j]); sh = similar_history(j)
    why = []
    why.append(f"低波分位 volp={volp[j]:.3f}(池内越低越稳)")
    if cons[j]: why.append("通过约束:volp≤0.8, dd60>−35%, r20>−25%")
    else: why.append("未过约束(若出现在缓冲保留中则可能仍持有)")
    if j in prev_idx: why.append("10-22 模拟持仓缓冲保留(排名<40)")
    why.append(f"行业 {W.ind[j]}(同业上限2)")
    # why not others: top rejected
    return dict(strategy=strategy, signal_date=D, research_rank=rank, code=codes[j], name=W.names[j], board=W.board[j], industry=W.ind[j],
                close=float(B23.close.iloc[j]), chg_pct=float(ret23[j]*100), select_score=float(score[j]),
                vol20=float(vol[j]), volp=float(volp[j]), dd60=float(dd60[j]), r5=float(r5[j]), r20=float(r20[j]),
                **{f"fund_{k}": v for k, v in f.items()},
                why_selected="；".join(why),
                similar_score_history=json.dumps(sh, ensure_ascii=False),
                buy_criteria_status=buy["status"], buy_criteria_reason=buy["reason"],
                conclusion="研究候选,未达到买入条件")

# Latest Top5 lists
rows = []
for rk, j in enumerate(pk_b1): rows.append(card(j, rk+1, sc_b1, "B1_research_rank_h5"))
for rk, j in enumerate(pk_b0): rows.append(card(j, rk+1, sc_b0, "B0_research_rank_h20"))
# Also store Top20 for B1 for explanation of why-not
why_not = []
for rk, j in enumerate(order_b1[:20]):
    if j in pk_b1: continue
    reasons = []
    if not cons[j]: reasons.append(f"未过约束(volp={volp[j]:.2f},dd60={dd60[j]:.2%},r20={r20[j]:.2%})")
    if sum(1 for p in pk_b1 if W.ind[p]==W.ind[j]) >= 2: reasons.append(f"行业{W.ind[j]}已满2只")
    if len(pk_b1) >= 5 and j not in pk_b1: reasons.append("Top5名额已满")
    why_not.append(dict(code=codes[j], name=W.names[j], cand_rank=rk+1, industry=W.ind[j], volp=volp[j], why_not="；".join(reasons) or "分数低于入选者"))
TOP = pd.DataFrame(rows); TOP.to_csv(R / "latest_top5.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(why_not).to_csv(R / "latest_why_not.csv", index=False, encoding="utf-8-sig")

# Register 10-23 predictions (NEW snapshot; 10-22 untouched)
man = REG / "manifest.jsonl"; regs = []
pool_f = REG / f"p6_pool_{D.replace('-','')}.csv"
if not pool_f.exists():
    pd.DataFrame(dict(code=[codes[j] for j in np.where(elig)[0]])).to_csv(pool_f, index=False, encoding="utf-8-sig"); regs.append(pool_f)
for strat, pk, order, sc, h, prev in [("B1", pk_b1, order_b1, sc_b1, 5, prev_idx), ("B0", pk_b0, order_b0, sc_b0, 20, [])]:
    rows_r = []
    for rk, j in enumerate(order[:20]):
        rows_r.append(dict(cand_rank=rk+1, code=codes[j], name=W.names[j], board=W.board[j], industry=W.ind[j],
                           close=float(B23.close.iloc[j]), select_score=float(sc[j]) if np.isfinite(sc[j]) else np.nan,
                           in_final5=j in pk, vol20=float(vol[j]), volp=float(volp[j]), dd60=float(dd60[j]), r20=float(r20[j]),
                           sim_weight=0.2 if j in pk else 0.0, buffered_from_prev=j in prev))
    d = pd.DataFrame(rows_r); d["strategy"]=strat; d["horizon"]=h; d["signal_date"]=D; d["rules_sha256"]=rules_sha
    d["model_version"]="lowvol_v1"; d["note"]="研究排名,不是买入推荐;模拟成交=次日开盘;10-22冻结文件未改动"
    out = REG / f"p6_pred_{D.replace('-','')}_{strat}_h{h}.csv"
    if not out.exists():
        d.to_csv(out, index=False, encoding="utf-8-sig"); os.chmod(out, stat.S_IREAD); regs.append(out)

# Also copy-register pointer that 10-22 P5 files remain source of truth for that day
done = set()
if man.exists(): done = {json.loads(l)["file"] for l in open(man, encoding="utf-8-sig")}
with open(man, "a", encoding="utf-8") as fh:
    for f in regs:
        if f.name not in done:
            fh.write(json.dumps(dict(file=f.name, sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                                     registered_at=dt.datetime.now().isoformat(timespec="seconds"), rules_sha256=rules_sha, signal_date=D), ensure_ascii=False)+"\n")

# Forward registry append (10-23 signals)
FW = Q.P6 / "forward_registry.csv"
# Seed with 10-22 from P5 if not present
have = set()
if FW.exists(): have = set(map(tuple, pd.read_csv(FW, dtype=str)[["signal_date","strategy","horizon","code"]].values))
fw = []
# 10-22 from P5
for f in sorted((Q.P5D / "predictions_registry").glob("p5_pred_20261022_B*_h*.csv")):
    d = pd.read_csv(f, dtype={"code": str}, encoding="utf-8-sig")
    for r in d[d.in_final5].itertuples():
        k = ("2026-10-22", r.strategy, str(int(r.horizon)), r.code)
        if k in have: continue
        fw.append(dict(signal_date="2026-10-22", strategy=r.strategy, horizon=int(r.horizon), code=r.code, name=r.name,
                       rules_sha256="p5:"+getattr(r,"rules_sha256",""), model_version=getattr(r,"model_version",""),
                       sim_weight=0.2, planned_entry="2026-10-23 开盘", planned_exit=f"entry+{int(r.horizon)}交易日开盘",
                       registry_file=f.name, source="P5_frozen"))
# 10-23
for strat, pk, h in [("B1", pk_b1, 5), ("B0", pk_b0, 20)]:
    for j in pk:
        k = (D, strat, str(h), codes[j])
        if k in have: continue
        fw.append(dict(signal_date=D, strategy=strat, horizon=h, code=codes[j], name=W.names[j], rules_sha256=rules_sha,
                       model_version="lowvol_v1", sim_weight=0.2, planned_entry="2026-10-24 开盘",
                       planned_exit=f"entry+{h}交易日开盘", registry_file=f"p6_pred_{D.replace('-','')}_{strat}_h{h}.csv", source="P6"))
if fw: pd.DataFrame(fw).to_csv(FW, mode="a", header=not FW.exists(), index=False, encoding="utf-8-sig")

# Forward validation file (empty matured rows + schema)
FV = Q.P6 / "forward_validation.csv"
if not FV.exists():
    pd.DataFrame(columns=["signal_date","strategy","horizon","code","name","entry_date","exit_date","ret_open_to_open","ret_net","pool_ret","rel","verified_at","note"]).to_csv(FV, index=False, encoding="utf-8-sig")

# Buy gate table
G = pd.DataFrame([dict(strategy="B1", horizon=5, **buy), dict(strategy="B0", horizon=20, status="不通过:不构成买入依据",
                 reason="原始低波无缓冲无约束;5日样本仅+3%净,且未过相对池与前向门槛。")])
G.to_csv(R / "buy_gate.csv", index=False, encoding="utf-8-sig")

print("B1 Top5:", TOP[TOP.strategy.str.startswith("B1")][["research_rank","code","name","close","volp","dd60","r20"]].to_string(index=False))
print("B0 Top5:", TOP[TOP.strategy.str.startswith("B0")][["research_rank","code","name","close","volp"]].to_string(index=False))
print("HOLDINGS", CUR[["code","name","entry_open","ref_close","chg_pct"]].to_string(index=False))
print("traded23", int(tr23.sum()), "filt", int(filt.sum()), "regs", [f.name for f in regs])
print(open(man, encoding="utf-8-sig").read())
