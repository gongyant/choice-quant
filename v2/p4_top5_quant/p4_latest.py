# -*- coding: utf-8 -*-
"""P4 batch 1: rank the latest complete trading day with the chosen E1 model (final model trained only on matured labels) and register it.
python p4_latest.py <MODEL> <h>. 'research_rank' is a ranking; 'meets_buy_criteria' is a separate, pre-stated gate (see BUY_GATE). Not investment advice."""
import sys, json, hashlib, pickle, datetime as dt, os, stat
import numpy as np, pandas as pd
import p4lib as P
MODEL, h = sys.argv[1], int(sys.argv[2]); C = P.P4 / "cache"; REG = P.P4 / "predictions_registry"; REG.mkdir(exist_ok=True)
fl = np.load(C / "flags.npz"); X = pickle.load(open(C / "world.pkl", "rb")); cal = list(fl["cal"]); iL = len(cal) - 1; D = cal[iL].replace("-", "")
out = REG / f"p4_pred_{D}_{MODEL}_h{h}.csv"; assert not out.exists(), f"registry file exists, refusing to overwrite: {out}"
lt = pd.read_parquet(P.P4 / "preds" / f"{MODEL}_h{h}_latest.parquet"); assert (lt.signal_date == cal[iL]).all() and lt.label_end_max.max() < cal[iL]
SU = pd.read_csv(P.P4 / "results" / "bt_summary_e0_e1.csv", encoding="utf-8-sig")
codes = list(fl["codes"]); col = {c: j for j, c in enumerate(codes)}; j = lt.code.map(col).values; u = X.u
d = lt.assign(name=u.name.values[j], board=u.board.values[j], industry=u.industry_l2.values[j], close_unadj=X.C[iL, j], amt20=fl["amt20"][iL, j],
              traded_t=X.traded[iL, j], limit_up_locked_t=X.lim_up[iL, j], limit_down_locked_t=X.lim_dn[iL, j], st_proxy=fl["st"][iL, j],
              st_name_current=(u.is_st_name.fillna(False) | u.name_has_delist_mark.fillna(False)).values[j], star_market=(u.board.values[j] == "科创板"))
d["tradable_filter"] = d.traded_t & ~d.limit_up_locked_t & (d.amt20 >= 1e8) & ~d.st_proxy & ~d.st_name_current
d["score_pctile"] = d.score.rank(pct=True); d = d.sort_values("score", ascending=False)
d["research_rank_all"] = np.arange(1, len(d) + 1); d["research_rank_tradable"] = np.where(d.tradable_filter, d.tradable_filter.cumsum(), np.nan)
# BUY_GATE (fixed before looking at today's list): the model's own historical Top5 (same horizon, same engine) must have
# (a) net mean excess vs pool > 0 with t >= 2 over the full history AND (b) net excess > 0 in BOTH the design and exploratory sub-periods.
nm = {"E1reg": "E1_LGBM回归(增强因子)", "E1lrank": "E1_LambdaRank", "E1xendcg": "E1_rank_xendcg"}[MODEL]
g = SU[(SU.strategy == nm) & (SU.horizon == h) & (SU.K == 5)].set_index("period")
gate = bool(g.loc["全期", "t_excess"] >= 2 and (g.mean_excess_net > 0).all())
d["meets_buy_criteria"] = gate & d.tradable_filter & (d.research_rank_tradable <= 5)
d["conclusion"] = np.where(d.research_rank_tradable <= 5, np.where(d.meets_buy_criteria, "满足预设买入标准(仍非投资建议)", "研究排名前五·未满足买入标准·建议观察"), "")
d["note"] = "研究排名不是买入推荐;score 是排序分,不是上涨概率或收益率"
d.to_csv(out, index=False, encoding="utf-8-sig"); sh = hashlib.sha256(out.read_bytes()).hexdigest(); os.chmod(out, stat.S_IREAD)
rec = dict(file=out.name, sha256=sh, model=MODEL, horizon=h, signal_date=cal[iL], n_rows=len(d), buy_gate_passed=gate, written_at_local=dt.datetime.now().isoformat(timespec="seconds"),
           first_verifiable=f"t+1 open ({h} trading days later)", exploratory=True)
open(REG / "manifest.jsonl", "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
print(json.dumps(rec, ensure_ascii=False)); print(d[d.research_rank_tradable <= 10][["research_rank_tradable", "code", "name", "board", "industry", "close_unadj", "score", "score_pctile", "meets_buy_criteria"]].round(4).to_string())
