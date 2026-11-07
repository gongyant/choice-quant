# -*- coding: utf-8 -*-
"""P5 Batch A v2 (corrected t17u definitions). Old a1_* files are kept untouched (labelled 'v1 assumed defs'); this writes a1v2_*.
B0 = raw low-vol Top5 (tradable filter only, no constraints, no buffer)
B1 = pure low-vol factor + SAME constraints + SAME buffer as S4 (Top20 -> volp<=0.8, dd60>-35%, r20>-25%, <=2/industry_l2 -> keep held while rank<40)
B2 = E2 LightGBM (E2reg) score + same constraints + same buffer
B3 = S4 = 0.5*pct(E2reg) + 0.5*pct(low-vol) + same constraints + same buffer
Merger delistings: positions the engine leaves MARKED_LAST_CLOSE_UNSOLD are re-valued at the swap terms (base) when the code is a known merger;
non-merger delistings keep the engine's last-traded-close value. Sensitivity: all-last-price / swap at listing-day open / non-merger -100%. EXPLORATORY."""
import json, time
import numpy as np, pandas as pd
import p5_common as Q
t0 = time.time(); R = Q.P5 / "results"; W = Q.load(); P = Q.P
MERGERS = {"002013": dict(name="中航机电", acquirer="600372", acquirer_name="中航电子(现中航机载)", ratio=0.6647, record_date="2023-03-16", new_share_list_date="2023-04-14", acq_close_record=17.56, acq_open_listing=17.25, acq_price_source="Eastmoney push2his 600372 不复权日K: 2023-03-16 收 17.56, 2023-04-14 开 17.25(600372 不在合格池缓存内)",
                          source="http://static.cninfo.com.cn/finalpage/2023-03-17/1216135561.PDF (终止上市公告: 1:0.6647, 股权登记日 2023-03-16, 3-17 摘牌)")}
E2 = {h: Q.mat(W, pd.read_parquet(Q.PR / f"E2reg_h{h}.parquet")) for h in [5, 20]}
DEF = {"B0_原始低波Top5": lambda h, rows: Q.plain(W, W.lowvol, rows),
       "B1_低波+约束+缓冲": lambda h, rows: Q.select(W, W.lowvol, rows, True, False),
       "B2_E2+约束+缓冲": lambda h, rows: Q.select(W, E2[h], rows, True, False),
       "B3_S4": lambda h, rows: Q.select(W, E2[h], rows, True, True)}
per, hold = [], []
for h in [5, 20]:
    rows = Q.rows_for(W, h)
    for nm, f in DEF.items(): pr, ho = Q.run(W, f(h, rows), rows, h, nm); per.append(pr); hold.append(ho)
PER = pd.concat(per, ignore_index=True); HO = pd.concat(hold, ignore_index=True); PER["year"] = PER.signal_date.str[:4]
cal = W.cal; X = W.X
def ti(d):
    k = cal.searchsorted(pd.Timestamp(d)); return int(k)
mk = HO[HO.status == "MARKED_LAST_CLOSE_UNSOLD"].copy(); recs = []
for r in mk.itertuples():
    j = W.col[r.code]; e = ti(r.entry_date); last = int(np.where(X.traded[:, j])[0].max()); base_last = r.slot_ret_gross
    m = MERGERS.get(r.code)
    if m:
        cr, ol = m["acq_close_record"], m["acq_open_listing"]
        fadj = X.F[last, j] / X.F[e, j]          # dividends of target between entry and last trade (raw->adj)
        swap_rec = m["ratio"] * cr * fadj / X.O[e, j] - 1; swap_list = m["ratio"] * ol * fadj / X.O[e, j] - 1
        recs.append(dict(strategy=r.strategy, horizon=r.horizon, signal_date=r.signal_date, code=r.code, name=r.name, kind="merger_swap", last_trade=str(cal[last].date()),
                         ret_last_price=base_last, ret_swap_record=swap_rec, ret_swap_listing=swap_list, ret_minus100=-1.0, acq_close_record=cr, acq_open_listing=ol, entry_open_raw=X.O[e, j], fadj=fadj))
    else:
        recs.append(dict(strategy=r.strategy, horizon=r.horizon, signal_date=r.signal_date, code=r.code, name=r.name, kind="non_merger_delist_or_unsold", last_trade=str(cal[last].date()),
                         ret_last_price=base_last, ret_swap_record=base_last, ret_swap_listing=base_last, ret_minus100=-1.0))
MK = pd.DataFrame(recs); MK.to_csv(R / "a1v2_merger_delist_valuation.csv", index=False, encoding="utf-8-sig")
def adjust(a, col_merger, nonmerger_minus100=False):
    adj = a.set_index("signal_date").ret_net.copy(); nm, h = a.strategy.iloc[0], a.horizon.iloc[0]
    for r in MK[(MK.strategy == nm) & (MK.horizon == h)].itertuples():
        new = getattr(r, col_merger) if r.kind == "merger_swap" else (-1.0 if nonmerger_minus100 else r.ret_last_price)
        adj[r.signal_date] += (new - r.ret_last_price) / 5
    return adj.values
SCEN = {"base_swap_record": ("ret_swap_record", False), "all_last_price(engine)": ("ret_last_price", False), "swap_listing_open": ("ret_swap_listing", False), "swap_record+nonmerger_-100%": ("ret_swap_record", True)}
PER["ret_net_engine"] = PER.ret_net
for (nm, h), a in PER.groupby(["strategy", "horizon"]): PER.loc[(PER.strategy == nm) & (PER.horizon == h), "ret_net"] = adjust(a, *SCEN["base_swap_record"])
PER.to_csv(R / "a1v2_periods.csv", index=False, encoding="utf-8-sig"); HO.to_csv(R / "a1v2_holdings.csv", index=False, encoding="utf-8-sig")
def summ(a, rn=None):
    rn = a.ret_net.values if rn is None else rn; e = rn - a.pool_ret.values
    return dict(n=len(a), cum_gross=np.prod(1 + a.ret_gross) - 1, cum_net=np.prod(1 + rn) - 1, cum_pool_gross=np.prod(1 + a.pool_ret) - 1, mean_net=rn.mean(), mean_excess=e.mean(),
                t_hac_excess=Q.hac_t(e, 4), beat_pool=(e > 0).mean(), win_abs=(rn > 0).mean(), mdd=P.mdd(pd.Series(rn)), turnover=a.turnover.mean(), fees_sum=a.fees.sum(), avg_hold=a.n_held.mean())
S, Y, SE = [], [], []
for (nm, h), a in PER.groupby(["strategy", "horizon"]):
    S.append(dict(strategy=nm, horizon=h, **summ(a)))
    for y, b in a.groupby("year"): Y.append(dict(strategy=nm, horizon=h, year=y, **summ(b)))
    a0 = a.copy(); a0["ret_net"] = a0.ret_net_engine
    for sc, (c, m100) in SCEN.items(): SE.append(dict(strategy=nm, horizon=h, scenario=sc, cum_net=np.prod(1 + adjust(a0, c, m100)) - 1, mdd=P.mdd(pd.Series(adjust(a0, c, m100)))))
for h in [5, 20]:
    a = PER[(PER.strategy == "B0_原始低波Top5") & (PER.horizon == h)]
    S.append(dict(strategy="POOL_合格池等权(毛,参照)", horizon=h, n=len(a), cum_gross=np.prod(1 + a.pool_ret) - 1, mdd=P.mdd(a.pool_ret)))
    for y, b in a.groupby("year"): Y.append(dict(strategy="POOL_合格池等权(毛,参照)", horizon=h, year=y, cum_gross=np.prod(1 + b.pool_ret) - 1, mdd=P.mdd(b.pool_ret)))
S, Y, SE = pd.DataFrame(S), pd.DataFrame(Y), pd.DataFrame(SE)
S.to_csv(R / "a1v2_summary.csv", index=False, encoding="utf-8-sig"); Y.to_csv(R / "a1v2_by_year.csv", index=False, encoding="utf-8-sig"); SE.to_csv(R / "a1v2_merger_sensitivity.csv", index=False, encoding="utf-8-sig")
PT = []
for h, blk, lag in [(5, 8, 4), (20, 4, 2)]:
    for a_, b_ in [("B3_S4", "B1_低波+约束+缓冲"), ("B3_S4", "B2_E2+约束+缓冲"), ("B1_低波+约束+缓冲", "B0_原始低波Top5")]:
        x = PER[(PER.strategy == a_) & (PER.horizon == h)].set_index("signal_date").ret_net - PER[(PER.strategy == b_) & (PER.horizon == h)].set_index("signal_date").ret_net
        lo, hi = Q.block_ci(x.values, blk); PT.append(dict(pair=f"{a_} - {b_}", horizon=h, n=len(x), mean_diff=x.mean(), hac_t=Q.hac_t(x.values, lag), ci_lo=lo, ci_hi=hi, share_pos=(x > 0).mean(), block=blk, hac_lag=lag))
PT = pd.DataFrame(PT); PT.to_csv(R / "a1v2_paired_tests.csv", index=False, encoding="utf-8-sig")
json.dump(dict(definitions=__doc__, mergers=MERGERS, n_marked=len(MK), v1_files_kept="a1_*.csv = v1 (assumed defs: B0=pool,B1=raw lowvol,B2=lowvol+cons+buffer); a1v2_* = corrected t17u defs"),
          open(R / "a1v2_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pd.set_option("display.width", 300)
print(S.round(4).to_string()); print(PT.round(4).to_string()); print(MK.round(4).to_string()); print(SE.round(4).to_string())
print(Y.pivot_table(index=["horizon", "year"], columns="strategy", values="cum_net").round(3).to_string()); print("secs", round(time.time() - t0))
