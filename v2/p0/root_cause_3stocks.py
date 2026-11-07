"""P0-C / P0-Q root-cause analysis for the three invalid-price stocks (600188, 600066, 000408): Tencent raw/qfq vs Sina raw/qfq/factor."""
import pickle, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, '.'); import p0lib as L
pr = pickle.load(open('data/probe3.pkl', 'rb')); out = {}; ev = []
for code, r in pr.items():
    tr, tq, sr, sq, fac = r['tx_raw'].copy(), r['tx_qfq'].copy(), r['sina_raw'].copy(), r['sina_qfq'].copy(), r['sina_qfq-factor'].copy()
    for d in (tr, tq, sr, sq): d['date'] = pd.to_datetime(d['date']); d.sort_values('date', inplace=True); d.reset_index(drop=True, inplace=True)
    fac['date'] = pd.to_datetime(fac['date']); fac['qfq_factor'] = fac.qfq_factor.astype(float)
    for nm, d in (('tx_raw', tr), ('tx_qfq', tq), ('sina_raw', sr), ('sina_qfq', sq)): d.to_parquet(f'data/probe3_{code}_{nm}.parquet', index=False)
    f = L.factor_on_dates(sr.date, fac); adj = sr.close.values / f
    m = pd.DataFrame({'date': sr.date, 'raw': sr.close.values, 'f': f, 'adj_mine': adj}).merge(tq[['date', 'close']].rename(columns={'close': 'tx_qfq'}), on='date').merge(sq[['date', 'close']].rename(columns={'close': 'sina_qfq'}), on='date')
    m['raw_ret'] = m.raw.pct_change(); m['adj_ret'] = m.adj_mine.pct_change(); m['txq_ret'] = m.tx_qfq.pct_change().replace([np.inf, -np.inf], np.nan); m['sinaq_ret'] = m.sina_qfq.pct_change()
    m['tx_offset'] = m.tx_qfq - m.raw
    hl = 'tx raw vs sina raw max|close diff|'
    rawdiff = float((tr.set_index('date').close - sr.set_index('date').close).abs().max())
    neg = m[m.tx_qfq <= 0]
    rate = 0.10
    o = dict(code=code, bars=len(m), tx_raw_vs_sina_raw_max_abs_close_diff=rawdiff,
             tx_qfq_nonpositive_days=int((tq[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()), first_nonpositive=str(neg.date.min().date()) if len(neg) else None, last_nonpositive=str(neg.date.max().date()) if len(neg) else None,
             tx_qfq_min_close=float(m.tx_qfq.min()), tx_qfq_first_close=float(m.tx_qfq.iloc[0]), raw_first_close=float(m.raw.iloc[0]), tx_offset_first=float(m.tx_offset.iloc[0]), tx_offset_last=float(m.tx_offset.iloc[-1]),
             sina_qfq_nonpositive_days=int((sq[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()),
             mine_vs_sina_qfq_max_abs_diff=float((m.adj_mine - m.sina_qfq).abs().max()), mine_nonpositive=int((m.adj_mine <= 0).sum()),
             raw_max_abs_ret=float(m.raw_ret.abs().max()), raw_days_abs_ret_gt_11pct=int((m.raw_ret.abs() > 0.11).sum()),
             txqfq_days_abs_ret_gt_11pct=int((m.txq_ret.abs() > 0.11).sum()), txqfq_max_abs_ret=float(np.nanmax(np.abs(m.txq_ret.replace(np.inf, np.nan)))) if m.txq_ret.notna().any() else None, txqfq_inf_ret_days=int(np.isinf(m.tx_qfq.pct_change()).sum()),
             corrected_max_abs_ret=float(m.adj_ret.abs().max()), corrected_days_abs_ret_gt_10_5pct=int((m.adj_ret.abs() > 0.105).sum()),
             factor_changes_in_window=int((np.diff(f) != 0).sum()))
    # ex-rights days: raw drop vs factor ratio
    chg = np.where(np.diff(f) != 0)[0] + 1
    for k in chg:
        ratio = f[k] / f[k - 1]; ref = m.raw.iloc[k - 1] * ratio
        ev.append(dict(code=code, ex_date=str(m.date.iloc[k].date()), prev_raw_close=m.raw.iloc[k - 1], raw_close=m.raw.iloc[k], factor_prev=f[k - 1], factor_new=f[k], ex_ref_price=round(ref, 3),
                       implied_cash_equiv_per_share=round(m.raw.iloc[k - 1] - ref, 3), raw_ret=m.raw_ret.iloc[k], corrected_ret=m.adj_ret.iloc[k], tx_qfq_ret=m.txq_ret.iloc[k], tx_offset_step=m.tx_offset.iloc[k] - m.tx_offset.iloc[k - 1]))
    out[code] = o
ev = pd.DataFrame(ev); ev.to_csv('results/root_cause_3stocks_exrights_events.csv', index=False, encoding='utf-8-sig')
json.dump(out, open('results/root_cause_3stocks.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False)); print(ev.head(25).to_string())
