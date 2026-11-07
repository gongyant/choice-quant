from pathlib import Path
p=Path('p0lib.py'); s=p.read_text(encoding='utf-8-sig')
old_a=s.index('    X.has_row = np.isfinite(X.C)'); old_b=s.index('def ffill_mat')
tail=s[old_a:old_b]
new_tail='''    return finalize(X)
def finalize(X):
    """Derived masks from the raw matrices (shared by build_data and the synthetic unit-test fixtures)."""
    u, cal = X.u, X.cal
    X.has_row = np.isfinite(X.C)
    X.traded = X.has_row & (X.V > 0) & (X.C > 0)
    X.ADJ_unadj_based = np.where(X.has_row, X.C / X.F, np.nan)   # corporate-action adjusted close (dividends reinvested)
    # historical listing mask (P0-D): list_date <= t and (delist_date > t or none) -- uses only dated facts
    ld = u.list_date.values.astype("datetime64[ns]"); dd = u.delist_date.values.astype("datetime64[ns]"); cd = cal.values[:, None]
    X.listed = (np.isnat(ld)[None, :] | (cd >= ld[None, :])) & (np.isnat(dd)[None, :] | (cd < dd[None, :]))
    return X
'''
s=s[:old_a]+new_tail+s[old_b:]
# train_rows helper before simulate
tr_fn='''def train_rows(Xq, yw, m_tr, cal, i, years=3):
    """Training rows available at signal index i: label windows (t+1..t+6) must END before i-5 (purge): train dates d <= i-12; 3-year lookback."""
    lo = cal.searchsorted(cal[i] - pd.Timedelta(days=365 * years)); hi = i - 12
    rows = [(Xq[d][m_tr[d]], yw[d][m_tr[d]]) for d in range(lo, hi + 1) if m_tr[d].any()]
    if not rows: return np.zeros((0, Xq.shape[2])), np.zeros(0)
    return np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows])
def simulate('''
s=s.replace('def simulate(',tr_fn,1)
p.write_text(s,encoding='utf-8')
b=Path('baseline_fixed.py'); t=b.read_text(encoding='utf-8-sig')
t=t.replace('''            lo = cal.searchsorted(cal[i] - pd.Timedelta(days=3 * 365)); hi = i - 12   # label end (t+6) must be < signal-5  => purge
            rows = [(Xq[d][m_tr[d]], yw[d][m_tr[d]]) for d in range(lo, hi + 1) if m_tr[d].any()]
            Xt = np.concatenate([r[0] for r in rows]); yt = np.concatenate([r[1] for r in rows])
''','''            Xt, yt = L.train_rows(Xq, yw, m_tr, cal, i)   # label end (t+6) must be < signal-5  => purge
''')
b.write_text(t,encoding='utf-8')
