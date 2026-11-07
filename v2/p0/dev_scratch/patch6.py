from pathlib import Path
b=Path('baseline_fixed.py'); t=b.read_text(encoding='utf-8')
t=t.replace('''    m_tr = elig & np.isfinite(yw)
''','''    m_tr = elig & np.isfinite(yw); isd = (X.u.status.values != "上市")   # P0-D diagnostic only: delisted-in-sample flag (future information, NOT used for selection)
''')
t=t.replace('''        yy = y[i][pool]; fin = np.isfinite(yy); rec = {"signal_date": str(cal[i].date()), "n_pool": int(len(pool))}
        for n in strat:
            s = np.nan_to_num(sc[n], nan=-1e9); top = pool[np.argsort(-s, kind="stable")[:a.topk]]; topk_sets[n].append(top)
''','''        yy = y[i][pool]; fin = np.isfinite(yy); rec = {"signal_date": str(cal[i].date()), "n_pool": int(len(pool))}
        lp = lab[i][pool]; rec["pool_delist_share"] = float(isd[pool].mean()); rec["pool_label_listed"] = float(np.nanmean(lp[~isd[pool]])); rec["pool_label_delisted"] = float(np.nanmean(lp[isd[pool]])) if isd[pool].any() else np.nan
        for n in strat:
            s = np.nan_to_num(sc[n], nan=-1e9); top = pool[np.argsort(-s, kind="stable")[:a.topk]]; topk_sets[n].append(top)
            lt = lab[i][top]; rec[f"{n}_delist_share"] = float(isd[top].mean()); rec[f"{n}_label_listed"] = float(np.nanmean(lt[~isd[top]])) if (~isd[top]).any() else np.nan; rec[f"{n}_label_delisted"] = float(np.nanmean(lt[isd[top]])) if isd[top].any() else np.nan
''')
b.write_text(t,encoding='utf-8')
