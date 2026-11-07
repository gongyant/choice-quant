from pathlib import Path
p=Path('p0lib.py'); s=p.read_text(encoding='utf-8')
s=s.replace("def simulate(X, ADJf, sig_idx, targets, cost_rt, capital=1e7, lot=100, log_name=None):","def simulate(X, ADJf, sig_idx, targets, cost_rt, capital=1e7, lot=100, log_name=None, haircut=0.0):")
s=s.replace('''    navs_pre, per, logs = [], [], []
''','''    navs_pre, per, logs = [], [], []
    last_bar = np.where(X.has_row.any(0), T - 1 - np.argmax(X.has_row[::-1], axis=0), -1)
    end_map = {}
    if haircut > 0:   # delisting stress: a position still held when its stock's data ends (delisted, never sellable again) is written down once by `haircut` the day after the final bar
        for j in np.where((last_bar >= 0) & (last_bar < T - 1))[0]: end_map.setdefault(int(last_bar[j]) + 1, []).append(j)
''',1)
s=s.replace('''        k = order_k.get(d)
        if k is not None:
            val = np.nan_to_num''','''        k = order_k.get(d)
        for j in end_map.get(d, []):
            if units[j] > 0: units[j] *= (1.0 - haircut)
        if k is not None:
            val = np.nan_to_num''',1)
s=s.replace('''    last_bar = np.where(X.has_row.any(0), T - 1 - np.argmax(X.has_row[::-1], axis=0), -1)
    stuck =''','''    stuck =''')
p.write_text(s,encoding='utf-8')
