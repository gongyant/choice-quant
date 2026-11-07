# -*- coding: utf-8 -*-
"""Register the low-volatility factor Top5 at the last complete day (same tradable filter as P4) so it can be verified forward alongside S4."""
import json, hashlib, pickle, os, stat, datetime as dt
import numpy as np, pandas as pd
import p4lib as P
C = P.P4 / "cache"; REG = P.P4 / "predictions_registry"; fl = np.load(C / "flags.npz"); X = pickle.load(open(C / "world.pkl", "rb")); cal = list(fl["cal"]); i = len(cal) - 1
out = REG / f"p4_pred_{cal[i].replace('-', '')}_LOWVOL_top5.csv"; assert not out.exists()
u = X.u; cur = (u.is_st_name.fillna(False) | u.name_has_delist_mark.fillna(False)).values.astype(bool)
A = pd.DataFrame(np.where(X.traded, X.C / X.F, np.nan)).ffill(); vol = A.pct_change(fill_method=None).rolling(20, min_periods=15).std().values[i]
f = fl["elig"][i] & X.traded[i] & ~X.lim_up[i] & (np.nan_to_num(fl["amt20"][i]) >= 1e8) & ~fl["st"][i] & ~cur & np.isfinite(vol)
o = np.where(f)[0][np.argsort(vol[f])][:5]
d = pd.DataFrame(dict(rank=range(1, 6), code=[X.codes[j] for j in o], name=u.name.values[o], board=u.board.values[o], close_unadj=X.C[i, o], vol20=vol[o], score=-vol[o], in_final5=True, meets_buy_criteria=False,
                      signal_date=cal[i], strategy="因子_低波动", note="对照策略,研究用,不是买入推荐"))
d.to_csv(out, index=False, encoding="utf-8-sig"); sh = hashlib.sha256(out.read_bytes()).hexdigest(); os.chmod(out, stat.S_IREAD)
open(REG / "manifest.jsonl", "a", encoding="utf-8").write(json.dumps(dict(file=out.name, sha256=sh, model="LOWVOL", horizon="5,20", signal_date=cal[i], buy_gate_passed=False, written_at_local=dt.datetime.now().isoformat(timespec="seconds"), exploratory=True), ensure_ascii=False) + "\n")
print(out.name, sh); print(d.to_string())
