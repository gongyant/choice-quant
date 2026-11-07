"""P1 步骤3：对约1500只股票做分层抽样；保存方法与随机种子。
分层 = 板块(沪主板/科创板/深主板/深主板原中小板/创业板/北交所) x 巨潮行业门类 x 流动性三分位(最近交易日成交额，仅用于分层)。
2021-01-01以后退市的A股全部纳入(降低幸存者偏差)；其余按层比例分配、每层至少1只(若可能)。
B股不在抽样范围(币种与交易机制不同)，但保留在证券主表中。
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
V2 = Path(__file__).resolve().parent; DATA = V2 / "data"; REP = V2 / "reports"
SEED = 20261016; TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
u = pd.read_csv(DATA / "universe_master_base.csv", dtype={"code": str})
ind = pd.read_csv(DATA / "industry_cninfo.csv", dtype={"code": str}).drop_duplicates("code", keep="last")
u = u.merge(ind, on="code", how="left")
u["delist_date"] = pd.to_datetime(u.delist_date, errors="coerce")
u["list_date"] = pd.to_datetime(u.list_date, errors="coerce")
el = u[~u.is_b_share].copy()
el = el[(el.status == "上市") | (el.delist_date >= "2021-01-01")].copy()
el["industry_l1"] = el.industry_l1.fillna("未分类")
# 流动性分层：按板块内成交额三分位；无成交额或为0单独成层
def liq(g):
    a = g.spot_amount.fillna(0)
    lab = pd.Series("L0无成交/缺失", index=g.index)
    pos = a > 0
    if pos.sum() >= 3:
        q = pd.qcut(a[pos].rank(method="first"), 3, labels=["L1低", "L2中", "L3高"])
        lab[pos] = q.astype(str)
    return lab
el["liq_stratum"] = el.groupby("board", group_keys=False).apply(liq)
el["stratum"] = el.board + "|" + el.industry_l1 + "|" + el.liq_stratum
dele = el[el.status != "上市"]
lst = el[el.status == "上市"].copy()
n_target = TARGET - len(dele)
sizes = lst.groupby("stratum").size()
raw = sizes * n_target / sizes.sum()
alloc = np.maximum(1, np.floor(raw)).astype(int)
rng = np.random.default_rng(SEED)
# 最大余数法微调到n_target
diff = n_target - alloc.sum()
if diff > 0:
    order = (raw - np.floor(raw)).sort_values(ascending=False).index
    for s in order[:diff]:
        if alloc[s] < sizes[s]: alloc[s] += 1
elif diff < 0:
    order = (alloc - raw).sort_values(ascending=False).index
    k = 0
    while diff < 0 and k < len(order) * 5:
        s = order[k % len(order)]
        if alloc[s] > 1: alloc[s] -= 1; diff += 1
        k += 1
alloc = np.minimum(alloc, sizes)
picked = []
for s, g in lst.groupby("stratum"):
    idx = rng.choice(len(g), size=int(alloc[s]), replace=False)
    picked.append(g.iloc[sorted(idx)])
samp = pd.concat(picked + [dele]).copy()
samp["sample_seed"] = SEED
samp["sample_group"] = np.where(samp.status == "上市", "分层抽样(在市)", "全部纳入(2021年后退市)")
samp = samp.sort_values(["exchange", "code"]).reset_index(drop=True)
samp.to_csv(DATA / f"sample_{TARGET}.csv", index=False, encoding="utf-8-sig")
el.to_csv(DATA / "universe_master.csv", index=False, encoding="utf-8-sig")
rep = {"seed": SEED, "target": TARGET, "sample_size": int(len(samp)), "eligible_universe": int(len(el)),
       "delisted_included": int(len(dele)), "strata_count": int(el.stratum.nunique()),
       "by_board_sample": samp.board.value_counts().to_dict(), "by_board_universe": el.board.value_counts().to_dict(),
       "by_liq_sample": samp.liq_stratum.value_counts().to_dict(),
       "by_industry_sample": samp.industry_l1.value_counts().to_dict(),
       "by_industry_universe": el.industry_l1.value_counts().to_dict(),
       "list_year_sample": samp.list_date.dt.year.value_counts().sort_index().astype(int).to_dict(),
       "industry_missing_in_universe": int((el.industry_l1 == "未分类").sum()),
       "method": "board x 巨潮行业门类 x 板块内成交额三分位；比例分配、每层至少1只；numpy default_rng(seed)；退市股全部纳入。"}
with open(REP / f"sampling_report_{TARGET}.json", "w", encoding="utf-8") as f: json.dump(rep, f, ensure_ascii=False, indent=2, default=int)
print(json.dumps(rep, ensure_ascii=False, indent=1, default=int)[:3000])