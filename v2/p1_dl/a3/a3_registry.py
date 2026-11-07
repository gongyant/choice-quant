"""P1-A3: forward-validation registry. Freezes the 2026-10-16 predictions (Top50 per model + file hashes). Created once; do NOT regenerate after 2026-10-19 data arrives."""
import json, hashlib, datetime as dt
from pathlib import Path
import pandas as pd, numpy as np
H = Path(__file__).resolve().parent.parent; V2 = H.parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = H / "forward_registry_20261016.csv"; man = H / "forward_registry_20261016_manifest.json"
assert not out.exists() and not man.exists(), "registry already frozen - refusing to overwrite"
v2p = H / "latest_all_stocks_20261016_v2.csv"; d = pd.read_csv(v2p, dtype=str, encoding="utf-8-sig", keep_default_na=False); e = d[d.eligible_asof == "True"].copy(); h2 = sha(v2p)
MV = {"hgb": "HGB|p1A1-2026-10-16-seed42-train2023-10-17_2026-09-22", "lgbm": "LGBM|p1A1-2026-10-16-seed42-train2023-10-17_2026-09-22", "ridge": "Ridge|p1A1-2026-10-16-seed42-train2023-10-17_2026-09-22",
      "dl_mlp": "MLP|mlp_latest(3seeds mean) train_end 2026-06-16 val_end 2026-09-22", "dl_gru": "GRU|gru_latest(3seeds mean,GPU) train_end 2026-06-16 val_end 2026-09-22", "dl_tf": "TF|tf_latest(seed42 only) train_end 2026-06-16 val_end 2026-09-22"}
rows = []; now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
for k, mv in MV.items():
    m, v = mv.split("|", 1); t = e.assign(r=pd.to_numeric(e[f"rank_{k}"])).sort_values("r").head(50)
    for r in t.itertuples(): rows.append(dict(asof_date="2026-10-16", model=m, model_version=v, rank=int(r.r), code=r.code, name=r.name, score=getattr(r, f"{k}_score"), n_eligible_pool=len(e), planned_label_start="2026-10-19(t+1收盘)", planned_label_end="2026-10-26(t+6收盘)",
                                              latest_limit_up_locked=("最新日涨停封板" in r.quality_flags), st_name=("ST" in r.name), source_table_sha256=h2, registry_created_at=now))
R = pd.DataFrame(rows); assert len(R) == 300 and R.groupby("model").code.nunique().eq(50).all(); R.to_csv(out, index=False, encoding="utf-8-sig")
files = [v2p, H / "latest_all_stocks_20261016.csv", H / "scores" / "dl_mlp_scores_20261016.csv", H / "scores" / "dl_gru_scores_20261016.csv", H / "scores" / "dl_tf_scores_20261016.csv", H / "scores" / "score_run_info.json", H / "models" / "model_meta_2026-10-16.json"] + sorted((H / "models").glob("*_frozen_2026-10-16.joblib")) + \
        sorted((H / "dl" / "checkpoints").glob("*/*_latest_*_r999.pt")) + [V2 / "p0" / "p0lib.py", H / "p1lib.py", H / "dl" / "dl_common.py", H / "dl" / "dl_mlp.py", H / "dl" / "dl_seq.py"]
M = dict(created_at=now, asof_signal_date="2026-10-16", status="FROZEN: this version's predictions must not be changed or re-run with different parameters. Later retrained models are NEW versions and must be registered separately.",
         protocol=dict(label="P0 标签:信号日t收盘后给出排名;t+1收盘(2026-10-19)买入 -> t+6收盘(2026-10-26)卖出的复权收益,减去同日合格池等权均值(相对收益)",
                       first_label_available_after="2026-10-26 收盘后(需要 2026-10-19 起的新数据;2026-10-16 之前的数据不得用于评估本冻结版本)",
                       evaluation="预先声明:Rank IC(分数 vs 相对收益)、Top50 等权相对收益、扣 0.30% 往返成本后的净超额、同制度等权基准;报告 HAC(4)/bootstrap,不在看到结果后改模型/参数/阈值;6个模型存在多重比较,不做事后择优",
                       min_periods_note="至少累积 ≥30 个独立5日期(约 150 个交易日,2027-05 前后)才有初步统计功效;此前仅描述",
                       caveats=["研究排名≠买入建议;score 非概率非收益", "样本为1500只分层抽样,新上市股票不在其中", "涨停封板标记的股票次日大概率无法买入(latest_limit_up_locked)"]),
         n_rows=len(R), models=list(MV.values()), sha256={str(p.relative_to(H.parent)).replace("/", "\\"): sha(p) for p in files})
json.dump(M, open(man, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(R.groupby("model").size().to_dict(), h2, len(files), now)
