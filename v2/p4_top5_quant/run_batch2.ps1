$env:PYTHONIOENCODING="utf-8"; $py="D:\choice_quant_starter\.venv\Scripts\python.exe"; Set-Location $PSScriptRoot
& $py p4_fin_build.py *> logs\fin_build.log
foreach ($h in 5,20) { foreach ($m in "E2reg","E2lrank") { & $py p4_train.py $m $h *> "logs\train_${m}_h$h.log" } }
& $py p4_e2_eval.py *> logs\e2_eval.log
$v = & $py -c "import pandas as pd;s=pd.read_csv('results/bt_summary_e2.csv');s=s[(s.horizon==5)&(s.period=='全期')&s.strategy.str.startswith('S')];print(s.sort_values('mean_excess_net').strategy.iloc[-1][:2])"
& $py p4_latest_e2.py $v *> logs\latest_e2.log
& $py -m pytest -q tests *> logs\pytest_e2.log
"ALLDONE $v" | Out-File logs\batch2_done.txt
