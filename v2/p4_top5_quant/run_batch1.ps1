$env:PYTHONIOENCODING="utf-8"; $py="D:\choice_quant_starter\.venv\Scripts\python.exe"; Set-Location $PSScriptRoot
& $py p4_build.py *> logs\build.log
foreach ($h in 5,20) { foreach ($m in "E1lrank","E1xendcg","E1reg") { & $py p4_train.py $m $h *> "logs\train_${m}_h$h.log" } }
& $py p4_eval.py *> logs\eval.log
"ALLDONE" | Out-File logs\batch1_done.txt
