$env:PYTHONIOENCODING="utf-8"; $py="D:\choice_quant_starter\.venv_cuda\Scripts\python.exe"; Set-Location $PSScriptRoot
foreach ($j in @("E2 5","E2 20","E1 5","E1 20")) { $a=$j.Split(" "); & $py p4_sm_train.py $a[0] $a[1] *> "logs\sm_$($a[0])_h$($a[1]).log" }
"SMDONE" | Out-File logs\batch3_train_done.txt
