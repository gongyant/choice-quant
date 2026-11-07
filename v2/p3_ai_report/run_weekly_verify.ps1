# 每周行情更新 + 前向核验(研究用,不连券商、不下单)。用法:在 PowerShell 中运行  .\run_weekly_verify.ps1 20261029
param([string]$End = (Get-Date).ToString("yyyyMMdd"))
$ErrorActionPreference = "Stop"; Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONWARNINGS = 'ignore'; $py = "D:\choice_quant_starter\.venv\Scripts\python.exe"
# 1) 增量抓取(6 个进程分片,单进程单线程)
$ps = 0..5 | ForEach-Object { Start-Process $py -ArgumentList "-u","p3_fetch.py",$End,$_,"6" -NoNewWindow -PassThru -RedirectStandardOutput "logs\wk_fetch_$_.out" -RedirectStandardError "logs\wk_fetch_$_.err" }
$ps | Wait-Process
# 2) 合并到 data\merged(P0 原始文件不动)
& $py p3_merge.py ("{0}-{1}-{2}" -f $End.Substring(0,4), $End.Substring(4,2), $End.Substring(6,2))
# 3) 核验(只追加到 verification\)
& $py p3_verify_forward.py