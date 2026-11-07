# P5 每周前向核验(研究用,不连接券商、不下单)。先跑 P4 周核验(其中包含 P3 抓取+合并, 只追加), 再跑 P5 只追加核验。用法: .\run_weekly_verify_p5.ps1 20261030
param([string]$End = (Get-Date).ToString("yyyyMMdd"))
Set-Location $PSScriptRoot; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONWARNINGS = 'ignore'
& "D:\choice_quant_starter\v2\p4_top5_quant\run_weekly_verify_p4.ps1" $End
Set-Location $PSScriptRoot
& "D:\choice_quant_starter\.venv\Scripts\python.exe" p5_verify_forward.py
