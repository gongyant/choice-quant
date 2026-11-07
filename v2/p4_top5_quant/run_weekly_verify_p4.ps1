# P4 每周前向核验(研究用,不连接券商、不下单)。用法: .\run_weekly_verify_p4.ps1 20261029
param([string]$End = (Get-Date).ToString("yyyyMMdd"))
Set-Location $PSScriptRoot; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONWARNINGS = 'ignore'
& "D:\choice_quant_starter\v2\p3_ai_report\run_weekly_verify.ps1" $End     # P3 抓取+合并+P3 核验(只追加)
& "D:\choice_quant_starter\.venv\Scripts\python.exe" p4_verify_forward.py   # P4: S4 与低波动 Top5(只追加)
