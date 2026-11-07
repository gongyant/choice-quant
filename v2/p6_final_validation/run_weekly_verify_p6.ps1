param([string]$End = (Get-Date).ToString("yyyyMMdd"))
Set-Location $PSScriptRoot; $env:PYTHONIOENCODING = 'utf-8'
& "D:\choice_quant_starter\v2\p4_top5_quant\run_weekly_verify_p4.ps1" $End
Set-Location $PSScriptRoot
& "D:\choice_quant_starter\.venv\Scripts\python.exe" p6_verify_forward.py
