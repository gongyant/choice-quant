# 启动只读「每日AI选股报告」仪表盘: http://127.0.0.1:8503 (Ctrl+C 停止)。不连券商、不下单。-NoBrowser 不自动打开浏览器。
param([switch]$NoBrowser)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = "D:\choice_quant_starter\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONWARNINGS = "ignore"
Set-Location (Join-Path $here "dashboard")
Write-Host "P3 dashboard: http://127.0.0.1:8503  (Ctrl+C to stop)"
if (-not $NoBrowser) { Start-Job -ScriptBlock { Start-Sleep 6; Start-Process "http://127.0.0.1:8503" } | Out-Null }
& $py -m streamlit run "每日AI选股报告.py" --server.address 127.0.0.1 --server.port 8503 --server.headless true --browser.gatherUsageStats false