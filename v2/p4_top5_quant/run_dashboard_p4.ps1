# P4 Top5 量化选股仪表盘: http://127.0.0.1:8504 (研究用,不连接券商、不下单)
param([switch]$NoBrowser)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path; $py = "D:\choice_quant_starter\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONWARNINGS = "ignore"; Set-Location (Join-Path $here "dashboard")
Write-Host "P4 dashboard: http://127.0.0.1:8504  (Ctrl+C to stop)"
if (-not $NoBrowser) { Start-Job -ScriptBlock { Start-Sleep 6; Start-Process "http://127.0.0.1:8504" } | Out-Null }
& $py -m streamlit run "p4_app.py" --server.address 127.0.0.1 --server.port 8504 --server.headless true --browser.gatherUsageStats false
