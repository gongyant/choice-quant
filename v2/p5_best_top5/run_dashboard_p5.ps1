# P5 Top5 研究仪表盘: http://127.0.0.1:8505 (研究用,不连接券商、不下单;P4 的 8504 不受影响)
param([switch]$NoBrowser)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path; $py = "D:\choice_quant_starter\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONWARNINGS = "ignore"; Set-Location (Join-Path $here "dashboard")
Write-Host "P5 dashboard: http://127.0.0.1:8505  (Ctrl+C to stop)"
if (-not $NoBrowser) { Start-Job -ScriptBlock { Start-Sleep 6; Start-Process "http://127.0.0.1:8505" } | Out-Null }
& $py -m streamlit run "p5_app.py" --server.address 127.0.0.1 --server.port 8505 --server.headless true --browser.gatherUsageStats false
