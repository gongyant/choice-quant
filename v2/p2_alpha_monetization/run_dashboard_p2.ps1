# Starts the read-only P2 research dashboard at http://127.0.0.1:8502 (Ctrl+C to stop). P1 dashboards (8501) are untouched. Does not place orders.
# -NoBrowser : do not auto-open the browser (used by the smoke test).
param([switch]$NoBrowser)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $here "..\..\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONWARNINGS = "ignore"
Set-Location $here
Write-Host "P2 dashboard: http://127.0.0.1:8502  (Ctrl+C to stop)"
if (-not $NoBrowser) { Start-Job -ScriptBlock { Start-Sleep 6; Start-Process "http://127.0.0.1:8502" } | Out-Null }
& $py -m streamlit run stock_dashboard_p2.py --server.address 127.0.0.1 --server.port 8502 --server.headless true --browser.gatherUsageStats false