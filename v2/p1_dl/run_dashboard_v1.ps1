# Starts the read-only dashboard at http://127.0.0.1:8501 (Ctrl+C to stop). Does not place orders or touch other projects.
# -NoBrowser : do not auto-open the browser (used by the smoke test).
param([switch]$NoBrowser)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $here "..\..\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONWARNINGS = "ignore"
Set-Location $here
Write-Host "Dashboard: http://127.0.0.1:8501  (Ctrl+C to stop)"
if (-not $NoBrowser) { Start-Job -ScriptBlock { Start-Sleep 6; Start-Process "http://127.0.0.1:8501" } | Out-Null }
& $py -m streamlit run stock_dashboard.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
