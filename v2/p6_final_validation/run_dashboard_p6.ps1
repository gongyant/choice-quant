param([switch]$NoBrowser)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path; $py = "D:\choice_quant_starter\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"; Set-Location (Join-Path $here "dashboard")
Write-Host "P6 dashboard: http://127.0.0.1:8506  (Ctrl+C to stop)"
if (-not $NoBrowser) { Start-Job -ScriptBlock { Start-Sleep 6; Start-Process "http://127.0.0.1:8506" } | Out-Null }
& $py -m streamlit run "p6_app.py" --server.address 127.0.0.1 --server.port 8506 --server.headless true --browser.gatherUsageStats false
