$env:PYTHONIOENCODING='utf-8'; $env:PYTHONWARNINGS='ignore'; $env:PYTHONDONTWRITEBYTECODE='1'
$py='D:\choice_quant_starter\.venv\Scripts\python.exe'
Set-Location D:\choice_quant_starter\v2\p2_alpha_monetization
& $py -m pytest ..\p0\leakage_unit_tests -q -p no:cacheprovider 2>&1 | Out-File -Encoding utf8 tests\pytest_p0_rerun_c.txt
& $py -m pytest ..\p1_dl\tests -q -p no:cacheprovider 2>&1 | Out-File -Encoding utf8 tests\pytest_p1_rerun_c.txt
& $py -m pytest tests\test_p2a.py -q -p no:cacheprovider 2>&1 | Out-File -Encoding utf8 tests\pytest_p2a_rerun_c.txt
& $py -m pytest tests\test_p2b.py -q -p no:cacheprovider 2>&1 | Out-File -Encoding utf8 tests\pytest_p2b_rerun_c.txt
'done' | Out-File results\c_regress.done