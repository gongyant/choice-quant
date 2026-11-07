$py='D:\choice_quant_starter\.venv_cuda\Scripts\python.exe'
& $py -m pip install numpy pandas scipy pyarrow scikit-learn pytest --retries 10 --timeout 60 2>&1 | Out-File -Encoding utf8 D:\choice_quant_starter\v2\p1_dl\dl\logs\cuda_deps.log
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128 --retries 20 --resume-retries 20 --timeout 60 2>&1 | Out-File -Encoding utf8 D:\choice_quant_starter\v2\p1_dl\dl\logs\cuda_torch.log
"DONE $(Get-Date)" | Out-File -Append -Encoding utf8 D:\choice_quant_starter\v2\p1_dl\dl\logs\cuda_torch.log