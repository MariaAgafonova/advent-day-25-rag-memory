$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
& "$PSScriptRoot\.venv\Scripts\python.exe" -m streamlit run ui/app.py --server.port 8505 --server.address 127.0.0.1
exit $LASTEXITCODE
