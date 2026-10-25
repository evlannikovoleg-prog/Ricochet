$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "$PSScriptRoot\src"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
python -m uvicorn ricochet.api.main:app --host 127.0.0.1 --port 8077
