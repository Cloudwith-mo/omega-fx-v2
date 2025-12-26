@echo off
cd "%~dp0"
if exist mt5_creds.local.bat call mt5_creds.local.bat

call .venv\Scripts\activate
set PYTHONPATH=src

python -c "import pandas, MetaTrader5" 2>nul
if errorlevel 1 (
  echo Installing dependencies...
  pip install -r requirements.txt
  python -c "import pandas, MetaTrader5" 2>nul
  if errorlevel 1 (
    echo ERROR: Missing deps. Please run: pip install -r requirements.txt
    pause
    exit /b 1
  )
)

python scripts\run_fastpass_usdjpy_live.py --mode demo --portfolio multi
pause
