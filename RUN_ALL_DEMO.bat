@echo off
cd "%~dp0"
if exist mt5_creds.local.bat call mt5_creds.local.bat

call .venv\Scripts\activate
set PYTHONPATH=src

python -c "import pandas, flask, MetaTrader5" 2>nul
if errorlevel 1 (
  echo Installing dependencies...
  pip install -r requirements.txt
)

start "OmegaFX Runner" cmd /k "call .venv\Scripts\activate && set PYTHONPATH=src && python scripts\run_fastpass_usdjpy_live.py --mode demo --portfolio multi"
start "OmegaFX Watchdog" cmd /k "call .venv\Scripts\activate && set PYTHONPATH=src && python scripts\watchdog_runner.py"
python dashboard\app.py
pause
