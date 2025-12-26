@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
set PYTHONPATH=src
call mt5_creds.B.local.bat
set ACCOUNT_ID=B
set LOG_ROOT=logs\acct_B
set MT5_PATH=C:\MT5_B\terminal64.exe
python scripts\run_fastpass_usdjpy_live.py --mode demo --portfolio multi
pause
