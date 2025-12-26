@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
set PYTHONPATH=src
call mt5_creds.A.local.bat
set ACCOUNT_ID=A
set LOG_ROOT=logs\acct_A
set MT5_PATH=C:\MT5_A\terminal64.exe
python scripts\run_fastpass_usdjpy_live.py --mode demo --portfolio multi
pause
