@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
set PYTHONPATH=src
call mt5_creds.B.local.bat
set ACCOUNT_ID=B
set LOG_ROOT=logs\acct_B
set DASH_PORT=5001
python dashboard\app.py
pause
