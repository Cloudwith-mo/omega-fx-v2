@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
set PYTHONPATH=src
call mt5_creds.A.local.bat
set ACCOUNT_ID=A
set LOG_ROOT=logs\acct_A
set DASH_PORT=5000
python dashboard\app.py
pause
