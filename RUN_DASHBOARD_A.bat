@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
set PYTHONPATH=src
if exist "%~dp0omega_secrets.local.bat" (
  call "%~dp0omega_secrets.local.bat"
) else (
  echo Missing omega_secrets.local.bat in %~dp0
)
set ACCOUNT_ID=A
set LOG_ROOT=logs\acct_A
set DASH_PORT=5000
python dashboard\app.py
pause
