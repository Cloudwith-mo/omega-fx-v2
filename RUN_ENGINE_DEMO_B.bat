@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
set PYTHONPATH=src
if exist "%~dp0mt5_creds.B.local.bat" (
  call "%~dp0mt5_creds.B.local.bat"
) else (
  echo Missing mt5_creds.B.local.bat in %~dp0
)
if exist "%~dp0mt5_expect.B.local.bat" (
  call "%~dp0mt5_expect.B.local.bat"
) else (
  echo Missing mt5_expect.B.local.bat in %~dp0
)
set ACCOUNT_ID=B
set LOG_ROOT=logs\acct_B
set MT5_PATH=C:\MT5_B\terminal64.exe
python scripts\run_fastpass_usdjpy_live.py --mode demo --portfolio multi
pause
