@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
set PYTHONPATH=src
if exist "%~dp0mt5_creds.A.local.bat" (
  call "%~dp0mt5_creds.A.local.bat"
) else (
  echo Missing mt5_creds.A.local.bat in %~dp0
)
if exist "%~dp0mt5_expect.A.local.bat" (
  call "%~dp0mt5_expect.A.local.bat"
) else (
  echo Missing mt5_expect.A.local.bat in %~dp0
)
set ACCOUNT_ID=A
set LOG_ROOT=logs\acct_A
set MT5_PATH=C:\MT5_A\terminal64.exe
echo ACCOUNT_ID=%ACCOUNT_ID%
echo LOG_ROOT=%LOG_ROOT%
echo MT5_PATH=%MT5_PATH%
python scripts\run_fastpass_usdjpy_live.py --mode demo --portfolio multi
pause
