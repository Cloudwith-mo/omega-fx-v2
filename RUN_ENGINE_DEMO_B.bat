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
set OMEGAFX_PASS_SYMBOLS=USDJPY
set OMEGAFX_MAX_OPEN_POSITIONS=1
set OMEGAFX_BUCKET_CAPS=JPY=1,USD=1
set DYNAMIC_RISK=1
set OMEGAFX_POLICY_DD_COOLDOWN=0.02
set OMEGAFX_POLICY_COOLDOWN_MINUTES=1440
python scripts\run_fastpass_usdjpy_live.py --mode demo --portfolio westbrook
pause
