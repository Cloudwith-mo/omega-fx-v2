@echo off
cd /d "%~dp0"

if exist "%~dp0omega_secrets.local.bat" (
  call "%~dp0omega_secrets.local.bat"
) else (
  echo Missing omega_secrets.local.bat in %~dp0
)

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

rem Research knobs (edit as needed)
set OMEGAFX_ACTIVE_EDGE_TAG=FixFade
set OMEGAFX_ACTIVE_SYMBOLS=EURUSD
set OMEGAFX_SESSION_WINDOW=13:00-16:00
set OMEGAFX_PORTFOLIO=research

echo ACCOUNT_ID=%ACCOUNT_ID%
echo LOG_ROOT=%LOG_ROOT%
echo MT5_PATH=%MT5_PATH%
echo OMEGAFX_ACTIVE_EDGE_TAG=%OMEGAFX_ACTIVE_EDGE_TAG%
echo OMEGAFX_ACTIVE_SYMBOLS=%OMEGAFX_ACTIVE_SYMBOLS%
echo OMEGAFX_SESSION_WINDOW=%OMEGAFX_SESSION_WINDOW%

start "Omega Engine A (Research)" cmd /k "call .venv\\Scripts\\activate && set PYTHONPATH=src && python scripts\\run_fastpass_usdjpy_live.py --mode demo --portfolio research"
start "Omega Dashboard A" "%~dp0RUN_DASHBOARD_A.bat"

echo Dash A: http://127.0.0.1:5000
pause
