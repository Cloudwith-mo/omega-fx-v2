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
set OMEGAFX_PASS_SYMBOLS=USDJPY,GBPJPY,EURUSD,GBPUSD
set OMEGAFX_SHADOW_SHOTS=1
set OMEGAFX_SCORERS_RISK_PER_TRADE_PCT=0.001
set OMEGAFX_SCORERS_DAILY_LOSS_PCT=0.01
set OMEGAFX_SCORERS_MAX_OPEN_POSITIONS=4
set OMEGAFX_SCORERS_MAX_TOTAL_OPEN_RISK_PCT=0.01
set OMEGAFX_SCORERS_BUCKET_CAPS=JPY=1
set OMEGAFX_WESTBROOK_DAILY_LOSS_PCT=0.02
set OMEGAFX_WESTBROOK_MAX_OPEN_POSITIONS=6
set OMEGAFX_WESTBROOK_MAX_TOTAL_OPEN_RISK_PCT=0.0075
set OMEGAFX_WESTBROOK_BUCKET_CAPS=JPY=1,USD=1
set OMEGAFX_PORTFOLIO_OPEN_RISK_CAP_PCT=0.02
echo ACCOUNT_ID=%ACCOUNT_ID%
echo LOG_ROOT=%LOG_ROOT%
echo MT5_PATH=%MT5_PATH%
echo OMEGAFX_PASS_SYMBOLS=%OMEGAFX_PASS_SYMBOLS%
python scripts\run_fastpass_usdjpy_live.py --mode demo --portfolio combined
pause
