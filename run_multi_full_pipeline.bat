@echo off
cd /d "%~dp0"
cd "%~dp0"
call .venv\Scripts\activate
set PYTHONPATH=src
set FTMO_NUM_RUNS=300
set FTMO_MAX_DAYS=90
set FTMO_TOTAL_HISTORY_DAYS=365
rem Load MT5 credentials from a local, git-ignored file if present:
rem   copy .\mt5_creds.local.bat.template -> .\mt5_creds.local.bat
if exist .\mt5_creds.local.bat call .\mt5_creds.local.bat
python scripts\demo_ftmo_multi_usdjpy_gbpjpy_full_pipeline_mt5.py
pause
