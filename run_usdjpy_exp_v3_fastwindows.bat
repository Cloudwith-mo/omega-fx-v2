@echo off
cd "%~dp0"
call .venv\Scripts\activate
set PYTHONPATH=src
set USE_REGIME_MASKS=1
set NUM_EVALS=200
set TOTAL_HISTORY_DAYS=270
rem Load MT5 credentials from a local, git-ignored file if present:
rem   copy mt5_creds.local.bat.template -> mt5_creds.local.bat
if exist mt5_creds.local.bat call mt5_creds.local.bat
python scripts\demo_fastpass_usdjpy_exp_v3_fastwindows_mt5.py
pause
