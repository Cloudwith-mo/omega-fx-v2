@echo off
cd /d "%~dp0"
cd "%~dp0"
call .venv\Scripts\activate
set PYTHONPATH=src
rem Load MT5 credentials from a local, git-ignored file if present:
rem   copy .\mt5_creds.local.bat.template -> .\mt5_creds.local.bat
if exist .\mt5_creds.local.bat call .\mt5_creds.local.bat
set NUM_EVALS=100
set TOTAL_HISTORY_DAYS=270
set USE_REGIME_MASKS=1
set PYTHONUNBUFFERED=1
echo Running EXP_V2 optimizer... please wait. Progress will print every ~10%% of the grid.
python scripts\demo_optimize_usdjpy_exp_v2_risk_mt5.py
echo Finished. Press any key to close.
pause
