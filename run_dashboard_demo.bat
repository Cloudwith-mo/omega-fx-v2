@echo off
cd "%~dp0"
rem Load MT5 credentials from a local, git-ignored file if present:
rem   copy mt5_creds.local.bat.template -> mt5_creds.local.bat
if exist mt5_creds.local.bat call mt5_creds.local.bat

call .venv\Scripts\activate
set PYTHONPATH=src
python -c "import flask, pandas" 2>nul
if errorlevel 1 (
  echo Installing dependencies...
  pip install -r requirements.txt
)
python dashboard\app.py
pause
