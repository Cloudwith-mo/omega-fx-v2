@echo off
cd /d "%~dp0"
cd "%~dp0"
if exist .\mt5_creds.local.bat call .\mt5_creds.local.bat

call .venv\Scripts\activate
set PYTHONPATH=src

python scripts\watchdog_runner.py
pause
