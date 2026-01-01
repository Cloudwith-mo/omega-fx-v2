@echo off
cd /d "%~dp0"
cd "%~dp0"
call .venv\Scripts\activate
python dashboard\app.py
pause
