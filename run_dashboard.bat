@echo off
cd "%~dp0"
call .venv\Scripts\activate
python dashboard\app.py
pause
