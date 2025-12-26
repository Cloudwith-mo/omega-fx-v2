@echo off
cd /d "%~dp0"
start "ENGINE_A" cmd /c "%~dp0RUN_ENGINE_DEMO_A.bat"
start "DASH_A" cmd /c "%~dp0RUN_DASHBOARD_A.bat"
start "ENGINE_B" cmd /c "%~dp0RUN_ENGINE_DEMO_B.bat"
start "DASH_B" cmd /c "%~dp0RUN_DASHBOARD_B.bat"
echo Dashboards:
echo   http://127.0.0.1:5000 (Acct A)
echo   http://127.0.0.1:5001 (Acct B)
pause
