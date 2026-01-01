@echo off
cd /d "%~dp0"

if exist "%~dp0omega_secrets.local.bat" (
  call "%~dp0omega_secrets.local.bat"
) else (
  echo Missing omega_secrets.local.bat in %~dp0
)

echo Launching Omega B (engine + dashboard)...

start "Omega Engine B" "%~dp0RUN_ENGINE_DEMO_B.bat"
start "Omega Dashboard B" "%~dp0RUN_DASHBOARD_B.bat"

echo Dash B: http://127.0.0.1:5001
pause
