@echo off
cd /d "%~dp0"

if exist "%~dp0omega_secrets.local.bat" (
  call "%~dp0omega_secrets.local.bat"
) else (
  echo Missing omega_secrets.local.bat in %~dp0
)

echo Launching Omega A Combined (engine + dashboard)...

start "Omega Engine A Combined" "%~dp0RUN_ENGINE_DEMO_A_COMBINED.bat"
start "Omega Dashboard A" "%~dp0RUN_DASHBOARD_A.bat"

echo Dash A: http://127.0.0.1:5000
pause
