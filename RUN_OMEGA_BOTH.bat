@echo off
cd /d "%~dp0"
set OMEGAFX_WATCHDOG=1
set OMEGAFX_WATCHDOG_STALE_SECONDS=60
set OMEGAFX_WATCHDOG_POLL_SECONDS=10
start "Omega A" "%~dp0RUN_OMEGA_A.bat"
start "Omega B" "%~dp0RUN_OMEGA_B.bat"
