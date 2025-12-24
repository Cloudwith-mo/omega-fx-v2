# Omega FX v2

Lean research and execution toolkit for FastPass-style prop challenge testing on FX/metal pairs. Includes multi-edge evaluation, MT5 adapters, and live/shadow runners.

## Quickstart (Windows + PowerShell)
```powershell
cd "C:\Users\muham\Documents\Omega Fx V2\omega-fx-v2\repo_full"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .          # or: pip install -r requirements.txt
```

### Run USDJPY FastPass Core (shadow)
```powershell
$env:PYTHONPATH="src"
$env:OMEGAFX_LIVE_MODE="0"
$env:OMEGAFX_INITIAL_EQUITY="10000"
$env:OMEGAFX_POLL_SECONDS="30"
$env:OMEGAFX_SHADOW_LOG="logs\shadow_fastpass_usdjpy_core.csv"
mkdir logs -ErrorAction SilentlyContinue
python scripts\run_fastpass_usdjpy_live.py
```
Shadow trades are appended to `logs\shadow_fastpass_usdjpy_core.csv`.

### MT5 credentials (Demo/FTMO + MT5 scripts)
Set credentials via env vars (recommended) or use a local bat file:
- Env vars: `OMEGAFX_MT5_LOGIN`, `OMEGAFX_MT5_PASSWORD`, `OMEGAFX_MT5_SERVER` (also supports `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`)
- Bat helper: copy `mt5_creds.local.bat.template` → `mt5_creds.local.bat` and fill your credentials (git-ignored)

### Generate a consolidated FastPass research report
```powershell
$env:PYTHONPATH="src"
python scripts\report_fastpass_research.py > reports\fastpass_research_report.md
```
Optional: set `FASTPASS_RUN_EVALS=1` to re-run light MT5 evals during report generation.

## Version
- Current frozen version: `FastPass_V3_multi_0.1` (see VERSION.md)
- Includes USDJPY FastPass V3, GBPJPY Core, multi FTMO pipelines, and dashboard wiring.
