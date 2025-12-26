from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
_log_root = os.getenv("LOG_ROOT")
if _log_root:
    _log_root_path = Path(_log_root)
    if not _log_root_path.is_absolute():
        _log_root_path = ROOT / _log_root_path
else:
    _log_root_path = ROOT / "logs"
LOG_DIR = _log_root_path
STATE_FILE = LOG_DIR / "state.json"
HEARTBEAT_FILE = LOG_DIR / "runner_heartbeat.txt"
WATCHDOG_LOG = LOG_DIR / "watchdog.log"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    WATCHDOG_LOG.write_text(
        (WATCHDOG_LOG.read_text(encoding="utf-8") if WATCHDOG_LOG.exists() else "") + line,
        encoding="utf-8",
    )


def read_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_heartbeat() -> dict | None:
    if not HEARTBEAT_FILE.exists():
        return None
    raw = HEARTBEAT_FILE.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    parts = raw.split("|")
    if len(parts) < 2:
        return None
    try:
        return {
            "time": float(parts[0]),
            "equity": float(parts[1]),
            "portfolio": parts[2] if len(parts) >= 3 else None,
            "mode": parts[3] if len(parts) >= 4 else None,
            "account_id": parts[4] if len(parts) >= 5 else None,
        }
    except Exception:
        return None


def spawn_runner(mode: str, portfolio: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["OMEGAFX_MODE"] = mode
    env["OMEGAFX_PORTFOLIO"] = portfolio
    cmd = [
        os.environ.get("PYTHON", "python"),
        str(ROOT / "scripts" / "run_fastpass_usdjpy_live.py"),
        "--mode",
        mode,
        "--portfolio",
        portfolio,
    ]
    subprocess.Popen(cmd, cwd=ROOT, env=env)


def main() -> None:
    mode = os.getenv("OMEGAFX_MODE", "demo")
    portfolio = os.getenv("OMEGAFX_PORTFOLIO", "multi")
    heartbeat_seconds = int(os.getenv("OMEGAFX_HEARTBEAT_SECONDS", "30"))
    stale_seconds = int(os.getenv("OMEGAFX_WATCHDOG_STALE_SECONDS", str(heartbeat_seconds * 4)))
    interval = int(os.getenv("OMEGAFX_WATCHDOG_INTERVAL", "15"))
    min_restart_gap = int(os.getenv("OMEGAFX_WATCHDOG_MIN_RESTART", "60"))

    last_restart = 0.0
    log(f"WATCHDOG START | mode={mode} | portfolio={portfolio} | stale={stale_seconds}s")

    while True:
        state = read_state()
        state_val = (state.get("state") or "").upper()
        hb = read_heartbeat()
        now = time.time()
        hb_age = (now - hb["time"]) if hb else None

        if state_val != "STOPPED":
            stale = hb_age is None or hb_age > stale_seconds
            if stale and now - last_restart >= min_restart_gap:
                log(f"HEARTBEAT STALE | age={hb_age} | restarting runner")
                spawn_runner(mode, portfolio)
                last_restart = now

        time.sleep(interval)


if __name__ == "__main__":
    main()
