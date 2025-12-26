from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _run(cmd: list[str], env: dict[str, str]) -> int:
    print(f"-> {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=ROOT, env=env)


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)

    smoke_cmds = [
        [
            sys.executable,
            "-c",
            (
                "from omegafx_v2.config import "
                "DEFAULT_PORTFOLIO_USDJPY_FASTPASS_CORE, "
                "DEFAULT_PORTFOLIO_USDJPY_FASTPASS_V3, "
                "MULTI_PORTFOLIO_USDJPY_GBPJPY_FASTPASS; "
                "print('config_import_ok')"
            ),
        ],
        [sys.executable, "scripts/run_fastpass_usdjpy_live.py", "--help"],
    ]

    failures = 0
    for cmd in smoke_cmds:
        code = _run(cmd, env)
        if code != 0:
            failures += 1

    if failures:
        print(f"Integrity check failed: {failures} command(s) failed.")
        return 1
    print("Integrity check OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
