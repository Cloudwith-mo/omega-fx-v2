from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _parse_symbols(raw: str) -> List[str]:
    parts: List[str] = []
    for chunk in raw.replace(";", ",").replace("|", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        for token in chunk.split():
            if token:
                parts.append(token.upper())
    return parts


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _progress_bar(pct: float, width: int = 20) -> str:
    filled = int(round((pct / 100.0) * width))
    filled = max(0, min(width, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _tail_lines(path: Path, max_lines: int = 500) -> List[str]:
    if not path.exists():
        return []
    lines: List[str] = []
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        chunk = min(size, 1024 * 1024)
        f.seek(-chunk, 2)
        data = f.read().decode("utf-8", errors="ignore")
    for line in data.splitlines():
        if line.strip():
            lines.append(line.strip())
    return lines[-max_lines:]


def _spot_check_csv(path: Path) -> Tuple[bool, int]:
    lines = _tail_lines(path, max_lines=500)
    if not lines:
        return True, 0
    if lines[0].lower().startswith("time,"):
        lines = lines[1:]
    times: List[datetime] = []
    for line in lines:
        ts = line.split(",", 1)[0].strip()
        try:
            times.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
        except Exception:
            return False, 0
    monotonic_ok = all(times[i] < times[i + 1] for i in range(len(times) - 1))
    duplicates = len(times) - len(set(times))
    return monotonic_ok, duplicates


def _rate_and_eta(sample_prev: Dict[str, object], sample_cur: Dict[str, object]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        t0 = datetime.fromisoformat(sample_prev["ts_utc"])
        t1 = datetime.fromisoformat(sample_cur["ts_utc"])
        dt_min = max((t1 - t0).total_seconds() / 60.0, 0.001)
        dd = float(sample_cur["processed_days"]) - float(sample_prev.get("processed_days", 0))
        db = float(sample_cur.get("size_bytes", 0)) - float(sample_prev.get("size_bytes", 0))
        days_per_min = dd / dt_min
        mb_per_min = (db / (1024 * 1024)) / dt_min
        return days_per_min, mb_per_min, dt_min
    except Exception:
        return None, None, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Report Dukascopy download progress by symbol.")
    parser.add_argument("--root", required=True, help="Root folder that contains per-symbol folders.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols.")
    parser.add_argument("--start", default="2006-01-01", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", default="2025-12-31", help="End date YYYY-MM-DD.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    symbols = _parse_symbols(args.symbols)
    start = datetime.fromisoformat(args.start).date()
    end = datetime.fromisoformat(args.end).date()
    total_days = (end - start).days + 1

    for sym in symbols:
        sym_dir = root / sym
        manifest = _read_json(sym_dir / f"manifest_{sym}.json")
        last_day_done = manifest.get("last_day_done")
        processed_days = int(manifest.get("processed_days") or 0)
        success_days = int(manifest.get("success_days") or 0)
        missing_days = int(manifest.get("missing_days") or 0)
        error_count = int(manifest.get("error_count") or 0)
        pct = (processed_days / total_days) * 100 if total_days else 0.0
        bar = _progress_bar(pct)

        sample_prev = manifest.get("progress_sample_prev") or {}
        sample_cur = manifest.get("progress_sample") or {}
        days_per_min, mb_per_min, _ = _rate_and_eta(sample_prev, sample_cur)
        eta_min = None
        if days_per_min and days_per_min > 0:
            remaining = max(total_days - processed_days, 0)
            eta_min = remaining / days_per_min

        rate_str = "n/a"
        if days_per_min is not None:
            rate_str = f"{days_per_min:.2f} days/min"
        elif mb_per_min is not None:
            rate_str = f"{mb_per_min:.2f} MB/min"

        eta_str = f"{eta_min/60:.1f}h" if eta_min is not None else "n/a"

        err_count = error_count
        if err_count == 0:
            err_log = Path("logs") / f"dukascopy_{sym}.err.log"
            if err_log.exists():
                try:
                    err_count = len([line for line in err_log.read_text(encoding="utf-8").splitlines() if line.strip()])
                except Exception:
                    err_count = 0

        m1_path = sym_dir / f"{sym}_M1.csv"
        monotonic_ok, dup_count = _spot_check_csv(m1_path)

        print(
            f"{sym} {bar} {pct:6.2f}% | last_day={last_day_done} "
            f"| processed={processed_days} success={success_days} missing={missing_days} "
            f"| rate={rate_str} eta={eta_str} | errors={err_count} "
            f"| monotonic_ok={monotonic_ok} dup_ts={dup_count}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
