from __future__ import annotations

import argparse
import csv
import json
import lzma
import os
import struct
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

BASE_URL = "https://datafeed.dukascopy.com/datafeed"


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


def _parse_timeframes(raw: str) -> List[str]:
    parts: List[str] = []
    for chunk in raw.replace(";", ",").replace("|", ",").split(","):
        token = chunk.strip().upper()
        if token:
            parts.append(token)
    return parts


def _iter_dates(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _month_path_component(dt: date) -> str:
    # Dukascopy uses zero-based months (00=Jan)
    return f"{dt.month - 1:02d}"


def _dukascopy_url(symbol: str, day: date, side: str) -> str:
    return (
        f"{BASE_URL}/{symbol}/{day.year}/{_month_path_component(day)}/{day.day:02d}/"
        f"{side}_candles_min_1.bi5"
    )


def _download(url: str, timeout: int, retries: int) -> Optional[bytes]:
    last_err: Optional[Exception] = None
    for _ in range(max(retries, 1)):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            last_err = exc
    if last_err:
        raise last_err
    return None


def _price_scale(symbol: str) -> float:
    return 1000.0 if symbol.upper().endswith("JPY") else 100000.0


def _parse_candles(raw: bytes, day: date, scale: float) -> pd.DataFrame:
    data = lzma.decompress(raw)
    record_size = 24
    if len(data) % record_size != 0:
        raise ValueError("Unexpected candle record size")
    rows = []
    day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    for offset, open_p, close_p, low_p, high_p, volume in struct.iter_unpack(">6i", data):
        ts = day_start + timedelta(seconds=offset)
        rows.append({
            "time": ts,
            "open": open_p / scale,
            "high": high_p / scale,
            "low": low_p / scale,
            "close": close_p / scale,
            "volume": float(volume),
        })
    return pd.DataFrame(rows)


def _write_rows(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["time", "open", "high", "low", "close", "volume"])
        for row in df.itertuples(index=False):
            writer.writerow([
                row.time.isoformat(),
                f"{row.open:.6f}",
                f"{row.high:.6f}",
                f"{row.low:.6f}",
                f"{row.close:.6f}",
                f"{row.volume:.0f}",
            ])


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.set_index("time")
    df.index = pd.to_datetime(df.index, utc=True)
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    out = pd.concat([o, h, l, c, v], axis=1).dropna()
    out.columns = ["open", "high", "low", "close", "volume"]
    out = out.reset_index()
    return out


def _tf_rules(timeframes: Sequence[str]) -> Dict[str, str]:
    mapping = {
        "M1": "1min",
        "M15": "15min",
        "H4": "4h",
        "D1": "1d",
    }
    return {tf: mapping[tf] for tf in timeframes if tf in mapping}


def _manifest_path(out_dir: Path) -> Path:
    return out_dir / "dukascopy_manifest.json"


def _symbol_manifest_path(out_dir: Path, symbol: str) -> Path:
    return out_dir / f"manifest_{symbol}.json"


def _load_manifest(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_manifest(path: Path, data: Dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _last_success_date(entry: Dict[str, object]) -> Optional[date]:
    raw = entry.get("last_success_date")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).date()
    except Exception:
        return None


def _read_last_line(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return None
        offset = min(size, 4096)
        f.seek(-offset, os.SEEK_END)
        chunk = f.read().decode("utf-8", errors="ignore")
    lines = [line for line in chunk.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _last_timestamp_from_csv(path: Path) -> Optional[datetime]:
    line = _read_last_line(path)
    if not line or line.lower().startswith("time,"):
        return None
    try:
        ts = line.split(",", 1)[0].strip()
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _lock_path(out_dir: Path, symbol: str) -> Path:
    return out_dir / f".download_{symbol}.lock"


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _acquire_lock(out_dir: Path, symbol: str, force: bool) -> bool:
    lock_path = _lock_path(out_dir, symbol)
    if lock_path.exists() and not force:
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        pid = payload.get("pid")
        if isinstance(pid, int) and _pid_running(pid):
            print(f"LOCKED: {symbol} already running with pid={pid}")
            return False
    lock_path.write_text(json.dumps({
        "symbol": symbol,
        "pid": os.getpid(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }), encoding="utf-8")
    return True


def _progress_bar(pct: float, width: int = 20) -> str:
    filled = int(round((pct / 100.0) * width))
    filled = max(0, min(width, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _update_symbol_manifest(
    path: Path,
    symbol: str,
    start_date: date,
    end_date: date,
    side: str,
    timeframes: Sequence[str],
    success_days: int,
    missing_days: int,
    processed_days: int,
    last_day_done: Optional[date],
    last_success_day: Optional[date],
    missing_sample: Sequence[str],
    error_count: int,
    last_error: Optional[str],
    sample: Dict[str, object],
) -> None:
    payload = {
        "symbol": symbol,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "side": side,
        "timeframes": list(timeframes),
        "success_days": success_days,
        "missing_days": missing_days,
        "processed_days": processed_days,
        "completed_days": processed_days,
        "last_day_done": last_day_done.isoformat() if last_day_done else None,
        "last_success_date": last_success_day.isoformat() if last_success_day else None,
        "missing_sample": list(missing_sample),
        "error_count": error_count,
        "last_error": last_error,
        "progress_sample_prev": sample.get("prev"),
        "progress_sample": sample.get("current"),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resume_from_existing(
    m1_path: Path,
    start_date: date,
    end_date: date,
) -> Tuple[Optional[date], int]:
    last_ts = _last_timestamp_from_csv(m1_path)
    if last_ts is None:
        return None, 0
    last_day = last_ts.date()
    if last_day < start_date or last_day > end_date:
        return None, 0
    processed = (last_day - start_date).days + 1
    return last_day, processed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Dukascopy M1 candles and resample to higher timeframes."
    )
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD.")
    parser.add_argument("--out-dir", required=True, help="Output folder for CSVs.")
    parser.add_argument("--timeframes", default="M1,M15,H4,D1", help="Comma-separated timeframes.")
    parser.add_argument("--side", default="BID", choices=["BID", "ASK"], help="Price side.")
    parser.add_argument("--timeout", type=int, default=10, help="Download timeout seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Download retry count.")
    parser.add_argument("--resume", action="store_true", help="Resume from last success date.")
    parser.add_argument("--force-lock", action="store_true", help="Ignore existing lock file.")
    parser.add_argument("--progress-every", type=int, default=1, help="Log progress every N successful days.")
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    timeframes = _parse_timeframes(args.timeframes)
    start_date = datetime.fromisoformat(args.start).date()
    end_date = datetime.fromisoformat(args.end).date()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tf_rules = _tf_rules(timeframes)
    manifest_path = _manifest_path(out_dir)
    manifest = _load_manifest(manifest_path)
    manifest["source"] = "dukascopy"
    manifest["side"] = args.side
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    for symbol in symbols:
        if not _acquire_lock(out_dir, symbol, args.force_lock):
            continue
        entry = manifest.get(symbol, {})
        symbol_manifest_path = _symbol_manifest_path(out_dir, symbol)
        symbol_manifest = _load_manifest(symbol_manifest_path)
        success_days = int(symbol_manifest.get("success_days") or 0)
        missing_count = int(symbol_manifest.get("missing_days") or 0)
        processed_days = int(symbol_manifest.get("processed_days") or 0)
        missing_sample = list(symbol_manifest.get("missing_sample") or [])
        error_count = int(symbol_manifest.get("error_count") or 0)
        last_error = symbol_manifest.get("last_error")
        last_day_done = None
        last_success_day = None

        if args.resume:
            last_ok = _last_success_date(symbol_manifest) or _last_success_date(entry)
            if last_ok and last_ok >= start_date:
                start = last_ok + timedelta(days=1)
                last_day_done = last_ok
                last_success_day = last_ok
            else:
                resume_last_day, resume_processed = _resume_from_existing(
                    out_dir / f"{symbol}_M1.csv",
                    start_date,
                    end_date,
                )
                if resume_last_day:
                    start = resume_last_day + timedelta(days=1)
                    processed_days = max(processed_days, resume_processed)
                    success_days = max(success_days, resume_processed)
                    last_day_done = resume_last_day
                    last_success_day = resume_last_day
                else:
                    start = start_date
        else:
            start = start_date

        scale = _price_scale(symbol)
        last_success_date = symbol_manifest.get("last_success_date") or entry.get("last_success_date")
        last_error = last_error or None
        total_days = (end_date - start_date).days + 1
        progress_prev = symbol_manifest.get("progress_sample") or {}

        for day in _iter_dates(start, end_date):
            url = _dukascopy_url(symbol, day, args.side)
            try:
                raw = _download(url, args.timeout, args.retries)
                if not raw:
                    raise RuntimeError("empty response")
                df_m1 = _parse_candles(raw, day, scale)
            except Exception as exc:
                last_error = str(exc)
                missing_count += 1
                if len(missing_sample) < 20:
                    missing_sample.append(day.isoformat())
                error_count += 1
                processed_days += 1
                last_day_done = day
                progress_prev = progress_prev or {}
                sample = {
                    "prev": progress_prev or None,
                    "current": {
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "processed_days": processed_days,
                        "success_days": success_days,
                        "missing_days": missing_count,
                        "size_bytes": (out_dir / f"{symbol}_M1.csv").stat().st_size if (out_dir / f"{symbol}_M1.csv").exists() else 0,
                    },
                }
                _update_symbol_manifest(
                    symbol_manifest_path,
                    symbol,
                    start_date,
                    end_date,
                    args.side,
                    timeframes,
                    success_days,
                    missing_count,
                    processed_days,
                    last_day_done,
                    last_success_day,
                    missing_sample,
                    error_count,
                    last_error,
                    sample,
                )
                continue

            df_m1 = df_m1.dropna()
            if df_m1.empty:
                missing_count += 1
                if len(missing_sample) < 20:
                    missing_sample.append(day.isoformat())
                error_count += 1
                processed_days += 1
                last_day_done = day
                progress_prev = progress_prev or {}
                sample = {
                    "prev": progress_prev or None,
                    "current": {
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "processed_days": processed_days,
                        "success_days": success_days,
                        "missing_days": missing_count,
                        "size_bytes": (out_dir / f"{symbol}_M1.csv").stat().st_size if (out_dir / f"{symbol}_M1.csv").exists() else 0,
                    },
                }
                _update_symbol_manifest(
                    symbol_manifest_path,
                    symbol,
                    start_date,
                    end_date,
                    args.side,
                    timeframes,
                    success_days,
                    missing_count,
                    processed_days,
                    last_day_done,
                    last_success_day,
                    missing_sample,
                    error_count,
                    last_error,
                    sample,
                )
                continue

            if "M1" in timeframes:
                _write_rows(out_dir / f"{symbol}_M1.csv", df_m1)
            for tf, rule in tf_rules.items():
                if tf == "M1":
                    continue
                df_tf = _resample(df_m1.copy(), rule)
                _write_rows(out_dir / f"{symbol}_{tf}.csv", df_tf)

            success_days += 1
            processed_days += 1
            last_success_date = day.isoformat()
            last_day_done = day
            last_success_day = day

            progress_prev = progress_prev or {}
            size_bytes = (out_dir / f"{symbol}_M1.csv").stat().st_size if (out_dir / f"{symbol}_M1.csv").exists() else 0
            sample = {
                "prev": progress_prev or None,
                "current": {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "processed_days": processed_days,
                    "success_days": success_days,
                    "missing_days": missing_count,
                    "size_bytes": size_bytes,
                },
            }
            _update_symbol_manifest(
                symbol_manifest_path,
                symbol,
                start_date,
                end_date,
                args.side,
                timeframes,
                success_days,
                missing_count,
                processed_days,
                last_day_done,
                last_success_day,
                missing_sample,
                error_count,
                last_error,
                sample,
            )
            progress_prev = sample["current"]

            if success_days % max(args.progress_every, 1) == 0:
                pct = (processed_days / total_days) * 100 if total_days else 0.0
                bar = _progress_bar(pct)
                rate = None
                prev = sample.get("prev") or {}
                if prev:
                    try:
                        t0 = datetime.fromisoformat(prev["ts_utc"])
                        t1 = datetime.fromisoformat(sample["current"]["ts_utc"])
                        dt_min = max((t1 - t0).total_seconds() / 60.0, 0.001)
                        dd = sample["current"]["processed_days"] - prev.get("processed_days", 0)
                        rate = dd / dt_min
                    except Exception:
                        rate = None
                eta = None
                if rate and rate > 0:
                    remaining = max(total_days - processed_days, 0)
                    eta = remaining / rate
                eta_str = f"{eta:.1f}m" if eta is not None else "n/a"
                rate_str = f"{rate:.2f} days/min" if rate is not None else "n/a"
                print(
                    f"PROGRESS {symbol} {bar} {pct:.2f}% "
                    f"days={processed_days}/{total_days} rate={rate_str} eta={eta_str}"
                )

        manifest[symbol] = {
            "symbol": symbol,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "success_days": success_days,
            "missing_days": missing_count,
            "missing_sample": missing_sample,
            "last_success_date": last_success_date,
            "last_error": last_error,
        }
        _save_manifest(manifest_path, manifest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
