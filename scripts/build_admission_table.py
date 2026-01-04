from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


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


def _avg(vals: Iterable[float]) -> float:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0


def _median(vals: Iterable[float]) -> float:
    vals = list(vals)
    return float(pd.Series(vals).median()) if vals else 0.0


def _quantile(vals: Iterable[float], q: float) -> float:
    vals = list(vals)
    return float(pd.Series(vals).quantile(q)) if vals else 0.0


def _edge_for_symbol(symbol: str) -> str:
    return "NY Trend Pullback" if symbol in {"AUDUSD", "USDCAD"} else "Curry/Klay"


def _parse_trade_time(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _overlap_count(trades_a: List[Dict[str, object]], trades_b: List[Dict[str, object]]) -> int:
    count = 0
    b_windows: List[Tuple[datetime, datetime]] = []
    for t in trades_b:
        start = _parse_trade_time(t.get("entry_time"))
        end = _parse_trade_time(t.get("exit_time"))
        if start and end:
            b_windows.append((start, end))
    for t in trades_a:
        a_start = _parse_trade_time(t.get("entry_time"))
        a_end = _parse_trade_time(t.get("exit_time"))
        if not a_start or not a_end:
            continue
        for b_start, b_end in b_windows:
            if a_start <= b_end and b_start <= a_end:
                count += 1
                break
    return count


def _concurrency_metrics(all_trades: List[Dict[str, object]]) -> Dict[str, object]:
    events: List[Tuple[datetime, int]] = []
    for t in all_trades:
        start = _parse_trade_time(t.get("entry_time"))
        end = _parse_trade_time(t.get("exit_time"))
        if not start or not end:
            continue
        events.append((start, 1))
        events.append((end, -1))
    if not events:
        return {
            "max_concurrent_trades": 0,
            "pct_time_over_2": 0.0,
            "over_2_time_minutes": 0.0,
        }
    events.sort(key=lambda x: x[0])
    open_trades = 0
    max_open = 0
    over_2_seconds = 0.0
    total_seconds = 0.0
    prev_time = events[0][0]
    for ts, delta in events:
        dt = (ts - prev_time).total_seconds()
        if dt > 0:
            total_seconds += dt
            if open_trades > 2:
                over_2_seconds += dt
        open_trades += delta
        if open_trades > max_open:
            max_open = open_trades
        prev_time = ts
    pct_over_2 = (over_2_seconds / total_seconds) * 100.0 if total_seconds else 0.0
    return {
        "max_concurrent_trades": max_open,
        "pct_time_over_2": pct_over_2,
        "over_2_time_minutes": over_2_seconds / 60.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build admission table from Splash backtest report.")
    parser.add_argument("--report", required=True, help="Path to splash_backtest_report.json")
    parser.add_argument("--data-root", required=True, help="Root folder containing per-symbol subfolders + manifests")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--out-csv", default="reports/admission_table.csv", help="Output CSV path")
    parser.add_argument("--out-json", default="reports/admission_table.json", help="Output JSON path")
    args = parser.parse_args()

    report = _read_json(Path(args.report))
    data_root = Path(args.data_root).expanduser().resolve()
    symbols = _parse_symbols(args.symbols)
    initial_equity = float(report.get("initial_equity") or 10000.0)
    portfolio_report = report.get("portfolio", {}) or {}

    rows: List[Dict[str, object]] = []
    portfolio_daily_pnl: Dict[str, float] = defaultdict(float)
    portfolio_weekly_counts: Dict[str, int] = defaultdict(int)
    all_trades: List[Dict[str, object]] = []

    for symbol in symbols:
        payload = report.get("symbols", {}).get(symbol, {})
        trades = payload.get("trades", []) or []
        stats = payload.get("stats", {}) or {}
        daily_counts = stats.get("daily_trade_counts", {}) or {}
        weekly_counts = stats.get("weekly_trade_counts", {}) or {}
        daily_vals = list(daily_counts.values())
        weekly_vals = list(weekly_counts.values())

        r_vals = [t.get("r_mult", 0.0) for t in trades if t.get("r_mult") is not None]
        trades_total = int(stats.get("trades_total", 0) or 0)
        win_rate = float(stats.get("win_rate", 0.0) or 0.0)
        avg_r = float(stats.get("avg_r", 0.0) or 0.0)
        median_r = float(stats.get("median_r", 0.0) or 0.0)
        p25_r = _quantile(r_vals, 0.25)
        p75_r = _quantile(r_vals, 0.75)

        worst_day_pnl = float(stats.get("worst_day_pnl", 0.0) or 0.0)
        worst_day_dd_pct = abs(worst_day_pnl) / initial_equity * 100.0 if initial_equity else 0.0
        max_drawdown_pct = float(payload.get("max_drawdown_pct", 0.0) or 0.0) * 100.0

        pass_sample = trades_total >= 50
        pass_dd = (worst_day_dd_pct <= 5.0) and (max_drawdown_pct <= 10.0)

        if trades_total >= 200 and pass_dd:
            confidence = "HIGH"
        elif trades_total >= 50:
            confidence = "MED"
        else:
            confidence = "LOW"

        # Missing day integrity from manifest
        manifest = _read_json(data_root / symbol / f"manifest_{symbol}.json")
        missing_days = int(manifest.get("missing_days") or 0)
        total_days = int(manifest.get("processed_days") or 0) or 0
        missing_pct = (missing_days / total_days * 100.0) if total_days else 0.0
        missing_material = missing_pct >= 1.0

        row = {
            "edge": _edge_for_symbol(symbol),
            "symbol": symbol,
            "trades_day_avg": _avg(daily_vals),
            "trades_day_median": _median(daily_vals),
            "trades_week_avg": _avg(weekly_vals),
            "trades_week_median": _median(weekly_vals),
            "win_pct": win_rate * 100.0,
            "avg_R": avg_r,
            "median_R": median_r,
            "p25_R": p25_r,
            "p75_R": p75_r,
            "worst_day_DD_pct": worst_day_dd_pct,
            "max_losing_streak_plays": int(stats.get("max_losing_streak", 0) or 0),
            "sample_trades": trades_total,
            "pass_DD": pass_dd,
            "pass_sample": pass_sample,
            "pass_freq": False,
            "confidence": confidence,
        }
        rows.append(row)

        # portfolio aggregations
        for day, pnl in (payload.get("daily_pnl", {}) or {}).items():
            try:
                portfolio_daily_pnl[day] += float(pnl)
            except Exception:
                continue
        for week, count in weekly_counts.items():
            portfolio_weekly_counts[week] += int(count)
        all_trades.extend(trades)

    # Portfolio-level summary
    portfolio_weekly_vals = list(portfolio_weekly_counts.values())
    portfolio_weekly_avg = _avg(portfolio_weekly_vals)
    portfolio_weekly_median = _median(portfolio_weekly_vals)
    portfolio_trades_week_pass = 25.0 <= portfolio_weekly_avg <= 50.0

    worst_day = None
    worst_day_pnl = 0.0
    if portfolio_daily_pnl:
        worst_day = min(portfolio_daily_pnl, key=portfolio_daily_pnl.get)
        worst_day_pnl = portfolio_daily_pnl[worst_day]
    portfolio_worst_day_dd_pct = abs(worst_day_pnl) / initial_equity * 100.0 if initial_equity else 0.0

    # Approx daily equity path for max drawdown
    equity = initial_equity
    peak = initial_equity
    max_dd_pct = 0.0
    for day in sorted(portfolio_daily_pnl.keys()):
        equity += portfolio_daily_pnl[day]
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100.0
            if dd > max_dd_pct:
                max_dd_pct = dd

    portfolio_pass_dd = (portfolio_worst_day_dd_pct <= 5.0) and (max_dd_pct <= 10.0)
    all_r = [t.get("r_mult", 0.0) for t in all_trades if t.get("r_mult") is not None]
    portfolio_avg_r = _avg(all_r)

    cluster = _concurrency_metrics(all_trades)
    jpy_overlaps = _overlap_count(
        [t for t in all_trades if t.get("symbol") == "USDJPY"],
        [t for t in all_trades if t.get("symbol") == "GBPJPY"],
    )
    open_risk_near_cap_pct = float(portfolio_report.get("open_risk_pct_near_cap_pct") or 0.0)
    open_risk_cap_pct = float(portfolio_report.get("max_open_risk_pct") or report.get("max_open_risk_pct") or 0.0)

    portfolio_summary = {
        "total_trades_week_avg": portfolio_weekly_avg,
        "total_trades_week_median": portfolio_weekly_median,
        "worst_day_DD_pct": portfolio_worst_day_dd_pct,
        "max_DD_pct": max_dd_pct,
        "avg_R": portfolio_avg_r,
        "open_risk_pct_near_cap_pct": open_risk_near_cap_pct,
        "open_risk_cap_pct": open_risk_cap_pct,
        "pass_DD": portfolio_pass_dd,
        "notes": (
            f">2 concurrent trades pct_time={cluster['pct_time_over_2']:.2f}% "
            f"(minutes={cluster['over_2_time_minutes']:.1f}), "
            f"max_concurrent={cluster['max_concurrent_trades']}, "
            f"JPY_overlap_count={jpy_overlaps}, "
            f"open_risk_near_cap_pct={open_risk_near_cap_pct:.2f}%"
        ),
    }

    # apply portfolio frequency to each row
    for row in rows:
        row["pass_freq"] = portfolio_trades_week_pass

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(
        json.dumps(
            {
                "rows": rows,
                "portfolio": portfolio_summary,
                "integrity": {
                    "missing_material_symbols": [
                        symbol for symbol in symbols
                        if (
                            (_read_json(data_root / symbol / f"manifest_{symbol}.json").get("missing_days") or 0)
                            / (_read_json(data_root / symbol / f"manifest_{symbol}.json").get("processed_days") or 1)
                        ) * 100.0 >= 0.2
                    ],
                    "missing_days_note": (
                        "Missing days are excluded (no fill). Outcomes already reflect exclusion."
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
