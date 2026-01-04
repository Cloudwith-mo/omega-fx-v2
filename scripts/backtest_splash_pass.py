from __future__ import annotations

import argparse
import heapq
import json
import time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from omegafx_v2.config import (
    DEFAULT_COSTS,
    DEFAULT_SESSION,
    DEFAULT_STRATEGY,
    DEFAULT_LONDON_BREAKOUT_SIGNAL_CONFIG,
    DEFAULT_LONDON_BREAKOUT_SIGNAL_CONFIG_GBPJPY,
    DEFAULT_LIQUIDITY_SWEEP_SIGNAL_CONFIG,
    DEFAULT_LIQUIDITY_SWEEP_SIGNAL_CONFIG_GBPJPY,
    DEFAULT_TREND_SIGNAL_CONFIG,
    NY_SESSION,
)
from omegafx_v2.data import load_ohlc_csv
from omegafx_v2.signals import build_london_breakout_signals, build_liquidity_sweep_signals, build_trend_signals
from omegafx_v2.sim import simulate_trade_path


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


def _find_csv(data_dir: Path, symbol: str) -> Optional[Path]:
    symbol_dir = data_dir / symbol
    search_roots = [data_dir]
    if symbol_dir.exists():
        search_roots.insert(0, symbol_dir)
    for root in search_roots:
        candidates = [
            root / f"{symbol}.csv",
            root / f"{symbol.lower()}.csv",
            root / f"{symbol.upper()}.csv",
            root / f"{symbol}_M15.csv",
            root / f"{symbol}_m15.csv",
            root / f"{symbol}_M1.csv",
            root / f"{symbol}_m1.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
    return None


def _signal_configs_for_symbol(symbol: str):
    if symbol.upper() == "GBPJPY":
        return DEFAULT_LONDON_BREAKOUT_SIGNAL_CONFIG_GBPJPY, DEFAULT_LIQUIDITY_SWEEP_SIGNAL_CONFIG_GBPJPY
    return DEFAULT_LONDON_BREAKOUT_SIGNAL_CONFIG, DEFAULT_LIQUIDITY_SWEEP_SIGNAL_CONFIG


def _symbol_buckets(symbol: str) -> List[str]:
    sym = (symbol or "").upper().strip()
    if len(sym) >= 6:
        return [sym[:3], sym[3:6]]
    if sym:
        return [sym]
    return ["UNKNOWN"]


def _parse_bucket_caps(raw: Optional[str]) -> Dict[str, int]:
    caps: Dict[str, int] = {}
    if not raw:
        return caps
    for chunk in raw.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, val = chunk.split("=", 1)
        key = key.strip().upper()
        val = val.strip()
        if not key or not val:
            continue
        try:
            cap = int(val)
        except ValueError:
            continue
        if cap >= 0:
            caps[key] = cap
    return caps


def _build_signals(ohlc: pd.DataFrame, symbol: str) -> pd.Series:
    if symbol.upper() in {"AUDUSD", "USDCAD"}:
        return build_trend_signals(
            ohlc,
            signal_config=DEFAULT_TREND_SIGNAL_CONFIG,
            session=NY_SESSION,
        )
    sig_london_cfg, sig_sweep_cfg = _signal_configs_for_symbol(symbol)
    london = build_london_breakout_signals(
        ohlc,
        signal_config=sig_london_cfg,
        session=DEFAULT_SESSION,
        symbol=symbol,
    )
    sweep = build_liquidity_sweep_signals(
        ohlc,
        signal_config=sig_sweep_cfg,
        session=DEFAULT_SESSION,
        symbol=symbol,
    )
    return london | sweep


def _prepare_portfolio_inputs(
    symbols: List[str],
    data_dir: Path,
    timeframe: str,
    initial_equity: float,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, str]]:
    symbol_data: Dict[str, Dict[str, object]] = {}
    errors: Dict[str, str] = {}
    for symbol in symbols:
        csv_path = _find_csv(data_dir, symbol)
        if not csv_path:
            errors[symbol] = "missing_csv"
            continue
        try:
            ohlc = load_ohlc_csv(csv_path, timeframe=timeframe)
            signals = _build_signals(ohlc, symbol).reindex(ohlc.index).fillna(False)
            strategy_cfg = replace(DEFAULT_STRATEGY, symbol=symbol)
            signal_indices = np.flatnonzero(signals.values)
            risk_pct = float(strategy_cfg.risk_per_trade_pct or 0.0)
            symbol_data[symbol] = {
                "ohlc": ohlc,
                "signals": signals,
                "config": strategy_cfg,
                "signal_indices": signal_indices,
                "risk_pct": risk_pct,
            }
        except Exception as exc:
            errors[symbol] = str(exc)
    symbol_data["__cache__"] = {}
    return symbol_data, errors


def _symbol_max_drawdown_pct(trades: List[Dict[str, object]], initial_equity: float) -> float:
    equity = initial_equity
    peak = initial_equity
    max_dd_pct = 0.0
    def _parse_time(raw: Optional[str]) -> Optional[datetime]:
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return None
    ordered = sorted(
        trades,
        key=lambda t: _parse_time(t.get("exit_time")) or datetime.min,
    )
    for trade in ordered:
        equity += float(trade.get("pnl", 0.0) or 0.0)
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak
            if dd > max_dd_pct:
                max_dd_pct = dd
    return max_dd_pct


def _open_risk_metrics(
    trades: List[Dict[str, object]],
    open_risk_cap_pct: float,
    near_cap_threshold: float = 0.9,
) -> Dict[str, object]:
    events: List[Tuple[datetime, float]] = []
    for trade in trades:
        entry = trade.get("entry_time")
        exit_time = trade.get("exit_time")
        try:
            start = datetime.fromisoformat(str(entry).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(exit_time).replace("Z", "+00:00"))
        except Exception:
            continue
        risk_pct = float(trade.get("risk_pct", 0.0) or 0.0)
        events.append((start, risk_pct))
        events.append((end, -risk_pct))
    if not events:
        return {
            "open_risk_pct_max": 0.0,
            "open_risk_pct_near_cap_pct": 0.0,
            "open_risk_total_seconds": 0.0,
        }
    events.sort(key=lambda x: x[0])
    open_risk_pct = 0.0
    open_risk_pct_max = 0.0
    near_cap_seconds = 0.0
    total_seconds = 0.0
    prev_time = events[0][0]
    threshold = open_risk_cap_pct * near_cap_threshold if open_risk_cap_pct > 0 else 0.0
    for ts, delta in events:
        dt = (ts - prev_time).total_seconds()
        if dt > 0:
            total_seconds += dt
            if open_risk_cap_pct > 0 and open_risk_pct >= threshold:
                near_cap_seconds += dt
        open_risk_pct += delta
        if open_risk_pct > open_risk_pct_max:
            open_risk_pct_max = open_risk_pct
            threshold = open_risk_cap_pct * near_cap_threshold if open_risk_cap_pct > 0 else 0.0
        prev_time = ts
    pct_near = (near_cap_seconds / total_seconds * 100.0) if total_seconds else 0.0
    return {
        "open_risk_pct_max": open_risk_pct_max,
        "open_risk_pct_near_cap_pct": pct_near,
        "open_risk_total_seconds": total_seconds,
    }


def _parse_trade_time(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _position_time_metrics(trades: List[Dict[str, object]]) -> Dict[str, object]:
    events: List[Tuple[datetime, int]] = []
    durations: List[float] = []
    for trade in trades:
        start = _parse_trade_time(trade.get("entry_time"))
        end = _parse_trade_time(trade.get("exit_time"))
        if not start or not end:
            continue
        events.append((start, 1))
        events.append((end, -1))
        durations.append((end - start).total_seconds())
    if not events:
        return {
            "minutes_with_any_position_open": 0.0,
            "pct_time_with_any_position_open": 0.0,
            "avg_position_hold_minutes": 0.0,
        }
    events.sort(key=lambda item: item[0])
    open_count = 0
    open_seconds = 0.0
    total_seconds = 0.0
    prev_time = events[0][0]
    for ts, delta in events:
        dt = (ts - prev_time).total_seconds()
        if dt > 0:
            total_seconds += dt
            if open_count > 0:
                open_seconds += dt
        open_count += delta
        prev_time = ts
    avg_hold_minutes = (sum(durations) / len(durations) / 60.0) if durations else 0.0
    pct_open = (open_seconds / total_seconds * 100.0) if total_seconds else 0.0
    return {
        "minutes_with_any_position_open": open_seconds / 60.0,
        "pct_time_with_any_position_open": pct_open,
        "avg_position_hold_minutes": avg_hold_minutes,
    }


def _log_progress(message: str, log_path: Path) -> None:
    print(message)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception:
        pass


def _trade_stats(trades: List[Dict[str, object]], daily_pnl: Dict[object, float]) -> Dict[str, object]:
    total = len(trades)
    r_values = [t["r_mult"] for t in trades]
    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r < 0]
    win_rate = (len(wins) / total) if total else 0.0
    avg_r = sum(r_values) / total if total else 0.0
    median_r = float(pd.Series(r_values).median()) if total else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    expectancy = avg_r

    max_losing_streak = 0
    current_streak = 0
    for r in r_values:
        if r < 0:
            current_streak += 1
            max_losing_streak = max(max_losing_streak, current_streak)
        else:
            current_streak = 0

    worst_trades = sorted(r_values)[:3] if r_values else []
    worst_trade = worst_trades[0] if worst_trades else 0.0

    daily_trade_counts: Dict[str, int] = {}
    for t in trades:
        day = t["entry_time"].split("T", 1)[0]
        daily_trade_counts[day] = daily_trade_counts.get(day, 0) + 1
    daily_counts = list(daily_trade_counts.values())
    trades_per_day_avg = sum(daily_counts) / len(daily_counts) if daily_counts else 0.0
    trades_per_day_max = max(daily_counts) if daily_counts else 0

    weekly_counts: Dict[str, int] = {}
    for day_str, count in daily_trade_counts.items():
        day = datetime.fromisoformat(day_str).date()
        year, week, _ = day.isocalendar()
        key = f"{year}-W{week:02d}"
        weekly_counts[key] = weekly_counts.get(key, 0) + count
    weekly_vals = list(weekly_counts.values())
    trades_per_week_avg = sum(weekly_vals) / len(weekly_vals) if weekly_vals else 0.0

    worst_day = None
    worst_day_pnl = 0.0
    if daily_pnl:
        worst_day = min(daily_pnl, key=daily_pnl.get)
        worst_day_pnl = daily_pnl[worst_day]

    hold_minutes_vals: List[float] = []
    exit_reason_counts = {"tp": 0, "sl": 0, "time": 0, "end": 0}
    for t in trades:
        entry = _parse_trade_time(t.get("entry_time"))
        exit_time = _parse_trade_time(t.get("exit_time"))
        if entry and exit_time:
            hold_minutes_vals.append((exit_time - entry).total_seconds() / 60.0)
        reason = str(t.get("exit_reason") or "").lower()
        if reason in exit_reason_counts:
            exit_reason_counts[reason] += 1
    hold_series = pd.Series(hold_minutes_vals) if hold_minutes_vals else pd.Series([], dtype="float")
    hold_median = float(hold_series.median()) if not hold_series.empty else 0.0
    hold_p95 = float(hold_series.quantile(0.95)) if not hold_series.empty else 0.0
    hold_max = float(hold_series.max()) if not hold_series.empty else 0.0
    tp_pct = (exit_reason_counts["tp"] / total * 100.0) if total else 0.0
    sl_pct = (exit_reason_counts["sl"] / total * 100.0) if total else 0.0
    time_pct = (exit_reason_counts["time"] / total * 100.0) if total else 0.0
    end_pct = (exit_reason_counts["end"] / total * 100.0) if total else 0.0

    return {
        "trades_total": total,
        "win_rate": win_rate,
        "avg_r": avg_r,
        "median_r": median_r,
        "avg_win_r": avg_win,
        "avg_loss_r": avg_loss,
        "expectancy_r": expectancy,
        "max_losing_streak": max_losing_streak,
        "worst_trade_r": worst_trade,
        "worst_3_trades_r": worst_trades,
        "trades_per_day_avg": trades_per_day_avg,
        "trades_per_day_max": trades_per_day_max,
        "trades_per_week_avg": trades_per_week_avg,
        "daily_trade_counts": daily_trade_counts,
        "weekly_trade_counts": weekly_counts,
        "worst_day": str(worst_day) if worst_day else None,
        "worst_day_pnl": worst_day_pnl,
        "hold_minutes_median": hold_median,
        "hold_minutes_p95": hold_p95,
        "hold_minutes_max": hold_max,
        "forced_close_count": exit_reason_counts["time"],
        "exit_reason_counts": exit_reason_counts,
        "exit_tp_pct": tp_pct,
        "exit_sl_pct": sl_pct,
        "exit_time_pct": time_pct,
        "exit_end_pct": end_pct,
    }


def _admission_check(stats: Dict[str, object], min_trades: int, min_avg_r: float) -> Dict[str, object]:
    reasons: List[str] = []
    trades_total = stats.get("trades_total", 0) or 0
    avg_r = stats.get("avg_r", 0.0) or 0.0
    if trades_total < min_trades:
        reasons.append(f"trades<{min_trades}")
    if avg_r < min_avg_r:
        reasons.append(f"avg_r<{min_avg_r}")
    return {
        "pass": len(reasons) == 0,
        "reasons": reasons,
    }


def run_portfolio_backtest(
    symbols: List[str],
    data_dir: Path,
    timeframe: str,
    initial_equity: float,
    daily_loss_pct: float,
    min_trades: int,
    min_avg_r: float,
    max_open_risk_pct: float,
    bucket_caps: Dict[str, int],
    prepared: Optional[Tuple[Dict[str, Dict[str, object]], Dict[str, str]]] = None,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, str]]:
    symbol_data: Dict[str, Dict[str, object]]
    errors: Dict[str, str]
    if prepared is None:
        symbol_data, errors = _prepare_portfolio_inputs(symbols, data_dir, timeframe, initial_equity)
    else:
        symbol_data, errors = prepared
    skip_bucket_caps: Dict[str, int] = defaultdict(int)
    skip_open_risk_cap = 0
    skip_daily_loss_cap = 0
    skip_symbol_open = 0
    signals_total = 0
    trades_taken = 0
    signals_processed = 0
    progress_every = 50000
    progress_log = Path("reports") / "backtest_progress.log"
    start_time = time.monotonic()

    event_heap: List[Tuple[pd.Timestamp, str, int]] = []
    for symbol in symbols:
        data = symbol_data.get(symbol)
        if not data:
            continue
        indices = data.get("signal_indices")
        if indices is None:
            continue
        signals_total += int(len(indices))
        if len(indices) > 0:
            first_idx = int(indices[0])
            entry_time = data["ohlc"].index[first_idx]
            heapq.heappush(event_heap, (entry_time, symbol, 0))

    equity = initial_equity
    max_equity = equity
    min_equity = equity
    day_limit = initial_equity * daily_loss_pct if daily_loss_pct > 0 else 0.0

    trades_by_symbol: Dict[str, List[Dict[str, object]]] = {symbol: [] for symbol in symbols}
    daily_pnl_by_symbol: Dict[str, Dict[object, float]] = {symbol: defaultdict(float) for symbol in symbols}
    portfolio_daily_pnl: Dict[object, float] = defaultdict(float)
    open_positions: List[Dict[str, object]] = []

    _log_progress(
        f"Backtest start | symbols={','.join(symbols)} | signals_total={signals_total}",
        progress_log,
    )
    while event_heap:
        entry_time, symbol, pos_idx = heapq.heappop(event_heap)
        if symbol not in symbol_data:
            continue
        data = symbol_data[symbol]
        indices = data.get("signal_indices")
        if indices is None or pos_idx >= len(indices):
            continue
        entry_idx = int(indices[pos_idx])
        # Push next event for this symbol.
        next_pos = pos_idx + 1
        if next_pos < len(indices):
            next_idx = int(indices[next_pos])
            next_time = data["ohlc"].index[next_idx]
            heapq.heappush(event_heap, (next_time, symbol, next_pos))

        signals_processed += 1
        if signals_processed % progress_every == 0:
            elapsed_sec = time.monotonic() - start_time
            elapsed_min = elapsed_sec / 60.0 if elapsed_sec else 0.0
            rate = signals_processed / elapsed_sec if elapsed_sec > 0 else 0.0
            remaining = signals_total - signals_processed
            est_remaining_min = (remaining / rate / 60.0) if rate > 0 else 0.0
            _log_progress(
                (
                    f"Progress | signals_processed={signals_processed} "
                    f"| trades_taken={trades_taken} "
                    f"| elapsed_min={elapsed_min:.1f} "
                    f"| est_remaining_min={est_remaining_min:.1f} "
                    f"| symbol={symbol} "
                    f"| date={entry_time.date().isoformat()}"
                ),
                progress_log,
            )

        # Close positions that have exited by this time.
        if open_positions:
            still_open: List[Dict[str, object]] = []
            for pos in open_positions:
                if pos["exit_time"] <= entry_time:
                    equity += pos["pnl"]
                    day_key = pos["exit_time"].date()
                    daily_pnl_by_symbol[pos["symbol"]][day_key] += pos["pnl"]
                    portfolio_daily_pnl[day_key] += pos["pnl"]
                    if equity > max_equity:
                        max_equity = equity
                    if equity < min_equity:
                        min_equity = equity
                else:
                    still_open.append(pos)
            open_positions = still_open

        # Daily loss cap (portfolio-level).
        if day_limit > 0:
            day_key = entry_time.date()
            if portfolio_daily_pnl.get(day_key, 0.0) <= -day_limit:
                skip_daily_loss_cap += 1
                continue

        # One open trade per symbol at a time.
        if any(pos["symbol"] == symbol for pos in open_positions):
            skip_symbol_open += 1
            continue

        risk_pct = float(data.get("risk_pct") or 0.0)
        open_risk_pct = sum(float(pos["risk_pct"]) for pos in open_positions)
        if max_open_risk_pct > 0 and (open_risk_pct + risk_pct) > max_open_risk_pct:
            skip_open_risk_cap += 1
            continue

        buckets = _symbol_buckets(symbol)
        blocked = False
        for bucket in buckets:
            cap = bucket_caps.get(bucket)
            if cap is None:
                continue
            open_bucket = sum(1 for pos in open_positions if bucket in pos["buckets"])
            if open_bucket + 1 > cap:
                blocked = True
                skip_bucket_caps[bucket] += 1
                break
        if blocked:
            continue

        exit_time = None
        r_mult = None
        exit_reason = None
        if exit_time is None or r_mult is None:
            cache = symbol_data.get("__cache__", {})
            cache_key = (symbol, entry_idx)
            cached = cache.get(cache_key)
            if cached:
                exit_time, r_mult, exit_reason = cached
            else:
                cfg = data["config"]
                ohlc = data["ohlc"]
                try:
                    outcome = simulate_trade_path(
                        ohlc=ohlc,
                        entry_idx=entry_idx,
                        account_balance=equity,
                        config=cfg,
                        costs=DEFAULT_COSTS,
                    )
                except Exception:
                    continue
                exit_time = outcome.exit_time
                risk_amount = equity * risk_pct
                r_mult = outcome.pnl / risk_amount if risk_amount else 0.0
                exit_reason = outcome.exit_reason
                cache[cache_key] = (exit_time, r_mult, exit_reason)
        r_mult = float(r_mult or 0.0)
        pnl = equity * risk_pct * r_mult
        trade = {
            "symbol": symbol,
            "entry_time": entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
            "pnl": pnl,
            "pnl_pct": risk_pct * r_mult,
            "r_mult": r_mult,
            "exit_reason": exit_reason,
            "risk_pct": risk_pct,
        }
        trades_by_symbol[symbol].append(trade)
        open_positions.append(
            {
                "symbol": symbol,
                "exit_time": exit_time,
                "pnl": pnl,
                "buckets": buckets,
                "risk_pct": risk_pct,
            }
        )
        trades_taken += 1

    # Close remaining open positions in exit order.
    if open_positions:
        open_positions.sort(key=lambda pos: pos["exit_time"])
        for pos in open_positions:
            equity += pos["pnl"]
            day_key = pos["exit_time"].date()
            daily_pnl_by_symbol[pos["symbol"]][day_key] += pos["pnl"]
            portfolio_daily_pnl[day_key] += pos["pnl"]
            if equity > max_equity:
                max_equity = equity
            if equity < min_equity:
                min_equity = equity
        open_positions = []

    results: Dict[str, object] = {}
    all_trades: List[Dict[str, object]] = []
    for symbol in symbols:
        if symbol not in symbol_data:
            continue
        trades = trades_by_symbol.get(symbol, [])
        daily_pnl = daily_pnl_by_symbol.get(symbol, {})
        stats = _trade_stats(trades, daily_pnl)
        admission = _admission_check(stats, min_trades, min_avg_r)
        ohlc = symbol_data[symbol]["ohlc"]
        max_drawdown_pct = _symbol_max_drawdown_pct(trades, initial_equity)
        daily_pnl_str = {str(day): pnl for day, pnl in daily_pnl.items()}
        results[symbol] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data_start": ohlc.index.min().isoformat() if not ohlc.empty else None,
            "data_end": ohlc.index.max().isoformat() if not ohlc.empty else None,
            "max_drawdown_pct": max_drawdown_pct,
            "daily_pnl": daily_pnl_str,
            "trades": trades,
            "stats": stats,
            "admission": admission,
        }
        all_trades.extend(trades)

    portfolio_max_dd_pct = 0.0
    if max_equity > 0:
        portfolio_max_dd_pct = (max_equity - min_equity) / max_equity

    open_risk = _open_risk_metrics(all_trades, max_open_risk_pct)
    position_time = _position_time_metrics(all_trades)
    forced_close_count = sum(
        1 for trade in all_trades if str(trade.get("exit_reason") or "").lower() == "time"
    )
    total_trades = len(all_trades)
    forced_close_pct = (forced_close_count / total_trades * 100.0) if total_trades else 0.0
    skipped_total = skip_open_risk_cap + skip_daily_loss_cap + skip_symbol_open + sum(skip_bucket_caps.values())
    skip_other = signals_total - trades_taken - skipped_total
    if skip_other < 0:
        skip_other = 0
    portfolio = {
        "equity_end": equity,
        "max_drawdown_pct": portfolio_max_dd_pct,
        "daily_pnl": {str(day): pnl for day, pnl in portfolio_daily_pnl.items()},
        "max_open_risk_pct": max_open_risk_pct,
        "bucket_caps": bucket_caps,
        "open_risk_pct_max": open_risk["open_risk_pct_max"],
        "open_risk_pct_near_cap_pct": open_risk["open_risk_pct_near_cap_pct"],
        "open_risk_total_seconds": open_risk["open_risk_total_seconds"],
        "open_risk_near_cap_threshold_pct": 90.0,
        "minutes_with_any_position_open": position_time["minutes_with_any_position_open"],
        "pct_time_with_any_position_open": position_time["pct_time_with_any_position_open"],
        "avg_position_hold_minutes": position_time["avg_position_hold_minutes"],
        "forced_close_count": forced_close_count,
        "forced_close_pct": forced_close_pct,
        "debug": {
            "signals_total": signals_total,
            "trades_taken": trades_taken,
            "count_skipped_due_to_open_risk_cap": skip_open_risk_cap,
            "count_skipped_due_to_bucket_cap": dict(skip_bucket_caps),
            "count_skipped_due_to_daily_loss_cap": skip_daily_loss_cap,
            "count_skipped_due_to_symbol_open": skip_symbol_open,
            "count_skipped_due_to_other_gates": skip_other,
        },
    }
    elapsed_min = (time.monotonic() - start_time) / 60.0
    _log_progress(
        (
            f"Backtest done | signals_processed={signals_processed} "
            f"| trades_taken={trades_taken} "
            f"| elapsed_min={elapsed_min:.1f}"
        ),
        progress_log,
    )

    return results, portfolio, errors


def _portfolio_weekly_counts(results: Dict[str, object]) -> Dict[str, int]:
    combined: Dict[str, int] = defaultdict(int)
    for payload in results.values():
        stats = payload.get("stats", {}) or {}
        for week, count in (stats.get("weekly_trade_counts", {}) or {}).items():
            combined[week] += int(count)
    return combined


def _portfolio_avg_r(results: Dict[str, object]) -> float:
    r_vals: List[float] = []
    for payload in results.values():
        for trade in payload.get("trades", []) or []:
            r_val = trade.get("r_mult")
            if r_val is not None:
                r_vals.append(float(r_val))
    return sum(r_vals) / len(r_vals) if r_vals else 0.0


def _portfolio_trade_count(results: Dict[str, object]) -> int:
    total = 0
    for payload in results.values():
        stats = payload.get("stats", {}) or {}
        total += int(stats.get("trades_total", 0) or 0)
    return total


def _portfolio_worst_day_dd_pct(portfolio: Dict[str, object], initial_equity: float) -> float:
    daily_pnl = portfolio.get("daily_pnl", {}) or {}
    if not daily_pnl or not initial_equity:
        return 0.0
    worst_day = min(daily_pnl, key=daily_pnl.get)
    worst_pnl = float(daily_pnl[worst_day] or 0.0)
    return abs(worst_pnl) / initial_equity * 100.0


def run_grid_search(
    symbols: List[str],
    data_dir: Path,
    timeframe: str,
    initial_equity: float,
    daily_loss_pct: float,
    min_trades: int,
    min_avg_r: float,
) -> List[Dict[str, object]]:
    open_risk_grid = [0.02, 0.03, 0.04, 0.05]
    bucket_grid = [1, 2]
    rows: List[Dict[str, object]] = []
    prepared = _prepare_portfolio_inputs(symbols, data_dir, timeframe, initial_equity)
    for open_risk_pct in open_risk_grid:
        for usd_cap in bucket_grid:
            for jpy_cap in bucket_grid:
                for gbp_cap in bucket_grid:
                    bucket_caps = {
                        "USD": usd_cap,
                        "JPY": jpy_cap,
                        "GBP": gbp_cap,
                        "EUR": 1,
                        "AUD": 1,
                        "CAD": 1,
                    }
                    results, portfolio, errors = run_portfolio_backtest(
                        symbols=symbols,
                        data_dir=data_dir,
                        timeframe=timeframe,
                        initial_equity=initial_equity,
                        daily_loss_pct=daily_loss_pct,
                        min_trades=min_trades,
                        min_avg_r=min_avg_r,
                        max_open_risk_pct=open_risk_pct,
                        bucket_caps=bucket_caps,
                        prepared=prepared,
                    )
                    weekly_counts = _portfolio_weekly_counts(results)
                    weekly_vals = list(weekly_counts.values())
                    trades_week_avg = sum(weekly_vals) / len(weekly_vals) if weekly_vals else 0.0
                    worst_day_dd_pct = _portfolio_worst_day_dd_pct(portfolio, initial_equity)
                    max_dd_pct = float(portfolio.get("max_drawdown_pct", 0.0) or 0.0) * 100.0
                    sample_trades = _portfolio_trade_count(results)
                    avg_r = _portfolio_avg_r(results) if sample_trades >= 200 else None
                    rows.append(
                        {
                            "max_open_risk_pct": open_risk_pct,
                            "bucket_USD": usd_cap,
                            "bucket_JPY": jpy_cap,
                            "bucket_GBP": gbp_cap,
                            "trades_week_avg": trades_week_avg,
                            "worst_day_DD_pct": worst_day_dd_pct,
                            "max_DD_pct": max_dd_pct,
                            "avg_R": avg_r,
                            "sample_trades": sample_trades,
                            "open_risk_near_cap_pct": float(
                                portfolio.get("open_risk_pct_near_cap_pct", 0.0) or 0.0
                            ),
                            "errors": errors,
                        }
                    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest Splash PASS symbols with London+Liquidity signals.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. USDJPY,GBPJPY,EURUSD")
    parser.add_argument("--data-dir", required=True, help="Folder containing CSVs per symbol")
    parser.add_argument("--timeframe", default="M15", help="Timeframe for signals, e.g. M15")
    parser.add_argument("--initial-equity", type=float, default=10000.0, help="Starting equity")
    parser.add_argument("--daily-loss-pct", type=float, default=0.03, help="Daily loss cap pct")
    parser.add_argument("--min-trades", type=int, default=50, help="Minimum trades for admission")
    parser.add_argument("--min-avg-r", type=float, default=2.0, help="Minimum avg R for admission")
    parser.add_argument(
        "--max-open-risk-pct",
        type=float,
        default=0.02,
        help="Max total open risk as pct of equity (e.g., 0.02 for 2%)",
    )
    parser.add_argument(
        "--bucket-risk-caps",
        default="USD=1,JPY=1,GBP=1,EUR=1,AUD=1,CAD=1",
        help="Comma-separated bucket caps in R units, e.g. USD=1,JPY=1",
    )
    parser.add_argument("--out", default="reports/splash_backtest_report.json", help="Output JSON path")
    parser.add_argument("--grid-search", action="store_true", help="Run risk budget grid search.")
    parser.add_argument("--grid-out-csv", default="reports/risk_budget_grid.csv", help="Grid CSV output")
    parser.add_argument("--grid-out-json", default="reports/risk_budget_grid.json", help="Grid JSON output")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    symbols = _parse_symbols(args.symbols)
    bucket_caps = _parse_bucket_caps(args.bucket_risk_caps)
    results, portfolio, errors = run_portfolio_backtest(
        symbols=symbols,
        data_dir=data_dir,
        timeframe=args.timeframe,
        initial_equity=args.initial_equity,
        daily_loss_pct=args.daily_loss_pct,
        min_trades=args.min_trades,
        min_avg_r=args.min_avg_r,
        max_open_risk_pct=args.max_open_risk_pct,
        bucket_caps=bucket_caps,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "timeframe": args.timeframe,
        "daily_loss_pct": args.daily_loss_pct,
        "min_trades": args.min_trades,
        "min_avg_r": args.min_avg_r,
        "initial_equity": args.initial_equity,
        "max_open_risk_pct": args.max_open_risk_pct,
        "bucket_risk_caps": bucket_caps,
        "symbols": results,
        "errors": errors,
        "portfolio": portfolio,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    if errors:
        print(f"Errors: {errors}")

    if args.grid_search:
        rows = run_grid_search(
            symbols=symbols,
            data_dir=data_dir,
            timeframe=args.timeframe,
            initial_equity=args.initial_equity,
            daily_loss_pct=args.daily_loss_pct,
            min_trades=args.min_trades,
            min_avg_r=args.min_avg_r,
        )
        import csv
        grid_csv = Path(args.grid_out_csv)
        grid_json = Path(args.grid_out_json)
        grid_csv.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            with grid_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            grid_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
            print(f"Wrote {grid_csv}")
            print(f"Wrote {grid_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
