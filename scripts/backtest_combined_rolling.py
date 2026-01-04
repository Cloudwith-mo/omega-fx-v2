from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from omegafx_v2.config import (
    LiquiditySweepSignalConfig,
    LondonBreakoutSignalConfig,
    MomentumPinballSignalConfig,
    MultiSymbolPortfolioProfile,
    PortfolioProfile,
    SignalConfig,
    StrategyProfile,
    TrendContinuationSignalConfig,
    DEFAULT_COSTS,
    MULTI_PORTFOLIO_COMBINED,
)
from omegafx_v2.data import load_ohlc_csv
from omegafx_v2.signals import (
    build_london_breakout_signals,
    build_liquidity_sweep_signals,
    build_momentum_pinball_signals_m5,
    build_signals,
    build_trend_signals,
)
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


def _find_csv(data_dir: Path, symbol: str, timeframe: str) -> Optional[Path]:
    symbol_dir = data_dir / symbol
    search_roots = [data_dir]
    if symbol_dir.exists():
        search_roots.insert(0, symbol_dir)
    tf = timeframe.upper()
    candidates = [
        f"{symbol}_{tf}.csv",
        f"{symbol}_{tf.lower()}.csv",
        f"{symbol}.csv",
        f"{symbol.lower()}.csv",
        f"{symbol.upper()}.csv",
    ]
    if tf != "M1":
        candidates.extend([f"{symbol}_M1.csv", f"{symbol}_m1.csv"])
    for root in search_roots:
        for name in candidates:
            path = root / name
            if path.exists():
                return path
    return None


def _build_signals_for_profile(ohlc: pd.DataFrame, profile: StrategyProfile) -> pd.Series:
    sig_cfg = profile.signals
    if isinstance(sig_cfg, SignalConfig):
        return build_signals(ohlc, signal_config=sig_cfg, session=profile.session)
    if isinstance(sig_cfg, TrendContinuationSignalConfig):
        return build_trend_signals(ohlc, signal_config=sig_cfg, session=profile.session)
    if isinstance(sig_cfg, LondonBreakoutSignalConfig):
        return build_london_breakout_signals(
            ohlc, signal_config=sig_cfg, session=profile.session, symbol=profile.symbol_key
        )
    if isinstance(sig_cfg, LiquiditySweepSignalConfig):
        return build_liquidity_sweep_signals(
            ohlc, signal_config=sig_cfg, session=profile.session, symbol=profile.symbol_key
        )
    raise TypeError(f"Unsupported signals config type: {type(sig_cfg)!r}")


def _iter_profiles(portfolio) -> List[Tuple[StrategyProfile, float]]:
    profiles: List[Tuple[StrategyProfile, float]] = []
    if isinstance(portfolio, PortfolioProfile):
        scales = getattr(portfolio, "risk_scales", None) or [1.0] * len(portfolio.strategies)
        for prof, scale in zip(portfolio.strategies, scales):
            profiles.append((prof, scale))
        return profiles
    if isinstance(portfolio, MultiSymbolPortfolioProfile):
        sym_scales = getattr(portfolio, "symbol_risk_scales", None) or [1.0] * len(portfolio.portfolios)
        for sub, sym_scale in zip(portfolio.portfolios, sym_scales):
            sub_scales = getattr(sub, "risk_scales", None) or [1.0] * len(sub.strategies)
            for prof, scale in zip(sub.strategies, sub_scales):
                profiles.append((prof, scale * sym_scale))
        return profiles
    raise TypeError(f"Unsupported portfolio type: {type(portfolio)}")


def _profile_group(profile: StrategyProfile) -> str:
    if profile.name == "USDJPY_M5_MomentumPinball_V1":
        return "westbrook"
    return "scorers"


def _symbol_buckets(symbol: str) -> List[str]:
    sym = (symbol or "").upper().strip()
    if len(sym) >= 6:
        return [sym[:3], sym[3:6]]
    if sym:
        return [sym]
    return ["UNKNOWN"]


def _date_range(start: date, end: date) -> List[date]:
    days = (end - start).days
    return [start + timedelta(days=offset) for offset in range(days + 1)]


def _risk_per_trade_pct(
    base_risk_pct: float,
    group: str,
    scorers_override_pct: float,
    scorers_cap_pct: float,
    scorers_max_open: int,
    westbrook_cap_pct: float,
    westbrook_max_open: int,
) -> float:
    risk_pct = base_risk_pct
    cap_per_trade = 0.0
    if group == "scorers":
        if scorers_override_pct > 0:
            risk_pct = min(risk_pct, scorers_override_pct) if risk_pct > 0 else scorers_override_pct
        if scorers_cap_pct > 0 and scorers_max_open > 0:
            cap_per_trade = scorers_cap_pct / scorers_max_open
    else:
        if westbrook_cap_pct > 0 and westbrook_max_open > 0:
            cap_per_trade = westbrook_cap_pct / westbrook_max_open
    if cap_per_trade > 0 and risk_pct > 0:
        risk_pct = min(risk_pct, cap_per_trade)
    return max(0.0, risk_pct)


def _days_to_threshold(
    daily_pnl: Dict[date, float],
    start_date: date,
    end_date: date,
    initial_equity: float,
    target_pct: float,
) -> Optional[int]:
    equity = initial_equity
    threshold = initial_equity * (1.0 + target_pct)
    for idx, day in enumerate(_date_range(start_date, end_date), start=1):
        equity += daily_pnl.get(day, 0.0)
        if target_pct >= 0 and equity >= threshold:
            return idx
        if target_pct < 0 and equity <= threshold:
            return idx
    return None


def _max_drawdown_pct(trades: List[Dict[str, object]], initial_equity: float) -> float:
    equity = initial_equity
    peak = initial_equity
    max_dd = 0.0
    ordered = sorted(trades, key=lambda t: t.get("exit_time") or "")
    for trade in ordered:
        equity += float(trade.get("pnl", 0.0) or 0.0)
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd * 100.0


def _window_stats(
    trades: List[Dict[str, object]],
    daily_pnl: Dict[date, float],
    start_date: date,
    end_date: date,
    initial_equity: float,
) -> Dict[str, object]:
    total = len(trades)
    r_vals = [float(t.get("r_mult", 0.0) or 0.0) for t in trades]
    wins = [r for r in r_vals if r > 0]
    win_rate = (len(wins) / total) if total else 0.0
    avg_r = sum(r_vals) / total if total else 0.0
    max_streak = 0
    streak = 0
    for r in r_vals:
        if r < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    window_days = (end_date - start_date).days + 1
    trades_per_day = total / window_days if window_days > 0 else 0.0
    worst_day_pnl = min(daily_pnl.values()) if daily_pnl else 0.0
    worst_day_dd_pct = abs(worst_day_pnl) / initial_equity * 100.0 if initial_equity else 0.0
    return {
        "trades_total": total,
        "trades_per_day": trades_per_day,
        "win_rate": win_rate,
        "avg_r": avg_r,
        "max_losing_streak": max_streak,
        "worst_day_dd_pct": worst_day_dd_pct,
    }


def _load_data_cache(
    data_dir: Path,
    symbol_timeframes: Dict[str, List[str]],
) -> Tuple[Dict[Tuple[str, str], pd.DataFrame], Dict[str, str]]:
    cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    errors: Dict[str, str] = {}
    for symbol, timeframes in symbol_timeframes.items():
        for tf in timeframes:
            key = (symbol, tf)
            if key in cache:
                continue
            csv_path = _find_csv(data_dir, symbol, tf)
            if not csv_path:
                errors[f"{symbol}:{tf}"] = "missing_csv"
                continue
            try:
                cache[key] = load_ohlc_csv(csv_path, timeframe=tf)
            except Exception as exc:
                errors[f"{symbol}:{tf}"] = str(exc)
    return cache, errors


def _prepare_streams(
    profiles: List[Tuple[StrategyProfile, float]],
    data_cache: Dict[Tuple[str, str], pd.DataFrame],
    scorers_symbols: List[str],
    westbrook_symbol: str,
) -> List[Dict[str, object]]:
    streams: List[Dict[str, object]] = []
    for prof, scale in profiles:
        symbol = prof.symbol_key.upper()
        group = _profile_group(prof)
        if group == "scorers" and symbol not in scorers_symbols:
            continue
        if group == "westbrook" and symbol != westbrook_symbol:
            continue
        if isinstance(prof.signals, MomentumPinballSignalConfig):
            ohlc_m5 = data_cache.get((symbol, "M5"))
            ohlc_m15 = data_cache.get((symbol, "M15"))
            if ohlc_m5 is None or ohlc_m15 is None:
                continue
            signals = build_momentum_pinball_signals_m5(ohlc_m5, ohlc_m15, prof.signals, prof.session)
            signals = signals.reindex(ohlc_m5.index).fillna(False)
            streams.append(
                {
                    "profile": prof,
                    "scale": scale,
                    "group": group,
                    "symbol": symbol,
                    "ohlc": ohlc_m5,
                    "signals": signals,
                    "timeframe": "M5",
                }
            )
        else:
            tf = prof.timeframe.upper()
            ohlc = data_cache.get((symbol, tf))
            if ohlc is None:
                continue
            signals = _build_signals_for_profile(ohlc, prof).reindex(ohlc.index).fillna(False)
            streams.append(
                {
                    "profile": prof,
                    "scale": scale,
                    "group": group,
                    "symbol": symbol,
                    "ohlc": ohlc,
                    "signals": signals,
                    "timeframe": tf,
                }
            )
    return streams


def _window_stream(stream: Dict[str, object], start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> Optional[Dict[str, object]]:
    ohlc = stream["ohlc"]
    signals = stream["signals"]
    window_ohlc = ohlc.loc[(ohlc.index >= start_dt) & (ohlc.index <= end_dt)]
    if window_ohlc.empty or len(window_ohlc) < 2:
        return None
    window_signals = signals.loc[window_ohlc.index].fillna(False)
    signal_indices = np.flatnonzero(window_signals.values)
    return {
        **stream,
        "ohlc": window_ohlc,
        "signals": window_signals,
        "signal_indices": signal_indices,
    }


def run_window_backtest(
    streams: List[Dict[str, object]],
    start_date: date,
    end_date: date,
    initial_equity: float,
    scorers_daily_loss_pct: float,
    westbrook_daily_loss_pct: float,
    scorers_max_open: int,
    westbrook_max_open: int,
    scorers_open_risk_cap_pct: float,
    westbrook_open_risk_cap_pct: float,
    portfolio_open_risk_cap_pct: float,
    scorers_bucket_caps: Dict[str, int],
    westbrook_bucket_caps: Dict[str, int],
    scorers_override_pct: float,
) -> Dict[str, object]:
    start_dt = pd.Timestamp(datetime.combine(start_date, datetime.min.time()), tz=timezone.utc)
    end_dt = pd.Timestamp(datetime.combine(end_date, datetime.max.time()), tz=timezone.utc)

    window_streams: List[Dict[str, object]] = []
    for stream in streams:
        ws = _window_stream(stream, start_dt, end_dt)
        if ws is not None:
            window_streams.append(ws)

    event_heap: List[Tuple[pd.Timestamp, int, int]] = []
    for idx, stream in enumerate(window_streams):
        indices = stream.get("signal_indices") or []
        if len(indices) == 0:
            continue
        entry_idx = int(indices[0])
        entry_time = stream["ohlc"].index[entry_idx]
        event_heap.append((entry_time, idx, 0))
    event_heap.sort(key=lambda x: x[0])

    equity = initial_equity
    trades: List[Dict[str, object]] = []
    daily_pnl: Dict[date, float] = {}

    open_positions: List[Dict[str, object]] = []
    scorers_open = 0
    westbrook_open = 0
    scorers_open_risk = 0.0
    westbrook_open_risk = 0.0

    current_day = None
    scorers_day_start = None
    westbrook_day_start = None
    scorers_stop_hit = False
    westbrook_stop_hit = False

    while event_heap:
        entry_time, stream_idx, pos_idx = event_heap.pop(0)
        stream = window_streams[stream_idx]
        indices = stream.get("signal_indices") or []
        if pos_idx >= len(indices):
            continue
        entry_idx = int(indices[pos_idx])

        # Schedule next signal for this stream.
        next_idx = pos_idx + 1
        if next_idx < len(indices):
            next_entry_idx = int(indices[next_idx])
            next_time = stream["ohlc"].index[next_entry_idx]
            event_heap.append((next_time, stream_idx, next_idx))
            event_heap.sort(key=lambda x: x[0])

        # Close positions that have exited by this time.
        if open_positions:
            still_open: List[Dict[str, object]] = []
            for pos in open_positions:
                if pos["exit_time"] <= entry_time:
                    equity += pos["pnl"]
                    day_key = pos["exit_time"].date()
                    daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + pos["pnl"]
                    if pos["group"] == "scorers":
                        scorers_open -= 1
                        scorers_open_risk -= pos["risk_pct"]
                    else:
                        westbrook_open -= 1
                        westbrook_open_risk -= pos["risk_pct"]
                else:
                    still_open.append(pos)
            open_positions = still_open
            scorers_open = max(scorers_open, 0)
            westbrook_open = max(westbrook_open, 0)
            scorers_open_risk = max(scorers_open_risk, 0.0)
            westbrook_open_risk = max(westbrook_open_risk, 0.0)

        day_key = entry_time.date()
        if current_day != day_key:
            current_day = day_key
            scorers_day_start = equity
            westbrook_day_start = equity
            scorers_stop_hit = False
            westbrook_stop_hit = False

        if scorers_day_start and scorers_daily_loss_pct > 0:
            if equity <= scorers_day_start * (1.0 - scorers_daily_loss_pct):
                scorers_stop_hit = True
        if westbrook_day_start and westbrook_daily_loss_pct > 0:
            if equity <= westbrook_day_start * (1.0 - westbrook_daily_loss_pct):
                westbrook_stop_hit = True

        group = stream["group"]
        if group == "scorers" and scorers_stop_hit:
            continue
        if group == "westbrook" and westbrook_stop_hit:
            continue

        base_risk = float(stream["profile"].strategy.risk_per_trade_pct or 0.0) * float(stream["scale"] or 1.0)
        risk_pct = _risk_per_trade_pct(
            base_risk,
            group=group,
            scorers_override_pct=scorers_override_pct,
            scorers_cap_pct=scorers_open_risk_cap_pct,
            scorers_max_open=scorers_max_open,
            westbrook_cap_pct=westbrook_open_risk_cap_pct,
            westbrook_max_open=westbrook_max_open,
        )
        if risk_pct <= 0:
            continue

        if group == "scorers" and scorers_max_open > 0 and scorers_open >= scorers_max_open:
            continue
        if group == "westbrook" and westbrook_max_open > 0 and westbrook_open >= westbrook_max_open:
            continue

        if group == "scorers" and scorers_open_risk_cap_pct > 0 and (scorers_open_risk + risk_pct) > scorers_open_risk_cap_pct:
            continue
        if group == "westbrook" and westbrook_open_risk_cap_pct > 0 and (westbrook_open_risk + risk_pct) > westbrook_open_risk_cap_pct:
            continue
        total_open_risk = scorers_open_risk + westbrook_open_risk
        if portfolio_open_risk_cap_pct > 0 and (total_open_risk + risk_pct) > portfolio_open_risk_cap_pct:
            continue

        symbol = stream["symbol"]
        buckets = _symbol_buckets(symbol)
        caps = scorers_bucket_caps if group == "scorers" else westbrook_bucket_caps
        blocked = False
        for bucket in buckets:
            cap = caps.get(bucket)
            if cap is None:
                continue
            open_bucket = sum(1 for pos in open_positions if pos["group"] == group and bucket in pos["buckets"])
            if open_bucket + 1 > cap:
                blocked = True
                break
        if blocked:
            continue

        ohlc = stream["ohlc"]
        if entry_idx >= len(ohlc) - 1:
            continue
        reward_base = float(stream["profile"].strategy.reward_per_trade_pct or 0.0) * float(stream["scale"] or 1.0)
        reward_pct = reward_base
        if base_risk > 0:
            reward_pct = reward_base * (risk_pct / base_risk)
        cfg = replace(
            stream["profile"].strategy,
            symbol=symbol,
            risk_per_trade_pct=risk_pct,
            reward_per_trade_pct=reward_pct,
        )
        outcome = simulate_trade_path(
            ohlc=ohlc,
            entry_idx=entry_idx,
            account_balance=equity,
            config=cfg,
            costs=DEFAULT_COSTS,
        )
        entry_time = outcome.entry_time
        exit_time = outcome.exit_time
        pnl = float(outcome.pnl or 0.0)
        risk_amount = equity * risk_pct
        r_mult = pnl / risk_amount if risk_amount else 0.0

        trade = {
            "group": group,
            "symbol": symbol,
            "entry_time": entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
            "pnl": pnl,
            "r_mult": r_mult,
            "risk_pct": risk_pct,
            "exit_reason": outcome.exit_reason,
        }
        trades.append(trade)
        open_positions.append(
            {
                "group": group,
                "symbol": symbol,
                "exit_time": exit_time,
                "pnl": pnl,
                "buckets": buckets,
                "risk_pct": risk_pct,
            }
        )
        if group == "scorers":
            scorers_open += 1
            scorers_open_risk += risk_pct
        else:
            westbrook_open += 1
            westbrook_open_risk += risk_pct

    # Close remaining positions
    if open_positions:
        open_positions.sort(key=lambda pos: pos["exit_time"])
        for pos in open_positions:
            equity += pos["pnl"]
            day_key = pos["exit_time"].date()
            daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + pos["pnl"]

    stats = _window_stats(trades, daily_pnl, start_date, end_date, initial_equity)
    max_dd_pct = _max_drawdown_pct(trades, initial_equity)
    days_to_pos = _days_to_threshold(daily_pnl, start_date, end_date, initial_equity, 0.05)
    days_to_minus2 = _days_to_threshold(daily_pnl, start_date, end_date, initial_equity, -0.02)
    days_to_minus5 = _days_to_threshold(daily_pnl, start_date, end_date, initial_equity, -0.05)

    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "window_days": (end_date - start_date).days + 1,
        "trades_total": stats["trades_total"],
        "trades_per_day": stats["trades_per_day"],
        "win_rate": stats["win_rate"],
        "avg_r": stats["avg_r"],
        "max_losing_streak": stats["max_losing_streak"],
        "worst_day_dd_pct": stats["worst_day_dd_pct"],
        "max_dd_pct": max_dd_pct,
        "days_to_+5pct": days_to_pos,
        "days_to_-2pct": days_to_minus2,
        "days_to_-5pct": days_to_minus5,
    }


def _percentiles(values: List[float], pct_low: float, pct_high: float) -> Dict[str, float]:
    if not values:
        return {"p_low": 0.0, "p_high": 0.0, "median": 0.0}
    arr = np.array(values, dtype=float)
    return {
        "p_low": float(np.percentile(arr, pct_low)),
        "p_high": float(np.percentile(arr, pct_high)),
        "median": float(np.percentile(arr, 50)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rolling-window combined backtest for Scorers + Westbrook.")
    parser.add_argument("--data-dir", required=True, help="Folder containing OHLC CSVs")
    parser.add_argument("--runs", type=int, default=12, help="Number of rolling windows")
    parser.add_argument("--min-days", type=int, default=14, help="Min window days")
    parser.add_argument("--max-days", type=int, default=28, help="Max window days")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--initial-equity", type=float, default=10000.0, help="Starting equity")
    parser.add_argument("--out-json", default="reports/combined_rolling_runs.json", help="Output JSON path")
    parser.add_argument("--out-csv", default="reports/combined_rolling_runs.csv", help="Output CSV path")
    parser.add_argument("--summary-json", default="reports/combined_rolling_summary.json", help="Summary JSON path")
    parser.add_argument("--scorers-symbols", default="USDJPY,GBPJPY,EURUSD,GBPUSD", help="Scorers symbols list")
    parser.add_argument("--westbrook-symbol", default="USDJPY", help="Westbrook symbol")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    scorers_symbols = _parse_symbols(args.scorers_symbols)
    westbrook_symbol = args.westbrook_symbol.upper()
    if args.seed is not None:
        random.seed(args.seed)

    scorers_daily_loss_pct = float(
        os.getenv("OMEGAFX_SCORERS_DAILY_LOSS_PCT", "0.01")
    )
    westbrook_daily_loss_pct = float(
        os.getenv("OMEGAFX_WESTBROOK_DAILY_LOSS_PCT", "0.02")
    )
    scorers_max_open = int(os.getenv("OMEGAFX_SCORERS_MAX_OPEN_POSITIONS", "4"))
    westbrook_max_open = int(os.getenv("OMEGAFX_WESTBROOK_MAX_OPEN_POSITIONS", "6"))
    scorers_open_risk_cap_pct = float(os.getenv("OMEGAFX_SCORERS_MAX_TOTAL_OPEN_RISK_PCT", "0.01"))
    westbrook_open_risk_cap_pct = float(os.getenv("OMEGAFX_WESTBROOK_MAX_TOTAL_OPEN_RISK_PCT", "0.0075"))
    portfolio_open_risk_cap_pct = float(os.getenv("OMEGAFX_PORTFOLIO_OPEN_RISK_CAP_PCT", "0.02"))
    scorers_override_pct = float(os.getenv("OMEGAFX_SCORERS_RISK_PER_TRADE_PCT", "0.001"))
    scorers_bucket_caps = {
        "JPY": int(os.getenv("OMEGAFX_SCORERS_BUCKET_CAP_JPY", "1")),
    }
    westbrook_bucket_caps = {
        "JPY": int(os.getenv("OMEGAFX_WESTBROOK_BUCKET_CAP_JPY", "1")),
        "USD": int(os.getenv("OMEGAFX_WESTBROOK_BUCKET_CAP_USD", "1")),
    }

    symbol_timeframes: Dict[str, List[str]] = {}
    for sym in scorers_symbols:
        symbol_timeframes.setdefault(sym, [])
        if "M15" not in symbol_timeframes[sym]:
            symbol_timeframes[sym].append("M15")
    symbol_timeframes.setdefault(westbrook_symbol, [])
    for tf in ("M5", "M15"):
        if tf not in symbol_timeframes[westbrook_symbol]:
            symbol_timeframes[westbrook_symbol].append(tf)

    data_cache, errors = _load_data_cache(data_dir, symbol_timeframes)
    if errors:
        raise SystemExit(f"Data load errors: {errors}")

    profiles = _iter_profiles(MULTI_PORTFOLIO_COMBINED)
    streams = _prepare_streams(profiles, data_cache, scorers_symbols, westbrook_symbol)
    if not streams:
        raise SystemExit("No streams prepared; check symbols/timeframes.")

    min_dates = []
    max_dates = []
    for stream in streams:
        idx = stream["ohlc"].index
        if idx.empty:
            continue
        min_dates.append(idx.min().date())
        max_dates.append(idx.max().date())
    if not min_dates or not max_dates:
        raise SystemExit("No data ranges available.")
    global_start = max(min_dates)
    global_end = min(max_dates)
    today = datetime.now(timezone.utc).date()
    cutoff_end = min(global_end, today - timedelta(days=1))
    if cutoff_end <= global_start:
        raise SystemExit("Insufficient data range for rolling windows.")

    runs: List[Dict[str, object]] = []
    attempts = 0
    max_attempts = args.runs * 20
    while len(runs) < args.runs and attempts < max_attempts:
        attempts += 1
        window_days = random.randint(args.min_days, args.max_days)
        latest_start = cutoff_end - timedelta(days=window_days - 1)
        if latest_start <= global_start:
            break
        start_offset = random.randint(0, (latest_start - global_start).days)
        start_date = global_start + timedelta(days=start_offset)
        end_date = start_date + timedelta(days=window_days - 1)
        run_result = run_window_backtest(
            streams=streams,
            start_date=start_date,
            end_date=end_date,
            initial_equity=args.initial_equity,
            scorers_daily_loss_pct=scorers_daily_loss_pct,
            westbrook_daily_loss_pct=westbrook_daily_loss_pct,
            scorers_max_open=scorers_max_open,
            westbrook_max_open=westbrook_max_open,
            scorers_open_risk_cap_pct=scorers_open_risk_cap_pct,
            westbrook_open_risk_cap_pct=westbrook_open_risk_cap_pct,
            portfolio_open_risk_cap_pct=portfolio_open_risk_cap_pct,
            scorers_bucket_caps=scorers_bucket_caps,
            westbrook_bucket_caps=westbrook_bucket_caps,
            scorers_override_pct=scorers_override_pct,
        )
        runs.append(run_result)

    if len(runs) < args.runs:
        raise SystemExit(f"Only generated {len(runs)} runs after {attempts} attempts.")

    trades_per_day_vals = [float(r["trades_per_day"]) for r in runs]
    worst_day_vals = [float(r["worst_day_dd_pct"]) for r in runs]
    max_dd_vals = [float(r["max_dd_pct"]) for r in runs]
    violations = [
        r for r in runs if float(r["worst_day_dd_pct"]) > 5.0 or float(r["max_dd_pct"]) > 10.0
    ]
    passed_14 = [r for r in runs if r.get("days_to_+5pct") is not None and r["days_to_+5pct"] <= 14]
    summary = {
        "runs": len(runs),
        "trades_per_day": _percentiles(trades_per_day_vals, 25, 75),
        "worst_day_dd_pct": _percentiles(worst_day_vals, 25, 75),
        "max_dd_pct": _percentiles(max_dd_vals, 25, 75),
        "violations_prop_rules_pct": (len(violations) / len(runs) * 100.0) if runs else 0.0,
        "passes_within_14_days_pct": (len(passed_14) / len(runs) * 100.0) if runs else 0.0,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "settings": {
                    "scorers_symbols": scorers_symbols,
                    "westbrook_symbol": westbrook_symbol,
                    "scorers_daily_loss_pct": scorers_daily_loss_pct,
                    "westbrook_daily_loss_pct": westbrook_daily_loss_pct,
                    "scorers_max_open": scorers_max_open,
                    "westbrook_max_open": westbrook_max_open,
                    "scorers_open_risk_cap_pct": scorers_open_risk_cap_pct,
                    "westbrook_open_risk_cap_pct": westbrook_open_risk_cap_pct,
                    "portfolio_open_risk_cap_pct": portfolio_open_risk_cap_pct,
                    "scorers_risk_per_trade_pct": scorers_override_pct,
                    "scorers_bucket_caps": scorers_bucket_caps,
                    "westbrook_bucket_caps": westbrook_bucket_caps,
                    "window_days_min": args.min_days,
                    "window_days_max": args.max_days,
                    "data_dir": str(data_dir),
                },
                "runs": runs,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    out_csv = Path(args.out_csv)
    if runs:
        df = pd.DataFrame(runs)
        df.to_csv(out_csv, index=False)

    summary_json = Path(args.summary_json)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {summary_json}")
    return 0


if __name__ == "__main__":
    import os

    raise SystemExit(main())
