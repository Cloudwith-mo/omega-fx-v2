from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import List

import pandas as pd

from .config import StrategyProfile
from .data import fetch_symbol_ohlc
from .sim import simulate_trade_path
from .config import DEFAULT_COSTS
from .profile_summary import _build_signals_for_profile


@dataclass
class LiveTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    exit_reason: str
    pnl: float
    pnl_pct: float
    equity_after: float


@dataclass
class LiveRunResult:
    profile_name: str
    symbol_key: str
    initial_equity: float
    final_equity: float
    trades: List[LiveTrade]


def run_offline_live_simulation(
    profile: StrategyProfile,
    days: int = 365,
    initial_equity: float = 10_000.0,
) -> LiveRunResult:
    """
    Offline 'live' simulation for a StrategyProfile:
    - Fetches recent data for profile.symbol_key.
    - Builds signals using the profile's signal config + session.
    - Steps through signals in time order; simulates trades sequentially.
    """
    end = date.today()
    start = end - timedelta(days=days)

    ohlc = fetch_symbol_ohlc(
        profile.symbol_key,
        start=start.isoformat(),
        end=end.isoformat(),
    )

    signals = _build_signals_for_profile(ohlc, profile)

    strategy_cfg = replace(
        profile.strategy,
        profit_target_pct=profile.challenge.profit_target_pct,
    )

    equity = initial_equity
    trades: List[LiveTrade] = []

    target_equity = initial_equity * (1.0 + profile.challenge.profit_target_pct)
    loss_limit_equity = initial_equity * (1.0 - profile.challenge.max_total_loss_pct)

    dates = ohlc.index.date
    daily_pnl: dict[object, float] = {}

    signal_indices = [i for i, flag in enumerate(signals.values) if flag]
    last_exit_pos = -1

    for idx in signal_indices:
        if idx <= last_exit_pos:
            continue

        if equity >= target_equity or equity <= loss_limit_equity:
            break

        if profile.challenge.daily_loss_pct is not None:
            day = dates[idx]
            day_pnl = daily_pnl.get(day, 0.0)
            day_limit = initial_equity * profile.challenge.daily_loss_pct
            if day_pnl <= -day_limit:
                continue

        if idx >= len(ohlc) - 1:
            break

        outcome = simulate_trade_path(
            ohlc=ohlc,
            entry_idx=idx,
            account_balance=equity,
            config=strategy_cfg,
            costs=DEFAULT_COSTS,
        )

        equity_after = equity + outcome.pnl

        if profile.challenge.daily_loss_pct is not None:
            day = dates[idx]
            daily_pnl[day] = daily_pnl.get(day, 0.0) + outcome.pnl

        trades.append(
            LiveTrade(
                entry_time=outcome.entry_time,
                exit_time=outcome.exit_time,
                entry_price=outcome.entry_price,
                exit_price=outcome.exit_price,
                exit_reason=outcome.exit_reason,
                pnl=outcome.pnl,
                pnl_pct=outcome.pnl_pct,
                equity_after=equity_after,
            )
        )

        equity = equity_after

        try:
            exit_pos = ohlc.index.get_loc(outcome.exit_time)
        except KeyError:
            break

        last_exit_pos = exit_pos

    return LiveRunResult(
        profile_name=profile.name,
        symbol_key=profile.symbol_key,
        initial_equity=initial_equity,
        final_equity=equity,
        trades=trades,
    )
