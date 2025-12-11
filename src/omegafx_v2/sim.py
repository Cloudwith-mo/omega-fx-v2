from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import DEFAULT_STRATEGY, StrategyConfig, XAUUSD_SPEC
from .strategy import plan_single_trade


@dataclass
class TradeOutcome:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    exit_reason: str  # "tp" | "sl" | "end"
    pnl: float
    pnl_pct: float


def simulate_trade_path(
    ohlc: pd.DataFrame,
    entry_idx: int,
    account_balance: float,
    config: StrategyConfig = DEFAULT_STRATEGY,
) -> TradeOutcome:
    """
    Simulate a single long trade on XAUUSD:
    - Use close at `entry_idx` as the entry price.
    - Use plan_single_trade to compute SL/TP levels.
    - Walk forward bar by bar; check if SL or TP is hit.
    - If both hit in same candle, assume SL first.
    - If neither hit before data ends, close at final close.
    """
    if entry_idx < 0 or entry_idx >= len(ohlc) - 1:
        raise IndexError("entry_idx must point to a bar with at least one bar after it")

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(ohlc.columns):
        raise ValueError(f"OHLC data must include columns {required_cols}")

    entry_time = ohlc.index[entry_idx]
    entry_price = float(ohlc["close"].iloc[entry_idx])

    trade_plan = plan_single_trade(
        account_balance=account_balance,
        current_price=entry_price,
        config=config,
    )

    sl = trade_plan.stop_loss_price
    tp = trade_plan.take_profit_price

    exit_time = ohlc.index[-1]
    exit_price = float(ohlc["close"].iloc[-1])
    exit_reason = "end"

    for ts, row in ohlc.iloc[entry_idx + 1 :].iterrows():
        low = float(row["low"])
        high = float(row["high"])

        hit_sl = low <= sl
        hit_tp = high >= tp

        if hit_sl and hit_tp:
            exit_time = ts
            exit_price = sl
            exit_reason = "sl"
            break
        if hit_sl:
            exit_time = ts
            exit_price = sl
            exit_reason = "sl"
            break
        if hit_tp:
            exit_time = ts
            exit_price = tp
            exit_reason = "tp"
            break

    price_move = exit_price - entry_price
    pips_move = price_move / XAUUSD_SPEC.pip_size
    pnl = pips_move * XAUUSD_SPEC.pip_value_per_lot * trade_plan.lot_size
    pnl_pct = pnl / account_balance

    return TradeOutcome(
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        exit_reason=exit_reason,
        pnl=pnl,
        pnl_pct=pnl_pct,
    )
