from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from typing import List

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


@dataclass
class EvaluationResult:
    initial_equity: float
    final_equity: float
    total_return: float  # final_equity / initial_equity - 1
    profit_target_hit: bool
    trades: List[TradeOutcome]
    num_trades: int
    num_wins: int
    num_losses: int
    max_equity: float
    min_equity: float
    max_drawdown_pct: float  # positive number, e.g. 0.05 for 5% DD
    verdict: str  # "target_hit" | "max_loss_breached" | "data_exhausted"


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


def run_sequential_evaluation(
    ohlc: pd.DataFrame,
    initial_equity: float,
    start_idx: int = 0,
    config: StrategyConfig = DEFAULT_STRATEGY,
    max_total_loss_pct: float = 0.06,
) -> EvaluationResult:
    """
    Run sequential trades until the profit target is hit or data ends.
    Each trade uses the current equity for position sizing.
    """
    equity = initial_equity
    trades: List[TradeOutcome] = []
    target_equity = initial_equity * (1.0 + config.profit_target_pct)
    loss_limit_equity = initial_equity * (1.0 - max_total_loss_pct)

    idx = start_idx
    n_bars = len(ohlc)
    max_equity = equity
    min_equity = equity

    while idx < n_bars - 1 and equity < target_equity and equity > loss_limit_equity:
        outcome = simulate_trade_path(
            ohlc=ohlc,
            entry_idx=idx,
            account_balance=equity,
            config=config,
        )
        trades.append(outcome)
        equity += outcome.pnl

        if equity > max_equity:
            max_equity = equity
        if equity < min_equity:
            min_equity = equity

        try:
            exit_pos = ohlc.index.get_loc(outcome.exit_time)
        except KeyError:
            break

        idx = exit_pos + 1

    num_trades = len(trades)
    num_wins = sum(1 for t in trades if t.pnl > 0)
    num_losses = sum(1 for t in trades if t.pnl < 0)

    total_return = 0.0
    if initial_equity > 0.0:
        total_return = equity / initial_equity - 1.0

    profit_target_hit = equity >= target_equity
    if max_equity > 0:
        max_drawdown_pct = (max_equity - min_equity) / max_equity
    else:
        max_drawdown_pct = 0.0

    if profit_target_hit:
        verdict = "target_hit"
    elif equity <= loss_limit_equity:
        verdict = "max_loss_breached"
    else:
        verdict = "data_exhausted"

    return EvaluationResult(
        initial_equity=initial_equity,
        final_equity=equity,
        total_return=total_return,
        profit_target_hit=profit_target_hit,
        trades=trades,
        num_trades=num_trades,
        num_wins=num_wins,
        num_losses=num_losses,
        max_equity=max_equity,
        min_equity=min_equity,
        max_drawdown_pct=max_drawdown_pct,
        verdict=verdict,
    )
