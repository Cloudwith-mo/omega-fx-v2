from __future__ import annotations

from dataclasses import dataclass
import random

import pandas as pd
from typing import List, Optional

from .config import (
    DEFAULT_CHALLENGE,
    DEFAULT_COSTS,
    DEFAULT_STRATEGY,
    ChallengeProfile,
    StrategyConfig,
    TradingCosts,
    XAUUSD_SPEC,
)
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


@dataclass
class EvaluationBatchResult:
    evaluations: List["EvaluationResult"]
    num_evals: int
    target_hit_count: int
    max_loss_count: int
    data_exhausted_count: int
    pass_rate: float
    average_return: float
    average_max_drawdown: float


def simulate_trade_path(
    ohlc: pd.DataFrame,
    entry_idx: int,
    account_balance: float,
    config: StrategyConfig = DEFAULT_STRATEGY,
    costs: Optional[TradingCosts] = DEFAULT_COSTS,
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
    gross_pnl = pips_move * XAUUSD_SPEC.pip_value_per_lot * trade_plan.lot_size

    round_trip_cost = 0.0
    if costs is not None:
        round_trip_cost = (
            costs.spread_pips * XAUUSD_SPEC.pip_value_per_lot * trade_plan.lot_size
            + costs.commission_per_lot_round_trip * trade_plan.lot_size
        )

    pnl = gross_pnl - round_trip_cost
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
    costs: Optional[TradingCosts] = DEFAULT_COSTS,
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
            costs=costs,
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


def run_randomized_evaluations(
    ohlc: pd.DataFrame,
    initial_equity: float,
    num_evals: int = 100,
    min_bars_per_eval: int = 500,
    start_offset: int = 5,
    config: StrategyConfig = DEFAULT_STRATEGY,
    max_total_loss_pct: float = 0.06,
    costs: Optional[TradingCosts] = DEFAULT_COSTS,
) -> EvaluationBatchResult:
    """
    Run multiple sequential evaluations starting at random indices within the dataset.
    """
    n_bars = len(ohlc)
    if n_bars < min_bars_per_eval + start_offset:
        raise ValueError("Not enough data for requested eval length")

    max_start = n_bars - min_bars_per_eval
    evaluations: List[EvaluationResult] = []

    for _ in range(num_evals):
        entry_idx = random.randint(start_offset, max_start)
        result = run_sequential_evaluation(
            ohlc=ohlc,
            initial_equity=initial_equity,
            start_idx=entry_idx,
            config=config,
            max_total_loss_pct=max_total_loss_pct,
            costs=costs,
        )
        evaluations.append(result)

    num_evals_actual = len(evaluations)
    target_hit_count = sum(1 for r in evaluations if r.verdict == "target_hit")
    max_loss_count = sum(1 for r in evaluations if r.verdict == "max_loss_breached")
    data_exhausted_count = sum(1 for r in evaluations if r.verdict == "data_exhausted")

    if num_evals_actual > 0:
        pass_rate = target_hit_count / num_evals_actual
        average_return = sum(r.total_return for r in evaluations) / num_evals_actual
        average_max_drawdown = (
            sum(r.max_drawdown_pct for r in evaluations) / num_evals_actual
        )
    else:
        pass_rate = 0.0
        average_return = 0.0
        average_max_drawdown = 0.0

    return EvaluationBatchResult(
        evaluations=evaluations,
        num_evals=num_evals_actual,
        target_hit_count=target_hit_count,
        max_loss_count=max_loss_count,
        data_exhausted_count=data_exhausted_count,
        pass_rate=pass_rate,
        average_return=average_return,
        average_max_drawdown=average_max_drawdown,
    )


def run_randomized_signal_evaluations(
    ohlc: pd.DataFrame,
    signals: pd.Series,
    initial_equity: float,
    challenge: ChallengeProfile = DEFAULT_CHALLENGE,
    config: StrategyConfig = DEFAULT_STRATEGY,
    costs: Optional[TradingCosts] = DEFAULT_COSTS,
    num_evals: int = 100,
    min_bars_per_eval: Optional[int] = None,
    start_offset: int = 5,
) -> EvaluationBatchResult:
    """
    Run multiple signal-driven evaluations starting at randomized indices.
    """
    if min_bars_per_eval is None:
        min_bars_per_eval = challenge.min_bars_per_eval

    signals = signals.reindex(ohlc.index).fillna(False)

    n_bars = len(ohlc)
    if n_bars < min_bars_per_eval + start_offset:
        raise ValueError("Not enough data for requested eval length")

    max_start = n_bars - min_bars_per_eval
    evaluations: List[EvaluationResult] = []

    for _ in range(num_evals):
        start_idx = random.randint(start_offset, max_start)
        ohlc_slice = ohlc.iloc[start_idx:]
        signals_slice = signals.iloc[start_idx:]

        result = run_signal_driven_evaluation(
            ohlc=ohlc_slice,
            signals=signals_slice,
            initial_equity=initial_equity,
            challenge=challenge,
            config=config,
            costs=costs,
        )
        evaluations.append(result)

    num_evals_actual = len(evaluations)
    target_hit_count = sum(1 for r in evaluations if r.verdict == "target_hit")
    max_loss_count = sum(1 for r in evaluations if r.verdict == "max_loss_breached")
    data_exhausted_count = sum(1 for r in evaluations if r.verdict == "data_exhausted")

    if num_evals_actual > 0:
        pass_rate = target_hit_count / num_evals_actual
        average_return = sum(r.total_return for r in evaluations) / num_evals_actual
        average_max_drawdown = (
            sum(r.max_drawdown_pct for r in evaluations) / num_evals_actual
        )
    else:
        pass_rate = 0.0
        average_return = 0.0
        average_max_drawdown = 0.0

    return EvaluationBatchResult(
        evaluations=evaluations,
        num_evals=num_evals_actual,
        target_hit_count=target_hit_count,
        max_loss_count=max_loss_count,
        data_exhausted_count=data_exhausted_count,
        pass_rate=pass_rate,
        average_return=average_return,
        average_max_drawdown=average_max_drawdown,
    )

def run_signal_driven_evaluation(
    ohlc: pd.DataFrame,
    signals: pd.Series,
    initial_equity: float,
    challenge: ChallengeProfile = DEFAULT_CHALLENGE,
    config: StrategyConfig = DEFAULT_STRATEGY,
    costs: Optional[TradingCosts] = DEFAULT_COSTS,
) -> EvaluationResult:
    """
    Evaluate using signal-driven entries; skip signals until the prior trade exits.
    """
    signals = signals.reindex(ohlc.index).fillna(False)

    equity = initial_equity
    trades: list[TradeOutcome] = []

    target_equity = initial_equity * (1.0 + challenge.profit_target_pct)
    loss_limit_equity = initial_equity * (1.0 - challenge.max_total_loss_pct)

    max_equity = equity
    min_equity = equity

    signal_indices = [i for i, flag in enumerate(signals.values) if flag]
    last_exit_pos = -1

    for idx in signal_indices:
        if idx <= last_exit_pos:
            continue

        if equity >= target_equity or equity <= loss_limit_equity:
            break

        if idx >= len(ohlc) - 1:
            break

        outcome = simulate_trade_path(
            ohlc=ohlc,
            entry_idx=idx,
            account_balance=equity,
            config=config,
            costs=costs,
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

        last_exit_pos = exit_pos

    num_trades = len(trades)
    num_wins = sum(1 for t in trades if t.pnl > 0)
    num_losses = sum(1 for t in trades if t.pnl < 0)

    total_return = equity / initial_equity - 1.0 if initial_equity > 0 else 0.0
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
