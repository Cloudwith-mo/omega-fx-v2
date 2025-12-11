"""Omega FX v2 core package."""

from .config import DEFAULT_STRATEGY, InstrumentSpec, StrategyConfig, XAUUSD_SPEC
from .data import fetch_xauusd_ohlc
from .strategy import PlannedTrade, plan_single_trade
from .sim import (
    EvaluationBatchResult,
    EvaluationResult,
    TradeOutcome,
    run_randomized_evaluations,
    run_sequential_evaluation,
    simulate_trade_path,
)

__all__ = [
    "StrategyConfig",
    "InstrumentSpec",
    "DEFAULT_STRATEGY",
    "XAUUSD_SPEC",
    "fetch_xauusd_ohlc",
    "PlannedTrade",
    "plan_single_trade",
    "TradeOutcome",
    "simulate_trade_path",
    "run_sequential_evaluation",
    "EvaluationResult",
    "run_randomized_evaluations",
    "EvaluationBatchResult",
]
