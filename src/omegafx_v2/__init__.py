"""Omega FX v2 core package."""

from .config import (
    DEFAULT_COSTS,
    DEFAULT_STRATEGY,
    InstrumentSpec,
    StrategyConfig,
    TradingCosts,
    XAUUSD_SPEC,
)
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
    "TradingCosts",
    "InstrumentSpec",
    "DEFAULT_STRATEGY",
    "DEFAULT_COSTS",
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
