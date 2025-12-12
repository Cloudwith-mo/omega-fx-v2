"""Omega FX v2 core package."""

from .config import (
    DEFAULT_COSTS,
    DEFAULT_CHALLENGE,
    DEFAULT_SESSION,
    DEFAULT_STRATEGY,
    DEFAULT_SIGNAL_CONFIG,
    ChallengeProfile,
    InstrumentSpec,
    SignalConfig,
    StrategyConfig,
    TradingCosts,
    TradingSession,
    XAUUSD_SPEC,
)
from .data import fetch_xauusd_ohlc
from .signals import (
    build_atr_filter,
    build_signals,
    build_session_mask,
    compute_h4_sma_filter,
    compute_atr,
    generate_breakout_signals,
)
from .strategy import PlannedTrade, plan_single_trade
from .sim import (
    EvaluationBatchResult,
    EvaluationResult,
    TradeOutcome,
    run_randomized_evaluations,
    run_randomized_signal_evaluations,
    run_signal_driven_evaluation,
    run_sequential_evaluation,
    simulate_trade_path,
)

__all__ = [
    "StrategyConfig",
    "TradingCosts",
    "ChallengeProfile",
    "TradingSession",
    "InstrumentSpec",
    "DEFAULT_STRATEGY",
    "DEFAULT_COSTS",
    "DEFAULT_CHALLENGE",
    "DEFAULT_SESSION",
    "DEFAULT_SIGNAL_CONFIG",
    "XAUUSD_SPEC",
    "SignalConfig",
    "build_atr_filter",
    "build_signals",
    "build_session_mask",
    "compute_h4_sma_filter",
    "compute_atr",
    "generate_breakout_signals",
    "fetch_xauusd_ohlc",
    "PlannedTrade",
    "plan_single_trade",
    "TradeOutcome",
    "simulate_trade_path",
    "run_sequential_evaluation",
    "run_signal_driven_evaluation",
    "EvaluationResult",
    "run_randomized_evaluations",
    "run_randomized_signal_evaluations",
    "EvaluationBatchResult",
]
