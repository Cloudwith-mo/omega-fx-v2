"""Omega FX v2 core package."""

from .config import DEFAULT_STRATEGY, InstrumentSpec, StrategyConfig, XAUUSD_SPEC
from .strategy import PlannedTrade, plan_single_trade

__all__ = [
    "StrategyConfig",
    "InstrumentSpec",
    "DEFAULT_STRATEGY",
    "XAUUSD_SPEC",
    "PlannedTrade",
    "plan_single_trade",
]
