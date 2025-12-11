from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    pip_size: float  # minimum price increment
    pip_value_per_lot: float  # account currency value per pip for 1.0 lot


XAUUSD_SPEC = InstrumentSpec(
    symbol="XAUUSD",
    pip_size=0.01,
    pip_value_per_lot=1.0,  # 1 USD per 0.01 move per lot
)


@dataclass(frozen=True)
class TradingCosts:
    spread_pips: float  # one-way spread measured in pips
    commission_per_lot_round_trip: float  # currency per 1.0 lot round-trip


# Example default costs for XAUUSD – adjust to your broker
DEFAULT_COSTS = TradingCosts(
    spread_pips=20.0,  # 0.20 in price with pip_size=0.01
    commission_per_lot_round_trip=7.0,
)


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str
    fixed_lot_size: float
    profit_target_pct: float  # e.g. 0.07 for 7% account target
    risk_per_trade_pct: float  # 0.02 = 2% of account
    reward_per_trade_pct: float  # 0.04 = 4% of account


# Default config for the current plan
DEFAULT_STRATEGY = StrategyConfig(
    symbol="XAUUSD",  # gold
    fixed_lot_size=2.0,
    profit_target_pct=0.07,  # 7% account target
    risk_per_trade_pct=0.02,  # 2% risk
    reward_per_trade_pct=0.04,  # 4% reward
)
