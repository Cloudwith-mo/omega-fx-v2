from dataclasses import dataclass

from .config import DEFAULT_STRATEGY, InstrumentSpec, StrategyConfig, XAUUSD_SPEC


@dataclass
class PlannedTrade:
    symbol: str
    direction: str  # "long" | "short"
    lot_size: float
    account_balance: float
    risk_amount: float  # currency amount at risk
    risk_pct: float
    reward_amount: float  # currency amount targeted
    reward_pct: float
    account_profit_target: float  # currency amount for 7% target
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    risk_pips: float
    reward_pips: float


def plan_single_trade(
    account_balance: float,
    current_price: float,
    config: StrategyConfig = DEFAULT_STRATEGY,
    instrument: InstrumentSpec = XAUUSD_SPEC,
) -> PlannedTrade:
    """
    Plan a single long trade on XAUUSD:
    - Fixed lot size from config
    - Risk a fixed percent of the account, target a fixed percent reward
    - Compute stop-loss and take-profit prices from pip value assumptions
    """
    risk_amount = account_balance * config.risk_per_trade_pct
    reward_amount = account_balance * config.reward_per_trade_pct
    account_profit_target = account_balance * config.profit_target_pct
    risk_pips = risk_amount / (instrument.pip_value_per_lot * config.fixed_lot_size)
    reward_pips = reward_amount / (instrument.pip_value_per_lot * config.fixed_lot_size)

    entry_price = current_price
    stop_loss_price = entry_price - risk_pips * instrument.pip_size
    take_profit_price = entry_price + reward_pips * instrument.pip_size

    return PlannedTrade(
        symbol=config.symbol,
        direction="long",
        lot_size=config.fixed_lot_size,
        account_balance=account_balance,
        risk_amount=risk_amount,
        risk_pct=config.risk_per_trade_pct,
        reward_amount=reward_amount,
        reward_pct=config.reward_per_trade_pct,
        account_profit_target=account_profit_target,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        risk_pips=risk_pips,
        reward_pips=reward_pips,
    )
