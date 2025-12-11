from dataclasses import dataclass

from .config import DEFAULT_STRATEGY, StrategyConfig


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


def plan_single_trade(
    account_balance: float,
    current_price: float,
    config: StrategyConfig = DEFAULT_STRATEGY,
) -> PlannedTrade:
    """
    Compute a simple single-trade plan:
    - assume a long trade at a fixed lot size
    - risk a fixed percent of the account, target a fixed percent reward
    - ignore exact stop-loss / take-profit price math until broker specs are added
    """
    # Account-level math only; price is included for future compatibility.
    _ = current_price

    risk_amount = account_balance * config.risk_per_trade_pct
    reward_amount = account_balance * config.reward_per_trade_pct
    account_profit_target = account_balance * config.profit_target_pct

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
    )
