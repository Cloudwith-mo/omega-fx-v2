from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from omegafx_v2.config import DEFAULT_STRATEGY
from omegafx_v2.strategy import plan_single_trade


def main() -> None:
    # Example inputs — adjust later or wire into CLI args.
    account_balance = 10_000.0
    current_price = 2_400.0  # example gold price

    trade = plan_single_trade(account_balance, current_price, DEFAULT_STRATEGY)

    print(f"Symbol: {trade.symbol}")
    print(f"Direction: {trade.direction}")
    print(f"Lot size: {trade.lot_size}")
    print(f"Account balance: {trade.account_balance:,.2f}")
    print(f"Risk: {trade.risk_amount:,.2f} ({trade.risk_pct:.2%})")
    print(f"Reward: {trade.reward_amount:,.2f} ({trade.reward_pct:.2%})")
    print(f"Account profit target (7%): {trade.account_profit_target:,.2f}")


if __name__ == "__main__":
    main()
