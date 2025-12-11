from datetime import date, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from omegafx_v2.data import fetch_xauusd_ohlc
from omegafx_v2.sim import simulate_trade_path


def main() -> None:
    end = date.today()
    start = end - timedelta(days=30)  # ~1 month of 1h bars

    ohlc = fetch_xauusd_ohlc(start.isoformat(), end.isoformat())
    print(f"Fetched {len(ohlc)} bars of XAUUSD 1h data")

    account_balance = 10_000.0
    entry_idx = 5

    outcome = simulate_trade_path(
        ohlc=ohlc,
        entry_idx=entry_idx,
        account_balance=account_balance,
    )

    print("--- Trade outcome ---")
    print(f"Entry: {outcome.entry_time}, price={outcome.entry_price:.2f}")
    print(f"Exit:  {outcome.exit_time}, price={outcome.exit_price:.2f}, reason={outcome.exit_reason}")
    print(f"PnL:   {outcome.pnl:.2f} ({outcome.pnl_pct:.2%})")


if __name__ == "__main__":
    main()
