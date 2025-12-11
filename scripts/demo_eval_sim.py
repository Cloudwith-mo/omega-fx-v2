from datetime import date, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from omegafx_v2.data import fetch_xauusd_ohlc
from omegafx_v2.sim import run_sequential_evaluation


def main() -> None:
    end = date.today()
    start = end - timedelta(days=90)  # ~3 months of 1h bars

    ohlc = fetch_xauusd_ohlc(start.isoformat(), end.isoformat())
    print(f"Fetched {len(ohlc)} XAUUSD 1h bars from {start} to {end}")

    initial_equity = 10_000.0

    result = run_sequential_evaluation(
        ohlc=ohlc,
        initial_equity=initial_equity,
        start_idx=5,
    )

    print("\n--- Evaluation summary ---")
    print(f"Initial equity: {result.initial_equity:,.2f}")
    print(f"Final equity:   {result.final_equity:,.2f}")
    print(f"Total return:   {result.total_return:.2%}")
    print(f"Profit target hit (7%): {result.profit_target_hit}")
    print(f"Trades: {result.num_trades}, wins: {result.num_wins}, losses: {result.num_losses}")

    if result.trades:
        first = result.trades[0]
        last = result.trades[-1]
        print("\nFirst trade:")
        print(f"  {first.entry_time} -> {first.exit_time}, reason={first.exit_reason}, PnL={first.pnl:.2f}")
        print("Last trade:")
        print(f"  {last.entry_time} -> {last.exit_time}, reason={last.exit_reason}, PnL={last.pnl:.2f}")


if __name__ == "__main__":
    main()
