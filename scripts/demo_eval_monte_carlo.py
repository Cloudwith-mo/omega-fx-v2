from datetime import date, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from omegafx_v2.data import fetch_xauusd_ohlc
from omegafx_v2.sim import run_randomized_evaluations


def main() -> None:
    end = date.today()
    start = end - timedelta(days=365)  # ~1 year of 1h bars

    ohlc = fetch_xauusd_ohlc(start.isoformat(), end.isoformat())
    print(f"Fetched {len(ohlc)} XAUUSD 1h bars from {start} to {end}")

    initial_equity = 10_000.0
    num_evals = 100

    batch = run_randomized_evaluations(
        ohlc=ohlc,
        initial_equity=initial_equity,
        num_evals=num_evals,
        min_bars_per_eval=500,
        start_offset=5,
        max_total_loss_pct=0.06,
    )

    print("\n--- Batch evaluation summary ---")
    print(f"Evals run:      {batch.num_evals}")
    print(f"Target hits:    {batch.target_hit_count}")
    print(f"Max loss hits:  {batch.max_loss_count}")
    print(f"Data exhausted: {batch.data_exhausted_count}")
    print(f"Pass rate (hit 7% before -6%): {batch.pass_rate:.2%}")
    print(f"Avg total return:              {batch.average_return:.2%}")
    print(f"Avg max drawdown:              {batch.average_max_drawdown:.2%}")


if __name__ == "__main__":
    main()
