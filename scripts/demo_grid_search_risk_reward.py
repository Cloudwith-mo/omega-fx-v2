from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from omegafx_v2.config import DEFAULT_CHALLENGE, DEFAULT_COSTS, DEFAULT_STRATEGY
from omegafx_v2.data import fetch_xauusd_ohlc
from omegafx_v2.sim import run_randomized_evaluations


def main() -> None:
    end = date.today()
    start = end - timedelta(days=365)  # ~1 year

    ohlc = fetch_xauusd_ohlc(start.isoformat(), end.isoformat())
    print(f"Fetched {len(ohlc)} XAUUSD 1h bars from {start} to {end}")

    initial_equity = 10_000.0
    num_evals = 100
    challenge = DEFAULT_CHALLENGE

    risk_values = [0.01, 0.015, 0.02]  # 1%, 1.5%, 2%
    reward_values = [0.02, 0.03, 0.04]  # 2%, 3%, 4%

    print("\nChallenge profile:")
    print(f"  Name:              {challenge.name}")
    print(f"  Profit target:     {challenge.profit_target_pct:.2%}")
    print(f"  Max total loss:    {challenge.max_total_loss_pct:.2%}")
    print(f"  Min bars per eval: {challenge.min_bars_per_eval}")

    print("\nrisk_pct  reward_pct  pass_rate  avg_ret  avg_max_dd")
    print("-----------------------------------------------------")

    for risk_pct in risk_values:
        for reward_pct in reward_values:
            cfg = replace(
                DEFAULT_STRATEGY,
                risk_per_trade_pct=risk_pct,
                reward_per_trade_pct=reward_pct,
                profit_target_pct=challenge.profit_target_pct,
            )

            batch = run_randomized_evaluations(
                ohlc=ohlc,
                initial_equity=initial_equity,
                num_evals=num_evals,
                min_bars_per_eval=challenge.min_bars_per_eval,
                start_offset=5,
                config=cfg,
                max_total_loss_pct=challenge.max_total_loss_pct,
                costs=DEFAULT_COSTS,
            )

            print(
                f"{risk_pct:7.3f}  {reward_pct:10.3f}  "
                f"{batch.pass_rate:9.3f}  "
                f"{batch.average_return:7.3f}  "
                f"{batch.average_max_drawdown:11.3f}"
            )


if __name__ == "__main__":
    main()
