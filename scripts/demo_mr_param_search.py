from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from omegafx_v2.config import (
    DEFAULT_CHALLENGE,
    DEFAULT_COSTS,
    DEFAULT_SESSION,
    DEFAULT_STRATEGY,
    MeanReversionSignalConfig,
)
from omegafx_v2.data import fetch_xauusd_ohlc
from omegafx_v2.signals import build_mean_reversion_signals
from omegafx_v2.sim import run_randomized_signal_evaluations


def main() -> None:
    end = date.today()
    start = end - timedelta(days=365)

    ohlc = fetch_xauusd_ohlc(start.isoformat(), end.isoformat())
    print(f"Fetched {len(ohlc)} XAUUSD 1h bars from {start} to {end}")

    challenge = DEFAULT_CHALLENGE
    costs = DEFAULT_COSTS
    session = DEFAULT_SESSION

    strategy_cfg = replace(
        DEFAULT_STRATEGY,
        profit_target_pct=challenge.profit_target_pct,
    )

    initial_equity = 10_000.0
    num_evals = 100

    ma_periods = [20, 50, 100]
    entry_ks = [0.5, 1.0, 1.5]

    rows = []

    for ma_period in ma_periods:
        for entry_k in entry_ks:
            signal_cfg = MeanReversionSignalConfig(
                ma_period=ma_period,
                atr_period=14,
                entry_k=entry_k,
                exit_k=0.0,
                h4_sma_period=50,
            )

            signals = build_mean_reversion_signals(
                ohlc,
                signal_config=signal_cfg,
                session=session,
            )

            if signals.sum() == 0:
                continue

            batch = run_randomized_signal_evaluations(
                ohlc=ohlc,
                signals=signals,
                initial_equity=initial_equity,
                challenge=challenge,
                config=strategy_cfg,
                costs=costs,
                num_evals=num_evals,
                min_bars_per_eval=challenge.min_bars_per_eval,
                start_offset=5,
            )

            avg_trades = (
                sum(len(ev.trades) for ev in batch.evaluations) / batch.num_evals
                if batch.num_evals > 0
                else 0.0
            )

            rows.append(
                (
                    ma_period,
                    entry_k,
                    batch.pass_rate,
                    batch.average_return,
                    batch.average_max_drawdown,
                    avg_trades,
                )
            )

    rows.sort(key=lambda r: r[2], reverse=True)

    print("\nma  entry_k  pass_rate  avg_ret  avg_dd  avg_trades")
    print("---------------------------------------------------")
    for ma_period, entry_k, pass_rate, avg_ret, avg_dd, avg_trades in rows:
        print(
            f"{ma_period:3d}  {entry_k:7.2f}  "
            f"{pass_rate:9.3f}  {avg_ret:7.3f}  {avg_dd:7.3f}  {avg_trades:10.2f}"
        )


if __name__ == "__main__":
    main()
