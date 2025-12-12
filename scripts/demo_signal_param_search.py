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
    DEFAULT_SIGNAL_CONFIG,
    SignalConfig,
)
from omegafx_v2.data import fetch_xauusd_ohlc
from omegafx_v2.signals import build_signals
from omegafx_v2.sim import run_randomized_signal_evaluations


def main() -> None:
    end = date.today()
    start = end - timedelta(days=365)

    ohlc = fetch_xauusd_ohlc(start.isoformat(), end.isoformat())
    print(f"Fetched {len(ohlc)} XAUUSD 1h bars from {start} to {end}")

    session = DEFAULT_SESSION

    challenge = DEFAULT_CHALLENGE
    cfg_base = replace(
        DEFAULT_STRATEGY,
        profit_target_pct=challenge.profit_target_pct,
    )

    initial_equity = 10_000.0
    num_evals = 100

    breakout_lookbacks = [10, 20, 30]
    atr_percentiles = [40, 50, 60, 70]

    rows = []

    for lookback in breakout_lookbacks:
        for perc in atr_percentiles:
            signal_cfg = SignalConfig(
                breakout_lookback=lookback,
                atr_period=DEFAULT_SIGNAL_CONFIG.atr_period,
                atr_percentile=perc,
                h4_sma_period=DEFAULT_SIGNAL_CONFIG.h4_sma_period,
            )

            signals = build_signals(
                ohlc, signal_config=signal_cfg, session=session
            )

            if signals.sum() == 0:
                continue

            batch = run_randomized_signal_evaluations(
                ohlc=ohlc,
                signals=signals,
                initial_equity=initial_equity,
                challenge=challenge,
                config=cfg_base,
                costs=DEFAULT_COSTS,
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
                    signal_cfg,
                    batch.pass_rate,
                    batch.average_return,
                    batch.average_max_drawdown,
                    avg_trades,
                )
            )

    rows.sort(key=lambda r: r[1], reverse=True)

    print("\nsignal_cfg                                pass_rate  avg_ret  avg_dd  avg_trades")
    print("-------------------------------------------------------------------------------")
    for signal_cfg, pass_rate, avg_ret, avg_dd, avg_trades in rows:
        print(
            f"{str(signal_cfg):35s}  "
            f"{pass_rate:9.3f}  {avg_ret:7.3f}  {avg_dd:7.3f}  {avg_trades:10.2f}"
        )


if __name__ == "__main__":
    main()
