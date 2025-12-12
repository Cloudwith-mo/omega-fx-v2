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

    challenge = DEFAULT_CHALLENGE
    session = DEFAULT_SESSION

    # Candidate signal config (tweak as desired)
    signal_cfg = SignalConfig(
        breakout_lookback=30,
        atr_period=14,
        atr_percentile=40,
        h4_sma_period=50,
    )

    signals = build_signals(ohlc, signal_config=signal_cfg, session=session)
    if signals.sum() == 0:
        print("No signals generated with current config; aborting sweep.")
        return

    print(f"Signal config: {signal_cfg}")
    print(f"Total signals after filters: {int(signals.sum())}")

    risk_values = [0.005, 0.01, 0.015, 0.02]
    reward_multiples = [1.0, 1.5, 2.0]

    initial_equity = 10_000.0
    num_evals = 100

    rows = []

    for risk_pct in risk_values:
        for reward_mult in reward_multiples:
            reward_pct = risk_pct * reward_mult
            strategy_cfg = replace(
                DEFAULT_STRATEGY,
                risk_per_trade_pct=risk_pct,
                reward_per_trade_pct=reward_pct,
                profit_target_pct=challenge.profit_target_pct,
            )

            batch = run_randomized_signal_evaluations(
                ohlc=ohlc,
                signals=signals,
                initial_equity=initial_equity,
                challenge=challenge,
                config=strategy_cfg,
                costs=DEFAULT_COSTS,
                num_evals=num_evals,
                min_bars_per_eval=challenge.min_bars_per_eval,
                start_offset=5,
                daily_loss_pct=challenge.daily_loss_pct,
            )

            avg_trades = (
                sum(len(ev.trades) for ev in batch.evaluations) / batch.num_evals
                if batch.num_evals > 0
                else 0.0
            )

            rows.append(
                (
                    risk_pct,
                    reward_pct,
                    reward_mult,
                    batch.pass_rate,
                    batch.average_return,
                    batch.average_max_drawdown,
                    avg_trades,
                )
            )

    rows.sort(key=lambda r: r[3], reverse=True)

    print("\nrisk_pct  reward_pct  R_mult  pass_rate  avg_ret  avg_dd  avg_trades")
    print("--------------------------------------------------------------------")
    for risk_pct, reward_pct, reward_mult, pass_rate, avg_ret, avg_dd, avg_trades in rows:
        print(
            f"{risk_pct:7.3f}  {reward_pct:10.3f}  {reward_mult:6.2f}  "
            f"{pass_rate:9.3f}  {avg_ret:7.3f}  {avg_dd:7.3f}  {avg_trades:10.2f}"
        )


if __name__ == "__main__":
    main()
