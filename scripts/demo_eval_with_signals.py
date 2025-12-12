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
    DEFAULT_SIGNAL_CONFIG,
    DEFAULT_SESSION,
    DEFAULT_STRATEGY,
)
from omegafx_v2.data import fetch_xauusd_ohlc
from omegafx_v2.signals import build_signals
from omegafx_v2.sim import run_signal_driven_evaluation


def main() -> None:
    end = date.today()
    start = end - timedelta(days=365)

    ohlc = fetch_xauusd_ohlc(start.isoformat(), end.isoformat())
    print(f"Fetched {len(ohlc)} XAUUSD 1h bars from {start} to {end}")

    signal_cfg = DEFAULT_SIGNAL_CONFIG
    session = DEFAULT_SESSION
    signals = build_signals(ohlc, signal_config=signal_cfg, session=session)

    print(f"Final signals after all filters: {int(signals.sum())}")
    print(f"Signal config: {signal_cfg}")

    challenge = DEFAULT_CHALLENGE

    cfg = replace(
        DEFAULT_STRATEGY,
        profit_target_pct=challenge.profit_target_pct,
    )

    initial_equity = 10_000.0

    result = run_signal_driven_evaluation(
        ohlc=ohlc,
        signals=signals,
        initial_equity=initial_equity,
        challenge=challenge,
        config=cfg,
        costs=DEFAULT_COSTS,
        daily_loss_pct=challenge.daily_loss_pct,
    )

    print("\n--- Signal-driven evaluation summary ---")
    print(f"Challenge:       {challenge.name}")
    print(f"Profit target:   {challenge.profit_target_pct:.2%}")
    print(f"Max total loss:  {challenge.max_total_loss_pct:.2%}")
    print(f"Daily loss cap:  {challenge.daily_loss_pct:.2%}")
    print(
        f"Session:         {session.name} (weekdays={session.allowed_weekdays}, hours={session.start_hour}-{session.end_hour})"
    )

    print(f"\nInitial equity:  {result.initial_equity:,.2f}")
    print(f"Final equity:    {result.final_equity:,.2f}")
    print(f"Total return:    {result.total_return:.2%}")
    print(f"Verdict:         {result.verdict}")
    print(f"Trades:          {result.num_trades}, wins: {result.num_wins}, losses: {result.num_losses}")
    print(f"Max drawdown:    {result.max_drawdown_pct:.2%}")


if __name__ == "__main__":
    main()
