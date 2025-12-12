from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from omegafx_v2.config import DEFAULT_CHALLENGE, DEFAULT_COSTS, DEFAULT_SESSION, DEFAULT_STRATEGY
from omegafx_v2.data import fetch_xauusd_ohlc
from omegafx_v2.signals import (
    build_atr_filter,
    build_session_mask,
    compute_atr,
    compute_h4_sma_filter,
    generate_breakout_signals,
)
from omegafx_v2.sim import run_randomized_signal_evaluations


def main() -> None:
    end = date.today()
    start = end - timedelta(days=365)

    ohlc = fetch_xauusd_ohlc(start.isoformat(), end.isoformat())
    print(f"Fetched {len(ohlc)} XAUUSD 1h bars from {start} to {end}")

    session = DEFAULT_SESSION
    session_mask = build_session_mask(ohlc, session=session)

    raw_signals = generate_breakout_signals(ohlc, lookback=20)
    atr = compute_atr(ohlc, period=14)
    atr_mask = build_atr_filter(atr, percentile=50)
    h4_trend_mask = compute_h4_sma_filter(ohlc, sma_period=50)

    signals = raw_signals & session_mask & atr_mask & h4_trend_mask

    print(f"Generated {int(raw_signals.sum())} raw breakout signals over 1y")
    print(f"{int(raw_signals.sum() - signals.sum())} signals removed by filters")
    print(f"{int(signals.sum())} signals remain after session + ATR + trend filter ({session.name})")

    challenge = DEFAULT_CHALLENGE
    cfg = replace(
        DEFAULT_STRATEGY,
        profit_target_pct=challenge.profit_target_pct,
    )

    initial_equity = 10_000.0
    num_evals = 100

    batch = run_randomized_signal_evaluations(
        ohlc=ohlc,
        signals=signals,
        initial_equity=initial_equity,
        challenge=challenge,
        config=cfg,
        costs=DEFAULT_COSTS,
        num_evals=num_evals,
        min_bars_per_eval=challenge.min_bars_per_eval,
        start_offset=5,
        daily_loss_pct=challenge.daily_loss_pct,
    )

    print("\n--- Signal-driven batch evaluation summary ---")
    print(f"Challenge:        {challenge.name}")
    print(f"Profit target:    {challenge.profit_target_pct:.2%}")
    print(f"Max total loss:   {challenge.max_total_loss_pct:.2%}")
    print(f"Min bars per eval:{challenge.min_bars_per_eval}")
    print(f"Daily loss cap:   {challenge.daily_loss_pct:.2%}")
    print(
        f"Session:         {session.name} (weekdays={session.allowed_weekdays}, hours={session.start_hour}-{session.end_hour})"
    )

    print(f"\nEvals run:        {batch.num_evals}")
    print(f"Target hits:      {batch.target_hit_count}")
    print(f"Max loss hits:    {batch.max_loss_count}")
    print(f"Data exhausted:   {batch.data_exhausted_count}")
    print(f"Pass rate:        {batch.pass_rate:.2%}")
    print(f"Avg total return: {batch.average_return:.2%}")
    print(f"Avg max drawdown: {batch.average_max_drawdown:.2%}")

    print("\nCosts used:")
    print(f"  spread_pips: {DEFAULT_COSTS.spread_pips}")
    print(f"  commission_per_lot_round_trip: {DEFAULT_COSTS.commission_per_lot_round_trip}")


if __name__ == "__main__":
    main()
