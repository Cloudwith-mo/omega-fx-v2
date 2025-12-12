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
from omegafx_v2.data import fetch_symbol_ohlc
from omegafx_v2.signals import build_mean_reversion_signals
from omegafx_v2.sim import run_randomized_signal_evaluations


def main() -> None:
    end = date.today()
    start = end - timedelta(days=365)

    symbols = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]

    challenge = DEFAULT_CHALLENGE
    costs = DEFAULT_COSTS
    session = DEFAULT_SESSION
    strategy_cfg = replace(
        DEFAULT_STRATEGY,
        profit_target_pct=challenge.profit_target_pct,
    )

    initial_equity = 10_000.0
    num_evals = 50  # smaller for multi-symbol speed

    ma_periods = [30, 50]
    entry_ks = [1.0, 1.5]

    print("=== Mean Reversion Multi-Symbol Scan (1y, H1) ===\n")

    results = []

    for symbol in symbols:
        try:
            ohlc = fetch_symbol_ohlc(symbol, start=start.isoformat(), end=end.isoformat())
        except RuntimeError as exc:
            print(f"{symbol}: data fetch failed ({exc})")
            continue

        best_row = None

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

                row = (
                    symbol,
                    ma_period,
                    entry_k,
                    batch.pass_rate,
                    batch.average_return,
                    batch.average_max_drawdown,
                    avg_trades,
                )

                if best_row is None or row[3] > best_row[3] or (
                    row[3] == best_row[3] and row[4] > best_row[4]
                ):
                    best_row = row

        if best_row:
            results.append(best_row)

    results.sort(key=lambda r: r[3], reverse=True)

    print("Symbol   ma   entry_k   pass_rate  avg_ret  avg_dd  avg_trades")
    print("---------------------------------------------------------------")
    for symbol, ma_period, entry_k, pass_rate, avg_ret, avg_dd, avg_trades in results:
        print(
            f"{symbol:6s}  {ma_period:3d}  {entry_k:7.2f}  "
            f"{pass_rate:9.3f}  {avg_ret:7.3f}  {avg_dd:7.3f}  {avg_trades:10.2f}"
        )


if __name__ == "__main__":
    main()
