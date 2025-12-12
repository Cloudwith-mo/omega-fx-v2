from datetime import date, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from omegafx_v2.config import DEFAULT_PROFILE_USDJPY_MR_M15_V1
from omegafx_v2.data import fetch_symbol_ohlc
from omegafx_v2.runtime_loop import DummyBrokerAdapter, run_live_runtime


def bar_stream_from_df(df):
    for ts, row in df.iterrows():
        yield ts, row


def main() -> None:
    profile = DEFAULT_PROFILE_USDJPY_MR_M15_V1
    end = date.today()
    start = end - timedelta(days=365)

    df = fetch_symbol_ohlc(
        profile.symbol_key,
        start=start.isoformat(),
        end=end.isoformat(),
        interval=profile.timeframe,
    )

    broker = DummyBrokerAdapter()
    state = run_live_runtime(
        profile=profile,
        bar_stream=bar_stream_from_df(df),
        broker=broker,
        initial_equity=10_000.0,
    )

    if state.last_bar_time:
        print(f"Processed bars up to: {state.last_bar_time}")
    print(f"Trades executed: {state.trades_executed}")
    print(f"Final equity: {state.equity:,.2f}")
    if state.equity > 0:
        print(f"Total return: {state.equity / 10_000.0 - 1:.2%}")


if __name__ == "__main__":
    main()
