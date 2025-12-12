import os
from datetime import datetime, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from omegafx_v2.config import DEFAULT_PROFILE_USDJPY_MR_V1
from omegafx_v2.data import fetch_symbol_ohlc
from omegafx_v2.mt5_adapter import Mt5BrokerAdapter, Mt5ConnectionConfig
from omegafx_v2.runtime_loop import run_live_runtime


def bar_stream_from_df(df):
    for ts, row in df.iterrows():
        yield ts, row


def main() -> None:
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if not all([login, password, server]):
        print("MT5 credentials not set in environment (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER). Exiting.")
        return

    profile = DEFAULT_PROFILE_USDJPY_MR_V1

    # For now, replay recent historical bars from MT5 as the bar stream.
    end = datetime.now()
    start = end - timedelta(days=365)

    try:
        conn = Mt5ConnectionConfig(login=int(login), password=password, server=server)
        broker = Mt5BrokerAdapter(conn=conn, symbol=profile.symbol_key)
    except Exception as exc:
        print(f"Failed to initialize MT5 broker: {exc}")
        return

    # Pull historical bars from MT5
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 package not installed; cannot fetch MT5 historical data.")
        return

    rates = mt5.copy_rates_range(
        profile.symbol_key,
        mt5.TIMEFRAME_H1,
        start,
        end,
    )
    if rates is None or len(rates) == 0:
        print("No rates returned from MT5; aborting.")
        return

    import pandas as pd

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")[["open", "high", "low", "close"]]

    state = run_live_runtime(
        profile=profile,
        bar_stream=bar_stream_from_df(df),
        broker=broker,
        initial_equity=10_000.0,
    )

    print(f"Completed live runtime sim. Final equity: {state.equity:,.2f}, trades: {state.trades_executed}")


if __name__ == "__main__":
    main()
