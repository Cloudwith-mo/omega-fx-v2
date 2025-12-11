from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_xauusd_ohlc(start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLC data for gold (XAUUSD) using yfinance.

    Returns a DataFrame indexed by datetime with columns:
    ['open', 'high', 'low', 'close'].
    """
    tickers = ["XAUUSD=X", "XAU=X", "GC=F"]  # try spot, fallback to gold futures
    df = pd.DataFrame()

    for ticker in tickers:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1h",
            progress=False,
        )
        if not df.empty:
            break

    if df.empty:
        raise RuntimeError("No data returned for XAUUSD in given range")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]]
    df = df.dropna()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df
