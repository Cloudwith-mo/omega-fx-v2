from __future__ import annotations

import pandas as pd
import yfinance as yf


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]]
    df = df.dropna()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def fetch_ohlc(symbol: str, start: str, end: str, interval: str = "1h") -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        progress=False,
    )
    return _normalize_df(df)


def fetch_symbol_ohlc(symbol_key: str, start: str, end: str, interval: str = "1h") -> pd.DataFrame:
    """
    Fetch OHLC data for a logical symbol key using yfinance tickers.
    """
    # yfinance intraday intervals are limited; cap range to 60 days for minute/hour intervals.
    if interval.lower().endswith("m"):
        end_dt = pd.to_datetime(end)
        start_dt = pd.to_datetime(start)
        max_span = pd.Timedelta(days=59)
        if end_dt - start_dt > max_span:
            start_dt = end_dt - max_span
            start = start_dt.date().isoformat()

    mapping = {
        "XAUUSD": ["XAUUSD=X", "XAU=X", "GC=F"],
        "EURUSD": ["EURUSD=X"],
        "GBPUSD": ["GBPUSD=X"],
        "USDJPY": ["USDJPY=X"],
    }
    tickers = mapping.get(symbol_key.upper(), [symbol_key])

    for ticker in tickers:
        df = fetch_ohlc(ticker, start=start, end=end, interval=interval)
        if not df.empty:
            return df

    raise RuntimeError(f"No data returned for {symbol_key} in given range")


def fetch_xauusd_ohlc(start: str, end: str, interval: str = "1h") -> pd.DataFrame:
    """
    Fetch OHLC data for gold (XAUUSD) using yfinance.

    Returns a DataFrame indexed by datetime with columns:
    ['open', 'high', 'low', 'close'].
    """
    return fetch_symbol_ohlc("XAUUSD", start, end, interval=interval)
