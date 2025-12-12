from __future__ import annotations

import pandas as pd

from .config import DEFAULT_SESSION, TradingSession


def generate_breakout_signals(
    ohlc: pd.DataFrame,
    lookback: int = 20,
) -> pd.Series:
    """
    Simple long-only breakout signal:
    - Signal True when the close exceeds the prior `lookback`-bar high.
    """
    if not {"high", "close"}.issubset(ohlc.columns):
        raise ValueError("ohlc must contain 'high' and 'close' columns")

    highs = ohlc["high"]
    closes = ohlc["close"]
    previous_high = highs.rolling(lookback, min_periods=lookback).max().shift(1)

    signals = closes > previous_high
    return signals.fillna(False)


def build_session_mask(
    ohlc: pd.DataFrame,
    session: TradingSession = DEFAULT_SESSION,
) -> pd.Series:
    """
    Boolean mask aligned with ohlc.index for allowed weekdays and hours.
    """
    if not isinstance(ohlc.index, pd.DatetimeIndex):
        raise ValueError("ohlc index must be a DatetimeIndex")

    idx = ohlc.index
    weekdays = idx.weekday
    hours = idx.hour

    allowed_days = pd.Series(weekdays).isin(session.allowed_weekdays).values
    in_hours = (hours >= session.start_hour) & (hours < session.end_hour)

    mask = allowed_days & in_hours
    return pd.Series(mask, index=ohlc.index)


def compute_atr(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR) over a given period.
    """
    if not {"high", "low", "close"}.issubset(ohlc.columns):
        raise ValueError("ohlc must contain 'high', 'low', and 'close' columns")

    high = ohlc["high"]
    low = ohlc["low"]
    close_prev = ohlc["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


def build_atr_filter(
    atr: pd.Series,
    percentile: float = 50,
) -> pd.Series:
    """
    Boolean mask where ATR is above a given percentile (e.g., top half of vols).
    """
    if atr.empty:
        return pd.Series([], dtype=bool)
    threshold = atr.quantile(percentile / 100)
    return atr > threshold
