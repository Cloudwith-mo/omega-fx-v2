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
