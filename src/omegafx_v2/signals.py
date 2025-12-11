from __future__ import annotations

import pandas as pd


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
