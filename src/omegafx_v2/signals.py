from __future__ import annotations

import pandas as pd

from .config import (
    DEFAULT_SESSION,
    DEFAULT_SIGNAL_CONFIG,
    DEFAULT_MR_SIGNAL_CONFIG,
    DEFAULT_TREND_SIGNAL_CONFIG,
    MeanReversionSignalConfig,
    TrendContinuationSignalConfig,
    SignalConfig,
    TradingSession,
)


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


def compute_h4_sma_filter(
    ohlc: pd.DataFrame,
    sma_period: int = 50,
) -> pd.Series:
    """
    Resample to 4H bars, compute SMA, and map an uptrend mask back to 1H index.
    """
    if not isinstance(ohlc.index, pd.DatetimeIndex):
        raise ValueError("OHLC index must be DatetimeIndex")

    h4 = (
        ohlc.resample("4H")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )

    sma = h4["close"].rolling(sma_period).mean()
    trend_up = h4["close"] > sma

    trend_up_1h = trend_up.reindex(ohlc.index, method="ffill")
    return trend_up_1h.fillna(False)


def build_signals(
    ohlc: pd.DataFrame,
    signal_config: SignalConfig = DEFAULT_SIGNAL_CONFIG,
    session: TradingSession = DEFAULT_SESSION,
) -> pd.Series:
    """
    Combined signal pipeline: breakout + session + ATR + trend filters.
    """
    session_mask = build_session_mask(ohlc, session=session)

    raw_signals = generate_breakout_signals(
        ohlc, lookback=signal_config.breakout_lookback
    )

    atr = compute_atr(ohlc, period=signal_config.atr_period)
    atr_mask = build_atr_filter(atr, percentile=signal_config.atr_percentile)

    h4_trend_mask = compute_h4_sma_filter(
        ohlc, sma_period=signal_config.h4_sma_period
    )

    signals = raw_signals & session_mask & atr_mask & h4_trend_mask
    return signals


def build_mean_reversion_signals(
    ohlc: pd.DataFrame,
    signal_config: MeanReversionSignalConfig = DEFAULT_MR_SIGNAL_CONFIG,
    session: TradingSession = DEFAULT_SESSION,
) -> pd.Series:
    """
    Mean-reversion long-only signals using MA/ATR pullback plus session and trend filters.
    """
    if not isinstance(ohlc.index, pd.DatetimeIndex):
        raise ValueError("ohlc index must be DatetimeIndex")

    close = ohlc["close"]

    ma = close.rolling(signal_config.ma_period).mean()
    atr = compute_atr(ohlc, period=signal_config.atr_period)
    atr_safe = atr.replace(0, pd.NA)
    z = (ma - close) / atr_safe

    raw_entry = z >= signal_config.entry_k

    if signal_config.exit_k is not None:
        near_mean = z <= signal_config.exit_k
        raw_entry = raw_entry & ~near_mean

    session_mask = build_session_mask(ohlc, session=session)
    h4_up = compute_h4_sma_filter(ohlc, sma_period=signal_config.h4_sma_period)

    signals = raw_entry & session_mask & h4_up
    return signals.fillna(False)


def build_trend_signals(
    ohlc: pd.DataFrame,
    signal_config: TrendContinuationSignalConfig = DEFAULT_TREND_SIGNAL_CONFIG,
    session: TradingSession = DEFAULT_SESSION,
) -> pd.Series:
    """
    Long-only trend-continuation signals on M15 (or given timeframe):
    - Uptrend: fast MA > slow MA
    - Entry: close crosses above fast MA after being at/under it, with ATR + H4 trend + session filters.
    """
    if not isinstance(ohlc.index, pd.DatetimeIndex):
        raise ValueError("ohlc index must be DatetimeIndex")

    close = ohlc["close"]

    ma_fast = close.rolling(signal_config.fast_ma_period).mean()
    ma_slow = close.rolling(signal_config.slow_ma_period).mean()
    uptrend = ma_fast > ma_slow

    prev_close = close.shift(1)
    prev_ma_fast = ma_fast.shift(1)
    crossed_up = (close > ma_fast) & (prev_close <= prev_ma_fast)

    atr = compute_atr(ohlc, period=signal_config.atr_period)
    atr_mask = build_atr_filter(atr, percentile=signal_config.atr_percentile)

    h4_up = compute_h4_sma_filter(
        ohlc,
        sma_period=signal_config.h4_sma_period,
    )

    session_mask = build_session_mask(ohlc, session=session)

    signals = crossed_up & uptrend & atr_mask & h4_up & session_mask
    return signals.fillna(False)
