from __future__ import annotations

import pandas as pd
import numpy as np

from .config import (
    DEFAULT_SESSION,
    DEFAULT_SIGNAL_CONFIG,
    DEFAULT_MR_SIGNAL_CONFIG,
    DEFAULT_TREND_SIGNAL_CONFIG,
    DEFAULT_LONDON_BREAKOUT_SIGNAL_CONFIG,
    DEFAULT_LIQUIDITY_SWEEP_SIGNAL_CONFIG,
    DEFAULT_MOMENTUM_PINBALL_CONFIG_M5,
    DEFAULT_TREND_KD_SIGNAL_CONFIG,
    MeanReversionSignalConfig,
    TrendContinuationSignalConfig,
    LondonBreakoutSignalConfig,
    LiquiditySweepSignalConfig,
    MomentumPinballSignalConfig,
    TrendKDSignalConfig,
    VanVleetSignalConfig,
    BigManSignalConfig,
    SignalConfig,
    TradingSession,
    get_pip_size,
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


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_london_breakout_signals(
    ohlc: pd.DataFrame,
    signal_config: LondonBreakoutSignalConfig = DEFAULT_LONDON_BREAKOUT_SIGNAL_CONFIG,
    session: TradingSession = DEFAULT_SESSION,
    symbol: str | None = None,
) -> pd.Series:
    """
    Simple London breakout: breakout above prior pre-session high with ATR + session filters.
    """
    if not isinstance(ohlc.index, pd.DatetimeIndex):
        raise ValueError("ohlc index must be DatetimeIndex")

    idx = ohlc.index
    hours = idx.hour
    dates = idx.date

    pre_mask = (hours >= signal_config.pre_session_start_hour) & (hours < signal_config.pre_session_end_hour)
    pre_session_high = ohlc["high"].where(pre_mask).groupby(dates).transform("max").shift(1)

    pip_size = get_pip_size(symbol or "") if symbol else 0.01
    buffer = signal_config.breakout_buffer_pips * pip_size
    breakout = ohlc["close"] > (pre_session_high + buffer)

    atr = compute_atr(ohlc, period=signal_config.atr_period)
    atr_mask = build_atr_filter(atr, percentile=signal_config.atr_filter_percentile)
    atr_rising = atr.diff() > 0
    atr_vol_ok = atr_rising | build_atr_filter(atr, percentile=signal_config.atr_min_percentile)
    session_mask = build_session_mask(ohlc, session=session)

    # Higher timeframe trend filter (default H1 EMA slope)
    trend_ok = session_mask
    if signal_config.trend_tf.lower() == "h1":
        htf = (
            ohlc.resample("1h")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
        )
        ema = htf["close"].ewm(span=50, adjust=False).mean()
        slope = ema.diff()
        trend_ok = (slope > signal_config.trend_strength_min).reindex(ohlc.index, method="ffill").fillna(False)

    signals = breakout & atr_mask & atr_vol_ok & session_mask & trend_ok
    return signals.fillna(False)


def build_liquidity_sweep_signals(
    ohlc: pd.DataFrame,
    signal_config: LiquiditySweepSignalConfig = DEFAULT_LIQUIDITY_SWEEP_SIGNAL_CONFIG,
    session: TradingSession = DEFAULT_SESSION,
    symbol: str | None = None,
) -> pd.Series:
    """
    Basic liquidity sweep: look for sweeps of recent lows that fail and close back above.
    Long-only for simplicity.
    """
    if not isinstance(ohlc.index, pd.DatetimeIndex):
        raise ValueError("ohlc index must be DatetimeIndex")

    pip_size = get_pip_size(symbol or "") if symbol else 0.01
    threshold = signal_config.sweep_threshold_pips * pip_size

    recent_low = ohlc["low"].rolling(signal_config.lookback_levels).min().shift(1)
    sweep = (ohlc["low"] < recent_low - threshold) & (ohlc["close"] > recent_low)

    atr = compute_atr(ohlc, period=signal_config.atr_period)
    atr_mask = build_atr_filter(atr, percentile=signal_config.atr_filter_percentile)
    session_mask = build_session_mask(ohlc, session=session)

    trend_ok = session_mask
    if signal_config.trend_tf.lower() == "h4":
        htf = (
            ohlc.resample("4h")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
        )
        ema = htf["close"].ewm(span=50, adjust=False).mean()
        slope = ema.diff()
        trend_ok = (slope > signal_config.trend_strength_min).reindex(ohlc.index, method="ffill").fillna(False)

    signals = sweep & atr_mask & session_mask & trend_ok
    return signals.fillna(False)


def build_momentum_pinball_signals_m5(
    ohlc_m5: pd.DataFrame,
    ohlc_m15: pd.DataFrame,
    signal_config: MomentumPinballSignalConfig = DEFAULT_MOMENTUM_PINBALL_CONFIG_M5,
    session: TradingSession = DEFAULT_SESSION,
) -> pd.Series:
    """
    M5 momentum pinball: long-only RSI(2) dip buys in an M15 uptrend.
    """
    if not isinstance(ohlc_m5.index, pd.DatetimeIndex) or not isinstance(ohlc_m15.index, pd.DatetimeIndex):
        raise ValueError("ohlc indexes must be DatetimeIndex")

    close_m5 = ohlc_m5["close"]
    rsi = compute_rsi(close_m5, period=signal_config.rsi_period)

    close_m15 = ohlc_m15["close"]
    ema_m15 = close_m15.ewm(span=signal_config.trend_ma_period, adjust=False).mean()
    ema_slope = ema_m15.diff()
    uptrend_m15 = ema_slope > 0
    uptrend_on_m5 = uptrend_m15.reindex(ohlc_m5.index, method="ffill").fillna(False)

    # H1 trend alignment
    h1 = (
        ohlc_m15.resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    h1_ema = h1["close"].ewm(span=50, adjust=False).mean()
    h1_slope = h1_ema.diff()
    h1_up = (h1_slope > 0).reindex(ohlc_m5.index, method="ffill").fillna(False)

    atr_m5 = compute_atr(ohlc_m5, period=signal_config.atr_period)
    atr_mask = build_atr_filter(atr_m5, percentile=signal_config.atr_min_percentile)

    # NY session only for momentum (configurable)
    hours = ohlc_m5.index.hour
    ny_mask = (hours >= signal_config.ny_session_start_hour) & (hours < signal_config.ny_session_end_hour)

    # Also apply provided session mask to keep weekday filters
    session_mask = build_session_mask(ohlc_m5, session=session) & ny_mask

    long_signals = (uptrend_on_m5) & h1_up & (rsi < signal_config.rsi_oversold) & atr_mask & session_mask
    long_signals = long_signals.fillna(False)

    # Limit signals to max trades per day
    if signal_config.max_trades_per_day is not None and signal_config.max_trades_per_day > 0:
        dates = ohlc_m5.index.date
        counts = {}
        limited = []
        for flag, d in zip(long_signals, dates):
            if not flag:
                limited.append(False)
                continue
            c = counts.get(d, 0)
            if c < signal_config.max_trades_per_day:
                limited.append(True)
                counts[d] = c + 1
            else:
                limited.append(False)
        long_signals = pd.Series(limited, index=ohlc_m5.index)

    return long_signals


def build_trend_kd_signals_m15(
    ohlc_m15: pd.DataFrame,
    signal_config: "TrendKDSignalConfig",
    session: TradingSession = DEFAULT_SESSION,
) -> pd.Series:
    """
    Experimental KD trend-continuation in high-volatility conditions.
    Long-only:
      - ATR percentile >= atr_percentile
      - Price above M15 EMA
      - H1 EMA slope positive and above threshold
      - Session filter
    """
    if not isinstance(ohlc_m15.index, pd.DatetimeIndex):
        raise ValueError("ohlc index must be DatetimeIndex")

    atr = compute_atr(ohlc_m15, period=14)
    atr_mask = build_atr_filter(atr, percentile=signal_config.atr_percentile)

    ema_m15 = ohlc_m15["close"].ewm(span=signal_config.m15_ema_period, adjust=False).mean()
    above_m15 = ohlc_m15["close"] > ema_m15

    h1 = (
        ohlc_m15.resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    h1_ema = h1["close"].ewm(span=signal_config.h1_ema_period, adjust=False).mean()
    h1_slope = h1_ema.diff()
    strong_h1 = (h1_slope > signal_config.h1_slope_threshold).reindex(ohlc_m15.index, method="ffill").fillna(False)

    session_mask = build_session_mask(ohlc_m15, session=session)

    raw = atr_mask & above_m15 & strong_h1 & session_mask
    # limit per day
    if signal_config.max_trades_per_day and signal_config.max_trades_per_day > 0:
        dates = ohlc_m15.index.date
        counts = {}
        limited = []
        for flag, d in zip(raw, dates):
            if not flag:
                limited.append(False)
                continue
            c = counts.get(d, 0)
            if c < signal_config.max_trades_per_day:
                limited.append(True)
                counts[d] = c + 1
            else:
                limited.append(False)
        return pd.Series(limited, index=ohlc_m15.index)
    return raw.fillna(False)


def build_vanvleet_signals_m15(
    ohlc: pd.DataFrame,
    signal_config: VanVleetSignalConfig,
    session: TradingSession = DEFAULT_SESSION,
) -> pd.Series:
    """
    Experimental trend pullback (VanVleet) on M15.
    Long-only pullback to EMA with ATR + RSI filter.
    """
    close = ohlc["close"]
    ema_fast = close.ewm(span=signal_config.fast_ma_period, adjust=False).mean()
    ema_slow = close.ewm(span=signal_config.slow_ma_period, adjust=False).mean()
    ema_pull = close.ewm(span=signal_config.pullback_ma_period, adjust=False).mean()
    uptrend = ema_fast > ema_slow

    rsi = compute_rsi(close, period=signal_config.rsi_period)
    atr = compute_atr(ohlc, period=signal_config.atr_period)
    atr_mask = build_atr_filter(atr, percentile=signal_config.atr_percentile)
    session_mask = build_session_mask(ohlc, session=session)

    prev_close = close.shift(1)
    prev_pull = ema_pull.shift(1)
    pullback = (prev_close < prev_pull) & (rsi < signal_config.rsi_pullback)
    cross_back = (close > ema_pull) & pullback

    sig = uptrend & cross_back & atr_mask & session_mask
    sig = sig.fillna(False)

    # Optional daily cap
    max_trades = getattr(signal_config, "max_trades_per_day", None)
    if max_trades and max_trades > 0:
        dates = ohlc.index.date
        counts: dict = {}
        limited = []
        for flag, d in zip(sig, dates):
            if not flag:
                limited.append(False)
                continue
            c = counts.get(d, 0)
            if c < max_trades:
                limited.append(True)
                counts[d] = c + 1
            else:
                limited.append(False)
        sig = pd.Series(limited, index=ohlc.index)

    return sig


def build_bigman_signals_h1(
    ohlc: pd.DataFrame,
    signal_config: BigManSignalConfig,
    session: TradingSession = DEFAULT_SESSION,
) -> pd.Series:
    """
    Experimental range-reversion (BigMan) on H1:
      - Low trend strength (ADX < threshold)
      - Bands around SMA using ATR*k
      - Long near lower band, short near upper band
    """
    close = ohlc["close"]
    sma = close.rolling(signal_config.sma_period).mean()
    atr = compute_atr(ohlc, period=signal_config.atr_period)
    atr_mask = build_atr_filter(atr, percentile=signal_config.atr_percentile)
    session_mask = build_session_mask(ohlc, session=session)

    # Simple ADX approximation
    high = ohlc["high"]
    low = ohlc["low"]
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=ohlc.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=ohlc.index)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_adx = tr.rolling(signal_config.adx_period).mean()
    plus_di = plus_dm.rolling(signal_config.adx_period).sum() / atr_adx.replace(0, pd.NA) * 100
    minus_di = minus_dm.rolling(signal_config.adx_period).sum() / atr_adx.replace(0, pd.NA) * 100
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, pd.NA)) * 100
    adx = dx.rolling(signal_config.adx_period).mean()
    low_trend = adx < signal_config.adx_max

    upper = sma + atr * signal_config.band_k
    lower = sma - atr * signal_config.band_k

    long_sig = (close <= lower) & low_trend & atr_mask & session_mask
    short_sig = (close >= upper) & low_trend & atr_mask & session_mask
    sig = long_sig | short_sig
    return sig.fillna(False)
