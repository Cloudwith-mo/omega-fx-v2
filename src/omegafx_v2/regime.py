import pandas as pd


def tag_regimes(
    ohlc_m15: pd.DataFrame,
    atr: pd.Series,
    h1_slope: pd.Series,
    low_vol_pct: float = 0.4,
    high_vol_pct: float = 0.7,
    trend_slope_threshold: float = 0.0001,
) -> pd.Series:
    """
    Tag each M15 bar into regimes:
      - "low_vol_range": ATR percentile <= low_vol_pct
      - "high_vol_trend": ATR percentile >= high_vol_pct and |H1 slope| > threshold
      - "high_vol_reversal": ATR percentile >= high_vol_pct and |H1 slope| <= threshold
      - "chop": otherwise
    """
    atr_rank = atr.rank(pct=True)
    h1_slope_on_m15 = h1_slope.reindex(ohlc_m15.index, method="ffill").fillna(0)

    regimes = []
    for pct, slope in zip(atr_rank, h1_slope_on_m15):
        if pct <= low_vol_pct:
            regimes.append("low_vol_range")
        elif pct >= high_vol_pct and abs(slope) > trend_slope_threshold:
            regimes.append("high_vol_trend")
        elif pct >= high_vol_pct and abs(slope) <= trend_slope_threshold:
            regimes.append("high_vol_reversal")
        else:
            regimes.append("chop")
    return pd.Series(regimes, index=ohlc_m15.index, name="regime")
