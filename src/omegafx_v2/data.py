import pandas as pd
from pathlib import Path


def load_ohlc_csv(path: Path, timeframe: str, tz: str = "UTC") -> pd.DataFrame:
    """
    Load OHLCV from CSV with columns: time, open, high, low, close, volume.
    - Parses time to datetime (tz-aware).
    - Sets index to UTC timestamp.
    - Resamples to requested timeframe if needed (supports M1, M5, M15, H1).
    """
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError("CSV must contain a 'time' column.")
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)

    base = df[["open", "high", "low", "close", "volume"]].sort_index()

    def resample_tf(df_in, rule):
        o = df_in["open"].resample(rule).first()
        h = df_in["high"].resample(rule).max()
        l = df_in["low"].resample(rule).min()
        c = df_in["close"].resample(rule).last()
        v = df_in["volume"].resample(rule).sum()
        out = pd.concat([o, h, l, c, v], axis=1).dropna()
        out.columns = ["open", "high", "low", "close", "volume"]
        return out

    tf_map = {
        "M1": "1min",
        "M5": "5min",
        "M15": "15min",
        "H1": "1h",
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "60m": "1h",
    }
    rule = tf_map.get(timeframe, None)
    if rule is None:
        raise ValueError(f"Unsupported timeframe {timeframe}")

    # If base is already M1, resample up as needed
    return resample_tf(base, rule) if rule != "1min" else base


# Legacy stubs for compatibility with existing imports
def fetch_ohlc(*args, **kwargs):
    return None


def fetch_symbol_ohlc(*args, **kwargs):
    return None


def fetch_xauusd_ohlc(*args, **kwargs):
    return None
