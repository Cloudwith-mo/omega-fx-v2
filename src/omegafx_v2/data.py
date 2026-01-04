import pandas as pd
from pathlib import Path


def load_ohlc_csv(path: Path, timeframe: str, tz: str = "UTC") -> pd.DataFrame:
    """
    Load OHLCV from CSV with columns: time, open, high, low, close, volume.
    Supports MT5 export format with <DATE> and <TIME> columns (tab-delimited).
    - Parses time to datetime (tz-aware).
    - Sets index to UTC timestamp.
    - Resamples to requested timeframe if needed (supports M1, M5, M15, H1, H4, D1).
    """
    sep = ","
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline()
        if "\t" in header:
            sep = "\t"
        elif ";" in header and "," not in header:
            sep = ";"
    except OSError:
        pass
    df = pd.read_csv(path, sep=sep)
    # normalize column names
    def _norm(col: str) -> str:
        return col.strip().strip("<>").lower()
    df.columns = [_norm(c) for c in df.columns]
    if "time" not in df.columns:
        if "timestamp" in df.columns:
            df["time"] = df["timestamp"]
        elif "date" in df.columns and "time" in df.columns:
            df["time"] = df["date"].astype(str) + " " + df["time"].astype(str)
        elif "date" in df.columns and "time" not in df.columns:
            df["time"] = df["date"]
    if "time" not in df.columns:
        raise ValueError("CSV must contain a 'time' column.")
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)
    volume_col = None
    for cand in ("volume", "tickvol", "vol"):
        if cand in df.columns:
            volume_col = cand
            break
    if volume_col is None:
        df["volume"] = 0.0
        volume_col = "volume"
    base = df[["open", "high", "low", "close", volume_col]].sort_index()
    base.columns = ["open", "high", "low", "close", "volume"]

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
        "H4": "4h",
        "D1": "1d",
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "60m": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    rule = tf_map.get(timeframe, None)
    if rule is None:
        raise ValueError(f"Unsupported timeframe {timeframe}")

    if rule != "1min":
        try:
            diffs = base.index.to_series().diff().dropna()
            if not diffs.empty:
                target = pd.to_timedelta(rule)
                median_delta = diffs.median()
                if abs(median_delta - target) <= pd.Timedelta(seconds=1):
                    return base
        except Exception:
            pass
        return resample_tf(base, rule)
    return base


# Legacy stubs for compatibility with existing imports
def fetch_ohlc(*args, **kwargs):
    return None


def fetch_symbol_ohlc(*args, **kwargs):
    return None


def fetch_xauusd_ohlc(*args, **kwargs):
    return None
