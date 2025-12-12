import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def load_trades(path: Path, days: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No shadow trades file at {path}")
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    cutoff = datetime.now() - timedelta(days=days)
    return df[df["entry_time"] >= cutoff]


def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "pass_rate": 0.0,
            "avg_ret": 0.0,
            "max_dd": 0.0,
        }
    count = len(df)
    wins = (df["pnl"] > 0).sum()
    losses = (df["pnl"] < 0).sum()
    pass_rate = wins / count if count else 0.0
    avg_ret = df["pnl_pct"].mean()

    equity_curve = (1 + df["pnl_pct"]).cumprod()
    if equity_curve.empty:
        max_dd = 0.0
    else:
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_dd = drawdown.min()

    return {
        "count": count,
        "wins": wins,
        "losses": losses,
        "pass_rate": pass_rate,
        "avg_ret": avg_ret,
        "max_dd": max_dd,
    }


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    path = Path("logs/shadow_trades.csv")

    try:
        df = load_trades(path, days)
    except FileNotFoundError as exc:
        print(exc)
        return

    metrics = compute_metrics(df)

    print(f"=== Shadow Trades Evaluation (last {days} days) ===")
    print(f"Trades:   {metrics['count']}")
    print(f"Wins:     {metrics['wins']}  Losses: {metrics['losses']}")
    print(f"Pass rate: {metrics['pass_rate']:.2%}")
    print(f"Avg return: {metrics['avg_ret']:.2%}")
    print(f"Max drawdown: {metrics['max_dd']:.2%}")


if __name__ == "__main__":
    main()
