import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from omegafx_v2.config import DEFAULT_PROFILE_USDJPY_MR_V1


def load_trades(path: Path, days: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No shadow trades file at {path}")
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    cutoff = pd.Timestamp.now(tz='UTC') - timedelta(days=days)
    return df[df["entry_time"] >= cutoff].sort_values("entry_time")


def equity_metrics(df: pd.DataFrame) -> tuple[float, float, float, float]:
    if df.empty:
        return 0.0, 0.0, 0.0, 0.0
    start_equity = df["equity_before"].iloc[0]
    end_equity = df["equity_after"].iloc[-1]
    if start_equity <= 0:
        return start_equity, end_equity, 0.0, 0.0
    equity_curve = df["equity_after"]
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min() if not drawdown.empty else 0.0
    total_return = end_equity / start_equity - 1.0
    return start_equity, end_equity, total_return, max_dd


def challenge_replay(df: pd.DataFrame, profit_target: float, max_loss: float) -> dict:
    if df.empty:
        return {"challenges": 0, "passes": 0, "fails": 0, "trades_per_challenge": 0}

    equity = df["equity_before"].iloc[0]
    start_equity = equity
    passes = fails = 0
    trades_in_challenge = []
    current_trades = 0

    for _, row in df.iterrows():
        equity += row["pnl"]
        current_trades += 1

        target_equity = start_equity * (1 + profit_target)
        loss_limit = start_equity * (1 - max_loss)

        if equity >= target_equity:
            passes += 1
            trades_in_challenge.append(current_trades)
            start_equity = equity
            current_trades = 0
        elif equity <= loss_limit:
            fails += 1
            trades_in_challenge.append(current_trades)
            start_equity = equity
            current_trades = 0

    challenges = passes + fails
    avg_trades = sum(trades_in_challenge) / len(trades_in_challenge) if trades_in_challenge else 0.0
    return {
        "challenges": challenges,
        "passes": passes,
        "fails": fails,
        "avg_trades": avg_trades,
    }


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    path = Path("logs/shadow_trades.csv")
    profile = DEFAULT_PROFILE_USDJPY_MR_V1

    try:
        df = load_trades(path, days)
    except FileNotFoundError as exc:
        print(exc)
        return

    if df.empty:
        print("No trades to evaluate.")
        return

    count = len(df)
    wins = (df["pnl"] > 0).sum()
    losses = (df["pnl"] < 0).sum()
    win_rate = wins / count if count else 0.0
    avg_pnl = df["pnl"].mean()
    avg_pnl_pct = df["pnl_pct"].mean()
    avg_win = df.loc[df["pnl"] > 0, "pnl"].mean() if wins else 0.0
    avg_loss = df.loc[df["pnl"] < 0, "pnl"].mean() if losses else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    start_eq, end_eq, total_return, max_dd = equity_metrics(df)

    challenge_stats = challenge_replay(
        df,
        profit_target=profile.challenge.profit_target_pct,
        max_loss=profile.challenge.max_total_loss_pct,
    )

    days_active = df["entry_time"].dt.date.nunique()
    trades_per_day = count / days_active if days_active else 0.0

    print(f"=== Online Shadow Evaluation (last {days} days) ===")
    print(f"Profile: {profile.name} ({df['symbol'].iloc[0]})")
    print(f"Trades: {count} (wins: {wins}, losses: {losses})")
    print(f"Win rate:         {win_rate:.2%}")
    print(f"Avg pnl:          {avg_pnl:.2f} ({avg_pnl_pct:.2%})")
    print(f"Expectancy/trade: {expectancy:.2f}")
    print(f"Max win: {df['pnl'].max():.2f}  Max loss: {df['pnl'].min():.2f}")

    print("\nEquity:")
    print(f"  Start equity:   {start_eq:,.2f}")
    print(f"  End equity:     {end_eq:,.2f}")
    print(f"  Total return:   {total_return:.2%}")
    print(f"  Max drawdown:   {max_dd:.2%}")

    print("\nShadow challenges:")
    print(f"  Challenges:     {challenge_stats['challenges']}")
    print(f"  Passes:         {challenge_stats['passes']}")
    print(f"  Fails:          {challenge_stats['fails']}")
    if challenge_stats["challenges"] > 0:
        pass_rate = challenge_stats["passes"] / challenge_stats["challenges"]
        print(f"  Pass rate:      {pass_rate:.2%}")
    print(f"  Avg trades/challenge: {challenge_stats['avg_trades']:.2f}")

    print("\nDaily behaviour:")
    print(f"  Days active:    {days_active}")
    print(f"  Trades per day: {trades_per_day:.2f}")


if __name__ == "__main__":
    main()
