from omegafx_v2.config import DEFAULT_PORTFOLIO_USDJPY_V2
from omegafx_v2.portfolio_summary import run_portfolio_summary


def main() -> None:
    portfolio = DEFAULT_PORTFOLIO_USDJPY_V2

    metrics = run_portfolio_summary(
        portfolio=portfolio,
        days=365,
        initial_equity_per_strategy=10_000.0,
        num_evals=100,
    )

    print(f"=== Portfolio Summary: {portfolio.name} ===")
    print("\nPer-strategy metrics:")
    print("Name                           Symbol  TF    PassRate   AvgRet   MaxDD   AvgTrades  Signals")
    for s in metrics["strategies"]:
        print(
            f"{s['name'][:28]:28s}  {s['symbol']:6s}  {s['timeframe']:4s}  "
            f"{s['pass_rate']:.2%}  {s['avg_return']:.2%}  {s['max_dd']:.2%}  "
            f"{s['avg_trades']:.2f}     {s['signals_total']}"
        )

    port = metrics["portfolio"]
    print("\nPortfolio aggregates:")
    print(f"  Total signals: {port['total_signals']}")
    print(f"  Total avg trades: {port['total_avg_trades']:.2f}")
    print(f"  Avg return (mean of strategies): {port['avg_return']:.2%}")
    print(f"  Worst max DD: {port['worst_dd']:.2%}")


if __name__ == "__main__":
    main()
