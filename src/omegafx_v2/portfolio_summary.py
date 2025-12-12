from __future__ import annotations

from typing import Dict, List

from .config import PortfolioProfile
from .profile_summary import run_profile_summary


def run_portfolio_summary(
    portfolio: PortfolioProfile,
    days: int = 365,
    initial_equity_per_strategy: float = 10_000.0,
    num_evals: int = 100,
) -> Dict:
    """
    Run per-strategy summaries for each profile in the portfolio and aggregate.
    """
    strategy_results: List[Dict] = []
    total_signals = 0
    total_trades = 0.0
    returns = []
    dds = []

    for profile in portfolio.strategies:
        res = run_profile_summary(
            profile=profile,
            days=days,
            initial_equity=initial_equity_per_strategy,
            num_evals=num_evals,
        )
        strategy_results.append(
            {
                "name": profile.name,
                "symbol": profile.symbol_key,
                "timeframe": profile.timeframe,
                "pass_rate": res["batch_pass_rate"],
                "avg_return": res["batch_avg_return"],
                "max_dd": res["batch_avg_max_dd"],
                "avg_trades": res["batch_avg_trades"],
                "signals_total": res["signals_total"],
            }
        )
        total_signals += res["signals_total"]
        total_trades += res["batch_avg_trades"]
        returns.append(res["batch_avg_return"])
        dds.append(res["batch_avg_max_dd"])

    avg_return = sum(returns) / len(returns) if returns else 0.0
    worst_dd = max(dds) if dds else 0.0

    return {
        "portfolio": {
            "name": portfolio.name,
            "total_signals": total_signals,
            "total_avg_trades": total_trades,
            "avg_return": avg_return,
            "worst_dd": worst_dd,
        },
        "strategies": strategy_results,
    }
