from omegafx_v2.config import DEFAULT_PROFILE_USDJPY_MR_V1
from omegafx_v2.runtime_live import run_offline_live_simulation


def main() -> None:
    profile = DEFAULT_PROFILE_USDJPY_MR_V1

    result = run_offline_live_simulation(
        profile=profile,
        days=365,
        initial_equity=10_000.0,
    )

    print(f"=== Offline Live Sim – {result.profile_name} ({result.symbol_key}) ===")
    print(f"Initial equity: {result.initial_equity:,.2f}")
    print(f"Final equity:   {result.final_equity:,.2f}")
    if result.initial_equity > 0:
        total_ret = result.final_equity / result.initial_equity - 1.0
        print(f"Total return:   {total_ret:.2%}")

    print(f"\nTrades executed: {len(result.trades)}")
    for i, t in enumerate(result.trades, start=1):
        print(
            f"{i:02d}. {t.entry_time} -> {t.exit_time} "
            f"reason={t.exit_reason} "
            f"PnL={t.pnl:7.2f} ({t.pnl_pct:.2%}) "
            f"Equity after={t.equity_after:,.2f}"
        )


if __name__ == "__main__":
    main()
