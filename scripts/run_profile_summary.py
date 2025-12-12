from omegafx_v2.config import DEFAULT_PROFILE_XAU_MR_V1
from omegafx_v2.profile_summary import run_profile_summary


def main() -> None:
    profile = DEFAULT_PROFILE_XAU_MR_V1

    print(f"=== Strategy Profile (V1): {profile.name} ===")
    print("Signals:", profile.signals)
    print("Challenge:", profile.challenge)
    print("Costs:", profile.costs)
    print("Session:", profile.session)

    metrics = run_profile_summary(
        profile=profile,
        days=365,
        initial_equity=10_000.0,
        num_evals=100,
    )

    print("\n--- Data & Signals ---")
    print(f"Bars:            {metrics['bars']}")
    print(f"Total signals:   {metrics['signals_total']}")

    print("\n--- Single Evaluation ---")
    print(f"Verdict:         {metrics['single_verdict']}")
    print(f"Return:          {metrics['single_return']:.2%}")
    print(f"Max drawdown:    {metrics['single_max_dd']:.2%}")

    print("\n--- Monte Carlo Batch ---")
    print(f"Pass rate:       {metrics['batch_pass_rate']:.2%}")
    print(f"Avg return:      {metrics['batch_avg_return']:.2%}")
    print(f"Avg max DD:      {metrics['batch_avg_max_dd']:.2%}")
    print(f"Avg trades/eval: {metrics['batch_avg_trades']:.2f}")


if __name__ == "__main__":
    main()
