import os
from pathlib import Path


def main() -> None:
    # Inputs (can be overridden via env)
    account_size = float(os.getenv("FTMO_ACCOUNT_SIZE", "100000"))
    profit_target_pct = float(os.getenv("FTMO_PROFIT_TARGET_PCT", "0.10"))  # 10%
    payout_share = float(os.getenv("FTMO_PAYOUT_SHARE", "0.80"))  # 80%
    challenge_fee = float(os.getenv("FTMO_CHALLENGE_FEE", "540"))

    pass_rate = float(os.getenv("FTMO_PASS_RATE", "0.93"))
    fail_rate = float(os.getenv("FTMO_FAIL_RATE", "0.07"))

    # Derived
    payout_if_pass = account_size * profit_target_pct * payout_share
    # Break-even pass rate solving: p * payout - (1 - p) * fee = 0
    breakeven_pass = challenge_fee / (payout_if_pass + challenge_fee) if (payout_if_pass + challenge_fee) != 0 else 0.0
    ev_challenge = pass_rate * payout_if_pass - fail_rate * challenge_fee
    ev_per_fee = ev_challenge / challenge_fee if challenge_fee else 0.0

    print("=== FTMO USDJPY V3 Economics ===")
    print(f"Account size: {account_size:,.0f}")
    print(f"Profit target: {profit_target_pct:.0%}, Payout share: {payout_share:.0%}")
    print(f"Challenge fee: {challenge_fee:,.0f}")
    print()
    print(f"Pass rate: {pass_rate:.2f}")
    print(f"Fail rate: {fail_rate:.2f}")
    print()
    print(f"Payout if pass: {payout_if_pass:,.0f}")
    print(f"Break-even pass rate: {breakeven_pass:.2%}")
    print(f"EV per challenge: {ev_challenge:,.0f}")
    print(f"EV per $ of fee: {ev_per_fee:.2f}")


if __name__ == "__main__":
    main()
