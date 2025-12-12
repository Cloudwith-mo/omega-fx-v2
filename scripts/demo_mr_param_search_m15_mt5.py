import os
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from omegafx_v2.config import DEFAULT_PROFILE_USDJPY_MR_M15_V1, MeanReversionSignalConfig
from omegafx_v2.mt5_adapter import fetch_symbol_ohlc_mt5
from omegafx_v2.profile_summary import _build_signals_for_profile
from omegafx_v2.sim import run_randomized_signal_evaluations


def init_mt5() -> bool:
    if mt5 is None:
        print("MetaTrader5 package not installed.")
        return False
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}")
        return False
    if all([login, password, server]):
        if not mt5.login(login=int(login), password=password, server=server):
            print(f"MT5 login failed: {mt5.last_error()}")
            return False
    return True


def main() -> None:
    profile = DEFAULT_PROFILE_USDJPY_MR_M15_V1

    if not init_mt5():
        return

    end = datetime.now()
    start = end - timedelta(days=180)

    ohlc = fetch_symbol_ohlc_mt5(
        symbol=profile.symbol_key,
        timeframe=mt5.TIMEFRAME_M15,
        start=start,
        end=end,
    )

    ma_periods = [30, 50, 80]
    entry_ks = [0.5, 1.0, 1.5]

    rows = []

    for ma in ma_periods:
        for k in entry_ks:
            sig_cfg = MeanReversionSignalConfig(
                ma_period=ma,
                atr_period=14,
                entry_k=k,
                exit_k=0.0,
                h4_sma_period=50,
            )

            temp_profile = replace(profile, signals=sig_cfg)

            signals = _build_signals_for_profile(ohlc, temp_profile)
            if signals.sum() == 0:
                continue

            strategy_cfg = replace(
                temp_profile.strategy,
                profit_target_pct=temp_profile.challenge.profit_target_pct,
            )

            batch = run_randomized_signal_evaluations(
                ohlc=ohlc,
                signals=signals,
                initial_equity=10_000.0,
                challenge=temp_profile.challenge,
                config=strategy_cfg,
                num_evals=100,
                min_bars_per_eval=temp_profile.challenge.min_bars_per_eval,
                start_offset=5,
            )

            avg_trades = (
                sum(len(ev.trades) for ev in batch.evaluations) / batch.num_evals
                if batch.num_evals > 0
                else 0.0
            )

            rows.append(
                (
                    ma,
                    k,
                    batch.pass_rate,
                    batch.average_return,
                    batch.average_max_drawdown,
                    avg_trades,
                )
            )

    rows.sort(key=lambda r: r[2], reverse=True)

    print("ma  entry_k  pass_rate  avg_ret  avg_dd  avg_trades")
    print("---------------------------------------------------")
    for ma, k, pr, ar, dd, at in rows:
        print(
            f"{ma:3d}  {k:7.2f}  "
            f"{pr:9.3f}  {ar:7.3f}  {dd:7.3f}  {at:10.2f}"
        )

    if mt5:
        mt5.shutdown()


if __name__ == "__main__":
    main()
