import os
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

from omegafx_v2.config import (
    DEFAULT_CHALLENGE,
    DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1,
    MomentumPinballSignalConfig,
)
from omegafx_v2.mt5_adapter import fetch_symbol_ohlc_mt5
from omegafx_v2.signals import build_momentum_pinball_signals_m5
from omegafx_v2.sim import run_randomized_signal_evaluations, run_signal_driven_evaluation


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
    if not init_mt5():
        return

    end = datetime.now()
    start = end - timedelta(days=180)

    ohlc_m5 = fetch_symbol_ohlc_mt5("USDJPY", mt5.TIMEFRAME_M5, start, end)
    ohlc_m15 = fetch_symbol_ohlc_mt5("USDJPY", mt5.TIMEFRAME_M15, start, end)

    profile = DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1
    signal_cfg: MomentumPinballSignalConfig = profile.signals  # type: ignore

    signals = build_momentum_pinball_signals_m5(
        ohlc_m5=ohlc_m5,
        ohlc_m15=ohlc_m15,
        signal_config=signal_cfg,
        session=profile.session,
    )

    strategy_cfg = profile.strategy
    single = run_signal_driven_evaluation(
        ohlc=ohlc_m5,
        signals=signals,
        initial_equity=10_000.0,
        challenge=DEFAULT_CHALLENGE,
        config=strategy_cfg,
        costs=profile.costs,
        daily_loss_pct=DEFAULT_CHALLENGE.daily_loss_pct,
    )

    batch = run_randomized_signal_evaluations(
        ohlc=ohlc_m5,
        signals=signals,
        initial_equity=10_000.0,
        challenge=DEFAULT_CHALLENGE,
        config=strategy_cfg,
        costs=profile.costs,
        num_evals=100,
        min_bars_per_eval=DEFAULT_CHALLENGE.min_bars_per_eval,
        start_offset=5,
        daily_loss_pct=DEFAULT_CHALLENGE.daily_loss_pct,
    )

    print("=== USDJPY M5 MomentumPinball V1 (MT5) ===")
    print(f"Bars M5: {len(ohlc_m5)}, Bars M15: {len(ohlc_m15)}")
    print(f"Signals: {signals.sum()}")
    print(f"Single verdict: {single.verdict}, return={single.total_return:.2%}, max_dd={single.max_drawdown_pct:.2%}, trades={single.num_trades}")
    print(
        f"Pass_rate: {batch.pass_rate:.2%}, Avg_return: {batch.average_return:.2%}, "
        f"Avg_max_dd: {batch.average_max_drawdown:.2%}, Avg_trades/eval: {sum(r.num_trades for r in batch.evaluations)/batch.num_evals:.2f}"
    )

    if mt5:
        mt5.shutdown()


if __name__ == "__main__":
    main()
