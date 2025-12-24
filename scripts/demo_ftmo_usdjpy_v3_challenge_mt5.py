import bisect
import os
import random
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Iterable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from omegafx_v2.config import (
    DEFAULT_PORTFOLIO_USDJPY_FASTPASS_V3,
    DEFAULT_PROFILE_USDJPY_LIQUI_M15_V1,
    DEFAULT_PROFILE_USDJPY_LONDON_M15_V1,
    DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1,
    FTMO_CHALLENGE_USDJPY,
    LondonBreakoutSignalConfig,
    LiquiditySweepSignalConfig,
    MomentumPinballSignalConfig,
)
from omegafx_v2.mt5_adapter import fetch_symbol_ohlc_mt5
from omegafx_v2.signals import (
    build_liquidity_sweep_signals,
    build_london_breakout_signals,
    build_momentum_pinball_signals_m5,
    compute_atr,
)
from omegafx_v2.sim import simulate_trade_path
from omegafx_v2.regime import tag_regimes

USE_REGIME_MASKS = os.getenv("USE_REGIME_MASKS", "0") == "1"


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


def summarize(values: Iterable[float]) -> Optional[dict]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    arr = np.array(vals, dtype=float)
    return {
        "avg": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def build_signals(profile, ohlc_m15: pd.DataFrame, ohlc_m5: pd.DataFrame):
    sig_cfg = profile.signals
    if isinstance(sig_cfg, LondonBreakoutSignalConfig):
        return build_london_breakout_signals(ohlc_m15, signal_config=sig_cfg, session=profile.session)
    if isinstance(sig_cfg, LiquiditySweepSignalConfig):
        return build_liquidity_sweep_signals(ohlc_m15, signal_config=sig_cfg, session=profile.session)
    if isinstance(sig_cfg, MomentumPinballSignalConfig):
        return build_momentum_pinball_signals_m5(ohlc_m5, ohlc_m15, signal_config=sig_cfg, session=profile.session)
    raise TypeError(f"Unsupported signal config: {type(sig_cfg)}")


def classify_regime(
    ohlc_m15: pd.DataFrame,
    low_vol_pct: float = 0.4,
    high_vol_pct: float = 0.7,
    trend_slope_threshold: float = 0.0001,
):
    """Simple regime tagging using ATR percentile and H1 EMA slope."""
    atr = compute_atr(ohlc_m15, period=14)
    atr_rank = atr.rank(pct=True)
    h1_close = ohlc_m15["close"].resample("1h").last().ffill()
    h1_ema = h1_close.ewm(span=20, adjust=False).mean()
    h1_slope = h1_ema.diff()
    h1_slope_on_m15 = h1_slope.reindex(ohlc_m15.index, method="ffill").fillna(0)

    regimes = []
    for pct, slope in zip(atr_rank, h1_slope_on_m15):
        if pct <= low_vol_pct:
            regimes.append("low_vol_range")
        elif pct >= high_vol_pct and abs(slope) > trend_slope_threshold:
            regimes.append("high_vol_trend")
        elif pct >= high_vol_pct and abs(slope) <= trend_slope_threshold:
            regimes.append("high_vol_reversal")
        else:
            regimes.append("chop")
    return pd.Series(regimes, index=ohlc_m15.index, name="regime")


def prepare_events(profiles, ohlc_m15: pd.DataFrame, ohlc_m5: pd.DataFrame, regime_series: pd.Series):
    events: list[tuple[datetime, int, int, str, Optional[str]]] = []
    for idx, profile in enumerate(profiles):
        signals = build_signals(profile, ohlc_m15, ohlc_m5)
        if signals is None or signals.sum() == 0:
            continue
        data_key = "m5" if profile.timeframe.startswith("5") else "m15"
        for i, ts in enumerate(signals.index):
            if signals.iat[i]:
                regime = regime_series.get(ts, None)
                if USE_REGIME_MASKS and profile.edge_regime_config:
                    allowed = profile.edge_regime_config
                    allowed_map = {
                        "low_vol_range": allowed.allow_in_low_vol,
                        "high_vol_trend": allowed.allow_in_high_vol_trend,
                        "high_vol_reversal": allowed.allow_in_high_vol_reversal,
                        "chop": allowed.allow_in_chop,
                    }
                    if not allowed_map.get(regime, False):
                        continue
                events.append((ts, idx, i, data_key, regime))
    events.sort(key=lambda x: x[0])
    return events


def run_challenge(
    events,
    ts_list,
    profiles,
    data_map,
    risk_scales,
    max_days: int,
    initial_equity: float = 100_000.0,
    allowed_regimes: set[str] | None = None,
    early_stop_pct: float | None = 0.06,
):
    if not events:
        return None

    challenge = FTMO_CHALLENGE_USDJPY
    target_equity = initial_equity * (1.0 + challenge.profit_target_pct)
    loss_limit_equity = initial_equity * (1.0 - challenge.max_total_loss_pct)
    daily_loss_limit = initial_equity * (challenge.daily_loss_pct or 0.05)

    max_start_ts = events[-1][0] - timedelta(days=max_days)
    max_start_idx = bisect.bisect_right(ts_list, max_start_ts) - 1
    if max_start_idx < 0:
        return None

    start_idx = random.randint(0, max_start_idx)
    start_ts = ts_list[start_idx]
    horizon_end = start_ts + timedelta(days=max_days)

    equity = initial_equity
    max_equity = equity
    min_equity = equity
    trades_by_strategy: dict[str, int] = {}
    num_trades = 0
    verdict = "timeout"
    first_entry_ts = None
    verdict_exit_ts = None
    trades_per_day_per_strategy: dict[tuple[datetime.date, str], int] = {}
    trading_days_set = set()
    daily_pnl: dict = {}

    end_idx = bisect.bisect_right(ts_list, horizon_end)

    for ts, profile_idx, entry_idx, data_key, regime in events[start_idx:end_idx]:
        if allowed_regimes is not None and regime not in allowed_regimes:
            continue
        if ts > horizon_end:
            verdict = "timeout"
            break
        if equity >= target_equity:
            verdict = "challenge_pass"
            verdict_exit_ts = ts
            break
        if equity <= loss_limit_equity:
            verdict = "challenge_fail"
            verdict_exit_ts = ts
            break

        day = ts.date()
        if daily_pnl.get(day, 0.0) <= -daily_loss_limit:
            continue

        profile = profiles[profile_idx]
        key = (day, profile.name)
        max_trades = getattr(profile.signals, "max_trades_per_day", None)
        if max_trades and trades_per_day_per_strategy.get(key, 0) >= max_trades:
            continue

        scale = risk_scales.get(profile.name, 1.0)
        cfg = replace(
            profile.strategy,
            risk_per_trade_pct=profile.strategy.risk_per_trade_pct * scale,
            reward_per_trade_pct=profile.strategy.reward_per_trade_pct * scale,
            profit_target_pct=challenge.profit_target_pct,
        )
        data = data_map[data_key]
        try:
            outcome = simulate_trade_path(
                ohlc=data,
                entry_idx=entry_idx,
                account_balance=equity,
                config=cfg,
                costs=profile.costs,
            )
        except Exception:
            continue

        equity += outcome.pnl
        num_trades += 1
        trades_by_strategy[profile.name] = trades_by_strategy.get(profile.name, 0) + 1
        trades_per_day_per_strategy[key] = trades_per_day_per_strategy.get(key, 0) + 1
        daily_pnl[day] = daily_pnl.get(day, 0.0) + outcome.pnl
        trading_days_set.add(day)

        if first_entry_ts is None:
            first_entry_ts = ts
        max_equity = max(max_equity, equity)
        min_equity = min(min_equity, equity)

        if equity >= target_equity:
            equity = target_equity
            verdict = "challenge_pass"
            verdict_exit_ts = ts
            break
        if early_stop_pct is not None and equity >= initial_equity * (1.0 + early_stop_pct):
            verdict = "early_stop"
            verdict_exit_ts = ts
            break
        if equity <= loss_limit_equity:
            equity = loss_limit_equity
            min_equity = max(min_equity, loss_limit_equity)
            verdict = "challenge_fail"
            verdict_exit_ts = ts
            break

    total_return = equity / initial_equity - 1.0
    max_dd = (initial_equity - min_equity) / initial_equity if initial_equity > 0 else 0.0
    days_to_verdict = None
    trades_to_verdict = None
    if verdict in {"challenge_pass", "challenge_fail", "early_stop"} and first_entry_ts is not None:
        if verdict_exit_ts and verdict_exit_ts > horizon_end:
            verdict = "timeout"
        else:
            end_ts = min(verdict_exit_ts or horizon_end, horizon_end)
            days_to_verdict = (end_ts - start_ts).total_seconds() / (60 * 60 * 24)
            trades_to_verdict = num_trades

    return {
        "verdict": verdict,
        "total_return": total_return,
        "max_drawdown_pct": max_dd,
        "num_trades": num_trades,
        "trades_by_strategy": trades_by_strategy,
        "trades_to_verdict": trades_to_verdict,
        "days_to_verdict": days_to_verdict,
        "trading_days": len(trading_days_set),
    }


def main() -> None:
    if not init_mt5():
        return

    end = datetime.now()
    total_history_days = int(os.getenv("FTMO_TOTAL_HISTORY_DAYS", os.getenv("TOTAL_HISTORY_DAYS", "365")))
    max_days = int(os.getenv("FTMO_MAX_DAYS", "90"))
    num_runs = int(os.getenv("FTMO_NUM_RUNS", os.getenv("NUM_RUNS", "100")))
    initial_equity = 100_000.0
    backoff_steps = [total_history_days, 540, 360, 270, 180]

    def fetch_with_backoff(tf):
        for days in backoff_steps:
            start = end - timedelta(days=days)
            try:
                df = fetch_symbol_ohlc_mt5("USDJPY", tf, start, end)
                if df is not None and not df.empty:
                    return df
            except Exception:
                continue
        return None

    ohlc_m15_full = fetch_with_backoff(mt5.TIMEFRAME_M15)
    ohlc_m5_full = fetch_with_backoff(mt5.TIMEFRAME_M5)
    if ohlc_m15_full is None or ohlc_m5_full is None:
        print("Failed to fetch MT5 data.")
        return

    profiles = [
        DEFAULT_PROFILE_USDJPY_LONDON_M15_V1,
        DEFAULT_PROFILE_USDJPY_LIQUI_M15_V1,
        DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1,
    ]
    risk_scales_list = DEFAULT_PORTFOLIO_USDJPY_FASTPASS_V3.risk_scales or [1.60, 0.25, 0.01]
    risk_scales = {profiles[i].name: risk_scales_list[i] for i in range(len(profiles))}

    regime_series = classify_regime(ohlc_m15_full)
    events = prepare_events(profiles, ohlc_m15_full, ohlc_m5_full, regime_series)
    if not events:
        print("No signals found in history window.")
        if mt5:
            mt5.shutdown()
        return
    ts_list = [e[0] for e in events]
    data_map = {"m15": ohlc_m15_full, "m5": ohlc_m5_full}

    results = []
    for i in range(num_runs):
        if (i + 1) % max(1, num_runs // 10) == 0:
            print(f"Run {i + 1}/{num_runs}")
        ev = run_challenge(events, ts_list, profiles, data_map, risk_scales, max_days, initial_equity)
        if ev is not None:
            results.append(ev)

    passes = [r for r in results if r["verdict"] == "challenge_pass"]
    early = [r for r in results if r["verdict"] == "early_stop"]
    fails = [r for r in results if r["verdict"] == "challenge_fail"]
    timeouts = [r for r in results if r["verdict"] == "timeout"]

    def stats(subset, key):
        return summarize([r[key] for r in subset if r.get(key) is not None])

    pass_days = stats(passes, "days_to_verdict")
    fail_days = stats(fails, "days_to_verdict")
    pass_trades = stats(passes, "trades_to_verdict")
    fail_trades = stats(fails, "trades_to_verdict")
    avg_final_return = sum(r["total_return"] for r in results) / len(results) if results else 0.0
    avg_pass_dd = sum(r["max_drawdown_pct"] for r in passes) / len(passes) if passes else 0.0
    worst_pass_dd = max((r["max_drawdown_pct"] for r in passes), default=0.0)
    avg_fail_dd = sum(r["max_drawdown_pct"] for r in fails) / len(fails) if fails else 0.0
    worst_fail_dd = max((r["max_drawdown_pct"] for r in fails), default=0.0)
    min_days_ok = sum(1 for r in results if r.get("trading_days", 0) >= 4) / len(results) if results else 0.0

    print("=== FTMO USDJPY FastPass V3 – Challenge Simulation ===")
    if results:
        print(
            f"Runs: {len(results)} | Passes: {len(passes)} ({len(passes)/len(results):.2%}) | "
            f"Early stops: {len(early)} ({len(early)/len(results):.2%}) | "
            f"Fails: {len(fails)} ({len(fails)/len(results):.2%}) | Timeouts: {len(timeouts)} ({len(timeouts)/len(results):.2%})"
        )
    else:
        print("No runs completed.")
    print(f"Average final return: {avg_final_return:.2%}")
    print(f"Min trading days >=4: {min_days_ok:.2%} of runs")
    print("Passing runs:")
    if passes:
        print(
            f"  Days to pass avg/med/p90: {(pass_days or {}).get('avg', 0):.2f} / "
            f"{(pass_days or {}).get('median', 0):.2f} / {(pass_days or {}).get('p90', 0):.2f}"
        )
        print(
            f"  Trades to pass avg/med/p90: {(pass_trades or {}).get('avg', 0):.2f} / "
            f"{(pass_trades or {}).get('median', 0):.2f} / {(pass_trades or {}).get('p90', 0):.2f}"
        )
        print(f"  Max DD avg/worst: {avg_pass_dd:.2%} / {worst_pass_dd:.2%}")
    else:
        print("  None")
    print("Early-stop runs:")
    if early:
        es_days = stats(early, "days_to_verdict")
        es_trades = stats(early, "trades_to_verdict")
        print(
            f"  Days to early-stop avg/med/p90: {(es_days or {}).get('avg', 0):.2f} / "
            f"{(es_days or {}).get('median', 0):.2f} / {(es_days or {}).get('p90', 0):.2f}"
        )
        print(
            f"  Trades to early-stop avg/med/p90: {(es_trades or {}).get('avg', 0):.2f} / "
            f"{(es_trades or {}).get('median', 0):.2f} / {(es_trades or {}).get('p90', 0):.2f}"
        )
    else:
        print("  None")
    print("Failing runs:")
    if fails:
        print(
            f"  Days to fail avg/med/p90: {(fail_days or {}).get('avg', 0):.2f} / "
            f"{(fail_days or {}).get('median', 0):.2f} / {(fail_days or {}).get('p90', 0):.2f}"
        )
        print(
            f"  Trades to fail avg/med/p90: {(fail_trades or {}).get('avg', 0):.2f} / "
            f"{(fail_trades or {}).get('median', 0):.2f} / {(fail_trades or {}).get('p90', 0):.2f}"
        )
        print(f"  Max DD avg/worst: {avg_fail_dd:.2%} / {worst_fail_dd:.2%}")
    else:
        print("  None")

    regimes = ["low_vol_range", "high_vol_trend", "high_vol_reversal", "chop"]
    print("\n=== Per-Regime Results (separate runs) ===")
    for regime in regimes:
        regime_results = []
        for _ in range(num_runs):
            ev = run_challenge(
                events,
                ts_list,
                profiles,
                data_map,
                risk_scales,
                max_days,
                initial_equity,
                allowed_regimes={regime},
            )
            if ev is not None:
                regime_results.append(ev)
        if not regime_results:
            print(f"{regime}: no runs")
            continue
        reg_pass = [r for r in regime_results if r["verdict"] == "challenge_pass"]
        reg_fail = [r for r in regime_results if r["verdict"] == "challenge_fail"]
        reg_early = [r for r in regime_results if r["verdict"] == "early_stop"]
        reg_timeout = [r for r in regime_results if r["verdict"] == "timeout"]
        reg_avg_ret = sum(r["total_return"] for r in regime_results) / len(regime_results)
        reg_pass_days = stats(reg_pass, "days_to_verdict")
        reg_fail_days = stats(reg_fail, "days_to_verdict")
        reg_pass_dd_avg = sum(r["max_drawdown_pct"] for r in reg_pass) / len(reg_pass) if reg_pass else 0.0
        reg_fail_dd_avg = sum(r["max_drawdown_pct"] for r in reg_fail) / len(reg_fail) if reg_fail else 0.0
        print(
            f"{regime}: runs={len(regime_results)}, "
            f"pass={len(reg_pass)} ({len(reg_pass)/len(regime_results):.2%}), "
            f"early={len(reg_early)} ({len(reg_early)/len(regime_results):.2%}), "
            f"fail={len(reg_fail)} ({len(reg_fail)/len(regime_results):.2%}), "
            f"timeout={len(reg_timeout)} ({len(reg_timeout)/len(regime_results):.2%}), "
            f"avg_ret={reg_avg_ret:.2%}"
        )
        if reg_pass:
            print(
                f"  Pass days med: {(reg_pass_days or {}).get('median', 0):.2f}, "
                f"p90: {(reg_pass_days or {}).get('p90', 0):.2f}, "
                f"avg DD (pass): {reg_pass_dd_avg:.2%}"
            )
        if reg_fail:
            print(
                f"  Fail days med: {(reg_fail_days or {}).get('median', 0):.2f}, "
                f"avg DD (fail): {reg_fail_dd_avg:.2%}"
            )

    if mt5:
        mt5.shutdown()


if __name__ == "__main__":
    main()
