import os
import random
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Iterable, Optional

import numpy as np

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
    DEFAULT_PORTFOLIO_USDJPY_FASTPASS_V3,
    DEFAULT_PROFILE_USDJPY_LIQUI_M15_V1,
    DEFAULT_PROFILE_USDJPY_LONDON_M15_V1,
    DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1,
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
from omegafx_v2.regime import tag_regimes

USE_REGIME_MASKS = os.getenv("USE_REGIME_MASKS", "0") == "1"


def apply_regime_mask(signals, regime_series, profile):
    if not USE_REGIME_MASKS or profile.edge_regime_config is None:
        return signals
    allowed = profile.edge_regime_config
    allowed_map = {
        "low_vol_range": allowed.allow_in_low_vol,
        "high_vol_trend": allowed.allow_in_high_vol_trend,
        "high_vol_reversal": allowed.allow_in_high_vol_reversal,
        "chop": allowed.allow_in_chop,
    }
    mask = regime_series.map(lambda r: allowed_map.get(r, False))
    mask = mask.reindex(signals.index, method="ffill").fillna(False)
    return signals & mask
from omegafx_v2.sim import simulate_trade_path


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


def build_signals_for_profile(profile, ohlc_m15, ohlc_m5):
    sig_cfg = profile.signals
    if isinstance(sig_cfg, LondonBreakoutSignalConfig):
        sig = build_london_breakout_signals(ohlc_m15, signal_config=sig_cfg, session=profile.session)
        return apply_regime_mask(sig, regime_series, profile)
    if isinstance(sig_cfg, LiquiditySweepSignalConfig):
        sig = build_liquidity_sweep_signals(ohlc_m15, signal_config=sig_cfg, session=profile.session)
        return apply_regime_mask(sig, regime_series, profile)
    if isinstance(sig_cfg, MomentumPinballSignalConfig):
        sig = build_momentum_pinball_signals_m5(ohlc_m5, ohlc_m15, signal_config=sig_cfg, session=profile.session)
        if USE_REGIME_MASKS:
            mask = regime_series.reindex(sig.index, method="ffill").fillna("low_vol_range")
            allowed = profile.edge_regime_config
            allowed_map = {
                "low_vol_range": allowed.allow_in_low_vol if allowed else False,
                "high_vol_trend": allowed.allow_in_high_vol_trend if allowed else True,
                "high_vol_reversal": allowed.allow_in_high_vol_reversal if allowed else False,
                "chop": allowed.allow_in_chop if allowed else False,
            }
            regime_mask = mask.map(lambda r: allowed_map.get(r, False)).astype(bool)
            sig = sig & regime_mask
        return sig
    raise TypeError(f"Unsupported signal config: {type(sig_cfg)}")


def run_multi_strategy_eval(ohlc_m15, ohlc_m5, profiles, risk_scales, initial_equity=10_000.0):
    master_index = ohlc_m15.index
    challenge = DEFAULT_CHALLENGE
    target_equity = initial_equity * (1.0 + challenge.profit_target_pct)
    loss_limit_equity = initial_equity * (1.0 - challenge.max_total_loss_pct)
    daily_loss_limit = initial_equity * float(os.getenv("OMEGAFX_DAILY_LOSS_PCT", "0.05"))

    events = []
    for profile in profiles:
        signals = build_signals_for_profile(profile, ohlc_m15, ohlc_m5)
        if signals is None or signals.sum() == 0:
            continue
        data = ohlc_m5 if profile.timeframe.lower().startswith("5") else ohlc_m15
        for idx, ts in enumerate(signals.index):
            if signals.iat[idx]:
                events.append((ts, profile, idx, data))

    events.sort(key=lambda x: x[0])

    equity = initial_equity
    max_equity = equity
    min_equity = equity
    trades_by_strategy = {}
    num_trades = 0
    verdict = "data_exhausted"
    first_entry_ts = None
    verdict_exit_ts = None
    trades_per_day_per_strategy = {}
    daily_pnl: dict = {}

    for ts, profile, entry_idx, data in events:
        if equity >= target_equity:
            verdict = "target_hit"
            verdict_exit_ts = ts
            break
        if equity <= loss_limit_equity:
            verdict = "max_loss_breached"
            verdict_exit_ts = ts
            break
        day = ts.date()
        if daily_pnl.get(day, 0.0) <= -daily_loss_limit:
            continue

        day = ts.date()
        key = (day, profile.name)
        max_trades = getattr(profile.signals, "max_trades_per_day", None)
        if max_trades:
            current = trades_per_day_per_strategy.get(key, 0)
            if current >= max_trades:
                continue

        scale = risk_scales.get(profile.name, 1.0)
        scaled_strategy = replace(
            profile.strategy,
            risk_per_trade_pct=profile.strategy.risk_per_trade_pct * scale,
            reward_per_trade_pct=profile.strategy.reward_per_trade_pct * scale,
            profit_target_pct=challenge.profit_target_pct,
        )

        outcome = simulate_trade_path(
            ohlc=data,
            entry_idx=entry_idx,
            account_balance=equity,
            config=scaled_strategy,
            costs=profile.costs,
        )

        equity += outcome.pnl
        num_trades += 1
        trades_by_strategy[profile.name] = trades_by_strategy.get(profile.name, 0) + 1
        trades_per_day_per_strategy[key] = trades_per_day_per_strategy.get(key, 0) + 1
        daily_pnl[day] = daily_pnl.get(day, 0.0) + outcome.pnl

        if first_entry_ts is None:
            first_entry_ts = ts

        if equity > max_equity:
            max_equity = equity
        if equity < min_equity:
            min_equity = equity

        if equity >= target_equity:
            verdict = "target_hit"
            verdict_exit_ts = outcome.exit_time
            break
        if equity <= loss_limit_equity:
            verdict = "max_loss_breached"
            verdict_exit_ts = outcome.exit_time
            break

    if verdict == "data_exhausted":
        if equity >= target_equity:
            verdict = "target_hit"
        elif equity <= loss_limit_equity:
            verdict = "max_loss_breached"

    total_return = equity / initial_equity - 1.0
    max_dd = (max_equity - min_equity) / max_equity if max_equity > 0 else 0.0

    bars_to_verdict = None
    trades_to_verdict = None
    if first_entry_ts is not None and verdict_exit_ts is not None:
        try:
            start_idx = master_index.get_indexer([first_entry_ts], method="ffill")[0]
            end_idx = master_index.get_indexer([verdict_exit_ts], method="ffill")[0]
            if start_idx != -1 and end_idx != -1:
                bars_to_verdict = end_idx - start_idx + 1
        except Exception:
            bars_to_verdict = None
        trades_to_verdict = num_trades

    return {
        "verdict": verdict,
        "total_return": total_return,
        "max_drawdown_pct": max_dd,
        "num_trades": num_trades,
        "trades_by_strategy": trades_by_strategy,
        "trades_to_verdict": trades_to_verdict,
        "bars_to_verdict": bars_to_verdict,
    }


def eval_windows(window_days: int, ohlc_m15_full, ohlc_m5_full, profiles, risk_scales, num_evals: int, initial_equity: float):
    window_delta = timedelta(days=window_days)
    actual_start = max(ohlc_m15_full.index.min(), ohlc_m5_full.index.min())
    actual_end = min(ohlc_m15_full.index.max(), ohlc_m5_full.index.max())

    master_index = ohlc_m15_full.index
    eligible_indices = [
        i for i, ts in enumerate(master_index)
        if ts >= actual_start and ts + window_delta <= actual_end
    ]
    if not eligible_indices:
        print("No eligible windows for requested history.")
        return None

    evaluations = []
    trades_totals = {}
    # regime tagging on full data
    global regime_series
    regime_series = tag_regimes(ohlc_m15_full, compute_atr(ohlc_m15_full, period=14), ohlc_m15_full["close"].ewm(span=50, adjust=False).mean().diff())

    for i_eval in range(num_evals):
        if (i_eval + 1) % max(1, num_evals // 5) == 0:
            print(f"  Window {i_eval+1}/{num_evals} for {window_days}d")
        start_idx = random.choice(eligible_indices)
        start_ts = master_index[start_idx]
        end_ts = start_ts + window_delta

        ohlc_m15 = ohlc_m15_full.loc[(ohlc_m15_full.index >= start_ts) & (ohlc_m15_full.index <= end_ts)]
        ohlc_m5 = ohlc_m5_full.loc[(ohlc_m5_full.index >= start_ts) & (ohlc_m5_full.index <= end_ts)]

        ev = run_multi_strategy_eval(
            ohlc_m15=ohlc_m15,
            ohlc_m5=ohlc_m5,
            profiles=profiles,
            risk_scales=risk_scales,
            initial_equity=initial_equity,
        )
        evaluations.append(ev)
        for k, v in ev["trades_by_strategy"].items():
            trades_totals[k] = trades_totals.get(k, 0) + v

    passes = [ev for ev in evaluations if ev["verdict"] == "target_hit"]
    fails = [ev for ev in evaluations if ev["verdict"] == "max_loss_breached"]

    def stats(subset, attr):
        return summarize([ev[attr] for ev in subset if ev.get(attr) is not None])

    pass_trades_stats = stats(passes, "trades_to_verdict")
    pass_days_stats = summarize(
        [(ev["bars_to_verdict"] or 0) * (15 / (60 * 24)) for ev in passes if ev.get("bars_to_verdict") is not None]
    )
    fail_trades_stats = stats(fails, "trades_to_verdict")
    fail_days_stats = summarize(
        [(ev["bars_to_verdict"] or 0) * (15 / (60 * 24)) for ev in fails if ev.get("bars_to_verdict") is not None]
    )

    avg_return = sum(ev["total_return"] for ev in evaluations) / len(evaluations)
    avg_max_dd = sum(ev["max_drawdown_pct"] for ev in evaluations) / len(evaluations)
    avg_trades = sum(ev["num_trades"] for ev in evaluations) / len(evaluations)
    avg_trades_per_strategy = {k: v / len(evaluations) for k, v in trades_totals.items()}

    return {
        "passes": len(passes),
        "fails": len(fails),
        "pass_rate": len(passes) / len(evaluations) if evaluations else 0.0,
        "avg_return": avg_return,
        "avg_max_dd": avg_max_dd,
        "avg_trades": avg_trades,
        "avg_trades_per_strategy": avg_trades_per_strategy,
        "pass_trades_stats": pass_trades_stats,
        "pass_days_stats": pass_days_stats,
        "fail_trades_stats": fail_trades_stats,
        "fail_days_stats": fail_days_stats,
    }


def main() -> None:
    if not init_mt5():
        return

    end = datetime.now()
    total_history_days = int(os.getenv("TOTAL_HISTORY_DAYS", "270"))
    backoff_steps = [total_history_days, 180, 120, 90, 60]
    num_evals = int(os.getenv("NUM_EVALS", "200"))
    initial_equity = 10_000.0

    def fetch_with_backoff(tf):
        for days in backoff_steps:
            start = end - timedelta(days=days)
            try:
                df = fetch_symbol_ohlc_mt5("USDJPY", tf, start, end)
                if not df.empty:
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
    risk_scales_list = DEFAULT_PORTFOLIO_USDJPY_FASTPASS_V3.risk_scales or [1.0] * len(profiles)
    risk_scales = {profiles[i].name: (risk_scales_list[i] if i < len(risk_scales_list) else 1.0) for i in range(len(profiles))}

    for window_days in (30, 60):
        print(f"\n=== USDJPY FastPass V3 – {window_days}d window ===")
        result = eval_windows(window_days, ohlc_m15_full, ohlc_m5_full, profiles, risk_scales, num_evals, initial_equity)
        if result is None:
            continue
        print(f"Passes: {result['passes']} ({result['pass_rate']:.2%}), Fails: {result['fails']}")
        print(
            f"Avg_return: {result['avg_return']:.2%}, Avg_max_dd: {result['avg_max_dd']:.2%}, "
            f"Avg_trades/eval: {result['avg_trades']:.2f}"
        )
        print(f"Avg trades per strategy: {result['avg_trades_per_strategy']}")
        ps = result["pass_trades_stats"]
        pd = result["pass_days_stats"]
        fs = result["fail_trades_stats"]
        fd = result["fail_days_stats"]
        print("Passing evals:")
        if ps and pd:
            print(
                f"  Trades to pass avg/med/p90: {ps['avg']:.2f} / {ps['median']:.2f} / {ps['p90']:.2f}"
            )
            print(
                f"  Days to pass   avg/med/p90: {pd['avg']:.2f} / {pd['median']:.2f} / {pd['p90']:.2f}"
            )
        else:
            print("  None")
        print("Failing evals:")
        if fs and fd:
            print(
                f"  Trades to fail avg/med/p90: {fs['avg']:.2f} / {fs['median']:.2f} / {fs['p90']:.2f}"
            )
            print(
                f"  Days to fail   avg/med/p90: {fd['avg']:.2f} / {fd['median']:.2f} / {fd['p90']:.2f}"
            )
        else:
            print("  None")

    if mt5:
        mt5.shutdown()


if __name__ == "__main__":
    main()
