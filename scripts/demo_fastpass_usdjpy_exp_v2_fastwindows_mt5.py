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
    DEFAULT_PROFILE_USDJPY_LONDON_M15_V1,
    DEFAULT_PROFILE_USDJPY_LIQUI_M15_V1,
    DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1,
    EXPERIMENTAL_PROFILE_USDJPY_TREND_KD_M15_V1,
    EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V2,
    LondonBreakoutSignalConfig,
    LiquiditySweepSignalConfig,
    MomentumPinballSignalConfig,
    TrendKDSignalConfig,
)
from omegafx_v2.mt5_adapter import fetch_symbol_ohlc_mt5
from omegafx_v2.signals import (
    build_liquidity_sweep_signals,
    build_london_breakout_signals,
    build_momentum_pinball_signals_m5,
    build_trend_kd_signals_m15,
    compute_atr,
)
from omegafx_v2.regime import tag_regimes
from omegafx_v2.sim import simulate_trade_path


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


def build_signals_for_profile(profile, ohlc_m15, ohlc_m5, regime_series):
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
            mask = regime_series.reindex(ohlc_m5.index, method="ffill").fillna("low_vol_range")
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
    if isinstance(sig_cfg, TrendKDSignalConfig):
        sig = build_trend_kd_signals_m15(ohlc_m15, signal_config=sig_cfg, session=profile.session)
        return apply_regime_mask(sig, regime_series, profile)
    raise TypeError(f"Unsupported signal config: {type(sig_cfg)}")


def run_multi_strategy_eval(ohlc_m15, ohlc_m5, profiles, risk_scales, initial_equity=10_000.0, risk_control=None):
    master_index = ohlc_m15.index
    challenge = DEFAULT_CHALLENGE
    target_equity = initial_equity * (1.0 + challenge.profit_target_pct)
    loss_limit_equity = initial_equity * (1.0 - challenge.max_total_loss_pct)
    daily_loss_limit = initial_equity * float(os.getenv("OMEGAFX_DAILY_LOSS_PCT", "0.05"))

    # Regime tagging
    regimes = tag_regimes(ohlc_m15, compute_atr(ohlc_m15, period=14), ohlc_m15["close"].ewm(span=50, adjust=False).mean().diff())

    events = []
    for profile in profiles:
        signals = build_signals_for_profile(profile, ohlc_m15, ohlc_m5, regimes)
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

    current_scales = dict(risk_scales)
    reduced = False

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

        key = (day, profile.name)
        max_trades = getattr(profile.signals, "max_trades_per_day", None)
        if max_trades:
            current = trades_per_day_per_strategy.get(key, 0)
            if current >= max_trades:
                continue

        # risk control per-eval
        if risk_control:
            peak = max_equity
            dd = (peak - equity) / peak if peak > 0 else 0
            if (not reduced) and dd >= risk_control.dd_reduce_threshold_pct:
                current_scales = {k: v * risk_control.risk_scale_reduction_factor for k, v in current_scales.items()}
                reduced = True
            elif reduced and dd <= risk_control.dd_restore_threshold_pct:
                current_scales = dict(risk_scales)
                reduced = False

        scale = current_scales.get(profile.name, 1.0)
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

    for i_eval in range(num_evals):
        if (i_eval + 1) % max(1, num_evals // 5) == 0:
            print(f"Window {i_eval+1}/{num_evals} for {window_days}d")
        start_idx = random.choice(eligible_indices)
        start_ts = master_index[start_idx]
        end_ts = start_ts + window_delta

        ohlc_m15 = ohlc_m15_full.loc[start_ts:end_ts]
        ohlc_m5 = ohlc_m5_full.loc[start_ts:end_ts]

        res = run_multi_strategy_eval(
            ohlc_m15,
            ohlc_m5,
            profiles,
            risk_scales,
            initial_equity=initial_equity,
            risk_control=EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V2.risk_control,
        )
        if res is None:
            continue
        evaluations.append(res)
        for k, v in res["trades_by_strategy"].items():
            trades_totals[k] = trades_totals.get(k, 0) + v

    return evaluations, trades_totals


def main():
    if not init_mt5():
        return
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    end = datetime.now()
    total_history_days = int(os.getenv("TOTAL_HISTORY_DAYS", "270"))
    num_evals = int(os.getenv("NUM_EVALS", "200"))
    start = end - timedelta(days=total_history_days)

    ohlc_m15 = fetch_symbol_ohlc_mt5("USDJPY", mt5.TIMEFRAME_M15, start, end)
    ohlc_m5 = fetch_symbol_ohlc_mt5("USDJPY", mt5.TIMEFRAME_M5, start, end)
    if ohlc_m15 is None or ohlc_m5 is None:
        print("Failed to fetch MT5 data.")
        return

    profiles = [
        DEFAULT_PROFILE_USDJPY_LONDON_M15_V1,
        DEFAULT_PROFILE_USDJPY_LIQUI_M15_V1,
        DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1,
        EXPERIMENTAL_PROFILE_USDJPY_TREND_KD_M15_V1,
    ]
    base_scales = {
        p.name: (EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V2.risk_scales[i] if EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V2.risk_scales else 1.0)
        for i, p in enumerate(profiles)
    }

    results = {}
    for window_days in (30, 60):
        evals, trades_totals = eval_windows(window_days, ohlc_m15, ohlc_m5, profiles, base_scales, num_evals=num_evals, initial_equity=10_000.0)
        if not evals:
            continue
        passes = [e for e in evals if e["verdict"] == "target_hit"]
        fails = [e for e in evals if e["verdict"] == "max_loss_breached"]
        pass_rate = len(passes) / len(evals) if evals else 0.0
        avg_return = float(np.mean([e["total_return"] for e in evals])) if evals else 0.0
        avg_max_dd = float(np.mean([e["max_drawdown_pct"] for e in evals])) if evals else 0.0
        avg_trades = float(np.mean([e["num_trades"] for e in evals])) if evals else 0.0
        print(f"=== USDJPY Exp V2 – {window_days}d window ===")
        print(f"Passes: {len(passes)} ({pass_rate:.2%}), Fails: {len(fails)}")
        print(f"Avg_return: {avg_return:.2%}, Avg_max_dd: {avg_max_dd:.2%}, Avg_trades/eval: {avg_trades:.2f}")
        print(f"Avg trades per strategy: { {k: v/len(evals) for k,v in trades_totals.items()} }")
        res_block = {
            "pass_rate": pass_rate,
            "avg_return": avg_return,
            "avg_max_dd": avg_max_dd,
            "avg_trades": avg_trades,
            "trades_per_strategy": {k: v/len(evals) for k,v in trades_totals.items()},
        }
        results[f"{window_days}d"] = res_block
    out_path = reports_dir / "fastpass_usdjpy_exp_v2_fastwindows.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote summary to {out_path}")
    if mt5:
        mt5.shutdown()


if __name__ == "__main__":
    import json
    main()
