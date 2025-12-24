import bisect
import json
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
    EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V2,
    EXPERIMENTAL_PROFILE_USDJPY_TREND_KD_M15_V1,
    DEFAULT_PROFILE_USDJPY_LONDON_M15_V1,
    DEFAULT_PROFILE_USDJPY_LIQUI_M15_V1,
    DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1,
    LondonBreakoutSignalConfig,
    LiquiditySweepSignalConfig,
    MomentumPinballSignalConfig,
)
from omegafx_v2.mt5_adapter import fetch_symbol_ohlc_mt5
from omegafx_v2.signals import (
    build_london_breakout_signals,
    build_liquidity_sweep_signals,
    build_momentum_pinball_signals_m5,
    build_trend_kd_signals_m15,
)
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


def build_signals(profile, ohlc_m15, ohlc_m5):
    sig_cfg = profile.signals
    if isinstance(sig_cfg, LondonBreakoutSignalConfig):
        return build_london_breakout_signals(ohlc_m15, signal_config=sig_cfg, session=profile.session)
    if isinstance(sig_cfg, LiquiditySweepSignalConfig):
        return build_liquidity_sweep_signals(ohlc_m15, signal_config=sig_cfg, session=profile.session)
    if isinstance(sig_cfg, MomentumPinballSignalConfig):
        return build_momentum_pinball_signals_m5(ohlc_m5, ohlc_m15, signal_config=sig_cfg, session=profile.session)
    if profile == EXPERIMENTAL_PROFILE_USDJPY_TREND_KD_M15_V1:
        return build_trend_kd_signals_m15(ohlc_m15, signal_config=profile.signals, session=profile.session)
    raise TypeError(f"Unsupported signal config: {type(sig_cfg)}")


def prepare_events(profiles, ohlc_m15, ohlc_m5):
    events = []
    for idx, profile in enumerate(profiles):
        sigs = build_signals(profile, ohlc_m15, ohlc_m5)
        if sigs is None or sigs.sum() == 0:
            continue
        data = ohlc_m5 if profile.timeframe.startswith("5") else ohlc_m15
        for i, ts in enumerate(sigs.index):
            if sigs.iat[i]:
                events.append((ts, idx, i, data))
    events.sort(key=lambda x: x[0])
    return events, [e[0] for e in events]


def run_stage(events, ts_list, profiles, risk_scales, ohlc_m15, ohlc_m5, start_idx, horizon_days, initial_equity, target_pct, max_loss_pct, daily_loss_pct, risk_control=None):
    if not events or start_idx >= len(events):
        return None
    start_ts = ts_list[start_idx]
    horizon_end = start_ts + timedelta(days=horizon_days)
    end_idx = bisect.bisect_right(ts_list, horizon_end)

    equity = initial_equity
    max_eq = equity
    min_eq = equity
    trades_by_strategy = {}
    num_trades = 0
    verdict = "timeout"
    target_equity = initial_equity * (1 + target_pct)
    loss_equity = initial_equity * (1 - max_loss_pct)
    daily_loss_limit = initial_equity * daily_loss_pct
    daily_pnl = {}
    trades_per_day = {}
    current_scales = dict(risk_scales)
    reduced = False
    last_strategy = None

    for ts, p_idx, entry_idx, data in events[start_idx:end_idx]:
        if ts > horizon_end:
            verdict = "timeout"
            break
        if equity >= target_equity:
            verdict = "pass"
            break
        if equity <= loss_equity:
            equity = loss_equity
            verdict = "fail"
            break

        day = ts.date()
        if daily_pnl.get(day, 0.0) <= -daily_loss_limit:
            continue
        profile = profiles[p_idx]
        key = (day, profile.name)
        mtpd = getattr(profile.signals, "max_trades_per_day", None)
        if mtpd and trades_per_day.get(key, 0) >= mtpd:
            continue

        if risk_control:
            dd = (max_eq - equity) / max_eq if max_eq > 0 else 0
            if (not reduced) and dd >= risk_control.dd_reduce_threshold_pct:
                current_scales = {k: v * risk_control.risk_scale_reduction_factor for k, v in current_scales.items()}
                reduced = True
            elif reduced and dd <= risk_control.dd_restore_threshold_pct:
                current_scales = dict(risk_scales)
                reduced = False

        scale = current_scales.get(profile.name, 1.0)
        cfg = replace(
            profile.strategy,
            risk_per_trade_pct=profile.strategy.risk_per_trade_pct * scale,
            reward_per_trade_pct=profile.strategy.reward_per_trade_pct * scale,
            profit_target_pct=target_pct,
        )
        outcome = simulate_trade_path(
            ohlc=data,
            entry_idx=entry_idx,
            account_balance=equity,
            config=cfg,
            costs=profile.costs,
        )
        equity += outcome.pnl
        num_trades += 1
        last_strategy = profile.name
        trades_by_strategy[profile.name] = trades_by_strategy.get(profile.name, 0) + 1
        trades_per_day[key] = trades_per_day.get(key, 0) + 1
        daily_pnl[day] = daily_pnl.get(day, 0.0) + outcome.pnl
        max_eq = max(max_eq, equity)
        min_eq = min(min_eq, equity)
        if equity >= target_equity:
            verdict = "pass"
            break
        if equity <= loss_equity:
            verdict = "fail"
            break

    total_return = equity / initial_equity - 1.0
    max_dd = (max_eq - min_eq) / max_eq if max_eq > 0 else 0.0
    return {
        "verdict": verdict,
        "total_return": total_return,
        "max_dd": max_dd,
        "num_trades": num_trades,
        "trades_by_strategy": trades_by_strategy,
        "last_strategy": last_strategy,
    }


def main():
    if not init_mt5():
        return
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    end = datetime.now()
    total_history_days = int(os.getenv("FTMO_TOTAL_HISTORY_DAYS", "365"))
    max_days_phase = int(os.getenv("FTMO_MAX_DAYS", "90"))
    num_runs = int(os.getenv("FTMO_NUM_RUNS", "100"))
    initial_equity = 100_000.0
    live_days = int(os.getenv("FTMO_LIVE_DAYS", "30"))

    def fetch_with_backoff(tf):
        for days in [total_history_days, 540, 360, 270, 180]:
            start = end - timedelta(days=days)
            try:
                df = fetch_symbol_ohlc_mt5("USDJPY", tf, start, end)
                if df is not None and not df.empty:
                    return df
            except Exception:
                continue
        return None

    ohlc_m15 = fetch_with_backoff(mt5.TIMEFRAME_M15)
    ohlc_m5 = fetch_with_backoff(mt5.TIMEFRAME_M5)
    if ohlc_m15 is None or ohlc_m5 is None:
        print("Failed to fetch MT5 data.")
        return

    profiles = [
        DEFAULT_PROFILE_USDJPY_LONDON_M15_V1,
        DEFAULT_PROFILE_USDJPY_LIQUI_M15_V1,
        DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1,
        EXPERIMENTAL_PROFILE_USDJPY_TREND_KD_M15_V1,
    ]
    risk_scales = {p.name: EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V2.risk_scales[i] for i, p in enumerate(profiles)}
    rc = EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V2.risk_control

    events, ts_list = prepare_events(profiles, ohlc_m15, ohlc_m5)
    if not events:
        print("No signals found.")
        return

    results = []
    symbol_pass_counts = {}
    strategy_pass_counts = {}

    for i in range(num_runs):
        start_idx = random.randint(0, max(0, len(ts_list) - 1))
        phase1 = run_stage(events, ts_list, profiles, risk_scales, ohlc_m15, ohlc_m5, start_idx, max_days_phase, initial_equity, 0.10, 0.095, 0.05, rc)
        phase2 = None
        live_result = None
        if phase1 and phase1["verdict"] == "pass":
            strategy_pass_counts[phase1.get("last_strategy")] = strategy_pass_counts.get(phase1.get("last_strategy"), 0) + 1
            phase2 = run_stage(events, ts_list, profiles, risk_scales, ohlc_m15, ohlc_m5, start_idx, max_days_phase, initial_equity, 0.05, 0.095, 0.05, rc)
            if phase2 and phase2["verdict"] == "pass":
                live_result = run_stage(events, ts_list, profiles, risk_scales, ohlc_m15, ohlc_m5, start_idx, live_days, initial_equity, 10.0, 0.095, 0.05, rc)
        results.append({"phase1": phase1, "phase2": phase2, "live": live_result})
        print(f"PROGRESS: {i+1}/{num_runs}", flush=True)

    p1_pass = [r for r in results if r["phase1"] and r["phase1"]["verdict"] == "pass"]
    p2_pass = [r for r in p1_pass if r["phase2"] and r["phase2"]["verdict"] == "pass"]
    pipeline_success = [r for r in p2_pass if r.get("live") and r["live"]["verdict"] != "fail"]

    summary = {
        "runs": len(results),
        "phase1": {
            "pass_rate": len(p1_pass) / len(results) if results else 0.0,
        },
        "phase2": {
            "pass_rate": len(p2_pass) / len(p1_pass) if p1_pass else 0.0,
        },
        "pipeline_success_rate": len(pipeline_success) / len(results) if results else 0.0,
        "strategy_pass_counts": strategy_pass_counts,
    }
    if pipeline_success:
        live_returns = [r["live"]["total_return"] for r in pipeline_success if r.get("live")]
        live_dd = [r["live"]["max_dd"] for r in pipeline_success if r.get("live")]
        summary["live"] = {
            "avg_return_live": float(np.mean(live_returns)) if live_returns else None,
            "max_dd_live_avg": float(np.mean(live_dd)) if live_dd else None,
            "max_dd_live_worst": float(max(live_dd)) if live_dd else None,
        }
    out_path = reports_dir / "ftmo_usdjpy_exp_v2_full_pipeline.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote JSON summary to {out_path}")
    if mt5:
        mt5.shutdown()


if __name__ == "__main__":
    main()
