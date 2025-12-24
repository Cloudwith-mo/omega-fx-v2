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
        try:
            login_int = int(login)
        except ValueError:
            print("MT5 login must be numeric; please set MT5_LOGIN to your account number.")
            return False
        if not mt5.login(login=login_int, password=password, server=server):
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


def prepare_events(profiles, ohlc_m15: pd.DataFrame, ohlc_m5: pd.DataFrame):
    events: list[tuple[datetime, int, int, str]] = []
    for idx, profile in enumerate(profiles):
        signals = build_signals(profile, ohlc_m15, ohlc_m5)
        if signals is None or signals.sum() == 0:
            continue
        data_key = "m5" if profile.timeframe.startswith("5") else "m15"
        for i, ts in enumerate(signals.index):
            if signals.iat[i]:
                events.append((ts, idx, i, data_key))
    events.sort(key=lambda x: x[0])
    return events, [e[0] for e in events]


def run_stage(
    events,
    ts_list,
    profiles,
    data_map,
    risk_scales,
    start_idx: int,
    horizon_days: int,
    initial_equity: float,
    target_pct: float,
    max_loss_pct: float,
    daily_loss_pct: float,
    early_stop_pct: float | None = None,
    window_start_idx: int | None = None,
    window_end_idx: int | None = None,
):
    if not events or start_idx >= len(events):
        return None

    start_ts = ts_list[start_idx]
    horizon_end = start_ts + timedelta(days=horizon_days)
    end_idx = bisect.bisect_right(ts_list, horizon_end)
    if window_start_idx is not None and window_end_idx is not None:
        # Constrain evaluation to supplied window
        end_idx = min(end_idx, window_end_idx)
        start_idx = max(start_idx, window_start_idx)
        if start_idx >= end_idx:
            return None

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

    target_equity = initial_equity * (1.0 + target_pct)
    loss_limit_equity = initial_equity * (1.0 - max_loss_pct)
    daily_loss_limit = initial_equity * daily_loss_pct

    for ts, profile_idx, entry_idx, data_key in events[start_idx:end_idx]:
        if ts > horizon_end:
            verdict = "timeout"
            break
        if equity >= target_equity:
            verdict = "pass"
            verdict_exit_ts = ts
            break
        if equity <= loss_limit_equity:
            equity = loss_limit_equity
            min_equity = max(min_equity, loss_limit_equity)
            verdict = "fail"
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
            profit_target_pct=target_pct,
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
            verdict = "pass"
            verdict_exit_ts = ts
            break
        if early_stop_pct is not None and equity >= initial_equity * (1.0 + early_stop_pct):
            verdict = "early_stop"
            verdict_exit_ts = ts
            break
        if equity <= loss_limit_equity:
            equity = loss_limit_equity
            min_equity = max(min_equity, loss_limit_equity)
            verdict = "fail"
            verdict_exit_ts = ts
            break

    total_return = equity / initial_equity - 1.0
    max_dd = (initial_equity - min_equity) / initial_equity if initial_equity > 0 else 0.0
    days_to_verdict = None
    trades_to_verdict = None
    if verdict in {"pass", "fail", "early_stop"} and first_entry_ts is not None:
        end_ts = verdict_exit_ts or horizon_end
        days_to_verdict = (end_ts - start_ts).total_seconds() / (60 * 60 * 24)
        trades_to_verdict = num_trades

    next_start_idx = bisect.bisect_right(ts_list, verdict_exit_ts or horizon_end)

    return {
        "verdict": verdict,
        "end_equity": equity,
        "total_return": total_return,
        "max_drawdown_pct": max_dd,
        "num_trades": num_trades,
        "trades_by_strategy": trades_by_strategy,
        "trades_to_verdict": trades_to_verdict,
        "days_to_verdict": days_to_verdict,
        "trading_days": len(trading_days_set),
        "exit_idx": next_start_idx,
    }


def main() -> None:
    if not init_mt5():
        return

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    end = datetime.now()
    total_history_days = int(os.getenv("FTMO_TOTAL_HISTORY_DAYS", os.getenv("TOTAL_HISTORY_DAYS", "365")))
    max_days_phase = int(os.getenv("FTMO_MAX_DAYS", "90"))
    num_runs = int(os.getenv("FTMO_NUM_RUNS", os.getenv("NUM_RUNS", "100")))
    initial_equity = 100_000.0
    live_days = int(os.getenv("FTMO_LIVE_DAYS", "30"))
    live_scale = float(os.getenv("FTMO_LIVE_SCALE", "0.5"))
    early_stop_env = os.getenv("FTMO_EARLY_STOP_PCT", "")
    early_stop_pct = float(early_stop_env) if early_stop_env else None

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
    print(f"M15 history: {ohlc_m15_full.index.min()} -> {ohlc_m15_full.index.max()} | bars={len(ohlc_m15_full)}")
    print(f"M5  history: {ohlc_m5_full.index.min()} -> {ohlc_m5_full.index.max()} | bars={len(ohlc_m5_full)}")

    profiles = [
        DEFAULT_PROFILE_USDJPY_LONDON_M15_V1,
        DEFAULT_PROFILE_USDJPY_LIQUI_M15_V1,
        DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1,
    ]
    base_scales = DEFAULT_PORTFOLIO_USDJPY_FASTPASS_V3.risk_scales or [1.0] * len(profiles)
    risk_scales = {profiles[i].name: base_scales[i] for i in range(len(profiles))}

    events, ts_list = prepare_events(profiles, ohlc_m15_full, ohlc_m5_full)
    print(f"Signal counts: total_events={len(events)} | by profile:")
    if events:
        counts = {}
        for _ts, idx, _entry, _data in events:
            name = profiles[idx].name
            counts[name] = counts.get(name, 0) + 1
        for k, v in counts.items():
            print(f"  {k}: {v}")
    if not events:
        print("No signals found in history window.")
        if mt5:
            mt5.shutdown()
        return
    data_map = {"m15": ohlc_m15_full, "m5": ohlc_m5_full}

    phase1_target = FTMO_CHALLENGE_USDJPY.profit_target_pct
    max_loss_pct = FTMO_CHALLENGE_USDJPY.max_total_loss_pct
    daily_loss_pct = FTMO_CHALLENGE_USDJPY.daily_loss_pct or 0.05
    phase2_target = 0.05

    print(
        f"Starting FTMO pipeline: runs={num_runs}, max_days_per_phase={max_days_phase}, "
        f"total_history_days={total_history_days}, live_days={live_days}"
    )
    sys.stdout.flush()

    total_horizon_days = max_days_phase * 2 + live_days
    max_start_ts = ts_list[-1] - timedelta(days=total_horizon_days)
    max_start_idx = bisect.bisect_right(ts_list, max_start_ts) - 1
    if max_start_idx < 0:
        print("Not enough history for requested horizons.")
        return

    results = []
    for i in range(num_runs):
        print(f"PROGRESS: {i + 1}/{num_runs}")
        sys.stdout.flush()

        start_idx = random.randint(0, max_start_idx)

        phase1 = run_stage(
            events,
            ts_list,
            profiles,
            data_map,
            risk_scales,
            start_idx,
            max_days_phase,
            initial_equity,
            phase1_target,
            max_loss_pct,
            daily_loss_pct,
            early_stop_pct,
        )
        if phase1 is None:
            continue

        phase2 = None
        if phase1["verdict"] == "pass":
            phase2 = run_stage(
                events,
                ts_list,
                profiles,
                data_map,
                risk_scales,
                phase1["exit_idx"],
                max_days_phase,
                initial_equity,
                phase2_target,
                max_loss_pct,
                daily_loss_pct,
                early_stop_pct,
            )

        live_result = None
        if phase1["verdict"] == "pass" and phase2 and phase2["verdict"] == "pass":
            live_scales = {k: v * live_scale for k, v in risk_scales.items()}
            live_result = run_stage(
                events,
                ts_list,
                profiles,
                data_map,
                live_scales,
                phase2["exit_idx"],
                live_days,
                initial_equity,
                target_pct=10.0,  # effectively unreachable, just collect stats
                max_loss_pct=max_loss_pct,
                daily_loss_pct=daily_loss_pct,
                early_stop_pct=None,
            )

        results.append({"phase1": phase1, "phase2": phase2, "live": live_result})

    # Aggregate
    p1_pass = [r for r in results if r["phase1"]["verdict"] == "pass"]
    p1_fail = [r for r in results if r["phase1"]["verdict"] == "fail"]
    p1_timeout = [r for r in results if r["phase1"]["verdict"] == "timeout"]
    p1_early = [r for r in results if r["phase1"]["verdict"] == "early_stop"]

    p2_pass = [r for r in p1_pass if r.get("phase2") and r["phase2"]["verdict"] == "pass"]
    p2_fail = [r for r in p1_pass if r.get("phase2") and r["phase2"]["verdict"] == "fail"]
    p2_timeout = [r for r in p1_pass if r.get("phase2") and r["phase2"]["verdict"] == "timeout"]
    p2_early = [r for r in p1_pass if r.get("phase2") and r["phase2"]["verdict"] == "early_stop"]

    pipeline_success = [r for r in results if r["phase1"]["verdict"] == "pass" and r.get("phase2") and r["phase2"]["verdict"] == "pass"]

    live_ok = [r for r in pipeline_success if r.get("live") and r["live"]["verdict"] != "fail"]

    def stats(subset, key, level="phase1"):
        vals = [s[level][key] for s in subset if s.get(level) and s[level].get(key) is not None]
        return summarize(vals)

    print("=== FTMO USDJPY V3 – Full Pipeline (Phase1 + Phase2 + First Payout) ===")
    print(f"Runs: {len(results)}")
    print("Phase 1:")
    if results:
        print(
            f"  pass: {len(p1_pass)} ({len(p1_pass)/len(results):.2%}), "
            f"early: {len(p1_early)} ({len(p1_early)/len(results):.2%}), "
            f"fail: {len(p1_fail)} ({len(p1_fail)/len(results):.2%}), "
            f"timeout: {len(p1_timeout)} ({len(p1_timeout)/len(results):.2%})"
        )
    p1_days = stats(results, "days_to_verdict", "phase1")
    if p1_days:
        print(
            f"  days_to_verdict avg/med/p90: {(p1_days or {}).get('avg', 0):.2f} / "
            f"{(p1_days or {}).get('median', 0):.2f} / {(p1_days or {}).get('p90', 0):.2f}"
        )

    print("Phase 2 (conditional on Phase 1 pass):")
    if p1_pass:
        print(
            f"  pass: {len(p2_pass)} ({len(p2_pass)/len(p1_pass):.2%}), "
            f"early: {len(p2_early)} ({len(p2_early)/len(p1_pass):.2%}), "
            f"fail: {len(p2_fail)} ({len(p2_fail)/len(p1_pass):.2%}), "
            f"timeout: {len(p2_timeout)} ({len(p2_timeout)/len(p1_pass):.2%})"
        )
    p2_days = stats(p2_pass, "days_to_verdict", "phase2") if p2_pass else None
    if p2_days:
        print(
            f"  days_to_pass avg/med/p90: {(p2_days or {}).get('avg', 0):.2f} / "
            f"{(p2_days or {}).get('median', 0):.2f} / {(p2_days or {}).get('p90', 0):.2f}"
        )

    if results:
        print(f"Pipeline success rate (Phase1 & Phase2 pass): {len(pipeline_success)/len(results):.2%}")

    json_summary: dict = {
        "runs": len(results),
        "phase1": {
            "pass_rate": len(p1_pass) / len(results) if results else 0.0,
            "fail_rate": len(p1_fail) / len(results) if results else 0.0,
            "timeout_rate": len(p1_timeout) / len(results) if results else 0.0,
            "early_rate": len(p1_early) / len(results) if results else 0.0,
        },
        "phase2": {
            "pass_rate": len(p2_pass) / len(p1_pass) if p1_pass else 0.0,
            "fail_rate": len(p2_fail) / len(p1_pass) if p1_pass else 0.0,
            "timeout_rate": len(p2_timeout) / len(p1_pass) if p1_pass else 0.0,
            "early_rate": len(p2_early) / len(p1_pass) if p1_pass else 0.0,
        },
        "pipeline_success_rate": len(pipeline_success) / len(results) if results else 0.0,
    }

    if live_ok:
        live_returns = [r["live"]["total_return"] for r in live_ok if r.get("live")]
        live_dd = [r["live"]["max_drawdown_pct"] for r in live_ok if r.get("live")]
        total_days = []
        total_trades = []
        for r in live_ok:
            d1 = r["phase1"].get("days_to_verdict") or max_days_phase
            d2 = r["phase2"].get("days_to_verdict") if r.get("phase2") else 0
            d3 = r["live"].get("days_to_verdict") or live_days
            t1 = r["phase1"].get("trades_to_verdict") or r["phase1"].get("num_trades", 0)
            t2 = r["phase2"].get("trades_to_verdict") if r.get("phase2") else 0
            t3 = r["live"].get("trades_to_verdict") or r["live"].get("num_trades", 0)
            total_days.append(d1 + (d2 or 0) + d3)
            total_trades.append((t1 or 0) + (t2 or 0) + (t3 or 0))

        live_days_stats = summarize([r["live"]["days_to_verdict"] or live_days for r in live_ok])
        json_summary["live"] = {
            "avg_return_live": float(np.mean(live_returns)),
            "max_dd_live_avg": float(np.mean(live_dd)),
            "max_dd_live_worst": float(max(live_dd)),
        }
        days_stats = summarize(total_days)
        trades_stats = summarize(total_trades)
        if days_stats:
            json_summary["total_days_to_payout_avg"] = days_stats.get("avg")
            json_summary["total_days_to_payout_median"] = days_stats.get("median")
            json_summary["total_days_to_payout_p90"] = days_stats.get("p90")
        if trades_stats:
            json_summary["trades_to_payout_avg"] = trades_stats.get("avg")
            json_summary["trades_to_payout_median"] = trades_stats.get("median")
            json_summary["trades_to_payout_p90"] = trades_stats.get("p90")

        print("First funded month (for pipeline successes):")
        print(
            f"  avg_live_return: {np.mean(live_returns):.2%} | "
            f"avg_live_max_dd: {np.mean(live_dd):.2%} | "
            f"worst_live_dd: {max(live_dd):.2%}"
        )
        if live_days_stats:
            print(
                f"  live days avg/med/p90: {(live_days_stats or {}).get('avg', 0):.2f} / "
                f"{(live_days_stats or {}).get('median', 0):.2f} / {(live_days_stats or {}).get('p90', 0):.2f}"
            )
        days_stats = summarize(total_days)
        trades_stats = summarize(total_trades)
        print("First payout timing (pipeline successes):")
        if days_stats:
            print(
                f"  total days to payout avg/med/p90: {(days_stats or {}).get('avg', 0):.2f} / "
                f"{(days_stats or {}).get('median', 0):.2f} / {(days_stats or {}).get('p90', 0):.2f}"
            )
        if trades_stats:
            print(
                f"  trades to payout avg/med/p90: {(trades_stats or {}).get('avg', 0):.2f} / "
                f"{(trades_stats or {}).get('median', 0):.2f} / {(trades_stats or {}).get('p90', 0):.2f}"
            )

    out_path = reports_dir / "ftmo_usdjpy_v3_full_pipeline.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)
    print(f"\nWrote JSON summary to {out_path}")

    if mt5:
        mt5.shutdown()


if __name__ == "__main__":
    main()
