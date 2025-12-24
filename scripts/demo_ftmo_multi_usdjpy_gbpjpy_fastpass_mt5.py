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
    DEFAULT_CHALLENGE,
    DEFAULT_PROFILE_GBPJPY_LIQUI_M15_V1,
    DEFAULT_PROFILE_GBPJPY_LONDON_M15_V1,
    DEFAULT_PROFILE_USDJPY_LIQUI_M15_V1,
    DEFAULT_PROFILE_USDJPY_LONDON_M15_V1,
    DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1,
    MULTI_PORTFOLIO_USDJPY_GBPJPY_FASTPASS,
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


def build_signals(profile, ohlc_m15, ohlc_m5=None):
    sig_cfg = profile.signals
    if isinstance(sig_cfg, LondonBreakoutSignalConfig):
        return build_london_breakout_signals(ohlc_m15, signal_config=sig_cfg, session=profile.session)
    if isinstance(sig_cfg, LiquiditySweepSignalConfig):
        return build_liquidity_sweep_signals(ohlc_m15, signal_config=sig_cfg, session=profile.session)
    if isinstance(sig_cfg, MomentumPinballSignalConfig):
        return build_momentum_pinball_signals_m5(ohlc_m5, ohlc_m15, signal_config=sig_cfg, session=profile.session)
    raise TypeError(f"Unsupported signal config: {type(sig_cfg)}")


def prepare_events(profiles, ohlc_map):
    events = []
    for idx, profile in enumerate(profiles):
        ohlc_m15 = ohlc_map[profile.symbol_key]["m15"]
        ohlc_m5 = ohlc_map[profile.symbol_key].get("m5")
        signals = build_signals(profile, ohlc_m15, ohlc_m5)
        if signals is None or signals.sum() == 0:
            continue
        data_key = "m5" if profile.timeframe.startswith("5") else "m15"
        for i, ts in enumerate(signals.index):
            if signals.iat[i]:
                events.append((ts, idx, i, profile.symbol_key, data_key))
    events.sort(key=lambda x: x[0])
    return events


def run_stage(
    events,
    profiles,
    risk_scales,
    ohlc_map,
    start_idx: int,
    horizon_days: int,
    initial_equity: float,
    target_pct: float,
    max_loss_pct: float,
    daily_loss_pct: float,
):
    if not events or start_idx >= len(events):
        return None

    start_ts = events[start_idx][0]
    horizon_end = start_ts + timedelta(days=horizon_days)
    ts_list = [e[0] for e in events]
    end_idx = bisect.bisect_right(ts_list, horizon_end)

    equity = initial_equity
    max_equity = equity
    min_equity = equity
    trades_by_strategy = {}
    trades_by_symbol = {}
    num_trades = 0
    verdict = "timeout"
    first_entry_ts = None
    verdict_exit_ts = None
    trades_per_day_per_strategy = {}
    trading_days_set = set()
    daily_pnl: dict = {}
    last_symbol = None

    target_equity = initial_equity * (1.0 + target_pct)
    loss_limit_equity = initial_equity * (1.0 - max_loss_pct)
    daily_loss_limit = initial_equity * daily_loss_pct

    for ts, profile_idx, entry_idx, symbol_key, data_key in events[start_idx:end_idx]:
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
        data = ohlc_map[symbol_key][data_key]
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
        trades_by_symbol[symbol_key] = trades_by_symbol.get(symbol_key, 0) + 1
        trades_per_day_per_strategy[key] = trades_per_day_per_strategy.get(key, 0) + 1
        daily_pnl[day] = daily_pnl.get(day, 0.0) + outcome.pnl
        trading_days_set.add(day)
        last_symbol = symbol_key

        if first_entry_ts is None:
            first_entry_ts = ts
        if equity > max_equity:
            max_equity = equity
        if equity < min_equity:
            min_equity = equity

        if equity >= target_equity:
            equity = target_equity
            verdict = "pass"
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
    if verdict in {"pass", "fail"} and first_entry_ts is not None:
        end_ts = verdict_exit_ts or horizon_end
        days_to_verdict = (end_ts - start_ts).total_seconds() / (60 * 60 * 24)
        trades_to_verdict = num_trades

    return {
        "verdict": verdict,
        "end_equity": equity,
        "total_return": total_return,
        "max_drawdown_pct": max_dd,
        "num_trades": num_trades,
        "trades_by_strategy": trades_by_strategy,
        "trades_by_symbol": trades_by_symbol,
        "trades_to_verdict": trades_to_verdict,
        "days_to_verdict": days_to_verdict,
        "trading_days": len(trading_days_set),
        "exit_idx": bisect.bisect_right(ts_list, verdict_exit_ts or horizon_end),
        "last_symbol": last_symbol,
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
    backoff_steps = [total_history_days, 540, 360, 270, 180]

    def fetch_with_backoff(symbol: str, tf):
        for days in backoff_steps:
            start = end - timedelta(days=days)
            try:
                df = fetch_symbol_ohlc_mt5(symbol, tf, start, end)
                if df is not None and not df.empty:
                    return df
            except Exception:
                continue
        return None

    ohlc_m15_usd = fetch_with_backoff("USDJPY", mt5.TIMEFRAME_M15)
    ohlc_m5_usd = fetch_with_backoff("USDJPY", mt5.TIMEFRAME_M5)
    ohlc_m15_gbp = fetch_with_backoff("GBPJPY", mt5.TIMEFRAME_M15)
    if ohlc_m15_usd is None or ohlc_m5_usd is None or ohlc_m15_gbp is None:
        print("Failed to fetch MT5 data.")
        return
    print(f"USDJPY M15: {ohlc_m15_usd.index.min()} -> {ohlc_m15_usd.index.max()} ({len(ohlc_m15_usd)} bars)")
    print(f"USDJPY M5 : {ohlc_m5_usd.index.min()} -> {ohlc_m5_usd.index.max()} ({len(ohlc_m5_usd)} bars)")
    print(f"GBPJPY M15: {ohlc_m15_gbp.index.min()} -> {ohlc_m15_gbp.index.max()} ({len(ohlc_m15_gbp)} bars)")

    ohlc_map = {
        "USDJPY": {"m15": ohlc_m15_usd, "m5": ohlc_m5_usd},
        "GBPJPY": {"m15": ohlc_m15_gbp},
    }

    usd_portfolio = MULTI_PORTFOLIO_USDJPY_GBPJPY_FASTPASS.portfolios[0]
    gbp_portfolio = MULTI_PORTFOLIO_USDJPY_GBPJPY_FASTPASS.portfolios[1]
    symbol_scales = MULTI_PORTFOLIO_USDJPY_GBPJPY_FASTPASS.symbol_risk_scales or [1.0, 1.0]

    profiles = usd_portfolio.strategies + gbp_portfolio.strategies
    risk_scales = {}
    for idx, profile in enumerate(usd_portfolio.strategies):
        scale = (usd_portfolio.risk_scales[idx] if usd_portfolio.risk_scales and idx < len(usd_portfolio.risk_scales) else 1.0) * symbol_scales[0]
        risk_scales[profile.name] = scale
    for idx, profile in enumerate(gbp_portfolio.strategies):
        scale = (gbp_portfolio.risk_scales[idx] if gbp_portfolio.risk_scales and idx < len(gbp_portfolio.risk_scales) else 1.0) * symbol_scales[1]
        risk_scales[profile.name] = scale

    events = prepare_events(profiles, ohlc_map)
    if not events:
        print("No signals found in history window.")
        if mt5:
            mt5.shutdown()
        return
    print(f"Total events: {len(events)}")

    phase1_target = 0.10
    phase2_target = 0.05
    max_loss_pct = 0.095
    daily_loss_pct = 0.05

    results = []
    total_horizon_days = max_days_phase * 2
    ts_list = [e[0] for e in events]
    max_start_ts = ts_list[-1] - timedelta(days=total_horizon_days)
    max_start_idx = bisect.bisect_right(ts_list, max_start_ts) - 1
    if max_start_idx < 0:
        print("Not enough history for requested horizons.")
        return

    symbol_pass_counts = {"USDJPY": 0, "GBPJPY": 0}

    for i in range(num_runs):
        if (i + 1) % max(1, num_runs // 10) == 0:
            print(f"Run {i + 1}/{num_runs}")
        start_idx = random.randint(0, max_start_idx)

        phase1 = run_stage(
            events,
            profiles,
            risk_scales,
            ohlc_map,
            start_idx,
            max_days_phase,
            initial_equity,
            phase1_target,
            max_loss_pct,
            daily_loss_pct,
        )
        if phase1 is None:
            continue

        phase2 = None
        if phase1["verdict"] == "pass":
            phase2 = run_stage(
                events,
                profiles,
                risk_scales,
                ohlc_map,
                phase1["exit_idx"],
                max_days_phase,
                initial_equity,
                phase2_target,
                max_loss_pct,
                daily_loss_pct,
            )

        if phase1["verdict"] == "pass":
            symbol_pass_counts[phase1.get("last_symbol")] = symbol_pass_counts.get(phase1.get("last_symbol"), 0) + 1

        results.append({"phase1": phase1, "phase2": phase2})

    p1_pass = [r for r in results if r["phase1"]["verdict"] == "pass"]
    p1_fail = [r for r in results if r["phase1"]["verdict"] == "fail"]
    p1_timeout = [r for r in results if r["phase1"]["verdict"] == "timeout"]
    p1_early = [r for r in results if r["phase1"]["verdict"] == "early_stop"]

    p2_pass = [r for r in p1_pass if r.get("phase2") and r["phase2"]["verdict"] == "pass"]
    p2_fail = [r for r in p1_pass if r.get("phase2") and r["phase2"]["verdict"] == "fail"]
    p2_timeout = [r for r in p1_pass if r.get("phase2") and r["phase2"]["verdict"] == "timeout"]
    p2_early = [r for r in p1_pass if r.get("phase2") and r["phase2"]["verdict"] == "early_stop"]

    pipeline_success = [r for r in results if r["phase1"]["verdict"] == "pass" and r.get("phase2") and r["phase2"]["verdict"] == "pass"]

    def stats(subset, key, level="phase1"):
        vals = [s[level][key] for s in subset if s.get(level) and s[level].get(key) is not None]
        return summarize(vals)

    print("=== FTMO Multi (USDJPY+GBPJPY) – Pipeline ===")
    print(f"Runs: {len(results)}")
    print("Phase 1:")
    print(
        f"  pass: {len(p1_pass)} ({len(p1_pass)/len(results):.2%}), "
        f"early: {len(p1_early)} ({len(p1_early)/len(results):.2%}), "
        f"fail: {len(p1_fail)} ({len(p1_fail)/len(results):.2%}), "
        f"timeout: {len(p1_timeout)} ({len(p1_timeout)/len(results):.2%})"
    )
    if p1_pass:
        p1_days = stats(p1_pass, "days_to_verdict", "phase1")
        print(
            f"  days_to_pass avg/med/p90: {(p1_days or {}).get('avg', 0):.2f} / "
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
    if p2_pass:
        p2_days = stats(p2_pass, "days_to_verdict", "phase2")
        print(
            f"  days_to_pass avg/med/p90: {(p2_days or {}).get('avg', 0):.2f} / "
            f"{(p2_days or {}).get('median', 0):.2f} / {(p2_days or {}).get('p90', 0):.2f}"
        )

    if results:
        print(f"Pipeline success rate (Phase1 & Phase2 pass): {len(pipeline_success)/len(results):.2%}")

    total_days = []
    total_trades = []
    max_dds = []
    for r in pipeline_success:
        d1 = r["phase1"].get("days_to_verdict") or max_days_phase
        d2 = r["phase2"].get("days_to_verdict") if r.get("phase2") else 0
        t1 = r["phase1"].get("trades_to_verdict") or r["phase1"].get("num_trades", 0)
        t2 = r["phase2"].get("trades_to_verdict") if r.get("phase2") else 0
        total_days.append(d1 + (d2 or 0))
        total_trades.append((t1 or 0) + (t2 or 0))
        max_dds.append(max(r["phase1"]["max_drawdown_pct"], r["phase2"]["max_drawdown_pct"] if r.get("phase2") else 0))

    summary = {
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
        "symbol_pass_counts": symbol_pass_counts,
    }

    if pipeline_success:
        days_stats = summarize(total_days)
        trades_stats = summarize(total_trades)
        summary.update(
            {
                "total_days_to_payout_avg": days_stats.get("avg") if days_stats else None,
                "total_days_to_payout_median": days_stats.get("median") if days_stats else None,
                "total_days_to_payout_p90": days_stats.get("p90") if days_stats else None,
                "trades_to_payout_avg": trades_stats.get("avg") if trades_stats else None,
                "trades_to_payout_median": trades_stats.get("median") if trades_stats else None,
                "trades_to_payout_p90": trades_stats.get("p90") if trades_stats else None,
                "avg_max_dd": float(np.mean(max_dds)) if max_dds else None,
                "worst_max_dd": float(np.max(max_dds)) if max_dds else None,
            }
        )

    out_path = reports_dir / "ftmo_multi_usdjpy_gbpjpy_fastpass.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote JSON summary to {out_path}")

    if mt5:
        mt5.shutdown()


if __name__ == "__main__":
    main()
