import json
import os
import random
import traceback
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
    DEFAULT_PORTFOLIO_GBPJPY_FASTPASS_CORE,
    DEFAULT_PROFILE_GBPJPY_LIQUI_M15_V1,
    DEFAULT_PROFILE_GBPJPY_LONDON_M15_V1,
    LondonBreakoutSignalConfig,
    LiquiditySweepSignalConfig,
)
from omegafx_v2.mt5_adapter import fetch_symbol_ohlc_mt5
from omegafx_v2.signals import (
    build_liquidity_sweep_signals,
    build_london_breakout_signals,
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


def log_error(msg: str, exc: Exception | None = None):
    log_path = ROOT / "logs" / "gbpjpy_fastpass_core_errors.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")
        if exc:
            f.write("".join(traceback.format_exception(exc)))


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


def build_signals_for_profile(profile, ohlc_m15):
    sig_cfg = profile.signals
    if isinstance(sig_cfg, LondonBreakoutSignalConfig):
        return build_london_breakout_signals(ohlc_m15, signal_config=sig_cfg, session=profile.session)
    if isinstance(sig_cfg, LiquiditySweepSignalConfig):
        return build_liquidity_sweep_signals(ohlc_m15, signal_config=sig_cfg, session=profile.session)
    raise TypeError(f"Unsupported signal config: {type(sig_cfg)}")


def run_multi_strategy_eval(ohlc_m15, profiles, risk_scales, initial_equity=10_000.0):
    challenge = DEFAULT_CHALLENGE
    target_equity = initial_equity * (1.0 + challenge.profit_target_pct)
    loss_limit_equity = initial_equity * (1.0 - challenge.max_total_loss_pct)
    daily_loss_limit = initial_equity * float(os.getenv("OMEGAFX_DAILY_LOSS_PCT", "0.05"))

    events = []
    for profile in profiles:
        signals = build_signals_for_profile(profile, ohlc_m15)
        if signals is None or signals.sum() == 0:
            continue
        data = ohlc_m15
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
            master_index = ohlc_m15.index
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


def eval_windows(window_days: int, ohlc_m15_full, profiles, risk_scales, num_evals: int, initial_equity: float):
    window_delta = timedelta(days=window_days)
    actual_start = ohlc_m15_full.index.min()
    actual_end = ohlc_m15_full.index.max()

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
            print(f"  Window {i_eval+1}/{num_evals} for {window_days}d")
        start_idx = random.choice(eligible_indices)
        start_ts = master_index[start_idx]
        end_ts = start_ts + window_delta

        ohlc_m15 = ohlc_m15_full.loc[(ohlc_m15_full.index >= start_ts) & (ohlc_m15_full.index <= end_ts)]

        ev = run_multi_strategy_eval(
            ohlc_m15=ohlc_m15,
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
    try:
        if not init_mt5():
            return

        end = datetime.now()
        total_history_days = int(os.getenv("TOTAL_HISTORY_DAYS", "270"))
        backoff_steps = [total_history_days, 180, 120, 90, 60]
        num_evals = int(os.getenv("NUM_EVALS", "200"))
        initial_equity = 10_000.0
        symbol_base = os.getenv("GBPJPY_SYMBOL", "GBPJPY")

        def resolve_symbol():
            if mt5 is None:
                return None
            # Direct try
            if mt5.symbol_select(symbol_base, True):
                return symbol_base
            # Try to find a suffixed symbol
            candidates = [s.name for s in mt5.symbols_get() if s.name.upper().startswith("GBPJPY")]
            for c in candidates:
                if mt5.symbol_select(c, True):
                    return c
            return None

        symbol = resolve_symbol()
        if symbol is None:
            msg = f"GBPJPY symbol not found/selected (base {symbol_base})."
            print(f"GBPJPY fastwindows FAILED: {msg}")
            log_error(msg)
            return
        if symbol != symbol_base:
            print(f"Using MT5 symbol: {symbol}")

        def fetch_with_backoff(tf):
            for days in backoff_steps:
                start = end - timedelta(days=days)
                try:
                    df = fetch_symbol_ohlc_mt5(symbol, tf, start, end)
                    if df is not None and not df.empty:
                        return df
                except Exception:
                    continue
            return None

        ohlc_m15_full = fetch_with_backoff(mt5.TIMEFRAME_M15)
        if ohlc_m15_full is None:
            msg = "Failed to fetch MT5 data or insufficient history."
            print(f"GBPJPY fastwindows FAILED: {msg}")
            log_error(msg)
            return

        available_days = (ohlc_m15_full.index.max() - ohlc_m15_full.index.min()).days
        if available_days < 60:
            msg = f"Insufficient history: only {available_days} days available."
            print(f"GBPJPY fastwindows FAILED: {msg}")
            log_error(msg)
            return
        margin = 5
        effective_history_days = min(total_history_days, max(60, available_days - margin))
        print(f"GBPJPY history available ~{available_days}d; using effective_history_days={effective_history_days}")

        cutoff_start = ohlc_m15_full.index.max() - timedelta(days=effective_history_days)
        ohlc_m15_full = ohlc_m15_full.loc[ohlc_m15_full.index >= cutoff_start]

        profiles = [
            DEFAULT_PROFILE_GBPJPY_LONDON_M15_V1,
            DEFAULT_PROFILE_GBPJPY_LIQUI_M15_V1,
        ]
        risk_scales_list = DEFAULT_PORTFOLIO_GBPJPY_FASTPASS_CORE.risk_scales or [1.0] * len(profiles)
        risk_scales = {profiles[i].name: (risk_scales_list[i] if i < len(risk_scales_list) else 1.0) for i in range(len(profiles))}

        report = {}

        for window_days in (30, 60):
            print(f"\n=== GBPJPY FastPass Core – {window_days}d window ===")
            result = eval_windows(window_days, ohlc_m15_full, profiles, risk_scales, num_evals, initial_equity)
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

            key = f"{window_days}d"
            report[key] = {
                "pass_rate": result["pass_rate"],
                "avg_return": result["avg_return"],
                "avg_max_dd": result["avg_max_dd"],
                "avg_trades_per_eval": result["avg_trades"],
                "avg_trades_per_strategy": result["avg_trades_per_strategy"],
                "pass_days": result["pass_days_stats"],
                "fail_days": result["fail_days_stats"],
            }

        if report:
            out_path = ROOT / "reports" / "fastpass_gbpjpy_core_fastwindows.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"Wrote GBPJPY fastwindows report to {out_path}")

    except Exception as exc:
        msg = f"Unhandled error: {exc}"
        print(f"GBPJPY fastwindows FAILED: {msg}")
        log_error(msg, exc)
    finally:
        if mt5:
            mt5.shutdown()


if __name__ == "__main__":
    main()
