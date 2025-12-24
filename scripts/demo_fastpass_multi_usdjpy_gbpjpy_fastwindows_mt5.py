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
    DEFAULT_PROFILE_USDJPY_LIQUI_M15_V1,
    DEFAULT_PROFILE_USDJPY_LONDON_M15_V1,
    DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1,
    DEFAULT_PROFILE_GBPJPY_LIQUI_M15_V1,
    DEFAULT_PROFILE_GBPJPY_LONDON_M15_V1,
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


def build_signals(profile, ohlc_m15, ohlc_m5):
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


def run_eval(events, profiles, risk_scales, ohlc_map, initial_equity=10_000.0, daily_loss=0.05):
    challenge = DEFAULT_CHALLENGE
    target_equity = initial_equity * (1.0 + challenge.profit_target_pct)
    loss_limit_equity = initial_equity * (1.0 - challenge.max_total_loss_pct)
    daily_loss_limit = initial_equity * daily_loss

    equity = initial_equity
    max_equity = equity
    min_equity = equity
    trades_by_strategy = {}
    trades_by_symbol = {}
    num_trades = 0
    verdict = "data_exhausted"
    first_entry_ts = None
    verdict_exit_ts = None
    trades_per_day_per_strategy = {}
    daily_pnl: dict = {}
    master_index = sorted({ts for ts, *_ in events})

    for ts, profile_idx, entry_idx, symbol_key, data_key in events:
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
        profile = profiles[profile_idx]
        key = (day, profile.name)
        max_trades = getattr(profile.signals, "max_trades_per_day", None)
        if max_trades and trades_per_day_per_strategy.get(key, 0) >= max_trades:
            continue

        scale = risk_scales.get(profile.name, 1.0)
        scaled_strategy = replace(
            profile.strategy,
            risk_per_trade_pct=profile.strategy.risk_per_trade_pct * scale,
            reward_per_trade_pct=profile.strategy.reward_per_trade_pct * scale,
            profit_target_pct=challenge.profit_target_pct,
        )
        data = ohlc_map[symbol_key][data_key]
        try:
            outcome = simulate_trade_path(
                ohlc=data,
                entry_idx=entry_idx,
                account_balance=equity,
                config=scaled_strategy,
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

    total_return = equity / initial_equity - 1.0
    max_dd = (max_equity - min_equity) / max_equity if max_equity > 0 else 0.0
    bars_to_verdict = None
    trades_to_verdict = None
    if first_entry_ts is not None and verdict_exit_ts is not None:
        try:
            start_idx = master_index.index(first_entry_ts)
            end_idx = master_index.index(verdict_exit_ts)
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
        "trades_by_symbol": trades_by_symbol,
        "trades_to_verdict": trades_to_verdict,
        "bars_to_verdict": bars_to_verdict,
    }


def eval_windows(window_days: int, ohlc_map, profiles, risk_scales, num_evals: int, initial_equity: float):
    window_delta = timedelta(days=window_days)
    usdjpy_m15 = ohlc_map["USDJPY"]["m15"]
    gbpjpy_m15 = ohlc_map["GBPJPY"]["m15"]
    actual_start = max(usdjpy_m15.index.min(), gbpjpy_m15.index.min())
    actual_end = min(usdjpy_m15.index.max(), gbpjpy_m15.index.max())

    master_index = sorted(set(usdjpy_m15.index) | set(gbpjpy_m15.index))
    eligible = [ts for ts in master_index if ts >= actual_start and ts + window_delta <= actual_end]
    if not eligible:
        print("No eligible windows for requested history.")
        return None

    evaluations = []
    trades_totals = {}
    symbol_totals = {}

    for i_eval in range(num_evals):
        if (i_eval + 1) % max(1, num_evals // 5) == 0:
            print(f"  Window {i_eval+1}/{num_evals} for {window_days}d")
        start_ts = random.choice(eligible)
        end_ts = start_ts + window_delta
        ohlc_subset = {
            "USDJPY": {
                "m15": ohlc_map["USDJPY"]["m15"].loc[(usdjpy_m15.index >= start_ts) & (usdjpy_m15.index <= end_ts)],
                "m5": ohlc_map["USDJPY"]["m5"].loc[(ohlc_map["USDJPY"]["m5"].index >= start_ts) & (ohlc_map["USDJPY"]["m5"].index <= end_ts)],
            },
            "GBPJPY": {
                "m15": ohlc_map["GBPJPY"]["m15"].loc[(gbpjpy_m15.index >= start_ts) & (gbpjpy_m15.index <= end_ts)],
            },
        }
        events = prepare_events(profiles, ohlc_subset)
        ev = run_eval(events, profiles, risk_scales, ohlc_subset, initial_equity)
        evaluations.append(ev)
        for k, v in ev["trades_by_strategy"].items():
            trades_totals[k] = trades_totals.get(k, 0) + v
        for k, v in ev["trades_by_symbol"].items():
            symbol_totals[k] = symbol_totals.get(k, 0) + v

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
    avg_trades_per_symbol = {k: v / len(evaluations) for k, v in symbol_totals.items()}

    return {
        "passes": len(passes),
        "fails": len(fails),
        "pass_rate": len(passes) / len(evaluations) if evaluations else 0.0,
        "avg_return": avg_return,
        "avg_max_dd": avg_max_dd,
        "avg_trades": avg_trades,
        "avg_trades_per_strategy": avg_trades_per_strategy,
        "avg_trades_per_symbol": avg_trades_per_symbol,
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

    for window_days in (30, 60):
        print(f"\n=== FastPass USDJPY+GBPJPY – {window_days}d window ===")
        result = eval_windows(window_days, ohlc_map, profiles, risk_scales, num_evals, initial_equity)
        if result is None:
            continue
        print(f"Passes: {result['passes']} ({result['pass_rate']:.2%}), Fails: {result['fails']}")
        print(
            f"Avg_return: {result['avg_return']:.2%}, Avg_max_dd: {result['avg_max_dd']:.2%}, "
            f"Avg_trades/eval: {result['avg_trades']:.2f}"
        )
        print(f"Avg trades per symbol: {result['avg_trades_per_symbol']}")
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
