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
    MULTI_PORTFOLIO_USDJPY_EXP_GBPJPY_CORE,
    EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V3,
    DEFAULT_PROFILE_USDJPY_LONDON_M15_V1,
    DEFAULT_PROFILE_USDJPY_LIQUI_M15_V1,
    DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1,
    EXPERIMENTAL_PROFILE_USDJPY_TREND_KD_M15_V1,
    EXPERIMENTAL_PROFILE_USDJPY_VANVLEET_M15_V1,
    EXPERIMENTAL_PROFILE_USDJPY_BIGMAN_H1_V1,
    LondonBreakoutSignalConfig,
    LiquiditySweepSignalConfig,
    MomentumPinballSignalConfig,
    TrendKDSignalConfig,
    VanVleetSignalConfig,
    BigManSignalConfig,
)
from omegafx_v2.mt5_adapter import fetch_symbol_ohlc_mt5
from omegafx_v2.signals import (
    build_london_breakout_signals,
    build_liquidity_sweep_signals,
    build_momentum_pinball_signals_m5,
    build_trend_kd_signals_m15,
    build_vanvleet_signals_m15,
    build_bigman_signals_h1,
    compute_atr,
)
from omegafx_v2.regime import tag_regimes
from omegafx_v2.sim import simulate_trade_path


def init_mt5():
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
    cfg = getattr(profile, "edge_regime_config", None)
    if cfg is None:
        return signals
    allowed_map = {
        "low_vol_range": cfg.allow_in_low_vol,
        "high_vol_trend": cfg.allow_in_high_vol_trend,
        "high_vol_reversal": cfg.allow_in_high_vol_reversal,
        "chop": cfg.allow_in_chop,
    }
    mask = regime_series.map(lambda r: allowed_map.get(r, False))
    mask = mask.reindex(signals.index, method="ffill").fillna(False)
    return signals & mask


def build_signals(profile, ohlc_m15, ohlc_m5, regime_series, allowed_mask=None):
    sig_cfg = profile.signals
    if isinstance(sig_cfg, LondonBreakoutSignalConfig):
        sig = build_london_breakout_signals(ohlc_m15, signal_config=sig_cfg, session=profile.session)
        if allowed_mask is not None:
            sig = sig & allowed_mask(ohlc_m15.index, profile.name)
        return apply_regime_mask(sig, regime_series, profile)
    if isinstance(sig_cfg, LiquiditySweepSignalConfig):
        sig = build_liquidity_sweep_signals(ohlc_m15, signal_config=sig_cfg, session=profile.session)
        if allowed_mask is not None:
            sig = sig & allowed_mask(ohlc_m15.index, profile.name)
        return apply_regime_mask(sig, regime_series, profile)
    if isinstance(sig_cfg, MomentumPinballSignalConfig):
        sig = build_momentum_pinball_signals_m5(ohlc_m5, ohlc_m15, signal_config=sig_cfg, session=profile.session)
        if allowed_mask is not None:
            sig = sig & allowed_mask(sig.index, profile.name)
        return apply_regime_mask(sig, regime_series.reindex(sig.index, method="ffill"), profile)
    if isinstance(sig_cfg, TrendKDSignalConfig):
        sig = build_trend_kd_signals_m15(ohlc_m15, signal_config=sig_cfg, session=profile.session)
        if allowed_mask is not None:
            sig = sig & allowed_mask(ohlc_m15.index, profile.name)
        return apply_regime_mask(sig, regime_series, profile)
    if isinstance(sig_cfg, VanVleetSignalConfig):
        sig = build_vanvleet_signals_m15(ohlc_m15, signal_config=sig_cfg, session=profile.session)
        if allowed_mask is not None:
            sig = sig & allowed_mask(ohlc_m15.index, profile.name)
        return apply_regime_mask(sig, regime_series, profile)
    if isinstance(sig_cfg, BigManSignalConfig):
        sig = build_bigman_signals_h1(ohlc_m15, signal_config=sig_cfg, session=profile.session)
        if allowed_mask is not None:
            sig = sig & allowed_mask(ohlc_m15.index, profile.name)
        return apply_regime_mask(sig, regime_series, profile)
    raise TypeError(f"Unsupported signal config: {type(sig_cfg)}")


def prepare_events(profiles, ohlc_map):
    # only tagging USDJPY M15 for regimes; apply to all USDJPY signals; GBPJPY unchanged
    ohlc_m15_usd = ohlc_map["USDJPY"]["m15"]
    regimes = tag_regimes(ohlc_m15_usd, compute_atr(ohlc_m15_usd, period=14), ohlc_m15_usd["close"].ewm(span=50, adjust=False).mean().diff())
    allowed_by_regime = {
        "high_vol_trend": {"USDJPY_M15_TrendKD_V3", "USDJPY_M15_LondonBreakout_V3"},
        "high_vol_reversal": {"USDJPY_M15_LiquiditySweep_V3"},
        "low_vol_range": {"USDJPY_H1_BigMan_V1", "USDJPY_M15_VanVleet_V1"},
        "chop": set(),
    }

    def allowed_mask(index, profile_name):
        allowed_regimes = [r for r, names in allowed_by_regime.items() if profile_name in names]
        if not allowed_regimes:
            return pd.Series(False, index=index)
        mask = regimes.isin(allowed_regimes)
        return mask.reindex(index, method="ffill").fillna(False)

    events = []
    ts_list = set()
    for profile in profiles:
        if profile.symbol_key == "USDJPY":
            ohlc_m15 = ohlc_map["USDJPY"]["m15"]
            ohlc_m5 = ohlc_map["USDJPY"].get("m5")
            sig = build_signals(profile, ohlc_m15, ohlc_m5, regimes, allowed_mask=allowed_mask)
            data = ohlc_m5 if profile.timeframe.lower().startswith("5") else ohlc_m15
        else:
            ohlc_m15 = ohlc_map[profile.symbol_key]["m15"]
            sig = build_signals(profile, ohlc_m15, None, regimes.reindex(ohlc_m15.index, method="ffill"), allowed_mask=lambda idx, name: pd.Series(True, index=idx))
            data = ohlc_m15
        if sig is None or sig.sum() == 0:
            continue
        for idx, ts in enumerate(sig.index):
            if sig.iat[idx]:
                events.append((ts, profile, idx, data))
                ts_list.add(ts)
    events.sort(key=lambda x: x[0])
    ts_sorted = sorted(ts_list)
    return events, ts_sorted


def run_stage(events, ts_list, profiles, risk_scales, ohlc_map, start_idx, max_days, initial_equity, target_pct, max_loss_pct, daily_loss_pct, symbol_scales=None, risk_control=None):
    start_ts = ts_list[start_idx]
    end_ts = start_ts + timedelta(days=max_days)
    target_eq = initial_equity * (1 + target_pct)
    loss_eq = initial_equity * (1 - max_loss_pct)
    daily_limit = initial_equity * daily_loss_pct

    equity = initial_equity
    max_eq = equity
    min_eq = equity
    trades_by_strategy = {}
    trades_per_day: dict = {}
    daily_pnl: dict = {}
    num_trades = 0
    verdict = "timeout"
    last_symbol = None
    last_strategy = None
    current_scales = dict(risk_scales)
    reduced = False

    for ts, profile, idx, data in events:
        if ts < start_ts or ts > end_ts:
            continue
        if equity >= target_eq:
            verdict = "pass"
            break
        if equity <= loss_eq:
            verdict = "fail"
            break
        day = ts.date()
        if daily_pnl.get(day, 0.0) <= -daily_limit:
            continue
        key = (day, profile.name)
        max_trades = getattr(profile.signals, "max_trades_per_day", None)
        if max_trades:
            current = trades_per_day.get(key, 0)
            if current >= max_trades:
                continue

        peak = max_eq
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd >= 0.07:
            verdict = "fail"
            verdict_exit_ts = ts
            break
        if dd >= 0.06 and profile.symbol_key == "USDJPY":
            if profile.name not in {"USDJPY_M15_LondonBreakout_V3", "USDJPY_M15_VanVleet_V1"}:
                continue
        if risk_control and profile.symbol_key == "USDJPY":
            if dd >= 0.04 and not reduced:
                current_scales = {k: v * 0.3 for k, v in current_scales.items()}
                reduced = True
            elif dd < 0.04 and reduced:
                current_scales = dict(risk_scales)
                reduced = False

        scale = current_scales.get(profile.name, 1.0)
        sym_scale = symbol_scales.get(profile.symbol_key, 1.0) if symbol_scales else 1.0
        cfg = replace(profile.strategy, risk_per_trade_pct=profile.strategy.risk_per_trade_pct * scale * sym_scale, reward_per_trade_pct=profile.strategy.reward_per_trade_pct * scale * sym_scale, profit_target_pct=target_pct)
        outcome = simulate_trade_path(ohlc=data, entry_idx=idx, account_balance=equity, config=cfg, costs=profile.costs)
        equity += outcome.pnl
        num_trades += 1
        last_strategy = profile.name
        last_symbol = profile.symbol_key
        trades_by_strategy[profile.name] = trades_by_strategy.get(profile.name, 0) + 1
        trades_per_day[key] = trades_per_day.get(key, 0) + 1
        daily_pnl[day] = daily_pnl.get(day, 0.0) + outcome.pnl
        max_eq = max(max_eq, equity)
        min_eq = min(min_eq, equity)
    total_return = equity / initial_equity - 1.0
    max_dd = (max_eq - min_eq) / max_eq if max_eq > 0 else 0.0
    return {
        "verdict": verdict,
        "total_return": total_return,
        "max_dd": max_dd,
        "num_trades": num_trades,
        "trades_by_strategy": trades_by_strategy,
        "last_symbol": last_symbol,
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

    def fetch_with_backoff(symbol: str, tf):
        for days in [total_history_days, 540, 360, 270, 180]:
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

    usd_portfolio = EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V3
    gbp_portfolio = MULTI_PORTFOLIO_USDJPY_EXP_GBPJPY_CORE.portfolios[1]
    symbol_scales = {"USDJPY": MULTI_PORTFOLIO_USDJPY_EXP_GBPJPY_CORE.symbol_risk_scales[0], "GBPJPY": MULTI_PORTFOLIO_USDJPY_EXP_GBPJPY_CORE.symbol_risk_scales[1]}

    profiles = usd_portfolio.strategies + gbp_portfolio.strategies
    risk_scales = {}
    for idx, profile in enumerate(usd_portfolio.strategies):
        scale = (usd_portfolio.risk_scales[idx] if usd_portfolio.risk_scales and idx < len(usd_portfolio.risk_scales) else 1.0)
        risk_scales[profile.name] = scale
    for idx, profile in enumerate(gbp_portfolio.strategies):
        scale = (gbp_portfolio.risk_scales[idx] if gbp_portfolio.risk_scales and idx < len(gbp_portfolio.risk_scales) else 1.0)
        risk_scales[profile.name] = scale

    events, ts_list = prepare_events(profiles, ohlc_map)
    if not events:
        print("No signals found in history window.")
        return

    results = []
    symbol_pass_counts = {}
    strategy_pass_counts = {}
    trades_per_strategy = {}
    for i in range(num_runs):
        start_idx = random.randint(0, max(0, len(ts_list) - 1))
        phase1 = run_stage(events, ts_list, profiles, risk_scales, ohlc_map, start_idx, max_days_phase, initial_equity, 0.10, 0.095, 0.05, symbol_scales=symbol_scales, risk_control=usd_portfolio.risk_control)
        phase2 = None
        live_result = None
        if phase1 and phase1["verdict"] == "pass":
            strategy_pass_counts[phase1.get("last_strategy")] = strategy_pass_counts.get(phase1.get("last_strategy"), 0) + 1
            symbol_pass_counts[phase1.get("last_symbol")] = symbol_pass_counts.get(phase1.get("last_symbol"), 0) + 1
            phase2 = run_stage(events, ts_list, profiles, risk_scales, ohlc_map, start_idx, max_days_phase, initial_equity, 0.05, 0.095, 0.05, symbol_scales=symbol_scales, risk_control=usd_portfolio.risk_control)
            if phase2 and phase2["verdict"] == "pass":
                live_result = run_stage(events, ts_list, profiles, risk_scales, ohlc_map, start_idx, live_days, initial_equity, 10.0, 0.095, 0.05, symbol_scales=symbol_scales, risk_control=usd_portfolio.risk_control)
        results.append({"phase1": phase1, "phase2": phase2, "live": live_result})
        for stage in [phase1, phase2, live_result]:
            if stage and stage.get("trades_by_strategy"):
                for k, v in stage["trades_by_strategy"].items():
                    trades_per_strategy[k] = trades_per_strategy.get(k, 0) + v
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
        "symbol_pass_counts": symbol_pass_counts,
        "strategy_pass_counts": strategy_pass_counts,
        "trades_per_strategy": trades_per_strategy,
    }
    if pipeline_success:
        live_returns = [r["live"]["total_return"] for r in pipeline_success if r.get("live")]
        live_dd = [r["live"]["max_dd"] for r in pipeline_success if r.get("live")]
        summary["live"] = {
            "avg_return_live": float(np.mean(live_returns)) if live_returns else None,
            "max_dd_live_avg": float(np.mean(live_dd)) if live_dd else None,
            "max_dd_live_worst": float(max(live_dd)) if live_dd else None,
        }
        days = []
        for r in pipeline_success:
            d1 = max_days_phase
            d2 = max_days_phase
            d3 = live_days
            days.append(d1 + d2 + d3)
        if days:
            ds = summarize(days)
            summary["total_days_to_payout_avg"] = ds.get("avg")
            summary["total_days_to_payout_median"] = ds.get("median")
            summary["total_days_to_payout_p90"] = ds.get("p90")

    out_path = reports_dir / "ftmo_multi_usdjpy_exp_v3_full_pipeline.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote JSON summary to {out_path}")
    if mt5:
        mt5.shutdown()


if __name__ == "__main__":
    main()
