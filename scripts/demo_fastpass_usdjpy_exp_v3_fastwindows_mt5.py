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
    DEFAULT_PROFILE_USDJPY_LONDON_M15_V1,
    DEFAULT_PROFILE_USDJPY_LIQUI_M15_V1,
    DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1,
    EXPERIMENTAL_PROFILE_USDJPY_TREND_KD_M15_V1,
    EXPERIMENTAL_PROFILE_USDJPY_VANVLEET_M15_V1,
    EXPERIMENTAL_PROFILE_USDJPY_BIGMAN_H1_V1,
    EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V3,
    LondonBreakoutSignalConfig,
    LiquiditySweepSignalConfig,
    MomentumPinballSignalConfig,
    TrendKDSignalConfig,
    VanVleetSignalConfig,
    BigManSignalConfig,
)
from omegafx_v2.mt5_adapter import fetch_symbol_ohlc_mt5
from omegafx_v2.signals import (
    build_liquidity_sweep_signals,
    build_london_breakout_signals,
    build_momentum_pinball_signals_m5,
    build_trend_kd_signals_m15,
    build_vanvleet_signals_m15,
    build_bigman_signals_h1,
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


def build_signals_for_profile(profile, ohlc_m15, ohlc_m5, regime_series, allowed_mask=None):
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
        if allowed_mask is not None:
            sig = sig & allowed_mask(ohlc_m5.index, profile.name)
        return sig
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


def run_multi_strategy_eval(ohlc_m15, ohlc_m5, profiles, risk_scales, initial_equity=10_000.0, risk_control=None):
    master_index = ohlc_m15.index
    challenge = DEFAULT_CHALLENGE
    target_equity = initial_equity * (1.0 + challenge.profit_target_pct)
    loss_limit_equity = initial_equity * (1.0 - challenge.max_total_loss_pct)
    daily_loss_limit = initial_equity * float(os.getenv("OMEGAFX_DAILY_LOSS_PCT", "0.05"))

    regimes = tag_regimes(ohlc_m15, compute_atr(ohlc_m15, period=14), ohlc_m15["close"].ewm(span=50, adjust=False).mean().diff())

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
    for profile in profiles:
        signals = build_signals_for_profile(profile, ohlc_m15, ohlc_m5, regimes, allowed_mask=allowed_mask)
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

        peak = max_equity
        dd = (peak - equity) / peak if peak > 0 else 0
        # Draymond rules for EXP_V3
        if dd >= 0.07:
            verdict = "max_loss_breached"
            verdict_exit_ts = ts
            break
        if dd >= 0.06:
            # allow only London + VanVleet
            if profile.name not in {"USDJPY_M15_LondonBreakout_V3", "USDJPY_M15_VanVleet_V1"}:
                continue
        if dd >= 0.04 and not reduced:
            current_scales = {k: v * 0.3 for k, v in current_scales.items()}
            reduced = True
        elif dd < 0.04 and reduced:
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
        if first_entry_ts is None:
            first_entry_ts = ts
        equity += outcome.pnl
        num_trades += 1
        trades_by_strategy[profile.name] = trades_by_strategy.get(profile.name, 0) + 1
        trades_per_day_per_strategy[key] = trades_per_day_per_strategy.get(key, 0) + 1
        daily_pnl[day] = daily_pnl.get(day, 0.0) + outcome.pnl
        max_equity = max(max_equity, equity)
        min_equity = min(min_equity, equity)
    if verdict_exit_ts is None:
        verdict_exit_ts = ohlc_m15.index[-1] if len(ohlc_m15) else None

    total_return = equity / initial_equity - 1.0
    max_dd = (max_equity - min_equity) / max_equity if max_equity > 0 else 0.0
    bars_to_verdict = None
    trades_to_verdict = None
    if first_entry_ts and verdict_exit_ts:
        def _safe_loc(idx, ts):
            try:
                return idx.get_loc(ts)
            except Exception:
                pos = idx.get_indexer([ts], method="nearest")
                return pos[0] if pos is not None and len(pos) and pos[0] != -1 else None
        i_end = _safe_loc(ohlc_m15.index, verdict_exit_ts)
        i_start = _safe_loc(ohlc_m15.index, first_entry_ts)
        if i_end is not None and i_start is not None:
            bars_to_verdict = (i_end - i_start) + 1
            trades_to_verdict = num_trades
    return {
        "verdict": verdict,
        "total_return": total_return,
        "max_dd": max_dd,
        "num_trades": num_trades,
        "trades_by_strategy": trades_by_strategy,
        "bars_to_verdict": bars_to_verdict,
        "trades_to_verdict": trades_to_verdict,
    }


def main():
    if not init_mt5():
        return
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    end = datetime.now()
    total_history_days = int(os.getenv("TOTAL_HISTORY_DAYS", "270"))
    start = end - timedelta(days=total_history_days)
    num_evals = int(os.getenv("NUM_EVALS", "200"))

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
        EXPERIMENTAL_PROFILE_USDJPY_VANVLEET_M15_V1,
        EXPERIMENTAL_PROFILE_USDJPY_BIGMAN_H1_V1,
    ]
    risk_scales = {p.name: EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V3.risk_scales[i] for i, p in enumerate(profiles)}
    rc = EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V3.risk_control

    def eval_window(days: int):
        window_delta = timedelta(days=days)
        master_index = ohlc_m15.index
        eligible = [i for i, ts in enumerate(master_index) if ts + window_delta <= master_index[-1]]
        if not eligible:
            return None
        returns = []
        dds = []
        trades = []
        trades_by = []
        bars_to = []
        trades_to = []
        for _ in range(num_evals):
            start_idx = random.choice(eligible)
            start_ts = master_index[start_idx]
            end_ts = start_ts + window_delta
            ohlc_m15_w = ohlc_m15.loc[start_ts:end_ts]
            ohlc_m5_w = ohlc_m5.loc[start_ts:end_ts]
            res = run_multi_strategy_eval(ohlc_m15_w, ohlc_m5_w, profiles, risk_scales, initial_equity=10_000.0, risk_control=rc)
            returns.append(res["total_return"])
            dds.append(res["max_dd"])
            trades.append(res["num_trades"])
            trades_by.append(res["trades_by_strategy"])
            if res.get("bars_to_verdict") is not None:
                bars_to.append(res["bars_to_verdict"])
            if res.get("trades_to_verdict") is not None:
                trades_to.append(res["trades_to_verdict"])
        pass_rate = sum(1 for r in returns if r >= DEFAULT_CHALLENGE.profit_target_pct) / len(returns)
        avg_ret = float(np.mean(returns))
        avg_dd = float(np.mean(dds))
        avg_trades = float(np.mean(trades))
        trades_by_avg = {}
        for d in trades_by:
            for k, v in d.items():
                trades_by_avg[k] = trades_by_avg.get(k, 0) + v / len(trades_by)
        return {
            "pass_rate": pass_rate,
            "avg_return": avg_ret,
            "avg_max_dd": avg_dd,
            "avg_trades": avg_trades,
            "avg_trades_by_strategy": trades_by_avg,
            "bars_to_verdict": summarize(bars_to) if bars_to else None,
            "trades_to_verdict": summarize(trades_to) if trades_to else None,
        }

    out = {"30d": eval_window(30), "60d": eval_window(60)}
    print("PROGRESS: 1/2", flush=True)
    out_path = reports_dir / "fastpass_usdjpy_exp_v3_fastwindows.json"
    out_path.write_text(__import__("json").dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print("PROGRESS: 2/2", flush=True)
    if mt5:
        mt5.shutdown()


if __name__ == "__main__":
    main()
