import itertools
import json
import os
import random
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
import sys

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
    DEFAULT_PROFILE_USDJPY_LONDON_M15_V1,
    DEFAULT_PROFILE_USDJPY_LIQUI_M15_V1,
    DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1,
    EXPERIMENTAL_PROFILE_USDJPY_TREND_KD_M15_V1,
    LondonBreakoutSignalConfig,
    LiquiditySweepSignalConfig,
    MomentumPinballSignalConfig,
    TrendKDSignalConfig,
    RiskControlConfig,
)
from omegafx_v2.signals import (
    build_london_breakout_signals,
    build_liquidity_sweep_signals,
    build_momentum_pinball_signals_m5,
    build_trend_kd_signals_m15,
    compute_atr,
)
from omegafx_v2.regime import tag_regimes
from omegafx_v2.sim import simulate_trade_path
from omegafx_v2.mt5_adapter import fetch_symbol_ohlc_mt5

USE_REGIME_MASKS = True  # KD-primary assumes regime filters on


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


def apply_regime_mask(signals, regime_series, profile):
    if profile.edge_regime_config is None:
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


def build_signals(profile, ohlc_m15, ohlc_m5, regime_series):
    sig_cfg = profile.signals
    if isinstance(sig_cfg, LondonBreakoutSignalConfig):
        return apply_regime_mask(build_london_breakout_signals(ohlc_m15, signal_config=sig_cfg, session=profile.session), regime_series, profile)
    if isinstance(sig_cfg, LiquiditySweepSignalConfig):
        return apply_regime_mask(build_liquidity_sweep_signals(ohlc_m15, signal_config=sig_cfg, session=profile.session), regime_series, profile)
    if isinstance(sig_cfg, MomentumPinballSignalConfig):
        sig = build_momentum_pinball_signals_m5(ohlc_m5, ohlc_m15, signal_config=sig_cfg, session=profile.session)
        return apply_regime_mask(sig, regime_series.reindex(sig.index, method="ffill"), profile)
    if isinstance(sig_cfg, TrendKDSignalConfig):
        sig = build_trend_kd_signals_m15(ohlc_m15, signal_config=sig_cfg, session=profile.session)
        return apply_regime_mask(sig, regime_series, profile)
    raise TypeError(f"Unsupported signal config: {type(sig_cfg)}")


def run_eval(window_days, ohlc_m15_full, ohlc_m5_full, profiles, risk_scales, risk_control, num_evals, initial_equity=10_000.0):
    window_delta = timedelta(days=window_days)
    master_index = ohlc_m15_full.index
    actual_start = max(ohlc_m15_full.index.min(), ohlc_m5_full.index.min())
    actual_end = min(ohlc_m15_full.index.max(), ohlc_m5_full.index.max())
    eligible = [i for i, ts in enumerate(master_index) if ts >= actual_start and ts + window_delta <= actual_end]
    if not eligible:
        return []
    regimes = tag_regimes(ohlc_m15_full, compute_atr(ohlc_m15_full, period=14), ohlc_m15_full["close"].ewm(span=50, adjust=False).mean().diff())

    evals = []
    for i_eval in range(num_evals):
        start_idx = random.choice(eligible)
        start_ts = master_index[start_idx]
        end_ts = start_ts + window_delta
        ohlc_m15 = ohlc_m15_full.loc[start_ts:end_ts]
        ohlc_m5 = ohlc_m5_full.loc[start_ts:end_ts]

        events = []
        for profile in profiles:
            signals = build_signals(profile, ohlc_m15, ohlc_m5, regimes)
            data = ohlc_m5 if profile.timeframe.startswith("5") else ohlc_m15
            for idx, ts in enumerate(signals.index):
                if signals.iat[idx]:
                    events.append((ts, profile, idx, data))
        events.sort(key=lambda x: x[0])

        equity = initial_equity
        target_equity = initial_equity * 1.07
        loss_equity = initial_equity * (1 - 0.06)
        max_equity = equity
        min_equity = equity
        current_scales = dict(risk_scales)
        reduced = False
        verdict = "timeout"

        for ts, profile, entry_idx, data in events:
            peak = max_equity
            dd = (peak - equity) / peak if peak > 0 else 0
            if risk_control:
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
                profit_target_pct=0.07,
            )
            outcome = simulate_trade_path(
                ohlc=data,
                entry_idx=entry_idx,
                account_balance=equity,
                config=cfg,
                costs=profile.costs,
            )
            equity += outcome.pnl
            max_equity = max(max_equity, equity)
            min_equity = min(min_equity, equity)
            if equity >= target_equity:
                verdict = "pass"
                break
            if equity <= loss_equity:
                verdict = "fail"
                break
        evals.append(
            {
                "verdict": verdict,
                "total_return": equity / initial_equity - 1.0,
                "max_dd": (max_equity - min_equity) / max_equity if max_equity > 0 else 0.0,
            }
        )
    return evals


def main():
    if not init_mt5():
        return
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    end = datetime.now()
    start = end - timedelta(days=int(os.getenv("TOTAL_HISTORY_DAYS", "270")))
    num_evals = int(os.getenv("NUM_EVALS", "100"))

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

    london_risks = [0.5, 0.75, 1.0, 1.25, 1.5]
    momentum_risks = [0.0, 0.005, 0.01]
    kd_risks = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5]
    dd_thresholds = [0.04, 0.05, 0.06, 0.07]
    reduction_factors = [0.3, 0.5, 0.7]

    candidates = list(itertools.product(london_risks, momentum_risks, kd_risks, dd_thresholds, reduction_factors))
    random.shuffle(candidates)
    total = len(candidates)
    start_time = time.time()

    scored = []
    for idx, (l_risk, m_risk, kd_risk, dd_th, red_fac) in enumerate(candidates, 1):
        if idx % max(1, total // 10) == 0:
            elapsed = time.time() - start_time
            pct = idx / total * 100
            eta = (elapsed / idx) * (total - idx) if idx else 0
            # Two styles so dashboard and humans both see it
            print(f"PROGRESS: {idx}/{total}", flush=True)
            print(f"Progress: {idx}/{total} ({pct:.1f}%), ETA ~{eta/60:.1f} min", flush=True)
        risk_scales = {
            DEFAULT_PROFILE_USDJPY_LONDON_M15_V1.name: l_risk,
            DEFAULT_PROFILE_USDJPY_LIQUI_M15_V1.name: 0.25,
            DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1.name: m_risk,
            EXPERIMENTAL_PROFILE_USDJPY_TREND_KD_M15_V1.name: kd_risk,
        }
        rc = RiskControlConfig(dd_reduce_threshold_pct=dd_th, dd_restore_threshold_pct=0.03, risk_scale_reduction_factor=red_fac)
        evals = run_eval(30, ohlc_m15, ohlc_m5, profiles, risk_scales, rc, num_evals=num_evals, initial_equity=10_000.0)
        if not evals:
            continue
        pass_rate = sum(1 for e in evals if e["verdict"] == "pass") / len(evals)
        avg_ret = float(np.mean([e["total_return"] for e in evals]))
        avg_dd = float(np.mean([e["max_dd"] for e in evals]))
        score = avg_ret - max(0, avg_dd - 0.095)
        scored.append(
            {
                "london_risk": l_risk,
                "momentum_risk": m_risk,
                "kd_risk": kd_risk,
                "dd_reduce_threshold_pct": dd_th,
                "risk_scale_reduction_factor": red_fac,
                "pass_rate_30d": pass_rate,
                "avg_return_30d": avg_ret,
                "avg_max_dd_30d": avg_dd,
                "score": score,
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    top_n = scored[: min(20, len(scored))]
    out_path = ROOT / "reports" / "optimize_usdjpy_exp_v2_kd_primary.json"
    out_path.write_text(json.dumps(top_n, indent=2), encoding="utf-8")
    print(f"Wrote top configs to {out_path}")
    if mt5:
        mt5.shutdown()


if __name__ == "__main__":
    main()
