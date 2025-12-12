from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

from .config import (
    MeanReversionSignalConfig,
    SignalConfig,
    StrategyProfile,
)
from .data import fetch_symbol_ohlc
from .signals import build_mean_reversion_signals, build_signals
from .sim import run_randomized_signal_evaluations, run_signal_driven_evaluation


def _build_signals_for_profile(ohlc, profile: StrategyProfile):
    sig_cfg = profile.signals
    if isinstance(sig_cfg, SignalConfig):
        return build_signals(
            ohlc,
            signal_config=sig_cfg,
            session=profile.session,
        )
    if isinstance(sig_cfg, MeanReversionSignalConfig):
        return build_mean_reversion_signals(
            ohlc,
            signal_config=sig_cfg,
            session=profile.session,
        )
    raise TypeError(f"Unsupported signals config type: {type(sig_cfg)!r}")


def run_profile_summary(
    profile: StrategyProfile,
    days: int = 365,
    initial_equity: float = 10_000.0,
    num_evals: int = 100,
) -> dict:
    """
    Fetch data, build signals, and run single + batch evaluations for a profile.
    """
    end = date.today()
    start = end - timedelta(days=days)

    ohlc = fetch_symbol_ohlc(
        profile.symbol_key,
        start=start.isoformat(),
        end=end.isoformat(),
        interval=profile.timeframe,
    )

    signals = _build_signals_for_profile(ohlc, profile)

    strategy_cfg = replace(
        profile.strategy,
        profit_target_pct=profile.challenge.profit_target_pct,
    )

    single = run_signal_driven_evaluation(
        ohlc=ohlc,
        signals=signals,
        initial_equity=initial_equity,
        challenge=profile.challenge,
        config=strategy_cfg,
        costs=profile.costs,
        daily_loss_pct=profile.challenge.daily_loss_pct,
    )

    batch = run_randomized_signal_evaluations(
        ohlc=ohlc,
        signals=signals,
        initial_equity=initial_equity,
        challenge=profile.challenge,
        config=strategy_cfg,
        costs=profile.costs,
        num_evals=num_evals,
        min_bars_per_eval=profile.challenge.min_bars_per_eval,
        start_offset=5,
    )

    avg_trades = (
        sum(len(ev.trades) for ev in batch.evaluations) / batch.num_evals
        if batch.num_evals > 0
        else 0.0
    )

    return {
        "bars": len(ohlc),
        "signals_total": int(signals.sum()),
        "single_verdict": single.verdict,
        "single_return": single.total_return,
        "single_max_dd": single.max_drawdown_pct,
        "batch_pass_rate": batch.pass_rate,
        "batch_avg_return": batch.average_return,
        "batch_avg_max_dd": batch.average_max_drawdown,
        "batch_avg_trades": avg_trades,
    }
