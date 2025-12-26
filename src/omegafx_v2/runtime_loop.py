from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Optional, List, Iterable, Tuple
import json
import os
import csv
from pathlib import Path
from datetime import datetime

import pandas as pd

from .config import StrategyProfile
from .profile_summary import _build_signals_for_profile
from .sim import simulate_trade_path
from .strategy import plan_single_trade
from .logger import get_logger


class BrokerAdapter(ABC):
    """Abstract interface for sending trades to a broker."""

    @abstractmethod
    def send_order(self, trade) -> bool:
        """Execute a trade request. Returns True if accepted, False otherwise."""
        raise NotImplementedError

    @abstractmethod
    def log(self, message: str) -> None:
        """Optional logging hook."""
        raise NotImplementedError


class DummyBrokerAdapter(BrokerAdapter):
    def __init__(self) -> None:
        self.last_ticket: Optional[int] = None

    def send_order(self, trade) -> bool:
        print(f"[DummyBroker] EXECUTE: {trade}")
        self.last_ticket = None
        return True

    def log(self, message: str) -> None:
        print(f"[DummyBroker] {message}")


@dataclass
class LiveRuntimeState:
    equity: float
    trades_executed: int = 0
    daily_pnl: Optional[dict] = None
    last_bar_time: Optional[pd.Timestamp] = None


def run_live_runtime(
    profile: StrategyProfile,
    bar_stream: Iterable[Tuple[pd.Timestamp, pd.Series]],
    broker: BrokerAdapter,
    initial_equity: float = 10_000.0,
    state_file: str = "runtime_state.json",
    heartbeat_every: int = 50,
    shadow_trades_path: str = "logs/shadow_trades.csv",
) -> LiveRuntimeState:
    """
    Live-mode execution loop over a bar stream.
    - bar_stream yields (timestamp, row) one-by-one (historical or real-time).
    - For each new bar, build signals and, if the latest bar is a signal,
      simulate the trade and send to broker.
    """
    historical_idx: List[pd.Timestamp] = []
    historical_rows: List[pd.Series] = []

    equity = initial_equity
    daily_pnl: dict = {}
    trades_executed = 0
    last_processed = None
    strategy_version = os.getenv("OMEGAFX_STRATEGY_VERSION", "")
    run_id = os.getenv("OMEGAFX_RUN_ID", "")

    target_equity = initial_equity * (1.0 + profile.challenge.profit_target_pct)
    loss_limit_equity = initial_equity * (1.0 - profile.challenge.max_total_loss_pct)

    strategy_cfg = replace(
        profile.strategy,
        profit_target_pct=profile.challenge.profit_target_pct,
    )
    logger = get_logger()

    if state_file and os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                saved = json.load(f)
            equity = saved.get("equity", equity)
            daily_pnl = saved.get("daily_pnl", daily_pnl)
            ts_saved = saved.get("last_bar_time")
            if ts_saved:
                last_processed = pd.to_datetime(ts_saved)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to load runtime state: {exc}")

    for idx_stream, (ts, row) in enumerate(bar_stream):
        if last_processed is not None and ts <= last_processed:
            continue
        if historical_idx and ts <= historical_idx[-1]:
            logger.warning("Bar feed stalled or duplicate timestamp detected; stopping.")
            break

        historical_idx.append(ts)
        historical_rows.append(row[["open", "high", "low", "close"]])

        df = pd.DataFrame(historical_rows, index=pd.DatetimeIndex(historical_idx))

        if len(df) < 2:
            continue

        signals = _build_signals_for_profile(df, profile)
        entry_idx = len(df) - 2  # evaluate signal on penultimate bar to ensure an exit bar exists
        if entry_idx < 0 or not signals.iloc[entry_idx]:
            continue

        if equity >= target_equity or equity <= loss_limit_equity:
            broker.log("Equity target/loss limit reached; stopping.")
            break

        dates = df.index.date
        if profile.challenge.daily_loss_pct is not None:
            day = dates[entry_idx]
            day_pnl = daily_pnl.get(day, 0.0)
            day_limit = initial_equity * profile.challenge.daily_loss_pct
            if day_pnl <= -day_limit:
                broker.log(f"Daily loss cap hit for {day}; skipping signal.")
                continue

        equity_before = equity
        trade_plan = plan_single_trade(
            account_balance=equity,
            current_price=df["close"].iloc[entry_idx],
            config=strategy_cfg,
        )

        outcome = simulate_trade_path(
            ohlc=df,
            entry_idx=entry_idx,
            account_balance=equity,
            config=strategy_cfg,
            costs=profile.costs,
        )

        accepted = broker.send_order(trade_plan)
        if not accepted:
            broker.log("Order rejected by broker.")
            continue

        equity += outcome.pnl
        trades_executed += 1

        if profile.challenge.daily_loss_pct is not None:
            day = dates[entry_idx]
            daily_pnl[day] = daily_pnl.get(day, 0.0) + outcome.pnl

        broker.log(
            f"Trade closed: reason={outcome.exit_reason}, pnl={outcome.pnl:.2f}, equity={equity:.2f}"
        )

        if shadow_trades_path:
            try:
                path = Path(shadow_trades_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                write_header = not path.exists()
                with path.open("a", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "profile_name",
                            "symbol",
                            "strategy_version",
                            "entry_time",
                            "exit_time",
                            "entry_date",
                            "direction",
                            "lot_size",
                            "entry_price",
                            "exit_price",
                            "sl_price",
                            "tp_price",
                            "exit_reason",
                            "pnl",
                            "pnl_pct",
                            "equity_before",
                            "equity_after",
                            "challenge_id",
                            "run_id",
                        ],
                    )
                    if write_header:
                        writer.writeheader()
                    writer.writerow(
                        {
                            "profile_name": profile.name,
                            "symbol": profile.symbol_key,
                            "strategy_version": strategy_version,
                            "entry_time": outcome.entry_time.isoformat(),
                            "exit_time": outcome.exit_time.isoformat(),
                            "entry_date": outcome.entry_time.date().isoformat(),
                            "direction": trade_plan.direction,
                            "lot_size": trade_plan.lot_size,
                            "entry_price": outcome.entry_price,
                            "exit_price": outcome.exit_price,
                            "sl_price": trade_plan.stop_loss_price,
                            "tp_price": trade_plan.take_profit_price,
                            "exit_reason": outcome.exit_reason,
                            "pnl": outcome.pnl,
                            "pnl_pct": outcome.pnl_pct,
                            "equity_before": equity_before,
                            "equity_after": equity,
                            "challenge_id": os.getenv("OMEGAFX_CHALLENGE_ID", ""),
                            "run_id": run_id or datetime.utcnow().isoformat(),
                        }
                    )
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Failed to append shadow trade: {exc}")

        if state_file:
            try:
                with open(state_file, "w") as f:
                    json.dump(
                        {
                            "equity": equity,
                            "last_bar_time": ts.isoformat(),
                            "daily_pnl": daily_pnl,
                            "profile": profile.name,
                        },
                        f,
                    )
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Failed to persist runtime state: {exc}")

        if heartbeat_every and idx_stream % heartbeat_every == 0:
            logger.info(f"Heartbeat OK: bars={idx_stream}, equity={equity:.2f}")

    return LiveRuntimeState(
        equity=equity,
        trades_executed=trades_executed,
        daily_pnl=daily_pnl,
        last_bar_time=historical_idx[-1] if historical_idx else None,
    )
