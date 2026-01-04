from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


ROOT = Path(__file__).resolve().parents[2]
_log_root = os.getenv("LOG_ROOT")
if _log_root:
    _log_root_path = Path(_log_root)
    if not _log_root_path.is_absolute():
        _log_root_path = ROOT / _log_root_path
else:
    _log_root_path = ROOT / "logs"
LOG_DIR = _log_root_path
LOG_DIR.mkdir(parents=True, exist_ok=True)
POLICY_FILE = LOG_DIR / "policy.json"


@dataclass
class PolicyConfig:
    lose_streak_disable: int = 3
    symbol_disable_minutes: int = 120
    dd_reduce_threshold_pct: float = 0.02
    dd_cooldown_threshold_pct: float = 0.04
    cooldown_minutes: int = 120


def load_config_from_env() -> PolicyConfig:
    def _f(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, default))
        except ValueError:
            return float(default)

    def _i(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, default))
        except ValueError:
            return int(default)

    return PolicyConfig(
        lose_streak_disable=_i("OMEGAFX_POLICY_LOSE_STREAK", 3),
        symbol_disable_minutes=_i("OMEGAFX_POLICY_DISABLE_MINUTES", 120),
        dd_reduce_threshold_pct=_f("OMEGAFX_POLICY_DD_REDUCE", 0.02),
        dd_cooldown_threshold_pct=_f("OMEGAFX_POLICY_DD_COOLDOWN", 0.04),
        cooldown_minutes=_i("OMEGAFX_POLICY_COOLDOWN_MINUTES", 120),
    )


def init_state(initial_equity: float) -> Dict[str, Any]:
    now = time.time()
    day = datetime.utcfromtimestamp(now).strftime("%Y-%m-%d")
    return {
        "day_start_equity": float(initial_equity),
        "day_start_date": day,
        "min_equity_today": float(initial_equity),
        "symbol_streaks": {},
        "symbol_disabled_until": {},
        "closed_trades_today": 0,
        "cooldown_until": None,
    }


def default_policy(symbols) -> Dict[str, Any]:
    return {
        "global_risk_scale": 1.0,
        "symbol_enabled": {sym: True for sym in symbols},
        "cooldown_until": None,
        "reason": None,
    }


def _roll_day(state: Dict[str, Any], equity: float, now_ts: float) -> None:
    day = datetime.utcfromtimestamp(now_ts).strftime("%Y-%m-%d")
    if state.get("day_start_date") != day:
        state["day_start_date"] = day
        state["day_start_equity"] = float(equity)
        state["min_equity_today"] = float(equity)
        state["closed_trades_today"] = 0


def evaluate_policy(state: Dict[str, Any], equity: float, symbols, now_ts: float, cfg: PolicyConfig) -> Dict[str, Any]:
    _roll_day(state, equity, now_ts)
    state["min_equity_today"] = min(state.get("min_equity_today", equity), equity)
    day_start = state.get("day_start_equity") or equity
    dd_pct = 0.0
    if day_start:
        dd_pct = (day_start - state["min_equity_today"]) / day_start

    global_risk_scale = 1.0
    reason = None
    cooldown_until = state.get("cooldown_until")
    if dd_pct >= cfg.dd_reduce_threshold_pct:
        global_risk_scale = 0.5
        reason = "daily_dd_reduce"
    if dd_pct >= cfg.dd_cooldown_threshold_pct:
        until = now_ts + cfg.cooldown_minutes * 60
        cooldown_until = max(cooldown_until or 0, until)
        state["cooldown_until"] = cooldown_until
        reason = "daily_dd_cooldown"

    symbol_enabled = {}
    disabled_until = state.get("symbol_disabled_until", {})
    for sym in symbols:
        until = disabled_until.get(sym)
        symbol_enabled[sym] = not (until and now_ts < until)
    if cooldown_until and now_ts < cooldown_until:
        symbol_enabled = {sym: False for sym in symbols}

    return {
        "global_risk_scale": global_risk_scale,
        "symbol_enabled": symbol_enabled,
        "cooldown_until": cooldown_until,
        "reason": reason,
    }


def write_policy(policy: Dict[str, Any]) -> None:
    payload = dict(policy)
    payload["timestamp_utc"] = datetime.utcnow().isoformat()
    POLICY_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def record_trade(state: Dict[str, Any], symbol: str, pnl: float, ts: float, cfg: PolicyConfig) -> None:
    _roll_day(state, state.get("min_equity_today", 0) or 0, ts)
    state["closed_trades_today"] = int(state.get("closed_trades_today", 0)) + 1
    streaks = state.get("symbol_streaks", {})
    disabled_until = state.get("symbol_disabled_until", {})

    streak = streaks.get(symbol, 0)
    if pnl < 0:
        streak += 1
    else:
        streak = 0
    if streak >= cfg.lose_streak_disable:
        disabled_until[symbol] = ts + cfg.symbol_disable_minutes * 60
        streak = 0

    streaks[symbol] = streak
    state["symbol_streaks"] = streaks
    state["symbol_disabled_until"] = disabled_until
