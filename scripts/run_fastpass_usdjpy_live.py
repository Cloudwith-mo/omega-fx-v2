from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None

from omegafx_v2.config import (
    DEFAULT_PORTFOLIO_USDJPY_FASTPASS_CORE,
    DEFAULT_PORTFOLIO_USDJPY_FASTPASS_V3,
    MULTI_PORTFOLIO_USDJPY_GBPJPY_FASTPASS,
    MULTI_PORTFOLIO_SPLASH_PASS,
    MULTI_PORTFOLIO_SCORERS,
    MULTI_PORTFOLIO_COMBINED,
    WESTBROOK_PORTFOLIO_USDJPY,
    EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V2,
    EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V3,
    DEFAULT_STRATEGY,
    DEFAULT_CHALLENGE,
    DEFAULT_COSTS,
    DEFAULT_SESSION,
    PortfolioProfile,
    MultiSymbolPortfolioProfile,
    StrategyProfile,
    SignalConfig,
    MeanReversionSignalConfig,
    TrendContinuationSignalConfig,
    LondonBreakoutSignalConfig,
    LiquiditySweepSignalConfig,
    MomentumPinballSignalConfig,
    TrendKDSignalConfig,
    VanVleetSignalConfig,
    BigManSignalConfig,
)
from omegafx_v2.logger import get_logger
from omegafx_v2.mt5_adapter import Mt5BrokerAdapter, Mt5ConnectionConfig
from omegafx_v2.profile_summary import _build_signals_for_profile
from omegafx_v2.signals import (
    build_momentum_pinball_signals_m5,
    build_trend_kd_signals_m15,
    build_vanvleet_signals_m15,
    build_bigman_signals_h1,
    build_session_mask,
    compute_atr,
    build_atr_filter,
    compute_h4_sma_filter,
)
from omegafx_v2.sim import simulate_trade_path
from omegafx_v2.strategy import plan_single_trade, PlannedTrade
from omegafx_v2.policy_engine import (
    load_config_from_env,
    init_state,
    default_policy,
    evaluate_policy,
    write_policy,
    record_trade,
)

ROOT = Path(__file__).resolve().parent.parent
_log_root = os.getenv("LOG_ROOT")
if _log_root:
    _log_root_path = Path(_log_root)
    if not _log_root_path.is_absolute():
        _log_root_path = ROOT / _log_root_path
else:
    _log_root_path = ROOT / "logs"
LOG_DIR = _log_root_path
LOG_DIR.mkdir(parents=True, exist_ok=True)
ACCOUNT_ID = os.getenv("ACCOUNT_ID") or os.getenv("OMEGAFX_ACCOUNT_ID") or ""
CREDS_SOURCE = "missing"
MT5_IDENTITY: Dict[str, object] = {}
MT5_INIT_OK: Optional[bool] = None
MT5_LAST_ERROR: Optional[str] = None
LAST_TERMINAL_INFO: Optional[object] = None
LAST_TERMINAL_TRADE_ALLOWED: Optional[bool] = None
LAST_ACCOUNT_TRADE_ALLOWED: Optional[bool] = None
LAST_MT5_TRADE_ALLOWED: Optional[bool] = None
LAST_ORDER_SEND_RESULT: Optional[Dict[str, object]] = None
LAST_ORDER_SEND_ERROR: Optional[str] = None
ORDER_SEND_NONE_COUNT: int = 0
DRY_RUN_ENABLED: Optional[bool] = None
MT5_PATH_USED: Optional[str] = None
MT5_OPEN_POSITIONS_COUNT: Optional[int] = None
MT5_VERIFIED_CLOSED_TRADES_COUNT: int = 0
LOG_EVENTS_TOTAL: int = 0
MATCHED_MT5_TICKETS_COUNT: int = 0
UNMATCHED_LOG_ONLY_COUNT: int = 0
EVIDENCE_TIER: str = "practice"
TICK_STALE_SECONDS: int = int(os.getenv("OMEGAFX_TICK_STALE_SECONDS", "300"))
PLAY_COUNTER: int = 0
RESEARCH_DAY_START_EQUITY: Optional[float] = None
RESEARCH_DAY_START_DATE: Optional[str] = None
RESEARCH_DAILY_STOP_HIT: bool = False
WESTBROOK_DAY_START_EQUITY: Optional[float] = None
WESTBROOK_DAY_START_DATE: Optional[str] = None
WESTBROOK_DAILY_STOP_HIT: bool = False
SCORERS_DAY_START_EQUITY: Optional[float] = None
SCORERS_DAY_START_DATE: Optional[str] = None
SCORERS_DAILY_STOP_HIT: bool = False
LAST_NO_SETUP_LOG_TS: Dict[str, float] = {}
DECISION_TRACE: Dict[str, object] = {}
PASS_SYMBOLS_ACTIVE: List[str] = []
MAX_OPEN_POSITIONS: int = 0
BUCKET_CAPS: Dict[str, int] = {}
RISK_SETTINGS: Dict[str, object] = {}
OPEN_POSITIONS_SCORERS: int = 0
OPEN_POSITIONS_WESTBROOK: int = 0
OPEN_RISK_SCORERS_PCT: float = 0.0
OPEN_RISK_WESTBROOK_PCT: float = 0.0
TOTAL_OPEN_RISK_PCT: float = 0.0
try:
    MT5_INIT_DELAY_SEC: float = float(os.getenv("OMEGAFX_MT5_INIT_DELAY_SEC", "2"))
except Exception:
    MT5_INIT_DELAY_SEC = 2.0

STATE_FILE = LOG_DIR / "state.json"
HEARTBEAT_FILE = LOG_DIR / "runner_heartbeat.txt"
STOP_REASON_FILE = LOG_DIR / "last_stop_reason.txt"
LIFECYCLE_LOG = LOG_DIR / "runner_lifecycle.log"
ERROR_LOG = LOG_DIR / "runner_errors.log"
ERROR_CURRENT_LOG = LOG_DIR / "runner_errors_current.log"
ATTR_LOG = LOG_DIR / "trade_attribution.csv"
IDENTITY_FILE = LOG_DIR / "identity.json"
ACCOUNT_SNAPSHOT_FILE = LOG_DIR / "account_snapshot.json"
RESEARCH_PACKET_FILE = LOG_DIR / "research_packet.json"
EXPECTED_IDENTITY_FILE = LOG_DIR / "expected_identity.json"
PAPER_SHOTS_LOG = LOG_DIR / "paper_shots.csv"

IDENTITY_LOCK: Dict[str, Optional[str]] = {
    "status": "unknown",
    "reason": "identity_not_checked",
}
IDENTITY_DATA: Dict[str, object] = {}
LAST_ACCOUNT_INFO: Optional[object] = None
LAST_ACCOUNT_ERROR: Optional[str] = None

logger = get_logger("omega-fx-live")


def log_lifecycle(message: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{ts}] {message}\n"
    LIFECYCLE_LOG.write_text(
        (LIFECYCLE_LOG.read_text(encoding="utf-8") if LIFECYCLE_LOG.exists() else "") + line,
        encoding="utf-8",
    )


def log_error(message: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{ts}] {message}\n"
    ERROR_LOG.write_text(
        (ERROR_LOG.read_text(encoding="utf-8") if ERROR_LOG.exists() else "") + line,
        encoding="utf-8",
    )
    ERROR_CURRENT_LOG.write_text(
        (ERROR_CURRENT_LOG.read_text(encoding="utf-8") if ERROR_CURRENT_LOG.exists() else "") + line,
        encoding="utf-8",
    )


def _append_log(path: Path, line: str) -> None:
    path.write_text(
        (path.read_text(encoding="utf-8") if path.exists() else "") + line,
        encoding="utf-8",
    )


def _git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return "unknown"


def start_run_logs(mode: str, portfolio: str) -> Tuple[str, str]:
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    run_id = f"{ts}_{os.getpid()}"
    commit = _git_commit()
    marker = f"=== NEW RUN === ts={ts} run_id={run_id} commit={commit} mode={mode} portfolio={portfolio}\n"
    _append_log(LIFECYCLE_LOG, marker)
    _append_log(ERROR_LOG, marker)
    ERROR_CURRENT_LOG.write_text(marker, encoding="utf-8")
    return run_id, commit


def set_stop_reason(reason: str) -> None:
    STOP_REASON_FILE.write_text(reason, encoding="utf-8")


def write_state(
    state: str,
    mode: str,
    portfolio: str,
    reason: Optional[str] = None,
    equity: Optional[float] = None,
    symbols: Optional[List[str]] = None,
    cooldown_until: Optional[float] = None,
) -> None:
    data = {
        "state": state,
        "armed": state == "RUNNING",
        "mode": mode,
        "portfolio": portfolio,
        "reason": reason,
        "equity": equity,
        "symbols": symbols or [],
        "cooldown_until": cooldown_until,
        "account_id": ACCOUNT_ID,
        "log_root": str(LOG_DIR),
        "updated_at": time.time(),
        "pid": os.getpid(),
    }
    if MT5_IDENTITY:
        data.update(MT5_IDENTITY)
    STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_heartbeat(mode: str, portfolio: str, equity: float) -> None:
    HEARTBEAT_FILE.write_text(
        f"{time.time()}|{equity}|{portfolio}|{mode}|{ACCOUNT_ID}", encoding="utf-8"
    )


def _write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _clean_env_value(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    cleaned = val.strip().strip('"').strip("'")
    return cleaned or None


def _expected_identity_from_env() -> Dict[str, Optional[str]]:
    return {
        "login": _clean_env_value(os.getenv("OMEGAFX_EXPECT_LOGIN")),
        "server": _clean_env_value(os.getenv("OMEGAFX_EXPECT_SERVER")),
        "company": _clean_env_value(os.getenv("OMEGAFX_EXPECT_COMPANY")),
    }

def _read_expected_identity_file() -> Dict[str, Optional[str]]:
    if not EXPECTED_IDENTITY_FILE.exists():
        return {}
    try:
        data = json.loads(EXPECTED_IDENTITY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    login_val = data.get("login")
    server_val = data.get("server")
    company_val = data.get("company")
    return {
        "login": _clean_env_value(str(login_val)) if login_val is not None else None,
        "server": _clean_env_value(str(server_val)) if server_val is not None else None,
        "company": _clean_env_value(str(company_val)) if company_val is not None else None,
    }


def _expected_identity_from_env_or_file() -> Tuple[Dict[str, Optional[str]], str]:
    expected = _expected_identity_from_env()
    if any(expected.values()):
        return expected, "env"
    file_expected = _read_expected_identity_file()
    if any(file_expected.values()):
        return file_expected, "file"
    return expected, "none"


def _auto_set_expected_identity(account) -> Dict[str, Optional[str]]:
    login = getattr(account, "login", None) if account else None
    server = getattr(account, "server", None) if account else None
    if login is None or server is None:
        return {}
    company = getattr(account, "company", None) if account else None
    payload = {
        "account_id": ACCOUNT_ID,
        "login": str(login),
        "server": str(server),
        "company": str(company) if company is not None else None,
        "set_at": datetime.utcnow().isoformat(),
    }
    _write_json_atomic(EXPECTED_IDENTITY_FILE, payload)
    log_lifecycle(f"AUTO_EXPECT | login={login} | server={server}")
    return {
        "login": str(login),
        "server": str(server),
        "company": str(company) if company is not None else None,
    }


def _evaluate_identity_lock(
    actual_login: Optional[object],
    actual_server: Optional[str],
    actual_company: Optional[str],
    expected: Dict[str, Optional[str]],
) -> Tuple[str, Optional[str]]:
    mismatches = []
    expected_login = expected.get("login")
    expected_server = expected.get("server")
    expected_company = expected.get("company")

    if expected_login is not None:
        if actual_login is None:
            mismatches.append(f"login expected={expected_login} actual=missing")
        elif str(actual_login) != str(expected_login):
            mismatches.append(f"login expected={expected_login} actual={actual_login}")
    if expected_server is not None:
        if actual_server is None:
            mismatches.append(f"server expected={expected_server} actual=missing")
        elif actual_server.strip().lower() != expected_server.strip().lower():
            mismatches.append(f"server expected={expected_server} actual={actual_server}")
    if expected_company is not None:
        if actual_company is None:
            mismatches.append(f"company expected={expected_company} actual=missing")
        elif actual_company.strip().lower() != expected_company.strip().lower():
            mismatches.append(f"company expected={expected_company} actual={actual_company}")

    if mismatches:
        return "mismatch", "ACCOUNT_MISMATCH | " + " | ".join(mismatches)
    if expected_login or expected_server or expected_company:
        return "ok", None
    return "unverified", "expected_identity_not_set"


def _build_identity_payload(
    account,
    term,
    mt5_path: Optional[str],
    expected: Optional[Dict[str, Optional[str]]] = None,
    expected_source: Optional[str] = None,
) -> Dict[str, object]:
    if expected is None or expected_source is None:
        expected, expected_source = _expected_identity_from_env_or_file()
    return {
        "account_id": ACCOUNT_ID,
        "log_root": str(LOG_DIR),
        "mt5_path": mt5_path,
        "mt5_account_login": getattr(account, "login", None) if account else None,
        "mt5_account_server": getattr(account, "server", None) if account else None,
        "mt5_company": getattr(account, "company", None) if account else None,
        "mt5_account_name": getattr(account, "name", None) if account else None,
        "mt5_currency": getattr(account, "currency", None) if account else None,
        "mt5_leverage": getattr(account, "leverage", None) if account else None,
        "mt5_trade_mode": getattr(account, "trade_mode", None) if account else None,
        "mt5_margin_mode": getattr(account, "margin_mode", None) if account else None,
        "mt5_terminal_path": getattr(term, "path", None) if term else None,
        "mt5_terminal_name": getattr(term, "name", None) if term else None,
        "mt5_terminal_company": getattr(term, "company", None) if term else None,
        "mt5_terminal_version": getattr(term, "version", None) if term else None,
        "expected_login": expected.get("login"),
        "expected_server": expected.get("server"),
        "expected_company": expected.get("company"),
        "expected_source": expected_source,
        "creds_source": CREDS_SOURCE,
    }


def _compute_trade_allowed(account, terminal) -> Tuple[Optional[bool], Optional[bool], Optional[bool]]:
    term_allowed = getattr(terminal, "trade_allowed", None) if terminal else None
    acct_allowed = getattr(account, "trade_allowed", None) if account else None
    if term_allowed is None and acct_allowed is None:
        return None, term_allowed, acct_allowed
    if term_allowed is None:
        return bool(acct_allowed), term_allowed, acct_allowed
    if acct_allowed is None:
        return bool(term_allowed), term_allowed, acct_allowed
    return bool(term_allowed) and bool(acct_allowed), term_allowed, acct_allowed


def write_identity_file(payload: Dict[str, object]) -> None:
    if not payload:
        return
    data = dict(payload)
    data["timestamp_utc"] = datetime.utcnow().isoformat()
    _write_json_atomic(IDENTITY_FILE, data)


def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None


def write_account_snapshot(
    account,
    error: Optional[str] = None,
    raw_bot_status: Optional[str] = None,
    risk_policy: Optional[Dict[str, object]] = None,
    risk_policy_ok: Optional[bool] = None,
    risk_policy_reason: Optional[str] = None,
    mt5_truth: Optional[Dict[str, object]] = None,
    mt5_truth_status: Optional[str] = None,
    mt5_truth_reason: Optional[str] = None,
    mt5_ticks: Optional[Dict[str, object]] = None,
    mt5_tick_status: Optional[str] = None,
    mt5_tick_reason: Optional[str] = None,
    extra_block_reason: Optional[str] = None,
) -> None:
    mt5_path = MT5_PATH_USED or os.getenv("MT5_PATH") or os.getenv("OMEGAFX_MT5_PATH")
    mt5_initialized = MT5_INIT_OK is True
    connected = bool(mt5_initialized and account is not None)
    status_raw = raw_bot_status or "DISARMED"
    lock_status = IDENTITY_LOCK.get("status") or "unknown"
    trade_allowed, term_allowed, acct_allowed = _compute_trade_allowed(account, LAST_TERMINAL_INFO)
    dry_run = DRY_RUN_ENABLED if DRY_RUN_ENABLED is not None else (os.getenv("OMEGAFX_LIVE_MODE", "0") != "1")
    trading_block_reason = None
    bot_status_effective = "LOCKED" if lock_status != "ok" else status_raw
    if lock_status == "ok" and status_raw == "ARMED":
        if dry_run:
            bot_status_effective = "DISARMED"
            trading_block_reason = "dry_run"
        elif trade_allowed is not True:
            bot_status_effective = "DISARMED"
            trading_block_reason = "mt5_trade_not_allowed"
    risk_ok = True if risk_policy_ok is None else bool(risk_policy_ok)
    trading_allowed = bool(
        status_raw == "ARMED"
        and lock_status == "ok"
        and connected
        and risk_ok
        and (trade_allowed is True)
        and not dry_run
        and extra_block_reason is None
    )
    if trading_block_reason is None and not trading_allowed:
        if lock_status != "ok":
            trading_block_reason = "identity_lock"
        elif not connected:
            trading_block_reason = "mt5_disconnected"
        elif trade_allowed is False:
            trading_block_reason = "mt5_trade_not_allowed"
        elif dry_run:
            trading_block_reason = "dry_run"
        elif not risk_ok:
            trading_block_reason = risk_policy_reason or "risk_policy"
    if extra_block_reason and not trading_allowed:
        trading_block_reason = extra_block_reason
    payload = {
        "account_id": ACCOUNT_ID,
        "log_root": str(LOG_DIR),
        "mt5_path": mt5_path,
        "mt5_initialized": mt5_initialized,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "mt5_connected": connected,
        "mt5_trade_allowed": trade_allowed,
        "mt5_terminal_trade_allowed": term_allowed,
        "mt5_account_trade_allowed": acct_allowed,
        "balance": _safe_float(getattr(account, "balance", None)) if connected else None,
        "equity": _safe_float(getattr(account, "equity", None)) if connected else None,
        "margin": _safe_float(getattr(account, "margin", None)) if connected else None,
        "free_margin": _safe_float(getattr(account, "margin_free", None)) if connected else None,
        "margin_level": _safe_float(getattr(account, "margin_level", None)) if connected else None,
        "currency": getattr(account, "currency", None) if connected else None,
        "leverage": getattr(account, "leverage", None) if connected else None,
        "mt5_account_login": (getattr(account, "login", None) if connected else None) or MT5_IDENTITY.get("mt5_account_login"),
        "mt5_account_server": (getattr(account, "server", None) if connected else None) or MT5_IDENTITY.get("mt5_account_server"),
        "identity_lock_status": IDENTITY_LOCK.get("status"),
        "identity_lock_reason": IDENTITY_LOCK.get("reason"),
        "bot_status_effective": bot_status_effective,
        "trading_allowed": trading_allowed,
        "trading_block_reason": trading_block_reason,
        "dry_run_enabled": dry_run,
        "last_order_send_result": LAST_ORDER_SEND_RESULT,
        "last_order_send_error": LAST_ORDER_SEND_ERROR,
        "order_send_none_count": ORDER_SEND_NONE_COUNT,
        "mt5_verified_closed_trades_count": MT5_VERIFIED_CLOSED_TRADES_COUNT,
        "mt5_open_positions_count": MT5_OPEN_POSITIONS_COUNT,
        "log_events_total": LOG_EVENTS_TOTAL,
        "matched_to_mt5_tickets_count": MATCHED_MT5_TICKETS_COUNT,
        "unmatched_shadow_or_log_only_count": UNMATCHED_LOG_ONLY_COUNT,
        "evidence_tier": EVIDENCE_TIER,
        "decision_trace": DECISION_TRACE,
        "mt5_truth_status": mt5_truth_status,
        "mt5_truth_reason": mt5_truth_reason,
        "mt5_positions_count": None if not mt5_truth else mt5_truth.get("positions_count"),
        "mt5_positions_profit": None if not mt5_truth else mt5_truth.get("positions_profit"),
        "mt5_orders_count": None if not mt5_truth else mt5_truth.get("orders_count"),
        "mt5_deals_last_2h_count": None if not mt5_truth else mt5_truth.get("deals_last_2h_count"),
        "mt5_last_deal_time": None if not mt5_truth else mt5_truth.get("last_deal_time"),
        "mt5_tick_status": mt5_tick_status,
        "mt5_tick_reason": mt5_tick_reason,
        "mt5_tick_age_sec_by_symbol": None if not mt5_ticks else mt5_ticks.get("age_sec_by_symbol"),
        "mt5_tick_time_by_symbol": None if not mt5_ticks else mt5_ticks.get("time_by_symbol"),
        "mt5_tick_missing_symbols": None if not mt5_ticks else mt5_ticks.get("missing_symbols"),
        "mt5_tick_stale_seconds": None if not mt5_ticks else mt5_ticks.get("stale_seconds"),
        "open_positions_scorers": OPEN_POSITIONS_SCORERS,
        "open_positions_westbrook": OPEN_POSITIONS_WESTBROOK,
        "open_risk_scorers_pct": OPEN_RISK_SCORERS_PCT,
        "open_risk_westbrook_pct": OPEN_RISK_WESTBROOK_PCT,
        "total_open_risk_pct": TOTAL_OPEN_RISK_PCT,
        "scorers_max_open_positions": RISK_SETTINGS.get("scorers_max_open_positions"),
        "westbrook_max_open_positions": RISK_SETTINGS.get("westbrook_max_open_positions"),
        "scorers_open_risk_cap_pct": RISK_SETTINGS.get("scorers_open_risk_cap_pct"),
        "westbrook_open_risk_cap_pct": RISK_SETTINGS.get("westbrook_open_risk_cap_pct"),
        "portfolio_open_risk_cap_pct": RISK_SETTINGS.get("portfolio_open_risk_cap_pct"),
        "research_block_reason": extra_block_reason,
        "pass_symbols": PASS_SYMBOLS_ACTIVE,
        "per_trade_risk_pct": RISK_SETTINGS.get("per_trade_risk_pct"),
        "per_trade_risk_pct_by_strategy": RISK_SETTINGS.get("per_trade_risk_pct_by_strategy"),
        "daily_loss_cap_pct": RISK_SETTINGS.get("daily_loss_cap_pct"),
        "soft_daily_stop_pct": RISK_SETTINGS.get("soft_daily_stop_pct"),
        "hard_daily_stop_pct": RISK_SETTINGS.get("hard_daily_stop_pct"),
        "max_open_positions": RISK_SETTINGS.get("max_open_positions"),
        "bucket_caps": RISK_SETTINGS.get("bucket_caps"),
        "risk_policy": risk_policy,
        "risk_policy_ok": risk_ok,
        "risk_policy_reason": risk_policy_reason,
        "error": error,
    }
    _write_json_atomic(ACCOUNT_SNAPSHOT_FILE, payload)


def _build_risk_policy_payload(
    policy: Dict[str, object],
    policy_config: PolicyConfig,
    dynamic_risk: bool,
    symbols: List[str],
    now_ts: float,
) -> Tuple[Dict[str, object], bool, Optional[str]]:
    symbol_enabled = policy.get("symbol_enabled") or {}
    enabled_count = sum(1 for val in symbol_enabled.values() if val)
    cooldown_until = policy.get("cooldown_until")
    in_cooldown = bool(dynamic_risk and cooldown_until and cooldown_until > now_ts)
    risk_ok = True
    risk_reason = None
    if dynamic_risk:
        if in_cooldown:
            risk_ok = False
            risk_reason = "cooldown"
        elif symbol_enabled and enabled_count == 0:
            risk_ok = False
            risk_reason = "all_symbols_disabled"
    payload = {
        "dynamic_risk": dynamic_risk,
        "global_risk_scale": policy.get("global_risk_scale"),
        "symbol_enabled": symbol_enabled,
        "cooldown_until": cooldown_until,
        "reason": policy.get("reason"),
        "in_cooldown": in_cooldown,
        "enabled_symbols": enabled_count,
        "total_symbols": len(symbols),
        "policy_lose_streak": policy_config.lose_streak_disable,
        "policy_disable_minutes": policy_config.symbol_disable_minutes,
        "policy_dd_reduce": policy_config.dd_reduce_threshold_pct,
        "policy_dd_cooldown": policy_config.dd_cooldown_threshold_pct,
        "policy_cooldown_minutes": policy_config.cooldown_minutes,
    }
    return payload, risk_ok, risk_reason


def _state_for_lock(mode: str) -> Tuple[str, Optional[str]]:
    if mode in {"demo", "ftmo"}:
        if IDENTITY_LOCK.get("status") == "mismatch":
            return "DISARMED", IDENTITY_LOCK.get("reason")
        if DRY_RUN_ENABLED:
            return "DISARMED", "dry_run"
        if LAST_MT5_TRADE_ALLOWED is False:
            return "DISARMED", "mt5_trade_not_allowed"
    return "RUNNING", None


def _read_local_creds() -> Dict[str, str]:
    global CREDS_SOURCE
    account_id = ACCOUNT_ID or os.getenv("OMEGAFX_ACCOUNT_ID") or os.getenv("ACCOUNT_ID")
    allow_fallback = os.getenv("OMEGAFX_ALLOW_LOCAL_CREDS_FALLBACK", "").lower() in {"1", "true", "yes"}
    candidates: List[Tuple[Path, str]] = []
    if account_id:
        candidates.append((ROOT / f"mt5_creds.{account_id}.local.bat", f"{account_id}.local"))
        if allow_fallback:
            candidates.append((ROOT / "mt5_creds.local.bat", "local_fallback"))
    else:
        candidates.append((ROOT / "mt5_creds.local.bat", "local_fallback"))
    def _valid(val: Optional[str]) -> bool:
        if not val:
            return False
        upper = val.strip().upper()
        if "YOUR_" in upper or upper in {"YOURLOGIN", "YOURPASSWORD", "YOUR_PASS"}:
            return False
        return True
    def _has_valid_creds(creds: Dict[str, str]) -> bool:
        login = creds.get("OMEGAFX_MT5_LOGIN") or creds.get("MT5_LOGIN")
        password = creds.get("OMEGAFX_MT5_PASSWORD") or creds.get("MT5_PASSWORD")
        server = creds.get("OMEGAFX_MT5_SERVER") or creds.get("MT5_SERVER")
        return _valid(login) and _valid(password) and _valid(server)
    for path, source in candidates:
        if not path.exists():
            continue
        creds: Dict[str, str] = {}
        raw = path.read_bytes()
        text = None
        for enc in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1252"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = raw.decode("utf-8", errors="ignore")
        for line in text.splitlines():
            line = line.strip()
            if not line.lower().startswith("set "):
                continue
            rest = line[4:].strip()
            if rest.startswith(("\"", "'")) and rest.endswith(("\"", "'")) and len(rest) > 1:
                rest = rest[1:-1]
            if "=" not in rest:
                continue
            key, val = rest.split("=", 1)
            key = key.strip().strip('"').strip("'")
            val = val.strip().strip('"').strip("'")
            creds[key] = val
        if creds and _has_valid_creds(creds):
            CREDS_SOURCE = source
            return creds
    CREDS_SOURCE = "missing"
    return {}


def mt5_env_values() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    def _clean(val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        return val.strip().strip('"').strip("'")

    def _valid(val: Optional[str]) -> bool:
        if not val:
            return False
        upper = val.strip().upper()
        if "YOUR_" in upper or upper in {"YOURLOGIN", "YOURPASSWORD", "YOUR_PASS"}:
            return False
        return True

    env_login = _clean(os.getenv("OMEGAFX_MT5_LOGIN") or os.getenv("MT5_LOGIN"))
    env_password = _clean(os.getenv("OMEGAFX_MT5_PASSWORD") or os.getenv("MT5_PASSWORD"))
    env_server = _clean(os.getenv("OMEGAFX_MT5_SERVER") or os.getenv("MT5_SERVER"))
    if _valid(env_login) and _valid(env_password) and _valid(env_server):
        global CREDS_SOURCE
        CREDS_SOURCE = "env"
        return env_login, env_password, env_server
    local = _read_local_creds()
    login = _clean(local.get("OMEGAFX_MT5_LOGIN") or local.get("MT5_LOGIN"))
    password = _clean(local.get("OMEGAFX_MT5_PASSWORD") or local.get("MT5_PASSWORD"))
    server = _clean(local.get("OMEGAFX_MT5_SERVER") or local.get("MT5_SERVER"))
    return login, password, server


def mt5_login() -> Tuple[bool, str | None]:
    if mt5 is None:
        return False, "MetaTrader5 package not installed"
    login, password, server = mt5_env_values()
    if not all([login, password, server]):
        return False, "MT5 credentials not set (OMEGAFX_MT5_LOGIN/PASSWORD/SERVER)"
    login_value = int(login)
    mt5_path = os.getenv("MT5_PATH") or os.getenv("OMEGAFX_MT5_PATH")
    global MT5_INIT_OK, MT5_LAST_ERROR, LAST_TERMINAL_INFO, LAST_TERMINAL_TRADE_ALLOWED
    global LAST_ACCOUNT_TRADE_ALLOWED, LAST_MT5_TRADE_ALLOWED, MT5_PATH_USED
    MT5_PATH_USED = mt5_path
    def _terminal64_pids() -> List[int]:
        try:
            import subprocess
            import csv as _csv
            out = subprocess.check_output(
                ["tasklist", "/FI", "IMAGENAME eq terminal64.exe", "/FO", "CSV", "/NH"],
                text=True,
                stderr=subprocess.STDOUT,
            )
            pids: List[int] = []
            for row in _csv.reader(out.splitlines()):
                if not row:
                    continue
                if row[0].strip('"').lower() != "terminal64.exe":
                    continue
                try:
                    pids.append(int(row[1]))
                except Exception:
                    continue
            return pids
        except Exception:
            return []
    def _terminal64_running() -> bool:
        return len(_terminal64_pids()) > 0

    if mt5_path:
        p = Path(mt5_path)
        log_lifecycle(f"MT5_PATH check | path={mt5_path} | exists={p.exists()} | is_file={p.is_file()}")
        if not p.exists() or not p.is_file():
            return False, f"MT5_PATH invalid or missing: {mt5_path}"
        try:
            with p.open("rb") as f:
                f.read(1)
        except Exception as exc:
            return False, f"MT5_PATH not readable: {exc}"
        log_lifecycle(f"MT5 initialize start | path={mt5_path}")
        if MT5_INIT_DELAY_SEC > 0:
            time.sleep(MT5_INIT_DELAY_SEC)
        ok_init = mt5.initialize(path=mt5_path)
    else:
        log_lifecycle("MT5 initialize start | path=default")
        if MT5_INIT_DELAY_SEC > 0:
            time.sleep(MT5_INIT_DELAY_SEC)
        ok_init = mt5.initialize()
    if not ok_init:
        pids = _terminal64_pids()
        MT5_INIT_OK = False
        MT5_LAST_ERROR = str(mt5.last_error())
        log_lifecycle(
            f"MT5 initialize failed | mt5_path={mt5_path or 'default'} | terminal64_pids={pids} | err={mt5.last_error()}"
        )
        if mt5_path:
            if pids:
                log_lifecycle("MT5 start-process | skipped | reason=terminal64 already running")
            else:
                try:
                    import subprocess
                    proc = subprocess.Popen([mt5_path])
                    log_lifecycle(f"MT5 start-process | ok=True | pid={proc.pid}")
                except Exception as exc:
                    log_lifecycle(f"MT5 start-process | ok=False | error={exc}")
        return False, f"MT5 initialize failed: {mt5.last_error()} (terminal64_pids={pids})"
    MT5_INIT_OK = True
    MT5_LAST_ERROR = None
    if not mt5.login(login=login_value, password=password, server=server):
        mt5.shutdown()
        return False, f"MT5 login failed: {mt5.last_error()}"
    account = mt5.account_info()
    term = mt5.terminal_info()
    if not account:
        mt5.shutdown()
        return False, "MT5 account_info unavailable after login"
    LAST_TERMINAL_INFO = term
    trade_allowed, term_allowed, acct_allowed = _compute_trade_allowed(account, term)
    LAST_TERMINAL_TRADE_ALLOWED = term_allowed if term_allowed is None else bool(term_allowed)
    LAST_ACCOUNT_TRADE_ALLOWED = acct_allowed if acct_allowed is None else bool(acct_allowed)
    LAST_MT5_TRADE_ALLOWED = trade_allowed if trade_allowed is None else bool(trade_allowed)

    expected, expected_source = _expected_identity_from_env_or_file()
    if expected_source == "none":
        auto_expected = _auto_set_expected_identity(account)
        if auto_expected:
            expected = auto_expected
            expected_source = "auto"
    identity_payload = _build_identity_payload(account, term, mt5_path, expected, expected_source)
    lock_status, lock_reason = _evaluate_identity_lock(
        identity_payload.get("mt5_account_login"),
        identity_payload.get("mt5_account_server"),
        identity_payload.get("mt5_company"),
        expected,
    )
    identity_payload["lock_status"] = lock_status
    identity_payload["lock_reason"] = lock_reason

    global MT5_IDENTITY, IDENTITY_LOCK, IDENTITY_DATA, LAST_ACCOUNT_INFO, LAST_ACCOUNT_ERROR
    MT5_IDENTITY = identity_payload
    IDENTITY_DATA = identity_payload
    IDENTITY_LOCK = {"status": lock_status, "reason": lock_reason}
    LAST_ACCOUNT_INFO = account
    LAST_ACCOUNT_ERROR = None
    write_identity_file(identity_payload)

    if lock_status == "mismatch":
        log_error(f"IDENTITY LOCK | {lock_reason}")
        log_lifecycle(f"IDENTITY LOCK | {lock_reason}")

    return True, None


def mt5_shutdown() -> None:
    if mt5:
        mt5.shutdown()


def portfolio_from_name(name: str):
    if name == "core":
        return DEFAULT_PORTFOLIO_USDJPY_FASTPASS_CORE
    if name == "v3":
        return DEFAULT_PORTFOLIO_USDJPY_FASTPASS_V3
    if name == "multi":
        return MULTI_PORTFOLIO_SPLASH_PASS
    if name == "scorers":
        return MULTI_PORTFOLIO_SCORERS
    if name == "westbrook":
        return WESTBROOK_PORTFOLIO_USDJPY
    if name == "combined":
        return MULTI_PORTFOLIO_COMBINED
    if name == "research":
        return MULTI_PORTFOLIO_USDJPY_GBPJPY_FASTPASS
    if name == "exp_v2":
        return EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V2
    if name == "exp_v3":
        return EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V3
    raise ValueError(f"Unknown portfolio: {name}")


def iter_profiles(portfolio) -> List[Tuple[StrategyProfile, float]]:
    profiles: List[Tuple[StrategyProfile, float]] = []
    if isinstance(portfolio, PortfolioProfile):
        scales = getattr(portfolio, "risk_scales", None) or [1.0] * len(portfolio.strategies)
        for prof, scale in zip(portfolio.strategies, scales):
            profiles.append((prof, scale))
        return profiles
    if isinstance(portfolio, MultiSymbolPortfolioProfile):
        sym_scales = getattr(portfolio, "symbol_risk_scales", None) or [1.0] * len(portfolio.portfolios)
        for sub, sym_scale in zip(portfolio.portfolios, sym_scales):
            sub_scales = getattr(sub, "risk_scales", None) or [1.0] * len(sub.strategies)
            for prof, scale in zip(sub.strategies, sub_scales):
                profiles.append((prof, scale * sym_scale))
        return profiles
    raise TypeError(f"Unsupported portfolio type: {type(portfolio)}")


def magic_number(strategy_id: str) -> int:
    base = sum(ord(c) for c in strategy_id) % 100000
    return 910000 + base


def _profile_group(portfolio_name: str, profile: StrategyProfile) -> str:
    if portfolio_name != "combined":
        return "default"
    if profile.name == "USDJPY_M5_MomentumPinball_V1":
        return "westbrook"
    return "scorers"


def _is_westbrook_position(pos) -> bool:
    tags = _parse_comment_tags(getattr(pos, "comment", "") or "")
    edge = (tags.get("EDGE") or "").upper()
    lineup = (tags.get("LINEUP") or "").upper()
    if edge == "WE" or lineup == "W":
        return True
    magic = getattr(pos, "magic", None)
    try:
        return magic is not None and int(magic) == magic_number("USDJPY_M5_MomentumPinball_V1")
    except Exception:
        return False


def player_name(strategy_id: str) -> str:
    mapping = {
        "USDJPY_M15_LondonBreakout_V1": "Curry",
        "GBPJPY_M15_LondonBreakout_V1": "Klay",
        "EURUSD_M15_LondonBreakout_V1": "Curry",
        "GBPUSD_M15_LondonBreakout_V1": "Klay",
        "USDJPY_M15_LiquiditySweep_V1": "Kawhi",
        "GBPJPY_M15_LiquiditySweep_V1": "Kawhi (GBP)",
        "EURUSD_M15_LiquiditySweep_V1": "Kawhi",
        "GBPUSD_M15_LiquiditySweep_V1": "Kawhi",
        "USDJPY_M5_MomentumPinball_V1": "Westbrook",
        "USDJPY_M15_TrendKD_V1": "KD",
        "USDJPY_M15_VanVleet_V1": "VanVleet",
        "USDJPY_H1_BigMan_V1": "Big Man",
        "AUDUSD_M15_NYTrendPullback_V1": "NY Pullback",
        "USDCAD_M15_NYTrendPullback_V1": "NY Pullback",
    }
    return mapping.get(strategy_id, strategy_id)


def edge_tag(strategy_id: str) -> str:
    mapping = {
        "USDJPY_M15_LondonBreakout_V1": "Cu",
        "GBPJPY_M15_LondonBreakout_V1": "Kl",
        "EURUSD_M15_LondonBreakout_V1": "Cu",
        "GBPUSD_M15_LondonBreakout_V1": "Kl",
        "USDJPY_M15_LiquiditySweep_V1": "Kw",
        "GBPJPY_M15_LiquiditySweep_V1": "Kg",
        "EURUSD_M15_LiquiditySweep_V1": "Kw",
        "GBPUSD_M15_LiquiditySweep_V1": "Kw",
        "USDJPY_M5_MomentumPinball_V1": "We",
        "USDJPY_M15_TrendKD_V1": "KD",
        "USDJPY_M15_VanVleet_V1": "VV",
        "USDJPY_H1_BigMan_V1": "Bg",
        "AUDUSD_M15_NYTrendPullback_V1": "NY",
        "USDCAD_M15_NYTrendPullback_V1": "NY",
    }
    if strategy_id in mapping:
        return mapping[strategy_id]
    if "LondonBreakout" in strategy_id:
        return "Cu"
    if "LiquiditySweep" in strategy_id:
        return "Kw"
    return "Edge"


def lineup_tag(portfolio: str) -> str:
    mapping = {
        "core": "C",
        "v3": "V",
        "multi": "M",
        "scorers": "S",
        "westbrook": "W",
        "combined": "COMB",
        "research": "RESEARCH",
        "exp_v2": "2",
        "exp_v3": "3",
    }
    return mapping.get(portfolio, portfolio or "Unknown")

def _base36_two(value: int) -> str:
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    idx = value % (36 * 36)
    return digits[idx // 36] + digits[idx % 36]


def _next_play_id() -> str:
    global PLAY_COUNTER
    PLAY_COUNTER = (PLAY_COUNTER + 1) % (36 * 36)
    return _base36_two(PLAY_COUNTER)


def comment_for_trade(
    portfolio: str,
    strategy_id: str,
    play_id: Optional[str] = None,
    edge_override: Optional[str] = None,
    lineup_override: Optional[str] = None,
) -> str:
    edge = edge_override or edge_tag(strategy_id)
    lineup = lineup_override or lineup_tag(portfolio)
    acct = ACCOUNT_ID or "?"
    play = play_id or _next_play_id()
    return f"EDGE={edge}|PLAY={play}|ACCT={acct}|LINEUP={lineup}"


def _build_planned_trade(
    prof: StrategyProfile,
    scale: float,
    equity: float,
    price: float,
    portfolio_name: str,
    research_enabled: bool,
    research_risk_scale: float,
    policy: Dict[str, object],
    dynamic_risk: bool,
    edge_override: Optional[str],
    lineup_override: Optional[str] = None,
) -> PlannedTrade:
    strategy_cfg = replace(
        prof.strategy,
        symbol=prof.symbol_key,
        risk_per_trade_pct=prof.strategy.risk_per_trade_pct
        * scale
        * (research_risk_scale if research_enabled else 1.0)
        * (policy.get("global_risk_scale", 1.0) if dynamic_risk else 1.0),
        reward_per_trade_pct=prof.strategy.reward_per_trade_pct
        * scale
        * (research_risk_scale if research_enabled else 1.0)
        * (policy.get("global_risk_scale", 1.0) if dynamic_risk else 1.0),
    )
    planned = plan_single_trade(
        account_balance=equity,
        current_price=price,
        config=strategy_cfg,
    )
    planned.magic_number = magic_number(prof.name)
    planned.strategy_id = prof.name
    planned.player_name = player_name(prof.name)
    planned.regime = regime_for_profile(prof)
    planned.reason_code = reason_code_for_profile(prof)
    planned.risk_scale = scale
    planned.comment = comment_for_trade(
        portfolio_name,
        prof.name,
        edge_override=edge_override if research_enabled else None,
        lineup_override=lineup_override,
    )
    return planned


def _parse_active_edge_tag() -> Optional[str]:
    raw = os.getenv("OMEGAFX_ACTIVE_EDGE_TAG") or os.getenv("ACTIVE_EDGE_TAG")
    if not raw:
        return None
    raw = raw.strip()
    if raw.upper().startswith("EDGE="):
        raw = raw.split("=", 1)[1].strip()
    return raw or None

def _normalize_research_edge_tag(tag: Optional[str]) -> Optional[str]:
    if not tag:
        return None
    cleaned = tag.strip().lower()
    compact = cleaned.replace(" ", "").replace("-", "").replace("_", "")
    if compact == "fixfade":
        return "fixfade"
    if compact in {"trendpullback", "trendpullbackv1", "trendcontinuation", "trendpb"}:
        return "trendpullback"
    return cleaned


def _parse_active_symbols() -> List[str]:
    raw = os.getenv("OMEGAFX_ACTIVE_SYMBOLS") or os.getenv("ACTIVE_SYMBOLS")
    if not raw:
        return []
    parts = []
    for chunk in raw.replace(";", ",").replace("|", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        for token in chunk.split():
            if token:
                parts.append(token.upper())
    return parts


def _parse_pass_symbols() -> List[str]:
    raw = os.getenv("OMEGAFX_PASS_SYMBOLS") or os.getenv("PASS_SYMBOLS")
    if not raw:
        return []
    parts = []
    for chunk in raw.replace(";", ",").replace("|", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        for token in chunk.split():
            if token:
                parts.append(token.upper())
    return parts


def _parse_bucket_caps(raw: Optional[str] = None) -> Dict[str, int]:
    if raw is None:
        raw = os.getenv("OMEGAFX_BUCKET_CAPS") or os.getenv("BUCKET_CAPS")
    if not raw:
        return {}
    caps: Dict[str, int] = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip().upper()
        val = val.strip()
        if not key or not val:
            continue
        try:
            cap = int(val)
        except Exception:
            continue
        if cap >= 0:
            caps[key] = cap
    return caps


def _symbol_bucket(symbol: str) -> str:
    sym = (symbol or "").upper().strip()
    if len(sym) >= 6:
        return sym[-3:]
    return sym or "UNKNOWN"


def _parse_session_window() -> Optional[Tuple[int, int]]:
    raw = os.getenv("OMEGAFX_SESSION_WINDOW") or os.getenv("SESSION_WINDOW")
    if not raw:
        return None
    raw = raw.strip()
    if "-" not in raw:
        return None
    start_raw, end_raw = raw.split("-", 1)
    def _parse_hhmm(value: str) -> Optional[int]:
        value = value.strip()
        if not value:
            return None
        if ":" in value:
            hh, mm = value.split(":", 1)
        else:
            hh, mm = value[:2], value[2:]
        try:
            h = int(hh)
            m = int(mm)
        except Exception:
            return None
        if h < 0 or h > 23 or m < 0 or m > 59:
            return None
        return h * 60 + m
    start_min = _parse_hhmm(start_raw)
    end_min = _parse_hhmm(end_raw)
    if start_min is None or end_min is None:
        return None
    return start_min, end_min


def _session_allows(now_ts: float, window: Optional[Tuple[int, int]]) -> bool:
    if window is None:
        return True
    start_min, end_min = window
    now_dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)
    now_min = now_dt.hour * 60 + now_dt.minute
    if start_min <= end_min:
        return start_min <= now_min <= end_min
    return now_min >= start_min or now_min <= end_min


def _parse_event_calendar_guard() -> List[str]:
    raw = os.getenv("OMEGAFX_EVENT_CALENDAR_GUARD") or os.getenv("EVENT_CALENDAR_GUARD")
    if not raw:
        return []
    raw = raw.strip()
    if raw.lower() in {"off", "none", "false"}:
        return []
    return [val.strip() for val in raw.split(",") if val.strip()]


def _edge_matches(strategy_id: str, active_tag: str) -> bool:
    if not active_tag:
        return True
    tag = active_tag.strip()
    if not tag:
        return True
    return edge_tag(strategy_id).lower() == tag.lower() or strategy_id.lower() == tag.lower()


def _build_research_profile(symbol: str) -> StrategyProfile:
    return StrategyProfile(
        name=f"RESEARCH_FixFade_{symbol}",
        symbol_key=symbol,
        timeframe="15m",
        strategy=replace(DEFAULT_STRATEGY, symbol=symbol),
        signals=SignalConfig(),
        challenge=DEFAULT_CHALLENGE,
        costs=DEFAULT_COSTS,
        session=DEFAULT_SESSION,
    )

def _build_trendpullback_profile(symbol: str) -> StrategyProfile:
    return StrategyProfile(
        name=f"RESEARCH_TrendPullback_{symbol}",
        symbol_key=symbol,
        timeframe="5m",
        strategy=replace(DEFAULT_STRATEGY, symbol=symbol),
        signals=SignalConfig(),
        challenge=DEFAULT_CHALLENGE,
        costs=DEFAULT_COSTS,
        session=DEFAULT_SESSION,
    )


def _pip_size_for_symbol(symbol: str) -> float:
    symbol = (symbol or "").upper()
    if symbol.endswith("JPY"):
        return 0.01
    if symbol.startswith("XAU"):
        return 0.01
    return 0.0001


def reason_code_for_profile(profile: StrategyProfile) -> str:
    sig = profile.signals
    if isinstance(sig, LondonBreakoutSignalConfig):
        return "breakout"
    if isinstance(sig, LiquiditySweepSignalConfig):
        return "sweep"
    if isinstance(sig, MomentumPinballSignalConfig):
        return "momentum"
    if isinstance(sig, TrendKDSignalConfig):
        return "kd_trend"
    if isinstance(sig, VanVleetSignalConfig):
        return "pullback"
    if isinstance(sig, BigManSignalConfig):
        return "range"
    return "signal"


def regime_for_profile(profile: StrategyProfile) -> str:
    # Minimal placeholder; use session as proxy.
    return "trend"


def _mt5_timeframe(tf: str):
    if mt5 is None:
        return None
    tf = tf.lower()
    mapping = {
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "1h": mt5.TIMEFRAME_H1,
        "h1": mt5.TIMEFRAME_H1,
    }
    return mapping.get(tf, mt5.TIMEFRAME_M15)


def fetch_recent_ohlc(symbol: str, timeframe: str, bars: int = 400) -> Optional[pd.DataFrame]:
    if mt5 is None:
        return None
    tf = _mt5_timeframe(timeframe)
    if tf is None:
        return None
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    return df


def build_signals(profile: StrategyProfile, ohlc_by_tf: Dict[str, pd.DataFrame]) -> pd.Series:
    sig = profile.signals
    tf = profile.timeframe
    if isinstance(sig, MomentumPinballSignalConfig):
        ohlc_m5 = ohlc_by_tf.get("5m")
        ohlc_m15 = ohlc_by_tf.get("15m")
        if ohlc_m5 is None or ohlc_m15 is None:
            return pd.Series(dtype=bool)
        return build_momentum_pinball_signals_m5(ohlc_m5, ohlc_m15, sig, profile.session)
    if isinstance(sig, TrendKDSignalConfig):
        ohlc_m15 = ohlc_by_tf.get("15m")
        if ohlc_m15 is None:
            return pd.Series(dtype=bool)
        return build_trend_kd_signals_m15(ohlc_m15, sig, profile.session)
    if isinstance(sig, VanVleetSignalConfig):
        ohlc_m15 = ohlc_by_tf.get("15m")
        if ohlc_m15 is None:
            return pd.Series(dtype=bool)
        return build_vanvleet_signals_m15(ohlc_m15, sig, profile.session)
    if isinstance(sig, BigManSignalConfig):
        ohlc_h1 = ohlc_by_tf.get("1h")
        if ohlc_h1 is None:
            return pd.Series(dtype=bool)
        return build_bigman_signals_h1(ohlc_h1, sig, profile.session)
    if tf not in ohlc_by_tf:
        return pd.Series(dtype=bool)
    return _build_signals_for_profile(ohlc_by_tf[tf], profile)


def _maybe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd is not None and pd.isna(value):
            return None
    except Exception:
        pass
    return _safe_float(value)


def _build_london_breakout_trace(
    profile: StrategyProfile,
    ohlc: pd.DataFrame,
    entry_idx: int,
    now_ts: float,
    in_cooldown: bool,
    policy: Dict[str, object],
    dynamic_risk: bool,
    risk_scale: float,
) -> Optional[Dict[str, object]]:
    sig_cfg = profile.signals
    if not isinstance(sig_cfg, LondonBreakoutSignalConfig):
        return None
    player = player_name(profile.name)
    if player not in {"Curry", "Klay"}:
        return None
    if ohlc is None or entry_idx < 0 or entry_idx >= len(ohlc):
        return {
            "player": player,
            "strategy_id": profile.name,
            "symbol": profile.symbol_key,
            "status": "no_data",
            "reason": "insufficient_ohlc",
            "evaluated_at_utc": datetime.utcnow().isoformat(),
        }
    entry_time = ohlc.index[entry_idx]
    close_val = _maybe_float(ohlc["close"].iloc[entry_idx])
    session_mask = build_session_mask(ohlc, session=profile.session)
    session_ok = bool(session_mask.iloc[entry_idx]) if not session_mask.empty else False
    session_vals = {
        "now_hour": entry_time.hour,
        "now_weekday": entry_time.weekday(),
        "start_hour": profile.session.start_hour,
        "end_hour": profile.session.end_hour,
        "allowed_weekdays": list(profile.session.allowed_weekdays),
    }

    idx = ohlc.index
    hours = idx.hour
    dates = idx.date
    pre_mask = (hours >= sig_cfg.pre_session_start_hour) & (hours < sig_cfg.pre_session_end_hour)
    pre_session_high = (
        ohlc["high"]
        .where(pre_mask)
        .groupby(dates)
        .transform("max")
        .shift(1)
    )
    pre_high_val = _maybe_float(pre_session_high.iloc[entry_idx])
    pip_size = _pip_size_for_symbol(profile.symbol_key)
    buffer = sig_cfg.breakout_buffer_pips * pip_size
    breakout_level = (pre_high_val + buffer) if pre_high_val is not None else None
    trigger_ok = bool(
        pre_high_val is not None
        and close_val is not None
        and breakout_level is not None
        and close_val > breakout_level
    )

    atr = compute_atr(ohlc, period=sig_cfg.atr_period)
    atr_mask = build_atr_filter(atr, percentile=sig_cfg.atr_filter_percentile)
    atr_min_mask = build_atr_filter(atr, percentile=sig_cfg.atr_min_percentile)
    atr_rising = atr.diff() > 0 if not atr.empty else pd.Series([], dtype=bool)
    atr_val = _maybe_float(atr.iloc[entry_idx]) if not atr.empty else None
    atr_threshold = _maybe_float(atr.quantile(sig_cfg.atr_filter_percentile / 100)) if not atr.empty else None
    atr_min_threshold = _maybe_float(atr.quantile(sig_cfg.atr_min_percentile / 100)) if not atr.empty else None
    atr_mask_val = bool(atr_mask.iloc[entry_idx]) if not atr_mask.empty else False
    atr_min_val = bool(atr_min_mask.iloc[entry_idx]) if not atr_min_mask.empty else False
    atr_rising_val = bool(atr_rising.iloc[entry_idx]) if not atr_rising.empty else False
    atr_vol_ok = atr_rising_val or atr_min_val
    momentum_ok = atr_mask_val and atr_vol_ok

    trend_ok = session_ok
    trend_slope = None
    if sig_cfg.trend_tf.lower() == "h1":
        htf = (
            ohlc.resample("1h")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
        )
        if not htf.empty:
            ema = htf["close"].ewm(span=50, adjust=False).mean()
            slope = ema.diff()
            trend_series = (slope > sig_cfg.trend_strength_min).reindex(ohlc.index, method="ffill").fillna(False)
            slope_series = slope.reindex(ohlc.index, method="ffill")
            trend_ok = bool(trend_series.iloc[entry_idx])
            trend_slope = _maybe_float(slope_series.iloc[entry_idx])

    cooldown_ok = not in_cooldown
    cooldown_until = policy.get("cooldown_until") if dynamic_risk else None
    cooldown_remaining = None
    if cooldown_until and isinstance(cooldown_until, (int, float)) and cooldown_until > now_ts:
        cooldown_remaining = float(cooldown_until - now_ts)

    symbol_enabled = True
    if dynamic_risk:
        symbol_enabled = bool(policy.get("symbol_enabled", {}).get(profile.symbol_key, True))
    risk_ok = symbol_enabled

    return {
        "player": player,
        "strategy_id": profile.name,
        "symbol": profile.symbol_key,
        "entry_time": entry_time.isoformat(),
        "evaluated_at_utc": datetime.utcnow().isoformat(),
        "session_ok": session_ok,
        "cooldown_ok": cooldown_ok,
        "trend_ok": trend_ok,
        "momentum_ok": momentum_ok,
        "trigger_ok": trigger_ok,
        "risk_ok": risk_ok,
        "values": {
            "session": session_vals,
            "cooldown": {
                "in_cooldown": in_cooldown,
                "cooldown_until": cooldown_until,
                "cooldown_remaining_sec": cooldown_remaining,
            },
            "trend": {
                "trend_tf": sig_cfg.trend_tf,
                "trend_slope": trend_slope,
                "trend_strength_min": sig_cfg.trend_strength_min,
            },
            "momentum": {
                "atr": atr_val,
                "atr_period": sig_cfg.atr_period,
                "atr_percentile": sig_cfg.atr_filter_percentile,
                "atr_threshold": atr_threshold,
                "atr_min_percentile": sig_cfg.atr_min_percentile,
                "atr_min_threshold": atr_min_threshold,
                "atr_rising": atr_rising_val,
                "atr_mask": atr_mask_val,
                "atr_vol_ok": atr_vol_ok,
            },
            "trigger": {
                "pre_session_high": pre_high_val,
                "close": close_val,
                "buffer_pips": sig_cfg.breakout_buffer_pips,
                "buffer_price": buffer,
                "breakout_level": breakout_level,
            },
            "risk": {
                "symbol_enabled": symbol_enabled,
                "global_risk_scale": policy.get("global_risk_scale") if dynamic_risk else None,
                "risk_per_trade_pct": _safe_float(profile.strategy.risk_per_trade_pct),
                "risk_scale": _safe_float(risk_scale),
            },
        },
    }


def _build_trend_continuation_trace(
    profile: StrategyProfile,
    ohlc: pd.DataFrame,
    entry_idx: int,
    now_ts: float,
    in_cooldown: bool,
    policy: Dict[str, object],
    dynamic_risk: bool,
    risk_scale: float,
) -> Optional[Dict[str, object]]:
    sig_cfg = profile.signals
    if not isinstance(sig_cfg, TrendContinuationSignalConfig):
        return None
    if "NYTrendPullback" not in profile.name:
        return None
    if ohlc is None or entry_idx < 0 or entry_idx >= len(ohlc):
        return {
            "player": player_name(profile.name),
            "strategy_id": profile.name,
            "symbol": profile.symbol_key,
            "status": "no_data",
            "reason": "insufficient_ohlc",
            "evaluated_at_utc": datetime.utcnow().isoformat(),
        }
    entry_time = ohlc.index[entry_idx]
    close = ohlc["close"].astype(float)
    ma_fast = close.rolling(sig_cfg.fast_ma_period).mean()
    ma_slow = close.rolling(sig_cfg.slow_ma_period).mean()
    uptrend = ma_fast > ma_slow
    prev_close = close.shift(1)
    prev_ma_fast = ma_fast.shift(1)
    crossed_up = (close > ma_fast) & (prev_close <= prev_ma_fast)
    atr = compute_atr(ohlc, period=sig_cfg.atr_period)
    atr_mask = build_atr_filter(atr, percentile=sig_cfg.atr_percentile)
    h4_up = compute_h4_sma_filter(ohlc, sma_period=sig_cfg.h4_sma_period)
    session_mask = build_session_mask(ohlc, session=profile.session)

    session_ok = bool(session_mask.iloc[entry_idx]) if not session_mask.empty else False
    trend_ok = bool(uptrend.iloc[entry_idx]) if not uptrend.empty else False
    trigger_ok = bool(crossed_up.iloc[entry_idx]) if not crossed_up.empty else False
    atr_ok = bool(atr_mask.iloc[entry_idx]) if not atr_mask.empty else False
    h4_ok = bool(h4_up.iloc[entry_idx]) if not h4_up.empty else False
    momentum_ok = atr_ok and h4_ok
    cooldown_ok = not in_cooldown

    cooldown_until = policy.get("cooldown_until") if dynamic_risk else None
    cooldown_remaining = None
    if cooldown_until and isinstance(cooldown_until, (int, float)) and cooldown_until > now_ts:
        cooldown_remaining = float(cooldown_until - now_ts)

    symbol_enabled = True
    if dynamic_risk:
        symbol_enabled = bool(policy.get("symbol_enabled", {}).get(profile.symbol_key, True))
    risk_ok = symbol_enabled

    return {
        "player": player_name(profile.name),
        "strategy_id": profile.name,
        "symbol": profile.symbol_key,
        "entry_time": entry_time.isoformat(),
        "evaluated_at_utc": datetime.utcnow().isoformat(),
        "session_ok": session_ok,
        "cooldown_ok": cooldown_ok,
        "trend_ok": trend_ok,
        "momentum_ok": momentum_ok,
        "trigger_ok": trigger_ok,
        "risk_ok": risk_ok,
        "values": {
            "session": {
                "now_hour": entry_time.hour,
                "now_weekday": entry_time.weekday(),
                "start_hour": profile.session.start_hour,
                "end_hour": profile.session.end_hour,
                "allowed_weekdays": list(profile.session.allowed_weekdays),
            },
            "cooldown": {
                "in_cooldown": in_cooldown,
                "cooldown_until": cooldown_until,
                "cooldown_remaining_sec": cooldown_remaining,
            },
            "trend": {
                "fast_ma": _maybe_float(ma_fast.iloc[entry_idx]) if not ma_fast.empty else None,
                "slow_ma": _maybe_float(ma_slow.iloc[entry_idx]) if not ma_slow.empty else None,
                "fast_ma_period": sig_cfg.fast_ma_period,
                "slow_ma_period": sig_cfg.slow_ma_period,
            },
            "momentum": {
                "atr": _maybe_float(atr.iloc[entry_idx]) if not atr.empty else None,
                "atr_period": sig_cfg.atr_period,
                "atr_percentile": sig_cfg.atr_percentile,
                "atr_threshold": _maybe_float(atr.quantile(sig_cfg.atr_percentile / 100)) if not atr.empty else None,
                "atr_mask": atr_ok,
                "h4_up": h4_ok,
            },
            "trigger": {
                "close": _maybe_float(close.iloc[entry_idx]) if not close.empty else None,
                "ma_fast": _maybe_float(ma_fast.iloc[entry_idx]) if not ma_fast.empty else None,
                "crossed_up": trigger_ok,
            },
            "risk": {
                "symbol_enabled": symbol_enabled,
                "global_risk_scale": policy.get("global_risk_scale") if dynamic_risk else None,
                "risk_per_trade_pct": _safe_float(profile.strategy.risk_per_trade_pct),
                "risk_scale": _safe_float(risk_scale),
            },
        },
    }


TRADE_LOG_FIELDS = [
    "stage",
    "execution_label",
    "account_id",
    "account_login",
    "account_server",
    "mt5_path",
    "mode",
    "portfolio",
    "ticket",
    "ticket_id",
    "mt5_order_ticket",
    "mt5_deal_ticket",
    "mt5_position_ticket",
    "mt5_retcode",
    "mt5_last_error",
    "mt5_comment",
    "magic_number",
    "comment",
    "profile_name",
    "symbol",
    "entry_time",
    "exit_time",
    "close_time",
    "entry_date",
    "direction",
    "lot_size",
    "entry_price",
    "exit_price",
    "sl_price",
    "tp_price",
    "exit_reason",
    "realized_profit",
    "pnl",
    "pnl_pct",
    "win_loss",
    "equity_before",
    "equity_after",
    "run_id",
]

PAPER_SHOT_FIELDS = [
    "timestamp_utc",
    "entry_time",
    "entry_date",
    "mode",
    "portfolio",
    "profile_name",
    "symbol",
    "player",
    "edge_tag",
    "comment",
    "direction",
    "entry_price",
    "sl_price",
    "tp_price",
    "risk_pct",
    "risk_amount",
    "risk_scale",
    "session_ok",
    "cooldown_ok",
    "trend_ok",
    "momentum_ok",
    "trigger_ok",
    "risk_ok",
    "reason",
    "failed_gates",
    "gate_values_json",
]


def _mt5_context_fields(account_info=None) -> Dict[str, Optional[str]]:
    login = getattr(account_info, "login", None) if account_info is not None else None
    server = getattr(account_info, "server", None) if account_info is not None else None
    if login is None:
        login = MT5_IDENTITY.get("mt5_account_login")
    if server is None:
        server = MT5_IDENTITY.get("mt5_account_server")
    mt5_path = MT5_PATH_USED or os.getenv("MT5_PATH") or os.getenv("OMEGAFX_MT5_PATH")
    return {
        "account_login": str(login) if login is not None else None,
        "account_server": server,
        "mt5_path": mt5_path,
    }


def _execution_label(stage: str, deal_ticket=None, position_ticket=None) -> str:
    if stage in {"DEAL_EXECUTED", "POSITION_OPENED"} and (deal_ticket or position_ticket):
        return "EXECUTED"
    return "ATTEMPT"


def append_trade_log(path: Path, row: Dict[str, object]) -> None:
    fieldnames = TRADE_LOG_FIELDS
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_equity_log(path: Path, equity: float, mode: str, portfolio: str) -> None:
    fieldnames = ["time", "equity", "mode", "portfolio", "account_id"]
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "time": datetime.utcnow().isoformat(),
            "equity": equity,
            "mode": mode,
            "portfolio": portfolio,
            "account_id": ACCOUNT_ID,
        })


def ensure_trade_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                existing_fields = reader.fieldnames or []
                rows = list(reader)
        except Exception:
            return
        if set(TRADE_LOG_FIELDS).issubset(set(existing_fields)):
            return
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in TRADE_LOG_FIELDS})
        tmp_path.replace(path)
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
        writer.writeheader()


def ensure_paper_shot_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAPER_SHOT_FIELDS)
        writer.writeheader()


def append_paper_shot(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAPER_SHOT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in PAPER_SHOT_FIELDS})


def _paper_shot_failed_gates(trace: Optional[Dict[str, object]]) -> List[str]:
    if not trace:
        return []
    failed = []
    for key in ("session_ok", "cooldown_ok", "trend_ok", "momentum_ok", "risk_ok"):
        if trace.get(key) is False:
            failed.append(key)
    return failed


def _append_paper_shot(
    path: Path,
    mode: str,
    portfolio: str,
    prof: StrategyProfile,
    planned: PlannedTrade,
    entry_time: datetime,
    trace: Optional[Dict[str, object]],
    reason: str,
) -> None:
    failed_gates = _paper_shot_failed_gates(trace)
    row = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "entry_time": entry_time.isoformat(),
        "entry_date": entry_time.date().isoformat(),
        "mode": mode,
        "portfolio": portfolio,
        "profile_name": prof.name,
        "symbol": prof.symbol_key,
        "player": planned.player_name or player_name(prof.name),
        "edge_tag": edge_tag(prof.name),
        "comment": planned.comment or "",
        "direction": planned.direction,
        "entry_price": planned.entry_price,
        "sl_price": planned.stop_loss_price,
        "tp_price": planned.take_profit_price,
        "risk_pct": planned.risk_pct,
        "risk_amount": planned.risk_amount,
        "risk_scale": planned.risk_scale,
        "session_ok": trace.get("session_ok") if trace else "",
        "cooldown_ok": trace.get("cooldown_ok") if trace else "",
        "trend_ok": trace.get("trend_ok") if trace else "",
        "momentum_ok": trace.get("momentum_ok") if trace else "",
        "trigger_ok": trace.get("trigger_ok") if trace else "",
        "risk_ok": trace.get("risk_ok") if trace else "",
        "reason": reason,
        "failed_gates": ",".join(failed_gates),
        "gate_values_json": json.dumps(trace.get("values", {}), ensure_ascii=True) if trace else "",
    }
    append_paper_shot(path, row)


def ensure_equity_log(path: Path, equity: float, mode: str, portfolio: str) -> None:
    if path.exists():
        return
    append_equity_log(path, equity, mode, portfolio)


def _default_trade_log(mode: str, portfolio: str) -> Path:
    if mode == "demo":
        if portfolio == "multi":
            return LOG_DIR / "demo_multi_trades.csv"
        if portfolio == "scorers":
            return LOG_DIR / "demo_scorers_trades.csv"
        if portfolio == "westbrook":
            return LOG_DIR / "demo_westbrook_trades.csv"
        if portfolio == "combined":
            return LOG_DIR / "demo_combined_trades.csv"
        return LOG_DIR / "demo_usdjpy_v3_trades.csv"
    if mode == "shadow":
        if portfolio == "multi":
            return LOG_DIR / "shadow_multi_trades.csv"
        if portfolio == "scorers":
            return LOG_DIR / "shadow_scorers_trades.csv"
        if portfolio == "westbrook":
            return LOG_DIR / "shadow_westbrook_trades.csv"
        if portfolio == "combined":
            return LOG_DIR / "shadow_combined_trades.csv"
        return LOG_DIR / "shadow_fastpass_usdjpy_core.csv"
    return LOG_DIR / "live_trades.csv"


def _default_equity_log(mode: str, portfolio: str) -> Path:
    if mode == "demo":
        if portfolio == "multi":
            return LOG_DIR / "demo_multi_equity.csv"
        if portfolio == "scorers":
            return LOG_DIR / "demo_scorers_equity.csv"
        if portfolio == "westbrook":
            return LOG_DIR / "demo_westbrook_equity.csv"
        if portfolio == "combined":
            return LOG_DIR / "demo_combined_equity.csv"
        return LOG_DIR / "demo_usdjpy_v3_equity.csv"
    if mode == "shadow":
        if portfolio == "multi":
            return LOG_DIR / "shadow_multi_equity.csv"
        if portfolio == "scorers":
            return LOG_DIR / "shadow_scorers_equity.csv"
        if portfolio == "westbrook":
            return LOG_DIR / "shadow_westbrook_equity.csv"
        if portfolio == "combined":
            return LOG_DIR / "shadow_combined_equity.csv"
        return LOG_DIR / "shadow_fastpass_usdjpy_core_equity.csv"
    return LOG_DIR / "live_equity.csv"


def append_attribution(ticket: Optional[int], profile: StrategyProfile, regime: str, reason: str) -> None:
    fieldnames = ["timestamp", "ticket", "symbol", "player", "strategy_id", "regime", "reason_code", "account_id"]
    write_header = not ATTR_LOG.exists()
    with ATTR_LOG.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "ticket": ticket or "",
                "symbol": profile.symbol_key,
                "player": player_name(profile.name),
                "strategy_id": profile.name,
                "regime": regime,
                "reason_code": reason,
                "account_id": ACCOUNT_ID,
            }
        )


def _parse_comment_strategy(comment: str) -> Optional[str]:
    if not comment:
        return None
    if "strategy=" in comment:
        for part in comment.split("|"):
            if part.startswith("strategy="):
                return part.split("=", 1)[1].strip() or None
    if "EDGE=" in comment:
        for part in comment.split("|"):
            if part.startswith("EDGE="):
                return part.split("=", 1)[1].strip() or None
    return None


def _is_bot_comment(comment: str) -> bool:
    if not comment:
        return False
    if "OmegaFX|" in comment:
        return True
    if "EDGE=" in comment and "|ACCT=" in comment and "|LINEUP=" in comment:
        return True
    return False


def _parse_comment_tags(comment: str) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    if not comment:
        return tags
    for part in comment.split("|"):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip().upper()
        val = val.strip()
        if key in {"EDGE", "PLAY", "ACCT", "LINEUP"} and val:
            tags[key] = val
    return tags


def _append_stage_log(
    trade_log: Path,
    stage: str,
    mode: str,
    portfolio: str,
    prof: Optional[StrategyProfile],
    planned: Optional[PlannedTrade],
    entry_time: Optional[datetime],
    account_info=None,
    mt5_order_ticket: Optional[object] = None,
    mt5_deal_ticket: Optional[object] = None,
    mt5_position_ticket: Optional[object] = None,
    mt5_retcode: Optional[object] = None,
    mt5_last_error: Optional[str] = None,
    mt5_comment: Optional[str] = None,
    realized_profit: Optional[float] = None,
    pnl_pct: Optional[float] = None,
    exit_price: Optional[float] = None,
    close_time: Optional[str] = None,
    exit_reason: Optional[str] = None,
) -> None:
    identity = _mt5_context_fields(account_info)
    comment = planned.comment if planned is not None else None
    if not comment and mt5_comment:
        comment = mt5_comment
    entry_time_str = entry_time.isoformat() if entry_time else ""
    entry_date = entry_time.date().isoformat() if entry_time else ""
    execution_label = _execution_label(stage, mt5_deal_ticket, mt5_position_ticket)
    pnl_value = realized_profit if realized_profit is not None else ""
    row = {
        "stage": stage,
        "execution_label": execution_label,
        "account_id": ACCOUNT_ID,
        "account_login": identity.get("account_login"),
        "account_server": identity.get("account_server"),
        "mt5_path": identity.get("mt5_path"),
        "mode": mode,
        "portfolio": portfolio,
        "ticket": str(mt5_order_ticket) if mt5_order_ticket else "",
        "ticket_id": str(mt5_deal_ticket) if mt5_deal_ticket else "",
        "mt5_order_ticket": str(mt5_order_ticket) if mt5_order_ticket else "",
        "mt5_deal_ticket": str(mt5_deal_ticket) if mt5_deal_ticket else "",
        "mt5_position_ticket": str(mt5_position_ticket) if mt5_position_ticket else "",
        "mt5_retcode": mt5_retcode,
        "mt5_last_error": mt5_last_error,
        "mt5_comment": mt5_comment or "",
        "magic_number": planned.magic_number if planned is not None else "",
        "comment": comment or "",
        "profile_name": prof.name if prof is not None else "",
        "symbol": prof.symbol_key if prof is not None else (planned.symbol if planned is not None else ""),
        "entry_time": entry_time_str,
        "exit_time": "",
        "close_time": close_time or "",
        "entry_date": entry_date,
        "direction": planned.direction if planned is not None else "",
        "lot_size": planned.lot_size if planned is not None else "",
        "entry_price": planned.entry_price if planned is not None else "",
        "exit_price": exit_price if exit_price is not None else "",
        "sl_price": planned.stop_loss_price if planned is not None else "",
        "tp_price": planned.take_profit_price if planned is not None else "",
        "exit_reason": exit_reason or "",
        "realized_profit": realized_profit if realized_profit is not None else "",
        "pnl": pnl_value,
        "pnl_pct": pnl_pct if pnl_pct is not None else "",
        "win_loss": "",
        "equity_before": "",
        "equity_after": "",
        "run_id": "",
    }
    append_trade_log(trade_log, row)


def _append_deal_execution_log(
    trade_log: Path,
    deal,
    mode: str,
    portfolio: str,
    account_info=None,
    equity_hint: Optional[float] = None,
) -> bool:
    if deal is None:
        return False
    deal_ticket = getattr(deal, "ticket", None)
    deal_order = getattr(deal, "order", None)
    deal_position = getattr(deal, "position_id", None)
    deal_comment = (getattr(deal, "comment", "") or "").strip()
    deal_time = datetime.fromtimestamp(getattr(deal, "time", 0), tz=timezone.utc).isoformat()
    deal_price = _safe_float(getattr(deal, "price", None))
    deal_profit = _safe_float(getattr(deal, "profit", None))
    deal_type = getattr(deal, "type", None)
    direction = ""
    if mt5 is not None:
        if deal_type == getattr(mt5, "DEAL_TYPE_BUY", None):
            direction = "long"
        elif deal_type == getattr(mt5, "DEAL_TYPE_SELL", None):
            direction = "short"
    tags = _parse_comment_tags(deal_comment)
    prof_name = tags.get("EDGE", "")
    planned = PlannedTrade(
        symbol=getattr(deal, "symbol", "") or "",
        direction=direction,
        lot_size=_safe_float(getattr(deal, "volume", None)) or 0.0,
        account_balance=0.0,
        risk_amount=0.0,
        risk_pct=0.0,
        reward_amount=0.0,
        reward_pct=0.0,
        account_profit_target=0.0,
        entry_price=deal_price or 0.0,
        stop_loss_price=0.0,
        take_profit_price=0.0,
        risk_pips=0.0,
        reward_pips=0.0,
        comment=deal_comment or "",
        strategy_id=prof_name or "",
        player_name=prof_name or "",
    )
    planned.magic_number = getattr(deal, "magic", None)
    pnl_pct = None
    if equity_hint is not None and deal_profit is not None and equity_hint != 0:
        pnl_pct = (deal_profit / equity_hint) * 100.0
    exit_reason = "deal"
    if mt5 is not None:
        entry_val = getattr(deal, "entry", None)
        if entry_val == getattr(mt5, "DEAL_ENTRY_IN", None):
            exit_reason = "deal_in"
        elif entry_val == getattr(mt5, "DEAL_ENTRY_OUT", None):
            exit_reason = "deal_out"
    _append_stage_log(
        trade_log,
        "DEAL_EXECUTED",
        mode,
        portfolio,
        None,
        planned,
        datetime.fromisoformat(deal_time),
        account_info=account_info,
        mt5_order_ticket=deal_order,
        mt5_deal_ticket=deal_ticket,
        mt5_position_ticket=deal_position,
        mt5_comment=deal_comment,
        realized_profit=deal_profit,
        pnl_pct=pnl_pct,
        exit_price=deal_price,
        close_time=deal_time,
        exit_reason=exit_reason,
    )
    return True


def _trade_log_counts(trade_log: Path) -> Tuple[int, int]:
    if not trade_log.exists():
        return 0, 0
    try:
        import csv
        with trade_log.open(newline="") as f:
            reader = csv.DictReader(f)
            total = 0
            matched = 0
            for row in reader:
                total += 1
                stage = (row.get("stage") or "").upper()
                has_ticket = bool(
                    row.get("mt5_deal_ticket")
                    or row.get("mt5_position_ticket")
                    or row.get("ticket_id")
                    or row.get("ticket")
                )
                executed = stage in {"DEAL_EXECUTED", "POSITION_OPENED"} or (
                    not stage and has_ticket
                )
                if executed:
                    matched += 1
        return total, matched
    except Exception:
        return 0, 0


def _positions_snapshot():
    if mt5 is None:
        return [], {}
    positions = mt5.positions_get()
    if positions is None:
        return [], {}
    open_map: Dict[Tuple[str, str], int] = {}
    bucket_map: Dict[Tuple[str, str], int] = {}
    westbrook_open: Dict[Tuple[str, str], int] = {}
    westbrook_bucket: Dict[Tuple[str, str], int] = {}
    westbrook_total = 0
    for p in positions:
        symbol = getattr(p, "symbol", "")
        ptype = getattr(p, "type", None)
        direction = "long" if ptype == getattr(mt5, "POSITION_TYPE_BUY", 0) else "short"
        key = (symbol, direction)
        open_map[key] = open_map.get(key, 0) + 1
        bucket = _symbol_bucket(symbol)
        bkey = (bucket, direction)
        bucket_map[bkey] = bucket_map.get(bkey, 0) + 1
        if _is_westbrook_position(p):
            westbrook_total += 1
            westbrook_open[key] = westbrook_open.get(key, 0) + 1
            westbrook_bucket[bkey] = westbrook_bucket.get(bkey, 0) + 1
    return positions, {
        "symbol_dir": open_map,
        "bucket_dir": bucket_map,
        "westbrook_symbol_dir": westbrook_open,
        "westbrook_bucket_dir": westbrook_bucket,
        "westbrook_total": westbrook_total,
    }


def _mt5_tick_snapshot(
    symbols: List[str],
    now_ts: float,
) -> Tuple[Dict[str, object], str, Optional[str]]:
    if not symbols:
        return {}, "missing", "no_symbols"
    if mt5 is None:
        return {}, "missing", "mt5_not_installed"
    if MT5_INIT_OK is not True:
        return {}, "disconnected", "mt5_not_initialized"
    age_map: Dict[str, float] = {}
    time_map: Dict[str, str] = {}
    missing: List[str] = []
    for sym in symbols:
        if not mt5.symbol_select(sym, True):
            missing.append(sym)
            continue
        tick = mt5.symbol_info_tick(sym)
        if tick is None:
            missing.append(sym)
            continue
        tick_time = getattr(tick, "time_msc", None)
        if tick_time is not None and tick_time > 0:
            tick_sec = float(tick_time) / 1000.0
        else:
            raw_time = getattr(tick, "time", None)
            tick_sec = float(raw_time) if raw_time else None
        if not tick_sec:
            missing.append(sym)
            continue
        age_map[sym] = max(0.0, now_ts - tick_sec)
        time_map[sym] = datetime.fromtimestamp(tick_sec, tz=timezone.utc).isoformat()
    payload = {
        "age_sec_by_symbol": age_map,
        "time_by_symbol": time_map,
        "missing_symbols": missing,
        "stale_seconds": TICK_STALE_SECONDS,
    }
    if not age_map and missing:
        return payload, "missing", "ticks_missing"
    if missing:
        return payload, "partial", f"missing_ticks:{','.join(missing)}"
    return payload, "ok", None


def _mt5_truth_snapshot(
    now_ts: float,
    positions: Optional[list] = None,
) -> Tuple[Dict[str, object], str, Optional[str]]:
    if mt5 is None:
        return {}, "missing", "mt5_not_installed"
    if MT5_INIT_OK is not True:
        return {}, "disconnected", "mt5_not_initialized"
    positions_list = positions
    if positions_list is None:
        positions_list = mt5.positions_get()
    if positions_list is None:
        return {}, "error", f"positions_get_failed:{mt5.last_error()}"
    orders = mt5.orders_get()
    if orders is None:
        return {}, "error", f"orders_get_failed:{mt5.last_error()}"
    to_dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)
    from_dt = datetime.fromtimestamp(now_ts - 7200, tz=timezone.utc)
    deals = mt5.history_deals_get(from_dt, to_dt)
    if deals is None:
        return {}, "error", f"deals_get_failed:{mt5.last_error()}"
    positions_profit = 0.0
    for pos in positions_list:
        try:
            positions_profit += float(getattr(pos, "profit", 0) or 0)
        except Exception:
            continue
    last_deal_time = None
    if deals:
        last_deal = max(deals, key=lambda d: getattr(d, "time", 0))
        last_deal_time = datetime.fromtimestamp(last_deal.time, tz=timezone.utc).isoformat()
    payload = {
        "positions_count": len(positions_list),
        "positions_profit": positions_profit,
        "orders_count": len(orders),
        "deals_last_2h_count": len(deals),
        "last_deal_time": last_deal_time,
    }
    return payload, "ok", None


def _mt5_research_packet(
    now_ts: float,
    account,
    positions_list: Optional[list],
) -> Tuple[Dict[str, object], str, Optional[str]]:
    if mt5 is None:
        return {}, "missing", "mt5_not_installed"
    if MT5_INIT_OK is not True:
        return {}, "disconnected", "mt5_not_initialized"
    acct = account or mt5.account_info()
    if acct is None:
        return {}, "error", f"account_info_failed:{mt5.last_error()}"
    positions = positions_list if positions_list is not None else mt5.positions_get()
    if positions is None:
        return {}, "error", f"positions_get_failed:{mt5.last_error()}"
    start_dt = datetime.fromtimestamp(now_ts, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    to_dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)
    deals = mt5.history_deals_get(start_dt, to_dt)
    if deals is None:
        return {}, "error", f"deals_get_failed:{mt5.last_error()}"
    payload = {
        "packet_version": "play_packet_v1",
        "generated_at_utc": datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat(),
        "account": {
            "login": getattr(acct, "login", None),
            "server": getattr(acct, "server", None),
            "company": getattr(acct, "company", None),
            "balance": _safe_float(getattr(acct, "balance", None)),
            "equity": _safe_float(getattr(acct, "equity", None)),
            "currency": getattr(acct, "currency", None),
            "leverage": getattr(acct, "leverage", None),
        },
        "positions": [
            {
                "ticket": getattr(p, "ticket", None),
                "symbol": getattr(p, "symbol", None),
                "type": getattr(p, "type", None),
                "volume": _safe_float(getattr(p, "volume", None)),
                "price_open": _safe_float(getattr(p, "price_open", None)),
                "sl": _safe_float(getattr(p, "sl", None)),
                "tp": _safe_float(getattr(p, "tp", None)),
                "profit": _safe_float(getattr(p, "profit", None)),
                "comment": getattr(p, "comment", None),
                "magic": getattr(p, "magic", None),
                "tags": _parse_comment_tags(getattr(p, "comment", "") or ""),
            }
            for p in positions
        ],
        "deals_since_day_start": [
            {
                "ticket": getattr(d, "ticket", None),
                "order": getattr(d, "order", None),
                "position_id": getattr(d, "position_id", None),
                "symbol": getattr(d, "symbol", None),
                "type": getattr(d, "type", None),
                "entry": getattr(d, "entry", None),
                "time": datetime.fromtimestamp(getattr(d, "time", 0), tz=timezone.utc).isoformat(),
                "price": _safe_float(getattr(d, "price", None)),
                "profit": _safe_float(getattr(d, "profit", None)),
                "volume": _safe_float(getattr(d, "volume", None)),
                "comment": getattr(d, "comment", None),
                "magic": getattr(d, "magic", None),
                "tags": _parse_comment_tags(getattr(d, "comment", "") or ""),
            }
            for d in deals
        ],
    }
    return payload, "ok", None


def run():
    mode = os.getenv("OMEGAFX_MODE", "shadow").lower()
    portfolio_name = os.getenv("OMEGAFX_PORTFOLIO", "core").lower()
    trade_log = Path(os.getenv("OMEGAFX_TRADE_LOG", str(_default_trade_log(mode, portfolio_name))))
    equity_log = Path(os.getenv("OMEGAFX_EQUITY_LOG", str(_default_equity_log(mode, portfolio_name))))
    poll_seconds = int(os.getenv("OMEGAFX_POLL_SECONDS", "30"))
    heartbeat_seconds = int(os.getenv("OMEGAFX_HEARTBEAT_SECONDS", "30"))
    initial_equity = float(os.getenv("OMEGAFX_INITIAL_EQUITY", "10000"))
    dynamic_risk = os.getenv("DYNAMIC_RISK", "0").lower() in {"1", "true", "yes"}

    if "OMEGAFX_LIVE_MODE" not in os.environ or not os.environ.get("OMEGAFX_LIVE_MODE", "").strip():
        os.environ["OMEGAFX_LIVE_MODE"] = "0" if mode == "shadow" else "1"
    global DRY_RUN_ENABLED
    global LAST_TERMINAL_INFO, LAST_TERMINAL_TRADE_ALLOWED, LAST_ACCOUNT_TRADE_ALLOWED
    global LAST_MT5_TRADE_ALLOWED, LAST_ORDER_SEND_RESULT, LAST_ORDER_SEND_ERROR, ORDER_SEND_NONE_COUNT
    global MT5_OPEN_POSITIONS_COUNT, MT5_VERIFIED_CLOSED_TRADES_COUNT
    global LOG_EVENTS_TOTAL, MATCHED_MT5_TICKETS_COUNT, UNMATCHED_LOG_ONLY_COUNT, EVIDENCE_TIER
    DRY_RUN_ENABLED = os.getenv("OMEGAFX_LIVE_MODE", "0") != "1"

    portfolio = portfolio_from_name(portfolio_name)
    ensure_trade_log(trade_log)
    ensure_equity_log(equity_log, initial_equity, mode, portfolio_name)
    profiles = iter_profiles(portfolio)
    research_enabled = portfolio_name == "research"
    paper_shots_enabled = os.getenv("OMEGAFX_SHADOW_SHOTS", "").lower() in {"1", "true", "yes"}
    if not paper_shots_enabled:
        paper_shots_enabled = os.getenv("OMEGAFX_PAPER_SHOTS", "").lower() in {"1", "true", "yes"}
    if paper_shots_enabled:
        ensure_paper_shot_log(PAPER_SHOTS_LOG)
    active_edge_tag = _parse_active_edge_tag()
    active_edge_tag_norm = _normalize_research_edge_tag(active_edge_tag)
    active_symbols = _parse_active_symbols()
    pass_symbols = _parse_pass_symbols()
    session_window_raw = os.getenv("OMEGAFX_SESSION_WINDOW") or os.getenv("SESSION_WINDOW")
    session_window_raw = session_window_raw.strip() if session_window_raw else None
    session_window = _parse_session_window()
    event_guard_days = _parse_event_calendar_guard()
    try:
        research_risk_scale = float(os.getenv("OMEGAFX_RESEARCH_RISK_SCALE", "0.1"))
    except Exception:
        research_risk_scale = 0.1
    try:
        research_risk_cap_pct = float(os.getenv("OMEGAFX_RESEARCH_RISK_CAP_PCT", "0.001"))
    except Exception:
        research_risk_cap_pct = 0.001
    try:
        research_daily_loss_pct = float(os.getenv("OMEGAFX_RESEARCH_DAILY_LOSS_PCT", "0.002"))
    except Exception:
        research_daily_loss_pct = 0.002
    try:
        fixfade_shove_pips = float(os.getenv("OMEGAFX_FIXFADE_SHOVE_PIPS", "15"))
    except Exception:
        fixfade_shove_pips = 15.0
    try:
        fixfade_retrace_pips = float(os.getenv("OMEGAFX_FIXFADE_RETRACE_PIPS", "7"))
    except Exception:
        fixfade_retrace_pips = 7.0
    try:
        fixfade_band_pips = float(os.getenv("OMEGAFX_FIXFADE_ANCHOR_BAND_PIPS", "3"))
    except Exception:
        fixfade_band_pips = 3.0
    try:
        fixfade_sl_buffer_pips = float(os.getenv("OMEGAFX_FIXFADE_SL_BUFFER_PIPS", "2"))
    except Exception:
        fixfade_sl_buffer_pips = 2.0
    trendpb_fast_span = 20
    trendpb_slow_span = 50
    trendpb_swing_lookback = 8
    trendpb_sl_buffer_pips = 2.0
    trendpb_r_multiple = 3.0
    research_static_block_reason = None
    fixfade_enabled = research_enabled and (active_edge_tag_norm or "") == "fixfade"
    trendpb_enabled = research_enabled and (active_edge_tag_norm or "") == "trendpullback"
    if research_enabled:
        if not active_edge_tag:
            research_static_block_reason = "missing_active_edge_tag"
        elif not fixfade_enabled and not trendpb_enabled:
            research_static_block_reason = "unsupported_edge"
        if not active_symbols:
            research_static_block_reason = research_static_block_reason or "missing_active_symbols"
        if active_symbols and len(active_symbols) > 2:
            research_static_block_reason = "too_many_active_symbols"
        if fixfade_enabled and active_symbols:
            profiles = [(_build_research_profile(sym), 1.0) for sym in active_symbols]
        elif trendpb_enabled and active_symbols:
            profiles = [(_build_trendpullback_profile(sym), 1.0) for sym in active_symbols]
        else:
            if active_symbols:
                profiles = [
                    (p, scale) for p, scale in profiles if p.symbol_key.upper() in active_symbols
                ]
            if active_edge_tag:
                profiles = [
                    (p, scale) for p, scale in profiles if _edge_matches(p.name, active_edge_tag)
                ]
            if not profiles and research_static_block_reason is None:
                research_static_block_reason = "no_matching_profiles"
        if (fixfade_enabled or trendpb_enabled) and session_window is None:
            research_static_block_reason = research_static_block_reason or "missing_session_window"
    if not research_enabled:
        if not pass_symbols:
            if portfolio_name == "multi":
                pass_symbols = ["USDJPY", "GBPJPY"]
            else:
                pass_symbols = sorted({p.symbol_key for p, _ in profiles})
        profiles = [(p, scale) for p, scale in profiles if p.symbol_key.upper() in pass_symbols]
    symbols = sorted({p.symbol_key for p, _ in profiles})
    try:
        max_open_positions = int(os.getenv("OMEGAFX_MAX_OPEN_POSITIONS", "3"))
    except Exception:
        max_open_positions = 3
    if max_open_positions < 0:
        max_open_positions = 0
    bucket_caps = _parse_bucket_caps()
    if not bucket_caps:
        bucket_caps = {"JPY": 1, "USD": 2}
    scorers_bucket_caps = _parse_bucket_caps(os.getenv("OMEGAFX_SCORERS_BUCKET_CAPS") or "")
    if not scorers_bucket_caps:
        scorers_bucket_caps = {"JPY": 1}
    westbrook_bucket_caps = _parse_bucket_caps(os.getenv("OMEGAFX_WESTBROOK_BUCKET_CAPS") or "")
    if not westbrook_bucket_caps:
        westbrook_bucket_caps = {"JPY": 1, "USD": 1}
    try:
        scorers_daily_loss_pct = float(os.getenv("OMEGAFX_SCORERS_DAILY_LOSS_PCT", "0.01"))
    except Exception:
        scorers_daily_loss_pct = 0.01
    try:
        scorers_max_open_positions = int(os.getenv("OMEGAFX_SCORERS_MAX_OPEN_POSITIONS", "2"))
    except Exception:
        scorers_max_open_positions = 2
    if scorers_max_open_positions < 0:
        scorers_max_open_positions = 0
    scorers_open_risk_cap_raw = (
        os.getenv("OMEGAFX_SCORERS_MAX_TOTAL_OPEN_RISK_PCT")
        or os.getenv("OMEGAFX_SCORERS_OPEN_RISK_CAP_PCT")
        or ""
    )
    try:
        scorers_open_risk_cap_pct = float(scorers_open_risk_cap_raw) if scorers_open_risk_cap_raw else 0.0
    except Exception:
        scorers_open_risk_cap_pct = 0.0
    try:
        scorers_risk_override_pct = float(os.getenv("OMEGAFX_SCORERS_RISK_PER_TRADE_PCT", "0") or 0)
    except Exception:
        scorers_risk_override_pct = 0.0
    try:
        westbrook_daily_loss_pct = float(os.getenv("OMEGAFX_WESTBROOK_DAILY_LOSS_PCT", "0.02"))
    except Exception:
        westbrook_daily_loss_pct = 0.02
    try:
        westbrook_max_open_positions = int(os.getenv("OMEGAFX_WESTBROOK_MAX_OPEN_POSITIONS", "1"))
    except Exception:
        westbrook_max_open_positions = 1
    if westbrook_max_open_positions < 0:
        westbrook_max_open_positions = 0
    westbrook_open_risk_cap_raw = (
        os.getenv("OMEGAFX_WESTBROOK_MAX_TOTAL_OPEN_RISK_PCT")
        or os.getenv("OMEGAFX_WESTBROOK_OPEN_RISK_CAP_PCT")
        or ""
    )
    try:
        westbrook_open_risk_cap_pct = float(westbrook_open_risk_cap_raw) if westbrook_open_risk_cap_raw else 0.0
    except Exception:
        westbrook_open_risk_cap_pct = 0.0
    try:
        portfolio_open_risk_cap_pct = float(os.getenv("OMEGAFX_PORTFOLIO_OPEN_RISK_CAP_PCT", "0.02"))
    except Exception:
        portfolio_open_risk_cap_pct = 0.02
    per_trade_risk_base = DEFAULT_STRATEGY.risk_per_trade_pct
    per_trade_risk_by_strategy = {}
    for prof, scale in profiles:
        per_trade_risk_by_strategy[prof.name] = prof.strategy.risk_per_trade_pct * scale
    daily_loss_cap = getattr(portfolio, "portfolio_daily_loss_pct", None)
    hard_loss_cap = getattr(portfolio, "portfolio_max_loss_pct", None)
    global PASS_SYMBOLS_ACTIVE, MAX_OPEN_POSITIONS, BUCKET_CAPS, RISK_SETTINGS
    PASS_SYMBOLS_ACTIVE = [] if research_enabled else (pass_symbols or symbols)
    MAX_OPEN_POSITIONS = max_open_positions
    BUCKET_CAPS = bucket_caps
    RISK_SETTINGS = {
        "per_trade_risk_pct": per_trade_risk_base,
        "per_trade_risk_pct_by_strategy": per_trade_risk_by_strategy,
        "daily_loss_cap_pct": daily_loss_cap,
        "soft_daily_stop_pct": daily_loss_cap,
        "hard_daily_stop_pct": hard_loss_cap,
        "max_open_positions": max_open_positions,
        "bucket_caps": bucket_caps,
        "pass_symbols": PASS_SYMBOLS_ACTIVE,
    }
    if portfolio_name == "combined":
        RISK_SETTINGS.update({
            "scorers_daily_loss_pct": scorers_daily_loss_pct,
            "scorers_max_open_positions": scorers_max_open_positions,
            "scorers_bucket_caps": scorers_bucket_caps,
            "scorers_open_risk_cap_pct": scorers_open_risk_cap_pct,
            "scorers_risk_override_pct": scorers_risk_override_pct,
            "westbrook_daily_loss_pct": westbrook_daily_loss_pct,
            "westbrook_max_open_positions": westbrook_max_open_positions,
            "westbrook_bucket_caps": westbrook_bucket_caps,
            "westbrook_open_risk_cap_pct": westbrook_open_risk_cap_pct,
            "portfolio_open_risk_cap_pct": portfolio_open_risk_cap_pct,
        })
    policy_config = load_config_from_env()
    policy_state = init_state(initial_equity)
    policy = default_policy(symbols)
    write_policy(policy)

    log_lifecycle(f"START | mode={mode} | portfolio={portfolio_name} | symbols={','.join(symbols)}")
    state_flag, state_reason = _state_for_lock(mode)
    write_state(state_flag, mode, portfolio_name, reason=state_reason, symbols=symbols, equity=initial_equity)

    if pd is None:
        msg = "Missing dependency: pandas. Install with pip install -r requirements.txt"
        print(msg)
        log_error(msg)
        set_stop_reason(f"mode={mode} | portfolio={portfolio_name} | reason={msg}")
        write_state("STOPPED", mode, portfolio_name, reason=msg, symbols=symbols, equity=initial_equity)
        return

    if mode in {"demo", "ftmo"}:
        ok, reason = mt5_login()
        if not ok:
            if reason and "initialize failed" in reason.lower():
                detail = reason
                if "mt5 initialize failed:" in reason.lower():
                    detail = reason.split(":", 1)[1].strip()
                summary = f"MT5 init failed: {detail}. Runner will not start; dashboard will show UNKNOWN."
                log_lifecycle(summary)
                log_error(summary)
            msg = f"mode={mode} | portfolio={portfolio_name} | reason={reason}"
            set_stop_reason(msg)
            log_lifecycle(f"PRE-FLIGHT FAIL | {msg}")
            write_state("STOPPED", mode, portfolio_name, reason=reason, symbols=symbols, equity=initial_equity)
            return
        state_flag, state_reason = _state_for_lock(mode)
        write_state(state_flag, mode, portfolio_name, reason=state_reason, symbols=symbols, equity=initial_equity)
        if LAST_ACCOUNT_INFO is not None:
            raw_bot_status = "ARMED" if state_flag == "RUNNING" else "DISARMED"
            log_total, matched = _trade_log_counts(trade_log)
            LOG_EVENTS_TOTAL = log_total
            MATCHED_MT5_TICKETS_COUNT = matched
            UNMATCHED_LOG_ONLY_COUNT = max(0, log_total - matched)
            MT5_VERIFIED_CLOSED_TRADES_COUNT = 0
            EVIDENCE_TIER = "verified" if MATCHED_MT5_TICKETS_COUNT > 0 else "practice"
            now_ts = time.time()
            mt5_ticks, mt5_tick_status, mt5_tick_reason = _mt5_tick_snapshot(symbols, now_ts)
            mt5_truth, mt5_truth_status, mt5_truth_reason = _mt5_truth_snapshot(now_ts)
            research_block_reason = research_static_block_reason
            if research_enabled:
                if research_block_reason is None and session_window and not _session_allows(now_ts, session_window):
                    research_block_reason = "outside_session_window"
                if research_block_reason is None and event_guard_days:
                    today = datetime.fromtimestamp(now_ts, tz=timezone.utc).date().isoformat()
                    if today not in event_guard_days:
                        research_block_reason = "event_guard_block"
                if research_block_reason is None and DRY_RUN_ENABLED:
                    research_block_reason = "trading_path_not_ready"
                if research_block_reason is None and mt5_tick_status != "ok":
                    research_block_reason = mt5_tick_reason or mt5_tick_status
            risk_payload, risk_ok, risk_reason = _build_risk_policy_payload(
                policy, policy_config, dynamic_risk, symbols, now_ts
            )
            write_account_snapshot(
                LAST_ACCOUNT_INFO,
                LAST_ACCOUNT_ERROR,
                raw_bot_status,
                risk_policy=risk_payload,
                risk_policy_ok=risk_ok,
                risk_policy_reason=risk_reason,
                mt5_truth=mt5_truth,
                mt5_truth_status=mt5_truth_status,
                mt5_truth_reason=mt5_truth_reason,
                mt5_ticks=mt5_ticks,
                mt5_tick_status=mt5_tick_status,
                mt5_tick_reason=mt5_tick_reason,
                extra_block_reason=research_block_reason,
            )
    else:
        if mt5 is None:
            log_lifecycle("MT5 not installed; shadow will idle without data.")
        else:
            mt5_path = os.getenv("MT5_PATH") or os.getenv("OMEGAFX_MT5_PATH")
            if MT5_INIT_DELAY_SEC > 0:
                time.sleep(MT5_INIT_DELAY_SEC)
            ok_init = mt5.initialize(path=mt5_path) if mt5_path else mt5.initialize()
            if not ok_init:
                log_lifecycle(f"MT5 initialize failed (shadow): {mt5.last_error()}")
            else:
                login, password, server = mt5_env_values()
                if all([login, password, server]):
                    if not mt5.login(login=int(login), password=password, server=server):
                        log_lifecycle(f"MT5 login failed (shadow): {mt5.last_error()}")

    brokers: Dict[str, Mt5BrokerAdapter] = {}
    if mode in {"demo", "ftmo"} and mt5 is not None:
        login, password, server = mt5_env_values()
        conn = Mt5ConnectionConfig(login=int(login), password=password, server=server)
        for sym in symbols:
            brokers[sym] = Mt5BrokerAdapter(conn, sym, dry_run=None)

    if mode in {"demo", "ftmo"} and DRY_RUN_ENABLED:
        log_lifecycle("DRY RUN enabled for demo/ftmo; trading will be disarmed.")

    allow_trading = not (
        mode in {"demo", "ftmo"} and IDENTITY_LOCK.get("status") == "mismatch"
    )
    if mode in {"demo", "ftmo"} and DRY_RUN_ENABLED:
        allow_trading = False
    if not allow_trading:
        lock_reason = IDENTITY_LOCK.get("reason") or "ACCOUNT_MISMATCH"
        log_lifecycle(f"FORCING_LOCK | TRADING_ALLOWED=false | {lock_reason}")
        log_error(f"FORCING_LOCK | TRADING_ALLOWED=false | {lock_reason}")

    last_trade_time: Dict[str, pd.Timestamp] = {}
    last_heartbeat = 0.0
    equity = initial_equity
    processed_deals: set[int] = set()
    last_deal_check = time.time() - 86400

    try:
        while True:
            now_ts = time.time()
            if dynamic_risk:
                policy = evaluate_policy(policy_state, equity, symbols, now_ts, policy_config)
                write_policy(policy)
            cooldown_until = policy.get("cooldown_until") if dynamic_risk else None
            in_cooldown = bool(cooldown_until and cooldown_until > now_ts)
            trade_allowed_ok = True
            if mode in {"demo", "ftmo"}:
                trade_allowed_ok = LAST_MT5_TRADE_ALLOWED is True
            research_block_reason = research_static_block_reason
            if research_enabled:
                today = datetime.fromtimestamp(now_ts, tz=timezone.utc).date().isoformat()
                global RESEARCH_DAY_START_EQUITY, RESEARCH_DAY_START_DATE, RESEARCH_DAILY_STOP_HIT
                if RESEARCH_DAY_START_DATE != today:
                    RESEARCH_DAY_START_DATE = today
                    RESEARCH_DAY_START_EQUITY = equity
                    RESEARCH_DAILY_STOP_HIT = False
                if research_daily_loss_pct > 0 and RESEARCH_DAY_START_EQUITY:
                    if equity <= RESEARCH_DAY_START_EQUITY * (1.0 - research_daily_loss_pct):
                        RESEARCH_DAILY_STOP_HIT = True
                if RESEARCH_DAILY_STOP_HIT and research_block_reason is None:
                    research_block_reason = "research_daily_stop"
                if research_block_reason is None and session_window and not _session_allows(now_ts, session_window):
                    research_block_reason = "outside_session_window"
                if research_block_reason is None and event_guard_days:
                    if today not in event_guard_days:
                        research_block_reason = "event_guard_block"
                if research_block_reason is None:
                    if not allow_trading or DRY_RUN_ENABLED or not trade_allowed_ok or MT5_INIT_OK is not True:
                        research_block_reason = "trading_path_not_ready"
                if research_block_reason is None and mode in {"demo", "ftmo"}:
                    tick_symbols = active_symbols or symbols
                    mt5_ticks, mt5_tick_status, mt5_tick_reason = _mt5_tick_snapshot(tick_symbols, now_ts)
                    if mt5_tick_status != "ok":
                        research_block_reason = mt5_tick_reason or mt5_tick_status
                    else:
                        age_map = mt5_ticks.get("age_sec_by_symbol", {})
                        if any((val or 0) > TICK_STALE_SECONDS for val in age_map.values()):
                            research_block_reason = "tick_stale"
            westbrook_block_reason = None
            if portfolio_name == "combined":
                try:
                    today = datetime.fromtimestamp(now_ts, tz=timezone.utc).date().isoformat()
                    global WESTBROOK_DAY_START_EQUITY, WESTBROOK_DAY_START_DATE, WESTBROOK_DAILY_STOP_HIT
                    if WESTBROOK_DAY_START_DATE != today:
                        WESTBROOK_DAY_START_DATE = today
                        WESTBROOK_DAY_START_EQUITY = equity
                        WESTBROOK_DAILY_STOP_HIT = False
                    if westbrook_daily_loss_pct > 0 and WESTBROOK_DAY_START_EQUITY:
                        if equity <= WESTBROOK_DAY_START_EQUITY * (1.0 - westbrook_daily_loss_pct):
                            if not WESTBROOK_DAILY_STOP_HIT:
                                log_lifecycle(
                                    f"WESTBROOK STOP | loss_pct={westbrook_daily_loss_pct:.4f} | equity={equity:.2f}"
                                )
                            WESTBROOK_DAILY_STOP_HIT = True
                    if WESTBROOK_DAILY_STOP_HIT:
                        westbrook_block_reason = "westbrook_daily_stop"
                except Exception as exc:
                    westbrook_block_reason = "westbrook_guard_error"
                    log_error(f"WESTBROOK GUARD ERROR | {exc}")
                    log_lifecycle(f"WESTBROOK GUARD ERROR | {exc}")
            scorers_block_reason = None
            if portfolio_name == "combined":
                try:
                    today = datetime.fromtimestamp(now_ts, tz=timezone.utc).date().isoformat()
                    global SCORERS_DAY_START_EQUITY, SCORERS_DAY_START_DATE, SCORERS_DAILY_STOP_HIT
                    if SCORERS_DAY_START_DATE != today:
                        SCORERS_DAY_START_DATE = today
                        SCORERS_DAY_START_EQUITY = equity
                        SCORERS_DAILY_STOP_HIT = False
                    if scorers_daily_loss_pct > 0 and SCORERS_DAY_START_EQUITY:
                        if equity <= SCORERS_DAY_START_EQUITY * (1.0 - scorers_daily_loss_pct):
                            if not SCORERS_DAILY_STOP_HIT:
                                log_lifecycle(
                                    f"SCORERS STOP | loss_pct={scorers_daily_loss_pct:.4f} | equity={equity:.2f}"
                                )
                            SCORERS_DAILY_STOP_HIT = True
                    if SCORERS_DAILY_STOP_HIT:
                        scorers_block_reason = "scorers_daily_stop"
                except Exception as exc:
                    scorers_block_reason = "scorers_guard_error"
                    log_error(f"SCORERS GUARD ERROR | {exc}")
                    log_lifecycle(f"SCORERS GUARD ERROR | {exc}")
            open_positions = {}
            bucket_positions = {}
            positions_list = None
            open_positions_total = 0
            westbrook_open_positions: Dict[Tuple[str, str], int] = {}
            westbrook_bucket_positions: Dict[Tuple[str, str], int] = {}
            westbrook_open_positions_total = 0
            scorers_open_positions: Dict[Tuple[str, str], int] = {}
            scorers_bucket_positions: Dict[Tuple[str, str], int] = {}
            scorers_open_positions_total = 0
            if mode in {"demo", "ftmo"} and mt5 is not None:
                positions, pos_maps = _positions_snapshot()
                positions_list = positions
                MT5_OPEN_POSITIONS_COUNT = len(positions)
                open_positions = pos_maps.get("symbol_dir", {})
                bucket_positions = pos_maps.get("bucket_dir", {})
                open_positions_total = len(positions)
                westbrook_open_positions = pos_maps.get("westbrook_symbol_dir", {})
                westbrook_bucket_positions = pos_maps.get("westbrook_bucket_dir", {})
                westbrook_open_positions_total = pos_maps.get("westbrook_total", 0)
                scorers_open_positions_total = max(0, open_positions_total - westbrook_open_positions_total)
                if open_positions:
                    for key, count in open_positions.items():
                        w_count = westbrook_open_positions.get(key, 0)
                        remaining = count - w_count
                        if remaining > 0:
                            scorers_open_positions[key] = remaining
                if bucket_positions:
                    for key, count in bucket_positions.items():
                        w_count = westbrook_bucket_positions.get(key, 0)
                        remaining = count - w_count
                        if remaining > 0:
                            scorers_bucket_positions[key] = remaining

            if portfolio_name == "combined":
                scorers_trade_risk_cap_pct = (
                    (scorers_open_risk_cap_pct / max(1, scorers_max_open_positions))
                    if scorers_open_risk_cap_pct > 0
                    else 0.0
                )
                westbrook_trade_risk_cap_pct = (
                    (westbrook_open_risk_cap_pct / max(1, westbrook_max_open_positions))
                    if westbrook_open_risk_cap_pct > 0
                    else 0.0
                )
                scorers_risk_per_pos = (
                    scorers_trade_risk_cap_pct if scorers_trade_risk_cap_pct > 0 else per_trade_risk_base
                )
                if scorers_risk_override_pct > 0:
                    if scorers_risk_per_pos > 0:
                        scorers_risk_per_pos = min(scorers_risk_per_pos, scorers_risk_override_pct)
                    else:
                        scorers_risk_per_pos = scorers_risk_override_pct
                scorers_open_risk_pct = scorers_open_positions_total * scorers_risk_per_pos
                westbrook_open_risk_pct = westbrook_open_positions_total * (
                    westbrook_trade_risk_cap_pct if westbrook_trade_risk_cap_pct > 0 else per_trade_risk_base
                )
                total_open_risk_pct = scorers_open_risk_pct + westbrook_open_risk_pct
                global OPEN_POSITIONS_SCORERS, OPEN_POSITIONS_WESTBROOK
                global OPEN_RISK_SCORERS_PCT, OPEN_RISK_WESTBROOK_PCT, TOTAL_OPEN_RISK_PCT
                OPEN_POSITIONS_SCORERS = scorers_open_positions_total
                OPEN_POSITIONS_WESTBROOK = westbrook_open_positions_total
                OPEN_RISK_SCORERS_PCT = scorers_open_risk_pct
                OPEN_RISK_WESTBROOK_PCT = westbrook_open_risk_pct
                TOTAL_OPEN_RISK_PCT = total_open_risk_pct
            else:
                OPEN_POSITIONS_SCORERS = 0
                OPEN_POSITIONS_WESTBROOK = 0
                OPEN_RISK_SCORERS_PCT = 0.0
                OPEN_RISK_WESTBROOK_PCT = 0.0
                TOTAL_OPEN_RISK_PCT = 0.0

            # fetch data per symbol/timeframe
            ohlc_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
            for prof, _ in profiles:
                tf = prof.timeframe
                key = (prof.symbol_key, tf)
                if key not in ohlc_cache:
                    df = fetch_recent_ohlc(prof.symbol_key, tf)
                    if df is not None:
                        ohlc_cache[key] = df
                # momentum requires M5+M15
                if isinstance(prof.signals, MomentumPinballSignalConfig):
                    for tf_req in ("5m", "15m"):
                        key = (prof.symbol_key, tf_req)
                        if key not in ohlc_cache:
                            df = fetch_recent_ohlc(prof.symbol_key, tf_req)
                            if df is not None:
                                ohlc_cache[key] = df
                if isinstance(prof.signals, BigManSignalConfig):
                    key = (prof.symbol_key, "1h")
                    if key not in ohlc_cache:
                        df = fetch_recent_ohlc(prof.symbol_key, "1h")
                        if df is not None:
                            ohlc_cache[key] = df

            # run signals
            for prof, scale in profiles:
                if not allow_trading:
                    break
                if research_enabled and research_block_reason:
                    break
                if mode in {"demo", "ftmo"} and not trade_allowed_ok:
                    continue
                profile_group = _profile_group(portfolio_name, prof)
                profile_dynamic_risk = dynamic_risk and profile_group == "westbrook"
                lineup_override = None
                if portfolio_name == "combined":
                    lineup_override = "W" if profile_group == "westbrook" else "S"
                if profile_group == "westbrook" and westbrook_block_reason:
                    last_t = LAST_NO_SETUP_LOG_TS.get("westbrook_block")
                    if last_t is None or (now_ts - last_t) > 300:
                        log_lifecycle(f"WESTBROOK BLOCK | reason={westbrook_block_reason}")
                        LAST_NO_SETUP_LOG_TS["westbrook_block"] = now_ts
                    continue
                if profile_group == "scorers" and scorers_block_reason:
                    last_t = LAST_NO_SETUP_LOG_TS.get("scorers_block")
                    if last_t is None or (now_ts - last_t) > 300:
                        log_lifecycle(f"SCORERS BLOCK | reason={scorers_block_reason}")
                        LAST_NO_SETUP_LOG_TS["scorers_block"] = now_ts
                    continue
                if profile_dynamic_risk:
                    if in_cooldown:
                        continue
                    enabled = policy.get("symbol_enabled", {}).get(prof.symbol_key, True)
                    if not enabled:
                        continue
                if research_enabled and active_symbols and prof.symbol_key.upper() not in active_symbols:
                    continue
                if (
                    research_enabled
                    and active_edge_tag
                    and not (fixfade_enabled or trendpb_enabled)
                    and not _edge_matches(prof.name, active_edge_tag)
                ):
                    continue
                tf = prof.timeframe
                ohlc_by_tf = {
                    tf_key[1]: df for tf_key, df in ohlc_cache.items() if tf_key[0] == prof.symbol_key
                }
                base_df = ohlc_by_tf.get(tf)
                if base_df is None or len(base_df) < 5:
                    continue
                entry_idx = None
                entry_time = None
                fixfade_direction = None
                fixfade_anchor = None
                fixfade_session_high = None
                fixfade_session_low = None
                trendpb_direction = None
                trendpb_stop = None
                trendpb_target = None
                london_trace = None
                paper_shot_logged = False
                if research_enabled and fixfade_enabled:
                    closed_df = base_df.iloc[:-1] if len(base_df) > 1 else base_df
                    if closed_df.empty:
                        continue
                    entry_idx = len(closed_df) - 1
                    entry_time = closed_df.index[entry_idx]
                    if session_window is None:
                        continue
                    day_start = datetime.fromtimestamp(now_ts, tz=timezone.utc).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    start_min, _ = session_window
                    session_start = day_start + timedelta(minutes=start_min)
                    pre_df = closed_df[(closed_df.index >= day_start) & (closed_df.index < session_start)]
                    session_df = closed_df[closed_df.index >= session_start]
                    if pre_df.empty or session_df.empty:
                        last_t = LAST_NO_SETUP_LOG_TS.get(prof.symbol_key)
                        if last_t is None or (now_ts - last_t) > 300:
                            log_lifecycle(f"RESEARCH FIXFADE | symbol={prof.symbol_key} | no_setup")
                            LAST_NO_SETUP_LOG_TS[prof.symbol_key] = now_ts
                        continue
                    anchor = (float(pre_df["high"].max()) + float(pre_df["low"].min())) / 2.0
                    session_high = float(session_df["high"].max())
                    session_low = float(session_df["low"].min())
                    fixfade_anchor = anchor
                    fixfade_session_high = session_high
                    fixfade_session_low = session_low
                    pip_size = _pip_size_for_symbol(prof.symbol_key)
                    shove_up = session_high >= anchor + (fixfade_shove_pips * pip_size)
                    shove_down = session_low <= anchor - (fixfade_shove_pips * pip_size)
                    close_val = float(closed_df["close"].iloc[entry_idx])
                    retrace_level_up = anchor + max(0.0, fixfade_shove_pips - fixfade_retrace_pips) * pip_size
                    retrace_level_down = anchor - max(0.0, fixfade_shove_pips - fixfade_retrace_pips) * pip_size
                    in_band = abs(close_val - anchor) <= (fixfade_band_pips * pip_size)
                    if shove_up and close_val >= anchor and (close_val <= retrace_level_up or in_band):
                        fixfade_direction = "short"
                    elif shove_down and close_val <= anchor and (close_val >= retrace_level_down or in_band):
                        fixfade_direction = "long"
                    if fixfade_direction is None:
                        last_t = LAST_NO_SETUP_LOG_TS.get(prof.symbol_key)
                        if last_t is None or (now_ts - last_t) > 300:
                            log_lifecycle(f"RESEARCH FIXFADE | symbol={prof.symbol_key} | no_setup")
                            LAST_NO_SETUP_LOG_TS[prof.symbol_key] = now_ts
                        continue
                elif research_enabled and trendpb_enabled:
                    closed_df = base_df.iloc[:-1] if len(base_df) > 1 else base_df
                    if closed_df.empty:
                        continue
                    entry_idx = len(closed_df) - 1
                    entry_time = closed_df.index[entry_idx]
                    if session_window is None:
                        continue
                    if len(closed_df) < (trendpb_slow_span + 3):
                        continue
                    close_series = closed_df["close"].astype(float)
                    ema_fast = close_series.ewm(span=trendpb_fast_span, adjust=False).mean()
                    ema_slow = close_series.ewm(span=trendpb_slow_span, adjust=False).mean()
                    fast_now = float(ema_fast.iloc[-1])
                    slow_now = float(ema_slow.iloc[-1])
                    fast_prev = float(ema_fast.iloc[-3])
                    slow_prev = float(ema_slow.iloc[-3])
                    trend_up = fast_now > slow_now and fast_now > fast_prev and slow_now >= slow_prev
                    trend_down = fast_now < slow_now and fast_now < fast_prev and slow_now <= slow_prev
                    if not (trend_up or trend_down):
                        last_t = LAST_NO_SETUP_LOG_TS.get(prof.symbol_key)
                        if last_t is None or (now_ts - last_t) > 300:
                            log_lifecycle(f"RESEARCH TRENDPB | symbol={prof.symbol_key} | no_setup")
                            LAST_NO_SETUP_LOG_TS[prof.symbol_key] = now_ts
                        continue
                    close_val = float(closed_df["close"].iloc[entry_idx])
                    low_val = float(closed_df["low"].iloc[entry_idx])
                    high_val = float(closed_df["high"].iloc[entry_idx])
                    if trend_up and low_val <= fast_now and close_val >= fast_now:
                        trendpb_direction = "long"
                    elif trend_down and high_val >= fast_now and close_val <= fast_now:
                        trendpb_direction = "short"
                    if trendpb_direction is None:
                        last_t = LAST_NO_SETUP_LOG_TS.get(prof.symbol_key)
                        if last_t is None or (now_ts - last_t) > 300:
                            log_lifecycle(f"RESEARCH TRENDPB | symbol={prof.symbol_key} | no_setup")
                            LAST_NO_SETUP_LOG_TS[prof.symbol_key] = now_ts
                        continue
                    lookback = closed_df.tail(trendpb_swing_lookback)
                    if lookback.empty:
                        continue
                    pip_size = _pip_size_for_symbol(prof.symbol_key)
                    buffer = trendpb_sl_buffer_pips * pip_size
                    if trendpb_direction == "long":
                        swing_low = float(lookback["low"].min())
                        stop_price = swing_low - buffer
                        risk = close_val - stop_price
                        if risk <= 0:
                            log_lifecycle(f"RESEARCH TRENDPB | symbol={prof.symbol_key} | invalid_sl")
                            continue
                        target_price = close_val + (trendpb_r_multiple * risk)
                    else:
                        swing_high = float(lookback["high"].max())
                        stop_price = swing_high + buffer
                        risk = stop_price - close_val
                        if risk <= 0:
                            log_lifecycle(f"RESEARCH TRENDPB | symbol={prof.symbol_key} | invalid_sl")
                            continue
                        target_price = close_val - (trendpb_r_multiple * risk)
                    trendpb_stop = stop_price
                    trendpb_target = target_price
                else:
                    signals = build_signals(prof, ohlc_by_tf)
                    if signals.empty:
                        continue
                    entry_idx = len(signals) - 2
                    if entry_idx < 0:
                        continue
                    entry_time = base_df.index[entry_idx]
                    london_trace = _build_london_breakout_trace(
                        prof,
                        base_df,
                        entry_idx,
                        now_ts,
                        in_cooldown,
                        policy,
                        profile_dynamic_risk,
                        scale,
                    )
                    if london_trace:
                        DECISION_TRACE[london_trace["player"]] = london_trace
                    trend_trace = _build_trend_continuation_trace(
                        prof,
                        base_df,
                        entry_idx,
                        now_ts,
                        in_cooldown,
                        policy,
                        profile_dynamic_risk,
                        scale,
                    )
                    if trend_trace:
                        DECISION_TRACE[trend_trace["player"]] = trend_trace
                    signals_flag = bool(signals.iloc[entry_idx])
                    if paper_shots_enabled and london_trace and london_trace.get("trigger_ok"):
                        failed_gates = _paper_shot_failed_gates(london_trace)
                        if failed_gates and not signals_flag:
                            price = float(base_df["close"].iloc[entry_idx])
                            planned = _build_planned_trade(
                                prof,
                                scale,
                                equity,
                                price,
                                portfolio_name,
                                research_enabled,
                                research_risk_scale,
                                policy,
                                profile_dynamic_risk,
                                active_edge_tag,
                                lineup_override=lineup_override,
                            )
                            _append_paper_shot(
                                PAPER_SHOTS_LOG,
                                mode,
                                portfolio_name,
                                prof,
                                planned,
                                entry_time,
                                london_trace,
                                reason=f"gate_block:{','.join(failed_gates)}",
                            )
                            paper_shot_logged = True
                    if not signals_flag:
                        continue
                last_t = last_trade_time.get(prof.name)
                if last_t is not None and entry_time <= last_t:
                    continue

                base_risk_pct = prof.strategy.risk_per_trade_pct * scale
                if research_enabled:
                    base_risk_pct *= research_risk_scale
                if profile_dynamic_risk:
                    base_risk_pct *= (policy.get("global_risk_scale", 1.0) or 1.0)
                risk_scale_override = 1.0
                cap_per_trade_pct = 0.0
                if portfolio_name == "combined":
                    if profile_group == "westbrook":
                        cap_per_trade_pct = westbrook_trade_risk_cap_pct
                    else:
                        cap_per_trade_pct = scorers_trade_risk_cap_pct
                        if scorers_risk_override_pct > 0 and base_risk_pct > 0:
                            risk_scale_override = min(
                                risk_scale_override, scorers_risk_override_pct / base_risk_pct
                            )
                    if cap_per_trade_pct > 0 and base_risk_pct > 0:
                        risk_scale_override = min(
                            risk_scale_override, cap_per_trade_pct / base_risk_pct
                        )
                if risk_scale_override < 0:
                    risk_scale_override = 0.0
                scale_effective = scale * risk_scale_override
                planned_risk_pct = base_risk_pct * risk_scale_override

                price = float(base_df["close"].iloc[entry_idx])
                planned = _build_planned_trade(
                    prof,
                    scale_effective,
                    equity,
                    price,
                    portfolio_name,
                    research_enabled,
                    research_risk_scale,
                    policy,
                    profile_dynamic_risk,
                    active_edge_tag,
                    lineup_override=lineup_override,
                )
                if fixfade_direction:
                    if fixfade_anchor is None or fixfade_session_high is None or fixfade_session_low is None:
                        log_lifecycle(f"RESEARCH FIXFADE | symbol={prof.symbol_key} | missing_anchor")
                        continue
                    planned.direction = fixfade_direction
                    pip_size = _pip_size_for_symbol(prof.symbol_key)
                    buffer = fixfade_sl_buffer_pips * pip_size
                    if fixfade_direction == "short":
                        planned.stop_loss_price = fixfade_session_high + buffer
                        planned.take_profit_price = float(fixfade_anchor)
                        if planned.stop_loss_price <= planned.entry_price or planned.take_profit_price >= planned.entry_price:
                            log_lifecycle(f"RESEARCH FIXFADE | symbol={prof.symbol_key} | invalid_sl_tp")
                            continue
                    else:
                        planned.stop_loss_price = fixfade_session_low - buffer
                        planned.take_profit_price = float(fixfade_anchor)
                        if planned.stop_loss_price >= planned.entry_price or planned.take_profit_price <= planned.entry_price:
                            log_lifecycle(f"RESEARCH FIXFADE | symbol={prof.symbol_key} | invalid_sl_tp")
                            continue
                elif trendpb_direction:
                    planned.direction = trendpb_direction
                    planned.stop_loss_price = float(trendpb_stop) if trendpb_stop is not None else planned.stop_loss_price
                    planned.take_profit_price = float(trendpb_target) if trendpb_target is not None else planned.take_profit_price
                    if trendpb_direction == "long":
                        if planned.stop_loss_price >= planned.entry_price or planned.take_profit_price <= planned.entry_price:
                            log_lifecycle(f"RESEARCH TRENDPB | symbol={prof.symbol_key} | invalid_sl_tp")
                            continue
                    else:
                        if planned.stop_loss_price <= planned.entry_price or planned.take_profit_price >= planned.entry_price:
                            log_lifecycle(f"RESEARCH TRENDPB | symbol={prof.symbol_key} | invalid_sl_tp")
                            continue

                _append_stage_log(
                    trade_log,
                    "SIGNAL",
                    mode,
                    portfolio_name,
                    prof,
                    planned,
                    entry_time,
                    account_info=LAST_ACCOUNT_INFO,
                )

                if mode in {"demo", "ftmo"}:
                    cap_open_positions_total = open_positions_total
                    cap_open_positions = open_positions
                    cap_bucket_positions = bucket_positions
                    cap_max_open_positions = MAX_OPEN_POSITIONS
                    cap_bucket_caps = BUCKET_CAPS
                    if portfolio_name == "combined":
                        if profile_group == "westbrook":
                            cap_open_positions_total = westbrook_open_positions_total
                            cap_open_positions = westbrook_open_positions
                            cap_bucket_positions = westbrook_bucket_positions
                            cap_max_open_positions = westbrook_max_open_positions
                            cap_bucket_caps = westbrook_bucket_caps
                        else:
                            cap_open_positions_total = scorers_open_positions_total
                            cap_open_positions = scorers_open_positions
                            cap_bucket_positions = scorers_bucket_positions
                            cap_max_open_positions = scorers_max_open_positions
                            cap_bucket_caps = scorers_bucket_caps
                    if cap_max_open_positions > 0 and cap_open_positions_total >= cap_max_open_positions:
                        log_lifecycle(
                            f"EXPOSURE BLOCK | max_open_positions={cap_max_open_positions}"
                        )
                        if (
                            paper_shots_enabled
                            and london_trace
                            and london_trace.get("trigger_ok")
                            and not paper_shot_logged
                        ):
                            _append_paper_shot(
                                PAPER_SHOTS_LOG,
                                mode,
                                portfolio_name,
                                prof,
                                planned,
                                entry_time,
                                london_trace,
                                reason="exposure_block:max_open_positions",
                            )
                            paper_shot_logged = True
                            _append_stage_log(
                                trade_log,
                                "ORDER_REJECTED",
                                mode,
                                portfolio_name,
                                prof,
                                planned,
                                entry_time,
                                account_info=LAST_ACCOUNT_INFO,
                            mt5_last_error="max_open_positions",
                        )
                        continue
                    if portfolio_name == "combined":
                        group_open_risk_pct = scorers_open_risk_pct
                        group_cap_pct = scorers_open_risk_cap_pct
                        if profile_group == "westbrook":
                            group_open_risk_pct = westbrook_open_risk_pct
                            group_cap_pct = westbrook_open_risk_cap_pct
                        if (
                            group_cap_pct > 0
                            and planned_risk_pct > 0
                            and (group_open_risk_pct + planned_risk_pct) > group_cap_pct
                        ):
                            log_lifecycle(
                                f"EXPOSURE BLOCK | open_risk_cap={group_cap_pct:.4f} "
                                f"| open_risk={group_open_risk_pct:.4f} add={planned_risk_pct:.4f}"
                            )
                            if (
                                paper_shots_enabled
                                and london_trace
                                and london_trace.get("trigger_ok")
                                and not paper_shot_logged
                            ):
                                _append_paper_shot(
                                    PAPER_SHOTS_LOG,
                                    mode,
                                    portfolio_name,
                                    prof,
                                    planned,
                                    entry_time,
                                    london_trace,
                                    reason="exposure_block:open_risk_cap",
                                )
                                paper_shot_logged = True
                            _append_stage_log(
                                trade_log,
                                "ORDER_REJECTED",
                                mode,
                                portfolio_name,
                                prof,
                                planned,
                                entry_time,
                                account_info=LAST_ACCOUNT_INFO,
                                mt5_last_error="open_risk_cap",
                            )
                            continue
                        if (
                            portfolio_open_risk_cap_pct > 0
                            and planned_risk_pct > 0
                            and (total_open_risk_pct + planned_risk_pct) > portfolio_open_risk_cap_pct
                        ):
                            log_lifecycle(
                                f"EXPOSURE BLOCK | portfolio_open_risk_cap={portfolio_open_risk_cap_pct:.4f} "
                                f"| open_risk={total_open_risk_pct:.4f} add={planned_risk_pct:.4f}"
                            )
                            if (
                                paper_shots_enabled
                                and london_trace
                                and london_trace.get("trigger_ok")
                                and not paper_shot_logged
                            ):
                                _append_paper_shot(
                                    PAPER_SHOTS_LOG,
                                    mode,
                                    portfolio_name,
                                    prof,
                                    planned,
                                    entry_time,
                                    london_trace,
                                    reason="exposure_block:portfolio_open_risk_cap",
                                )
                                paper_shot_logged = True
                            _append_stage_log(
                                trade_log,
                                "ORDER_REJECTED",
                                mode,
                                portfolio_name,
                                prof,
                                planned,
                                entry_time,
                                account_info=LAST_ACCOUNT_INFO,
                                mt5_last_error="portfolio_open_risk_cap",
                            )
                            continue
                    bucket = _symbol_bucket(prof.symbol_key)
                    bucket_cap = cap_bucket_caps.get(bucket)
                    if bucket_cap is not None and cap_bucket_positions.get((bucket, planned.direction), 0) >= bucket_cap:
                        log_lifecycle(
                            f"EXPOSURE BLOCK | bucket={bucket} | direction={planned.direction} | cap={bucket_cap}"
                        )
                        if (
                            paper_shots_enabled
                            and london_trace
                            and london_trace.get("trigger_ok")
                            and not paper_shot_logged
                        ):
                            _append_paper_shot(
                                PAPER_SHOTS_LOG,
                                mode,
                                portfolio_name,
                                prof,
                                planned,
                                entry_time,
                                london_trace,
                                reason="exposure_block:bucket_cap",
                            )
                            paper_shot_logged = True
                            _append_stage_log(
                                trade_log,
                                "ORDER_REJECTED",
                                mode,
                                portfolio_name,
                                prof,
                                planned,
                                entry_time,
                                account_info=LAST_ACCOUNT_INFO,
                            mt5_last_error="bucket_cap",
                        )
                        continue
                    if portfolio_name != "combined" and cap_open_positions.get((prof.symbol_key, planned.direction), 0) > 0:
                        log_lifecycle(
                            f"EXPOSURE BLOCK | symbol={prof.symbol_key} | direction={planned.direction} | reason=position_exists"
                        )
                        if (
                            paper_shots_enabled
                            and london_trace
                            and london_trace.get("trigger_ok")
                            and not paper_shot_logged
                        ):
                            _append_paper_shot(
                                PAPER_SHOTS_LOG,
                                mode,
                                portfolio_name,
                                prof,
                                planned,
                                entry_time,
                                london_trace,
                                reason="exposure_block:position_exists",
                            )
                            paper_shot_logged = True
                        _append_stage_log(
                            trade_log,
                            "ORDER_REJECTED",
                            mode,
                            portfolio_name,
                            prof,
                            planned,
                            entry_time,
                                account_info=LAST_ACCOUNT_INFO,
                            mt5_last_error="position_exists",
                        )
                        continue
                    if portfolio_name != "combined" and bucket_cap is None and cap_bucket_positions.get((bucket, planned.direction), 0) > 0:
                        log_lifecycle(
                            f"EXPOSURE BLOCK | bucket={bucket} | direction={planned.direction} | reason=bucket_position_exists"
                        )
                        if (
                            paper_shots_enabled
                            and london_trace
                            and london_trace.get("trigger_ok")
                            and not paper_shot_logged
                        ):
                            _append_paper_shot(
                                PAPER_SHOTS_LOG,
                                mode,
                                portfolio_name,
                                prof,
                                planned,
                                entry_time,
                                london_trace,
                                reason="exposure_block:bucket_position_exists",
                            )
                            paper_shot_logged = True
                        _append_stage_log(
                            trade_log,
                            "ORDER_REJECTED",
                            mode,
                            portfolio_name,
                            prof,
                            planned,
                            entry_time,
                            account_info=LAST_ACCOUNT_INFO,
                            mt5_last_error="bucket_position_exists",
                        )
                        continue
                    if research_enabled and research_risk_cap_pct > 0:
                        max_risk = equity * research_risk_cap_pct
                        if planned.risk_amount > max_risk:
                            log_lifecycle(
                                f"RESEARCH BLOCK | symbol={prof.symbol_key} | reason=risk_cap "
                                f"| risk={planned.risk_amount:.4f} cap={max_risk:.4f}"
                            )
                            _append_stage_log(
                                trade_log,
                                "ORDER_REJECTED",
                                mode,
                                portfolio_name,
                                prof,
                                planned,
                                entry_time,
                                account_info=LAST_ACCOUNT_INFO,
                                mt5_last_error="risk_cap",
                            )
                            continue

                _append_stage_log(
                    trade_log,
                    "ORDER_ATTEMPT",
                    mode,
                    portfolio_name,
                    prof,
                    planned,
                    entry_time,
                    account_info=LAST_ACCOUNT_INFO,
                )

                if mode == "shadow":
                    outcome = simulate_trade_path(
                        ohlc=base_df,
                        entry_idx=entry_idx,
                        account_balance=equity,
                        config=strategy_cfg,
                        costs=prof.costs,
                    )
                    equity_before = equity
                    equity = equity + outcome.pnl
                    _append_stage_log(
                        trade_log,
                        "ORDER_REJECTED",
                        mode,
                        portfolio_name,
                        prof,
                        planned,
                        entry_time,
                        account_info=LAST_ACCOUNT_INFO,
                        mt5_retcode="SHADOW",
                        mt5_last_error="shadow_mode",
                    )
                    if dynamic_risk:
                        record_trade(policy_state, prof.symbol_key, outcome.pnl, now_ts, policy_config)
                else:
                    broker = brokers.get(prof.symbol_key)
                    if broker is None:
                        continue
                    ok = broker.send_order(planned)
                    LAST_ORDER_SEND_RESULT = broker.last_order_result
                    LAST_ORDER_SEND_ERROR = broker.last_order_error
                    ORDER_SEND_NONE_COUNT = sum(b.order_send_none_count for b in brokers.values())
                    if ok:
                        retcode = None
                        if broker.last_order_result:
                            retcode = broker.last_order_result.get("retcode")
                        if retcode == "DRY_RUN":
                            _append_stage_log(
                                trade_log,
                                "ORDER_REJECTED",
                                mode,
                                portfolio_name,
                                prof,
                                planned,
                                entry_time,
                                account_info=LAST_ACCOUNT_INFO,
                                mt5_retcode=retcode,
                                mt5_last_error="dry_run",
                                mt5_comment=broker.last_order_result.get("comment") if broker.last_order_result else None,
                            )
                        else:
                            _append_stage_log(
                                trade_log,
                                "ORDER_PLACED",
                                mode,
                                portfolio_name,
                                prof,
                                planned,
                                entry_time,
                                account_info=LAST_ACCOUNT_INFO,
                                mt5_order_ticket=broker.last_ticket,
                                mt5_retcode=retcode,
                                mt5_comment=broker.last_order_result.get("comment") if broker.last_order_result else None,
                            )
                        if mode in {"demo", "ftmo"}:
                            open_positions[(prof.symbol_key, planned.direction)] = (
                                open_positions.get((prof.symbol_key, planned.direction), 0) + 1
                            )
                            open_positions_total += 1
                            bucket = _symbol_bucket(prof.symbol_key)
                            bucket_positions[(bucket, planned.direction)] = (
                                bucket_positions.get((bucket, planned.direction), 0) + 1
                            )
                            if profile_group == "westbrook":
                                westbrook_open_positions[(prof.symbol_key, planned.direction)] = (
                                    westbrook_open_positions.get((prof.symbol_key, planned.direction), 0) + 1
                                )
                                westbrook_open_positions_total += 1
                                westbrook_bucket_positions[(bucket, planned.direction)] = (
                                    westbrook_bucket_positions.get((bucket, planned.direction), 0) + 1
                                )
                                westbrook_open_risk_pct += planned_risk_pct
                            else:
                                scorers_open_positions[(prof.symbol_key, planned.direction)] = (
                                    scorers_open_positions.get((prof.symbol_key, planned.direction), 0) + 1
                                )
                                scorers_open_positions_total += 1
                                scorers_bucket_positions[(bucket, planned.direction)] = (
                                    scorers_bucket_positions.get((bucket, planned.direction), 0) + 1
                                )
                                scorers_open_risk_pct += planned_risk_pct
                            total_open_risk_pct = scorers_open_risk_pct + westbrook_open_risk_pct
                            OPEN_POSITIONS_SCORERS = scorers_open_positions_total
                            OPEN_POSITIONS_WESTBROOK = westbrook_open_positions_total
                            OPEN_RISK_SCORERS_PCT = scorers_open_risk_pct
                            OPEN_RISK_WESTBROOK_PCT = westbrook_open_risk_pct
                            TOTAL_OPEN_RISK_PCT = total_open_risk_pct
                        append_attribution(broker.last_ticket, prof, planned.regime or "", planned.reason_code or "")
                    else:
                        retcode = None
                        if broker.last_order_result:
                            retcode = broker.last_order_result.get("retcode")
                        _append_stage_log(
                            trade_log,
                            "ORDER_REJECTED",
                            mode,
                            portfolio_name,
                            prof,
                            planned,
                            entry_time,
                            account_info=LAST_ACCOUNT_INFO,
                            mt5_retcode=retcode,
                            mt5_last_error=broker.last_order_error,
                            mt5_comment=broker.last_order_result.get("comment") if broker.last_order_result else None,
                        )

                last_trade_time[prof.name] = entry_time

            # closed trade reconciliation (demo/ftmo)
            if mode in {"demo", "ftmo"} and mt5 is not None:
                now_deal = time.time()
                from_dt = datetime.fromtimestamp(last_deal_check, tz=timezone.utc)
                to_dt = datetime.now(timezone.utc)
                deals = mt5.history_deals_get(from_dt, to_dt)
                last_deal_check = now_deal
                if deals:
                    in_entries = set()
                    out_entries = set()
                    for name in ("DEAL_ENTRY_IN", "DEAL_ENTRY_INOUT"):
                        if hasattr(mt5, name):
                            in_entries.add(getattr(mt5, name))
                    for name in ("DEAL_ENTRY_OUT", "DEAL_ENTRY_OUT_BY", "DEAL_ENTRY_OUT_BY_PROFIT", "DEAL_ENTRY_OUT_BY_LOSS"):
                        if hasattr(mt5, name):
                            out_entries.add(getattr(mt5, name))
                    for d in deals:
                        entry_flag = getattr(d, "entry", None)
                        if (in_entries or out_entries) and entry_flag not in (in_entries | out_entries):
                            continue
                        comment = (getattr(d, "comment", "") or "").strip()
                        if not _is_bot_comment(comment):
                            continue
                        deal_id = getattr(d, "ticket", None)
                        if deal_id in processed_deals:
                            continue
                        if _append_deal_execution_log(trade_log, d, mode, portfolio_name, account_info=LAST_ACCOUNT_INFO, equity_hint=equity):
                            processed_deals.add(deal_id)
                            if dynamic_risk and entry_flag in out_entries:
                                record_trade(policy_state, getattr(d, "symbol", ""), float(getattr(d, "profit", 0) or 0), now_deal, policy_config)

            # equity snapshot
            account_info = None
            account_error = None
            if mode in {"demo", "ftmo"} and mt5 is not None:
                account_info = mt5.account_info()
                term_info = mt5.terminal_info()
                if account_info is not None:
                    equity = float(getattr(account_info, "equity", equity))
                    trade_allowed, term_allowed, acct_allowed = _compute_trade_allowed(account_info, term_info)
                    LAST_TERMINAL_INFO = term_info
                    LAST_TERMINAL_TRADE_ALLOWED = term_allowed if term_allowed is None else bool(term_allowed)
                    LAST_ACCOUNT_TRADE_ALLOWED = acct_allowed if acct_allowed is None else bool(acct_allowed)
                    LAST_MT5_TRADE_ALLOWED = trade_allowed if trade_allowed is None else bool(trade_allowed)
                else:
                    account_error = "mt5_account_info_unavailable"
            if account_info is None and account_error is None:
                account_error = "mt5_disconnected" if mode in {"demo", "ftmo"} else "mt5_not_required"
            append_equity_log(equity_log, equity, mode, portfolio_name)

            now = time.time()
            if now - last_heartbeat >= heartbeat_seconds:
                write_heartbeat(mode, portfolio_name, equity)
                state_flag, state_reason = _state_for_lock(mode)
                write_state(state_flag, mode, portfolio_name, reason=state_reason, equity=equity, symbols=symbols)
                if IDENTITY_DATA:
                    write_identity_file(IDENTITY_DATA)
                raw_bot_status = "ARMED" if state_flag == "RUNNING" else "DISARMED"
                log_total, matched = _trade_log_counts(trade_log)
                LOG_EVENTS_TOTAL = log_total
                MATCHED_MT5_TICKETS_COUNT = matched
                UNMATCHED_LOG_ONLY_COUNT = max(0, log_total - matched)
                MT5_VERIFIED_CLOSED_TRADES_COUNT = len(processed_deals)
                EVIDENCE_TIER = "verified" if MATCHED_MT5_TICKETS_COUNT > 0 else "practice"
                mt5_ticks, mt5_tick_status, mt5_tick_reason = _mt5_tick_snapshot(symbols, now)
                mt5_truth, mt5_truth_status, mt5_truth_reason = _mt5_truth_snapshot(
                    now, positions_list
                )
                write_policy(policy)
                risk_payload, risk_ok, risk_reason = _build_risk_policy_payload(
                    policy, policy_config, dynamic_risk, symbols, now
                )
                write_account_snapshot(
                    account_info,
                    account_error,
                    raw_bot_status,
                    risk_policy=risk_payload,
                    risk_policy_ok=risk_ok,
                    risk_policy_reason=risk_reason,
                    mt5_truth=mt5_truth,
                    mt5_truth_status=mt5_truth_status,
                    mt5_truth_reason=mt5_truth_reason,
                    mt5_ticks=mt5_ticks,
                    mt5_tick_status=mt5_tick_status,
                    mt5_tick_reason=mt5_tick_reason,
                    extra_block_reason=research_block_reason,
                )
                if research_enabled and mode in {"demo", "ftmo"} and mt5 is not None:
                    research_meta = {
                        "active_edge_tag": active_edge_tag,
                        "active_symbols": active_symbols,
                        "session_window": session_window_raw,
                        "research_block_reason": research_block_reason,
                    }
                    packet, pkt_status, pkt_reason = _mt5_research_packet(now, account_info, positions_list)
                    if pkt_status == "ok":
                        packet["research"] = research_meta
                        _write_json_atomic(RESEARCH_PACKET_FILE, packet)
                    else:
                        _write_json_atomic(
                            RESEARCH_PACKET_FILE,
                            {
                                "packet_version": "play_packet_v1",
                                "generated_at_utc": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
                                "status": pkt_status,
                                "reason": pkt_reason,
                                "research": research_meta,
                            },
                        )
                log_lifecycle(f"HEARTBEAT | mode={mode} | portfolio={portfolio_name} | equity={equity:.2f}")
                last_heartbeat = now

            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        log_lifecycle(f"STOP | mode={mode} | portfolio={portfolio_name} | reason=user_interrupt")
        write_state("STOPPED", mode, portfolio_name, reason="user_interrupt", equity=equity, symbols=symbols)
    except Exception as exc:
        log_error(f"CRASH | mode={mode} | portfolio={portfolio_name} | error={exc}")
        set_stop_reason(f"mode={mode} | portfolio={portfolio_name} | reason=Runner crashed - see runner_errors.log")
        write_state("STOPPED", mode, portfolio_name, reason=str(exc), equity=equity, symbols=symbols)
    finally:
        mt5_shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OmegaFX live runner")
    parser.add_argument("--mode", choices=["shadow", "demo", "ftmo"], help="runner mode")
    parser.add_argument(
        "--portfolio",
        choices=["core", "v3", "multi", "scorers", "westbrook", "combined", "exp_v2", "exp_v3", "research"],
        help="portfolio selector",
    )
    args = parser.parse_args()
    if args.mode:
        os.environ["OMEGAFX_MODE"] = args.mode
    if args.portfolio:
        os.environ["OMEGAFX_PORTFOLIO"] = args.portfolio
    run()
