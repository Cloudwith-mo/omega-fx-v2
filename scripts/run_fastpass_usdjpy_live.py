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
    EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V2,
    EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V3,
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

STATE_FILE = LOG_DIR / "state.json"
HEARTBEAT_FILE = LOG_DIR / "runner_heartbeat.txt"
STOP_REASON_FILE = LOG_DIR / "last_stop_reason.txt"
LIFECYCLE_LOG = LOG_DIR / "runner_lifecycle.log"
ERROR_LOG = LOG_DIR / "runner_errors.log"
ERROR_CURRENT_LOG = LOG_DIR / "runner_errors_current.log"
ATTR_LOG = LOG_DIR / "trade_attribution.csv"
IDENTITY_FILE = LOG_DIR / "identity.json"
ACCOUNT_SNAPSHOT_FILE = LOG_DIR / "account_snapshot.json"

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


def _build_identity_payload(account, term, mt5_path: Optional[str]) -> Dict[str, object]:
    expected = _expected_identity_from_env()
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
) -> None:
    def _safe_float(val) -> Optional[float]:
        try:
            return float(val)
        except Exception:
            return None

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
        ok_init = mt5.initialize(path=mt5_path)
    else:
        log_lifecycle("MT5 initialize start | path=default")
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

    identity_payload = _build_identity_payload(account, term, mt5_path)
    expected = _expected_identity_from_env()
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


def player_name(strategy_id: str) -> str:
    mapping = {
        "USDJPY_M15_LondonBreakout_V1": "Curry",
        "GBPJPY_M15_LondonBreakout_V1": "Klay",
        "USDJPY_M15_LiquiditySweep_V1": "Kawhi",
        "GBPJPY_M15_LiquiditySweep_V1": "Kawhi (GBP)",
        "USDJPY_M5_MomentumPinball_V1": "Westbrook",
        "USDJPY_M15_TrendKD_V1": "KD",
        "USDJPY_M15_VanVleet_V1": "VanVleet",
        "USDJPY_H1_BigMan_V1": "Big Man",
    }
    return mapping.get(strategy_id, strategy_id)


def edge_tag(strategy_id: str) -> str:
    mapping = {
        "USDJPY_M15_LondonBreakout_V1": "Cu",
        "GBPJPY_M15_LondonBreakout_V1": "Kl",
        "USDJPY_M15_LiquiditySweep_V1": "Kw",
        "GBPJPY_M15_LiquiditySweep_V1": "Kg",
        "USDJPY_M5_MomentumPinball_V1": "We",
        "USDJPY_M15_TrendKD_V1": "KD",
        "USDJPY_M15_VanVleet_V1": "VV",
        "USDJPY_H1_BigMan_V1": "Bg",
    }
    return mapping.get(strategy_id, "Edge")


def lineup_tag(portfolio: str) -> str:
    mapping = {
        "core": "C",
        "v3": "V",
        "multi": "M",
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


def comment_for_trade(portfolio: str, strategy_id: str, play_id: Optional[str] = None) -> str:
    edge = edge_tag(strategy_id)
    lineup = lineup_tag(portfolio)
    acct = ACCOUNT_ID or "?"
    play = play_id or _next_play_id()
    return f"EDGE={edge}|PLAY={play}|ACCT={acct}|LINEUP={lineup}"


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


def append_trade_log(path: Path, row: Dict[str, object]) -> None:
    fieldnames = [
        "account_id",
        "ticket",
        "ticket_id",
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
    if path.exists():
        return
    fieldnames = [
        "account_id",
        "ticket",
        "ticket_id",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def ensure_equity_log(path: Path, equity: float, mode: str, portfolio: str) -> None:
    if path.exists():
        return
    append_equity_log(path, equity, mode, portfolio)


def _default_trade_log(mode: str, portfolio: str) -> Path:
    if mode == "demo":
        if portfolio == "multi":
            return LOG_DIR / "demo_multi_trades.csv"
        return LOG_DIR / "demo_usdjpy_v3_trades.csv"
    if mode == "shadow":
        if portfolio == "multi":
            return LOG_DIR / "shadow_multi_trades.csv"
        return LOG_DIR / "shadow_fastpass_usdjpy_core.csv"
    return LOG_DIR / "live_trades.csv"


def _default_equity_log(mode: str, portfolio: str) -> Path:
    if mode == "demo":
        if portfolio == "multi":
            return LOG_DIR / "demo_multi_equity.csv"
        return LOG_DIR / "demo_usdjpy_v3_equity.csv"
    if mode == "shadow":
        if portfolio == "multi":
            return LOG_DIR / "shadow_multi_equity.csv"
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


def _update_trade_log_with_close(trade_log: Path, deal, equity_hint: Optional[float]) -> bool:
    if not trade_log.exists():
        return False
    try:
        import csv
        with trade_log.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
    except Exception as exc:
        log_error(f"close_update read failed: {exc}")
        return False

    deal_ticket = str(getattr(deal, "ticket", ""))
    deal_order = str(getattr(deal, "order", ""))
    deal_position = str(getattr(deal, "position_id", ""))
    deal_comment = (getattr(deal, "comment", "") or "").strip()
    deal_symbol = getattr(deal, "symbol", "")
    deal_price = getattr(deal, "price", "")
    deal_time = datetime.fromtimestamp(deal.time).isoformat()
    deal_profit = float(getattr(deal, "profit", 0) or 0)

    updated = False
    for row in rows:
        ticket = (row.get("ticket") or "").strip()
        comment = (row.get("comment") or "").strip()
        if ticket and ticket in {deal_ticket, deal_order, deal_position}:
            pass
        elif comment and deal_comment and comment == deal_comment:
            pass
        else:
            continue

        if row.get("close_time"):
            return False

        equity_before = None
        try:
            equity_before = float(row.get("equity_before")) if row.get("equity_before") not in ("", None, "None") else None
        except Exception:
            equity_before = None
        if equity_before is None and equity_hint is not None:
            equity_before = equity_hint - deal_profit
        pnl_pct = None
        if equity_before and equity_before != 0:
            pnl_pct = (deal_profit / equity_before) * 100.0

        win_loss = "flat"
        if deal_profit > 0:
            win_loss = "win"
        elif deal_profit < 0:
            win_loss = "loss"

        row["close_time"] = deal_time
        row["exit_time"] = row.get("exit_time") or deal_time
        row["exit_price"] = row.get("exit_price") or deal_price
        row["realized_profit"] = deal_profit
        row["pnl_pct"] = pnl_pct if pnl_pct is not None else row.get("pnl_pct", "")
        row["win_loss"] = win_loss
        row["ticket_id"] = deal_ticket
        if equity_before is not None:
            row["equity_after"] = row.get("equity_after") or (equity_before + deal_profit)
        row["exit_reason"] = row.get("exit_reason") or "closed"

        updated = True
        break

    if not updated:
        # try to map by strategy in comment for a fallback row
        strategy_id = _parse_comment_strategy(deal_comment) or ""
        log_error(f"close_update no match: ticket={deal_ticket} order={deal_order} position={deal_position} comment={deal_comment} strategy={strategy_id}")
        return False

    # ensure headers include new fields
    extra_fields = [
        "ticket_id",
        "close_time",
        "realized_profit",
        "win_loss",
    ]
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    try:
        with trade_log.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
    except Exception as exc:
        log_error(f"close_update write failed: {exc}")
        return False

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
                if row.get("ticket_id") or row.get("close_time") or row.get("realized_profit"):
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
    for p in positions:
        symbol = getattr(p, "symbol", "")
        ptype = getattr(p, "type", None)
        direction = "long" if ptype == getattr(mt5, "POSITION_TYPE_BUY", 0) else "short"
        key = (symbol, direction)
        open_map[key] = open_map.get(key, 0) + 1
        bucket = "JPY" if symbol in {"USDJPY", "GBPJPY"} else symbol
        bkey = (bucket, direction)
        bucket_map[bkey] = bucket_map.get(bkey, 0) + 1
    return positions, {"symbol_dir": open_map, "bucket_dir": bucket_map}


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
    symbols = sorted({p.symbol_key for p, _ in profiles})
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
            )
    else:
        if mt5 is None:
            log_lifecycle("MT5 not installed; shadow will idle without data.")
        else:
            mt5_path = os.getenv("MT5_PATH") or os.getenv("OMEGAFX_MT5_PATH")
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
            open_positions = {}
            bucket_positions = {}
            positions_list = None
            if mode in {"demo", "ftmo"} and mt5 is not None:
                positions, pos_maps = _positions_snapshot()
                positions_list = positions
                MT5_OPEN_POSITIONS_COUNT = len(positions)
                open_positions = pos_maps.get("symbol_dir", {})
                bucket_positions = pos_maps.get("bucket_dir", {})

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
                if mode in {"demo", "ftmo"} and not trade_allowed_ok:
                    continue
                if dynamic_risk:
                    if in_cooldown:
                        continue
                    enabled = policy.get("symbol_enabled", {}).get(prof.symbol_key, True)
                    if not enabled:
                        continue
                tf = prof.timeframe
                ohlc_by_tf = {
                    tf_key[1]: df for tf_key, df in ohlc_cache.items() if tf_key[0] == prof.symbol_key
                }
                base_df = ohlc_by_tf.get(tf)
                if base_df is None or len(base_df) < 5:
                    continue
                signals = build_signals(prof, ohlc_by_tf)
                if signals.empty:
                    continue
                entry_idx = len(signals) - 2
                if entry_idx < 0 or not bool(signals.iloc[entry_idx]):
                    continue
                entry_time = base_df.index[entry_idx]
                last_t = last_trade_time.get(prof.name)
                if last_t is not None and entry_time <= last_t:
                    continue

                strategy_cfg = replace(
                    prof.strategy,
                    symbol=prof.symbol_key,
                    risk_per_trade_pct=prof.strategy.risk_per_trade_pct * scale * (policy.get("global_risk_scale", 1.0) if dynamic_risk else 1.0),
                    reward_per_trade_pct=prof.strategy.reward_per_trade_pct * scale * (policy.get("global_risk_scale", 1.0) if dynamic_risk else 1.0),
                )
                price = float(base_df["close"].iloc[entry_idx])
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
                planned.comment = comment_for_trade(portfolio_name, prof.name)

                if mode in {"demo", "ftmo"}:
                    bucket = "JPY" if prof.symbol_key in {"USDJPY", "GBPJPY"} else prof.symbol_key
                    if open_positions.get((prof.symbol_key, planned.direction), 0) > 0:
                        log_lifecycle(
                            f"EXPOSURE BLOCK | symbol={prof.symbol_key} | direction={planned.direction} | reason=position_exists"
                        )
                        continue
                    if bucket_positions.get((bucket, planned.direction), 0) > 0:
                        log_lifecycle(
                            f"EXPOSURE BLOCK | bucket={bucket} | direction={planned.direction} | reason=bucket_position_exists"
                        )
                        continue

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
                    append_trade_log(trade_log, {
                        "account_id": ACCOUNT_ID,
                        "ticket": "",
                        "magic_number": planned.magic_number,
                        "comment": planned.comment,
                        "profile_name": prof.name,
                        "symbol": prof.symbol_key,
                        "entry_time": outcome.entry_time.isoformat(),
                        "exit_time": outcome.exit_time.isoformat(),
                        "entry_date": outcome.entry_time.date().isoformat(),
                        "direction": planned.direction,
                        "lot_size": planned.lot_size,
                        "entry_price": outcome.entry_price,
                        "exit_price": outcome.exit_price,
                        "sl_price": planned.stop_loss_price,
                        "tp_price": planned.take_profit_price,
                        "exit_reason": outcome.exit_reason,
                        "pnl": outcome.pnl,
                        "pnl_pct": outcome.pnl_pct,
                        "equity_before": equity_before,
                        "equity_after": equity,
                        "run_id": "",
                    })
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
                        append_trade_log(trade_log, {
                            "account_id": ACCOUNT_ID,
                            "ticket": broker.last_ticket or "",
                            "magic_number": planned.magic_number,
                            "comment": planned.comment,
                            "profile_name": prof.name,
                            "symbol": prof.symbol_key,
                            "entry_time": entry_time.isoformat(),
                            "exit_time": "",
                            "entry_date": entry_time.date().isoformat(),
                            "direction": planned.direction,
                            "lot_size": planned.lot_size,
                            "entry_price": planned.entry_price,
                            "exit_price": "",
                            "sl_price": planned.stop_loss_price,
                            "tp_price": planned.take_profit_price,
                            "exit_reason": "sent",
                            "pnl": "",
                            "pnl_pct": "",
                            "equity_before": "",
                            "equity_after": "",
                            "run_id": "",
                        })
                        if mode in {"demo", "ftmo"}:
                            open_positions[(prof.symbol_key, planned.direction)] = (
                                open_positions.get((prof.symbol_key, planned.direction), 0) + 1
                            )
                            bucket = "JPY" if prof.symbol_key in {"USDJPY", "GBPJPY"} else prof.symbol_key
                            bucket_positions[(bucket, planned.direction)] = (
                                bucket_positions.get((bucket, planned.direction), 0) + 1
                            )
                        append_attribution(broker.last_ticket, prof, planned.regime or "", planned.reason_code or "")

                last_trade_time[prof.name] = entry_time

            # closed trade reconciliation (demo/ftmo)
            if mode in {"demo", "ftmo"} and mt5 is not None:
                now_deal = time.time()
                from_dt = datetime.fromtimestamp(last_deal_check, tz=timezone.utc)
                to_dt = datetime.now(timezone.utc)
                deals = mt5.history_deals_get(from_dt, to_dt)
                last_deal_check = now_deal
                if deals:
                    out_entries = set()
                    for name in ("DEAL_ENTRY_OUT", "DEAL_ENTRY_OUT_BY", "DEAL_ENTRY_OUT_BY_PROFIT", "DEAL_ENTRY_OUT_BY_LOSS"):
                        if hasattr(mt5, name):
                            out_entries.add(getattr(mt5, name))
                    for d in deals:
                        if out_entries and getattr(d, "entry", None) not in out_entries:
                            continue
                        comment = (getattr(d, "comment", "") or "").strip()
                        if not _is_bot_comment(comment):
                            continue
                        deal_id = getattr(d, "ticket", None)
                        if deal_id in processed_deals:
                            continue
                        if _update_trade_log_with_close(trade_log, d, equity):
                            processed_deals.add(deal_id)
                            if dynamic_risk:
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
        choices=["core", "v3", "multi", "exp_v2", "exp_v3"],
        help="portfolio selector",
    )
    args = parser.parse_args()
    if args.mode:
        os.environ["OMEGAFX_MODE"] = args.mode
    if args.portfolio:
        os.environ["OMEGAFX_PORTFOLIO"] = args.portfolio
    run()
