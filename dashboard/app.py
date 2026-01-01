import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import subprocess

from flask import Flask, jsonify, request, send_from_directory

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

ROOT = Path(__file__).resolve().parent.parent
_log_root = os.getenv("LOG_ROOT")
if _log_root:
    _log_root_path = Path(_log_root)
    if not _log_root_path.is_absolute():
        _log_root_path = ROOT / _log_root_path
else:
    _log_root_path = ROOT / "logs"
LOG_DIR = _log_root_path
REPORTS_DIR = ROOT / "reports"
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = LOG_DIR / "state.json"
COMMAND_FILE = LOG_DIR / "runner_command.json"
POLICY_FILE = LOG_DIR / "policy.json"
IDENTITY_FILE = LOG_DIR / "identity.json"
ACCOUNT_SNAPSHOT_FILE = LOG_DIR / "account_snapshot.json"
POLICY_FILE = LOG_DIR / "policy.json"
IDENTITY_STALE_SECONDS = int(os.getenv("OMEGAFX_IDENTITY_STALE_SECONDS", "300"))
ACCOUNT_SNAPSHOT_STALE_SECONDS = int(os.getenv("OMEGAFX_ACCOUNT_SNAPSHOT_STALE_SECONDS", "30"))
POLICY_STALE_SECONDS = int(os.getenv("OMEGAFX_POLICY_STALE_SECONDS", "300"))
WATCHDOG_ENABLED = os.getenv("OMEGAFX_WATCHDOG", "").lower() in {"1", "true", "yes"}
WATCHDOG_STALE_SECONDS = int(os.getenv("OMEGAFX_WATCHDOG_STALE_SECONDS", "60"))
WATCHDOG_POLL_SECONDS = int(os.getenv("OMEGAFX_WATCHDOG_POLL_SECONDS", "10"))

app = Flask(__name__)


def _read_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _age_seconds(ts_value: Optional[str], path: Path) -> Optional[float]:
    if ts_value:
        try:
            stamp = ts_value.rstrip("Z")
            dt = datetime.fromisoformat(stamp)
            return (datetime.utcnow() - dt).total_seconds()
        except Exception:
            pass
    try:
        return time.time() - path.stat().st_mtime
    except Exception:
        return None


def _read_identity():
    data = _read_json_file(IDENTITY_FILE)
    if not data:
        return {}, "missing", None, "identity.json missing"
    age = _age_seconds(data.get("timestamp_utc"), IDENTITY_FILE)
    if age is not None and age > IDENTITY_STALE_SECONDS:
        return data, "stale", age, f"identity stale ({int(age)}s)"
    return data, "ok", age, None


def _read_account_snapshot():
    data = _read_json_file(ACCOUNT_SNAPSHOT_FILE)
    if not data:
        return {}, "missing", None, "account_snapshot.json missing"
    age = _age_seconds(data.get("timestamp_utc"), ACCOUNT_SNAPSHOT_FILE)
    if age is not None and age > ACCOUNT_SNAPSHOT_STALE_SECONDS:
        return data, "stale", age, f"snapshot stale ({int(age)}s)"
    if data.get("mt5_connected") is False:
        return data, "disconnected", age, data.get("error") or "mt5_disconnected"
    return data, "ok", age, None


def _read_policy():
    data = _read_json_file(POLICY_FILE)
    if not data:
        return {}, "missing", None, "policy.json missing"
    age = _age_seconds(data.get("timestamp_utc"), POLICY_FILE)
    if age is not None and age > POLICY_STALE_SECONDS:
        return data, "stale", age, f"policy stale ({int(age)}s)"
    return data, "ok", age, None

CREDS_SOURCE = "missing"


def _read_local_creds():
    global CREDS_SOURCE
    account_id = os.getenv("ACCOUNT_ID") or os.getenv("OMEGAFX_ACCOUNT_ID")
    allow_fallback = os.getenv("OMEGAFX_ALLOW_LOCAL_CREDS_FALLBACK", "").lower() in {"1", "true", "yes"}
    candidates = []
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
    def _has_valid_creds(creds: dict) -> bool:
        login = creds.get("OMEGAFX_MT5_LOGIN") or creds.get("MT5_LOGIN")
        password = creds.get("OMEGAFX_MT5_PASSWORD") or creds.get("MT5_PASSWORD")
        server = creds.get("OMEGAFX_MT5_SERVER") or creds.get("MT5_SERVER")
        return _valid(login) and _valid(password) and _valid(server)
    for path, source in candidates:
        if not path.exists():
            continue
        creds = {}
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


def _inject_local_creds():
    local = _read_local_creds()
    def _should_override(val: str | None) -> bool:
        if not val:
            return True
        upper = val.strip().upper()
        if "YOUR_" in upper or upper in {"YOURLOGIN", "YOURPASSWORD", "YOUR_PASS"}:
            return True
        return False
    for key, val in local.items():
        if _should_override(os.getenv(key)):
            os.environ[key] = val

# Note: dashboard should not load MT5 creds; runner owns MT5 connectivity.

def _mt5_env_values():
    def _clean(val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        return val.strip().strip('"').strip("'")
    login = _clean(os.getenv("OMEGAFX_MT5_LOGIN") or os.getenv("MT5_LOGIN"))
    pw = _clean(os.getenv("OMEGAFX_MT5_PASSWORD") or os.getenv("MT5_PASSWORD"))
    srv = _clean(os.getenv("OMEGAFX_MT5_SERVER") or os.getenv("MT5_SERVER"))
    return login, pw, srv


# helper for MT5 credential status
def mt5_status():
    login, pw, srv = _mt5_env_values()
    return "ok" if all([login, pw, srv]) else "missing"


MT5_CACHE_SECONDS = int(os.getenv("OMEGAFX_MT5_CACHE_SECONDS", "10"))
_mt5_cache = {"ts": 0.0, "account": None, "deals": None, "error": None}


def _mt5_login():
    if mt5 is None:
        return False, "mt5_not_installed"
    login, password, server = _mt5_env_values()
    if not all([login, password, server]):
        return False, "mt5_creds_missing"
    mt5_path = os.getenv("MT5_PATH") or os.getenv("OMEGAFX_MT5_PATH")
    ok_init = mt5.initialize(path=mt5_path) if mt5_path else mt5.initialize()
    if not ok_init:
        return False, f"init_failed:{mt5.last_error()}"
    if not mt5.login(login=int(login), password=password, server=server):
        mt5.shutdown()
        return False, f"login_failed:{mt5.last_error()}"
    return True, None


def _mt5_snapshot():
    now = time.time()
    if _mt5_cache["ts"] and now - _mt5_cache["ts"] < MT5_CACHE_SECONDS:
        return _mt5_cache["account"], _mt5_cache["deals"], _mt5_cache["error"]
    ok, err = _mt5_login()
    if not ok:
        _mt5_cache.update({"ts": now, "account": None, "deals": None, "error": err})
        return None, None, err
    account = mt5.account_info()
    to_dt = datetime.utcnow()
    from_dt = to_dt - timedelta(days=10)
    deals = mt5.history_deals_get(from_dt, to_dt)
    mt5.shutdown()
    _mt5_cache.update({"ts": now, "account": account, "deals": deals, "error": None})
    return account, deals, None


def _use_mt5_metrics() -> bool:
    mode, _ = _current_mode_portfolio()
    if os.getenv("OMEGAFX_USE_MT5_METRICS", "").lower() in {"1", "true", "yes"}:
        return True
    return mode in {"demo", "ftmo"}


def _mt5_deals_to_trades(deals, account_balance):
    if not deals:
        return []
    trades = []
    for d in sorted(deals, key=lambda x: x.time, reverse=True):
        if not getattr(d, "symbol", None):
            continue
        profit = float(getattr(d, "profit", 0) or 0)
        if profit == 0:
            continue
        dtype = getattr(d, "type", None)
        side = "BUY" if dtype == getattr(mt5, "DEAL_TYPE_BUY", 0) else "SELL"
        ts = datetime.fromtimestamp(d.time).strftime("%Y-%m-%d %H:%M:%S")
        pnl_pct = None
        if account_balance:
            pnl_pct = (profit / account_balance) * 100.0
        trades.append({
            "timestamp": ts,
            "symbol": d.symbol,
            "strategy_id": getattr(d, "comment", "") or "",
            "strategy_name_friendly": _player_name(getattr(d, "comment", "") or ""),
            "side": side,
            "size": getattr(d, "volume", ""),
            "entry_price": getattr(d, "price", ""),
            "exit_price": "",
            "pnl_pct": pnl_pct,
            "equity_after_trade": None,
        })
        if len(trades) >= 50:
            break
    return trades

# Runner state
runner_proc: Optional[subprocess.Popen] = None
runner_mode: Optional[str] = None
runner_portfolio: Optional[str] = None
last_mode: Optional[str] = None
last_portfolio: Optional[str] = None
last_stop_reason: Optional[str] = None
last_heartbeat: Optional[float] = None
last_heartbeat_equity: Optional[float] = None
runner_active: bool = False
# current selection (even if runner fails to start)
current_mode: Optional[str] = None
current_portfolio: Optional[str] = None

# OpenAI client (expects OPENAI_API_KEY)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None

def _ai_status():
    if OpenAI is None:
        return False, "openai package not installed"
    if not OPENAI_API_KEY:
        return False, "OPENAI_API_KEY missing"
    return True, "ok"

# Full AI system prompt
AI_SYSTEM_PROMPT = """
You are OmegaFX AI Co-Pilot.
Roles: NBA-style announcer, prop trading assistant, quant analyst, risk manager, research coordinator, productivity coach.

ABSOLUTE RULES
- Use ONLY facts in the provided ai_context JSON.
- If a value is missing, say "unknown".
- Do NOT invent stats, pass rates, or optimizer results.
- If a field is not present, do not claim it exists.
- Always include an Evidence section listing the exact JSON fields used.

GOALS
- Explain live performance in plain language.
- Separate variance vs structural issues.
- Respect FTMO risk limits in all recommendations.
- Keep advice aligned with the $10k/month scaling goal.

STYLE
- Announcer Mode: energetic, concise, never reckless.
- Risk Manager Mode: serious, conservative, rule-based.
- Research Mode: analytical, cites evidence.

RESPONSE FORMAT (default)
- Summary: 2-4 sentences.
- Analysis: bullets.
- Recommendation: KEEP RUNNING / PAUSE / REVIEW (only if asked).
- Evidence: bullet list of fields used (example: live_metrics.pnl_today_pct, recent_trades[0].pnl_pct).

If asked about expected behavior, only compare against team_stats or optimizer_results if those values exist in ai_context.
"""

# AI Risk Brief cache (text-only, runs every X minutes)
AI_BRIEF_CACHE = {
    "ts": 0.0,
    "text": "",
    "error": None,
    "generated_at": None,
}
AI_BRIEF_LOCK = threading.Lock()

# simple helper to load comparison/pipeline stats by lineup
def _expected_stats(lineup: str):
    # map lineup to report filename
    mapping = {
        "core": "ftmo_usdjpy_v3_full_pipeline.json",
        "v3": "ftmo_usdjpy_v3_full_pipeline.json",
        "multi": "ftmo_multi_usdjpy_gbpjpy_full_pipeline.json",
        "exp_v2": "ftmo_usdjpy_exp_v2_full_pipeline.json",
        "exp_v2_multi": "ftmo_multi_usdjpy_exp_v2_full_pipeline.json",
        "exp_v3": "ftmo_usdjpy_exp_v3_full_pipeline.json",
        "exp_v3_multi": "ftmo_multi_usdjpy_exp_v3_full_pipeline.json",
    }
    fname = mapping.get(lineup)
    if not fname:
        return {}
    path = REPORTS_DIR / fname
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        "expected_winrate": data.get("pass_rate") or data.get("pass_rate_30d"),
        "expected_avg_return": data.get("avg_return") or data.get("avg_return_30d"),
        "expected_avg_loss_pct": data.get("avg_max_dd") or data.get("avg_max_dd_30d"),
        "expected_avg_win_pct": data.get("avg_return") or data.get("avg_return_30d"),
        "expected_dd_distribution": data.get("avg_max_dd"),
        "expected_losing_streak_range": data.get("losing_streak_range"),
    }

# Background task registry for pipelines/optimizers
task_procs: Dict[str, subprocess.Popen] = {}
task_progress: Dict[str, float] = {}
task_lock = threading.Lock()

RUNNER_ACTIVITY = LOG_DIR / "runner_activity.log"
LAST_STOP_FILE = LOG_DIR / "last_stop_reason.txt"


def write_activity(msg: str):
    RUNNER_ACTIVITY.parent.mkdir(exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    RUNNER_ACTIVITY.write_text(RUNNER_ACTIVITY.read_text() + line if RUNNER_ACTIVITY.exists() else line, encoding="utf-8")


def set_stop_reason(reason: str):
    global last_stop_reason
    last_stop_reason = reason
    LAST_STOP_FILE.write_text(reason, encoding="utf-8")


def read_stop_reason() -> Optional[str]:
    if LAST_STOP_FILE.exists():
        try:
            return LAST_STOP_FILE.read_text(encoding="utf-8").strip() or None
        except Exception:
            return None


def _read_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _write_state(state: str, mode: Optional[str], portfolio: Optional[str], reason: Optional[str] = None):
    data = {
        "state": state,
        "mode": mode,
        "portfolio": portfolio,
        "reason": reason,
        "updated_at": time.time(),
        "pid": os.getpid(),
    }
    try:
        STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def read_heartbeat():
    path = LOG_DIR / "runner_heartbeat.txt"
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
        parts = raw.split("|")
        if len(parts) >= 2:
            ts = float(parts[0])
            equity = float(parts[1])
            portfolio = parts[2] if len(parts) >= 3 else None
            mode = parts[3] if len(parts) >= 4 else None
            account_id = parts[4] if len(parts) >= 5 else None
            return {"time": ts, "equity": equity, "portfolio": portfolio, "mode": mode, "account_id": account_id}
    except Exception:
        return None
    return None


def clear_heartbeat():
    try:
        hb = LOG_DIR / "runner_heartbeat.txt"
        if hb.exists():
            hb.write_text("", encoding="utf-8")
    except Exception:
        pass
    return None


def mt5_preflight(mode: str, portfolio: str):
    if mode != "demo" and mode != "ftmo":
        return True, None
    if mt5 is None:
        return False, "MetaTrader5 package not installed"
    login, password, server = _mt5_env_values()
    if not all([login, password, server]):
        return False, "MT5 credentials not set (OMEGAFX_MT5_LOGIN/PASSWORD/SERVER)"
    mt5_path = os.getenv("MT5_PATH") or os.getenv("OMEGAFX_MT5_PATH")
    ok_init = mt5.initialize(path=mt5_path) if mt5_path else mt5.initialize()
    if not ok_init:
        return False, f"init_failed:{mt5.last_error()}"
    if not mt5.login(login=int(login), password=password, server=server):
        mt5.shutdown()
        return False, f"login_failed:{mt5.last_error()}"
    account = mt5.account_info()
    if not account or account.balance <= 0:
        mt5.shutdown()
        return False, "equity is 0"
    symbols = ["USDJPY"]
    if portfolio == "multi":
        symbols.append("GBPJPY")
    for sym in symbols:
        if not mt5.symbol_select(sym, True):
            mt5.shutdown()
            return False, f"symbol_select failed for {sym}"
    mt5.shutdown()
    return True, None


def start_runner(mode: str, portfolio: str):
    global runner_proc, runner_mode, runner_portfolio, last_mode, last_portfolio, last_heartbeat, last_heartbeat_equity, runner_active
    if runner_proc and runner_proc.poll() is None:
        # already running
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["OMEGAFX_MODE"] = mode
    env["OMEGAFX_LIVE_MODE"] = "0" if mode == "shadow" else "1"
    env["OMEGAFX_PORTFOLIO"] = portfolio
    # default logs
    if mode == "demo":
        if portfolio == "multi":
            env["OMEGAFX_EQUITY_LOG"] = str(LOG_DIR / "demo_multi_equity.csv")
            env["OMEGAFX_TRADE_LOG"] = str(LOG_DIR / "demo_multi_trades.csv")
        elif portfolio == "v3":
            env["OMEGAFX_EQUITY_LOG"] = str(LOG_DIR / "demo_usdjpy_v3_equity.csv")
            env["OMEGAFX_TRADE_LOG"] = str(LOG_DIR / "demo_usdjpy_v3_trades.csv")
    elif mode == "ftmo":
        env["OMEGAFX_EQUITY_LOG"] = str(LOG_DIR / "live_equity.csv")
        env["OMEGAFX_TRADE_LOG"] = str(LOG_DIR / "live_trades.csv")
    elif mode == "shadow":
        if portfolio == "multi":
            env.setdefault("OMEGAFX_TRADE_LOG", str(LOG_DIR / "shadow_multi_trades.csv"))
            env.setdefault("OMEGAFX_EQUITY_LOG", str(LOG_DIR / "shadow_multi_equity.csv"))
        else:
            env.setdefault("OMEGAFX_TRADE_LOG", str(LOG_DIR / "shadow_fastpass_usdjpy_core.csv"))
            env.setdefault("OMEGAFX_EQUITY_LOG", str(LOG_DIR / "shadow_fastpass_usdjpy_core_equity.csv"))

    cmd = ["python", str(ROOT / "scripts" / "run_fastpass_usdjpy_live.py")]
    runner_proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
    runner_mode = mode
    runner_portfolio = portfolio
    last_mode = mode
    last_portfolio = portfolio
    runner_active = True
    last_heartbeat = None
    last_heartbeat_equity = None
    clear_heartbeat()
    set_stop_reason(f"runner starting | mode={mode} | lineup={portfolio}")
    write_activity(f"Tip-Off | mode={mode} | portfolio={portfolio}")

    # monitor thread to capture exit
    def monitor():
        global runner_proc, runner_mode, runner_portfolio, runner_active
        if runner_proc is None:
            return
        code = runner_proc.wait()
        if code == 0 and read_stop_reason() in (None, "runner starting"):
            set_stop_reason("Runner exited without explicit stop reason")
        elif code != 0 and read_stop_reason() in (None, "runner starting"):
            set_stop_reason(f"Runner exited with code {code}")
        write_activity(f"Runner exited with code {code}")
        runner_proc = None
        runner_mode = None
        runner_portfolio = None
        runner_active = False

    threading.Thread(target=monitor, daemon=True).start()


def stop_runner(reason: str = "Stopped by user"):
    global runner_proc, runner_mode, runner_portfolio, current_mode, current_portfolio, runner_active
    if runner_proc and runner_proc.poll() is None:
        runner_proc.terminate()
        try:
            runner_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            runner_proc.kill()
    runner_proc = None
    runner_mode = None
    runner_portfolio = None
    runner_active = False
    clear_heartbeat()
    set_stop_reason(reason)
    write_activity(f"Timeout | {reason}")


def launch_task(name: str, script: Path):
    script_path = ROOT / "scripts" / script
    if not script_path.exists():
        return False
    with task_lock:
        if name in task_procs and task_procs[name].poll() is None:
            return True  # already running
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.Popen(["python", str(script_path)], cwd=ROOT, env=env)
    with task_lock:
        task_procs[name] = proc
        task_progress[name] = 0.0
    def monitor():
        proc.wait()
        with task_lock:
            task_progress[name] = 1.0
            task_procs.pop(name, None)
    threading.Thread(target=monitor, daemon=True).start()
    return True


@app.route("/start", methods=["POST"])
def start():
    global current_mode, current_portfolio
    data = request.get_json(force=True, silent=True) or {}
    mode = data.get("mode", "shadow")
    portfolio = data.get("portfolio", "core")
    current_mode, current_portfolio = mode, portfolio
    clear_heartbeat()
    _write_state("REQUESTED", mode, portfolio, reason="tip_off")
    stop_runner("runner restarting")
    start_runner(mode, portfolio)
    return jsonify({"ok": True})


@app.route("/stop", methods=["POST"])
def stop():
    stop_runner("Stopped by user")
    return jsonify({"ok": True})


@app.route("/reset_state", methods=["POST"])
def reset_state():
    global runner_proc, runner_mode, runner_portfolio, last_mode, last_portfolio, last_stop_reason, last_heartbeat, last_heartbeat_equity, current_mode, current_portfolio, runner_active
    stop_runner("Reset by user")
    runner_proc = None
    runner_mode = None
    runner_portfolio = None
    last_mode = None
    last_portfolio = None
    last_stop_reason = None
    last_heartbeat = None
    last_heartbeat_equity = None
    current_mode = None
    current_portfolio = None
    runner_active = False
    clear_heartbeat()
    try:
        LAST_STOP_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        STATE_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    write_activity("Hard reset invoked")
    return jsonify({"ok": True})

@app.route("/status")
def status():
    try:
        global runner_active
        state = _read_state() or {}
        state_state = state.get("state")
        state_mode = state.get("mode")
        state_portfolio = state.get("portfolio")
        hb = read_heartbeat()
        hb_time = hb["time"] if hb else None
        hb_equity = hb["equity"] if hb else None
        hb_mode = hb["mode"] if hb else None
        hb_portfolio = hb["portfolio"] if hb else None

        heartbeat_interval = int(os.getenv("OMEGAFX_HEARTBEAT_SECONDS", "30"))
        stale_window = heartbeat_interval * 4  # tolerate a few missed beats
        now = time.time()

        # prefer heartbeat freshness; fall back to local proc state if available
        proc_alive = runner_proc is not None and runner_proc.poll() is None
        fresh_hb = hb_time is not None and now - hb_time < stale_window
        running = bool(fresh_hb)

        ls = read_stop_reason()
        parsed_mode, parsed_portfolio = _parse_stop_reason(ls or "")
        if ls == "runner starting":
            ls = None
        if running:
            ls = None

        stop_time = None
        try:
            if LAST_STOP_FILE.exists():
                stop_time = LAST_STOP_FILE.stat().st_mtime
        except Exception:
            stop_time = None

        mode_val, port_val = _current_mode_portfolio()
        if mode_val == "shadow" and ls and "MT5" in ls:
            ls = None
        if ls and "mode=" in ls and mode_val not in ls:
            ls = None

        # include minimal live metrics for convenience
        _, metrics = _read_trades_for_today()
        identity, identity_status, identity_age, identity_reason = _read_identity()
        snapshot, snapshot_status, snapshot_age, snapshot_reason = _read_account_snapshot()
        policy, policy_status, policy_age, policy_reason = _read_policy()
        lock_status = snapshot.get("identity_lock_status") or identity.get("lock_status") or "unknown"
        lock_status = str(lock_status).lower()
        lock_reason = snapshot.get("identity_lock_reason") or identity.get("lock_reason") or identity_reason or snapshot_reason
        mt5_connected = snapshot.get("mt5_connected") if snapshot else None
        mt5_trade_allowed = snapshot.get("mt5_trade_allowed") if snapshot else None
        mt5_truth_status = snapshot.get("mt5_truth_status")
        mt5_truth_reason = snapshot.get("mt5_truth_reason")
        mt5_tick_status = snapshot.get("mt5_tick_status")
        mt5_tick_reason = snapshot.get("mt5_tick_reason")
        if snapshot_status != "ok":
            if not mt5_truth_status:
                mt5_truth_status = "unknown"
            if not mt5_truth_reason:
                mt5_truth_reason = snapshot_reason or snapshot_status
            if not mt5_tick_status:
                mt5_tick_status = "unknown"
            if not mt5_tick_reason:
                mt5_tick_reason = snapshot_reason or snapshot_status
        elif not mt5_truth_status:
            mt5_truth_status = "unknown"
            mt5_truth_reason = "mt5_truth_missing"
        if snapshot_status == "ok" and not mt5_tick_status:
            mt5_tick_status = "unknown"
            mt5_tick_reason = "mt5_tick_missing"
        dry_run_enabled = snapshot.get("dry_run_enabled") if snapshot else None
        trading_block_reason = snapshot.get("trading_block_reason") if snapshot else None
        if snapshot_status == "ok":
            runner_mt5_status = "CONNECTED" if mt5_connected else "DISCONNECTED"
            runner_mt5_reason = snapshot_reason
        elif snapshot_status == "disconnected":
            runner_mt5_status = "DISCONNECTED"
            runner_mt5_reason = snapshot_reason
        else:
            runner_mt5_status = "UNKNOWN"
            runner_mt5_reason = snapshot_reason

        ai_enabled, ai_reason = _ai_status()
        if not fresh_hb:
            raw_bot_status = "DOWN"
        elif state_state == "RUNNING":
            raw_bot_status = "ARMED"
        else:
            raw_bot_status = "DISARMED"
        if not fresh_hb:
            bot_status_effective = "DOWN"
        else:
            bot_status_effective = snapshot.get("bot_status_effective") or (
                "LOCKED" if lock_status != "ok" else raw_bot_status
            )
        trading_allowed = False
        if fresh_hb and snapshot_status == "ok":
            if "trading_allowed" in snapshot:
                trading_allowed = bool(snapshot.get("trading_allowed"))
            else:
                trading_allowed = bool(
                    raw_bot_status == "ARMED" and lock_status == "ok" and mt5_connected is True
                )
        bot_status = bot_status_effective
        if bot_status == "ARMED" and not trading_allowed:
            bot_status = "DISARMED"
        expected_login = identity.get("expected_login")
        expected_server = identity.get("expected_server")
        actual_login = identity.get("mt5_account_login")
        actual_server = identity.get("mt5_account_server")
        expected_ok = False
        if expected_login or expected_server:
            expected_ok = (
                (not expected_login or str(expected_login) == str(actual_login))
                and (not expected_server or expected_server == actual_server)
            )
        evidence_tier = snapshot.get("evidence_tier") or "practice"
        trading_path_ready = bool(
            (dry_run_enabled is False)
            and mt5_connected is True
            and mt5_trade_allowed is True
            and expected_ok
        )

        if running:
            runner_state = "RUNNING"
        else:
            runner_state = state_state or "STOPPED"
        if not fresh_hb:
            bot_status_reason = "heartbeat_stale"
        elif lock_status != "ok":
            bot_status_reason = lock_reason
        elif dry_run_enabled is True:
            bot_status_reason = "dry_run"
        elif snapshot.get("risk_policy_ok") is False:
            bot_status_reason = snapshot.get("risk_policy_reason") or policy_reason
        elif trading_block_reason:
            bot_status_reason = trading_block_reason
        else:
            bot_status_reason = ls
        return jsonify({
            "running": running,
            "runner_state": runner_state,
            "bot_status": bot_status,
            "bot_status_effective": bot_status_effective,
            "bot_status_reason": bot_status_reason,
            "mode": mode_val,
            "portfolio": port_val,
            "last_stop_reason": ls,
            "last_stop_time": stop_time,
            "last_mode": last_mode,
            "last_portfolio": last_portfolio,
            "last_heartbeat_time": hb_time or last_heartbeat,
            "last_heartbeat_equity": hb_equity or last_heartbeat_equity,
            "metrics": metrics,
            "identity": identity,
            "identity_status": identity_status,
            "identity_age_sec": identity_age,
            "identity_reason": identity_reason,
            "identity_lock_status": lock_status,
            "identity_lock_reason": lock_reason,
            "account_snapshot_status": snapshot_status,
            "account_snapshot_age_sec": snapshot_age,
            "account_snapshot_reason": snapshot_reason,
            "balance": snapshot.get("balance"),
            "equity": snapshot.get("equity"),
            "mt5_connected": mt5_connected,
            "mt5_trade_allowed": mt5_trade_allowed,
            "mt5_terminal_trade_allowed": snapshot.get("mt5_terminal_trade_allowed"),
            "mt5_account_trade_allowed": snapshot.get("mt5_account_trade_allowed"),
            "mt5_path": snapshot.get("mt5_path"),
            "dry_run_enabled": dry_run_enabled,
            "last_order_send_result": snapshot.get("last_order_send_result"),
            "last_order_send_error": snapshot.get("last_order_send_error"),
            "order_send_none_count": snapshot.get("order_send_none_count"),
            "mt5_verified_closed_trades_count": snapshot.get("mt5_verified_closed_trades_count"),
            "mt5_open_positions_count": snapshot.get("mt5_open_positions_count"),
            "log_events_total": snapshot.get("log_events_total"),
            "matched_to_mt5_tickets_count": snapshot.get("matched_to_mt5_tickets_count"),
            "unmatched_shadow_or_log_only_count": snapshot.get("unmatched_shadow_or_log_only_count"),
            "evidence_tier": evidence_tier,
            "mt5_account_login": identity.get("mt5_account_login"),
            "mt5_account_server": identity.get("mt5_account_server"),
            "mt5_company": identity.get("mt5_company") or identity.get("mt5_terminal_company"),
            "mt5_truth_status": mt5_truth_status,
            "mt5_truth_reason": mt5_truth_reason,
            "mt5_positions_count": snapshot.get("mt5_positions_count"),
            "mt5_positions_profit": snapshot.get("mt5_positions_profit"),
            "mt5_orders_count": snapshot.get("mt5_orders_count"),
            "mt5_deals_last_2h_count": snapshot.get("mt5_deals_last_2h_count"),
            "mt5_last_deal_time": snapshot.get("mt5_last_deal_time"),
            "mt5_tick_status": mt5_tick_status,
            "mt5_tick_reason": mt5_tick_reason,
            "mt5_tick_age_sec_by_symbol": snapshot.get("mt5_tick_age_sec_by_symbol"),
            "mt5_tick_time_by_symbol": snapshot.get("mt5_tick_time_by_symbol"),
            "mt5_tick_missing_symbols": snapshot.get("mt5_tick_missing_symbols"),
            "mt5_tick_stale_seconds": snapshot.get("mt5_tick_stale_seconds"),
            "trading_allowed": trading_allowed,
            "trading_path_ready": trading_path_ready,
            "trading_block_reason": trading_block_reason,
            "risk_policy": policy,
            "risk_policy_status": policy_status,
            "risk_policy_age_sec": policy_age,
            "risk_policy_reason": policy_reason,
            "risk_policy_ok": snapshot.get("risk_policy_ok"),
            "risk_policy_state": snapshot.get("risk_policy"),
            "risk_policy_state_reason": snapshot.get("risk_policy_reason"),
            "runner_mt5_status": runner_mt5_status,
            "runner_mt5_reason": runner_mt5_reason,
            "ai_enabled": ai_enabled,
            "ai_reason": ai_reason,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/diag")
def diag():
    try:
        state = _read_state() or {}
        hb = read_heartbeat()
        hb_time = hb["time"] if hb else None
        hb_equity = hb["equity"] if hb else None
        hb_mode = hb["mode"] if hb else None
        hb_portfolio = hb["portfolio"] if hb else None
        hb_account = hb.get("account_id") if hb else None
        now = time.time()
        hb_age = (now - hb_time) if hb_time else None
        proc_alive = runner_proc is not None and runner_proc.poll() is None
        stop_time = None
        try:
            if LAST_STOP_FILE.exists():
                stop_time = LAST_STOP_FILE.stat().st_mtime
        except Exception:
            stop_time = None
        trade_path = _trade_log_path()
        equity_path = _equity_log_path()
        identity, identity_status, identity_age, identity_reason = _read_identity()
        snapshot, snapshot_status, snapshot_age, snapshot_reason = _read_account_snapshot()
        policy, policy_status, policy_age, policy_reason = _read_policy()
        ai_enabled, ai_reason = _ai_status()
        dynamic_risk_enabled = os.getenv("DYNAMIC_RISK", "0").lower() in {"1", "true", "yes"}
        closed_stats = _closed_trade_stats()
        account_id = os.getenv("ACCOUNT_ID") or os.getenv("OMEGAFX_ACCOUNT_ID")
        dash_port = os.getenv("DASH_PORT") or os.getenv("OMEGAFX_DASH_PORT") or "5000"
        mt5_path = os.getenv("MT5_PATH") or os.getenv("OMEGAFX_MT5_PATH")
        effective_mode, effective_portfolio = _current_mode_portfolio()
        expected_login = identity.get("expected_login")
        expected_server = identity.get("expected_server")
        actual_login = identity.get("mt5_account_login")
        actual_server = identity.get("mt5_account_server")
        expected_ok = False
        if expected_login or expected_server:
            expected_ok = (
                (not expected_login or str(expected_login) == str(actual_login))
                and (not expected_server or expected_server == actual_server)
            )
        dry_run_enabled = snapshot.get("dry_run_enabled")
        trading_path_ready = bool(
            (dry_run_enabled is False)
            and snapshot.get("mt5_connected") is True
            and snapshot.get("mt5_trade_allowed") is True
            and expected_ok
        )
        return jsonify({
            "state": state,
            "current_mode": effective_mode,
            "current_portfolio": effective_portfolio,
            "runner_mode": runner_mode,
            "runner_portfolio": runner_portfolio,
            "last_mode": last_mode,
            "last_portfolio": last_portfolio,
            "runner_active": runner_active,
            "runner_pid": runner_proc.pid if runner_proc else None,
            "runner_poll": runner_proc.poll() if runner_proc else None,
            "proc_alive": proc_alive,
            "account_id": account_id,
            "log_root": str(LOG_DIR),
            "dash_port": dash_port,
            "mt5_path": mt5_path,
            "heartbeat": {
                "time": hb_time,
                "age_seconds": hb_age,
                "equity": hb_equity,
                "mode": hb_mode,
                "portfolio": hb_portfolio,
                "account_id": hb_account,
                "path": str(LOG_DIR / "runner_heartbeat.txt"),
            },
            "last_stop_reason": read_stop_reason(),
            "last_stop_time": stop_time,
            "identity": identity,
            "identity_status": identity_status,
            "identity_age_sec": identity_age,
            "identity_reason": identity_reason,
            "account_snapshot": snapshot,
            "account_snapshot_status": snapshot_status,
            "account_snapshot_age_sec": snapshot_age,
            "account_snapshot_reason": snapshot_reason,
            "mt5_connected": snapshot.get("mt5_connected"),
            "mt5_trade_allowed": snapshot.get("mt5_trade_allowed"),
            "mt5_terminal_trade_allowed": snapshot.get("mt5_terminal_trade_allowed"),
            "mt5_account_trade_allowed": snapshot.get("mt5_account_trade_allowed"),
            "mt5_path_used": snapshot.get("mt5_path"),
            "mt5_truth_status": snapshot.get("mt5_truth_status"),
            "mt5_truth_reason": snapshot.get("mt5_truth_reason"),
            "mt5_positions_count": snapshot.get("mt5_positions_count"),
            "mt5_positions_profit": snapshot.get("mt5_positions_profit"),
            "mt5_orders_count": snapshot.get("mt5_orders_count"),
            "mt5_deals_last_2h_count": snapshot.get("mt5_deals_last_2h_count"),
            "mt5_last_deal_time": snapshot.get("mt5_last_deal_time"),
            "mt5_tick_status": snapshot.get("mt5_tick_status"),
            "mt5_tick_reason": snapshot.get("mt5_tick_reason"),
            "mt5_tick_age_sec_by_symbol": snapshot.get("mt5_tick_age_sec_by_symbol"),
            "mt5_tick_time_by_symbol": snapshot.get("mt5_tick_time_by_symbol"),
            "mt5_tick_missing_symbols": snapshot.get("mt5_tick_missing_symbols"),
            "mt5_tick_stale_seconds": snapshot.get("mt5_tick_stale_seconds"),
            "dry_run_enabled": dry_run_enabled,
            "last_order_send_result": snapshot.get("last_order_send_result"),
            "last_order_send_error": snapshot.get("last_order_send_error"),
            "order_send_none_count": snapshot.get("order_send_none_count"),
            "mt5_verified_closed_trades_count": snapshot.get("mt5_verified_closed_trades_count"),
            "mt5_open_positions_count": snapshot.get("mt5_open_positions_count"),
            "log_events_total": snapshot.get("log_events_total"),
            "matched_to_mt5_tickets_count": snapshot.get("matched_to_mt5_tickets_count"),
            "unmatched_shadow_or_log_only_count": snapshot.get("unmatched_shadow_or_log_only_count"),
            "evidence_tier": snapshot.get("evidence_tier") or "practice",
            "trading_path_ready": trading_path_ready,
            "ai_enabled": ai_enabled,
            "ai_reason": ai_reason,
            "dynamic_risk_enabled": dynamic_risk_enabled,
            "policy": policy,
            "policy_status": policy_status,
            "policy_age_sec": policy_age,
            "policy_read_reason": policy_reason,
            "policy_global_risk_scale": policy.get("global_risk_scale"),
            "policy_symbol_enabled": policy.get("symbol_enabled"),
            "policy_cooldown_until": policy.get("cooldown_until"),
            "policy_reason": policy.get("reason"),
            "last_closed_trade_time": closed_stats.get("last_closed_trade_time"),
            "closed_trades_seen_today": closed_stats.get("closed_trades_seen_today"),
            "streaks_by_symbol": closed_stats.get("streaks_by_symbol"),
            "log_paths": {
                "runner_lifecycle": str(LOG_DIR / "runner_lifecycle.log"),
                "runner_errors": str(LOG_DIR / "runner_errors.log"),
                "last_stop_reason": str(LAST_STOP_FILE),
                "trade_log": str(trade_path),
                "trade_log_exists": trade_path.exists(),
                "equity_log": str(equity_path),
                "equity_log_exists": equity_path.exists(),
            },
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/daily_summary")
def daily_summary():
    return jsonify({
        "trades_today": 0,
        "pnl_today_pct": 0.0,
        "max_dd_today_pct": 0.0,
        "guardrail_hits": 0,
        "symbols": ["USDJPY", "GBPJPY"] if runner_portfolio == "multi" else ["USDJPY"],
    })


@app.route("/config_snapshot")
def config_snapshot():
    portfolio = runner_portfolio or "core"
    symbols = ["USDJPY", "GBPJPY"] if portfolio == "multi" else ["USDJPY"]
    return jsonify({
        "mode": runner_mode or "shadow",
        "portfolio": portfolio,
        "symbols": symbols,
        "risk_scales": {},
        "caps": {"daily_loss_pct": 0.05, "max_loss_pct": 0.095},
        "lineup": [],
    })


@app.route("/tasks")
def tasks():
    with task_lock:
        state = {k: (p.poll() is None) for k, p in task_procs.items()}
        progress = {k+"_progress": v for k, v in task_progress.items()}
    state.update(progress)
    return jsonify(state)


@app.route("/run_ftmo", methods=["POST"])
def run_ftmo():
    launch_task("ftmo", Path("scripts/run_ftmo_usdjpy_v3_full_pipeline_mt5.py").name)
    return jsonify({"ok": True})


@app.route("/run_ftmo_multi", methods=["POST"])
def run_ftmo_multi():
    launch_task("ftmo_multi", Path("scripts/run_ftmo_multi_usdjpy_gbpjpy_full_pipeline_mt5.py").name)
    return jsonify({"ok": True})


@app.route("/run_ftmo_exp", methods=["POST"])
def run_ftmo_exp():
    launch_task("ftmo_exp", Path("scripts/run_ftmo_usdjpy_exp_v2_full_pipeline_mt5.py").name)
    return jsonify({"ok": True})


@app.route("/run_ftmo_exp_multi", methods=["POST"])
def run_ftmo_exp_multi():
    launch_task("ftmo_exp_multi", Path("scripts/run_ftmo_multi_usdjpy_exp_v2_full_pipeline_mt5.py").name)
    return jsonify({"ok": True})


@app.route("/run_ftmo_exp_v3", methods=["POST"])
def run_ftmo_exp_v3():
    launch_task("ftmo_exp_v3", Path("scripts/demo_ftmo_usdjpy_exp_v3_full_pipeline_mt5.py").name)
    return jsonify({"ok": True})


@app.route("/run_ftmo_exp_v3_multi", methods=["POST"])
def run_ftmo_exp_v3_multi():
    launch_task("ftmo_exp_v3_multi", Path("scripts/demo_ftmo_multi_usdjpy_exp_v3_full_pipeline_mt5.py").name)
    return jsonify({"ok": True})


@app.route("/run_exp_optimizer", methods=["POST"])
def run_exp_optimizer():
    launch_task("exp_opt", Path("scripts/demo_optimize_usdjpy_exp_v2_risk_mt5.py").name)
    return jsonify({"ok": True})


@app.route("/run_exp_v2_kd_primary_optimizer", methods=["POST"])
def run_exp_v2_kd_primary_optimizer():
    launch_task("exp_opt_kd", Path("scripts/demo_optimize_usdjpy_exp_v2_kd_primary_mt5.py").name)
    return jsonify({"ok": True})


@app.route("/run_exp_v3_optimizer", methods=["POST"])
def run_exp_v3_optimizer():
    launch_task("exp_opt_v3", Path("scripts/demo_optimize_usdjpy_exp_v3_risk_mt5.py").name)
    return jsonify({"ok": True})


@app.route("/run_exp_v3_fastwindows", methods=["POST"])
def run_exp_v3_fastwindows():
    launch_task("exp_v3_fw", Path("scripts/demo_fastpass_usdjpy_exp_v3_fastwindows_mt5.py").name)
    return jsonify({"ok": True})


@app.route("/run_research", methods=["POST"])
def run_research():
    launch_task("research", Path("scripts/report_fastpass_research.py").name)
    return jsonify({"ok": True})


@app.route("/run_gbpjpy_test", methods=["POST"])
def run_gbpjpy_test():
    launch_task("gbpjpy", Path("scripts/demo_fastpass_gbpjpy_core_fastwindows_mt5.py").name)
    return jsonify({"ok": True})


@app.route("/download_ftmo_json")
def download_ftmo_json():
    return send_from_directory(REPORTS_DIR, "ftmo_usdjpy_v3_full_pipeline.json", as_attachment=True)


@app.route("/download_shadow_log")
def download_shadow_log():
    return send_from_directory(LOG_DIR, "shadow_fastpass_usdjpy_core.csv", as_attachment=True)


@app.route("/download_demo_log")
def download_demo_log():
    fname = "demo_multi_equity.csv" if runner_portfolio == "multi" else "demo_usdjpy_v3_equity.csv"
    return send_from_directory(LOG_DIR, fname, as_attachment=True)


@app.route("/apply_exp_v2_config", methods=["POST"])
def apply_exp_v2_config():
    src = REPORTS_DIR / "optimize_usdjpy_exp_v2_risk.json"
    if not src.exists():
        return jsonify({"error": "optimizer file missing"}), 400
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
        top = (data.get("results") or data)[0]
        (REPORTS_DIR / "exp_v2_selected_config.json").write_text(json.dumps(top, indent=2), encoding="utf-8")
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/apply_exp_v3_config", methods=["POST"])
def apply_exp_v3_config():
    src = REPORTS_DIR / "optimize_usdjpy_exp_v3_risk.json"
    if not src.exists():
        return jsonify({"error": "optimizer file missing"}), 400
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
        top_list = data.get("results") or data
        if not top_list:
            return jsonify({"error": "no optimizer rows"}), 400
        top = top_list[0]
        (REPORTS_DIR / "exp_v3_selected_config.json").write_text(json.dumps(top, indent=2), encoding="utf-8")
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/optimizer")
def optimizer():
    out = {}
    exp_opt = REPORTS_DIR / "optimize_usdjpy_exp_v2_risk.json"
    kd_opt = REPORTS_DIR / "optimize_usdjpy_exp_v2_kd_primary.json"
    v3_opt = REPORTS_DIR / "optimize_usdjpy_exp_v3_risk.json"
    v3_selected = REPORTS_DIR / "exp_v3_selected_config.json"
    def load(path):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None
    out["results"] = load(exp_opt)
    out["kd_primary"] = load(kd_opt)
    out["v3"] = load(v3_opt)
    best = None
    if v3_selected.exists():
        best = load(v3_selected)
    elif out["v3"]:
        try:
            if isinstance(out["v3"], list):
                best = out["v3"][0] if out["v3"] else None
            else:
                best = (out["v3"].get("results") or out["v3"])[0]
        except Exception:
            best = None
    out["v3_best"] = best
    return jsonify(out)


@app.route("/exp_optimizer_results")
def exp_optimizer_results():
    return optimizer()


@app.route("/comparison")
def comparison():
    def load(name):
        path = REPORTS_DIR / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return jsonify({
        "v3": load("ftmo_usdjpy_v3_full_pipeline.json"),
        "gbpjpy_core": load("fastpass_gbpjpy_core_fastwindows.json"),
        "multi_symbol": load("ftmo_multi_usdjpy_gbpjpy_full_pipeline.json"),
        "exp_v2": load("ftmo_usdjpy_exp_v2_full_pipeline.json"),
        "exp_v2_multi": load("ftmo_multi_usdjpy_exp_v2_full_pipeline.json"),
        "exp_v3": load("ftmo_usdjpy_exp_v3_full_pipeline.json"),
        "exp_v3_multi": load("ftmo_multi_usdjpy_exp_v3_full_pipeline.json"),
    })


@app.route("/portfolio_comparison")
def portfolio_comparison():
    return comparison()


@app.route("/recent_reports")
def recent_reports():
    items = []
    for p in REPORTS_DIR.glob("*.json"):
        try:
            stat = p.stat()
            items.append({"name": p.name, "mtime": stat.st_mtime, "size": stat.st_size})
        except Exception:
            continue
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return jsonify(items[:20])


@app.route("/events")
def events():
    if RUNNER_ACTIVITY.exists():
        try:
            return jsonify(RUNNER_ACTIVITY.read_text(encoding="utf-8").splitlines()[-50:])
        except Exception:
            return jsonify([])
    return jsonify([])


@app.route("/")
def index():
    return send_from_directory(ROOT / "dashboard" / "static", "index.html")


@app.route("/events_stream")
def events_stream():
    # simple placeholder to avoid 404 noise
    return jsonify({"ok": True})


def _player_name(profile_name: str) -> str:
    mapping = {
        "USDJPY_M15_LondonBreakout_V1": "Curry",
        "GBPJPY_M15_LondonBreakout_V1": "Klay",
        "USDJPY_H1_BigMan_V1": "Big Man",
        "USDJPY_M15_VanVleet_V1": "VanVleet",
        "USDJPY_M15_LiquiditySweep_V1": "Kawhi",
        "GBPJPY_M15_LiquiditySweep_V1": "Kawhi (GBP)",
        "USDJPY_M5_MomentumPinball_V1": "Westbrook (bench)",
        "USDJPY_M15_TrendKD_V1": "KD",
    }
    return mapping.get(profile_name, profile_name)


def _current_mode_portfolio():
    state = _read_state() or {}
    state_mode = state.get("mode")
    state_portfolio = state.get("portfolio")
    state_state = state.get("state")
    hb = read_heartbeat()
    hb_mode = hb["mode"] if hb else None
    hb_portfolio = hb["portfolio"] if hb else None
    hb_time = hb["time"] if hb else None
    heartbeat_interval = int(os.getenv("OMEGAFX_HEARTBEAT_SECONDS", "30"))
    stale_window = heartbeat_interval * 4
    hb_fresh = hb_time is not None and (time.time() - hb_time) < stale_window
    if hb_mode == "live":
        hb_mode = current_mode or runner_mode or last_mode or "demo"
    stop_text = read_stop_reason() or ""
    parsed_mode, parsed_portfolio = _parse_stop_reason(stop_text)
    if current_mode or current_portfolio:
        mode = current_mode or state_mode or hb_mode or runner_mode or last_mode or parsed_mode or "shadow"
        portfolio = current_portfolio or state_portfolio or hb_portfolio or runner_portfolio or last_portfolio or parsed_portfolio or "core"
        return mode, portfolio
    if state_mode or state_portfolio:
        mode = state_mode or hb_mode or runner_mode or last_mode or parsed_mode or "shadow"
        portfolio = state_portfolio or hb_portfolio or runner_portfolio or last_portfolio or parsed_portfolio or "core"
        return mode, portfolio
    if hb_fresh:
        mode = hb_mode or state_mode or runner_mode or last_mode or parsed_mode or "shadow"
        portfolio = hb_portfolio or state_portfolio or runner_portfolio or last_portfolio or parsed_portfolio or "core"
    else:
        mode = state_mode or runner_mode or last_mode or parsed_mode or hb_mode or "shadow"
        portfolio = state_portfolio or runner_portfolio or last_portfolio or parsed_portfolio or hb_portfolio or "core"
    return mode, portfolio


def _parse_stop_reason(text: str) -> tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    mode_val = None
    port_val = None
    for chunk in text.split("|"):
        chunk = chunk.strip()
        if chunk.startswith("mode="):
            mode_val = chunk.split("=", 1)[1].strip()
        if chunk.startswith("portfolio="):
            port_val = chunk.split("=", 1)[1].strip()
        if chunk.startswith("lineup="):
            port_val = chunk.split("=", 1)[1].strip()
    return mode_val, port_val


def _trade_log_path():
    mode, portfolio = _current_mode_portfolio()
    if mode == "demo":
        if portfolio == "multi":
            return LOG_DIR / "demo_multi_trades.csv"
        return LOG_DIR / "demo_usdjpy_v3_trades.csv"
    if mode == "shadow":
        if portfolio == "multi":
            return LOG_DIR / "shadow_multi_trades.csv"
        return LOG_DIR / "shadow_fastpass_usdjpy_core.csv"
    # ftmo/live fallback
    return LOG_DIR / "live_trades.csv"


def _equity_log_path():
    mode, portfolio = _current_mode_portfolio()
    if mode == "demo":
        if portfolio == "multi":
            return LOG_DIR / "demo_multi_equity.csv"
        return LOG_DIR / "demo_usdjpy_v3_equity.csv"
    if mode == "shadow":
        if portfolio == "multi":
            return LOG_DIR / "shadow_multi_equity.csv"
        return LOG_DIR / "shadow_fastpass_usdjpy_core_equity.csv"
    # ftmo/live fallback
    return LOG_DIR / "live_equity.csv"


def _read_trades_for_today():
    path = _trade_log_path()
    import csv
    from collections import deque
    today = time.strftime("%Y-%m-%d")
    rows = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            dq = deque(reader, maxlen=500)  # bound memory
        for r in dq:
            ts = r.get("entry_time") or r.get("exit_time") or ""
            if today in ts:
                rows.append(r)
    except Exception:
        rows = []
    # compute metrics
    mode_val, port_val = _current_mode_portfolio()

    hb = read_heartbeat()
    hb_time = hb["time"] if hb else last_heartbeat
    hb_equity = hb["equity"] if hb else last_heartbeat_equity

    metrics = {
        "trades_today": 0,
        "wins_today": 0,
        "losses_today": 0,
        "pnl_today_pct": 0.0,
        "max_dd_today_pct": 0.0,
        "symbols_today": [],
        "last_trade_time": None,
        "start_equity_today": None,
        "end_equity_today": None,
        "last_heartbeat_time": hb_time,
        "last_heartbeat_equity": hb_equity,
        "mode": mode_val,
        "lineup_name": port_val,
        "losing_streak_current": 0,
        "losing_streak_max_today": 0,
        "losing_streak_by_player": {},
        "losing_streak_by_symbol": {},
        "trades_by_player": {},
        "wins_by_player": {},
        "losses_by_player": {},
        "pnl_by_player": {},
        "trades_by_symbol": {},
        "pnl_by_symbol": {},
    }
    if not path.exists():
        return rows, metrics
    if not rows:
        return rows, metrics

    symbols = set()
    start_eq = None
    last_eq = None
    min_eq = None

    current_streak = 0
    max_streak = 0
    streak_player = {}
    streak_symbol = {}

    for r in rows:
        pnl = float(r.get("pnl", 0) or 0)
        eq_after = r.get("equity_after")
        try:
            eq_after = float(eq_after) if eq_after not in (None, "", "None") else None
        except Exception:
            eq_after = None
        if eq_after is not None:
            if start_eq is None:
                start_eq = eq_after - pnl  # rough back-out
            last_eq = eq_after
            min_eq = eq_after if min_eq is None else min(min_eq, eq_after)

        metrics["trades_today"] += 1
        player = _player_name(r.get("profile_name") or "")
        sym = r.get("symbol")

        # streaks
        if pnl < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
            if player:
                streak_player[player] = max(streak_player.get(player, 0), current_streak)
            if sym:
                streak_symbol[sym] = max(streak_symbol.get(sym, 0), current_streak)
        else:
            current_streak = 0

        if pnl > 0:
            metrics["wins_today"] += 1
            metrics["wins_by_player"][player] = metrics["wins_by_player"].get(player, 0) + 1
        elif pnl < 0:
            metrics["losses_today"] += 1
            metrics["losses_by_player"][player] = metrics["losses_by_player"].get(player, 0) + 1

        if sym:
            symbols.add(sym)
            metrics["trades_by_symbol"][sym] = metrics["trades_by_symbol"].get(sym, 0) + 1
            metrics["pnl_by_symbol"][sym] = metrics["pnl_by_symbol"].get(sym, 0.0) + pnl
        if player:
            metrics["trades_by_player"][player] = metrics["trades_by_player"].get(player, 0) + 1
            metrics["pnl_by_player"][player] = metrics["pnl_by_player"].get(player, 0.0) + pnl
        ts = r.get("entry_time") or r.get("exit_time")
        if ts:
            metrics["last_trade_time"] = ts

    metrics["symbols_today"] = sorted(symbols)
    metrics["start_equity_today"] = start_eq if start_eq is not None else 0.0
    metrics["end_equity_today"] = last_eq if last_eq is not None else 0.0
    if start_eq and last_eq is not None:
        metrics["pnl_today_pct"] = ((last_eq - start_eq) / start_eq) * 100.0
    if start_eq and min_eq is not None:
        metrics["max_dd_today_pct"] = ((min_eq - start_eq) / start_eq) * 100.0
    metrics["losing_streak_current"] = current_streak
    metrics["losing_streak_max_today"] = max_streak
    metrics["losing_streak_by_player"] = streak_player
    metrics["losing_streak_by_symbol"] = streak_symbol

    return rows, metrics


def _equity_stats_today() -> Dict[str, Optional[float]]:
    path = _equity_log_path()
    if not path.exists():
        return {}
    import csv
    today = time.strftime("%Y-%m-%d")
    equities: List[float] = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                ts = r.get("time") or ""
                if today not in ts:
                    continue
                raw = r.get("equity")
                try:
                    eq = float(raw) if raw not in (None, "", "None") else None
                except Exception:
                    eq = None
                if eq is not None:
                    equities.append(eq)
    except Exception:
        return {}
    if not equities:
        return {}
    start_eq = equities[0]
    end_eq = equities[-1]
    min_eq = min(equities)
    out = {
        "start_equity_today": start_eq,
        "end_equity_today": end_eq,
        "pnl_today_pct": ((end_eq - start_eq) / start_eq) * 100.0 if start_eq else 0.0,
        "max_dd_today_pct": ((min_eq - start_eq) / start_eq) * 100.0 if start_eq else 0.0,
    }
    return out


def _closed_trade_stats():
    path = _trade_log_path()
    if not path.exists():
        return {"last_closed_trade_time": None, "closed_trades_seen_today": 0, "streaks_by_symbol": {}}
    import csv
    rows = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    except Exception:
        return {"last_closed_trade_time": None, "closed_trades_seen_today": 0, "streaks_by_symbol": {}}

    def _ts(row):
        return row.get("close_time") or row.get("exit_time") or ""

    rows_sorted = sorted(rows, key=_ts)
    today = time.strftime("%Y-%m-%d")
    closed_today = 0
    last_closed = None
    streaks: Dict[str, int] = {}
    for r in rows_sorted:
        ts = _ts(r)
        if not ts:
            continue
        last_closed = ts
        if today in ts:
            closed_today += 1
        sym = r.get("symbol") or ""
        pnl = 0.0
        raw = r.get("realized_profit") or r.get("pnl") or 0
        try:
            pnl = float(raw)
        except Exception:
            pnl = 0.0
        if pnl < 0:
            streaks[sym] = streaks.get(sym, 0) + 1
        else:
            streaks[sym] = 0

    return {
        "last_closed_trade_time": last_closed,
        "closed_trades_seen_today": closed_today,
        "streaks_by_symbol": streaks,
    }


@app.route("/recent_trades")
def recent_trades():
    path = _trade_log_path()
    if not path.exists():
        return jsonify([])
    rows = []
    try:
        import csv
        from collections import deque
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            dq = deque(reader, maxlen=200)
        for r in dq:
            pnl_pct = r.get("pnl_pct")
            try:
                pnl_pct = float(pnl_pct) if pnl_pct not in (None, "", "None") else None
            except Exception:
                pnl_pct = None
            eq_after = r.get("equity_after")
            try:
                eq_after = float(eq_after) if eq_after not in (None, "", "None") else None
            except Exception:
                eq_after = None
            rows.append({
                "timestamp": r.get("close_time") or r.get("exit_time") or r.get("entry_time") or "",
                "symbol": r.get("symbol") or "",
                "strategy_id": r.get("profile_name") or "",
                "strategy_name_friendly": _player_name(r.get("profile_name") or ""),
                "side": (r.get("direction") or "").upper(),
                "size": r.get("lot_size") or "",
                "entry_price": r.get("entry_price") or "",
                "exit_price": r.get("exit_price") or "",
                "pnl_pct": pnl_pct,
                "equity_after_trade": eq_after,
            })
    except Exception as exc:
        try:
            errlog = LOG_DIR / "runner_errors.log"
            errlog.parent.mkdir(exist_ok=True)
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            with errlog.open("a", encoding="utf-8") as f:
                f.write(f"[{ts}] recent_trades error: {exc}\n")
        except Exception:
            pass
        return jsonify({"error": str(exc)}), 500
    rows.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return jsonify(rows)


@app.route("/live_metrics")
def live_metrics():
    try:
        rows, metrics = _read_trades_for_today()
        eq_stats = _equity_stats_today()
        if eq_stats:
            for key, val in eq_stats.items():
                if val is not None:
                    metrics[key] = val

        metrics["source"] = "logs"
        snapshot, snapshot_status, snapshot_age, snapshot_reason = _read_account_snapshot()
        identity, identity_status, identity_age, identity_reason = _read_identity()
        if snapshot_status == "ok":
            metrics["account_balance"] = snapshot.get("balance")
            metrics["account_equity"] = snapshot.get("equity")
        else:
            metrics["account_balance"] = None
            metrics["account_equity"] = None
        metrics["account_snapshot_status"] = snapshot_status
        metrics["account_snapshot_age_sec"] = snapshot_age
        metrics["account_snapshot_reason"] = snapshot_reason
        metrics["account_login"] = identity.get("mt5_account_login")
        metrics["account_server"] = identity.get("mt5_account_server")
        # enrich with heartbeat if available
        hb_path = LOG_DIR / "runner_heartbeat.txt"
        if hb_path.exists():
            try:
                raw = hb_path.read_text(encoding="utf-8").strip()
                parts = raw.split("|")
                if len(parts) >= 2:
                    metrics["last_heartbeat_time"] = float(parts[0])
                    metrics["last_heartbeat_equity"] = float(parts[1])
                    if len(parts) >= 3:
                        metrics["last_heartbeat_portfolio"] = parts[2]
                    if len(parts) >= 4:
                        metrics["last_heartbeat_mode"] = parts[3]
            except Exception:
                pass
        mode_val, port_val = _current_mode_portfolio()
        metrics.setdefault("mode", mode_val)
        metrics.setdefault("lineup_name", port_val)
        return jsonify(metrics)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/ai_context")
def ai_context():
    try:
        ctx = {}
        try:
            ctx["status"] = status().get_json()
        except Exception:
            ctx["status"] = {}
        try:
            ctx["live_metrics"] = live_metrics().get_json()
        except Exception:
            ctx["live_metrics"] = {}
        try:
            ctx["recent_trades"] = recent_trades().get_json()
        except Exception:
            ctx["recent_trades"] = []
        try:
            ctx["config_snapshot"] = config_snapshot().get_json()
        except Exception:
            ctx["config_snapshot"] = {}
        try:
            ctx["team_stats"] = comparison().get_json()
        except Exception:
            ctx["team_stats"] = {}
        try:
            ctx["optimizer_results"] = optimizer().get_json()
        except Exception:
            ctx["optimizer_results"] = {}

        # simple events/hooks
        events = []
        lm = ctx.get("live_metrics", {}) or {}
        hb_time = lm.get("last_heartbeat_time")
        if hb_time:
            age = time.time() - hb_time
            if age > 15:
                events.append({"type": "heartbeat_stale", "age_seconds": age})
        trades = ctx.get("recent_trades", []) or []
        # losing streak detection (last 3 negatives)
        neg_streak = 0
        for t in reversed(trades[-5:]):
            try:
                pnl = float(t.get("pnl_pct", 0) or 0)
            except Exception:
                pnl = 0
            if pnl < 0:
                neg_streak += 1
            else:
                break
        if neg_streak >= 3:
            events.append({"type": "losing_streak", "length": neg_streak})
        try:
            dd = float(lm.get("max_dd_today_pct", 0) or 0)
            if dd < -3:
                events.append({"type": "dd_alert", "value": dd})
        except Exception:
            pass
        ctx["events"] = events
        return jsonify(ctx)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/ai_risk_brief")
def ai_risk_brief():
    """
    Returns a cached, text-only risk brief. Cache interval controlled by
    OMEGAFX_AI_BRIEF_MINUTES (default 5).
    """
    try:
        interval_minutes = float(os.getenv("OMEGAFX_AI_BRIEF_MINUTES", "5") or "5")
    except ValueError:
        interval_minutes = 5.0
    interval_seconds = max(30.0, interval_minutes * 60.0)
    now = time.time()

    def _iso(ts: float | None) -> str | None:
        if not ts:
            return None
        return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")

    with AI_BRIEF_LOCK:
        ts = AI_BRIEF_CACHE.get("ts", 0.0) or 0.0
        if ts and now - ts < interval_seconds:
            return jsonify({
                "text": AI_BRIEF_CACHE.get("text", "") or "",
                "generated_at": AI_BRIEF_CACHE.get("generated_at"),
                "cached": True,
                "next_update_in_seconds": max(0, int(interval_seconds - (now - ts))),
                "error": AI_BRIEF_CACHE.get("error"),
            })

    if client is None:
        detail = "OPENAI_API_KEY is missing." if not OPENAI_API_KEY else "openai package is not installed."
        text = f"AI Risk Brief unavailable: {detail}"
        with AI_BRIEF_LOCK:
            AI_BRIEF_CACHE.update({
                "ts": now,
                "text": text,
                "error": "ai_not_configured",
                "generated_at": _iso(now),
            })
        return jsonify({
            "text": text,
            "generated_at": _iso(now),
            "cached": False,
            "next_update_in_seconds": int(interval_seconds),
            "error": "ai_not_configured",
        })

    try:
        resp = ai_context()
        ctx = resp.get_json() if hasattr(resp, "get_json") else {}
    except Exception:
        ctx = {}
    context_str = json.dumps(ctx or {}, ensure_ascii=False, default=str)
    messages = [
        {"role": "system", "content": AI_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": (
            "Provide a concise, text-only AI Risk Brief using ONLY the JSON below.\n"
            "If a value is missing, say 'unknown'.\n"
            "Summarize:\n"
            "- Account state (mode, lineup, runner, equity, pnl, dd)\n"
            "- Risk flags (streaks, dd alerts, heartbeat)\n"
            "- Action recommended: YES/NO (judgment only, no instructions)\n"
            "Include one Evidence line listing JSON fields used (e.g., live_metrics.pnl_today_pct).\n"
            "Keep it under ~6 short lines. Do NOT include action steps.\n\n"
            "Context JSON:\n"
            f"```json\n{context_str}\n```"
        )},
    ]
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.2,
        )
        text = completion.choices[0].message.content.strip()
    except Exception as exc:
        text = "AI Risk Brief unavailable: failed to call AI backend."
        with AI_BRIEF_LOCK:
            AI_BRIEF_CACHE.update({
                "ts": now,
                "text": text,
                "error": str(exc),
                "generated_at": _iso(now),
            })
        return jsonify({
            "text": text,
            "generated_at": _iso(now),
            "cached": False,
            "next_update_in_seconds": int(interval_seconds),
            "error": str(exc),
        })

    with AI_BRIEF_LOCK:
        AI_BRIEF_CACHE.update({
            "ts": now,
            "text": text,
            "error": None,
            "generated_at": _iso(now),
        })
    return jsonify({
        "text": text,
        "generated_at": _iso(now),
        "cached": False,
        "next_update_in_seconds": int(interval_seconds),
        "error": None,
    })


@app.route("/health")
def health():
    return jsonify({"ok": True, "message": "dashboard backend up"}), 200


@app.route("/ai_query", methods=["POST"])
def ai_query():
    """
    Expected JSON payload:
    {
      "user_message": "string",
      "ai_context": { ... /ai_context payload ... }
    }
    """
    if client is None:
        detail = "OPENAI_API_KEY is missing." if not OPENAI_API_KEY else "openai package is not installed."
        return jsonify({
            "answer": f"AI backend not configured: {detail}",
            "error": "ai_not_configured"
        }), 500

    data = request.get_json(force=True, silent=True) or {}
    user_message = (data.get("user_message") or "").strip()
    ai_ctx = data.get("ai_context") or {}

    if not user_message:
        return jsonify({
            "answer": "Please ask a question or describe what you want me to analyze.",
            "error": "empty_message"
        }), 400

    try:
        st = status()
        st_json = st.get_json() if hasattr(st, "get_json") else {}
    except Exception:
        st_json = {}
    if (st_json.get("matched_to_mt5_tickets_count") or 0) == 0:
        tier = st_json.get("evidence_tier") or "practice"
        return jsonify({
            "answer": (
                "AI disabled for roster/risk calls: evidence tier is "
                f"{tier.upper()} (no MT5-verified trades)."
            ),
            "error": "insufficient_evidence",
        }), 403

    if not ai_ctx:
        try:
            resp = ai_context()
            ai_ctx = resp.get_json() if hasattr(resp, "get_json") else {}
        except Exception:
            ai_ctx = {}
    context_str = json.dumps(ai_ctx, ensure_ascii=False, default=str)
    messages = [
        {"role": "system", "content": AI_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": (
            "Here is the current trading context as JSON. "
            "Use it as the basis for your analysis:\n\n"
            f"```json\n{context_str}\n```\n\n"
            "If any value is missing, say 'unknown'. "
            "Include one Evidence line listing JSON fields used (e.g., live_metrics.pnl_today_pct).\n\n"
            f"Now answer this question from the Coach:\n\n{user_message}"
        )},
    ]
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.25,
        )
        answer = completion.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({
            "answer": "I ran into an error while calling the AI backend. Please check the server logs.",
            "error": str(e),
        }), 500


@app.route("/run_ledger")
def run_ledger():
    path = LOG_DIR / "run_events"
    if not path.exists():
        return jsonify([])
    try:
        return jsonify(path.read_text(encoding="utf-8").splitlines())
    except Exception:
        return jsonify([])


@app.route("/edge_impact")
def edge_impact():
    rows, _ = _read_trades_for_today()
    impact: Dict[str, Dict[str, float]] = {}
    for r in rows:
        player = _player_name(r.get("profile_name") or "")
        pnl = float(r.get("pnl", 0) or 0)
        pnl_pct = float(r.get("pnl_pct", 0) or 0)
        stats = impact.setdefault(player, {"trades": 0, "pnl_pct": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "wins": 0, "losses": 0})
        stats["trades"] += 1
        stats["pnl_pct"] += pnl_pct
        if pnl > 0:
            stats["wins"] += 1
            stats["avg_win"] += pnl_pct
        elif pnl < 0:
            stats["losses"] += 1
            stats["avg_loss"] += pnl_pct
    for stats in impact.values():
        if stats["wins"]:
            stats["avg_win"] = stats["avg_win"] / stats["wins"]
        if stats["losses"]:
            stats["avg_loss"] = stats["avg_loss"] / stats["losses"]
        stats.pop("wins", None)
        stats.pop("losses", None)
    return jsonify(impact)


@app.route("/check_market_data")
def check_market_data():
    snapshot, snapshot_status, snapshot_age, snapshot_reason = _read_account_snapshot()
    if snapshot_status != "ok":
        reason = snapshot_reason or snapshot_status
        return jsonify({"ok": False, "error": reason, "missing": [reason]}), 400
    return jsonify({"ok": True, "missing": []})


def _watchdog_loop():
    while True:
        time.sleep(WATCHDOG_POLL_SECONDS)
        state = _read_state() or {}
        state_flag = state.get("state")
        if state_flag == "STOPPED":
            continue
        hb = read_heartbeat()
        now = time.time()
        hb_time = hb.get("time") if hb else None
        if hb_time and (now - hb_time) <= WATCHDOG_STALE_SECONDS:
            continue
        mode = state.get("mode") or current_mode or runner_mode or last_mode
        portfolio = state.get("portfolio") or current_portfolio or runner_portfolio or last_portfolio
        if not mode or not portfolio:
            continue
        write_activity(f"Watchdog restart | mode={mode} | portfolio={portfolio} | heartbeat_stale")
        stop_runner("watchdog restarting")
        start_runner(mode, portfolio)


if WATCHDOG_ENABLED:
    threading.Thread(target=_watchdog_loop, daemon=True).start()


if __name__ == "__main__":
    port = int(os.getenv("DASH_PORT") or os.getenv("OMEGAFX_DASH_PORT") or "5000")
    app.run(host="0.0.0.0", port=port, debug=False)
