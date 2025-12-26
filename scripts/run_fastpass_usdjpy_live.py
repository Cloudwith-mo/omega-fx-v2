from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import replace
from datetime import datetime, timedelta
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

STATE_FILE = LOG_DIR / "state.json"
HEARTBEAT_FILE = LOG_DIR / "runner_heartbeat.txt"
STOP_REASON_FILE = LOG_DIR / "last_stop_reason.txt"
LIFECYCLE_LOG = LOG_DIR / "runner_lifecycle.log"
ERROR_LOG = LOG_DIR / "runner_errors.log"
ATTR_LOG = LOG_DIR / "trade_attribution.csv"

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
    ERROR_LOG.write_text(
        (ERROR_LOG.read_text(encoding="utf-8") if ERROR_LOG.exists() else "") + f"[{ts}] {message}\n",
        encoding="utf-8",
    )


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
    STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_heartbeat(mode: str, portfolio: str, equity: float) -> None:
    HEARTBEAT_FILE.write_text(
        f"{time.time()}|{equity}|{portfolio}|{mode}|{ACCOUNT_ID}", encoding="utf-8"
    )


def _read_local_creds() -> Dict[str, str]:
    account_id = ACCOUNT_ID or os.getenv("OMEGAFX_ACCOUNT_ID") or os.getenv("ACCOUNT_ID")
    candidates = []
    if account_id:
        candidates.append(ROOT / f"mt5_creds.{account_id}.local.bat")
    candidates.append(ROOT / "mt5_creds.local.bat")
    for path in candidates:
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
            _, rest = line.split(" ", 1)
            if "=" not in rest:
                continue
            key, val = rest.split("=", 1)
            creds[key.strip()] = val.strip().strip('"').strip("'")
        if creds:
            return creds
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

    local = _read_local_creds()
    env_login = _clean(os.getenv("OMEGAFX_MT5_LOGIN") or os.getenv("MT5_LOGIN"))
    env_password = _clean(os.getenv("OMEGAFX_MT5_PASSWORD") or os.getenv("MT5_PASSWORD"))
    env_server = _clean(os.getenv("OMEGAFX_MT5_SERVER") or os.getenv("MT5_SERVER"))
    login = env_login if _valid(env_login) else _clean(local.get("OMEGAFX_MT5_LOGIN") or local.get("MT5_LOGIN"))
    password = env_password if _valid(env_password) else _clean(local.get("OMEGAFX_MT5_PASSWORD") or local.get("MT5_PASSWORD"))
    server = env_server if _valid(env_server) else _clean(local.get("OMEGAFX_MT5_SERVER") or local.get("MT5_SERVER"))
    return login, password, server


def mt5_login() -> Tuple[bool, str | None]:
    if mt5 is None:
        return False, "MetaTrader5 package not installed"
    login, password, server = mt5_env_values()
    if not all([login, password, server]):
        return False, "MT5 credentials not set (OMEGAFX_MT5_LOGIN/PASSWORD/SERVER)"
    mt5_path = os.getenv("MT5_PATH") or os.getenv("OMEGAFX_MT5_PATH")
    if mt5_path:
        ok_init = mt5.initialize(path=mt5_path)
    else:
        ok_init = mt5.initialize()
    if not ok_init:
        return False, f"MT5 initialize failed: {mt5.last_error()}"
    if not mt5.login(login=int(login), password=password, server=server):
        mt5.shutdown()
        return False, f"MT5 login failed: {mt5.last_error()}"
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


def comment_for_trade(mode: str, portfolio: str, strategy_id: str, regime: str) -> str:
    return (
        f"OmegaFX|mode={mode}|portfolio={portfolio}|player={player_name(strategy_id)}|"
        f"strategy={strategy_id}|regime={regime}"
    )


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
    if "strategy=" not in comment:
        return None
    for part in comment.split("|"):
        if part.startswith("strategy="):
            return part.split("=", 1)[1].strip() or None
    return None


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


def run():
    mode = os.getenv("OMEGAFX_MODE", "shadow").lower()
    portfolio_name = os.getenv("OMEGAFX_PORTFOLIO", "core").lower()
    trade_log = Path(os.getenv("OMEGAFX_TRADE_LOG", str(LOG_DIR / "shadow_fastpass_usdjpy_core.csv")))
    equity_log = Path(os.getenv("OMEGAFX_EQUITY_LOG", str(LOG_DIR / "shadow_fastpass_usdjpy_core_equity.csv")))
    poll_seconds = int(os.getenv("OMEGAFX_POLL_SECONDS", "30"))
    heartbeat_seconds = int(os.getenv("OMEGAFX_HEARTBEAT_SECONDS", "30"))
    initial_equity = float(os.getenv("OMEGAFX_INITIAL_EQUITY", "10000"))
    dynamic_risk = os.getenv("DYNAMIC_RISK", "0").lower() in {"1", "true", "yes"}

    portfolio = portfolio_from_name(portfolio_name)
    profiles = iter_profiles(portfolio)
    symbols = sorted({p.symbol_key for p, _ in profiles})
    policy_config = load_config_from_env()
    policy_state = init_state(initial_equity)
    policy = default_policy(symbols)
    write_policy(policy)

    log_lifecycle(f"START | mode={mode} | portfolio={portfolio_name} | symbols={','.join(symbols)}")
    write_state("RUNNING", mode, portfolio_name, symbols=symbols, equity=initial_equity)

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
            brokers[sym] = Mt5BrokerAdapter(conn, sym, dry_run=False)

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
                planned.comment = comment_for_trade(mode, portfolio_name, prof.name, planned.regime)

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
                        if "OmegaFX|" not in comment:
                            continue
                        deal_id = getattr(d, "ticket", None)
                        if deal_id in processed_deals:
                            continue
                        if _update_trade_log_with_close(trade_log, d, equity):
                            processed_deals.add(deal_id)
                            if dynamic_risk:
                                record_trade(policy_state, getattr(d, "symbol", ""), float(getattr(d, "profit", 0) or 0), now_deal, policy_config)

            # equity snapshot
            if mode in {"demo", "ftmo"} and mt5 is not None:
                acct = mt5.account_info()
                if acct is not None:
                    equity = float(getattr(acct, "equity", equity))
            append_equity_log(equity_log, equity, mode, portfolio_name)

            now = time.time()
            if now - last_heartbeat >= heartbeat_seconds:
                write_heartbeat(mode, portfolio_name, equity)
                write_state("RUNNING", mode, portfolio_name, equity=equity, symbols=symbols)
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
