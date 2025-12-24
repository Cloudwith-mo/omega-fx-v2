import csv
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

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
    DEFAULT_PORTFOLIO_USDJPY_FASTPASS_V3,
    MULTI_PORTFOLIO_USDJPY_GBPJPY_FASTPASS,
    LondonBreakoutSignalConfig,
    LiquiditySweepSignalConfig,
    MomentumPinballSignalConfig,
    PortfolioProfile,
    StrategyProfile,
)
from omegafx_v2.profile_summary import _build_signals_for_profile
from omegafx_v2.runtime_loop import DummyBrokerAdapter
from omegafx_v2.sim import simulate_trade_path
from omegafx_v2.strategy import plan_single_trade
from omegafx_v2.signals import (
    build_london_breakout_signals,
    build_liquidity_sweep_signals,
    build_momentum_pinball_signals_m5,
)
from omegafx_v2.mt5_adapter import Mt5BrokerAdapter, Mt5ConnectionConfig
from omegafx_v2.logger import get_logger

logger = get_logger(__name__)

LOG_DIR = ROOT / "logs"
ERROR_LOG = LOG_DIR / "runner_errors.log"
STOP_REASON_FILE = LOG_DIR / "last_stop_reason.txt"
LIFECYCLE_LOG = LOG_DIR / "runner_lifecycle.log"
HEARTBEAT_FILE = LOG_DIR / "runner_heartbeat.txt"


def log_lifecycle(msg: str):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    LIFECYCLE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with LIFECYCLE_LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def write_stop_reason(reason: str, mode: str = "", portfolio: str = "", symbols=None):
    symbols = symbols or []
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload = f"{ts} | reason={reason}"
    if mode:
        payload += f" | mode={mode}"
    if portfolio:
        payload += f" | portfolio={portfolio}"
    if symbols:
        payload += f" | symbols={','.join(symbols)}"
    STOP_REASON_FILE.parent.mkdir(parents=True, exist_ok=True)
    STOP_REASON_FILE.write_text(payload, encoding="utf-8")


def get_mt5_creds():
    login = os.getenv("OMEGAFX_MT5_LOGIN") or os.getenv("MT5_LOGIN")
    password = os.getenv("OMEGAFX_MT5_PASSWORD") or os.getenv("MT5_PASSWORD")
    server = os.getenv("OMEGAFX_MT5_SERVER") or os.getenv("MT5_SERVER")
    return login, password, server


def init_mt5(mode: str, portfolio_name: str, symbols, require_login: bool) -> bool:
    if mt5 is None:
        if require_login:
            msg = "MetaTrader5 package is not installed; cannot run live/demo."
            write_stop_reason(msg, mode=mode, portfolio=portfolio_name, symbols=symbols)
            log_lifecycle(f"EXIT | {msg}")
            return False
        return True
    login, password, server = get_mt5_creds()
    if require_login and not all([login, password, server]):
        msg = "MT5 credentials not set (OMEGAFX_MT5_LOGIN/PASSWORD/SERVER)"
        write_stop_reason(msg, mode=mode, portfolio=portfolio_name, symbols=symbols)
        log_lifecycle(f"EXIT | {msg}")
        return False
    if not mt5.initialize():
        msg = f"MT5 initialize failed: {mt5.last_error()}"
        write_stop_reason(msg, mode=mode, portfolio=portfolio_name, symbols=symbols)
        log_lifecycle(f"EXIT | {msg}")
        return False
    if require_login:
        if not mt5.login(login=int(login), password=password, server=server):
            msg = f"MT5 login failed: {mt5.last_error()}"
            write_stop_reason(msg, mode=mode, portfolio=portfolio_name, symbols=symbols)
            log_lifecycle(f"EXIT | {msg}")
            mt5.shutdown()
            return False
    return True


def fetch_mt5_df(symbol: str, timeframe, lookback: int = 300):
    if mt5 is None:
        return None
    try:
        mt5.symbol_select(symbol, True)
    except Exception:
        return None
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, lookback)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")[["open", "high", "low", "close"]]
    return df.dropna()


def in_session(ts: pd.Timestamp, session) -> bool:
    if ts.weekday() not in session.allowed_weekdays:
        return False
    start = session.start_hour
    end = session.end_hour
    hour = ts.hour
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def prepare_profiles(portfolio: PortfolioProfile) -> list[StrategyProfile]:
    profiles: list[StrategyProfile] = []
    profit_target_fallback = float(os.getenv("OMEGAFX_PROFIT_TARGET_PCT", "0.07"))

    if hasattr(portfolio, "portfolios"):
        symbol_scales = getattr(portfolio, "symbol_risk_scales", []) or []
        for sym_idx, sub in enumerate(getattr(portfolio, "portfolios", [])):
            sym_scale = symbol_scales[sym_idx] if sym_idx < len(symbol_scales) else 1.0
            sub_scales = getattr(sub, "risk_scales", None) or [1.0] * len(sub.strategies)
            for idx, base in enumerate(sub.strategies):
                scale = sym_scale * (sub_scales[idx] if idx < len(sub_scales) else 1.0)
                pt = getattr(base, "challenge", None).profit_target_pct if getattr(base, "challenge", None) else profit_target_fallback
                scaled_strategy = replace(
                    base.strategy,
                    risk_per_trade_pct=base.strategy.risk_per_trade_pct * scale,
                    reward_per_trade_pct=base.strategy.reward_per_trade_pct * scale,
                    profit_target_pct=pt,
                )
                profiles.append(replace(base, strategy=scaled_strategy))
        return profiles

    scales = getattr(portfolio, "risk_scales", None) or [1.0] * len(portfolio.strategies)
    for idx, base in enumerate(portfolio.strategies):
        scale = scales[idx] if idx < len(scales) else 1.0
        pt = getattr(base, "challenge", None).profit_target_pct if getattr(base, "challenge", None) else profit_target_fallback
        scaled_strategy = replace(
            base.strategy,
            risk_per_trade_pct=base.strategy.risk_per_trade_pct * scale,
            reward_per_trade_pct=base.strategy.reward_per_trade_pct * scale,
            profit_target_pct=pt,
        )
        profiles.append(replace(base, strategy=scaled_strategy))
    return profiles


def append_trade(path: Path, profile: StrategyProfile, trade_plan, outcome, equity_before: float, equity_after: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "profile_name",
                "symbol",
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
                "run_id",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "profile_name": profile.name,
                "symbol": profile.symbol_key,
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
                "equity_after": equity_after,
                "run_id": os.getenv("OMEGAFX_RUN_ID", ""),
            }
        )


def write_equity(path: Path, ts: pd.Timestamp, equity: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["time", "equity"])
        writer.writerow([ts.isoformat(), equity])


def get_portfolio_choice() -> PortfolioProfile:
    choice = os.getenv("OMEGAFX_PORTFOLIO", "core").lower()
    if choice == "multi":
        return MULTI_PORTFOLIO_USDJPY_GBPJPY_FASTPASS
    return DEFAULT_PORTFOLIO_USDJPY_FASTPASS_V3


def main():
    live_mode = os.getenv("OMEGAFX_LIVE_MODE", "0") == "1"
    initial_equity = float(os.getenv("OMEGAFX_INITIAL_EQUITY", "10000"))
    trade_log = Path(os.getenv("OMEGAFX_TRADE_LOG", LOG_DIR / "shadow_fastpass_usdjpy_core.csv"))
    equity_log = Path(os.getenv("OMEGAFX_SHADOW_LOG", trade_log))
    heartbeat_interval = int(os.getenv("OMEGAFX_HEARTBEAT_SECONDS", "30"))
    poll_seconds = int(os.getenv("OMEGAFX_POLL_SECONDS", "30"))

    portfolio = get_portfolio_choice()
    profiles = prepare_profiles(portfolio)
    symbols = sorted({p.symbol_key for p in profiles})
    challenge = getattr(profiles[0], "challenge", None)
    daily_loss_pct = float(os.getenv("OMEGAFX_DAILY_LOSS_PCT", portfolio.portfolio_daily_loss_pct or (challenge.daily_loss_pct if challenge else 0.05)))
    max_loss_pct = portfolio.portfolio_max_loss_pct or (challenge.max_total_loss_pct if challenge else 0.10)
    profit_target_pct = (challenge.profit_target_pct if challenge else float(os.getenv("OMEGAFX_PROFIT_TARGET_PCT", "0.07")))
    target_equity = initial_equity * (1.0 + profit_target_pct) if profit_target_pct else float("inf")
    loss_limit_equity = initial_equity * (1.0 - max_loss_pct) if max_loss_pct else 0

    mode = "live" if live_mode else "shadow"
    write_stop_reason("runner starting", mode=mode, portfolio=portfolio.name, symbols=symbols)
    log_lifecycle(f"START | mode={mode} | portfolio={portfolio.name} | symbols={symbols} | start_equity={initial_equity}")
    logger.info("Starting runner | mode=%s | portfolio=%s | symbols=%s", mode, portfolio.name, symbols)

    require_login = live_mode  # demo/ftmo => live_mode True; shadow False
    if not init_mt5(mode, portfolio.name, symbols, require_login=require_login):
        return

    broker_map = {}
    if live_mode:
        login, password, server = get_mt5_creds()
        conn = Mt5ConnectionConfig(login=int(login), password=password, server=server)
        for sym in symbols:
            broker_map[sym] = Mt5BrokerAdapter(conn=conn, symbol=sym, dry_run=False)
    else:
        dummy = DummyBrokerAdapter()
        for sym in symbols:
            broker_map[sym] = dummy

    equity = initial_equity
    daily_pnl: dict = {}
    last_heartbeat = time.time()

    try:
        while True:
            df_m15_map = {}
            df_m5_map = {}
            for sym in symbols:
                df15 = fetch_mt5_df(sym, mt5.TIMEFRAME_M15, lookback=400) if mt5 else None
                df5 = fetch_mt5_df(sym, mt5.TIMEFRAME_M5, lookback=400) if mt5 else None
                if df15 is None or df5 is None or len(df15) < 2 or len(df5) < 2:
                    continue
                df_m15_map[sym] = df15
                df_m5_map[sym] = df5

            if not df_m15_map:
                time.sleep(poll_seconds)
                now = time.time()
                if now - last_heartbeat >= heartbeat_interval:
                    log_lifecycle(f"HEARTBEAT | mode={mode} | portfolio={portfolio.name} | equity={equity:.2f}")
                    HEARTBEAT_FILE.write_text(f"{int(now)}|{equity:.4f}|{portfolio.name}|{mode}", encoding="utf-8")
                    last_heartbeat = now
                continue

            log_lifecycle("ENTER MAIN LOOP | data fetched")

            for profile in profiles:
                sym = profile.symbol_key
                df_m15 = df_m15_map.get(sym)
                df_m5 = df_m5_map.get(sym)
                if df_m15 is None or df_m5 is None:
                    continue

                if isinstance(profile.signals, MomentumPinballSignalConfig):
                    signals = build_momentum_pinball_signals_m5(df_m5, df_m15, signal_config=profile.signals, session=profile.session)
                    df = df_m5
                elif isinstance(profile.signals, LondonBreakoutSignalConfig):
                    signals = build_london_breakout_signals(df_m15, signal_config=profile.signals, session=profile.session)
                    df = df_m15
                elif isinstance(profile.signals, LiquiditySweepSignalConfig):
                    signals = build_liquidity_sweep_signals(df_m15, signal_config=profile.signals, session=profile.session)
                    df = df_m15
                else:
                    signals = _build_signals_for_profile(df_m15, profile)
                    df = df_m15

                entry_idx = len(df) - 2
                if entry_idx < 0 or entry_idx >= len(signals):
                    continue
                sig_ts = df.index[entry_idx]

                day = sig_ts.date()
                day_limit = initial_equity * daily_loss_pct
                day_pnl = daily_pnl.get(day, 0.0)
                if day_limit and day_pnl <= -day_limit:
                    continue

                if equity >= target_equity or equity <= loss_limit_equity:
                    continue
                if not signals.iloc[entry_idx]:
                    continue
                if not in_session(sig_ts, profile.session):
                    continue

                key = (day, profile.name)
                max_trades = getattr(profile.signals, "max_trades_per_day", None)
                if max_trades:
                    if trades_per_day := daily_pnl.get(key):
                        if trades_per_day >= max_trades:
                            continue

                equity_before = equity
                trade_plan = plan_single_trade(
                    account_balance=equity,
                    current_price=df["close"].iloc[entry_idx],
                    config=profile.strategy,
                )

                outcome = simulate_trade_path(
                    ohlc=df,
                    entry_idx=entry_idx,
                    account_balance=equity,
                    config=profile.strategy,
                    costs=profile.costs,
                )

                broker = broker_map.get(sym, DummyBrokerAdapter())
                broker.send_order(trade_plan)  # in shadow, this is a dummy

                equity += outcome.pnl
                daily_pnl[day] = daily_pnl.get(day, 0.0) + outcome.pnl

                append_trade(trade_log, profile, trade_plan, outcome, equity_before, equity)
                write_equity(equity_log, outcome.exit_time, equity)

                if equity >= target_equity:
                    reason = "Challenge target hit; stopping loop."
                    write_stop_reason(reason, mode=mode, portfolio=portfolio.name, symbols=symbols)
                    log_lifecycle(f"EXIT | {reason}")
                    raise StopIteration
                if equity <= loss_limit_equity:
                    reason = "Max loss breached; stopping loop."
                    write_stop_reason(reason, mode=mode, portfolio=portfolio.name, symbols=symbols)
                    log_lifecycle(f"EXIT | {reason}")
                    raise StopIteration

            now = time.time()
            if now - last_heartbeat >= heartbeat_interval:
                log_lifecycle(f"HEARTBEAT | mode={mode} | portfolio={portfolio.name} | equity={equity:.2f}")
                HEARTBEAT_FILE.write_text(f"{int(now)}|{equity:.4f}|{portfolio.name}|{mode}", encoding="utf-8")
                last_heartbeat = now

            time.sleep(poll_seconds)

    except KeyboardInterrupt:
        write_stop_reason("Stopped by user (keyboard)", mode=mode, portfolio=portfolio.name, symbols=symbols)
        log_lifecycle("EXIT | keyboard interrupt")
    except StopIteration:
        pass  # stop reason already written
    except Exception:
        err = traceback.format_exc()
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        ERROR_LOG.write_text(err, encoding="utf-8")
        write_stop_reason("Runner error; see runner_errors.log", mode=mode, portfolio=portfolio.name, symbols=symbols)
        log_lifecycle("CRASH | see runner_errors.log")
        logger.error(err)
    finally:
        if mt5:
            mt5.shutdown()
        current_reason = STOP_REASON_FILE.read_text(encoding="utf-8").strip() if STOP_REASON_FILE.exists() else ""
        if (not current_reason) or (current_reason.lower().strip() == "runner starting"):
            write_stop_reason("Runner exited without explicit stop reason", mode=mode, portfolio=portfolio.name, symbols=symbols)
        log_lifecycle(f"EXIT COMPLETE | reason={current_reason or 'unknown'}")


if __name__ == "__main__":
    main()
