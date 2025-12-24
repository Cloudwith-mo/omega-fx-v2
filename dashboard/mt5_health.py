import os
from datetime import datetime, timedelta
from pathlib import Path

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


def check_mt5_health(symbols=None, days: int = 10):
    if symbols is None:
        symbols = ["USDJPY", "GBPJPY"]
    result = {
        "mt5_installed": mt5 is not None,
        "connected": False,
        "login": None,
        "server": None,
        "symbols": {},
        "error": None,
    }
    if mt5 is None:
        result["error"] = "MetaTrader5 package not installed"
        return result

    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    result["login"] = login
    result["server"] = server
    if not mt5.initialize():
        result["error"] = f"init_failed:{mt5.last_error()}"
        return result
    if all([login, password, server]):
        if not mt5.login(login=int(login), password=password, server=server):
            result["error"] = f"login_failed:{mt5.last_error()}"
            mt5.shutdown()
            return result
    result["connected"] = True

    end = datetime.now()
    start = end - timedelta(days=days)
    for sym in symbols:
        info = {"selected": False, "history_span": None, "bars_m15": 0, "bars_m5": 0, "error": None}
        try:
            if not mt5.symbol_select(sym, True):
                info["error"] = f"symbol_select_failed:{mt5.last_error()}"
            else:
                info["selected"] = True
                rates_m15 = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M15, start, end)
                rates_m5 = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M5, start, end)
                info["bars_m15"] = len(rates_m15) if rates_m15 is not None else 0
                info["bars_m5"] = len(rates_m5) if rates_m5 is not None else 0
                if rates_m15 is not None and len(rates_m15) > 0:
                    info["history_span"] = {
                        "start": datetime.fromtimestamp(rates_m15[0]["time"]).isoformat(),
                        "end": datetime.fromtimestamp(rates_m15[-1]["time"]).isoformat(),
                    }
        except Exception as exc:
            info["error"] = str(exc)
        result["symbols"][sym] = info

    mt5.shutdown()
    return result
