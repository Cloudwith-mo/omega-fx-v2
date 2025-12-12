from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - optional dependency
    mt5 = None

from .runtime_loop import BrokerAdapter
from .strategy import PlannedTrade
from .logger import get_logger

DRY_RUN = os.getenv("OMEGAFX_LIVE_MODE", "0") != "1"

@dataclass
class Mt5ConnectionConfig:
    login: int
    password: str
    server: str


class Mt5BrokerAdapter(BrokerAdapter):
    def __init__(self, conn: Mt5ConnectionConfig, symbol: str, dry_run: bool = False):
        if mt5 is None and not DRY_RUN:
            raise RuntimeError("MetaTrader5 package is not installed.")

        self.conn = conn
        self.symbol = symbol
        self.dry_run = dry_run or DRY_RUN
        self.logger = get_logger()
        self.digits: Optional[int] = None
        self.point: Optional[float] = None
        self.stop_level: Optional[float] = None
        self.contract_size: Optional[float] = None

        if self.dry_run:
            self.log("MT5 adapter in DRY RUN mode (no real connection).")
            return

        self._connect()
        self._load_symbol_info()

    def send_order(self, trade: PlannedTrade) -> bool:
        """
        Map a PlannedTrade to an MT5 market order. Long-only assumed for now.
        """
        if self.dry_run:
            self.log(
                f"DRY RUN: would send order: symbol={self.symbol}, lot={trade.lot_size}, "
                f"sl={trade.stop_loss_price}, tp={trade.take_profit_price}"
            )
            return True

        if mt5 is None:
            return False

        # Ensure symbol is selected
        if not mt5.symbol_select(trade.symbol, True):
            self.log(f"Failed to select symbol {trade.symbol}")
            return False
        price = self._normalize_price(trade.entry_price)
        sl = self._normalize_price(trade.stop_loss_price)
        tp = self._normalize_price(trade.take_profit_price)

        if not self._validate_stops(price, sl, tp):
            self.log("Aborting order due to invalid SL/TP distances")
            return False

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": trade.lot_size,
            "type": mt5.ORDER_TYPE_BUY if trade.direction == "long" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 20251212,
            "comment": "Omega-FX-v2",
            "type_filling": mt5.ORDER_FILLING_FOK,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        result = mt5.order_send(request)
        if result is None:
            self.log("MT5 order_send returned None")
            return False

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.log(f"Order success: ticket={result.order}")
            return True

        code = result.retcode
        comment = result.comment

        if code == mt5.TRADE_RETCODE_INVALID_VOLUME:
            self.log(f"Order failed: INVALID_VOLUME (volume={trade.lot_size}), comment={comment}")
        elif code == mt5.TRADE_RETCODE_INVALID_STOPS:
            self.log(f"Order failed: INVALID_STOPS (price={price}, sl={sl}, tp={tp}), comment={comment}")
        elif code == mt5.TRADE_RETCODE_INVALID_PRICE:
            self.log(f"Order failed: INVALID_PRICE (price={price}), comment={comment}")
        elif code in {
            mt5.TRADE_RETCODE_NO_CONNECTION,
            mt5.TRADE_RETCODE_SERVER_BUSY,
            mt5.TRADE_RETCODE_TRADE_DISABLED,
            mt5.TRADE_RETCODE_TERMINAL_OFFLINE,
        }:
            self.log(f"Order failed (connection issue): retcode={code}, comment={comment}. Retrying...")
            if self._connect():
                retry = mt5.order_send(request)
                if retry and retry.retcode == mt5.TRADE_RETCODE_DONE:
                    self.log(f"Order sent after retry: ticket={retry.order}")
                    return True
                self.log(f"Retry failed: retcode={getattr(retry,'retcode',None)}")
        else:
            self.log(f"Order failed: retcode={code}, comment={comment}")

        return False

    def log(self, message: str) -> None:
        self.logger.info(f"[MT5Broker] {message}")

    def close(self) -> None:
        if mt5:
            mt5.shutdown()

    def _connect(self) -> bool:
        if not mt5.initialize():
            self.log(f"MT5 initialize() failed, error code: {mt5.last_error()}")
            return False
        authorized = mt5.login(
            login=self.conn.login,
            password=self.conn.password,
            server=self.conn.server,
        )
        if not authorized:
            self.log(f"MT5 login failed, error: {mt5.last_error()}")
            return False
        return True

    def _load_symbol_info(self) -> None:
        if mt5 is None:
            return
        info = mt5.symbol_info(self.symbol)
        if info is None:
            raise RuntimeError(f"MT5 symbol {self.symbol} not found")
        self.digits = info.digits
        self.point = info.point
        self.stop_level = info.stop_level
        self.contract_size = info.trade_contract_size or info.contract_size
        self.log(
            f"Symbol {self.symbol}: digits={self.digits}, point={self.point}, "
            f"stop_level={self.stop_level}, contract_size={self.contract_size}"
        )

    def _normalize_price(self, price: float) -> float:
        if self.digits is None:
            return price
        factor = 10 ** self.digits
        return round(price * factor) / factor

    def _validate_stops(self, entry_price: float, sl: float, tp: float) -> bool:
        if self.stop_level is None or self.point is None:
            return True
        sl_dist = abs(entry_price - sl) / self.point
        tp_dist = abs(entry_price - tp) / self.point
        if sl_dist < self.stop_level or tp_dist < self.stop_level:
            self.log(
                f"Invalid stops: SL distance={sl_dist:.1f}, TP distance={tp_dist:.1f}, "
                f"required >= {self.stop_level}"
            )
            return False
        return True


# MT5 historical fetch helper
def fetch_symbol_ohlc_mt5(symbol: str, timeframe, start, end):
    """
    Fetch OHLC from MT5 between start and end for the given symbol/timeframe.
    Returns a DataFrame indexed by datetime with ['open','high','low','close'].
    Assumes MT5 is initialized and logged in elsewhere.
    """
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is not installed.")

    logger = get_logger(__name__)

    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None or len(rates) == 0:
        logger.error("No MT5 rates returned for %s between %s and %s", symbol, start, end)
        raise RuntimeError("No MT5 rates returned")

    import pandas as pd

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]]
    df = df.dropna()
    return df
