from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - optional dependency
    mt5 = None

from .runtime_loop import BrokerAdapter
from .strategy import PlannedTrade
from .logger import get_logger


@dataclass
class Mt5ConnectionConfig:
    login: int
    password: str
    server: str


class Mt5BrokerAdapter(BrokerAdapter):
    def __init__(self, conn: Mt5ConnectionConfig, symbol: str, dry_run: bool = False):
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package is not installed.")

        self.conn = conn
        self.symbol = symbol
        self.dry_run = dry_run
        self.logger = get_logger()

        self._connect()

    def send_order(self, trade: PlannedTrade) -> bool:
        """
        Map a PlannedTrade to an MT5 market order. Long-only assumed for now.
        """
        if mt5 is None:
            return False

        if self.dry_run:
            self.log(f"[DRY RUN] Would send order: {trade}")
            return True

        # Ensure symbol is selected
        if not mt5.symbol_select(trade.symbol, True):
            self.log(f"Failed to select symbol {trade.symbol}")
            return False

        price = trade.entry_price
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": trade.symbol,
            "volume": trade.lot_size,
            "type": mt5.ORDER_TYPE_BUY if trade.direction == "long" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": trade.stop_loss_price,
            "tp": trade.take_profit_price,
            "deviation": 20,
            "magic": 1001,
            "comment": "omegafx_v2",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            self.log(f"MT5 order_send returned None; last_error={mt5.last_error()}")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.log(f"MT5 order failed: retcode={result.retcode}, comment={result.comment}")
            if result.retcode in {
                mt5.TRADE_RETCODE_NO_CONNECTION,
                mt5.TRADE_RETCODE_SERVER_BUSY,
                mt5.TRADE_RETCODE_TRADE_DISABLED,
                mt5.TRADE_RETCODE_TERMINAL_OFFLINE,
            }:
                self.log("Attempting MT5 reconnect and retry...")
                if self._connect():
                    retry = mt5.order_send(request)
                    if retry and retry.retcode == mt5.TRADE_RETCODE_DONE:
                        self.log(f"Order sent after retry: ticket={retry.order}")
                        return True
                    self.log(f"Retry failed: retcode={getattr(retry,'retcode',None)}")
            return False

        self.log(f"Order sent: ticket={result.order}, price={price}")
        return True

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
