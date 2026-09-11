"""Alpaca adapter (paper by default).

Requires `pip install algofun[alpaca]` and these environment variables:

    ALPACA_API_KEY, ALPACA_SECRET_KEY
    ALPACA_PAPER=true|false   (default true; live needs an explicit false AND --live on the CLI)

Alpaca is commission-free, has an official REST API, supports fractional
market orders, and gives you a paper account with fake money. That is the
whole reason the live layer targets it rather than Robinhood, which has no
official stock-trading API for retail accounts.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterable
from functools import partial
from typing import TypeVar

import pandas as pd

from .base import Account, Broker, Order, OrderResult, Position

log = logging.getLogger(__name__)
T = TypeVar("T")


def with_retries(fn: Callable[[], T], attempts: int = 3, base_delay: float = 1.0,
                 what: str = "request") -> T:
    """Call fn(), retrying with exponential backoff. Re-raises the last error."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - broker/network errors are heterogeneous
            last = e
            if i + 1 < attempts:
                delay = base_delay * (2 ** i)
                log.warning("%s failed (%s); retry %d/%d in %.0fs", what, e, i + 1, attempts - 1, delay)
                time.sleep(delay)
    assert last is not None
    raise last


class AlpacaBroker(Broker):
    name = "alpaca"
    supports_fractional = True

    def __init__(self, api_key: str | None = None, secret_key: str | None = None,
                 paper: bool | None = None, data_feed: str | None = None):
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.trading.client import TradingClient
        except ImportError as e:  # pragma: no cover
            raise ImportError("alpaca-py is not installed: pip install 'algofun[alpaca]'") from e
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not self.api_key or not self.secret_key:
            raise RuntimeError("set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        if paper is None:
            paper = os.environ.get("ALPACA_PAPER", "true").strip().lower() != "false"
        self.paper = paper
        # free data plans only allow the IEX feed for recent trades; SIP needs a subscription
        self.data_feed = (data_feed or os.environ.get("ALPACA_DATA_FEED", "iex")).lower()
        self.name = "alpaca-paper" if paper else "alpaca-LIVE"
        self.trading = TradingClient(self.api_key, self.secret_key, paper=paper)
        self.data = StockHistoricalDataClient(self.api_key, self.secret_key)

    def account(self) -> Account:
        a = with_retries(self.trading.get_account, what="get account")
        return Account(cash=float(a.cash), equity=float(a.equity), buying_power=float(a.buying_power))

    def positions(self) -> dict[str, Position]:
        out = {}
        for p in with_retries(self.trading.get_all_positions, what="get positions"):
            out[p.symbol] = Position(p.symbol, float(p.qty), float(p.avg_entry_price), float(p.market_value))
        return out

    def latest_prices(self, tickers: Iterable[str]) -> pd.Series:
        """Latest trade price per ticker. Never raises: on a persistent data-API
        error it returns what it has (possibly empty) and the planner falls back
        to the last cached close."""
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestTradeRequest
        tickers = list(tickers)
        if not tickers:
            return pd.Series(dtype="float64")
        feed = DataFeed(self.data_feed) if self.data_feed in {"iex", "sip", "otc"} else None
        out: dict[str, float] = {}
        for i in range(0, len(tickers), 100):
            batch = tickers[i:i + 100]
            req = StockLatestTradeRequest(symbol_or_symbols=batch, feed=feed)
            try:
                trades = with_retries(partial(self.data.get_stock_latest_trade, req), attempts=3,
                                      what=f"latest trades for {len(batch)} symbols")
            except Exception as e:  # noqa: BLE001
                log.warning("latest prices unavailable for %d symbols (%s); using last close", len(batch), e)
                continue
            out.update({t: float(trades[t].price) for t in batch if t in trades})
        return pd.Series(out, dtype="float64")

    def is_market_open(self) -> bool:
        return bool(self.trading.get_clock().is_open)

    def cancel_open_orders(self) -> int:
        return len(self.trading.cancel_orders())

    def submit(self, order: Order) -> OrderResult:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        qty = float(order.quantity)
        # Alpaca requires DAY time-in-force for fractional quantities
        req = MarketOrderRequest(
            symbol=order.ticker, qty=round(qty, 6),
            side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        try:
            o = self.trading.submit_order(req)
        except Exception as e:
            return OrderResult(order, "", "rejected", message=str(e))
        filled = float(o.filled_qty or 0)
        price = float(o.filled_avg_price) if o.filled_avg_price else None
        return OrderResult(order, str(o.id), str(o.status.value if hasattr(o.status, "value") else o.status),
                           filled_quantity=filled, filled_price=price)
