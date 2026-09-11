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

import os
from collections.abc import Iterable

import pandas as pd

from .base import Account, Broker, Order, OrderResult, Position


class AlpacaBroker(Broker):
    name = "alpaca"
    supports_fractional = True

    def __init__(self, api_key: str | None = None, secret_key: str | None = None,
                 paper: bool | None = None):
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
        self.name = "alpaca-paper" if paper else "alpaca-LIVE"
        self.trading = TradingClient(self.api_key, self.secret_key, paper=paper)
        self.data = StockHistoricalDataClient(self.api_key, self.secret_key)

    def account(self) -> Account:
        a = self.trading.get_account()
        return Account(cash=float(a.cash), equity=float(a.equity), buying_power=float(a.buying_power))

    def positions(self) -> dict[str, Position]:
        out = {}
        for p in self.trading.get_all_positions():
            out[p.symbol] = Position(p.symbol, float(p.qty), float(p.avg_entry_price), float(p.market_value))
        return out

    def latest_prices(self, tickers: Iterable[str]) -> pd.Series:
        from alpaca.data.requests import StockLatestTradeRequest
        tickers = list(tickers)
        if not tickers:
            return pd.Series(dtype="float64")
        trades = self.data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=tickers))
        return pd.Series({t: float(trades[t].price) for t in tickers if t in trades}, dtype="float64")

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
