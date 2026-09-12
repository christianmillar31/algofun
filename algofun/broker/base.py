"""Broker abstraction shared by the paper broker and real broker adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd


@dataclass
class Account:
    cash: float
    equity: float
    buying_power: float
    last_equity: float | None = None   # equity at the prior close, when the broker provides it


@dataclass
class Position:
    ticker: str
    quantity: float
    avg_price: float
    market_value: float


@dataclass
class Order:
    ticker: str
    side: str            # 'buy' | 'sell'
    quantity: float      # shares (fractional allowed if broker supports it)
    kind: str = "market"
    time_in_force: str = "day"
    #: idempotency key: brokers that support it (Alpaca) reject a duplicate, so a
    #: retried submit can never double-fill. Stamped by plan_rebalance.
    client_order_id: str | None = None

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side == "buy" else -self.quantity


@dataclass
class OrderResult:
    order: Order
    order_id: str
    status: str                       # 'filled' | 'accepted' | 'rejected' | 'dry_run'
    filled_quantity: float = 0.0
    filled_price: float | None = None
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""


class Broker(ABC):
    name: str = "base"
    supports_fractional: bool = True

    @abstractmethod
    def account(self) -> Account: ...

    @abstractmethod
    def positions(self) -> dict[str, Position]: ...

    @abstractmethod
    def latest_prices(self, tickers: Iterable[str]) -> pd.Series: ...

    @abstractmethod
    def submit(self, order: Order) -> OrderResult: ...

    def submit_all(self, orders: Iterable[Order]) -> list[OrderResult]:
        """Sells first, then buys, so cash is available."""
        orders = sorted(orders, key=lambda o: 0 if o.side == "sell" else 1)
        return [self.submit(o) for o in orders]

    def is_market_open(self) -> bool:
        return True

    def cancel_open_orders(self) -> int:
        return 0

    def refresh(self, results: list[OrderResult]) -> list[OrderResult]:
        """Update statuses/fills of recently submitted orders (brokers that fill
        asynchronously override this). Default: nothing to do."""
        return results

    def equity_history(self) -> pd.Series | None:
        """Daily account equity history if the broker can provide one (date -> equity)."""
        return None

    def flatten(self) -> list[OrderResult]:
        """Close every position with market orders. The emergency exit."""
        self.cancel_open_orders()
        orders = [Order(t, "sell" if p.quantity > 0 else "buy", abs(p.quantity))
                  for t, p in self.positions().items() if p.quantity != 0]
        return [self.submit(o) for o in orders]
