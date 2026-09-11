"""In-memory paper broker with optional JSON persistence.

Fills happen immediately at the price you give it (`set_prices`) through the
same CostModel the backtester uses, so a paper run and a backtest agree on
how expensive a trade is.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ..backtest.costs import CostModel
from .base import Account, Broker, Order, OrderResult, Position


class PaperBroker(Broker):
    name = "paper"

    def __init__(self, cash: float = 10_000.0, costs: CostModel | None = None,
                 state_file: str | Path | None = None):
        self.cash = float(cash)
        self.costs = costs or CostModel.retail()
        self._positions: dict[str, dict] = {}   # ticker -> {qty, avg_price}
        self._prices: pd.Series = pd.Series(dtype="float64")
        self.state_file = Path(state_file) if state_file else None
        self.fills: list[OrderResult] = []
        if self.state_file and self.state_file.exists():
            self._load()

    # ---- prices ---------------------------------------------------------
    def set_prices(self, prices: pd.Series | dict) -> None:
        self._prices = pd.Series(prices, dtype="float64")

    def latest_prices(self, tickers: Iterable[str]) -> pd.Series:
        return self._prices.reindex(list(tickers))

    # ---- state ----------------------------------------------------------
    def account(self) -> Account:
        mv = sum(p["qty"] * float(self._prices.get(t, p["avg_price"])) for t, p in self._positions.items())
        eq = self.cash + mv
        return Account(cash=self.cash, equity=eq, buying_power=self.cash)

    def positions(self) -> dict[str, Position]:
        out = {}
        for t, p in self._positions.items():
            px = float(self._prices.get(t, p["avg_price"]))
            out[t] = Position(t, p["qty"], p["avg_price"], p["qty"] * px)
        return out

    # ---- orders ---------------------------------------------------------
    def submit(self, order: Order) -> OrderResult:
        ref = self._prices.get(order.ticker)
        if ref is None or pd.isna(ref):
            return OrderResult(order, str(uuid.uuid4()), "rejected", message="no price")
        side = 1 if order.side == "buy" else -1
        price = self.costs.fill_price(float(ref), side)
        qty = float(order.quantity)
        if not self.supports_fractional:
            qty = float(int(qty))
        if qty <= 0:
            return OrderResult(order, str(uuid.uuid4()), "rejected", message="zero quantity")
        comm = self.costs.commission(qty, price)
        pos = self._positions.get(order.ticker, {"qty": 0.0, "avg_price": 0.0})
        if side == 1:
            cost = qty * price + comm
            if cost > self.cash + 1e-9:
                qty = max(0.0, (self.cash - comm) / price)
                if not self.supports_fractional:
                    qty = float(int(qty))
                if qty <= 0:
                    return OrderResult(order, str(uuid.uuid4()), "rejected", message="insufficient cash")
                cost = qty * price + comm
            new_qty = pos["qty"] + qty
            pos["avg_price"] = (pos["qty"] * pos["avg_price"] + qty * price) / new_qty
            pos["qty"] = new_qty
            self.cash -= cost
        else:
            if qty > pos["qty"] + 1e-9:
                qty = pos["qty"]
            if qty <= 0:
                return OrderResult(order, str(uuid.uuid4()), "rejected", message="nothing to sell")
            pos["qty"] -= qty
            self.cash += qty * price - comm
        if pos["qty"] <= 1e-9:
            self._positions.pop(order.ticker, None)
        else:
            self._positions[order.ticker] = pos
        res = OrderResult(order, str(uuid.uuid4()), "filled", filled_quantity=qty, filled_price=price)
        self.fills.append(res)
        self._save()
        return res

    # ---- persistence ----------------------------------------------------
    def _save(self) -> None:
        if not self.state_file:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps({"cash": self.cash, "positions": self._positions}, indent=2))

    def _load(self) -> None:
        d = json.loads(self.state_file.read_text())
        self.cash = float(d["cash"])
        self._positions = {k: {"qty": float(v["qty"]), "avg_price": float(v["avg_price"])}
                           for k, v in d["positions"].items()}
