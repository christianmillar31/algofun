"""Turn a strategy's target weights into broker orders.

Run this once per day after the close. Market DAY orders submitted after
hours queue for the next open, which is exactly the "decide at close, fill
at next open" assumption the backtester makes.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..backtest.view import MarketView
from ..broker.base import Broker, Order, OrderResult
from ..data.store import BarStore
from ..risk.limits import RiskLimits
from ..strategies.base import Strategy, clean_weights

log = logging.getLogger(__name__)


@dataclass
class RebalancePlan:
    as_of: pd.Timestamp
    equity: float
    cash: float
    target_weights: pd.Series
    current_weights: pd.Series
    orders: list[Order]
    prices: pd.Series
    results: list[OrderResult] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for o in self.orders:
            px = float(self.prices.get(o.ticker, float("nan")))
            rows.append({"ticker": o.ticker, "side": o.side, "quantity": round(o.quantity, 4),
                         "est_price": px, "est_notional": round(o.quantity * px, 2),
                         "current_w": round(float(self.current_weights.get(o.ticker, 0.0)), 4),
                         "target_w": round(float(self.target_weights.get(o.ticker, 0.0)), 4)})
        return pd.DataFrame(rows, columns=["ticker", "side", "quantity", "est_price", "est_notional",
                                           "current_w", "target_w"])

    def describe(self) -> str:
        head = (f"as of {self.as_of.date()}  equity={self.equity:,.2f}  cash={self.cash:,.2f}  "
                f"targets={int((self.target_weights != 0).sum())} names  orders={len(self.orders)}")
        if not self.orders:
            return head + "\n  (no trades needed)"
        return head + "\n" + self.to_frame().to_string(index=False)


def compute_orders(target_weights: pd.Series, current_qty: pd.Series, prices: pd.Series, equity: float,
                   min_trade_notional: float = 1.0, min_trade_weight: float = 0.002,
                   allow_fractional: bool = True) -> list[Order]:
    """Diff target weights against current holdings and emit market orders.

    Same no-trade band as the backtester: skip trades smaller than
    min_trade_weight of equity unless they close a position.
    """
    tickers = sorted(set(target_weights.index) | set(current_qty.index))
    tw = target_weights.reindex(tickers).fillna(0.0)
    cq = current_qty.reindex(tickers).fillna(0.0)
    px = prices.reindex(tickers)
    orders: list[Order] = []
    for t in tickers:
        p = px[t]
        if pd.isna(p) or p <= 0:
            if tw[t] != 0 or cq[t] != 0:
                log.warning("no price for %s; skipping", t)
            continue
        target_qty = tw[t] * equity / p
        if not allow_fractional:
            target_qty = float(int(target_qty))
        delta = target_qty - cq[t]
        notional = abs(delta) * p
        closing = tw[t] == 0 and cq[t] != 0
        if notional < min_trade_notional:
            continue
        if not closing and notional < min_trade_weight * equity:
            continue
        qty = abs(delta) if allow_fractional else float(int(abs(delta)))
        if qty <= 0:
            continue
        orders.append(Order(t, "buy" if delta > 0 else "sell", qty))
    return orders


def plan_rebalance(strategy: Strategy, store: BarStore, broker: Broker, universe: Sequence[str],
                   limits: RiskLimits | None = None, min_trade_notional: float = 1.0,
                   min_trade_weight: float = 0.002, min_bars: int | None = None) -> RebalancePlan:
    limits = limits or RiskLimits()
    panel = store.load_panel(list(universe), min_bars=min_bars or 0)
    if len(panel) <= strategy.warmup:
        raise ValueError(f"need > {strategy.warmup} bars of history, have {len(panel)}; run `algofun fetch`")
    view = MarketView(panel, len(panel) - 1)
    weights = limits.apply(clean_weights(strategy.target_weights(view), panel.tickers))
    weights = weights[weights != 0]

    acct = broker.account()
    positions = broker.positions()
    current_qty = pd.Series({t: p.quantity for t, p in positions.items()}, dtype="float64")
    need_px = sorted(set(weights.index) | set(current_qty.index))
    prices = broker.latest_prices(need_px) if need_px else pd.Series(dtype="float64")
    # fall back to last close from our own data when the broker has no quote
    last_close = panel.close.iloc[-1]
    prices = prices.combine_first(last_close.reindex(need_px))
    current_w = (current_qty * prices.reindex(current_qty.index)) / acct.equity if acct.equity else current_qty * 0
    orders = compute_orders(weights, current_qty, prices, acct.equity, min_trade_notional,
                            min_trade_weight, broker.supports_fractional)
    return RebalancePlan(as_of=view.date, equity=acct.equity, cash=acct.cash, target_weights=weights,
                         current_weights=current_w.fillna(0.0), orders=orders, prices=prices)


def execute_plan(plan: RebalancePlan, broker: Broker, log_path: str | Path | None = "runs/rebalance_log.jsonl"
                 ) -> list[OrderResult]:
    results = broker.submit_all(plan.orders)
    plan.results = results
    if log_path:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            for r in results:
                f.write(json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(), "broker": broker.name,
                    "as_of": str(plan.as_of.date()), "ticker": r.order.ticker, "side": r.order.side,
                    "quantity": r.order.quantity, "status": r.status, "filled_quantity": r.filled_quantity,
                    "filled_price": r.filled_price, "order_id": r.order_id, "message": r.message,
                }) + "\n")
    return results
