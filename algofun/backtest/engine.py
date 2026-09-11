"""Event-driven daily backtester.

Timeline for one bar i:
    1. Fill orders that were decided at the close of bar i-1, at bar i's OPEN
       (after slippage/spread/commission).
    2. Mark the book to market at bar i's CLOSE and record equity.
    3. If bar i is a rebalance day, hand the strategy a MarketView cut off at
       bar i and store its target weights as pending orders for bar i+1.

Deciding on today's close and filling at tomorrow's open is the simplest
schedule that cannot leak the future: the strategy never sees the price it
gets filled at.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..data.store import Panel
from ..risk.limits import RiskLimits
from ..strategies.base import Strategy, clean_weights
from .costs import CostModel
from .metrics import compute_metrics, format_metrics
from .schedule import rebalance_mask
from .view import MarketView

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100_000.0
    costs: CostModel = field(default_factory=CostModel.retail)
    limits: RiskLimits = field(default_factory=RiskLimits)
    allow_fractional: bool = True   # Alpaca allows fractional market orders
    min_trade_notional: float = 1.0  # skip rebalancing dust smaller than this ($)
    # no-trade band: skip a rebalance trade smaller than this fraction of equity
    # unless it closes the position. Cuts churn from weight drift.
    min_trade_weight: float = 0.002
    rebalance: str | int | None = None  # override the strategy's schedule
    benchmark: str | None = "SPY"     # ticker in the panel to compare against


@dataclass
class BacktestResult:
    strategy: str
    params: dict[str, Any]
    config: BacktestConfig
    equity: pd.Series          # end-of-day equity (cash + positions at close)
    cash: pd.Series
    positions: pd.DataFrame    # shares held at close, date x ticker
    weights: pd.DataFrame      # market-value weights at close
    trades: pd.DataFrame       # one row per fill
    turnover: pd.Series        # traded notional / equity, per day
    benchmark: pd.Series | None = None  # benchmark equity rebased to initial cash

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)

    def metrics(self) -> dict[str, float]:
        return compute_metrics(self.equity, trades=self.trades, turnover=self.turnover,
                               benchmark=self.benchmark)

    def summary(self) -> str:
        head = f"{self.strategy} {self.params}  [{self.equity.index[0].date()} -> {self.equity.index[-1].date()}]"
        return head + "\n" + format_metrics(self.metrics())

    def save(self, out_dir) -> None:
        import json
        from pathlib import Path
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.equity.rename("equity").to_frame().join(self.cash.rename("cash")).to_csv(d / "equity.csv")
        self.positions.to_csv(d / "positions.csv")
        self.weights.to_csv(d / "weights.csv")
        self.trades.to_csv(d / "trades.csv", index=False)
        if self.benchmark is not None:
            self.benchmark.rename("benchmark").to_csv(d / "benchmark.csv")
        (d / "metrics.json").write_text(json.dumps(
            {"strategy": self.strategy, "params": self.params, "metrics": self.metrics()},
            indent=2, default=str))


def _locate(dates: pd.DatetimeIndex, when, default: int) -> int:
    if when is None:
        return default
    ts = pd.Timestamp(when)
    return int(dates.searchsorted(ts))


def run_backtest(panel: Panel, strategy: Strategy, config: BacktestConfig | None = None,
                 start=None, end=None) -> BacktestResult:
    """Run `strategy` over `panel`.

    `start`/`end` bound the trading window. History before `start` is still
    visible to the strategy (that is what makes walk-forward test folds valid),
    but no trades or equity are recorded before it.
    """
    cfg = config or BacktestConfig()
    dates = panel.dates
    tickers = panel.tickers
    n_t = len(tickers)
    if len(dates) == 0 or n_t == 0:
        raise ValueError("empty panel")

    open_px = panel.open.to_numpy(dtype="float64")
    # valuation price: last known close (a name that stops printing keeps its last mark)
    mark_px = panel.close.ffill().to_numpy(dtype="float64")

    first = max(_locate(dates, start, 0), int(strategy.warmup))
    last = _locate(dates, end, len(dates) - 1) if end is not None else len(dates) - 1
    last = min(last, len(dates) - 1)
    if end is not None and last < len(dates) and dates[last] > pd.Timestamp(end):
        last -= 1
    if first > last:
        raise ValueError(f"no bars to trade: warmup={strategy.warmup}, start={start}, end={end}, "
                         f"panel {dates[0].date()}..{dates[-1].date()} ({len(dates)} bars)")

    schedule = cfg.rebalance if cfg.rebalance is not None else strategy.rebalance
    rb_mask = rebalance_mask(dates, schedule)
    strategy.reset()

    cash = float(cfg.initial_cash)
    shares = np.zeros(n_t)
    pending: np.ndarray | None = None  # target weights decided at previous close

    n_rec = last - first + 1
    eq_rec = np.empty(n_rec)
    cash_rec = np.empty(n_rec)
    pos_rec = np.empty((n_rec, n_t))
    to_rec = np.zeros(n_rec)
    trades: list[dict[str, Any]] = []
    costs = cfg.costs

    for k, i in enumerate(range(first, last + 1)):
        date = dates[i]
        traded_notional = 0.0

        # ---- 1. fill pending orders at today's open ------------------------
        if pending is not None:
            px_open = open_px[i]
            px_fill_ref = np.where(np.isnan(px_open), mark_px[i - 1] if i > 0 else np.nan, px_open)
            equity_open = cash + float(np.nansum(shares * px_fill_ref))
            target_shares = np.zeros(n_t)
            ok = ~np.isnan(px_fill_ref) & (px_fill_ref > 0)
            target_shares[ok] = pending[ok] * equity_open / px_fill_ref[ok]
            if not cfg.allow_fractional:
                target_shares = np.trunc(target_shares)
            # names without a price today: hold what we have
            target_shares[~ok] = shares[~ok]
            delta = target_shares - shares
            # sells first so cash is available for buys
            order = np.argsort(delta)
            for j in order:
                d = delta[j]
                if d == 0 or not ok[j]:
                    continue
                notional_ref = abs(d) * px_fill_ref[j]
                closing = target_shares[j] == 0 and shares[j] != 0
                if notional_ref < cfg.min_trade_notional:
                    continue
                if not closing and notional_ref < cfg.min_trade_weight * equity_open:
                    continue
                side = 1 if d > 0 else -1
                price = costs.fill_price(px_fill_ref[j], side)
                if side == 1 and not cfg.limits.allow_short:
                    # cap buys at available cash (after commission)
                    max_qty = cash / (price + costs.commission_per_share + price * costs.commission_pct)
                    if max_qty <= 0:
                        continue
                    d = min(d, max_qty if cfg.allow_fractional else math.floor(max_qty))
                    if d <= 0:
                        continue
                comm = costs.commission(d, price)
                cash -= side * abs(d) * price + comm
                shares[j] += d
                traded_notional += abs(d) * price
                trades.append({"date": date, "ticker": tickers[j], "side": "buy" if side == 1 else "sell",
                               "quantity": abs(d), "price": price, "commission": comm,
                               "notional": abs(d) * price})
            shares[np.abs(shares) < 1e-12] = 0.0
            pending = None

        # ---- 2. mark to market at the close --------------------------------
        equity = cash + float(np.nansum(shares * mark_px[i]))
        eq_rec[k] = equity
        cash_rec[k] = cash
        pos_rec[k] = shares
        to_rec[k] = traded_notional / equity if equity > 0 else 0.0

        # ---- 3. decide at the close -----------------------------------------
        if rb_mask[i] and i < last:
            view = MarketView(panel, i)
            raw = strategy.target_weights(view)
            w = clean_weights(raw, tickers)
            w = cfg.limits.apply(w, sectors=panel.sectors)
            pending = w.to_numpy(dtype="float64")

    idx = dates[first:last + 1]
    equity_s = pd.Series(eq_rec, index=idx, name="equity")
    positions = pd.DataFrame(pos_rec, index=idx, columns=tickers)
    mv = positions * mark_px[first:last + 1]
    weights = mv.div(equity_s, axis=0).fillna(0.0)
    trades_df = pd.DataFrame(trades, columns=["date", "ticker", "side", "quantity", "price",
                                              "commission", "notional"])
    bench = None
    if cfg.benchmark and cfg.benchmark in panel.close.columns:
        b = panel.close[cfg.benchmark].ffill().iloc[first:last + 1]
        if b.notna().all() and b.iloc[0] > 0:
            bench = (b / b.iloc[0] * cfg.initial_cash).rename(cfg.benchmark)

    return BacktestResult(
        strategy=strategy.name, params=dict(strategy.params), config=cfg,
        equity=equity_s, cash=pd.Series(cash_rec, index=idx, name="cash"),
        positions=positions, weights=weights, trades=trades_df,
        turnover=pd.Series(to_rec, index=idx, name="turnover"), benchmark=bench,
    )
