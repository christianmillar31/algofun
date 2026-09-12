from __future__ import annotations

import pandas as pd

from ..backtest.view import MarketView
from .base import Strategy


class BuyAndHold(Strategy):
    """Equal-weight buy and hold of a fixed ticker list. The benchmark to beat."""

    name = "buy_and_hold"
    rebalance = "monthly"  # only rebalances back to equal weight monthly
    defaults = {"tickers": "SPY"}

    def configure(self) -> None:
        if isinstance(self.tickers, str):
            self.tickers = [t.strip().upper() for t in self.tickers.split(",") if t.strip()]

    def target_weights(self, view: MarketView) -> pd.Series:
        live = view.tradable()
        names = [t for t in self.tickers if t in live.index and live[t]]
        if not names:
            return pd.Series(dtype="float64")
        return pd.Series(1.0 / len(names), index=names)
