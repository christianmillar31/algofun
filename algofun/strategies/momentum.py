from __future__ import annotations

import pandas as pd

from ..backtest.view import MarketView
from ..risk.sizing import equal_weight, inverse_vol
from .base import Strategy


class Momentum(Strategy):
    """Cross-sectional momentum (Jegadeesh & Titman style).

    Rank every name by its return over the last `lookback` bars, skipping
    the most recent `skip` bars (short-term reversal), and hold the top N.
    Optionally only invest when the market filter (SPY above its own long SMA)
    is on, which is the usual crash protection for momentum.
    """

    name = "momentum"
    defaults = {"lookback": 252, "skip": 21, "top_n": 10, "sizing": "equal",
                "market_filter": "SPY", "filter_sma": 200, "rebalance": "monthly"}
    param_grid = {"lookback": [126, 252], "top_n": [10, 20]}

    def configure(self) -> None:
        self.warmup = int(max(self.lookback, self.filter_sma)) + 1
        self.rebalance = self.params["rebalance"]

    def target_weights(self, view: MarketView) -> pd.Series:
        px = view.history("close", self.lookback + 1)
        if self.market_filter and self.market_filter in px.columns:
            spy = view.history("close", self.filter_sma)[self.market_filter]
            if spy.notna().sum() >= self.filter_sma and spy.iloc[-1] < spy.mean():
                return pd.Series(dtype="float64")  # risk-off: all cash
        end = px.iloc[-1 - self.skip] if self.skip > 0 else px.iloc[-1]
        start = px.iloc[0]
        score = (end / start - 1.0)
        score = score[view.tradable() & score.notna()]
        if self.market_filter in score.index:
            score = score.drop(self.market_filter)
        names = list(score.sort_values(ascending=False).index[: self.top_n])
        if not names:
            return pd.Series(dtype="float64")
        if self.sizing == "inverse_vol":
            return inverse_vol(px[names].pct_change().iloc[-60:], names)
        return equal_weight(names)
