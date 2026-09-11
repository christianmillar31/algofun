from __future__ import annotations

import pandas as pd

from ..backtest.view import MarketView
from ..risk.sizing import equal_weight, inverse_vol
from .base import Strategy


class SMACrossover(Strategy):
    """Trend following: long every name whose fast SMA is above its slow SMA.

    Classic 50/200 "golden cross". Equal weight (or inverse-vol) across the
    names currently in an uptrend; cash otherwise. Weekly rebalance keeps
    turnover sane.
    """

    name = "sma_crossover"
    defaults = {"fast": 50, "slow": 200, "sizing": "equal", "max_names": 30, "rebalance": "weekly"}
    param_grid = {"fast": [20, 50], "slow": [100, 200]}

    def configure(self) -> None:
        if self.fast >= self.slow:
            raise ValueError("fast must be < slow")
        self.warmup = int(self.slow) + 1
        self.rebalance = self.params["rebalance"]

    def target_weights(self, view: MarketView) -> pd.Series:
        px = view.history("close", self.slow + 1)
        fast = px.rolling(self.fast, min_periods=self.fast).mean().iloc[-1]
        slow = px.rolling(self.slow, min_periods=self.slow).mean().iloc[-1]
        signal = (fast > slow) & view.tradable()
        # strength = how far above the slow line; used to rank when capping names
        strength = (fast / slow - 1.0)[signal]
        names = list(strength.sort_values(ascending=False).index[: self.max_names])
        if not names:
            return pd.Series(dtype="float64")
        if self.sizing == "inverse_vol":
            return inverse_vol(px[names].pct_change().iloc[-60:], names)
        return equal_weight(names)
