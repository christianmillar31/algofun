from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.view import MarketView
from ..risk.sizing import equal_weight
from .base import Strategy


class MeanReversion(Strategy):
    """Short-term mean reversion on liquid names.

    Buy names whose close has dropped more than `z_entry` standard deviations
    below their `lookback`-day mean (a z-score), but only while the name is
    still in a longer-term uptrend (`trend_sma`), so we are buying dips and
    not falling knives. Exit when the z-score recovers past `z_exit`.
    Daily rebalance: this is a higher-turnover strategy and costs matter.
    """

    name = "mean_reversion"
    defaults = {"lookback": 20, "z_entry": -2.0, "z_exit": 0.0, "trend_sma": 100,
                "max_names": 10, "rebalance": "daily"}
    param_grid = {"lookback": [10, 20], "z_entry": [-1.5, -2.0, -2.5]}

    def __init__(self, **params):
        super().__init__(**params)
        self._held: set[str] = set()

    def reset(self) -> None:
        self._held = set()

    def configure(self) -> None:
        self.warmup = int(max(self.lookback, self.trend_sma)) + 1
        self.rebalance = self.params["rebalance"]

    def target_weights(self, view: MarketView) -> pd.Series:
        px = view.history("close", self.trend_sma)
        recent = px.iloc[-self.lookback:]
        mean, sd = recent.mean(), recent.std(ddof=1)
        last = px.iloc[-1]
        z = (last - mean) / sd.replace(0, np.nan)
        uptrend = last > px.mean()
        ok = view.tradable() & z.notna()
        entries = ok & uptrend & (z < self.z_entry)
        # keep current holdings until they mean-revert past z_exit (or lose the trend)
        stay = pd.Series({t: (t in self._held) for t in px.columns})
        stay = stay & ok & (z < self.z_exit)
        candidates = z[entries | stay].sort_values()  # most oversold first
        names = list(candidates.index[: self.max_names])
        self._held = set(names)
        if not names:
            return pd.Series(dtype="float64")
        return equal_weight(names)
