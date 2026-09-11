from __future__ import annotations

import pandas as pd

from ..backtest.view import MarketView
from ..risk.sizing import equal_weight, inverse_vol, vol_target
from .base import Strategy


class Momentum(Strategy):
    """Cross-sectional momentum (Jegadeesh & Titman style).

    Rank every name by its return over the last `lookback` bars, skipping
    the most recent `skip` bars (short-term reversal), and hold the top N.
    Optionally only invest when the market filter (SPY above its own long SMA)
    is on, which is the usual crash protection for momentum.

    Defaults follow the evidence: inverse-vol sizing, a portfolio vol target
    (scale-down only) to blunt momentum crashes (Barroso & Santa-Clara), and
    at most `max_per_sector` names from any one sector so the basket is not
    a single-industry bet (Moskowitz & Grinblatt). Set vol_target=0 and
    max_per_sector=0 for the plain equal-weight version.
    """

    name = "momentum"
    defaults = {"lookback": 252, "skip": 21, "top_n": 10, "sizing": "inverse_vol", "vol_target": 0.15,
                "vol_lookback": 60, "max_per_sector": 3,
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
        names = self._select(score.sort_values(ascending=False).index, view)
        if not names:
            return pd.Series(dtype="float64")
        rets = px[names].pct_change().iloc[-int(self.vol_lookback):]
        w = inverse_vol(rets, names) if self.sizing == "inverse_vol" else equal_weight(names)
        if self.vol_target:
            w = vol_target(w, rets, float(self.vol_target), max_leverage=1.0)
        return w

    def _select(self, ranked, view: MarketView) -> list[str]:
        """Top N by score, taking at most max_per_sector from any known sector."""
        cap = int(self.max_per_sector or 0)
        if cap <= 0 or not view.sectors:
            return list(ranked[: self.top_n])
        names, counts = [], {}
        for t in ranked:
            s = view.sector_of(t)
            if s != "Unknown" and counts.get(s, 0) >= cap:
                continue
            names.append(t)
            counts[s] = counts.get(s, 0) + 1
            if len(names) >= self.top_n:
                break
        return names
