from __future__ import annotations

import logging

import pandas as pd

from ..backtest.view import MarketView
from ..risk.sizing import equal_weight, inverse_vol, vol_target
from .base import Strategy, pick_top

log = logging.getLogger(__name__)


class Sentiment(Strategy):
    """News-tone ranking on the Loughran-McDonald finance lexicon.

    Each name is scored by the article-weighted mean tone of the news tagged
    with it over the last `lookback` sessions (score="tone"), or by minus its
    mean negativity (score="neg", the part Loughran & McDonald found actually
    predicts anything). Names with fewer than `min_articles` in the window are
    not ranked: no news is not good news, it is no information. The top N are
    held with the same inverse-vol sizing, vol target, sector cap and index
    filter as momentum. Weekly by default because news tone decays in days.

    Needs the panel to carry news features (`algofun backtest --news`); without
    them it stays in cash and says so once.
    """

    name = "sentiment"
    defaults = {"lookback": 10, "min_articles": 3, "top_n": 20, "score": "tone", "sizing": "inverse_vol",
                "vol_target": 0.15, "vol_lookback": 60, "max_per_sector": 4,
                "market_filter": "SPY", "filter_sma": 200, "rebalance": "weekly"}
    param_grid = {"lookback": [5, 10, 21], "top_n": [10, 20]}

    def configure(self) -> None:
        need = [self.lookback, self.vol_lookback] + ([self.filter_sma] if self.market_filter else [])
        self.warmup = int(max(need)) + 1
        self.rebalance = self.params["rebalance"]
        self._warned = False

    def reset(self) -> None:
        self._warned = False

    def scores(self, view: MarketView) -> pd.DataFrame:
        """Per-name score and article count over the window (unfiltered, for reports)."""
        lb = int(self.lookback)
        count = view.feature("news_count", lb).sum()
        if self.score == "neg":
            s = -(view.feature("news_neg_sum", lb).sum() / count.replace(0.0, float("nan")))
        else:
            s = view.feature("news_tone_sum", lb).sum() / count.replace(0.0, float("nan"))
        return pd.DataFrame({"score": s, "articles": count})

    def target_weights(self, view: MarketView) -> pd.Series:
        if not (view.has_feature("news_count") and view.has_feature("news_tone_sum")):
            if not self._warned:
                log.warning("sentiment strategy: panel has no news features (use --news); staying in cash")
                self._warned = True
            return pd.Series(dtype="float64")
        px = view.history("close", int(max(self.vol_lookback, self.filter_sma)) + 1)
        if self.market_filter and self.market_filter in px.columns:
            spy = px[self.market_filter].iloc[-int(self.filter_sma):]
            if spy.notna().sum() >= self.filter_sma and spy.iloc[-1] < spy.mean():
                return pd.Series(dtype="float64")
        sc = self.scores(view)
        ok = (sc["articles"] >= int(self.min_articles)) & sc["score"].notna() & view.tradable().reindex(sc.index).fillna(False)
        score = sc.loc[ok, "score"]
        if self.market_filter in score.index:
            score = score.drop(self.market_filter)
        names = pick_top(score.sort_values(ascending=False).index, view, int(self.top_n), int(self.max_per_sector or 0))
        if not names:
            return pd.Series(dtype="float64")
        rets = px[names].pct_change().iloc[-int(self.vol_lookback):]
        w = inverse_vol(rets, names) if self.sizing == "inverse_vol" else equal_weight(names)
        if self.vol_target:
            w = vol_target(w, rets, float(self.vol_target), max_leverage=1.0)
        return w
