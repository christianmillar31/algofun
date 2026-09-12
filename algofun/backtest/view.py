"""MarketView: the only thing a Strategy is allowed to see.

It wraps the full Panel but every accessor slices to bars at or before the
decision date. There is no method that returns future data, so a strategy
cannot peek at tomorrow without going out of its way to break the abstraction.
"""
from __future__ import annotations

import pandas as pd

from ..data.store import Panel


class MarketView:
    __slots__ = ("_i", "_panel")

    def __init__(self, panel: Panel, i: int):
        if i < 0 or i >= len(panel):
            raise IndexError(f"bar index {i} out of range for panel of length {len(panel)}")
        self._panel = panel
        self._i = i

    # ---- identity --------------------------------------------------------
    @property
    def date(self) -> pd.Timestamp:
        """The decision date. The strategy sees this bar's close, nothing later."""
        return self._panel.dates[self._i]

    @property
    def bar_index(self) -> int:
        return self._i

    @property
    def tickers(self) -> list[str]:
        return self._panel.tickers

    @property
    def sectors(self) -> dict[str, str]:
        """Ticker -> sector, when the panel was loaded with a sector map (else empty)."""
        return self._panel.sectors

    def sector_of(self, ticker: str) -> str:
        return self._panel.sector_of(ticker)

    def __len__(self) -> int:
        """Number of bars visible (including today)."""
        return self._i + 1

    # ---- history accessors ----------------------------------------------
    def history(self, field: str = "close", lookback: int | None = None) -> pd.DataFrame:
        """Bars up to and including today. lookback=N returns the last N rows.

        Prefer a bounded lookback: it is faster and it forces you to state how
        much history the strategy actually needs (its `warmup`).
        """
        df = self._panel.field(field)
        stop = self._i + 1
        start = 0 if lookback is None else max(0, stop - lookback)
        return df.iloc[start:stop]

    @property
    def close(self) -> pd.DataFrame:
        return self.history("close")

    @property
    def open(self) -> pd.DataFrame:
        return self.history("open")

    @property
    def high(self) -> pd.DataFrame:
        return self.history("high")

    @property
    def low(self) -> pd.DataFrame:
        return self.history("low")

    @property
    def volume(self) -> pd.DataFrame:
        return self.history("volume")

    def last(self, field: str = "close") -> pd.Series:
        """Today's value of a field for every ticker (NaN if it did not trade)."""
        return self._panel.field(field).iloc[self._i]

    def returns(self, lookback: int, field: str = "close") -> pd.DataFrame:
        """Daily simple returns over the last `lookback` bars."""
        px = self.history(field, lookback + 1)
        return px.pct_change().iloc[1:]

    # ---- external features (news sentiment, ...) --------------------------
    @property
    def features(self) -> list[str]:
        """Names of the date x ticker features attached to the panel."""
        return sorted(self._panel.features)

    def has_feature(self, name: str) -> bool:
        return name in self._panel.features

    def feature(self, field: str, lookback: int | None = None) -> pd.DataFrame:
        """A feature frame up to and including today, exactly like `history`.
        Features are aligned to bars, so a value on date d is what was known at
        d's close (see algofun.text.sentiment.session_dates for how news is mapped)."""
        df = self._panel.feature(field)
        stop = self._i + 1
        start = 0 if lookback is None else max(0, stop - lookback)
        return df.iloc[start:stop]

    def tradable(self) -> pd.Series:
        """Tickers with a valid close today and, when a point-in-time membership
        history is attached, in the index today."""
        return self._panel.eligible(self._i)
