"""Local parquet cache of daily bars, one file per ticker, plus the Panel
structure the backtester consumes.

    store = BarStore("data/cache")
    store.update(["SPY", "AAPL"], start="2000-01-01")   # fetch/incremental
    panel = store.load_panel(["SPY", "AAPL"])            # aligned wide frames
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .sources import BAR_COLUMNS, BarSource, empty_bars, get_source, normalize_bars

log = logging.getLogger(__name__)


@dataclass
class Panel:
    """Wide, date-aligned OHLCV frames. Rows are dates, columns are tickers.

    A ticker that did not trade on a date holds NaN in every field. The
    engine treats NaN close as "not tradable today".
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def __len__(self) -> int:
        return len(self.close)

    def field(self, name: str) -> pd.DataFrame:
        return getattr(self, name)

    def slice(self, start=None, end=None) -> Panel:
        return Panel(**{f: getattr(self, f).loc[start:end] for f in BAR_COLUMNS})

    def select(self, tickers: Sequence[str]) -> Panel:
        cols = [t for t in tickers if t in self.close.columns]
        return Panel(**{f: getattr(self, f)[cols] for f in BAR_COLUMNS})

    def with_min_history(self, min_bars: int) -> Panel:
        """Drop tickers with fewer than min_bars non-NaN closes."""
        keep = self.close.notna().sum() >= min_bars
        return self.select(list(self.close.columns[keep]))

    def returns(self) -> pd.DataFrame:
        return self.close.pct_change(fill_method=None)

    @classmethod
    def from_bars(cls, bars: dict[str, pd.DataFrame], start=None, end=None) -> Panel:
        if not bars:
            raise ValueError("no bars supplied")
        frames = {f: {} for f in BAR_COLUMNS}
        for ticker, df in bars.items():
            df = normalize_bars(df).loc[start:end]
            for f in BAR_COLUMNS:
                frames[f][ticker] = df[f]
        wide = {f: pd.DataFrame(frames[f]).sort_index() for f in BAR_COLUMNS}
        idx = wide["close"].index
        idx.name = "date"
        wide = {f: d.reindex(idx) for f, d in wide.items()}
        return cls(**wide)


class BarStore:
    """Parquet cache: <root>/bars/<TICKER>.parquet."""

    def __init__(self, root: str | os.PathLike = "data/cache"):
        self.root = Path(root)
        self.bars_dir = self.root / "bars"
        self.bars_dir.mkdir(parents=True, exist_ok=True)

    # -- basic file ops ----------------------------------------------------
    def path(self, ticker: str) -> Path:
        return self.bars_dir / f"{ticker.upper()}.parquet"

    def has(self, ticker: str) -> bool:
        return self.path(ticker).exists()

    def tickers(self) -> list[str]:
        return sorted(p.stem for p in self.bars_dir.glob("*.parquet"))

    def load(self, ticker: str) -> pd.DataFrame:
        p = self.path(ticker)
        if not p.exists():
            return empty_bars()
        return normalize_bars(pd.read_parquet(p))

    def save(self, ticker: str, df: pd.DataFrame, merge: bool = True) -> pd.DataFrame:
        df = normalize_bars(df)
        if merge and self.has(ticker):
            old = self.load(ticker)
            df = pd.concat([old, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_parquet(self.path(ticker))
        return df

    def last_date(self, ticker: str) -> pd.Timestamp | None:
        df = self.load(ticker)
        return None if df.empty else df.index[-1]

    # -- fetching ----------------------------------------------------------
    def update(self, tickers: Iterable[str], start: str | None = "1990-01-01",
               end: str | None = None, source: BarSource | str = "auto",
               force: bool = False, overlap_days: int = 7) -> dict[str, int]:
        """Fetch new bars for each ticker and merge into the cache.

        Incremental: a cached ticker is re-fetched from (last_date - overlap_days)
        so late adjustments to the last few bars are picked up. Returns a map
        ticker -> number of rows now cached (0 means nothing could be fetched).
        """
        src = get_source(source) if isinstance(source, str) else source
        tickers = list(dict.fromkeys(t.upper() for t in tickers))
        full, incremental = [], {}
        for t in tickers:
            last = None if force else self.last_date(t)
            if last is None:
                full.append(t)
            else:
                incremental[t] = (last - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")

        counts: dict[str, int] = {}
        if full:
            log.info("full fetch of %d tickers from %s via %s", len(full), start, src.name)
            got = src.fetch_many(full, start, end)
            for t in full:
                df = got.get(t)
                if df is None or df.empty:
                    log.warning("no data for %s", t)
                    counts[t] = len(self.load(t))
                    continue
                counts[t] = len(self.save(t, df, merge=not force))

        # group incremental fetches by their start date to keep batch calls small
        by_start: dict[str, list[str]] = {}
        for t, s in incremental.items():
            by_start.setdefault(s, []).append(t)
        for s, ts in by_start.items():
            got = src.fetch_many(ts, s, end)
            for t in ts:
                df = got.get(t)
                if df is not None and not df.empty:
                    self.save(t, df, merge=True)
                counts[t] = len(self.load(t))
        return counts

    # -- panel -------------------------------------------------------------
    def load_panel(self, tickers: Sequence[str] | None = None, start=None, end=None,
                   min_bars: int = 0) -> Panel:
        tickers = list(tickers) if tickers else self.tickers()
        bars = {}
        for t in tickers:
            df = self.load(t)
            if not df.empty:
                bars[t.upper()] = df
        missing = sorted(set(t.upper() for t in tickers) - set(bars))
        if missing:
            log.warning("%d tickers not in cache (run `algofun fetch`): %s%s",
                        len(missing), ", ".join(missing[:10]), "..." if len(missing) > 10 else "")
        panel = Panel.from_bars(bars, start, end)
        if min_bars:
            panel = panel.with_min_history(min_bars)
        return panel


def import_long_csv(path: str | os.PathLike, store: BarStore, ticker_col: str = "Name",
                    date_col: str = "date", column_map: dict[str, str] | None = None,
                    merge: bool = True) -> dict[str, int]:
    """Bulk-import a long-format CSV (one row per ticker-day) into the store.

    Works out of the box with the plotly `all_stocks_5yr.csv` layout:
        date,open,high,low,close,volume,Name
    """
    df = pd.read_csv(path)
    if column_map:
        df = df.rename(columns=column_map)
    counts = {}
    for ticker, g in df.groupby(ticker_col):
        g = g.drop(columns=[ticker_col]).rename(columns={date_col: "date"})
        try:
            bars = normalize_bars(g)
        except ValueError as e:
            log.warning("skipping %s: %s", ticker, e)
            continue
        t = str(ticker).upper().replace(".", "-")
        counts[t] = len(store.save(t, bars, merge=merge))
    return counts
