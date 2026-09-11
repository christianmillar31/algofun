"""Bar data sources. Each returns a standardized daily OHLCV frame:

    index: DatetimeIndex named 'date' (tz-naive, sorted, unique)
    columns: open, high, low, close, volume  (floats; prices split/dividend adjusted where the source allows)

Primary source is Yahoo via yfinance (free, adjusted). Stooq is a fallback
that is split-adjusted but not dividend-adjusted, so total-return numbers
from Stooq data will be a little low for dividend payers.
"""
from __future__ import annotations

import io
import logging
from collections.abc import Iterable
from datetime import date, datetime
from typing import Protocol

import pandas as pd
import requests

log = logging.getLogger(__name__)

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary OHLCV frame into the canonical layout."""
    if df is None or len(df) == 0:
        return empty_bars()
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    if "adj_close" in out.columns and "close" not in out.columns:
        out = out.rename(columns={"adj_close": "close"})
    if "date" in out.columns:
        out = out.set_index("date")
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    out.index.name = "date"
    missing = [c for c in BAR_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"bars missing columns {missing}; have {list(out.columns)}")
    out = out[BAR_COLUMNS].astype("float64")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["close"])
    return out


def empty_bars() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], name="date")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLUMNS}, index=idx)


def _as_str(d: str | date | datetime | None) -> str | None:
    if d is None:
        return None
    if isinstance(d, (date, datetime)):
        return d.strftime("%Y-%m-%d")
    return str(d)


class BarSource(Protocol):
    name: str

    def fetch_many(self, tickers: Iterable[str], start: str | None, end: str | None
                   ) -> dict[str, pd.DataFrame]: ...


class YFinanceSource:
    """Yahoo Finance via yfinance. Adjusted prices (auto_adjust=True)."""

    name = "yfinance"

    def __init__(self, batch_size: int = 100, threads: bool = True):
        self.batch_size = batch_size
        self.threads = threads

    def fetch_many(self, tickers, start=None, end=None) -> dict[str, pd.DataFrame]:
        import yfinance as yf  # imported lazily so the package is optional at import time

        tickers = list(dict.fromkeys(t.upper() for t in tickers))
        out: dict[str, pd.DataFrame] = {}
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i + self.batch_size]
            raw = yf.download(
                batch, start=_as_str(start), end=_as_str(end), auto_adjust=True,
                progress=False, group_by="ticker", threads=self.threads,
            )
            if raw is None or len(raw) == 0:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                for t in batch:
                    if t not in raw.columns.get_level_values(0):
                        continue
                    try:
                        out[t] = normalize_bars(raw[t].dropna(how="all"))
                    except ValueError as e:  # pragma: no cover - depends on Yahoo payload
                        log.warning("skipping %s: %s", t, e)
            else:  # single ticker returns flat columns
                out[batch[0]] = normalize_bars(raw.dropna(how="all"))
        return out


class StooqSource:
    """Stooq daily CSV endpoint. Split-adjusted, not dividend-adjusted."""

    name = "stooq"
    URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"

    def __init__(self, suffix: str = ".us", session: requests.Session | None = None):
        self.suffix = suffix
        self.session = session or requests.Session()

    def fetch_one(self, ticker: str, start=None, end=None) -> pd.DataFrame:
        symbol = ticker.lower().replace("-", ".") + self.suffix
        resp = self.session.get(self.URL.format(symbol=symbol), timeout=30)
        resp.raise_for_status()
        if "No data" in resp.text[:100]:
            return empty_bars()
        df = pd.read_csv(io.StringIO(resp.text))
        df = normalize_bars(df)
        if start is not None:
            df = df.loc[pd.Timestamp(start):]
        if end is not None:
            df = df.loc[:pd.Timestamp(end)]
        return df

    def fetch_many(self, tickers, start=None, end=None) -> dict[str, pd.DataFrame]:
        out = {}
        for t in tickers:
            try:
                df = self.fetch_one(t, start, end)
            except Exception as e:  # network or parse error: skip, don't abort the run
                log.warning("stooq failed for %s: %s", t, e)
                continue
            if len(df):
                out[t.upper()] = df
        return out


class AlpacaBarsSource:
    """Daily bars from Alpaca's market-data API (split and dividend adjusted).

    Free plans may query the consolidated (SIP) feed as long as the window
    ends at least 15 minutes ago, so `end` is clamped to now - lag_minutes.
    History is shallow compared with Yahoo (roughly 2016 onward), which is
    why the automatic chain uses this source for incremental updates only.
    Needs ALPACA_API_KEY / ALPACA_SECRET_KEY or an injected client.
    """

    name = "alpaca"

    def __init__(self, api_key: str | None = None, secret_key: str | None = None, feed: str = "sip",
                 batch_size: int = 100, lag_minutes: int = 16, client=None):
        import os
        self.feed = feed
        self.batch_size = batch_size
        self.lag_minutes = lag_minutes
        if client is not None:
            self.client = client
            return
        key = api_key or os.environ.get("ALPACA_API_KEY")
        secret = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("AlpacaBarsSource needs ALPACA_API_KEY and ALPACA_SECRET_KEY")
        from alpaca.data.historical import StockHistoricalDataClient
        self.client = StockHistoricalDataClient(key, secret)

    def fetch_many(self, tickers, start=None, end=None) -> dict[str, pd.DataFrame]:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        tickers = list(dict.fromkeys(t.upper() for t in tickers))
        now = pd.Timestamp.now(tz="UTC")
        end_ts = min(pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) if end else now, now) - pd.Timedelta(minutes=self.lag_minutes)
        start_ts = pd.Timestamp(start or "2015-01-01", tz="UTC")
        out: dict[str, pd.DataFrame] = {}
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i + self.batch_size]
            req = StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day, start=start_ts.to_pydatetime(),
                                   end=end_ts.to_pydatetime(), adjustment=Adjustment.ALL, feed=DataFeed(self.feed))
            try:
                bars = self.client.get_stock_bars(req)
            except Exception as e:  # noqa: BLE001 - network/API error: let the chain fall through
                log.warning("alpaca bars failed for %d symbols: %s", len(batch), e)
                continue
            df = getattr(bars, "df", None)
            if df is None or len(df) == 0:
                continue
            if isinstance(df.index, pd.MultiIndex):
                for sym, g in df.groupby(level=0):
                    g = g.droplevel(0)
                    out[str(sym).upper()] = normalize_bars(g[["open", "high", "low", "close", "volume"]])
            elif len(batch) == 1:
                out[batch[0]] = normalize_bars(df[["open", "high", "low", "close", "volume"]])
        return out


class ChainedSource:
    """Try sources in order; a ticker missing from one falls through to the next."""

    name = "chained"

    def __init__(self, *sources: BarSource):
        self.sources = list(sources)
        self.name = "+".join(s.name for s in self.sources)

    def fetch_many(self, tickers, start=None, end=None) -> dict[str, pd.DataFrame]:
        remaining = list(dict.fromkeys(t.upper() for t in tickers))
        out: dict[str, pd.DataFrame] = {}
        for src in self.sources:
            if not remaining:
                break
            try:
                got = src.fetch_many(remaining, start, end)
            except Exception as e:
                log.warning("source %s failed: %s", src.name, e)
                got = {}
            out.update({k: v for k, v in got.items() if len(v)})
            remaining = [t for t in remaining if t not in out]
        return out


def get_source(name: str = "auto") -> BarSource:
    name = name.lower()
    if name == "yfinance":
        return YFinanceSource()
    if name == "stooq":
        return StooqSource()
    if name == "alpaca":
        return AlpacaBarsSource()
    if name == "auto":
        return ChainedSource(YFinanceSource(), StooqSource())
    if name == "auto-incremental":
        # official broker data first for the recent bars that drive live decisions
        try:
            return ChainedSource(AlpacaBarsSource(), YFinanceSource(), StooqSource())
        except (RuntimeError, ImportError):
            return ChainedSource(YFinanceSource(), StooqSource())
    raise ValueError(f"unknown source {name!r}; choose auto, yfinance, stooq, or alpaca")
