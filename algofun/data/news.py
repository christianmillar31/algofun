"""News articles from Alpaca's market-data API (Benzinga feed), cached by month.

    store = NewsStore("data/cache")
    store.update("2023-01-01")                 # fetch/incremental, resumable
    df = store.load("2023-01-01", "2023-12-31")

Every article carries `created_at` in UTC. Nothing here decides which trading
session an article belongs to; see algofun.text.sentiment.session_dates for
the lookahead-safe mapping.

The Alpaca history goes back to 2015 and the endpoint is free on the basic
data plan. It pages 50 articles at a time and the account is limited to 200
requests a minute, so the first fetch of a few years takes hours. The cache
is written one month at a time and finished months are never re-fetched, so
a fetch can be interrupted (or given a time budget) and resumed later.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .sources import from_alpaca_symbol, to_alpaca_symbol

log = logging.getLogger(__name__)

NEWS_COLUMNS = ["id", "created_at", "updated_at", "headline", "summary", "source", "url", "symbols"]
PAGE_SIZE = 50


def empty_news() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in NEWS_COLUMNS})
    df["id"] = df["id"].astype("int64")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)
    return df


def normalize_news(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to NEWS_COLUMNS: int id, UTC timestamps, our '-' share-class symbols,
    one row per article id (the latest update wins)."""
    if df is None or len(df) == 0:
        return empty_news()
    df = df.copy()
    for c in NEWS_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[NEWS_COLUMNS]
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("int64")
    for c in ("created_at", "updated_at"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    df["updated_at"] = df["updated_at"].fillna(df["created_at"])
    df["symbols"] = df["symbols"].apply(lambda s: [from_alpaca_symbol(str(x)) for x in (s if isinstance(s, (list, tuple)) else (list(s) if hasattr(s, "__iter__") and not isinstance(s, str) else []))])
    for c in ("headline", "summary", "source", "url"):
        df[c] = df[c].fillna("").astype(str)
    df = df.dropna(subset=["created_at"])
    df = df.sort_values(["updated_at", "id"]).drop_duplicates("id", keep="last")
    return df.sort_values(["created_at", "id"]).reset_index(drop=True)


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


class AlpacaNewsSource:
    """Paged fetch through Alpaca's /v1beta1/news.

    `client` is anything with `get(path, data) -> dict` returning {"news": [...],
    "next_page_token": ...}; the real alpaca-py NewsClient by default.
    `min_interval` throttles requests to stay under the account's limit.
    """

    name = "alpaca-news"

    def __init__(self, client=None, min_interval: float = 0.35, include_content: bool = False):
        self._client = client
        self.min_interval = float(min_interval)
        self.include_content = include_content
        self.requests_made = 0

    @property
    def client(self):
        if self._client is None:
            from alpaca.data.historical.news import NewsClient
            key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
            if not key or not secret:
                raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY are not set")
            self._client = NewsClient(api_key=key, secret_key=secret)
        return self._client

    def fetch(self, start, end, symbols: Sequence[str] | None = None,
              deadline: float | None = None) -> tuple[pd.DataFrame, bool]:
        """Articles created in [start, end]. Returns (frame, complete); complete is
        False when `deadline` (time.monotonic()) passed before the last page."""
        params: dict = {"start": _utc(start).isoformat().replace("+00:00", "Z"),
                        "end": _utc(end).isoformat().replace("+00:00", "Z"),
                        "sort": "asc", "limit": PAGE_SIZE,
                        "include_content": self.include_content, "exclude_contentless": False}
        if symbols:
            params["symbols"] = ",".join(to_alpaca_symbol(s) for s in symbols)
        rows: list[dict] = []
        token = None
        last = 0.0
        while True:
            if deadline is not None and time.monotonic() > deadline:
                return normalize_news(pd.DataFrame(rows)), False
            wait = self.min_interval - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            if token:
                params["page_token"] = token
            last = time.monotonic()
            resp = self.client.get("/news", data=dict(params))
            self.requests_made += 1
            for a in resp.get("news", []) or []:
                rows.append({c: a.get(c) for c in NEWS_COLUMNS})
            token = resp.get("next_page_token")
            if not token:
                break
        return normalize_news(pd.DataFrame(rows)), True


def month_starts(start, end) -> list[pd.Timestamp]:
    s, e = _utc(start).normalize(), _utc(end).normalize()
    first = pd.Timestamp(year=s.year, month=s.month, day=1, tz="UTC")
    out = []
    cur = first
    while cur <= e:
        out.append(cur)
        cur = (cur + pd.offsets.MonthBegin(1)).normalize()
    return out


class NewsStore:
    """Parquet cache: <root>/news/<YYYY-MM>.parquet, one file per calendar month (UTC)."""

    def __init__(self, root: str | os.PathLike = "data/cache"):
        self.root = Path(root)
        self.news_dir = self.root / "news"
        self.news_dir.mkdir(parents=True, exist_ok=True)

    def path(self, month: pd.Timestamp) -> Path:
        return self.news_dir / f"{month:%Y-%m}.parquet"

    def months(self) -> list[pd.Timestamp]:
        return sorted(pd.Timestamp(p.stem + "-01", tz="UTC") for p in self.news_dir.glob("????-??.parquet"))

    def load_month(self, month: pd.Timestamp) -> pd.DataFrame:
        p = self.path(month)
        return normalize_news(pd.read_parquet(p)) if p.exists() else empty_news()

    def save_month(self, month: pd.Timestamp, df: pd.DataFrame, merge: bool = True) -> pd.DataFrame:
        df = normalize_news(df)
        if merge:
            df = normalize_news(pd.concat([self.load_month(month), df]))
        df.to_parquet(self.path(month))
        return df

    def load(self, start=None, end=None, symbols: Iterable[str] | None = None) -> pd.DataFrame:
        months = self.months()
        if start is not None:
            s0 = _utc(start).normalize()
            months = [m for m in months if m >= pd.Timestamp(year=s0.year, month=s0.month, day=1, tz="UTC")]
        if end is not None:
            months = [m for m in months if m <= _utc(end)]
        frames = [self.load_month(m) for m in months]
        df = normalize_news(pd.concat(frames)) if frames else empty_news()
        if start is not None:
            df = df[df["created_at"] >= _utc(start)]
        if end is not None:
            df = df[df["created_at"] <= _utc(end)]
        if symbols is not None:
            want = {s.upper() for s in symbols}
            df = df[df["symbols"].apply(lambda ss: any(s in want for s in ss))]
        return df.reset_index(drop=True)

    def coverage(self) -> pd.DataFrame:
        rows = []
        for m in self.months():
            df = self.load_month(m)
            rows.append({"month": f"{m:%Y-%m}", "articles": len(df),
                         "last_created": df["created_at"].max() if len(df) else pd.NaT})
        return pd.DataFrame(rows)

    def update(self, start, end=None, source: AlpacaNewsSource | None = None,
               symbols: Sequence[str] | None = None, max_minutes: float | None = None,
               settle_hours: float = 24.0) -> dict[str, int]:
        """Fetch month by month from `start` to `end` (default now).

        A month whose file exists and whose end is more than `settle_hours` in
        the past is complete and skipped. The current (or a partial) month is
        re-fetched from its last cached article minus a day, then merged.
        `max_minutes` stops cleanly after the budget; whatever finished is saved
        and the next call resumes there. Returns {month: articles cached}.
        """
        src = source or AlpacaNewsSource()
        now = datetime.now(timezone.utc)
        end_ts = _utc(end) if end is not None else pd.Timestamp(now)
        end_ts = min(end_ts, pd.Timestamp(now))
        deadline = time.monotonic() + max_minutes * 60 if max_minutes is not None else None
        counts: dict[str, int] = {}
        for m in month_starts(start, end_ts):
            key = f"{m:%Y-%m}"
            m_end = min((m + pd.offsets.MonthBegin(1)).normalize() - pd.Timedelta(seconds=1), end_ts)
            settled = m_end < pd.Timestamp(now) - timedelta(hours=settle_hours)
            existing = self.load_month(m)
            if self.path(m).exists() and settled:
                counts[key] = len(existing)
                continue
            fetch_from = max(_utc(start), m)
            if len(existing):
                fetch_from = max(fetch_from, existing["created_at"].max() - pd.Timedelta(days=1))
            if deadline is not None and time.monotonic() > deadline:
                log.warning("news fetch stopped by time budget before %s; rerun to continue", key)
                break
            log.info("fetching news %s..%s", fetch_from.date(), m_end.date())
            df, complete = src.fetch(fetch_from, m_end, symbols=symbols, deadline=deadline)
            if len(df):
                existing = self.save_month(m, df, merge=True)
            counts[key] = len(existing)
            if not complete:
                log.warning("news fetch for %s is partial (time budget); rerun to continue", key)
                break
        return counts
