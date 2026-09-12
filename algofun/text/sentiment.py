"""Turn timestamped news into date x ticker features a strategy can read
through MarketView without any chance of peeking.

The one rule that matters: an article belongs to the first trading session
whose close it precedes. A story published at 15:59 New York time on Tuesday
is known at Tuesday's close and may drive Tuesday's decision (filled at
Wednesday's open). A story at 16:01 belongs to Wednesday. A story on
Saturday belongs to Monday. Anything else is lookahead.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from ..live.calendar import is_trading_day, next_trading_day
from .lexicon import Lexicon, score_text

log = logging.getLogger(__name__)

NY_TZ = "America/New_York"
NEWS_FEATURES = ["news_tone_sum", "news_neg_sum", "news_count"]


def session_dates(created_at: pd.Series, close_hour: int = 16) -> pd.Series:
    """Map UTC article timestamps to the (tz-naive) NYSE session date at whose close
    the article is first known."""
    ts = pd.to_datetime(created_at, utc=True)
    local = ts.dt.tz_convert(NY_TZ)
    day = local.dt.normalize().dt.tz_localize(None)
    secs = local.dt.hour * 3600 + local.dt.minute * 60 + local.dt.second
    after_close = secs > close_hour * 3600
    day = day + pd.to_timedelta(after_close.astype(int), unit="D")
    mapping = {d: (d if is_trading_day(d) else next_trading_day(d)) for d in day.dropna().unique()}
    return day.map(mapping)


def score_articles(news: pd.DataFrame, lexicon: Lexicon) -> pd.DataFrame:
    """One row per article: session date, symbols, tone, negativity, counts."""
    if len(news) == 0:
        return pd.DataFrame(columns=["id", "created_at", "session", "symbols", "tone", "negativity",
                                     "positive", "negative", "n_tokens"])
    text = news["headline"].fillna("") + ". " + news["summary"].fillna("")
    scores = [score_text(t, lexicon) for t in text]
    out = pd.DataFrame({
        "id": news["id"].to_numpy(),
        "created_at": news["created_at"].to_numpy(),
        "session": session_dates(news["created_at"]).to_numpy(),
        "symbols": news["symbols"].to_numpy(),
        "tone": [s.tone for s in scores],
        "negativity": [s.negativity for s in scores],
        "positive": [s.positive for s in scores],
        "negative": [s.negative for s in scores],
        "n_tokens": [s.n_tokens for s in scores],
    })
    return out


def daily_sentiment(scored: pd.DataFrame, dates: pd.DatetimeIndex, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Aggregate scored articles into date x ticker frames aligned to the panel.

    news_tone_sum  sum of per-article tone           (divide by count for the mean)
    news_neg_sum   sum of per-article negativity
    news_count     articles tagged with the ticker that session

    Days and names without news hold 0, so rolling sums are well defined; the
    count says whether there was anything to read.
    """
    idx = pd.DatetimeIndex(dates)
    zeros = pd.DataFrame(0.0, index=idx, columns=list(tickers))
    if len(scored) == 0:
        return {f: zeros.copy() for f in NEWS_FEATURES}
    long = scored[["session", "symbols", "tone", "negativity"]].explode("symbols").dropna(subset=["symbols"])
    long = long[long["symbols"].isin(set(tickers))]
    if len(long) == 0:
        return {f: zeros.copy() for f in NEWS_FEATURES}
    g = long.groupby(["session", "symbols"])
    agg = pd.DataFrame({"news_tone_sum": g["tone"].sum(), "news_neg_sum": g["negativity"].sum(),
                        "news_count": g["tone"].size().astype(float)}).reset_index()
    out = {}
    for f in NEWS_FEATURES:
        wide = agg.pivot(index="session", columns="symbols", values=f)
        wide.index = pd.DatetimeIndex(wide.index)
        out[f] = wide.reindex(index=idx, columns=list(tickers)).fillna(0.0)
    return out


def build_daily_features(store, lexicon: Lexicon, dates: pd.DatetimeIndex, tickers: list[str],
                         start=None, end=None) -> tuple[dict[str, pd.DataFrame], dict]:
    """Score every cached month once (cached beside the raw month) and aggregate.

    `store` is a NewsStore. Returns (features, info) where info reports how much
    news there was and how many ticker-days it touched.
    """
    idx = pd.DatetimeIndex(dates)
    lo = pd.Timestamp(start) if start is not None else idx[0]
    hi = pd.Timestamp(end) if end is not None else idx[-1]
    frames = []
    for m in store.months():
        m_naive = m.tz_localize(None)
        if m_naive > hi or (m_naive + pd.offsets.MonthEnd(1)) < lo - pd.Timedelta(days=5):
            continue
        frames.append(_scored_month(store, m, lexicon))
    scored = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    feats = daily_sentiment(scored, idx, tickers)
    count = feats["news_count"]
    if len(scored):
        first, last = scored["session"].min(), scored["session"].max()
        window = count.loc[max(first, idx[0]):min(last, idx[-1])]   # only the span the news covers
    else:
        first = last = None
        window = count.iloc[0:0]
    info = {
        "articles": int(len(scored)),
        "articles_tagged": int(scored["symbols"].apply(len).sum()) if len(scored) else 0,
        "ticker_days_with_news": float((window > 0).to_numpy().mean()) if window.size else 0.0,
        "first": str(first.date()) if first is not None else None,
        "last": str(last.date()) if last is not None else None,
    }
    return feats, info


def _scored_month(store, month: pd.Timestamp, lexicon: Lexicon) -> pd.DataFrame:
    raw = store.path(month)
    cached = store.news_dir / f"scored_{lexicon.name}_{month:%Y-%m}.parquet"
    if cached.exists() and cached.stat().st_mtime >= raw.stat().st_mtime:
        df = pd.read_parquet(cached)
        df["session"] = pd.to_datetime(df["session"])
        return df
    df = score_articles(store.load_month(month), lexicon)
    try:
        df.to_parquet(cached)
    except (OSError, ValueError) as e:  # cache is an optimisation, never a requirement
        log.debug("could not cache scores for %s: %s", month, e)
    return df


def lexicon_path(cache_dir: str | os.PathLike) -> Path:
    return Path(cache_dir) / "text"
