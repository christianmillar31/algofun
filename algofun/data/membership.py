"""Point-in-time S&P 500 membership.

Source: fja05680/sp500 on GitHub (MIT), a daily membership history from
1996 built on Andreas Clenow's dataset and maintained from Wikipedia's
index-change announcements. It removes selection bias: a backtest can only
hold names that were in the index on that date. It does not remove
delisting-return bias, because free price sources rarely carry the final
prints of names that went to zero. Treat results as closer to the truth,
not as the truth.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .universe import to_yahoo_symbol

log = logging.getLogger(__name__)

MEMBERSHIP_URLS = [
    "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv",
    "https://raw.githubusercontent.com/fja05680/sp500/main/S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv",
]
MEMBERSHIP_FILE = "sp500_membership.csv"


def load_membership(cache_dir: str | os.PathLike = "data/cache/universe", refresh: bool = False) -> pd.DataFrame:
    """Return a frame with a DatetimeIndex of snapshot dates and one column
    `tickers` holding the list of Yahoo-style symbols in the index that day."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / MEMBERSHIP_FILE
    if not path.exists() or refresh:
        text = None
        for url in MEMBERSHIP_URLS:
            try:
                resp = requests.get(url, timeout=60)
                if resp.ok and resp.text.startswith("date,tickers"):
                    text = resp.text
                    break
            except requests.RequestException as e:
                log.warning("membership download failed from %s: %s", url, e)
        if text is None:
            raise RuntimeError("could not download S&P 500 membership history and no cached copy exists")
        path.write_text(text)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["tickers"] = df["tickers"].astype(str).apply(lambda s: sorted({to_yahoo_symbol(t) for t in s.split(",") if t.strip()}))
    return df.set_index("date").sort_index()


def all_members(membership: pd.DataFrame, start=None, end=None) -> list[str]:
    """Union of every ticker that was a member at any snapshot in [start, end]."""
    m = membership.loc[start:end] if (start is not None or end is not None) else membership
    out: set[str] = set()
    for lst in m["tickers"]:
        out.update(lst)
    return sorted(out)


def membership_mask(membership: pd.DataFrame, dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """date x ticker boolean frame: True when the ticker was in the index on that
    date, using the most recent snapshot at or before the date. Dates before
    the first snapshot are all False."""
    snaps = membership.index
    pos = snaps.searchsorted(dates, side="right") - 1
    cols = list(tickers)
    col_index = {t: i for i, t in enumerate(cols)}
    data = np.zeros((len(dates), len(cols)), dtype=bool)
    cache: dict[int, list[int]] = {}
    for row, p in enumerate(pos):
        if p < 0:
            continue
        if p not in cache:
            cache[p] = [col_index[t] for t in membership["tickers"].iloc[p] if t in col_index]
        data[row, cache[p]] = True
    return pd.DataFrame(data, index=dates, columns=cols)


def pit_universe(cache_dir: str | os.PathLike = "data/cache/universe", start=None, end=None) -> list[str]:
    return all_members(load_membership(cache_dir), start, end)
