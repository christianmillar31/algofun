"""Ticker universes.

Note on survivorship bias: any "current S&P 500" list is today's survivors.
Backtesting only on those names flatters results because companies that
went bankrupt or were dropped never appear. Treat results on such a
universe as an upper bound, not an expectation.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import requests

SP500_CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)

# Broad, liquid ETFs. Useful both as a small universe and as benchmarks.
ETFS: list[str] = [
    "SPY",  # S&P 500
    "QQQ",  # Nasdaq 100
    "IWM",  # Russell 2000
    "DIA",  # Dow 30
    "VTI",  # Total US market
    "EFA",  # Developed ex-US
    "EEM",  # Emerging markets
    "TLT",  # 20y+ Treasuries
    "IEF",  # 7-10y Treasuries
    "LQD",  # IG corporate bonds
    "HYG",  # High yield
    "GLD",  # Gold
    "SLV",  # Silver
    "USO",  # Oil
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC",
]

# A small hand-picked set of large caps for quick experiments when you
# don't want to wait on a 500-ticker download.
MEGA_CAPS: list[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "JPM", "V", "UNH",
    "XOM", "JNJ", "PG", "MA", "HD", "COST", "ABBV", "WMT", "KO", "PEP",
]


def to_yahoo_symbol(symbol: str) -> str:
    """Yahoo uses '-' where exchanges use '.' (BRK.B -> BRK-B)."""
    return symbol.strip().upper().replace(".", "-")


def load_sp500_constituents(cache_dir: str | os.PathLike = "data/cache/universe",
                            refresh: bool = False) -> pd.DataFrame:
    """Download (or load cached) the current S&P 500 constituents table."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "sp500_constituents.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    resp = requests.get(SP500_CONSTITUENTS_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.to_csv(path, index=False)
    return df


def sector_map(cache_dir: str | os.PathLike = "data/cache/universe", refresh: bool = False,
               extra: dict[str, str] | None = None) -> dict[str, str]:
    """Ticker -> GICS sector from the constituents table; ETFs map to 'ETF'.

    Returns {} (with a warning) if the table cannot be loaded, so callers
    degrade to "no sector caps" rather than failing.
    """
    out: dict[str, str] = {t: "ETF" for t in ETFS}
    try:
        df = load_sp500_constituents(cache_dir, refresh=refresh)
        col = "GICS Sector" if "GICS Sector" in df.columns else None
        if col:
            out.update({to_yahoo_symbol(s): str(sec) for s, sec in zip(df["Symbol"].astype(str), df[col], strict=True)})
    except Exception as e:  # noqa: BLE001 - network or parse failure
        import logging
        logging.getLogger(__name__).warning("sector map unavailable (%s); sector caps will not apply", e)
    if extra:
        out.update(extra)
    return out


def sp500_tickers(**kwargs) -> list[str]:
    df = load_sp500_constituents(**kwargs)
    return [to_yahoo_symbol(s) for s in df["Symbol"].astype(str)]


def resolve_universe(spec: str, **kwargs) -> list[str]:
    """Turn a universe spec into a ticker list.

    Accepts: 'sp500', 'etfs', 'megacaps', combinations joined by '+',
    a path to a text/CSV file with one ticker per line (or a 'Symbol' column),
    or a comma-separated list of tickers.
    """
    spec = spec.strip()
    if os.path.exists(spec):
        text = Path(spec).read_text()
        if text.lower().startswith("symbol"):
            df = pd.read_csv(spec)
            return sorted({to_yahoo_symbol(s) for s in df["Symbol"].astype(str)})
        return sorted({to_yahoo_symbol(t) for t in text.split() if t.strip()})

    out: set[str] = set()
    for part in spec.split("+"):
        key = part.strip().lower()
        if key == "sp500":
            out.update(sp500_tickers(**kwargs))
        elif key == "etfs":
            out.update(ETFS)
        elif key == "megacaps":
            out.update(MEGA_CAPS)
        elif key:
            out.update(to_yahoo_symbol(t) for t in part.split(",") if t.strip())
    return sorted(out)
