import numpy as np
import pandas as pd
import pytest

from algofun.data.store import Panel


def make_panel(n_tickers=5, n_days=300, seed=0, drift=0.0005, vol=0.02, start="2020-01-01",
               tickers=None) -> Panel:
    """Random-walk OHLCV panel. open != close so fill-at-open is observable."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    tickers = tickers or [f"T{i}" for i in range(n_tickers)]
    n_tickers = len(tickers)
    rets = rng.normal(drift, vol, size=(n_days, n_tickers))
    close = 100.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = close * (1.0 + rng.normal(0, 0.003, size=close.shape))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.003, size=close.shape)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.003, size=close.shape)))
    volume = rng.integers(100_000, 1_000_000, size=close.shape).astype(float)
    bars = {}
    for j, t in enumerate(tickers):
        bars[t] = pd.DataFrame({"open": open_[:, j], "high": high[:, j], "low": low[:, j],
                                "close": close[:, j], "volume": volume[:, j]}, index=dates)
    return Panel.from_bars(bars)


@pytest.fixture
def panel() -> Panel:
    return make_panel()


@pytest.fixture
def big_panel() -> Panel:
    return make_panel(n_tickers=8, n_days=900, seed=1)
