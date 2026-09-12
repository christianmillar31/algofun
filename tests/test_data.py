import numpy as np
import pandas as pd
import pytest

from algofun.data import (
    BarStore,
    Panel,
    import_long_csv,
    normalize_bars,
    resolve_universe,
    to_yahoo_symbol,
)
from algofun.data.sources import ChainedSource, empty_bars


def _bars(n=5, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": np.arange(n) + 1.0,
                         "Volume": 100}, index=idx)


def test_normalize_bars_variants():
    df = normalize_bars(_bars())
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "date"
    tz = _bars()
    tz.index = tz.index.tz_localize("UTC")
    assert normalize_bars(tz).index.tz is None
    long = _bars().reset_index().rename(columns={"index": "date", "Close": "Adj Close"})
    assert "close" in normalize_bars(long).columns
    dup = pd.concat([_bars(), _bars()])
    assert len(normalize_bars(dup)) == 5
    with pytest.raises(ValueError):
        normalize_bars(pd.DataFrame({"close": [1.0]}, index=pd.bdate_range("2020-01-01", periods=1)))
    assert normalize_bars(None).empty and empty_bars().empty


def test_store_save_merge_and_panel(tmp_path):
    store = BarStore(tmp_path)
    store.save("aapl", _bars(5))
    assert store.tickers() == ["AAPL"] and store.has("AAPL")
    # overlapping merge keeps the newer values and de-duplicates
    newer = _bars(5, start="2020-01-06") * 10
    merged = store.save("AAPL", newer)
    assert len(merged) == 8
    assert merged["close"].iloc[-1] == 50.0
    assert store.last_date("AAPL") == merged.index[-1]
    store.save("MSFT", _bars(3, start="2020-01-03"))
    panel = store.load_panel(["AAPL", "MSFT", "NOPE"])
    assert panel.tickers == ["AAPL", "MSFT"]
    assert len(panel) == 8
    assert panel.close["MSFT"].isna().sum() == 5
    assert panel.with_min_history(4).tickers == ["AAPL"]
    assert panel.slice("2020-01-06", "2020-01-08").close.shape[0] == 3
    assert panel.select(["MSFT"]).tickers == ["MSFT"]
    assert store.load("missing").empty


def test_import_long_csv(tmp_path):
    rows = []
    for t in ("aapl", "brk.b"):
        for i, d in enumerate(pd.bdate_range("2021-01-01", periods=4)):
            rows.append({"date": d.date(), "open": 1, "high": 2, "low": 0.5, "close": i + 1, "volume": 9,
                         "Name": t})
    path = tmp_path / "long.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    store = BarStore(tmp_path / "cache")
    counts = import_long_csv(path, store)
    assert counts == {"AAPL": 4, "BRK-B": 4}
    assert store.load("BRK-B")["close"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_resolve_universe_and_symbols(tmp_path):
    assert to_yahoo_symbol("brk.b") == "BRK-B"
    assert resolve_universe("aapl, msft,AAPL") == ["AAPL", "MSFT"]
    assert "SPY" in resolve_universe("etfs") and "AAPL" in resolve_universe("megacaps+etfs")
    f = tmp_path / "u.txt"
    f.write_text("nvda\ntsla\n")
    assert resolve_universe(str(f)) == ["NVDA", "TSLA"]
    c = tmp_path / "u.csv"
    c.write_text("Symbol,Security\nBF.B,Brown-Forman\n")
    assert resolve_universe(str(c)) == ["BF-B"]


def test_chained_source_falls_through():
    class A:
        name = "a"
        def fetch_many(self, tickers, start=None, end=None):
            return {"X": _bars(3)} if "X" in tickers else {}

    class B:
        name = "b"
        def fetch_many(self, tickers, start=None, end=None):
            return {t: _bars(2) for t in tickers}

    class Boom:
        name = "boom"
        def fetch_many(self, tickers, start=None, end=None):
            raise RuntimeError("down")

    got = ChainedSource(Boom(), A(), B()).fetch_many(["X", "Y"])
    assert len(got["X"]) == 3 and len(got["Y"]) == 2


def test_store_update_incremental(tmp_path):
    calls = []

    class Src:
        name = "fake"
        def fetch_many(self, tickers, start=None, end=None):
            calls.append((tuple(tickers), start))
            return {t: _bars(10, start="2020-01-01") for t in tickers}

    store = BarStore(tmp_path)
    counts = store.update(["A", "B"], start="2020-01-01", source=Src())
    assert counts == {"A": 10, "B": 10}
    assert calls[0] == (("A", "B"), "2020-01-01")
    store.update(["A"], source=Src())
    assert calls[1][0] == ("A",) and calls[1][1] > "2020-01-01"   # incremental start after cache
    assert len(store.load("A")) == 10


def test_panel_from_bars_rejects_empty():
    with pytest.raises(ValueError):
        Panel.from_bars({})
