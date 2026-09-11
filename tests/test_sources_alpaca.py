import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from algofun.backtest import BacktestConfig, CostModel, run_backtest
from algofun.data import AlpacaBarsSource, BarStore, ChainedSource, get_source
from algofun.strategies.base import Strategy

from .conftest import make_panel

pytest.importorskip("alpaca")


class FakeDataClient:
    def __init__(self, fail=False):
        self.fail = fail
        self.requests = []

    def get_stock_bars(self, req):
        self.requests.append(req)
        if self.fail:
            raise RuntimeError("504")
        ts = pd.date_range("2026-09-01", periods=3, freq="B", tz="UTC") + pd.Timedelta(hours=4)
        rows = []
        for sym in req.symbol_or_symbols:
            for i, t in enumerate(ts):
                rows.append({"symbol": sym, "timestamp": t, "open": 10 + i, "high": 11 + i, "low": 9 + i,
                             "close": 10.5 + i, "volume": 1000, "trade_count": 5, "vwap": 10.4})
        df = pd.DataFrame(rows).set_index(["symbol", "timestamp"])
        return SimpleNamespace(df=df)


def test_alpaca_bars_source_parses_multiindex_and_clamps_end():
    client = FakeDataClient()
    src = AlpacaBarsSource(client=client, lag_minutes=16)
    got = src.fetch_many(["aapl", "MSFT"], start="2026-08-25", end="2026-09-11")
    assert set(got) == {"AAPL", "MSFT"}
    df = got["AAPL"]
    assert list(df.columns) == ["open", "high", "low", "close", "volume"] and df.index.tz is None
    assert df.index[0] == pd.Timestamp("2026-09-01") and df["close"].iloc[-1] == 12.5
    req = client.requests[0]
    end = pd.Timestamp(req.end)
    end = end.tz_convert(None) if end.tzinfo is not None else end
    assert end <= pd.Timestamp.now(tz="UTC").tz_convert(None) - pd.Timedelta(minutes=15)
    assert str(req.adjustment.value if hasattr(req.adjustment, "value") else req.adjustment) == "all"


def test_alpaca_failure_falls_through_chain():
    bad = AlpacaBarsSource(client=FakeDataClient(fail=True))

    class Backup:
        name = "backup"
        def fetch_many(self, tickers, start=None, end=None):
            idx = pd.bdate_range("2026-09-01", periods=3)
            return {t: pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx) for t in tickers}

    got = ChainedSource(bad, Backup()).fetch_many(["A"])
    assert "A" in got and got["A"]["close"].iloc[0] == 1.0


def test_auto_incremental_without_keys_is_yahoo_chain(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    src = get_source("auto-incremental")
    assert src.name == "yfinance+stooq"
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    assert get_source("auto-incremental").name.startswith("alpaca+")


def test_alpaca_symbol_mapping():
    from algofun.data.sources import from_alpaca_symbol, to_alpaca_symbol
    assert to_alpaca_symbol("BF-B") == "BF.B" and to_alpaca_symbol("brk-b") == "BRK.B" and to_alpaca_symbol("AAPL") == "AAPL"
    assert from_alpaca_symbol("BF.B") == "BF-B"
    client = FakeDataClient()
    AlpacaBarsSource(client=client).fetch_many(["BF-B", "AAPL"], start="2026-09-01")
    assert client.requests[0].symbol_or_symbols == ["BF.B", "AAPL"]


def test_partial_bar_refresh_is_not_a_disagreement_but_old_drift_is(tmp_path, caplog):
    store = BarStore(tmp_path)
    idx = pd.bdate_range("2026-09-01", periods=5)
    store.save("A", pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": [10.0] * 5, "volume": 1}, index=idx))
    # only the newest bar changed (yesterday's partial close replaced by the real one): fine
    fresh = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": [10.0, 10.0, 10.0, 10.0, 11.0], "volume": 1}, index=idx)
    assert store._history_disagrees("A", fresh) is False
    # an older bar moved 3%: history was re-adjusted or the vendor differs
    drift = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": [10.3, 10.3, 10.3, 10.0, 10.0], "volume": 1}, index=idx)
    with caplog.at_level(logging.WARNING):
        assert store._history_disagrees("A", drift) is True
    assert "settled closes differ" in caplog.text


def test_update_refetches_full_history_when_older_bars_drift(tmp_path):
    idx_old = pd.bdate_range("2026-08-01", periods=30)

    class Base:  # long-history vendor
        name = "base"
        calls = []
        def fetch_many(self, tickers, start=None, end=None):
            Base.calls.append((tuple(tickers), start))
            return {t: pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 20.0, "volume": 1}, index=idx_old) for t in tickers}

    class Inc:   # incremental vendor whose adjusted history disagrees on settled days
        name = "inc"
        def fetch_many(self, tickers, start=None, end=None):
            idx = idx_old[-8:]
            return {t: pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": [21.0] * 6 + [20.0, 20.0], "volume": 1}, index=idx) for t in tickers}

    store = BarStore(tmp_path)
    store.save("A", pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 20.0, "volume": 1}, index=idx_old[:-3]))
    import algofun.data.store as mod
    orig = mod.get_source
    mod.get_source = lambda name="auto": Inc() if name == "auto-incremental" else Base()
    try:
        counts = store.update(["A"], start="2026-08-01", source="auto")
    finally:
        mod.get_source = orig
    assert counts == {"A": 30}                                # full series re-downloaded from the base source
    assert (store.load("A")["close"] == 20.0).all()           # and it is internally consistent again
    assert Base.calls and Base.calls[-1][0] == ("A",)


def test_volume_cap_limits_fills_and_cost_scaling():
    p = make_panel(n_tickers=1, n_days=30, tickers=["A"])
    p.volume.iloc[:, 0] = 100.0   # tiny volume: 5% cap = 5 shares per bar

    class AllIn(Strategy):
        name = "all_in"
        rebalance = "daily"

        def target_weights(self, view):
            return pd.Series({"A": 1.0})

    from algofun.risk import RiskLimits
    cfg = BacktestConfig(benchmark=None, initial_cash=100_000.0, limits=RiskLimits(max_weight=1.0), costs=CostModel.zero())
    capped = run_backtest(p, AllIn(), cfg)
    assert capped.trades["quantity"].max() <= 5.0 + 1e-9
    uncapped = run_backtest(p, AllIn(), BacktestConfig(benchmark=None, initial_cash=100_000.0,
                                                       limits=RiskLimits(max_weight=1.0), costs=CostModel.zero(),
                                                       max_volume_share=None))
    assert uncapped.trades["quantity"].max() > 5.0
    c = CostModel.retail().scaled(2.0)
    assert c.slippage_bps == 10.0 and c.spread_bps == 4.0 and CostModel.zero().scaled(3).price_penalty == 0.0
