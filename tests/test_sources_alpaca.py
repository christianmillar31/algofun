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


def test_store_warns_on_vendor_disagreement(tmp_path, caplog):
    store = BarStore(tmp_path)
    idx = pd.bdate_range("2026-09-01", periods=5)
    store.save("A", pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": [10.0] * 5, "volume": 1}, index=idx))
    fresh = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": [10.0, 10.0, 10.0, 10.0, 11.0], "volume": 1}, index=idx)
    with caplog.at_level(logging.WARNING):
        worst = store._warn_if_disagree("A", fresh)
    assert worst == pytest.approx(0.10) and "differ from cached" in caplog.text


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
