import pandas as pd
import pytest

from algofun.backtest import BacktestConfig, MarketView, run_backtest
from algofun.data import all_members, load_membership, membership_mask
from algofun.strategies.base import Strategy

from .conftest import make_panel


def _write_membership(tmp_path):
    (tmp_path / "sp500_membership.csv").write_text(
        'date,tickers\n2020-01-01,"A,B,BF.B"\n2020-03-01,"A,C"\n2020-06-01,"C,D"\n')
    return tmp_path


def test_load_and_union(tmp_path):
    m = load_membership(_write_membership(tmp_path))
    assert list(m.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-01")]
    assert m["tickers"].iloc[0] == ["A", "B", "BF-B"]
    assert all_members(m) == ["A", "B", "BF-B", "C", "D"]
    assert all_members(m, start="2020-04-01") == ["C", "D"]


def test_mask_uses_latest_snapshot_at_or_before_date(tmp_path):
    m = load_membership(_write_membership(tmp_path))
    dates = pd.DatetimeIndex(["2019-12-31", "2020-01-01", "2020-02-15", "2020-03-01", "2020-05-01", "2020-07-01"])
    mask = membership_mask(m, dates, ["A", "B", "C", "D", "Z"])
    assert mask.loc["2019-12-31"].sum() == 0                      # before history: nothing eligible
    assert mask.loc["2020-01-01", ["A", "B"]].all() and not mask.loc["2020-01-01", "C"]
    assert mask.loc["2020-02-15", "B"] and not mask.loc["2020-02-15", "C"]
    assert mask.loc["2020-03-01", "C"] and not mask.loc["2020-03-01", "B"]
    assert mask.loc["2020-07-01", ["C", "D"]].all() and not mask.loc["2020-07-01", "A"]
    assert not mask["Z"].any()


def test_engine_never_holds_non_members(tmp_path):
    p = make_panel(n_tickers=3, n_days=120, tickers=["A", "B", "C"], start="2020-01-02")
    snaps = pd.DataFrame({"tickers": [["A", "B"], ["B", "C"]]},
                         index=pd.DatetimeIndex(["2019-12-01", p.dates[60]]))
    p = p.with_membership(snaps)
    assert p.membership.shape == (120, 3)
    v0, v100 = MarketView(p, 0), MarketView(p, 100)
    assert v0.tradable().to_dict() == {"A": True, "B": True, "C": False}
    assert v100.tradable().to_dict() == {"A": False, "B": True, "C": True}

    class WantsAll(Strategy):
        name = "wants_all"
        rebalance = "daily"

        def target_weights(self, view):
            return pd.Series({"A": 0.3, "B": 0.3, "C": 0.3})

    res = run_backtest(p, WantsAll(), BacktestConfig(benchmark=None))
    assert (res.positions["C"].iloc[:60] == 0).all() and (res.positions["C"].iloc[62:] > 0).all()
    assert (res.positions["A"].iloc[62:] == 0).all()               # sold once it left the index
    assert p.slice(p.dates[10], p.dates[20]).membership.shape == (11, 3)
    assert p.select(["A"]).membership.columns.tolist() == ["A"]


def test_membership_download_failure_is_clear(tmp_path, monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("no net")))
    with pytest.raises(RuntimeError):
        load_membership(tmp_path)
