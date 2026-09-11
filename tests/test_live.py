import pandas as pd
import pytest

from algofun.backtest import CostModel
from algofun.broker import Order, PaperBroker
from algofun.data import BarStore
from algofun.live import compute_orders, execute_plan, plan_rebalance
from algofun.strategies import Momentum

from .conftest import make_panel


def test_compute_orders_diff_and_band():
    target = pd.Series({"A": 0.5, "B": 0.5})
    current = pd.Series({"A": 40.0, "C": 3.0})
    prices = pd.Series({"A": 10.0, "B": 20.0, "C": 50.0})
    orders = compute_orders(target, current, prices, equity=1000.0)
    by = {o.ticker: o for o in orders}
    assert by["A"].side == "buy" and by["A"].quantity == pytest.approx(10.0)   # 50 target - 40 held
    assert by["B"].side == "buy" and by["B"].quantity == pytest.approx(25.0)
    assert by["C"].side == "sell" and by["C"].quantity == pytest.approx(3.0)   # closing ignores band
    # tiny drift is ignored
    assert compute_orders(pd.Series({"A": 0.5}), pd.Series({"A": 49.9}), prices, 1000.0) == []
    whole = compute_orders(pd.Series({"B": 0.5}), pd.Series(dtype=float), prices, 1010.0, allow_fractional=False)
    assert whole[0].quantity == 25.0


def test_paper_broker_fills_and_rejects():
    b = PaperBroker(cash=1000.0, costs=CostModel.zero())
    b.set_prices({"A": 10.0})
    r = b.submit(Order("A", "buy", 50))
    assert r.status == "filled" and b.cash == pytest.approx(500.0)
    assert b.positions()["A"].quantity == 50
    r2 = b.submit(Order("A", "buy", 1000))   # capped to cash
    assert r2.status == "filled" and r2.filled_quantity == pytest.approx(50.0)
    assert b.cash == pytest.approx(0.0)
    assert b.submit(Order("A", "buy", 1)).status == "rejected"
    assert b.submit(Order("Z", "buy", 1)).status == "rejected"
    r3 = b.submit(Order("A", "sell", 500))   # capped to holdings
    assert r3.filled_quantity == pytest.approx(100.0) and "A" not in b.positions()
    assert b.account().equity == pytest.approx(1000.0)


def test_paper_broker_persists(tmp_path):
    f = tmp_path / "state.json"
    b = PaperBroker(cash=100.0, state_file=f, costs=CostModel.zero())
    b.set_prices({"A": 10.0})
    b.submit(Order("A", "buy", 5))
    b2 = PaperBroker(cash=999.0, state_file=f)
    assert b2.cash == pytest.approx(50.0) and b2._positions["A"]["qty"] == 5


def test_plan_and_execute_end_to_end(tmp_path):
    p = make_panel(n_tickers=4, n_days=320, seed=7, tickers=["A", "B", "C", "SPY"])
    store = BarStore(tmp_path / "cache")
    for t in p.tickers:
        store.save(t, pd.DataFrame({f: p.field(f)[t] for f in ("open", "high", "low", "close", "volume")}))
    broker = PaperBroker(cash=1000.0)
    broker.set_prices(p.close.iloc[-1])
    strat = Momentum(lookback=126, skip=21, top_n=2, market_filter=None)
    plan = plan_rebalance(strat, store, broker, p.tickers, force=True)
    assert plan.as_of == p.dates[-1]
    assert len(plan.orders) == 2 and all(o.side == "buy" for o in plan.orders)
    assert "orders=2" in plan.describe()
    results = execute_plan(plan, broker, log_path=tmp_path / "log.jsonl")
    assert all(r.status == "filled" for r in results)
    assert (tmp_path / "log.jsonl").read_text().count("\n") == 2
    # converged: nothing to do on the second pass
    assert plan_rebalance(strat, store, broker, p.tickers, force=True).orders == []


def test_plan_honours_schedule_and_force(tmp_path):
    # panel ending on a Wednesday mid-month: monthly strategy must skip unless forced
    start = pd.bdate_range(end="2024-03-13", periods=320)[0]   # ends on a mid-month Wednesday
    p = make_panel(n_tickers=3, n_days=320, seed=3, start=start, tickers=["A", "B", "C"])
    assert p.dates[-1] == pd.Timestamp("2024-03-13")
    store = BarStore(tmp_path / "cache")
    for t in p.tickers:
        store.save(t, pd.DataFrame({f: p.field(f)[t] for f in ("open", "high", "low", "close", "volume")}))
    broker = PaperBroker(cash=1000.0)
    broker.set_prices(p.close.iloc[-1])
    strat = Momentum(lookback=126, skip=21, top_n=2, market_filter=None)   # monthly
    plan = plan_rebalance(strat, store, broker, p.tickers)
    assert plan.skipped and plan.orders == [] and "monthly" in plan.skipped
    forced = plan_rebalance(strat, store, broker, p.tickers, force=True)
    assert forced.skipped is None and len(forced.orders) == 2
    daily = plan_rebalance(Momentum(lookback=126, skip=21, top_n=2, market_filter=None, rebalance="daily"),
                           store, broker, p.tickers)
    assert daily.skipped is None and len(daily.orders) == 2


def test_plan_requires_history(tmp_path):
    p = make_panel(n_tickers=2, n_days=30, seed=1)
    store = BarStore(tmp_path / "cache")
    for t in p.tickers:
        store.save(t, pd.DataFrame({f: p.field(f)[t] for f in ("open", "high", "low", "close", "volume")}))
    with pytest.raises(ValueError):
        plan_rebalance(Momentum(), store, PaperBroker(), p.tickers)
