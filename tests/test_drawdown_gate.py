import json

import numpy as np
import pandas as pd
import pytest

from algofun.backtest import BacktestConfig, CostModel, run_backtest
from algofun.broker import PaperBroker
from algofun.data import BarStore
from algofun.live import execute_plan, gate_status, plan_rebalance, shortfall_report
from algofun.risk import DrawdownControl, RiskLimits
from algofun.strategies import Momentum
from algofun.strategies.base import Strategy

from .conftest import make_panel


def test_drawdown_control_thresholds_and_history():
    c = DrawdownControl(halve_at=0.10, flat_at=0.20)
    assert c.scale(0.05) == 1.0 and c.scale(0.10) == 0.5 and c.scale(0.25) == 0.0
    assert DrawdownControl.drawdown([100, 120, 90]) == pytest.approx(0.25)
    assert DrawdownControl.drawdown([100, 120], current=108) == pytest.approx(0.10)
    assert DrawdownControl.drawdown([]) == 0.0
    assert c.scale_from_history([100, 120], 100) == (0.5, pytest.approx(1 / 6))


class AllIn(Strategy):
    name = "all_in"
    rebalance = "daily"

    def target_weights(self, view):
        return pd.Series({"A": 1.0})


def _crash_panel():
    n = 120
    px = np.concatenate([np.linspace(100, 130, 40), np.linspace(130, 85, 20), np.full(60, 85.0)])
    p = make_panel(n_tickers=1, n_days=n, tickers=["A"])
    for f in ("open", "high", "low", "close"):
        p.field(f)["A"] = px
    return p


def test_engine_goes_flat_past_the_budget_and_recovers_nothing_by_itself():
    p = _crash_panel()
    base = dict(benchmark=None, costs=CostModel.zero(), limits=RiskLimits(max_weight=1.0))
    with_ctrl = run_backtest(p, AllIn(), BacktestConfig(**base, drawdown_control=DrawdownControl()))
    without = run_backtest(p, AllIn(), BacktestConfig(**base, drawdown_control=None))
    # after the 35% crash the controlled book is flat, the uncontrolled one is not
    assert (with_ctrl.positions["A"].iloc[70:] == 0).all()
    assert (without.positions["A"].iloc[70:] > 0).all()
    assert with_ctrl.equity.min() > without.equity.min()


def _store(tmp_path, tickers=("A", "B", "C")):
    p = make_panel(n_tickers=len(tickers), n_days=320, seed=5, tickers=list(tickers))
    store = BarStore(tmp_path / "cache")
    for t in p.tickers:
        store.save(t, pd.DataFrame({f: p.field(f)[t] for f in ("open", "high", "low", "close", "volume")}))
    return p, store


def test_plan_exposure_scale_halves_the_book(tmp_path):
    p, store = _store(tmp_path)
    broker = PaperBroker(cash=1000.0)
    broker.set_prices(p.close.iloc[-1])
    strat = Momentum(lookback=126, skip=21, top_n=2, market_filter=None, vol_target=0)
    full = plan_rebalance(strat, store, broker, p.tickers, force=True, limits=RiskLimits(max_weight=0.5))
    half = plan_rebalance(strat, store, broker, p.tickers, force=True, limits=RiskLimits(max_weight=0.5),
                          exposure_scale=0.5, drawdown=0.12)
    n_full = sum(o.quantity * full.prices[o.ticker] for o in full.orders)
    n_half = sum(o.quantity * half.prices[o.ticker] for o in half.orders)
    assert n_half == pytest.approx(n_full / 2, rel=1e-6)
    assert "drawdown budget" in half.describe() and "x0.50" in half.describe()


class SlowFill(PaperBroker):
    """Accepts first, fills on refresh (like a real broker)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.refreshed = 0

    def submit(self, order):
        r = super().submit(order)
        r.status, r.filled_quantity, r._fill = "accepted", 0.0, (r.filled_quantity, r.filled_price)
        r.filled_price = None
        return r

    def refresh(self, results):
        self.refreshed += 1
        for r in results:
            q, px = r._fill
            r.status, r.filled_quantity, r.filled_price = "filled", q, px
        return results


def test_execute_plan_settles_refreshes_and_logs_modelled_price(tmp_path, monkeypatch):
    p, store = _store(tmp_path)
    broker = SlowFill(cash=1000.0, costs=CostModel.retail())
    broker.set_prices(p.close.iloc[-1])
    plan = plan_rebalance(Momentum(lookback=126, skip=21, top_n=2, market_filter=None, vol_target=0),
                          store, broker, p.tickers, force=True, limits=RiskLimits(max_weight=0.5))
    import algofun.live.rebalance as mod
    monkeypatch.setattr(mod.time, "sleep", lambda s: None, raising=False)
    log = tmp_path / "fills.jsonl"
    results = execute_plan(plan, broker, log_path=log, settle_seconds=1)
    assert broker.refreshed == 1 and all(r.status == "filled" for r in results)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert all("modelled_price" in r and r["filled_price"] > r["modelled_price"] for r in rows if r["side"] == "buy")
    rep = shortfall_report(log)
    assert rep["n"] == 2 and rep["mean_bps"] == pytest.approx(6.0, abs=0.5)   # retail preset: 5 bps + half of 2 bps
    assert rep["total_cost"] > 0


def test_gate_status_counts_fills_and_clean_runs(tmp_path):
    fills = tmp_path / "fills.jsonl"
    guards = tmp_path / "guards.jsonl"
    with fills.open("w") as f:
        for i in range(30):
            f.write(json.dumps({"broker": "alpaca-paper", "as_of": f"2026-09-{(i % 10) + 1:02d}", "status": "filled",
                                "filled_quantity": 1}) + "\n")
        f.write(json.dumps({"broker": "alpaca-LIVE", "as_of": "2026-09-11", "status": "filled", "filled_quantity": 1}) + "\n")
        f.write(json.dumps({"broker": "alpaca-paper", "as_of": "2026-09-12", "status": "rejected", "filled_quantity": 0}) + "\n")
    with guards.open("w") as f:
        for ok in [True, True, False, True, True, True]:
            f.write(json.dumps({"broker": "alpaca-paper", "passed": ok, "as_of": "2026-09-11"}) + "\n")
    gs = gate_status(fills, guards, min_fills=25, min_sessions=3)
    assert gs.fills == 30 and gs.fill_sessions == 10 and gs.clean_sessions == 3 and gs.total_runs == 6
    assert gs.open and "OPEN" in gs.describe()
    assert not gate_status(fills, guards, min_fills=100, min_sessions=3).open
    assert not gate_status(fills, guards, min_fills=25, min_sessions=4).open
    empty = gate_status(tmp_path / "none.jsonl", tmp_path / "none2.jsonl")
    assert not empty.open and any("no paper fills" in n for n in empty.notes)
