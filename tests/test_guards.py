from datetime import datetime, timezone

import pandas as pd
import pytest

from algofun.backtest import CostModel
from algofun.broker import Order, PaperBroker
from algofun.broker.base import Account, Position
from algofun.live import (
    Guardrails,
    load_state,
    previous_trading_day,
    run_guards,
    save_state,
    snapshot_state,
)
from algofun.live.guards import (
    check_daily_loss,
    check_freshness,
    check_order_limits,
    check_reconciliation,
)
from algofun.live.rebalance import RebalancePlan


def _plan(as_of="2026-09-11", orders=None, prices=None, equity=1000.0):
    orders = orders or []
    prices = pd.Series(prices or {}, dtype="float64")
    return RebalancePlan(as_of=pd.Timestamp(as_of), equity=equity, cash=equity, target_weights=pd.Series(dtype=float),
                         current_weights=pd.Series(dtype=float), orders=orders, prices=prices)


def _now(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_freshness_rules():
    # Friday bar checked Friday evening / Saturday / Sunday: fresh
    for when in ("2026-09-11T22:30", "2026-09-12T12:00", "2026-09-13T12:00"):
        assert check_freshness(pd.Timestamp("2026-09-11"), 1, _now(when)) is None
    # Monday evening with only Friday's bar: allowed (one session stale)
    assert check_freshness(pd.Timestamp("2026-09-11"), 1, _now("2026-09-14T22:30")) is None
    # Tuesday with Friday's bar: two sessions stale -> blocked
    assert "stale" in check_freshness(pd.Timestamp("2026-09-11"), 1, _now("2026-09-15T22:30"))
    # bar from the future -> blocked
    assert "after today" in check_freshness(pd.Timestamp("2026-09-20"), 1, _now("2026-09-11T22:30"))
    assert previous_trading_day("2026-09-14") == pd.Timestamp("2026-09-11")


def test_daily_loss():
    f, n = check_daily_loss(Account(cash=0, equity=96.0, buying_power=0, last_equity=100.0), 0.03)
    assert f and "daily loss" in f
    f, n = check_daily_loss(Account(cash=0, equity=98.0, buying_power=0, last_equity=100.0), 0.03)
    assert f is None and "within limit" in n
    f, n = check_daily_loss(Account(cash=0, equity=98.0, buying_power=0), 0.03)
    assert f is None and "not checked" in n


def test_order_limits():
    g = Guardrails(max_orders=2, max_order_pct=0.30, max_gross_order_pct=1.10)
    plan = _plan(orders=[Order("A", "buy", 10), Order("B", "buy", 10), Order("C", "buy", 1)],
                 prices={"A": 10.0, "B": 10.0, "C": 500.0}, equity=1000.0)
    fails = check_order_limits(plan, 1000.0, g)
    assert any("3 orders exceeds" in f for f in fails)
    assert any("buy C" in f and "50%" in f for f in fails)
    ok = _plan(orders=[Order("A", "buy", 10)], prices={"A": 10.0})
    assert check_order_limits(ok, 1000.0, g) == []
    missing = _plan(orders=[Order("Z", "buy", 1)], prices={})
    assert any("no reference price" in f for f in check_order_limits(missing, 1000.0, g))


def test_reconciliation():
    pos = {"A": Position("A", 10.0, 1.0, 10.0), "B": Position("B", 5.0, 1.0, 5.0)}
    fails, note = check_reconciliation(pos, None)
    assert fails == [] and "first run" in note
    state = {"as_of": "2026-09-10", "positions": {"A": 10.0, "C": 3.0}, "pending": ["B"]}
    fails, _ = check_reconciliation(pos, state)
    assert len(fails) == 1 and "C" in fails[0] and "no longer held" in fails[0]
    state = {"as_of": "2026-09-10", "positions": {"A": 10.0}, "pending": []}
    fails, _ = check_reconciliation(pos, state)
    assert len(fails) == 1 and "unexpected position B" in fails[0]
    drift = {"A": Position("A", 10.05, 1.0, 10.0)}
    assert check_reconciliation(drift, {"positions": {"A": 10.0}, "pending": []})[0] == []
    big = {"A": Position("A", 20.0, 1.0, 20.0)}
    assert "differs" in check_reconciliation(big, {"positions": {"A": 10.0}, "pending": []})[0][0]


def test_run_guards_blocks_and_passes():
    plan = _plan(orders=[Order("A", "buy", 10)], prices={"A": 10.0})
    acct = Account(cash=1000, equity=1000, buying_power=1000, last_equity=1000)
    ok = run_guards(plan, acct, {}, Guardrails(), last_state=None, now=_now("2026-09-11T22:30"))
    assert ok.passed and "PASS" in ok.describe()
    bad = run_guards(plan, Account(cash=900, equity=900, buying_power=900, last_equity=1000), {}, Guardrails(),
                     now=_now("2026-09-11T22:30"))
    assert not bad.passed and "BLOCKED" in bad.describe()
    assert run_guards(plan, acct, {}, Guardrails.off(), now=_now("2026-12-31T00:00")).passed


def test_snapshot_roundtrip_and_paper_flatten(tmp_path):
    b = PaperBroker(cash=100.0, costs=CostModel.zero(), state_file=tmp_path / "paper.json")
    b.set_prices({"A": 10.0})
    r = b.submit(Order("A", "buy", 5))
    plan = _plan()
    state = snapshot_state(b, plan, [r])
    assert state["positions"] == {"A": 5.0} and state["pending"] == ["A"]
    save_state(tmp_path / "state.json", state)
    assert load_state(tmp_path / "state.json")["positions"] == {"A": 5.0}
    assert load_state(tmp_path / "missing.json") is None
    assert b.account().last_equity == pytest.approx(100.0)   # first observation of the day
    out = b.flatten()
    assert len(out) == 1 and out[0].status == "filled" and b.positions() == {}
    assert b.account().cash == pytest.approx(100.0)
