"""AlpacaBroker order logic against a fake TradingClient (no network)."""
from types import SimpleNamespace

import pytest

pytest.importorskip("alpaca")

from algofun.broker import Order  # noqa: E402
from algofun.broker.alpaca import AlpacaBroker  # noqa: E402


class FakeOrder(SimpleNamespace):
    pass


def _order(cid="x", status="accepted", filled_qty="0", price=None, oid="o1"):
    return FakeOrder(id=oid, status=SimpleNamespace(value=status), filled_qty=filled_qty,
                     filled_avg_price=price, client_order_id=cid)


class FakeTrading:
    def __init__(self, fail_times=0, duplicate_after_fail=False):
        self.fail_times = fail_times
        self.duplicate_after_fail = duplicate_after_fail
        self.submitted = []
        self.calls = 0

    def submit_order(self, req):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("500 Internal Server Error")
        if self.duplicate_after_fail and self.calls > 1:
            raise RuntimeError('{"code":40010001,"message":"client_order_id must be unique"}')
        self.submitted.append(req)
        return _order(cid=req.client_order_id, filled_qty=str(req.qty), price="10.5", status="filled")

    def get_order_by_client_id(self, cid):
        return _order(cid=cid, status="accepted", oid="looked-up")

    def get_clock(self):
        raise RuntimeError("500")


def _broker(trading) -> AlpacaBroker:
    b = AlpacaBroker.__new__(AlpacaBroker)   # skip __init__ (needs keys/network)
    b.trading = trading
    b.name = "alpaca-paper"
    return b


def test_submit_passes_client_order_id_and_retries(monkeypatch):
    from algofun.broker import alpaca as mod
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    t = FakeTrading(fail_times=1)
    r = _broker(t).submit(Order("MU", "buy", 1.5, client_order_id="algofun-1-MU-buy"))
    assert r.status == "filled" and r.filled_quantity == 1.5 and r.filled_price == 10.5
    assert t.calls == 2 and t.submitted[0].client_order_id == "algofun-1-MU-buy"


def test_submit_without_key_does_not_retry(monkeypatch):
    from algofun.broker import alpaca as mod
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    t = FakeTrading(fail_times=1)
    r = _broker(t).submit(Order("MU", "buy", 1.0))
    assert r.status == "rejected" and t.calls == 1


def test_submit_duplicate_means_already_placed(monkeypatch):
    from algofun.broker import alpaca as mod
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    # first attempt "fails" after Alpaca actually accepted it; retry says duplicate
    t = FakeTrading(fail_times=1, duplicate_after_fail=True)
    r = _broker(t).submit(Order("MU", "buy", 1.0, client_order_id="k"))
    assert r.status == "accepted" and r.order_id == "looked-up"
    assert len(t.submitted) == 0


def test_market_clock_failure_surfaces_as_exception(monkeypatch):
    from algofun.broker import alpaca as mod
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        _broker(FakeTrading()).is_market_open()
