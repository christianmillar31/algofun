import pytest

from algofun.backtest import CostModel


def test_fill_price_direction():
    c = CostModel(slippage_bps=10, spread_bps=10)   # penalty = 15 bps
    assert c.fill_price(100.0, 1) == pytest.approx(100.15)
    assert c.fill_price(100.0, -1) == pytest.approx(99.85)
    with pytest.raises(ValueError):
        c.fill_price(100.0, 0)


def test_commission_minimum_and_pct():
    c = CostModel(commission_per_share=0.005, commission_min=1.0)
    assert c.commission(10, 50.0) == 1.0          # below minimum
    assert c.commission(1000, 50.0) == 5.0
    assert c.commission(0, 50.0) == 0.0
    pct = CostModel(commission_pct=0.001)
    assert pct.commission(10, 100.0) == pytest.approx(1.0)


def test_presets():
    assert CostModel.zero().price_penalty == 0.0
    assert CostModel.zero().commission(100, 10) == 0.0
    assert CostModel.pessimistic().price_penalty > CostModel.retail().price_penalty
    assert CostModel.ibkr().commission(1, 10) == 1.0
