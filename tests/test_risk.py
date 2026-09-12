import numpy as np
import pandas as pd
import pytest

from algofun.risk import RiskLimits, equal_weight, inverse_vol, vol_target


def test_limits_cap_scale_and_clip():
    w = pd.Series({"A": 0.8, "B": 0.6, "C": -0.3})
    out = RiskLimits(max_weight=0.5, max_gross=1.0, allow_short=False).apply(w)
    assert (out >= 0).all()
    assert out.max() <= 0.5 + 1e-12
    assert out.sum() == pytest.approx(1.0)
    short_ok = RiskLimits(max_weight=1.0, max_gross=2.0, allow_short=True).apply(w)
    assert short_ok["C"] == pytest.approx(-0.3)
    dust = RiskLimits(min_weight=0.05).apply(pd.Series({"A": 0.01, "B": 0.5}))
    assert dust["A"] == 0.0
    net = RiskLimits(max_weight=1.0, max_gross=2.0, allow_short=True, max_net=0.5).apply(w)
    assert abs(net.sum()) <= 0.5 + 1e-12


def test_equal_weight():
    assert equal_weight(["A", "B"]).tolist() == [0.5, 0.5]
    assert equal_weight(pd.Series({"A": True, "B": False}))["A"] == 1.0
    assert equal_weight([]).empty


def test_inverse_vol_and_vol_target():
    rng = np.random.default_rng(0)
    r = pd.DataFrame({"A": rng.normal(0, 0.01, 100), "B": rng.normal(0, 0.03, 100)})
    w = inverse_vol(r)
    assert w.sum() == pytest.approx(1.0)
    assert w["A"] > w["B"]
    scaled = vol_target(w, r, target_annual_vol=0.05, max_leverage=1.0)
    assert scaled.sum() < w.sum()
    assert vol_target(pd.Series(dtype=float), r).empty
