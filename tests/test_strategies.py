import numpy as np
import pandas as pd
import pytest

from algofun.backtest import BacktestConfig, MarketView, run_backtest
from algofun.data.store import Panel
from algofun.strategies import (
    STRATEGIES,
    MeanReversion,
    Momentum,
    SMACrossover,
    get_strategy,
    parse_params,
)


def _panel_from_closes(closes: dict[str, np.ndarray]) -> Panel:
    dates = pd.bdate_range("2019-01-01", periods=len(next(iter(closes.values()))))
    bars = {t: pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6}, index=dates)
            for t, c in closes.items()}
    return Panel.from_bars(bars)


def test_sma_picks_uptrend_only():
    n = 260
    up = np.linspace(100, 200, n)
    down = np.linspace(200, 100, n)
    p = _panel_from_closes({"UP": up, "DOWN": down})
    w = SMACrossover(fast=20, slow=100).target_weights(MarketView(p, n - 1))
    assert list(w.index) == ["UP"] and w["UP"] == pytest.approx(1.0)


def test_momentum_ranks_by_trailing_return_skipping_recent():
    n = 300
    rng = np.random.default_rng(0)
    base = 100 + rng.normal(0, 0.1, n).cumsum()
    strong = base * np.linspace(1, 3, n)
    weak = base * np.linspace(1, 1.1, n)
    spy = np.linspace(100, 150, n)
    p = _panel_from_closes({"STRONG": strong, "WEAK": weak, "SPY": spy})
    w = Momentum(lookback=126, skip=21, top_n=1, market_filter="SPY", filter_sma=100).target_weights(
        MarketView(p, n - 1))
    assert list(w.index) == ["STRONG"]
    # market filter off when SPY below its SMA
    p2 = _panel_from_closes({"STRONG": strong, "WEAK": weak, "SPY": spy[::-1]})
    w2 = Momentum(lookback=126, skip=21, top_n=1, market_filter="SPY", filter_sma=100).target_weights(
        MarketView(p2, n - 1))
    assert w2.empty


def test_mean_reversion_buys_dip_in_uptrend_and_exits():
    n = 200
    trend = np.linspace(100, 160, n)
    dip = trend.copy()
    dip[-1] *= 0.95  # sharp one-day drop, still above the 100-day mean
    calm = trend.copy()
    p = _panel_from_closes({"DIP": dip, "CALM": calm})
    s = MeanReversion(lookback=20, z_entry=-2.0, trend_sma=100)
    s.reset()
    w = s.target_weights(MarketView(p, n - 1))
    assert list(w.index) == ["DIP"]
    # after the price recovers the position is released
    recovered = dip.copy()
    recovered = np.append(recovered, [trend[-1] * 1.02] * 5)
    p2 = _panel_from_closes({"DIP": recovered, "CALM": np.append(calm, [calm[-1]] * 5)})
    w2 = s.target_weights(MarketView(p2, len(recovered) - 1))
    assert w2.empty


def test_all_registered_strategies_run(big_panel):
    for name in STRATEGIES:
        params = {"tickers": "T0,T1"} if name == "buy_and_hold" else {}
        if name == "momentum":
            params = {"market_filter": None, "top_n": 3}
        strat = get_strategy(name, **params)
        res = run_backtest(big_panel, strat, BacktestConfig(benchmark=None))
        assert len(res.equity) > 0 and res.equity.iloc[-1] > 0


def test_param_validation_and_grid():
    with pytest.raises(TypeError):
        SMACrossover(bogus=1)
    with pytest.raises(ValueError):
        SMACrossover(fast=200, slow=50)
    assert len(SMACrossover.grid()) == 4
    with pytest.raises(KeyError):
        get_strategy("nope")


def test_parse_params():
    assert parse_params("fast=20,slow=100,sizing=inverse_vol,flag=true,x=none,z=-1.5") == {
        "fast": 20, "slow": 100, "sizing": "inverse_vol", "flag": True, "x": None, "z": -1.5}
    assert parse_params(None) == {} and parse_params("") == {}
