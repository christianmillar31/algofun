import numpy as np
import pandas as pd
import pytest

from algofun.backtest import BacktestConfig, CostModel, run_backtest
from algofun.risk import RiskLimits
from algofun.strategies import BuyAndHold
from algofun.strategies.base import Strategy

from .conftest import make_panel


class Constant(Strategy):
    name = "constant"
    rebalance = "daily"
    defaults = {"weights": None}

    def target_weights(self, view):
        return pd.Series(self.weights)


def test_accounting_identity(panel):
    res = run_backtest(panel, Constant(weights={"T0": 0.4, "T1": 0.3}), BacktestConfig(benchmark=None))
    mv = (res.positions * panel.close.ffill().loc[res.equity.index]).sum(axis=1)
    assert np.allclose(res.cash + mv, res.equity)
    assert (res.cash >= -1e-9).all()


def test_buy_and_hold_tracks_price_with_zero_costs():
    p = make_panel(n_tickers=1, n_days=100, seed=5, tickers=["SPY"])
    cfg = BacktestConfig(initial_cash=1000.0, costs=CostModel.zero(), benchmark=None,
                         limits=RiskLimits(max_weight=1.0), rebalance="daily")
    res = run_backtest(p, BuyAndHold(tickers="SPY"), cfg)
    fill_open = p.open["SPY"].iloc[1]
    expected = 1000.0 / fill_open * p.close["SPY"]
    # from the first fill on, equity is exactly shares * close (no cash left, no costs)
    assert np.allclose(res.equity.iloc[1:], expected.iloc[1:], rtol=1e-9)
    assert res.equity.iloc[0] == 1000.0
    assert len(res.trades) == 1   # bought once, then held; daily rebalances are no-ops


def test_costs_reduce_equity(panel):
    s = Constant(weights={"T0": 0.5, "T1": 0.5})
    free = run_backtest(panel, s, BacktestConfig(costs=CostModel.zero(), benchmark=None, drawdown_control=None))
    paid = run_backtest(panel, s, BacktestConfig(costs=CostModel.pessimistic(), benchmark=None, drawdown_control=None))
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]


def test_shorts_clipped_when_disallowed(panel):
    res = run_backtest(panel, Constant(weights={"T0": -0.5, "T1": 0.5}), BacktestConfig(benchmark=None))
    assert (res.positions["T0"] == 0).all()
    assert (res.positions["T1"] > 0).iloc[2:].all()


def test_gross_limit_scales_weights(panel):
    cfg = BacktestConfig(benchmark=None, limits=RiskLimits(max_weight=1.0, max_gross=1.0),
                         costs=CostModel.zero())
    res = run_backtest(panel, Constant(weights={"T0": 1.0, "T1": 1.0}), cfg)
    assert res.weights.iloc[-1].sum() <= 1.0 + 1e-6
    assert res.cash.min() >= -1e-9


def test_whole_shares_only(panel):
    cfg = BacktestConfig(benchmark=None, allow_fractional=False, initial_cash=5000.0)
    res = run_backtest(panel, Constant(weights={"T0": 0.5, "T1": 0.5}), cfg)
    assert np.allclose(res.positions.to_numpy(), np.round(res.positions.to_numpy()))
    assert res.cash.min() >= -1e-9


def test_start_end_window_and_history_visible(big_panel):
    class NeedsHistory(Strategy):
        name = "needs_history"
        warmup = 100
        rebalance = "daily"

        def target_weights(self, view):
            assert len(view.close) >= 100
            return pd.Series({"T0": 1.0})

    start, end = big_panel.dates[400], big_panel.dates[500]
    res = run_backtest(big_panel, NeedsHistory(), BacktestConfig(benchmark=None), start=start, end=end)
    assert res.equity.index[0] == start
    assert res.equity.index[-1] == end
    assert len(res.equity) == 101


def test_warmup_is_respected(panel):
    class W(Strategy):
        name = "w"
        warmup = 50
        rebalance = "daily"

        def target_weights(self, view):
            assert len(view) >= 50
            return pd.Series({"T0": 1.0})

    res = run_backtest(panel, W(), BacktestConfig(benchmark=None))
    assert res.equity.index[0] == panel.dates[50]


def test_rebalance_override_and_no_trade_band(panel):
    s = Constant(weights={"T0": 0.5, "T1": 0.5})
    daily = run_backtest(panel, s, BacktestConfig(benchmark=None, rebalance="daily", min_trade_weight=0.0))
    banded = run_backtest(panel, s, BacktestConfig(benchmark=None, rebalance="daily", min_trade_weight=0.02))
    monthly = run_backtest(panel, s, BacktestConfig(benchmark=None, rebalance="monthly"))
    assert len(banded.trades) < len(daily.trades)
    assert len(monthly.trades) < len(daily.trades)


def test_closing_trade_ignores_band(panel):
    class Flip(Strategy):
        name = "flip"
        rebalance = "daily"

        def target_weights(self, view):
            return pd.Series({"T0": 1.0}) if view.bar_index < 10 else pd.Series(dtype="float64")

    cfg = BacktestConfig(benchmark=None, min_trade_weight=0.5, limits=RiskLimits(max_weight=1.0))
    res = run_backtest(panel, Flip(), cfg)
    assert res.positions["T0"].iloc[-1] == 0
    assert set(res.trades["side"]) == {"buy", "sell"}


def test_untradable_ticker_is_skipped(panel):
    p = panel
    p.close.iloc[:150, 0] = np.nan  # T0 does not exist for the first half
    p.open.iloc[:150, 0] = np.nan
    res = run_backtest(p, Constant(weights={"T0": 0.5, "T1": 0.5}), BacktestConfig(benchmark=None))
    assert (res.positions["T0"].iloc[:150] == 0).all()
    assert (res.positions["T0"].iloc[152:] > 0).all()


def test_benchmark_and_metrics(big_panel):
    p = big_panel
    cfg = BacktestConfig(benchmark="T0")
    res = run_backtest(p, Constant(weights={"T1": 0.5, "T2": 0.5}), cfg)
    m = res.metrics()
    assert res.benchmark is not None and res.benchmark.iloc[0] == cfg.initial_cash
    for k in ("cagr", "sharpe", "max_drawdown", "benchmark_cagr", "beta", "annual_turnover", "win_rate"):
        assert k in m
    assert "T1" in res.summary() or "constant" in res.summary()


def test_save(tmp_path, panel):
    res = run_backtest(panel, Constant(weights={"T0": 1.0}), BacktestConfig(benchmark=None))
    res.save(tmp_path)
    assert (tmp_path / "equity.csv").exists() and (tmp_path / "metrics.json").exists()


def test_empty_window_raises(panel):
    with pytest.raises(ValueError):
        run_backtest(panel, Constant(weights={}), BacktestConfig(benchmark=None),
                     start=panel.dates[-1] + pd.Timedelta(days=30))
