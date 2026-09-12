import numpy as np
import pandas as pd
import pytest

from algofun.backtest.metrics import (
    cagr,
    compute_metrics,
    drawdown_series,
    max_drawdown,
    sharpe,
    sortino,
    trade_stats,
)


def test_max_drawdown_known_series():
    eq = pd.Series([100, 120, 90, 100, 130, 65, 70], dtype=float)
    assert max_drawdown(eq) == pytest.approx(-0.5)
    dd = drawdown_series(eq)
    assert dd.iloc[0] == 0 and dd.iloc[1] == 0 and dd.iloc[2] == pytest.approx(-0.25)


def test_cagr_doubling_in_one_year():
    eq = pd.Series(np.linspace(100, 200, 253))
    assert cagr(eq) == pytest.approx(1.0)
    assert cagr(pd.Series([100.0])) == 0.0


def test_sharpe_and_sortino_guards():
    flat = pd.Series([0.0] * 50)
    assert sharpe(flat) == 0.0 and sortino(flat) == 0.0
    up = pd.Series([0.001] * 50 + [-0.0005] * 50)
    assert sharpe(up) > 0 and sortino(up) > 0
    assert sortino(pd.Series([0.001] * 50)) == 0.0  # no downside -> guarded


def test_trade_stats_round_trips():
    trades = pd.DataFrame([
        {"date": 1, "ticker": "A", "side": "buy", "quantity": 10, "price": 10.0, "commission": 0.0},
        {"date": 2, "ticker": "A", "side": "sell", "quantity": 10, "price": 12.0, "commission": 0.0},
        {"date": 3, "ticker": "B", "side": "buy", "quantity": 5, "price": 20.0, "commission": 1.0},
        {"date": 4, "ticker": "B", "side": "sell", "quantity": 5, "price": 18.0, "commission": 1.0},
    ])
    s = trade_stats(trades)
    assert s["n_round_trips"] == 2
    assert s["win_rate"] == 0.5
    assert s["avg_win"] == pytest.approx(20.0)
    assert s["avg_loss"] == pytest.approx(-12.0)
    assert s["total_commission"] == 2.0
    assert trade_stats(None)["n_trades"] == 0


def test_compute_metrics_with_benchmark():
    idx = pd.bdate_range("2020-01-01", periods=300)
    eq = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.0005, 0.01, 300))), index=idx)
    bench = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0.0004, 0.01, 300))), index=idx)
    m = compute_metrics(eq, benchmark=bench)
    assert m["n_days"] == 300
    assert "beta" in m and "alpha_annual" in m and "excess_cagr" in m
