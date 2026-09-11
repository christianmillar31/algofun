import numpy as np

from algofun.backtest import BacktestConfig, walk_forward
from algofun.strategies import SMACrossover

from .conftest import make_panel


def test_walk_forward_folds_and_stitching():
    p = make_panel(n_tickers=4, n_days=700, seed=2)
    wf = walk_forward(p, SMACrossover, {"fast": [10, 20], "slow": [50]}, train_bars=200, test_bars=100,
                      config=BacktestConfig(benchmark="T0"))
    n = len(p)
    warm = SMACrossover(fast=10, slow=50).warmup
    expected_folds = 0
    ts = warm
    while ts + 200 + 1 < n:
        expected_folds += 1
        ts += 100
    assert len(wf.folds) == expected_folds
    assert wf.equity.index.is_monotonic_increasing and wf.equity.index.is_unique
    assert wf.equity.index[0] == p.dates[warm + 200]
    assert wf.benchmark is not None and len(wf.benchmark) == len(wf.equity)
    assert all(isinstance(x, dict) and x["slow"] == 50 for x in wf.folds["params"])
    assert "Stitched" in wf.summary() and "Honesty check" in wf.summary()
    assert wf.n_trials_per_fold == 2 and wf.n_trials_total == 2 * len(wf.folds)
    assert 0.0 <= wf.parameter_drift <= 1.0
    assert {"train_cagr", "test_cagr", "test_bars", "train_dsr", "n_trials"} <= set(wf.folds.columns)
    assert wf.folds["train_dsr"].between(0, 1).all()
    assert np.isfinite(wf.walk_forward_efficiency) or np.isnan(wf.walk_forward_efficiency)
