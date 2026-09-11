"""Walk-forward optimisation: the honest way to pick parameters.

For each fold, grid-search parameters on a training window, then run the
winner untouched on the following test window. Stitch the test windows
together. That stitched curve is the only performance number that has not
seen its own future.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..data.store import Panel
from ..strategies.base import Strategy
from .engine import BacktestConfig, BacktestResult, run_backtest
from .metrics import compute_metrics, format_metrics


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame       # one row per fold: windows, chosen params, train/test metric
    equity: pd.Series         # stitched out-of-sample equity
    benchmark: pd.Series | None
    test_results: list[BacktestResult]
    objective: str

    def metrics(self) -> dict[str, float]:
        trades = pd.concat([r.trades for r in self.test_results], ignore_index=True) if self.test_results else None
        turnover = pd.concat([r.turnover for r in self.test_results]) if self.test_results else None
        return compute_metrics(self.equity, trades=trades, turnover=turnover, benchmark=self.benchmark)

    def summary(self) -> str:
        cols = ["train_start", "test_start", "test_end", "params", f"train_{self.objective}",
                f"test_{self.objective}"]
        table = self.folds[cols].to_string(index=False)
        return ("Walk-forward folds:\n" + table + "\n\nStitched out-of-sample:\n"
                + format_metrics(self.metrics()))


def _stitch(results: list[BacktestResult], initial: float) -> tuple[pd.Series, pd.Series | None]:
    """Chain per-fold returns. Each fold starts flat (in cash) on its first bar,
    so that bar contributes a zero return; folds never share positions."""
    rets = [r.equity.pct_change().fillna(0.0) for r in results]
    brets = [r.benchmark.pct_change().fillna(0.0) for r in results if r.benchmark is not None]
    eq = (1.0 + pd.concat(rets)).cumprod() * initial
    bench = (1.0 + pd.concat(brets)).cumprod() * initial if len(brets) == len(results) else None
    return eq.rename("equity"), (bench.rename("benchmark") if bench is not None else None)


def walk_forward(panel: Panel, strategy_cls: type[Strategy], param_grid: dict[str, list[Any]] | None = None,
                 train_bars: int = 504, test_bars: int = 126, step_bars: int | None = None,
                 objective: str = "sharpe", config: BacktestConfig | None = None,
                 fixed_params: dict[str, Any] | None = None, verbose: bool = False) -> WalkForwardResult:
    """Rolling walk-forward.

    train_bars / test_bars are in trading days (252 = one year). The first
    fold's training window starts after the strategy's warmup.
    """
    cfg = config or BacktestConfig()
    grid = param_grid if param_grid is not None else strategy_cls.param_grid
    fixed = fixed_params or {}
    keys = list(grid)
    combos = [dict(zip(keys, v, strict=True)) for v in itertools.product(*(grid[k] for k in keys))] if keys else [{}]
    step = step_bars or test_bars
    dates = panel.dates
    n = len(dates)
    warm = max(strategy_cls(**{**fixed, **combos[0]}).warmup, 0)

    rows = []
    tests: list[BacktestResult] = []
    train_start = warm
    while train_start + train_bars + 1 < n:
        train_end = train_start + train_bars - 1
        test_start = train_end + 1
        test_end = min(test_start + test_bars - 1, n - 1)
        if test_end <= test_start:
            break
        best_score, best_params = -np.inf, None
        for params in combos:
            strat = strategy_cls(**{**fixed, **params})
            res = run_backtest(panel, strat, cfg, start=dates[train_start], end=dates[train_end])
            score = res.metrics().get(objective, -np.inf)
            if verbose:
                print(f"  fold {len(rows)} train {params}: {objective}={score:.3f}")
            if score > best_score:
                best_score, best_params = score, params
        strat = strategy_cls(**{**fixed, **best_params})
        test = run_backtest(panel, strat, cfg, start=dates[test_start], end=dates[test_end])
        tm = test.metrics()
        tests.append(test)
        rows.append({
            "fold": len(rows), "train_start": dates[train_start].date(), "train_end": dates[train_end].date(),
            "test_start": dates[test_start].date(), "test_end": dates[test_end].date(),
            "params": best_params, f"train_{objective}": best_score,
            f"test_{objective}": tm.get(objective, np.nan), "test_return": tm["total_return"],
            "test_max_drawdown": tm["max_drawdown"],
        })
        train_start += step
    if not rows:
        raise ValueError(f"not enough data for one fold: need {warm + train_bars + test_bars} bars, have {n}")
    eq, bench = _stitch(tests, cfg.initial_cash)
    return WalkForwardResult(folds=pd.DataFrame(rows), equity=eq, benchmark=bench,
                             test_results=tests, objective=objective)
