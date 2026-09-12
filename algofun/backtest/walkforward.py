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
from .metrics import cagr, compute_metrics, format_metrics
from .stats import honesty_report


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame       # one row per fold: windows, chosen params, train/test metric
    equity: pd.Series         # stitched out-of-sample equity
    benchmark: pd.Series | None
    test_results: list[BacktestResult]
    objective: str
    n_trials_per_fold: int = 1

    @property
    def n_trials_total(self) -> int:
        return self.n_trials_per_fold * len(self.folds)

    @property
    def walk_forward_efficiency(self) -> float:
        """Out-of-sample annualised return over the in-sample annualised return of
        the parameters that were chosen (Pardo). > 0.5 acceptable, > 0.6 robust."""
        is_cagr = float(np.average(self.folds["train_cagr"], weights=self.folds["test_bars"]))
        oos = cagr(self.equity)
        if is_cagr <= 0:
            return float("nan")
        return oos / is_cagr

    @property
    def parameter_drift(self) -> float:
        """Share of folds whose chosen parameters differ from the previous fold's."""
        params = [tuple(sorted(p.items())) for p in self.folds["params"]]
        if len(params) < 2:
            return 0.0
        return sum(a != b for a, b in zip(params[1:], params[:-1], strict=True)) / (len(params) - 1)

    def honesty(self):
        return honesty_report(self.equity.pct_change().dropna())

    def metrics(self) -> dict[str, float]:
        trades = pd.concat([r.trades for r in self.test_results], ignore_index=True) if self.test_results else None
        turnover = pd.concat([r.turnover for r in self.test_results]) if self.test_results else None
        return compute_metrics(self.equity, trades=trades, turnover=turnover, benchmark=self.benchmark)

    def summary(self) -> str:
        cols = ["train_start", "test_start", "test_end", "params", f"train_{self.objective}",
                f"test_{self.objective}", "train_dsr"]
        table = self.folds[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}")
        h = self.honesty()
        wfe = self.walk_forward_efficiency
        honesty = [
            "Honesty check:",
            f"  configurations per fold  {self.n_trials_per_fold:8d}  (total trials {self.n_trials_total})",
            f"  walk_forward_efficiency  {wfe:8.3f}  (OOS CAGR / IS CAGR; > 0.5 ok, > 0.6 robust)",
            f"  parameter_drift          {self.parameter_drift:8.3f}  (share of folds that switched params)",
            f"  mean_train_dsr           {float(self.folds['train_dsr'].mean()):8.3f}  (in-sample deflated Sharpe; want >= 0.95)",
            "  out-of-sample:",
        ] + h.lines()
        return ("Walk-forward folds:\n" + table + "\n\n" + "\n".join(honesty)
                + "\n\nStitched out-of-sample:\n" + format_metrics(self.metrics()))


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
        best_score, best_params, best_res = -np.inf, None, None
        trial_sharpes: list[float] = []
        for params in combos:
            strat = strategy_cls(**{**fixed, **params})
            res = run_backtest(panel, strat, cfg, start=dates[train_start], end=dates[train_end])
            m = res.metrics()
            score = m.get(objective, -np.inf)
            trial_sharpes.append(m.get("sharpe", np.nan))
            if verbose:
                print(f"  fold {len(rows)} train {params}: {objective}={score:.3f}")
            if score > best_score:
                best_score, best_params, best_res = score, params, res
        train_h = honesty_report(best_res.returns.iloc[1:], trial_sharpes_annual=trial_sharpes)
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
            "train_cagr": cagr(best_res.equity), "test_cagr": tm["cagr"], "test_bars": test_end - test_start + 1,
            "train_dsr": train_h.dsr if train_h.dsr is not None else train_h.psr_zero,
            "n_trials": len(combos),
        })
        train_start += step
    if not rows:
        raise ValueError(f"not enough data for one fold: need {warm + train_bars + test_bars} bars, have {n}")
    eq, bench = _stitch(tests, cfg.initial_cash)
    return WalkForwardResult(folds=pd.DataFrame(rows), equity=eq, benchmark=bench,
                             test_results=tests, objective=objective, n_trials_per_fold=len(combos))
