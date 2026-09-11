"""Strategy interface.

A strategy is a pure function from MarketView -> target portfolio weights.
It never touches orders, cash, or the broker. That is what lets the same
object drive the backtester and the live rebalancer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd

from ..backtest.view import MarketView


class Strategy(ABC):
    #: registry key and display name
    name: ClassVar[str] = "base"
    #: bars of history required before the first decision
    warmup: int = 0
    #: 'daily' | 'weekly' | 'monthly' | 'quarterly' | int
    rebalance: str | int = "daily"
    #: parameter grid used by walk-forward optimisation: {param: [values]}
    param_grid: ClassVar[dict[str, list[Any]]] = {}
    #: default parameter values, also used to validate kwargs
    defaults: ClassVar[dict[str, Any]] = {}

    def __init__(self, **params: Any):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise TypeError(f"{self.name}: unknown params {sorted(unknown)}; "
                            f"valid: {sorted(self.defaults)}")
        self.params: dict[str, Any] = {**self.defaults, **params}
        for k, v in self.params.items():
            setattr(self, k, v)
        self.configure()

    def configure(self) -> None:  # noqa: B027 - optional hook
        """Hook: derive warmup/rebalance from params after they are set."""

    def reset(self) -> None:  # noqa: B027 - optional hook
        """Hook: clear any per-run state. The engine calls this before a run."""

    @abstractmethod
    def target_weights(self, view: MarketView) -> pd.Series:
        """Return desired portfolio weights indexed by ticker.

        Weights are fractions of total equity. Omitted tickers mean 0.
        Long-only strategies return values in [0, 1] summing to <= 1.
        """

    def describe(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({p})"

    def __repr__(self) -> str:
        return self.describe()

    @classmethod
    def grid(cls) -> list[dict[str, Any]]:
        """All parameter combinations from param_grid."""
        import itertools
        if not cls.param_grid:
            return [dict()]
        keys = list(cls.param_grid)
        return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(cls.param_grid[k] for k in keys))]


def clean_weights(weights: pd.Series | dict | None, tickers: list[str]) -> pd.Series:
    """Coerce a strategy's output into a float Series over the panel's tickers."""
    if weights is None:
        return pd.Series(0.0, index=tickers)
    w = pd.Series(weights, dtype="float64")
    w = w.reindex(tickers).fillna(0.0)
    w = w.replace([float("inf"), float("-inf")], 0.0)
    return w
