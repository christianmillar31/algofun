from __future__ import annotations

from typing import Any

from .base import Strategy, clean_weights, pick_top
from .buy_and_hold import BuyAndHold
from .mean_reversion import MeanReversion
from .momentum import Momentum
from .sentiment import Sentiment
from .sma_crossover import SMACrossover

STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls for cls in (BuyAndHold, SMACrossover, Momentum, MeanReversion, Sentiment)
}


def get_strategy(name: str, **params: Any) -> Strategy:
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(STRATEGIES)}") from None
    return cls(**params)


def parse_params(spec: str | None) -> dict[str, Any]:
    """'fast=20,slow=100,sizing=inverse_vol' -> {'fast': 20, 'slow': 100, 'sizing': 'inverse_vol'}"""
    out: dict[str, Any] = {}
    if not spec:
        return out
    for item in spec.split(","):
        if not item.strip():
            continue
        k, _, v = item.partition("=")
        k, v = k.strip(), v.strip()
        for cast in (int, float):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            out[k] = {"true": True, "false": False, "none": None}.get(v.lower(), v)
    return out


__all__ = [
    "STRATEGIES",
    "BuyAndHold",
    "MeanReversion",
    "Momentum",
    "SMACrossover",
    "Sentiment",
    "Strategy",
    "clean_weights",
    "get_strategy",
    "parse_params",
    "pick_top",
]
