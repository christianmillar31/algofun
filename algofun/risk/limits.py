"""Hard portfolio limits applied after the strategy speaks and before orders exist."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RiskLimits:
    max_weight: float = 0.25     # per-name cap on |weight|
    max_gross: float = 1.0       # sum |weights| cap (1.0 = no leverage)
    max_net: float | None = None # cap on sum(weights); None = no cap
    allow_short: bool = False
    min_weight: float = 0.0      # drop dust positions below this |weight|

    def apply(self, weights: pd.Series) -> pd.Series:
        w = weights.astype("float64").copy()
        if not self.allow_short:
            w = w.clip(lower=0.0)
        w = w.clip(lower=-self.max_weight, upper=self.max_weight)
        if self.min_weight > 0:
            w[w.abs() < self.min_weight] = 0.0
        gross = w.abs().sum()
        if gross > self.max_gross and gross > 0:
            w = w * (self.max_gross / gross)
        if self.max_net is not None:
            net = w.sum()
            if abs(net) > self.max_net and net != 0:
                w = w * (self.max_net / abs(net))
        return w
