"""Portfolio-level drawdown budget.

Survival, not alpha: Carver's random-data test shows equity-curve rules
reduce the returns of a genuinely profitable system. This exists so a bad
month cannot become a ruinous year while a human is not watching. Exposure
halves past `halve_at` from the running peak and goes flat past `flat_at`;
it recovers automatically as equity climbs back (capital correction).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DrawdownControl:
    halve_at: float = 0.10
    flat_at: float = 0.20

    def scale(self, drawdown: float) -> float:
        """Exposure multiplier for a drawdown expressed as a positive fraction."""
        if drawdown >= self.flat_at:
            return 0.0
        if drawdown >= self.halve_at:
            return 0.5
        return 1.0

    @staticmethod
    def drawdown(history: Sequence[float], current: float | None = None) -> float:
        vals = [float(v) for v in history if v is not None and v == v]
        if current is not None:
            vals.append(float(current))
        if not vals:
            return 0.0
        peak = max(vals)
        return 0.0 if peak <= 0 else max(0.0, 1.0 - vals[-1] / peak)

    def scale_from_history(self, history: Sequence[float], current: float | None = None) -> tuple[float, float]:
        dd = self.drawdown(history, current)
        return self.scale(dd), dd
