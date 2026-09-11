"""Transaction cost model.

Every fill pays: adverse slippage + half the bid/ask spread on the price,
plus commission. Zero-commission brokers still cost you the spread and the
slippage, which is why "free trades" strategies with high turnover die.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission_per_share: float = 0.0   # e.g. IBKR Pro: 0.005
    commission_min: float = 0.0         # per order minimum
    commission_pct: float = 0.0         # fraction of notional (some brokers/crypto)
    slippage_bps: float = 5.0           # adverse move vs reference price, in basis points
    spread_bps: float = 2.0             # full quoted spread; you pay half on each side

    @property
    def price_penalty(self) -> float:
        """Fractional adverse price move applied to every fill."""
        return (self.slippage_bps + self.spread_bps / 2.0) / 10_000.0

    def fill_price(self, reference_price: float, side: int) -> float:
        """side = +1 for buy (pay more), -1 for sell (receive less)."""
        if side not in (1, -1):
            raise ValueError("side must be +1 (buy) or -1 (sell)")
        return reference_price * (1.0 + side * self.price_penalty)

    def commission(self, quantity: float, price: float) -> float:
        qty = abs(quantity)
        if qty == 0:
            return 0.0
        c = qty * self.commission_per_share + qty * price * self.commission_pct
        return max(c, self.commission_min)

    # ---- presets ---------------------------------------------------------
    @classmethod
    def zero(cls) -> CostModel:
        """No costs at all. Only useful to measure how much costs matter."""
        return cls(0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def retail(cls) -> CostModel:
        """Commission-free broker (Alpaca, Robinhood) trading liquid US names."""
        return cls(commission_per_share=0.0, commission_min=0.0, commission_pct=0.0,
                   slippage_bps=5.0, spread_bps=2.0)

    @classmethod
    def ibkr(cls) -> CostModel:
        return cls(commission_per_share=0.005, commission_min=1.0, commission_pct=0.0,
                   slippage_bps=5.0, spread_bps=2.0)

    @classmethod
    def pessimistic(cls) -> CostModel:
        """Small caps, thin books, market orders at the open. Use this to stress test."""
        return cls(commission_per_share=0.0, commission_min=0.0, commission_pct=0.0,
                   slippage_bps=20.0, spread_bps=10.0)


COST_PRESETS = {
    "zero": CostModel.zero,
    "retail": CostModel.retail,
    "ibkr": CostModel.ibkr,
    "pessimistic": CostModel.pessimistic,
}
