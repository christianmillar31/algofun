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
    max_sector_weight: float | None = None   # cap on sum |weights| within one sector (needs a sector map)

    def apply(self, weights: pd.Series, sectors: dict[str, str] | None = None) -> pd.Series:
        w = weights.astype("float64").copy()
        if not self.allow_short:
            w = w.clip(lower=0.0)
        w = w.clip(lower=-self.max_weight, upper=self.max_weight)
        if self.min_weight > 0:
            w[w.abs() < self.min_weight] = 0.0
        if self.max_sector_weight is not None and sectors:
            w = cap_sectors(w, sectors, self.max_sector_weight, self.max_weight)
        gross = w.abs().sum()
        if gross > self.max_gross and gross > 0:
            w = w * (self.max_gross / gross)
        if self.max_net is not None:
            net = w.sum()
            if abs(net) > self.max_net and net != 0:
                w = w * (self.max_net / abs(net))
        return w


def cap_sectors(weights: pd.Series, sectors: dict[str, str], cap: float, max_weight: float = 1.0,
                max_iter: int = 5) -> pd.Series:
    """Scale any sector above `cap` down to it and hand the excess pro rata to
    names in sectors that still have room (MSCI-style), never pushing a
    receiving sector above the cap or a name above max_weight. Whatever
    cannot be placed becomes cash. Names with no sector are exempt from the
    cap (the per-name limit still binds them) and may receive excess.
    """
    w = weights.astype("float64").copy()
    if w.empty:
        return w
    sec = pd.Series({t: sectors.get(t) for t in w.index}, dtype="object")
    known = sec.notna()
    closed: set[str] = set()
    for _ in range(max_iter):
        totals = w[known].abs().groupby(sec[known]).sum()
        over = totals[totals > cap * (1 + 1e-9)]
        if over.empty:
            break
        excess = 0.0
        for s_name, tot in over.items():
            mask = (sec == s_name).to_numpy()
            w[mask] = w[mask] * (cap / tot)
            excess += tot - cap
            closed.add(s_name)
        totals = w[known].abs().groupby(sec[known]).sum()
        receivers = (w.abs() > 0) & ~sec.isin(closed)
        if not receivers.any() or excess <= 0:
            break
        absw = w[receivers].abs()
        # room per name: the per-name cap, and a pro-rata share of the sector's remaining room
        name_room = (max_weight - absw).clip(lower=0)
        sector_room = pd.Series(index=absw.index, dtype="float64")
        for t in absw.index:
            s_name = sec[t]
            if s_name is None:
                sector_room[t] = float("inf")
            else:
                in_sector = absw[sec[absw.index] == s_name]
                sector_room[t] = max(cap - totals.get(s_name, 0.0), 0.0) * absw[t] / in_sector.sum()
        room = pd.concat([name_room, sector_room], axis=1).min(axis=1)
        add = (excess * absw / absw.sum()).clip(upper=room)
        sign = w[receivers].apply(lambda x: 1.0 if x >= 0 else -1.0)
        w[receivers] = w[receivers] + add * sign
    return w
