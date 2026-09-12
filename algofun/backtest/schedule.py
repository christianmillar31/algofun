"""Rebalance schedules over a trading-day index."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance_mask(dates: pd.DatetimeIndex, schedule: str | int) -> np.ndarray:
    """Boolean mask: True on bars where the strategy is asked for new targets.

    schedule: 'daily', 'weekly' (last trading day of each ISO week),
              'monthly' (last trading day of each month), 'quarterly',
              or an int N for every N bars.
    """
    n = len(dates)
    if isinstance(schedule, int):
        if schedule <= 0:
            raise ValueError("integer schedule must be >= 1")
        mask = np.zeros(n, dtype=bool)
        mask[::schedule] = True
        return mask
    s = schedule.lower()
    if s == "daily":
        return np.ones(n, dtype=bool)
    if s == "weekly":
        key = dates.isocalendar().year.to_numpy() * 100 + dates.isocalendar().week.to_numpy()
    elif s == "monthly":
        key = dates.year.to_numpy() * 100 + dates.month.to_numpy()
    elif s == "quarterly":
        key = dates.year.to_numpy() * 10 + ((dates.month.to_numpy() - 1) // 3)
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    # last bar of each period: next bar has a different key (or it's the final bar)
    mask = np.empty(n, dtype=bool)
    mask[:-1] = key[1:] != key[:-1]
    mask[-1] = True
    return mask
