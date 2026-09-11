"""NYSE trading calendar, enough to answer "is today the last trading day of
the week/month/quarter?" so the live rebalancer fires on the same days the
backtester did.

Unscheduled closures (days of mourning, disasters) are not known in advance;
the worst case is a rebalance shifted by one day.
"""
from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        # NYSE does not close on Dec 31 when Jan 1 falls on a Saturday
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday, start_date="2022-01-01"),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


_BDAY = CustomBusinessDay(calendar=NYSEHolidayCalendar())


def is_trading_day(date) -> bool:
    d = pd.Timestamp(date).normalize()
    return bool(_BDAY.is_on_offset(d))


def next_trading_day(date) -> pd.Timestamp:
    """First trading day strictly after `date`."""
    return (pd.Timestamp(date).normalize() + _BDAY).normalize()


def previous_trading_day(date) -> pd.Timestamp:
    """Last trading day strictly before `date`."""
    return (pd.Timestamp(date).normalize() - _BDAY).normalize()


def trading_days(start, end) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end, freq=_BDAY)


def _period_key(d: pd.Timestamp, schedule: str) -> int:
    if schedule == "weekly":
        iso = d.isocalendar()
        return iso.year * 100 + iso.week
    if schedule == "monthly":
        return d.year * 100 + d.month
    if schedule == "quarterly":
        return d.year * 10 + (d.month - 1) // 3
    raise ValueError(f"unknown schedule {schedule!r}")


def is_rebalance_day(date, schedule: str | int, bar_dates: pd.DatetimeIndex | None = None) -> bool:
    """True if a strategy on `schedule` would decide at the close of `date`.

    For 'weekly'/'monthly'/'quarterly' this asks whether the next NYSE
    trading day falls in a new period. For an integer schedule (every N bars)
    `bar_dates` (the panel's dates) is required and the answer mirrors the
    backtester's bar-count mask.
    """
    d = pd.Timestamp(date).normalize()
    if isinstance(schedule, int):
        if bar_dates is None:
            raise ValueError("integer schedules need bar_dates")
        from ..backtest.schedule import rebalance_mask
        mask = rebalance_mask(bar_dates, schedule)
        loc = bar_dates.get_indexer([d])[0]
        return bool(mask[loc]) if loc >= 0 else False
    s = schedule.lower()
    if s == "daily":
        return True
    return _period_key(d, s) != _period_key(next_trading_day(d), s)
