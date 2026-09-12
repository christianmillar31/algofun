import pandas as pd

from algofun.live import is_rebalance_day, is_trading_day, next_trading_day, trading_days
from tests.conftest import make_panel


def test_known_holidays_and_observance():
    assert not is_trading_day("2024-03-29")   # Good Friday
    assert not is_trading_day("2024-07-04")
    assert not is_trading_day("2024-11-28")   # Thanksgiving
    assert is_trading_day("2024-11-29")       # Black Friday half day is a trading day
    assert not is_trading_day("2023-01-02")   # New Year's observed (Jan 1 was Sunday)
    assert is_trading_day("2021-12-31")       # Jan 1 2022 was Saturday: NYSE open on the 31st
    assert not is_trading_day("2024-06-19")   # Juneteenth
    assert is_trading_day("2021-06-18")       # Juneteenth not observed by NYSE before 2022
    assert not is_trading_day("2024-09-14")   # Saturday


def test_next_trading_day_and_range():
    assert next_trading_day("2024-03-28") == pd.Timestamp("2024-04-01")
    assert next_trading_day("2024-07-03") == pd.Timestamp("2024-07-05")
    assert next_trading_day("2021-12-30") == pd.Timestamp("2021-12-31")
    days = trading_days("2024-11-25", "2024-11-29")
    assert list(days.strftime("%m-%d")) == ["11-25", "11-26", "11-27", "11-29"]


def test_is_rebalance_day_period_ends():
    assert is_rebalance_day("2024-03-28", "monthly")        # last trading day of March (Good Friday next)
    assert not is_rebalance_day("2024-03-27", "monthly")
    assert is_rebalance_day("2024-12-31", "monthly")
    assert is_rebalance_day("2022-12-30", "monthly")        # Jan 2 2023 is a holiday
    assert is_rebalance_day("2024-07-03", "weekly") is False  # Friday 7/5 is still this week
    assert is_rebalance_day("2024-07-05", "weekly")
    assert is_rebalance_day("2024-06-28", "quarterly")
    assert not is_rebalance_day("2024-05-31", "quarterly")
    assert is_rebalance_day("2024-05-31", "daily")


def test_is_rebalance_day_int_schedule_mirrors_backtester():
    p = make_panel(n_tickers=1, n_days=10)
    assert is_rebalance_day(p.dates[0], 5, p.dates)
    assert not is_rebalance_day(p.dates[1], 5, p.dates)
    assert is_rebalance_day(p.dates[5], 5, p.dates)
    assert not is_rebalance_day(pd.Timestamp("1999-01-01"), 5, p.dates)
