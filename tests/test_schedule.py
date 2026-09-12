import pandas as pd
import pytest

from algofun.backtest import rebalance_mask


def test_monthly_marks_last_trading_day():
    dates = pd.bdate_range("2021-01-01", "2021-04-30")
    mask = rebalance_mask(dates, "monthly")
    chosen = dates[mask]
    assert list(chosen.strftime("%Y-%m-%d")) == ["2021-01-29", "2021-02-26", "2021-03-31", "2021-04-30"]


def test_weekly_and_daily_and_int():
    dates = pd.bdate_range("2021-01-04", periods=15)  # three full weeks
    assert rebalance_mask(dates, "weekly").sum() == 3
    assert rebalance_mask(dates, "daily").all()
    m = rebalance_mask(dates, 5)
    assert m.sum() == 3 and m[0]
    assert rebalance_mask(dates, "quarterly").sum() == 1


def test_bad_schedule():
    dates = pd.bdate_range("2021-01-04", periods=5)
    with pytest.raises(ValueError):
        rebalance_mask(dates, "hourly")
    with pytest.raises(ValueError):
        rebalance_mask(dates, 0)
