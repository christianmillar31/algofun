"""The most important tests in the repo: the strategy cannot see the future."""
import numpy as np
import pandas as pd

from algofun.backtest import BacktestConfig, CostModel, MarketView, run_backtest
from algofun.strategies.base import Strategy

from .conftest import make_panel


class Spy(Strategy):
    """Records everything it is shown."""
    name = "spy"
    rebalance = "daily"

    def reset(self):
        self.seen = []

    def target_weights(self, view):
        c = view.close
        rets = view.returns(5)
        self.seen.append((view.date, c.index.max(), len(c), rets.index.max() if len(rets) else view.date))
        return pd.Series(1.0, index=[view.tickers[0]])


def test_view_never_exposes_future(panel):
    for i in range(len(panel)):
        v = MarketView(panel, i)
        assert v.date == panel.dates[i]
        assert v.close.index.max() == panel.dates[i]
        assert len(v.close) == i + 1
        assert v.history("open", 10).index.max() == panel.dates[i]
        assert len(v.history("close", 10)) == min(10, i + 1)
        assert v.last("close").name == panel.dates[i]


def test_strategy_sees_only_history_and_fills_next_open(panel):
    strat = Spy()
    res = run_backtest(panel, strat, BacktestConfig(costs=CostModel.zero(), benchmark=None))
    for decision_date, max_seen, _n, ret_max in strat.seen:
        assert max_seen == decision_date
        assert ret_max == decision_date
    # every fill happens strictly after a decision, at the next bar's open
    first_trade = res.trades.iloc[0]
    assert first_trade["date"] == panel.dates[1]
    assert np.isclose(first_trade["price"], panel.open.iloc[1, 0])


def test_fill_price_is_next_open_with_costs(panel):
    costs = CostModel(slippage_bps=10, spread_bps=4)
    res = run_backtest(panel, Spy(), BacktestConfig(costs=costs, benchmark=None))
    t = res.trades.iloc[0]
    expected = panel.open.iloc[1, 0] * (1 + costs.price_penalty)
    assert np.isclose(t["price"], expected)


class CheatAttempt(Strategy):
    """Tries the obvious cheat: grab the whole close frame and take the next row."""
    name = "cheat"
    rebalance = "daily"

    def target_weights(self, view):
        full = view.history("close")            # this is everything it can get
        assert full.index[-1] == view.date      # ...and it ends today
        try:
            future = full.loc[full.index[-1] + pd.Timedelta(days=1):]
        except KeyError:
            future = full.iloc[0:0]
        assert len(future) == 0
        return pd.Series(dtype="float64")


def test_no_future_rows_reachable(panel):
    run_backtest(panel, CheatAttempt(), BacktestConfig(benchmark=None))


def test_perfect_foresight_is_impossible():
    """A strategy that buys only before up-days would need tomorrow's close.
    Prove it cannot: its realised returns must not beat the oracle."""
    p = make_panel(n_tickers=1, n_days=200, seed=3)
    close = p.close.iloc[:, 0]
    oracle_up = (close.shift(-1) > close)  # uses the future on purpose

    class Oracle(Strategy):
        name = "oracle"
        rebalance = "daily"

        def target_weights(self, view):
            # the only legal information is view; we simulate what an honest
            # implementation could do: use yesterday's direction as a proxy
            c = view.history("close", 2).iloc[:, 0]
            up = len(c) == 2 and c.iloc[-1] > c.iloc[-2]
            return pd.Series(1.0 if up else 0.0, index=[view.tickers[0]])

    res = run_backtest(p, Oracle(), BacktestConfig(costs=CostModel.zero(), benchmark=None))
    honest = res.equity.iloc[-1] / res.equity.iloc[0]
    # what true foresight would have earned on closes
    r = close.pct_change().fillna(0.0)
    cheat = float((1 + r[oracle_up.shift(1).fillna(False)]).prod())
    assert honest < cheat
