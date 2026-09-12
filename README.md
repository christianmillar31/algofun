# algofun

An honest algorithmic trading lab. Data in, strategy, backtest with real costs,
walk-forward parameter selection, then the *same* strategy object drives a paper
or live rebalancer.

```
data -> strategy -> backtest -> risk -> execution
```

The backtester is the product. The engine is built so a strategy **cannot see
the future**: it decides on today's close through a `MarketView` that only
exposes bars up to today, and it gets filled at *tomorrow's open* after
slippage, spread and commission. The tests in `tests/test_no_lookahead.py`
enforce that.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,plot]"

# 1. get data (S&P 500 constituents + broad ETFs, full daily history from Yahoo)
algofun fetch --universe sp500+etfs --start 1990-01-01

# 2. see what is available
algofun list-strategies

# 3. backtest
algofun backtest --strategy momentum --min-bars 500 --plot --out runs/momentum
algofun backtest --strategy sma_crossover --params fast=20,slow=100 --costs pessimistic

# 4. pick parameters honestly (rolling train/test)
algofun walkforward --strategy sma_crossover --train-bars 504 --test-bars 126

# 5. paper trade with fake money (dry run prints the orders it would send)
algofun rebalance --strategy momentum --broker paper
algofun rebalance --strategy momentum --broker paper --execute
```

No network from where you are? Import any long-format CSV
(`date,open,high,low,close,volume,Name`), e.g. the public
[`all_stocks_5yr.csv`](https://github.com/plotly/datasets/blob/master/all_stocks_5yr.csv)
(505 S&P names, 2013 to 2018):

```bash
algofun import-csv all_stocks_5yr.csv
algofun backtest --strategy momentum --params market_filter=None --benchmark "" --min-bars 1000
```

## What is in the box

| Module | What it does |
| --- | --- |
| `algofun/data` | Universes (S&P 500 list, ETFs), Yahoo + Stooq sources with a fallback chain, incremental parquet cache, aligned OHLCV `Panel`, bulk CSV import |
| `algofun/backtest` | Event-driven daily engine, `CostModel` presets (`zero`, `retail`, `ibkr`, `pessimistic`), metrics (CAGR, Sharpe, Sortino, max drawdown, Calmar, turnover, win rate, beta/alpha vs benchmark), rebalance schedules, walk-forward optimisation |
| `algofun/strategies` | `buy_and_hold`, `sma_crossover`, `momentum` (cross-sectional, with market filter), `mean_reversion` (z-score dips in uptrends), `sentiment` (news tone on the Loughran-McDonald lexicon). Each declares its own `warmup`, `rebalance` and `param_grid` |
| `algofun/text` | Loughran-McDonald finance lexicon (cached download), tokenizer with the negation rule, per-article tone and negativity, and the mapping from article timestamps to the NYSE session at whose close they are first known |
| `algofun/risk` | Equal-weight, inverse-vol and vol-target sizing; hard `RiskLimits` (per-name cap, gross cap, no shorts by default) applied after every strategy decision |
| `algofun/broker` | `Broker` interface, in-memory `PaperBroker` (persists to JSON, uses the same `CostModel`), `AlpacaBroker` adapter |
| `algofun/live` | `plan_rebalance` diffs target weights against broker holdings with the same no-trade band as the backtester; `execute_plan` submits sells then buys and appends to `runs/rebalance_log.jsonl` |

## Writing a strategy

```python
import pandas as pd
from algofun.strategies.base import Strategy
from algofun.risk import equal_weight

class Breakout(Strategy):
    name = "breakout"
    defaults = {"lookback": 55, "max_names": 20}
    param_grid = {"lookback": [20, 55, 100]}

    def configure(self):
        self.warmup = self.lookback + 1
        self.rebalance = "weekly"

    def target_weights(self, view):            # view only knows bars <= today
        px = view.history("close", self.lookback + 1)
        at_high = (px.iloc[-1] >= px.iloc[:-1].max()) & view.tradable()
        return equal_weight(list(at_high[at_high].index[: self.max_names]))
```

Register it in `algofun/strategies/__init__.py` and it works in every command.
Ask for bounded history (`view.history(field, lookback)`); it is faster and it
documents what the strategy needs.

## News sentiment (Loughran-McDonald)

If you do not believe in price patterns, this is the other door, and it goes
through the same honesty checks. Nothing here needs a language model, on
purpose: a word list published in 2011 cannot know how a 2018 headline turned
out, whereas any model trained on the internet does, which makes a backtest
of model-scored history worthless.

- `algofun fetch-news --start 2023-01-01 --max-minutes 120` pulls Alpaca's
  (Benzinga) news, month by month, into `data/cache/news/`. Free on the basic
  data plan, history from 2015, 50 articles a page and 200 requests a minute,
  so the first fetch of a few years takes hours. It stops cleanly at the time
  budget and the next run resumes; finished months are never re-fetched.
- Every article is scored with the Loughran-McDonald dictionary (2,355
  negative and 354 positive finance words, with "not profitable" counted as
  negative) and assigned to the first NYSE session whose close it precedes: a
  story at 15:59 New York time on Tuesday belongs to Tuesday, at 16:01 to
  Wednesday, on Saturday to Monday. The decision at Tuesday's close may only
  read Tuesday's news, and is filled at Wednesday's open like everything else.
- `algofun backtest --strategy sentiment --news --pit --start 2023-03-01 --stress`
  ranks names by article-weighted mean tone over the last `lookback` sessions
  (or by minus negativity with `score=neg`, the half of the dictionary that
  Loughran & McDonald found actually predicts anything), requires
  `min_articles` in the window, and holds the top N with the same sizing,
  sector cap, vol target and index filter as momentum.
- `algofun sentiment --top 15` prints today's ranking from the cache;
  `algofun rebalance ... --strategy sentiment --news` trades it, paper first.
- The "Research (news sentiment)" workflow does all of the above on a GitHub
  runner with the paper keys and uploads the reports, momentum on the same
  window beside it for comparison.

What to expect, honestly: the literature finds negative tone predicts
short-horizon returns with small magnitude, mostly in small caps, and it
decays within days. Headline sentiment on large caps is arbitraged in
seconds by machines that read faster than a daily-bar system. Slower
information (earnings surprises, transcript tone) has better evidence. So
the numbers may well say "no edge after costs"; that is a result, not a bug.

### First real-news result (2026-09-12)

945,915 Benzinga articles, January 2023 to September 2026, scored once and
cached. Both strategies on the same window (March 2023 to September 2026),
point-in-time S&P 500 membership, retail costs:

| Strategy | CAGR | Sharpe | Max DD | Turnover | 2x costs | 3x costs |
| --- | --- | --- | --- | --- | --- | --- |
| Sentiment (tone, 10 sessions, top 20, weekly) | -1.6% | -0.09 | -15.0% | 64x | -4.5% | -6.3% |
| Momentum (12-1, top 10, monthly) | 10.3% | 0.71 | -11.6% | 5.6x | 9.9% | 9.6% |
| SPY buy and hold | 22.2% | 1.42 | -18.8% | | | |

Walk-forward on sentiment (six configurations per fold, seven folds, 42
trials counted): out-of-sample Sharpe -0.17 +/- 0.76, walk-forward
efficiency -0.22, in-sample deflated Sharpe 0.63. Headline tone on large
caps, held for a week, has no edge here, and it turns over the book 64
times a year paying for the privilege. That is the literature's expectation
and the pipeline reported it without flinching. Note also that both
strategies trailed a raw S&P 500 that compounded at 22% a year over this
window: the vol target and drawdown budget cap exposure at the cost of
upside in a straight-up market, which is the trade they are meant to make.

The dictionary itself is published by the University of Notre Dame's
Software Repository for Accounting and Finance and is free for personal and
academic use (commercial use needs their licence); we read the copy bundled
in the MIT-licensed `pysentiment2` package.

## Going live (read this part)

1. **Backtest with `--costs pessimistic` too.** If the edge disappears, it was
   never there.
2. **Walk-forward, not in-sample.** The stitched out-of-sample curve from
   `algofun walkforward` is the only number that has not seen its own future.
3. **Paper first, for months.** Create a free paper account at Alpaca, then:
   ```bash
   pip install -e ".[alpaca]"
   export ALPACA_API_KEY=... ALPACA_SECRET_KEY=... ALPACA_PAPER=true
   algofun status --broker alpaca
   algofun rebalance --strategy momentum --broker alpaca --refresh            # dry run
   algofun rebalance --strategy momentum --broker alpaca --refresh --execute  # sends DAY orders
   ```
   Run it once a day after the close (cron, GitHub Actions, whatever). Market
   DAY orders placed after hours fill at the next open, which is precisely
   what the backtester assumes.
4. **Live needs a double opt-in.** `ALPACA_PAPER=false` *and* `--live` on the
   command line, or the CLI refuses. Keep `--max-weight` and `--max-gross`
   conservative.
5. **Every run passes through a kill switch first.** Before any order is
   sent, `algofun rebalance` checks that the latest bar is fresh, that the
   account is not down more than 3% on the day, that no single order exceeds
   30% of equity and the plan has fewer than 60 orders, and that the broker's
   positions match the snapshot written after the previous run. Any failure
   blocks the run and exits non-zero so the scheduled job turns red. Tune with
   `--max-daily-loss-pct`, `--max-orders`, `--max-order-pct`,
   `--max-stale-sessions`; `--no-guards` disables everything and should never
   be in a scheduled job. `algofun flatten --broker alpaca --execute --yes`
   is the emergency exit: it cancels open orders and closes every position.
6. **Three more safety rails.** A drawdown budget halves exposure 10% below
   the equity peak and goes flat at 20%, in the backtest and live alike
   (`--dd-halve`, `--dd-flat`, `--no-drawdown-control`). `algofun gate`
   reports whether the paper record has at least 100 filled orders and 20
   consecutive clean runs; `--live` refuses to trade until it does unless you
   pass `--override-gate`. `algofun shortfall` compares every realised fill
   with the price the plan was sized at, which is the only number that says
   whether the backtest's cost model was honest.
7. **The rebalancer respects the strategy's schedule.** A monthly strategy
   only trades on the last NYSE trading day of the month, exactly like the
   backtest. Pass `--force` for the first deployment or a manual reset.

### Run it from GitHub Actions (no server needed)

`.github/workflows/rebalance.yml` runs every weekday at 22:17 UTC (after the
close), refreshes the bar cache, and rebalances the **paper** account.

1. In Alpaca, switch the dashboard to *Paper* and generate an API key pair.
2. In this repo: Settings -> Secrets and variables -> Actions -> add
   `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`.
3. Actions -> "Rebalance (Alpaca)" -> Run workflow, with *force* on for the
   first run so it takes its initial positions. Watch the log. From then on
   the schedule does the rest.

Going live later means adding `ALPACA_LIVE_API_KEY` / `ALPACA_LIVE_SECRET_KEY`
and choosing `mode=live` in a manual run. Scheduled runs never touch live keys.
Never paste keys into chat, issues, or commits.

### Why Alpaca and not Robinhood

Robinhood has no official API for retail stock trading. The third-party
libraries that exist drive its private endpoints, violate its terms of
service, and get accounts locked. Alpaca is also commission-free, supports
fractional shares, has a real paper-trading environment and an official SDK.
Same "free trades", much lower chance of losing your account. The `Broker`
interface is small if you ever want to add another one.

## Known limits, on purpose

- Daily bars only. Intraday is a different engine and a different data bill.
- Long-only by default. Shorting is supported in accounting (`allow_short=True`)
  but borrow fees and locate availability are not modelled.
- The current S&P 500 list is today's survivors. Backtests on it are an upper
  bound. Fetch a wider universe (`--universe sp500+etfs` plus your own lists)
  and prefer strategies that don't depend on a fixed constituent list.
- Yahoo data is free and mostly fine for daily bars; it is not audited.
  Dividends are folded into adjusted prices (total return), which is what you
  want for backtests.

## Tests

```bash
pytest
```

The suite covers: no lookahead (view boundaries, fill timing, an explicit
cheat attempt), cost model, cash/position accounting identity, share
rounding, rebalance schedules, risk limits, each strategy on synthetic data,
walk-forward fold construction, the paper broker, and the order planner.
