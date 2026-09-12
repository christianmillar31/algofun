import numpy as np
import pandas as pd

from algofun.backtest import BacktestConfig, MarketView, run_backtest
from algofun.data.news import NewsStore, normalize_news
from algofun.strategies import get_strategy
from algofun.text import (
    Lexicon,
    build_daily_features,
    daily_sentiment,
    score_articles,
    session_dates,
)

from .conftest import make_panel

LEX = Lexicon(positive=frozenset({"BEAT", "RECORD", "GROWTH", "UPGRADE"}),
              negative=frozenset({"MISS", "LAWSUIT", "DECLINE", "RECALL", "LOSS"}))


def test_session_dates_follow_the_new_york_close_and_calendar():
    ts = pd.Series(pd.to_datetime([
        "2025-03-11 19:59:00Z",   # Tue 15:59 EDT  -> Tue
        "2025-03-11 20:00:00Z",   # Tue 16:00      -> Tue (known at the close)
        "2025-03-11 20:00:01Z",   # Tue 16:00:01   -> Wed
        "2025-03-15 12:00:00Z",   # Saturday       -> Monday 17th
        "2025-04-18 14:00:00Z",   # Good Friday    -> Monday 21st
        "2025-01-10 21:30:00Z",   # Fri 16:30 EST  -> Monday 13th
    ]))
    got = [d.strftime("%Y-%m-%d") for d in session_dates(ts)]
    assert got == ["2025-03-11", "2025-03-11", "2025-03-12", "2025-03-17", "2025-04-21", "2025-01-13"]


def _news(rows):
    return normalize_news(pd.DataFrame([
        {"id": i, "created_at": ts, "updated_at": ts, "headline": h, "summary": "", "source": "x", "url": "",
         "symbols": syms} for i, (ts, syms, h) in enumerate(rows, start=1)]))


def test_daily_features_align_to_sessions_and_tickers():
    news = _news([
        ("2024-06-03T14:00:00Z", ["T0"], "record growth, analyst upgrade"),      # Mon 10:00 -> Mon
        ("2024-06-03T20:30:00Z", ["T0", "T1"], "lawsuit and recall"),           # Mon 16:30 -> Tue
        ("2024-06-04T15:00:00Z", ["T1"], "profit miss, sales decline"),          # Tue -> Tue
        ("2024-06-04T15:00:00Z", ["ZZZ"], "record beat"),                        # not in universe
    ])
    scored = score_articles(news, LEX)
    dates = pd.bdate_range("2024-06-03", periods=3)
    f = daily_sentiment(scored, dates, ["T0", "T1", "T2"])
    assert f["news_count"].loc["2024-06-03"].tolist() == [1.0, 0.0, 0.0]
    assert f["news_count"].loc["2024-06-04"].tolist() == [1.0, 2.0, 0.0]
    assert f["news_tone_sum"].loc["2024-06-03", "T0"] == 1.0            # 3 positive, 0 negative
    assert f["news_tone_sum"].loc["2024-06-04", "T0"] == -1.0           # the after-close lawsuit lands Tuesday
    assert f["news_tone_sum"].loc["2024-06-04", "T1"] == -2.0
    assert f["news_count"].loc["2024-06-05"].sum() == 0.0
    assert f["news_neg_sum"].loc["2024-06-04", "T1"] > 0


def test_view_feature_never_exposes_future():
    p = make_panel(n_tickers=3, n_days=30)
    feats = {"news_count": pd.DataFrame(np.arange(90, dtype=float).reshape(30, 3), index=p.dates, columns=p.tickers)}
    p = p.with_features(feats)
    for i in range(len(p)):
        v = MarketView(p, i)
        f = v.feature("news_count", 5)
        assert f.index.max() == p.dates[i] and len(f) == min(5, i + 1)
    assert v.features == ["news_count"] and v.has_feature("news_count")
    # slicing and selecting keep the features aligned
    q = p.slice(p.dates[10], p.dates[19]).select(["T2", "T0"])
    assert q.feature("news_count").shape == (10, 2) and list(q.feature("news_count").columns) == ["T2", "T0"]


def test_sentiment_strategy_ranks_by_tone_and_ignores_after_close_news():
    p = make_panel(n_tickers=4, n_days=120, seed=2)
    last = p.dates[-1]
    stamp = lambda d, hh: f"{d:%Y-%m-%d}T{hh}:00Z"  # noqa: E731
    news = _news([
        (stamp(p.dates[-3], "14:00"), ["T0"], "record growth"),
        (stamp(p.dates[-2], "14:00"), ["T0"], "upgrade"),
        (stamp(p.dates[-2], "14:00"), ["T1"], "lawsuit loss"),
        (stamp(p.dates[-1], "14:00"), ["T1"], "recall"),
        (stamp(last, "23:00"), ["T2"], "record record record beat beat"),   # after the last close: unusable
        (stamp(last, "23:00"), ["T2"], "growth growth upgrade"),
    ])
    feats = daily_sentiment(score_articles(news, LEX), p.dates, p.tickers)
    assert feats["news_count"]["T2"].sum() == 0.0   # never lands on a bar in this panel
    p = p.with_features(feats)
    s = get_strategy("sentiment", lookback=5, min_articles=2, top_n=1, market_filter=None, vol_target=0,
                     sizing="equal", max_per_sector=0)
    w = s.target_weights(MarketView(p, len(p) - 1))
    assert list(w.index) == ["T0"]
    s2 = get_strategy("sentiment", lookback=5, min_articles=2, top_n=2, score="neg", market_filter=None,
                      vol_target=0, sizing="equal", max_per_sector=0)
    w2 = s2.target_weights(MarketView(p, len(p) - 1))
    assert w2.sort_values(ascending=False).index[0] == "T0"     # zero negativity beats T1's
    # a panel without features stays in cash instead of crashing
    res = run_backtest(make_panel(n_tickers=3, n_days=80), get_strategy("sentiment", market_filter=None),
                       BacktestConfig(benchmark=None))
    assert np.isclose(res.equity.iloc[-1], res.equity.iloc[0])


def test_build_daily_features_scores_each_month_once(tmp_path):
    store = NewsStore(tmp_path)
    news = _news([("2024-06-03T14:00:00Z", ["T0"], "record growth"),
                  ("2024-06-04T14:00:00Z", ["T1"], "lawsuit")])
    store.save_month(pd.Timestamp("2024-06-01", tz="UTC"), news)
    dates = pd.bdate_range("2024-06-03", periods=5)
    feats, info = build_daily_features(store, LEX, dates, ["T0", "T1"])
    assert info["articles"] == 2 and feats["news_count"].to_numpy().sum() == 2.0
    cached = list(store.news_dir.glob("scored_lm_2024-06.parquet"))
    assert len(cached) == 1
    feats2, _ = build_daily_features(store, LEX, dates, ["T0", "T1"])   # served from the score cache
    pd.testing.assert_frame_equal(feats["news_tone_sum"], feats2["news_tone_sum"])
