import pandas as pd
import pytest

from algofun.data.news import AlpacaNewsSource, NewsStore, month_starts, normalize_news


class FakeNewsClient:
    """Pages of 2 through a fixed article list, honouring start/end and page_token."""

    def __init__(self, articles, page_size=2):
        self.articles = sorted(articles, key=lambda a: a["created_at"])
        self.page_size = page_size
        self.calls = []

    def get(self, path, data=None):
        assert path == "/news"
        self.calls.append(dict(data))
        start, end = pd.Timestamp(data["start"]), pd.Timestamp(data["end"])
        rows = [a for a in self.articles if start <= pd.Timestamp(a["created_at"]) <= end]
        if data.get("symbols"):
            want = set(data["symbols"].split(","))
            rows = [a for a in rows if want & set(a["symbols"])]
        offset = int(data.get("page_token") or 0)
        page = rows[offset:offset + self.page_size]
        nxt = offset + self.page_size
        return {"news": page, "next_page_token": str(nxt) if nxt < len(rows) else None}


def art(i, ts, symbols, headline="h", updated=None):
    return {"id": i, "created_at": ts, "updated_at": updated or ts, "headline": headline,
            "summary": "s", "source": "benzinga", "url": "u", "symbols": symbols}


ARTICLES = [
    art(1, "2024-01-03T14:00:00Z", ["AAPL"]),
    art(2, "2024-01-03T21:00:00Z", ["BRK.B", "AAPL"]),
    art(3, "2024-01-20T10:00:00Z", ["MSFT"]),
    art(4, "2024-02-02T12:00:00Z", ["AAPL"]),
    art(5, "2024-02-15T12:00:00Z", ["TSLA"]),
]


def test_fetch_pages_and_maps_symbols():
    src = AlpacaNewsSource(client=FakeNewsClient(ARTICLES), min_interval=0)
    df, complete = src.fetch("2024-01-01", "2024-01-31")
    assert complete and len(df) == 3 and src.requests_made == 2
    assert df.loc[df["id"] == 2, "symbols"].iloc[0] == ["BRK-B", "AAPL"]
    assert str(df["created_at"].dt.tz) == "UTC"


def test_fetch_stops_at_deadline_and_reports_partial():
    src = AlpacaNewsSource(client=FakeNewsClient(ARTICLES, page_size=1), min_interval=0)
    df, complete = src.fetch("2024-01-01", "2024-02-28", deadline=-1.0)   # already expired
    assert not complete and len(df) == 0


def test_normalize_keeps_latest_update_per_id():
    a = art(9, "2024-03-01T10:00:00Z", ["X"], headline="old", updated="2024-03-01T10:00:00Z")
    b = art(9, "2024-03-01T10:00:00Z", ["X"], headline="new", updated="2024-03-01T11:00:00Z")
    df = normalize_news(pd.DataFrame([b, a]))
    assert len(df) == 1 and df["headline"].iloc[0] == "new"


def test_month_starts():
    ms = month_starts("2024-11-15", "2025-01-03")
    assert [f"{m:%Y-%m}" for m in ms] == ["2024-11", "2024-12", "2025-01"]


def test_store_update_is_monthly_and_resumable(tmp_path):
    client = FakeNewsClient(ARTICLES)
    store = NewsStore(tmp_path)
    counts = store.update("2024-01-01", "2024-02-28", source=AlpacaNewsSource(client=client, min_interval=0))
    assert counts == {"2024-01": 3, "2024-02": 2}
    assert sorted(p.name for p in store.news_dir.glob("*.parquet")) == ["2024-01.parquet", "2024-02.parquet"]
    n_calls = len(client.calls)
    # both months are long settled: a second update touches nothing
    counts2 = store.update("2024-01-01", "2024-02-28", source=AlpacaNewsSource(client=client, min_interval=0))
    assert counts2 == counts and len(client.calls) == n_calls

    df = store.load("2024-01-01", "2024-02-10")
    assert df["id"].tolist() == [1, 2, 3, 4]
    assert store.load(symbols=["TSLA"])["id"].tolist() == [5]
    cov = store.coverage()
    assert cov["articles"].tolist() == [3, 2]


def test_store_update_time_budget_saves_finished_months(tmp_path):
    client = FakeNewsClient(ARTICLES, page_size=1)
    store = NewsStore(tmp_path)
    counts = store.update("2024-01-01", "2024-02-28", max_minutes=0.0,
                          source=AlpacaNewsSource(client=client, min_interval=0))
    # zero budget: nothing fetched, nothing written, no crash
    assert counts == {} and not list(store.news_dir.glob("*.parquet"))


def test_source_requires_keys(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError):
        AlpacaNewsSource().client  # noqa: B018 - property access is the test
