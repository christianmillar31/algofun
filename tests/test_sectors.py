import numpy as np
import pandas as pd
import pytest

from algofun.backtest import BacktestConfig, MarketView, run_backtest
from algofun.data import Panel, sector_map
from algofun.risk import RiskLimits
from algofun.risk.limits import cap_sectors
from algofun.strategies import Momentum

from .conftest import make_panel


def test_cap_sectors_redistributes_pro_rata():
    w = pd.Series({"A": 0.3, "B": 0.3, "C": 0.2, "D": 0.2})
    sectors = {"A": "Tech", "B": "Tech", "C": "Health", "D": "Energy"}
    out = cap_sectors(w, sectors, cap=0.40, max_weight=1.0)
    assert out[["A", "B"]].sum() == pytest.approx(0.40)
    assert out.sum() == pytest.approx(1.0)                 # excess moved, not lost
    assert out["C"] == pytest.approx(out["D"])             # pro rata across the free names
    # nowhere to put it -> becomes cash
    all_tech = cap_sectors(pd.Series({"A": 0.5, "B": 0.5}), {"A": "Tech", "B": "Tech"}, cap=0.30)
    assert all_tech.sum() == pytest.approx(0.30)
    # unknown sector names are never capped together
    unk = cap_sectors(pd.Series({"X": 0.5, "Y": 0.5}), {}, cap=0.30)
    assert unk.sum() == pytest.approx(1.0)
    # per-name max_weight bounds the redistribution
    bounded = cap_sectors(pd.Series({"A": 0.6, "B": 0.4}), {"A": "Tech", "B": "Health"}, cap=0.30, max_weight=0.45)
    assert bounded["B"] <= 0.45 + 1e-9 and bounded["A"] == pytest.approx(0.30)


def test_risk_limits_applies_sector_cap_only_with_map():
    w = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    lim = RiskLimits(max_weight=0.25, max_sector_weight=0.30)
    assert lim.apply(w).equals(w.astype("float64"))           # no map -> untouched
    out = lim.apply(w, sectors={"A": "T", "B": "T", "C": "T", "D": "H"})
    assert out[["A", "B", "C"]].sum() == pytest.approx(0.30) and out["D"] == pytest.approx(0.25)


def test_panel_and_view_carry_sectors():
    p = make_panel(n_tickers=3, n_days=50, tickers=["A", "B", "C"])
    p2 = Panel(**{f: p.field(f) for f in ("open", "high", "low", "close", "volume")}, sectors={"A": "Tech"})
    assert p2.slice(None, None).sectors == {"A": "Tech"} and p2.select(["A"]).sectors == {"A": "Tech"}
    v = MarketView(p2, 10)
    assert v.sector_of("A") == "Tech" and v.sector_of("B") == "Unknown" and v.sectors == {"A": "Tech"}


def test_momentum_per_sector_cap_and_vol_target():
    n = 320
    tick = [f"T{i}" for i in range(8)] + ["SPY"]
    p = make_panel(n_tickers=9, n_days=n, seed=4, tickers=tick)
    # make T0..T4 the strongest, all "Tech"; T5..T7 weaker, other sectors
    for i, t in enumerate(tick[:8]):
        mult = 6.0 - 0.5 * i if i < 5 else 1.3 - 0.05 * (i - 5)   # Tech names trend far harder than the rest
        p.close[t] = p.close[t] * np.geomspace(1, mult, n)
    sectors = {t: "Tech" for t in tick[:5]} | {"T5": "Health", "T6": "Energy", "T7": "Staples", "SPY": "ETF"}
    p.sectors = sectors
    s = Momentum(lookback=126, skip=21, top_n=5, max_per_sector=2, market_filter=None, vol_target=0.0)
    w = s.target_weights(MarketView(p, n - 1))
    assert sum(sectors[t] == "Tech" for t in w.index) == 2 and len(w) == 5
    assert w.sum() == pytest.approx(1.0)                      # inverse-vol, fully invested
    s2 = Momentum(lookback=126, skip=21, top_n=5, max_per_sector=0, market_filter=None, vol_target=0.05)
    w2 = s2.target_weights(MarketView(p, n - 1))
    assert sum(sectors[t] == "Tech" for t in w2.index) == 5   # cap off -> all Tech
    assert w2.sum() < 1.0                                     # 5% vol target scales the book down
    res = run_backtest(p, Momentum(lookback=126, skip=21, top_n=3, market_filter=None),
                       BacktestConfig(benchmark=None, limits=RiskLimits(max_sector_weight=0.5)))
    assert res.equity.iloc[-1] > 0


def test_sector_map_from_csv(tmp_path):
    (tmp_path / "sp500_constituents.csv").write_text("Symbol,Security,GICS Sector\nBRK.B,Berkshire,Financials\nNVDA,Nvidia,Information Technology\n")
    m = sector_map(tmp_path)
    assert m["BRK-B"] == "Financials" and m["NVDA"] == "Information Technology" and m["SPY"] == "ETF"
