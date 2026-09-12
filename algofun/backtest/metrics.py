"""Performance statistics from an equity curve."""
from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min()) if len(equity) else 0.0


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    if total <= 0:
        return -1.0
    return float(total ** (1.0 / years) - 1.0)


def sharpe(returns: pd.Series, periods_per_year: int = 252, rf_annual: float = 0.0) -> float:
    r = returns.dropna() - rf_annual / periods_per_year
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0


def sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    dd = np.sqrt((downside ** 2).sum() / len(r)) if len(downside) else 0.0
    return float(r.mean() / dd * np.sqrt(periods_per_year)) if dd > 0 else 0.0


def trade_stats(trades: pd.DataFrame | None) -> dict[str, float]:
    """Round-trip P&L per ticker using average-cost accounting."""
    out = {"n_trades": 0, "n_round_trips": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
           "profit_factor": 0.0, "total_commission": 0.0}
    if trades is None or len(trades) == 0:
        return out
    out["n_trades"] = len(trades)
    out["total_commission"] = float(trades["commission"].sum())
    pnls: list[float] = []
    for _, g in trades.sort_values("date").groupby("ticker", sort=False):
        qty = 0.0
        cost = 0.0  # total cost basis of the open position
        for row in g.itertuples(index=False):
            q = row.quantity
            if row.side == "buy":
                cost += q * row.price + row.commission
                qty += q
            else:
                if qty <= 0:
                    continue  # short leg; not tracked here
                q = min(q, qty)
                avg = cost / qty
                pnls.append(q * row.price - q * avg - row.commission)
                cost -= q * avg
                qty -= q
                if qty <= 1e-12:
                    qty, cost = 0.0, 0.0
    if not pnls:
        return out
    p = np.array(pnls)
    wins, losses = p[p > 0], p[p <= 0]
    out["n_round_trips"] = len(p)
    out["win_rate"] = float(len(wins) / len(p))
    out["avg_win"] = float(wins.mean()) if len(wins) else 0.0
    out["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
    gl = -losses.sum()
    out["profit_factor"] = float(wins.sum() / gl) if gl > 0 else float("inf") if len(wins) else 0.0
    return out


def compute_metrics(equity: pd.Series, trades: pd.DataFrame | None = None,
                    turnover: pd.Series | None = None, benchmark: pd.Series | None = None,
                    periods_per_year: int = 252) -> dict[str, float]:
    r = equity.pct_change().dropna()
    m: dict[str, float] = {
        "start_equity": float(equity.iloc[0]),
        "end_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": cagr(equity, periods_per_year),
        "annual_vol": float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else 0.0,
        "sharpe": sharpe(r, periods_per_year),
        "sortino": sortino(r, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "best_day": float(r.max()) if len(r) else 0.0,
        "worst_day": float(r.min()) if len(r) else 0.0,
        "n_days": len(equity),
    }
    m["calmar"] = m["cagr"] / abs(m["max_drawdown"]) if m["max_drawdown"] < 0 else 0.0
    from .stats import honesty_report
    h = honesty_report(r, periods_per_year)
    m["sharpe_se"] = h.sharpe_se_annual
    m["prob_sharpe_gt_zero"] = h.psr_zero
    m["min_track_years_95pct"] = h.min_track_years
    if turnover is not None and len(turnover):
        m["annual_turnover"] = float(turnover.mean() * periods_per_year)
    m.update(trade_stats(trades))
    if benchmark is not None and len(benchmark) == len(equity):
        br = benchmark.pct_change().dropna()
        m["benchmark_total_return"] = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1.0)
        m["benchmark_cagr"] = cagr(benchmark, periods_per_year)
        m["benchmark_sharpe"] = sharpe(br, periods_per_year)
        m["benchmark_max_drawdown"] = max_drawdown(benchmark)
        m["excess_cagr"] = m["cagr"] - m["benchmark_cagr"]
        aligned = pd.concat([r, br], axis=1).dropna()
        if len(aligned) > 2 and aligned.iloc[:, 1].var() > 0:
            beta = float(aligned.cov().iloc[0, 1] / aligned.iloc[:, 1].var())
            m["beta"] = beta
            m["alpha_annual"] = float((aligned.iloc[:, 0].mean() - beta * aligned.iloc[:, 1].mean())
                                      * periods_per_year)
    return m


_PCT = {"total_return", "cagr", "annual_vol", "max_drawdown", "best_day", "worst_day",
        "win_rate", "benchmark_total_return", "benchmark_cagr", "benchmark_max_drawdown",
        "excess_cagr", "alpha_annual", "annual_turnover"}
_MONEY = {"start_equity", "end_equity", "avg_win", "avg_loss", "total_commission"}
_YEARS = {"min_track_years_95pct", "min_backtest_years"}


def format_metrics(m: dict[str, float]) -> str:
    lines = []
    for k, v in m.items():
        if k in _PCT:
            s = f"{v * 100:8.2f}%"
        elif k in _MONEY:
            s = f"{v:12,.2f}"
        elif k in _YEARS:
            s = f"{v:8.1f} yr" if v != float("inf") else "     inf"
        elif isinstance(v, float):
            s = f"{v:8.3f}"
        else:
            s = f"{v}"
        lines.append(f"  {k:<24} {s}")
    return "\n".join(lines)
