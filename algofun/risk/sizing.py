"""Position sizing helpers. All return weights (fractions of equity) indexed by ticker."""
from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight(selected: pd.Series | list[str], gross: float = 1.0) -> pd.Series:
    """Equal weight across selected names. `selected` is a bool Series or a list of tickers."""
    if isinstance(selected, pd.Series):
        names = list(selected.index[selected.astype(bool)])
    else:
        names = list(selected)
    if not names:
        return pd.Series(dtype="float64")
    return pd.Series(gross / len(names), index=names)


def inverse_vol(returns: pd.DataFrame, selected: list[str] | None = None,
                gross: float = 1.0, min_periods: int = 20) -> pd.Series:
    """Weight each name by 1/vol so every position contributes similar risk."""
    if selected is not None:
        returns = returns[[t for t in selected if t in returns.columns]]
    vol = returns.std(ddof=1)
    vol = vol[(returns.count() >= min_periods) & (vol > 0)]
    if vol.empty:
        return pd.Series(dtype="float64")
    inv = 1.0 / vol
    return inv / inv.sum() * gross


def vol_target(weights: pd.Series, returns: pd.DataFrame, target_annual_vol: float = 0.15,
               max_leverage: float = 1.0, periods_per_year: int = 252) -> pd.Series:
    """Scale the whole book so its trailing realised vol hits the target.

    Leverage is capped at max_leverage (1.0 = never lever up, only de-risk).
    """
    if weights.empty:
        return weights
    r = returns.reindex(columns=weights.index).fillna(0.0)
    port = r.to_numpy() @ weights.to_numpy()
    realised = float(np.std(port, ddof=1)) * np.sqrt(periods_per_year) if len(port) > 1 else 0.0
    if realised <= 0:
        return weights
    scale = min(target_annual_vol / realised, max_leverage / max(weights.abs().sum(), 1e-12))
    return weights * scale
