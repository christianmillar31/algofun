"""Statistical honesty for backtests.

Formulas follow Bailey & López de Prado ("The Deflated Sharpe Ratio", 2014;
"Pseudo-Mathematics and Financial Charlatanism", 2014) and Lo ("The
Statistics of Sharpe Ratios", 2002). No SciPy dependency: the normal CDF
uses math.erf and the inverse CDF uses Acklam's approximation, accurate to
about 1e-9, which is more than these estimates deserve.

Sharpe ratios passed to the per-observation functions are NOT annualised:
use `sr_daily = sr_annual / sqrt(periods_per_year)`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

EULER_GAMMA = 0.5772156649015329


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def expected_max_sharpe(n_trials: int, sr_std: float = 1.0) -> float:
    """E[max Sharpe] over n_trials independent strategies with zero true skill
    and Sharpe estimates of standard deviation sr_std (same units as the SR)."""
    n = int(n_trials)
    if n <= 1 or sr_std <= 0:
        return 0.0
    return sr_std * ((1 - EULER_GAMMA) * norm_ppf(1 - 1 / n) + EULER_GAMMA * norm_ppf(1 - 1 / (n * math.e)))


def sharpe_std_error(sr: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Standard error of a per-observation Sharpe estimate (Lo 2002 with higher moments)."""
    if n_obs < 2:
        return math.inf
    var_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    return math.sqrt(max(var_term, 1e-12) / (n_obs - 1))


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """P(true Sharpe > sr_benchmark) given the estimate sr from n_obs observations."""
    se = sharpe_std_error(sr, n_obs, skew, kurt)
    if not math.isfinite(se) or se == 0:
        return 0.0
    return norm_cdf((sr - sr_benchmark) / se)


def deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float, n_trials: int,
                    trial_sr_std: float) -> tuple[float, float]:
    """Deflated Sharpe Ratio: PSR against the expected maximum Sharpe of n_trials
    noise strategies whose Sharpe estimates have std trial_sr_std.
    Returns (dsr, sr0). All Sharpes per observation."""
    sr0 = expected_max_sharpe(n_trials, trial_sr_std)
    return probabilistic_sharpe(sr, sr0, n_obs, skew, kurt), sr0


def min_track_record_length(sr: float, sr_benchmark: float = 0.0, skew: float = 0.0, kurt: float = 3.0,
                            confidence: float = 0.95) -> float:
    """Observations needed for PSR(sr_benchmark) to reach `confidence`. inf if sr <= benchmark."""
    if sr <= sr_benchmark:
        return math.inf
    z = norm_ppf(confidence)
    var_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    return 1.0 + max(var_term, 1e-12) * (z / (sr - sr_benchmark)) ** 2


def min_backtest_length_years(n_trials: int, target_sharpe_annual: float) -> float:
    """Years of backtest needed so that the best of n_trials noise strategies is
    NOT expected to reach target_sharpe_annual (upper bound 2 ln N / SR^2)."""
    if n_trials <= 1 or target_sharpe_annual <= 0:
        return 0.0
    return 2.0 * math.log(n_trials) / (target_sharpe_annual ** 2)


@dataclass
class HonestyReport:
    sharpe_annual: float
    sharpe_se_annual: float
    psr_zero: float                 # P(true Sharpe > 0)
    min_track_years: float          # years of live data to confirm Sharpe > 0 at 95%
    n_obs: int
    skew: float
    kurt: float
    n_trials: int = 1
    dsr: float | None = None        # deflated Sharpe if n_trials > 1
    expected_max_sharpe_annual: float | None = None
    min_backtest_years: float | None = None

    def lines(self) -> list[str]:
        out = [f"  sharpe_annual            {self.sharpe_annual:8.3f}  ± {self.sharpe_se_annual:.3f} (1 SE)",
               f"  prob_sharpe_gt_zero      {self.psr_zero:8.3f}",
               f"  min_track_years_95pct    {self.min_track_years:8.1f}" if math.isfinite(self.min_track_years)
               else "  min_track_years_95pct         inf"]
        if self.n_trials > 1 and self.dsr is not None:
            out += [f"  n_trials                 {self.n_trials:8d}",
                    f"  expected_max_noise_sr    {self.expected_max_sharpe_annual:8.3f}  (annualised, zero-skill best-of-N)",
                    f"  deflated_sharpe          {self.dsr:8.3f}  (want >= 0.95)",
                    f"  min_backtest_years       {self.min_backtest_years:8.1f}  (to rule out best-of-N luck)"]
        return out


def honesty_report(returns: pd.Series, periods_per_year: int = 252, trial_sharpes_annual: list[float] | None = None
                   ) -> HonestyReport:
    """Everything a backtest should print next to its Sharpe ratio.

    trial_sharpes_annual: annualised Sharpes of EVERY configuration tried in the
    search that produced this strategy (including this one). Their count and
    dispersion drive the deflated Sharpe.
    """
    r = returns.dropna().astype("float64")
    n = int(len(r))
    if n < 3 or r.std(ddof=1) == 0:
        return HonestyReport(0.0, math.inf, 0.0, math.inf, n, 0.0, 3.0)
    sr_d = float(r.mean() / r.std(ddof=1))
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0
    k = math.sqrt(periods_per_year)
    se_d = sharpe_std_error(sr_d, n, skew, kurt)
    rep = HonestyReport(
        sharpe_annual=sr_d * k, sharpe_se_annual=se_d * k,
        psr_zero=probabilistic_sharpe(sr_d, 0.0, n, skew, kurt),
        min_track_years=min_track_record_length(sr_d, 0.0, skew, kurt) / periods_per_year,
        n_obs=n, skew=skew, kurt=kurt,
    )
    if trial_sharpes_annual and len(trial_sharpes_annual) > 1:
        trials = np.array([s for s in trial_sharpes_annual if np.isfinite(s)], dtype="float64") / k
        if len(trials) > 1:
            dsr, sr0 = deflated_sharpe(sr_d, n, skew, kurt, len(trials), float(trials.std(ddof=1)))
            rep.n_trials = int(len(trials))
            rep.dsr = dsr
            rep.expected_max_sharpe_annual = sr0 * k
            rep.min_backtest_years = min_backtest_length_years(len(trials), max(sr0 * k, 1e-9))
    return rep
