import math

import numpy as np
import pandas as pd
import pytest

from algofun.backtest.stats import (
    deflated_sharpe,
    expected_max_sharpe,
    honesty_report,
    min_backtest_length_years,
    min_track_record_length,
    norm_cdf,
    norm_ppf,
    probabilistic_sharpe,
    sharpe_std_error,
)


def test_normal_helpers():
    assert norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert norm_cdf(norm_ppf(0.3)) == pytest.approx(0.3, abs=1e-7)
    assert norm_ppf(0.0) == -math.inf and norm_ppf(1.0) == math.inf


def test_expected_max_sharpe_and_minbtl():
    assert expected_max_sharpe(1) == 0.0
    e10, e1000 = expected_max_sharpe(10), expected_max_sharpe(1000)
    assert 1.4 < e10 < 1.7 and 3.1 < e1000 < 3.4   # ~1.54 and ~3.26 in the paper
    # the paper's headline: 5 years of data, ~45 trials -> best noise Sharpe of 1.0
    assert min_backtest_length_years(45, 1.0) == pytest.approx(2 * math.log(45), rel=1e-9)
    assert min_backtest_length_years(45, 1.0) > 5  # upper bound exceeds 5 years, so 5 years is not enough
    assert min_backtest_length_years(1, 1.0) == 0.0


def test_psr_dsr_and_mintrl_consistency():
    sr_d, n, skew, kurt = 0.06, 750, -0.3, 5.0   # ~Sharpe 0.95 annualised over 3 years, fat left tail
    psr = probabilistic_sharpe(sr_d, 0.0, n, skew, kurt)
    assert 0.9 < psr < 1.0
    mtrl = min_track_record_length(sr_d, 0.0, skew, kurt, 0.95)
    assert probabilistic_sharpe(sr_d, 0.0, int(round(mtrl)), skew, kurt) == pytest.approx(0.95, abs=0.01)
    assert min_track_record_length(0.0, 0.0) == math.inf
    dsr1, sr0_1 = deflated_sharpe(sr_d, n, skew, kurt, 1, 0.02)
    dsr50, sr0_50 = deflated_sharpe(sr_d, n, skew, kurt, 50, 0.02)
    assert sr0_1 == 0.0 and sr0_50 > 0 and dsr50 < dsr1
    assert sharpe_std_error(sr_d, n) < sharpe_std_error(sr_d, n // 4)


def test_honesty_report_from_returns():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0006, 0.01, 1000))
    h = honesty_report(r)
    assert h.n_obs == 1000 and h.sharpe_annual == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))
    assert 0 < h.psr_zero < 1 and h.sharpe_se_annual > 0 and h.dsr is None
    h2 = honesty_report(r, trial_sharpes_annual=[0.2, 0.5, 0.9, 1.1, h.sharpe_annual])
    assert h2.n_trials == 5 and h2.dsr is not None and h2.dsr <= h2.psr_zero
    assert h2.expected_max_sharpe_annual > 0 and h2.min_backtest_years > 0
    assert any("deflated_sharpe" in line for line in h2.lines())
    flat = honesty_report(pd.Series([0.0] * 10))
    assert flat.psr_zero == 0.0 and flat.min_track_years == math.inf
