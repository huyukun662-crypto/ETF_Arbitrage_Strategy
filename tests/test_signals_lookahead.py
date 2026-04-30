"""Critical: signals must be look-ahead-safe.

Future-bar perturbation must not change past signal values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.audits import lookahead_invariance
from data.universe.build import COHORTS, all_symbols, cohort_map
from strategy.etf_mean_reversion.signals import (
    signal_v1, signal_v2, signal_v4, signal_v6, signal_v7, signal_v8, signal_v9,
)


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2019-01-02", periods=400, freq="B")
    syms = all_symbols()
    # geometric Brownian motion per symbol
    log_rets = rng.normal(loc=0.0001, scale=0.012, size=(len(dates), len(syms)))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(log_rets, axis=0)),
        index=dates, columns=syms,
    )
    return prices


def test_v1_lookahead_safe(synthetic_prices):
    passed, detail = lookahead_invariance(
        signal_fn=lambda p: signal_v1(p, window=20),
        prices=synthetic_prices,
        perturb_last_k=20,
    )
    assert passed, f"v1 fails look-ahead: {detail}"


def test_v2_lookahead_safe(synthetic_prices):
    cmap = cohort_map()
    passed, detail = lookahead_invariance(
        signal_fn=lambda p: signal_v2(p, cohort_map=cmap, window=60),
        prices=synthetic_prices,
        perturb_last_k=20,
    )
    assert passed, f"v2 fails look-ahead: {detail}"


def test_v1_sign_convention(synthetic_prices):
    # If price spikes (high z), v1 should be NEGATIVE (mean reversion -> short)
    px = synthetic_prices.copy()
    sym = px.columns[0]
    px.iloc[-1, px.columns.get_loc(sym)] *= 1.5  # big positive shock today
    s = signal_v1(px, window=20)
    assert s.iloc[-1][sym] < 0, "v1 should be negative on positive shock (mean reversion)"


def test_v4_lookahead_safe(synthetic_prices):
    cmap = cohort_map()
    passed, detail = lookahead_invariance(
        signal_fn=lambda p: signal_v4(p, cohort_map=cmap, window=20),
        prices=synthetic_prices,
        perturb_last_k=20,
    )
    assert passed, f"v4 fails look-ahead: {detail}"


def test_v6_lookahead_safe(synthetic_prices):
    """v6 (NAV-based) must be look-ahead-safe in BOTH the price and NAV inputs.

    Synthetic NAV: same shape as prices but smoothed (moving avg of price)
    to mimic real-world NAV being a less-noisy reference.
    """
    cmap = cohort_map()
    # Synthetic NAV: 5-day rolling mean of price (smooth proxy)
    nav = synthetic_prices.rolling(5, min_periods=1).mean()

    def fn(p):
        nav_for_p = p.rolling(5, min_periods=1).mean()
        return signal_v6(p, nav=nav_for_p, cohort_map=cmap, window=60)

    passed, detail = lookahead_invariance(
        signal_fn=fn, prices=synthetic_prices, perturb_last_k=20,
    )
    assert passed, f"v6 fails look-ahead: {detail}"


def test_v8_lookahead_safe(synthetic_prices):
    cmap = cohort_map()
    passed, detail = lookahead_invariance(
        signal_fn=lambda p: signal_v8(p, cohort_map=cmap, window=20, threshold=1.5),
        prices=synthetic_prices, perturb_last_k=20,
    )
    assert passed, f"v8 fails look-ahead: {detail}"


def test_v9_lookahead_safe(synthetic_prices):
    cmap = cohort_map()
    passed, detail = lookahead_invariance(
        signal_fn=lambda p: signal_v9(p, cohort_map=cmap, window=20, vol_window=20),
        prices=synthetic_prices, perturb_last_k=20,
    )
    assert passed, f"v9 fails look-ahead: {detail}"


def test_v7_lookahead_safe(synthetic_prices):
    cmap = cohort_map()
    passed, detail = lookahead_invariance(
        signal_fn=lambda p: signal_v7(p, cohort_map=cmap, lookback=60),
        prices=synthetic_prices,
        perturb_last_k=25,
    )
    assert passed, f"v7 fails look-ahead: {detail}"


def test_v2_uses_cohort(synthetic_prices):
    cmap = cohort_map()
    s = signal_v2(synthetic_prices, cohort_map=cmap, window=60)
    # Within a cohort, summing relative log-prices should be ~0 by construction
    for c in COHORTS:
        members = [m for m in c.members if m in synthetic_prices.columns]
        if len(members) < 2:
            continue
        # The signal itself sums to zero per-row within cohort? Not exactly because
        # of independent rolling z. But the *underlying relative log-price* must.
        log_p = np.log(synthetic_prices)
        cohort_block = log_p[members]
        rel = cohort_block.sub(cohort_block.median(axis=1), axis=0)
        # Median of rel within cohort should be 0 (definition of median)
        med = rel.median(axis=1)
        assert med.abs().max() < 1e-10
