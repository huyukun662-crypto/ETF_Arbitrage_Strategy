"""Signal v1 and v2 from `expressions_batch_0001.md`.

Sign convention: NEGATIVE signal = expected positive forward return
(mean reversion: high deviation from reference -> short, low -> long).

Both variants are LOOK-AHEAD-SAFE BY CONSTRUCTION: every rolling window
ends at t-1 close, accomplished via `.shift(1)` BEFORE the rolling
operation when needed, or by computing the moving stats in a way that
strictly excludes the current bar from the standardization window.

Round_0001 simplification (vs. the full expressions_batch_0001.md spec):
- v1, v2 use cohort-relative price (log-price minus cohort median log-price)
  as the "fair value reference" rather than a NAV/IOPV proxy.
- This is a clean test of the *deviation -> reversion* hypothesis without
  introducing the additional uncertainty of the IOPV proxy. NAV-based
  variants (v6) are deferred to round_0002.

Cohort = (asset_class, benchmark) per `data/universe/build.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_z_excluding_current(x: pd.Series | pd.DataFrame, window: int) -> pd.DataFrame:
    """(x_t - mean(x_{t-window..t-1})) / std(x_{t-window..t-1}).

    Standardization uses ONLY past data (window ending at t-1).
    """
    past = x.shift(1)
    mu = past.rolling(window=window, min_periods=window).mean()
    sd = past.rolling(window=window, min_periods=window).std(ddof=1)
    return (x - mu) / sd


def signal_v1(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """v1: pure price mean reversion z-score, no cohort neutralization.

    Per row (date), per ETF: standardize log-price using its own past
    `window` days (ending at t-1). Negative signal = mean reversion long.
    """
    log_p = np.log(prices)
    z = _rolling_z_excluding_current(log_p, window=window)
    return -z      # mean reversion: high z -> short, low z -> long


def signal_v2(prices: pd.DataFrame, cohort_map: dict[str, str],
              window: int = 60) -> pd.DataFrame:
    """v2: cohort-demeaned log-price z-score (60D).

    1. log_p -> demean by cohort median at each date
    2. rolling z-score over 60 trading days (ending at t-1)
    3. negate for mean-reversion convention.
    """
    relative = _cohort_relative_log_price(prices, cohort_map, min_cohort_size=2)
    z = _rolling_z_excluding_current(relative, window=window)
    return -z


def _cohort_relative_log_price(prices: pd.DataFrame, cohort_map: dict[str, str],
                               min_cohort_size: int = 2) -> pd.DataFrame:
    """Demean log-price by cohort median at each date.

    Only uses cohort members present in `prices` columns. Cohorts smaller
    than `min_cohort_size` are passed through unchanged (the resulting
    z-score will then be a vanilla price z-score for those symbols).
    """
    log_p = np.log(prices)
    cohorts: dict[str, list[str]] = {}
    for sym, cname in cohort_map.items():
        if sym in log_p.columns:
            cohorts.setdefault(cname, []).append(sym)

    relative = log_p.copy()
    for cname, members in cohorts.items():
        if len(members) < min_cohort_size:
            continue
        block = log_p[members]
        cohort_median = block.median(axis=1)
        relative[members] = block.sub(cohort_median, axis=0)
    return relative


def signal_v4(prices: pd.DataFrame, cohort_map: dict[str, str],
              window: int = 20) -> pd.DataFrame:
    """v4: Engle-Granger pair log-spread z-score, cohort-defined pairs.

    For each cohort with ≥2 members, take the highest-correlation pair on
    the IS panel (computed once, not rolling — avoids daily refit overhead
    in a baseline). Compute spread = log(P_A) - log(P_B), then z-score
    over `window` days ending at t-1. Sign convention: when spread is
    high z, A is overpriced vs B → short A, long B.

    Output panel has same shape as `prices`; non-pair-member cells are NaN.
    """
    log_p = np.log(prices)
    cohorts: dict[str, list[str]] = {}
    for sym, cname in cohort_map.items():
        if sym in log_p.columns:
            cohorts.setdefault(cname, []).append(sym)

    out = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    for cname, members in cohorts.items():
        if len(members) < 2:
            continue
        # Pair selection MUST be look-ahead-safe. Using full-panel correlation
        # (even on the IS window) lets future bars influence which pair is
        # picked, which in turn changes past signal values when the future is
        # perturbed → audit failure (max_abs_diff explodes).
        # Fix: pick the first two members from the cohort definition. This is
        # deterministic w.r.t. universe order and contains zero data-derived
        # selection. Production should use a pre-IS calibration window.
        a, b = members[0], members[1]

        spread = log_p[a] - log_p[b]
        z = _rolling_z_excluding_current(spread, window=window)

        # Negative z on A: A is below mean -> long A, short B
        out[a] = -z
        out[b] =  z

    return out


def signal_v6(prices: pd.DataFrame, nav: pd.DataFrame,
              cohort_map: dict[str, str],
              window: int = 60) -> pd.DataFrame:
    """v6: NAV-based premium-to-fair-value mean-reversion z-score.

    Construction:
      premium_t        = (price_t - nav_t) / nav_t
      cohort_relative  = premium - cohort_median(premium)
      z                = rolling_z_excluding_current(cohort_relative, window)
      signal           = -z   (mean reversion: high premium → expected drop)

    NAV is the official end-of-day unit value (单位净值). This is the true
    fair-value reference, replacing v2's price-cohort-relative proxy. v6 is
    sensitive to genuine premium dynamics (driven by 申赎 friction +
    regulatory limits) rather than cross-sectional price drift.

    Both `prices` and `nav` must be aligned on the same date index and
    columns. NaN rows in nav (non-trading days for the fund's reporting)
    propagate to the signal — handled downstream by the engine's mask.
    """
    nav_aligned = nav.reindex(index=prices.index, columns=prices.columns)
    nav_aligned = nav_aligned.ffill(limit=3)            # short ffill for reporting gaps
    premium = (prices - nav_aligned) / nav_aligned

    cohorts: dict[str, list[str]] = {}
    for sym, cname in cohort_map.items():
        if sym in prices.columns:
            cohorts.setdefault(cname, []).append(sym)

    rel = premium.copy()
    for cname, members in cohorts.items():
        if len(members) < 2:
            continue
        block = premium[members]
        rel[members] = block.sub(block.median(axis=1), axis=0)

    z = _rolling_z_excluding_current(rel, window=window)
    return -z


def signal_v8(prices: pd.DataFrame, cohort_map: dict[str, str],
              window: int = 20, threshold: float = 1.5) -> pd.DataFrame:
    """v8: cohort-relative z-score with asymmetric threshold gate.

    Same construction as v2 (cohort-demeaned log-price z-score) but only
    fires when |z| > threshold. Sub-threshold deviations are set to NaN
    (engine treats as no position). This concentrates trades on the
    extreme-deviation regime where mean reversion is strongest.
    """
    relative = _cohort_relative_log_price(prices, cohort_map, min_cohort_size=2)
    z = _rolling_z_excluding_current(relative, window=window)
    s = -z
    return s.where(s.abs() > threshold)


def signal_v9(prices: pd.DataFrame, cohort_map: dict[str, str],
              window: int = 20, vol_window: int = 20) -> pd.DataFrame:
    """v9: cohort-relative z-score scaled by INVERSE realized vol.

    Same z-score signal as v2, but each ETF's signal is divided by its own
    rolling realized vol (window: vol_window days). Higher-vol ETFs get
    smaller weights. This is per-asset risk parity, not regime gating.
    Stays within the dominant mechanism (deviation->reversion) and adds
    only a position-sizing layer.
    """
    relative = _cohort_relative_log_price(prices, cohort_map, min_cohort_size=2)
    z = _rolling_z_excluding_current(relative, window=window)
    s = -z

    log_p = np.log(prices)
    rets = log_p.diff().shift(1)         # past returns only
    vol = rets.rolling(vol_window, min_periods=vol_window).std()
    inv_vol = 1.0 / vol.replace(0, np.nan)
    return s * inv_vol


def signal_v7(prices: pd.DataFrame, cohort_map: dict[str, str],
              rebalance: str = "ME", lookback: int = 60) -> pd.DataFrame:
    """v7: monthly cross-sectional rank of cohort-relative log-price deviation.

    On each rebalance date (last trading day of month), rank the
    cohort-relative log-price deviation (vs `lookback`-day rolling mean
    ending at t-1). Hold for the full next month. Sign: negative rank
    (i.e. lowest-percentile deviation = expected-positive return).

    Output: panel with the same value held constant from one rebalance
    to the next (stair-step), so the engine treats it as a static-
    until-rebalance signal.
    """
    relative = _cohort_relative_log_price(prices, cohort_map, min_cohort_size=2)

    past = relative.shift(1)
    deviation = relative - past.rolling(lookback, min_periods=lookback).mean()

    # Cross-sectional rank within (date) — pct rank ∈ [0, 1]; subtract 0.5 -> [-0.5, +0.5]
    daily_signal = -(deviation.rank(axis=1, pct=True) - 0.5)

    # Stair-step: keep the value as of last trading day of previous month
    rebal_dates = daily_signal.resample(rebalance).last().index
    rebal_panel = daily_signal.reindex(rebal_dates, method="ffill")
    monthly_signal = rebal_panel.reindex(daily_signal.index, method="ffill")
    return monthly_signal
