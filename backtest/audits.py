"""Mandatory audit functions for round_0001 (per skill mandatory audits).

All audits return (passed: bool, detail: dict). The backtest engine
records both fields per variant.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def lookahead_invariance(
    signal_fn: Callable[[pd.DataFrame], pd.DataFrame],
    prices: pd.DataFrame,
    perturb_last_k: int = 20,
    seed: int = 17,
) -> tuple[bool, dict]:
    """Future-bar perturbation test.

    Compute signal on `prices` and on `prices_perturbed` (where the last
    `perturb_last_k` rows are randomized). Past signal values must be
    bit-identical. Failure means the signal peeks at the future.
    """
    rng = np.random.default_rng(seed)
    sig_orig = signal_fn(prices)

    perturbed = prices.copy()
    last_idx = perturbed.index[-perturb_last_k:]
    noise = rng.normal(loc=1.0, scale=0.05, size=(perturb_last_k, perturbed.shape[1]))
    perturbed.loc[last_idx] = perturbed.loc[last_idx].values * noise

    sig_pert = signal_fn(perturbed)

    # Compare past values only (drop the perturbed tail)
    cutoff = perturbed.index[-perturb_last_k - 1]
    past_orig = sig_orig.loc[:cutoff]
    past_pert = sig_pert.loc[:cutoff]

    common_idx = past_orig.index.intersection(past_pert.index)
    if len(common_idx) == 0:
        return False, {"reason": "no overlapping past index", "n_compared": 0}

    a = past_orig.loc[common_idx]
    b = past_pert.loc[common_idx]

    # Allow tiny float noise from intermediate rolling stats but flag any
    # systematic divergence.
    diff = (a - b).abs().max().max()
    n_compared = a.notna().values.sum()
    passed = bool(diff < 1e-9)
    return passed, {"max_abs_diff": float(diff), "n_compared": int(n_compared)}


def per_year_sharpe(returns: pd.Series) -> dict:
    """Per-year Sharpe and worst-year/best-year-out diagnostics."""
    if returns.empty:
        return {"per_year": {}, "worst_year_sharpe": float("nan"),
                "best_year_out_sharpe": float("nan"), "headline_sharpe": float("nan")}

    rets = returns.dropna()
    headline = float(_sharpe(rets))

    by_year = rets.groupby(rets.index.year).apply(_sharpe)
    by_year_dict = {int(y): float(s) for y, s in by_year.items()}

    worst = float(min(by_year_dict.values())) if by_year_dict else float("nan")

    if len(by_year_dict) >= 2:
        best_year = max(by_year_dict, key=by_year_dict.get)
        rets_no_best = rets[rets.index.year != best_year]
        best_out = float(_sharpe(rets_no_best))
    else:
        best_out = float("nan")

    return {
        "per_year": by_year_dict,
        "worst_year_sharpe": worst,
        "best_year_out_sharpe": best_out,
        "headline_sharpe": headline,
    }


def _sharpe(rets: pd.Series, ann: float = 252.0) -> float:
    rets = rets.dropna()
    if len(rets) < 2:
        return float("nan")
    mu = rets.mean()
    sd = rets.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(mu / sd * np.sqrt(ann))


def ic_decay_by_delay(
    signal: pd.DataFrame, forward_returns: pd.DataFrame,
    delays: tuple[int, ...] = (0, 1, 2, 3, 5, 10),
) -> dict[int, float]:
    """Mean cross-sectional Spearman IC at each execution delay.

    Used to verify that delay=1 is not anomalously the only profitable lag
    (a signature of look-ahead).
    """
    from scipy.stats import spearmanr

    out: dict[int, float] = {}
    for d in delays:
        # signal at t -> return from t+d to t+d+horizon (already encoded in forward_returns)
        if d > 0:
            sig = signal.shift(d)
        else:
            sig = signal
        ics = []
        for ts in sig.index:
            if ts not in forward_returns.index:
                continue
            x = sig.loc[ts].dropna()
            y = forward_returns.loc[ts].reindex(x.index).dropna()
            common = x.index.intersection(y.index)
            if len(common) < 3:
                continue
            try:
                r, _ = spearmanr(x.loc[common], y.loc[common])
                if not np.isnan(r):
                    ics.append(r)
            except Exception:                          # noqa: BLE001
                continue
        out[d] = float(np.mean(ics)) if ics else float("nan")
    return out
