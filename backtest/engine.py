"""Minimal event-driven daily backtest engine for ETF mean-reversion signals.

Honors:
- delay=1 (signal at close of day t -> trade at open of day t+1)
- After-cost return computation via CostModel
- Long-only Q5 excess + dollar-neutral long-short reporting
- Per-year decomposition for audit checks

Not a full market-impact / portfolio-optimization framework. For round_0001
we want a simple, transparent baseline that the audits can definitively
pass or fail on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.audits import _sharpe, ic_decay_by_delay, per_year_sharpe
from backtest.costs import CostModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    horizon: int = 5            # forward-return horizon used for IC + LS ret
    delay: int = 1              # signal at close t -> trade at open t+1
    quintile_n: int = 5         # for Q5/Q1 long-only basket
    min_universe_size: int = 4  # minimum cross-sectional members to trade


@dataclass(frozen=True)
class BacktestResult:
    variant: str
    metrics: dict
    per_year: dict
    ic_decay: dict
    series: pd.DataFrame  # date-indexed columns: ls_ret, q5_excess_ret, n_assets


def run_backtest(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    variant: str,
    cost: CostModel,
    cfg: BacktestConfig = BacktestConfig(),
) -> BacktestResult:
    """Run the daily backtest for a single signal panel.

    Args:
        signal: date x symbol panel of standardized signals (positive = long).
        prices: date x symbol panel of close prices (后复权).
        variant: variant id for logging / output keys.
        cost: cost model.
        cfg: backtest config.
    """
    common_cols = signal.columns.intersection(prices.columns)
    sig = signal[common_cols].copy()
    px = prices[common_cols].copy()

    # Forward return: signal at close of day t -> trade at close of day t+delay
    # (proxy for open_{t+delay+1}; one-bar conservative latency baked in by
    # using close_{t+delay} as entry price). Hold for `horizon` bars; exit
    # at close of day t+delay+horizon. So:
    #     fwd_ret_t = P_{t+delay+horizon} / P_{t+delay} - 1
    # Implementation: diff(horizon) at row r gives log(P_r) - log(P_{r-horizon}).
    # Shift by -(delay+horizon) places the value computed at row (t+delay+horizon)
    # into row t, yielding log(P_{t+delay+horizon}) - log(P_{t+delay}). ✓
    fwd_log_ret = np.log(px).diff(cfg.horizon).shift(-(cfg.delay + cfg.horizon))
    fwd_ret = np.exp(fwd_log_ret) - 1

    # Cross-sectional dollar-neutral long-short: weight = sign(signal) / count
    # Apply minimum universe filter per row
    valid_mask = sig.notna() & fwd_ret.notna()
    n_per_day = valid_mask.sum(axis=1)
    keep_days = n_per_day >= cfg.min_universe_size

    # Long-short: rank signals → top half long, bottom half short, equal weight per side.
    ranks = sig.where(valid_mask).rank(axis=1, pct=True)
    long_mask = ranks > 0.5
    short_mask = ranks <= 0.5

    long_w = long_mask.div(long_mask.sum(axis=1), axis=0)
    short_w = short_mask.div(short_mask.sum(axis=1), axis=0)
    weights_ls = (long_w - short_w).fillna(0.0)

    ls_ret_gross = (weights_ls * fwd_ret).sum(axis=1)
    # Annual turnover-based cost: each rebalance is full position turnover on each leg.
    # At horizon=5 days hold, ~252/5 = 50 round trips per year. Per round trip: round_trip_bps for long + round_trip_bps for short.
    cost_per_rebalance_bp = cost.round_trip_bps(short_leg=False, hold_days=cfg.horizon) + \
                            cost.round_trip_bps(short_leg=True, hold_days=cfg.horizon)
    # Per-trade cost is applied each holding cycle (every `horizon` days).
    # Approximate by amortizing daily.
    daily_cost = cost_per_rebalance_bp / 1e4 / cfg.horizon
    ls_ret_net = ls_ret_gross / cfg.horizon - daily_cost      # daily PnL contribution

    # Long-only Q5 excess: top quintile vs cohort-equal-weight benchmark
    q5_ranks = sig.where(valid_mask).rank(axis=1, pct=True)
    in_q5 = q5_ranks > (1 - 1.0 / cfg.quintile_n)
    q5_w = in_q5.div(in_q5.sum(axis=1), axis=0)
    bench_w = valid_mask.div(valid_mask.sum(axis=1), axis=0)
    q5_ret_gross = (q5_w * fwd_ret).sum(axis=1) - (bench_w * fwd_ret).sum(axis=1)
    q5_cost_bp = cost.round_trip_bps(short_leg=False, hold_days=cfg.horizon)
    q5_daily_cost = q5_cost_bp / 1e4 / cfg.horizon
    q5_ret_net = q5_ret_gross / cfg.horizon - q5_daily_cost

    # Restrict to days with enough universe members
    ls_ret_net = ls_ret_net.where(keep_days)
    q5_ret_net = q5_ret_net.where(keep_days)
    ls_ret_gross_d = (ls_ret_gross / cfg.horizon).where(keep_days)
    q5_ret_gross_d = (q5_ret_gross / cfg.horizon).where(keep_days)

    # Information coefficient
    ic_decay = ic_decay_by_delay(signal=sig, forward_returns=fwd_ret)

    # Headline metrics
    metrics = {
        "ls_sharpe_gross": _sharpe(ls_ret_gross_d),
        "ls_sharpe_after_cost": _sharpe(ls_ret_net),
        "q5_sharpe_gross": _sharpe(q5_ret_gross_d),
        "q5_sharpe_after_cost": _sharpe(q5_ret_net),
        "ic_mean_h": ic_decay.get(cfg.delay, float("nan")),
        "n_trading_days": int(keep_days.sum()),
        "avg_universe_size": float(n_per_day[keep_days].mean()) if keep_days.any() else float("nan"),
    }

    py_ls = per_year_sharpe(ls_ret_net)
    py_q5 = per_year_sharpe(q5_ret_net)
    metrics["worst_year_ls_sharpe"] = py_ls["worst_year_sharpe"]
    metrics["best_year_out_ls_sharpe"] = py_ls["best_year_out_sharpe"]
    metrics["worst_year_q5_sharpe"] = py_q5["worst_year_sharpe"]
    metrics["best_year_out_q5_sharpe"] = py_q5["best_year_out_sharpe"]

    series = pd.DataFrame({
        "ls_ret_gross_daily": ls_ret_gross_d,
        "ls_ret_net_daily":   ls_ret_net,
        "q5_ret_gross_daily": q5_ret_gross_d,
        "q5_ret_net_daily":   q5_ret_net,
        "n_assets":           n_per_day,
    })

    return BacktestResult(
        variant=variant,
        metrics=metrics,
        per_year={"ls": py_ls, "q5": py_q5},
        ic_decay=ic_decay,
        series=series,
    )
