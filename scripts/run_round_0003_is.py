"""Round 0003 — IS-only: expanded universe + v8/v9 + concentrated baskets.

Goal: maximize Q5 long-only after-cost Sharpe on IS, while keeping all
look-ahead audits passing. Per user 2026-04-28: 目标依然是is最大sh.

Adds vs round_0002:
- Universe: 7 cohorts attempted (gold, hs300, zz500, nasdaq100, sp500,
  hang_seng, cy50). Effective: 4 well-populated cohorts due to
  AkShare rate-limit on the new symbols.
- v8 (asymmetric threshold gate)
- v9 (per-asset inverse-vol weighting)
- Sweep extends horizons (1..30) and adds quintile_n in {3, 5} so we
  can compare top-tercile (more concentrated) vs top-quintile baskets.

OOS remains LOCKED.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

from backtest.audits import lookahead_invariance      # noqa: E402
from backtest.costs import CostModel                  # noqa: E402
from backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from data.adapters.akshare_etf import fetch_etf_panel # noqa: E402
from data.splits import IS_END, IS_START, is_oos_unlocked   # noqa: E402
from data.universe.build import all_symbols, cohort_map  # noqa: E402
from strategy.etf_mean_reversion.signals import (
    signal_v1, signal_v2, signal_v4, signal_v7, signal_v8, signal_v9,
)                                                    # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("round_0003")

SESSION_DIR = REPO_ROOT / "logs" / "20260428_etf_meanrev_arbitrage"


@dataclass(frozen=True)
class SweepCfg:
    variant: str
    window: int
    horizon: int
    slippage_bp: float
    quintile_n: int = 5
    threshold: float = 0.0   # only used by v8


def make_signal(cfg: SweepCfg, panel: pd.DataFrame, cmap: dict[str, str]):
    if cfg.variant == "v1":
        return signal_v1(panel, window=cfg.window)
    if cfg.variant == "v2":
        return signal_v2(panel, cohort_map=cmap, window=cfg.window)
    if cfg.variant == "v4":
        return signal_v4(panel, cohort_map=cmap, window=cfg.window)
    if cfg.variant == "v7":
        return signal_v7(panel, cohort_map=cmap, lookback=cfg.window)
    if cfg.variant == "v8":
        return signal_v8(panel, cohort_map=cmap, window=cfg.window, threshold=cfg.threshold)
    if cfg.variant == "v9":
        return signal_v9(panel, cohort_map=cmap, window=cfg.window, vol_window=cfg.window)
    raise ValueError(cfg.variant)


def main() -> int:
    if is_oos_unlocked():
        logger.error("OOS_UNLOCKED set; refusing.")
        return 2

    syms = all_symbols()
    logger.info("Fetching IS panel for %d symbols (mostly cached) ...", len(syms))
    panel = fetch_etf_panel(syms, IS_START, IS_END, field="close")
    panel = panel.dropna(how="all", axis=0).ffill(limit=5)
    coverage = panel.notna().sum()
    keep = coverage[coverage >= 250].index.tolist()
    panel = panel[keep]
    logger.info("Effective panel: %s; columns: %s", panel.shape, list(panel.columns))

    cmap = {s: c for s, c in cohort_map().items() if s in panel.columns}
    cohort_sizes = {c: sum(1 for s, cc in cmap.items() if cc == c) for c in set(cmap.values())}
    logger.info("Cohort sizes: %s", cohort_sizes)

    sweep = []
    # v2 — best variant in r2: explore around w=20 h=21 with finer horizon grid
    sweep += [SweepCfg("v2", w, h, s, q) for w in [10, 20, 40, 60]
              for h in [5, 10, 15, 21, 30] for s in [0.5, 1.0]
              for q in [3, 5]]
    # v8 — asymmetric threshold variants
    sweep += [SweepCfg("v8", w, h, s, q, t) for w in [10, 20, 40]
              for h in [5, 10, 21] for s in [0.5, 1.0]
              for q in [3, 5] for t in [0.5, 1.0, 1.5, 2.0]]
    # v9 — inverse-vol scaled
    sweep += [SweepCfg("v9", w, h, s, q) for w in [10, 20, 40, 60]
              for h in [5, 10, 21] for s in [0.5, 1.0]
              for q in [3, 5]]
    # v1 / v4 / v7 baselines retained for completeness but smaller grid
    sweep += [SweepCfg("v1", w, h, s, q) for w in [10, 20] for h in [5, 21]
              for s in [0.5, 1.0] for q in [3, 5]]
    sweep += [SweepCfg("v4", w, h, s, q) for w in [20] for h in [5, 21]
              for s in [0.5, 1.0] for q in [3, 5]]
    sweep += [SweepCfg("v7", w, 21, s, q) for w in [60, 120]
              for s in [0.5, 1.0] for q in [3, 5]]

    logger.info("Sweep size: %d configs", len(sweep))

    rows: list[dict] = []
    audits: dict[str, dict] = {}
    audit_done: dict[tuple[str, int, float], bool] = {}

    for i, cfg in enumerate(sweep):
        akey = (cfg.variant, cfg.window, cfg.threshold)
        if akey not in audit_done:
            sig_fn = lambda p, c=cfg: make_signal(c, p, cmap)
            passed, detail = lookahead_invariance(signal_fn=sig_fn, prices=panel)
            audits[f"{cfg.variant}_w{cfg.window}_t{cfg.threshold}"] = {
                "passed": passed, "detail": detail
            }
            audit_done[akey] = passed
            if not passed:
                logger.error("AUDIT FAIL %s w=%d t=%.1f -> %s", cfg.variant, cfg.window, cfg.threshold, detail)
        if not audit_done[akey]:
            continue

        cost = CostModel(commission_oneway=0.00005, slippage_bp=cfg.slippage_bp)
        bt_cfg = BacktestConfig(horizon=cfg.horizon, delay=1,
                                quintile_n=cfg.quintile_n, min_universe_size=4)
        sig = make_signal(cfg, panel, cmap)
        bt = run_backtest(signal=sig, prices=panel, variant=cfg.variant,
                          cost=cost, cfg=bt_cfg)

        row = {
            "variant": cfg.variant, "window": cfg.window,
            "horizon": cfg.horizon, "slippage_bp": cfg.slippage_bp,
            "quintile_n": cfg.quintile_n, "threshold": cfg.threshold,
            **bt.metrics,
            "ic_at_delay_1": bt.ic_decay.get(1, float("nan")),
        }
        rows.append(row)

        if (i + 1) % 30 == 0:
            logger.info("  ... %d/%d done", i + 1, len(sweep))

    df = pd.DataFrame(rows).sort_values("q5_sharpe_after_cost", ascending=False)
    out = SESSION_DIR / "grid_search_round_0003_is.csv"
    df.to_csv(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(df))

    show_cols = ["variant", "window", "horizon", "slippage_bp", "quintile_n", "threshold",
                 "q5_sharpe_after_cost", "worst_year_q5_sharpe",
                 "best_year_out_q5_sharpe", "ic_at_delay_1"]
    logger.info("\nTop 15 by q5_sharpe_after_cost:\n%s", df[show_cols].head(15).to_string(index=False))

    best_per = df.loc[df.groupby("variant")["q5_sharpe_after_cost"].idxmax()]
    logger.info("\nBest per variant:\n%s", best_per[show_cols].to_string(index=False))

    with open(SESSION_DIR / "audit_round_0003.json", "w", encoding="utf-8") as f:
        json.dump({"audits": audits,
                   "is_window": [str(IS_START), str(IS_END)],
                   "panel_columns": list(panel.columns),
                   "cohort_sizes": cohort_sizes,
                   "n_rows_grid": len(rows)},
                  f, indent=2, default=str, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
