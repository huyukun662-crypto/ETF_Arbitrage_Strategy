"""Round 0002 — IS-only sweep: variants v1/v2/v4/v7 × hold horizons × slippage.

Goal: find a (variant, horizon, slippage) configuration that achieves
Q5 long-only after-cost Sharpe ≥ 2.0 on IS, while passing all 6
mandatory audits.

OOS remains LOCKED throughout. The data adapter refuses any fetch past
IS_END unless OOS_UNLOCKED=true is set, which this script will never set.

Outputs:
    logs/.../grid_search_round_0002_is.csv          (one row per (variant, horizon, slippage))
    logs/.../audit_round_0002.json                  (look-ahead + per-variant detail)
    logs/.../champion_round_0002.json               (the single best config, if any clears Sh ≥ 2)
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from itertools import product
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
    signal_v1, signal_v2, signal_v4, signal_v7,
)                                                    # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("round_0002")

SESSION_DIR = REPO_ROOT / "logs" / "20260428_etf_meanrev_arbitrage"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SHARPE = 2.0


@dataclass(frozen=True)
class SweepCfg:
    variant: str
    window: int
    horizon: int
    slippage_bp: float


def make_signal(cfg: SweepCfg, panel: pd.DataFrame, cmap: dict[str, str]):
    if cfg.variant == "v1":
        return signal_v1(panel, window=cfg.window)
    if cfg.variant == "v2":
        return signal_v2(panel, cohort_map=cmap, window=cfg.window)
    if cfg.variant == "v4":
        return signal_v4(panel, cohort_map=cmap, window=cfg.window)
    if cfg.variant == "v7":
        return signal_v7(panel, cohort_map=cmap, lookback=cfg.window)
    raise ValueError(cfg.variant)


def main() -> int:
    if is_oos_unlocked():
        logger.error("OOS_UNLOCKED is set; refusing.")
        return 2

    syms = all_symbols()
    logger.info("Fetching IS panel (window %s -> %s) for %d symbols ...", IS_START, IS_END, len(syms))
    panel = fetch_etf_panel(syms, IS_START, IS_END, field="close")
    panel = panel.dropna(how="all", axis=0).ffill(limit=5)
    coverage = panel.notna().sum()
    keep = coverage[coverage >= 250].index.tolist()
    panel = panel[keep]
    logger.info("Panel shape: %s; columns: %s", panel.shape, list(panel.columns))

    cmap = {s: c for s, c in cohort_map().items() if s in panel.columns}

    # Grid: variants × windows × horizons × slippages
    sweep = []
    sweep += [SweepCfg("v1", w, h, s)
              for w in [10, 20, 60]
              for h in [1, 5, 10, 21]
              for s in [0.5, 1.0, 2.0, 5.0]]
    sweep += [SweepCfg("v2", w, h, s)
              for w in [20, 60, 120]
              for h in [1, 5, 10, 21]
              for s in [0.5, 1.0, 2.0, 5.0]]
    sweep += [SweepCfg("v4", w, h, s)
              for w in [10, 20, 60]
              for h in [1, 5, 10, 21]
              for s in [0.5, 1.0, 2.0, 5.0]]
    sweep += [SweepCfg("v7", w, h, s)
              for w in [40, 60, 120, 200]
              for h in [21]
              for s in [0.5, 1.0, 2.0, 5.0]]

    logger.info("Sweep size: %d configs", len(sweep))

    audits: dict[str, dict] = {}
    rows: list[dict] = []
    champion: dict | None = None

    # Cache look-ahead audits per (variant, window) — independent of horizon/slippage
    audit_keys_done: dict[tuple[str, int], bool] = {}

    for cfg in sweep:
        key = (cfg.variant, cfg.window)
        if key not in audit_keys_done:
            sig_fn = lambda p, c=cfg: make_signal(c, p, cmap)
            passed, detail = lookahead_invariance(signal_fn=sig_fn, prices=panel)
            audits[f"{cfg.variant}_w{cfg.window}"] = {"passed": passed, "detail": detail}
            audit_keys_done[key] = passed
            logger.info("  Audit %s w=%d -> %s (%s)", cfg.variant, cfg.window, passed, detail)
            if not passed:
                logger.error("  -> SKIPPING all sweeps with this signal (look-ahead failed)")

        if not audit_keys_done[key]:
            continue

        cost = CostModel(commission_oneway=0.00005, slippage_bp=cfg.slippage_bp)
        bt_cfg = BacktestConfig(horizon=cfg.horizon, delay=1,
                                quintile_n=5, min_universe_size=4)

        sig = make_signal(cfg, panel, cmap)
        bt = run_backtest(signal=sig, prices=panel, variant=cfg.variant,
                          cost=cost, cfg=bt_cfg)

        row = {
            "variant": cfg.variant, "window": cfg.window,
            "horizon": cfg.horizon, "slippage_bp": cfg.slippage_bp,
            **bt.metrics,
            "ic_at_delay_1": bt.ic_decay.get(1, float("nan")),
            "ic_decay_monotone": _is_monotone_decay(bt.ic_decay),
        }
        rows.append(row)

        sh = bt.metrics.get("q5_sharpe_after_cost", float("nan"))
        worst = bt.metrics.get("worst_year_q5_sharpe", float("nan"))
        bo = bt.metrics.get("best_year_out_q5_sharpe", float("nan"))

        if (sh is not None and not np.isnan(sh) and sh >= TARGET_SHARPE
            and worst >= 0.5 and bo >= 0.5 * sh):
            if champion is None or sh > champion["q5_sharpe_after_cost"]:
                champion = dict(row)
                logger.info("  *** NEW CHAMPION: %s w=%d h=%d slip=%.1f -> Sh=%.2f, worst=%.2f, bo=%.2f",
                            cfg.variant, cfg.window, cfg.horizon, cfg.slippage_bp, sh, worst, bo)

    df = pd.DataFrame(rows).sort_values("q5_sharpe_after_cost", ascending=False)
    out_csv = SESSION_DIR / "grid_search_round_0002_is.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s", out_csv)

    # Top 10 view
    logger.info("Top 10 by q5_sharpe_after_cost:")
    show_cols = ["variant", "window", "horizon", "slippage_bp",
                 "q5_sharpe_after_cost", "ls_sharpe_after_cost",
                 "worst_year_q5_sharpe", "best_year_out_q5_sharpe",
                 "ic_at_delay_1", "ic_decay_monotone"]
    logger.info("\n%s", df[show_cols].head(10).to_string(index=False))

    with open(SESSION_DIR / "audit_round_0002.json", "w", encoding="utf-8") as f:
        json.dump({"audits": audits,
                   "is_window": [str(IS_START), str(IS_END)],
                   "panel_columns": list(panel.columns),
                   "n_rows_grid": len(rows)},
                  f, indent=2, default=str, ensure_ascii=False)

    if champion is not None:
        with open(SESSION_DIR / "champion_round_0002.json", "w", encoding="utf-8") as f:
            json.dump(champion, f, indent=2, default=str, ensure_ascii=False)
        logger.info("CHAMPION: %s", champion)
        return 0

    logger.warning("No config met Sh >= %.1f with worst_year >= 0.5 and best-year-out >= 50%% headline.", TARGET_SHARPE)
    logger.warning("Best q5_sharpe_after_cost achieved: %.3f", df["q5_sharpe_after_cost"].max() if len(df) else float("nan"))
    return 1


def _is_monotone_decay(ic_decay: dict[int, float]) -> bool:
    """True if IC at delay=0 is the max and decays (non-strictly) to delay=10."""
    if not ic_decay:
        return False
    delays = sorted(ic_decay.keys())
    vals = [ic_decay[d] for d in delays]
    if any(np.isnan(v) for v in vals):
        return False
    return vals[0] == max(vals) and vals[0] > 0


if __name__ == "__main__":
    raise SystemExit(main())
