"""Round 0004 — Train-only sweep (2018-01-01 → 2022-06-30).

Per user 2026-04-28 protocol:
  Train (2018-01 → 2022-06): grid sweep, pick top candidates by Q5 Sh.
  Validate (2022-07 → 2023-12): single-shot promotion check on top candidates only.
  Test/OOS (2024-01 → 2026-04): LOCKED until Validate passes.

This script handles the Train phase ONLY. It picks the top 3 by Sh
(plus mandates worst-year-train ≥ 0 to discard pure tail-driven configs).

Outputs:
    grid_search_round_0004_train.csv       (all configs × Train metrics)
    train_top_candidates.json              (top 3 selected for Validate)
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
from data.splits import (
    TRAIN_END, TRAIN_START, is_oos_unlocked,
)                                                     # noqa: E402
from data.universe.build import all_symbols, cohort_map  # noqa: E402
from strategy.etf_mean_reversion.signals import (
    signal_v1, signal_v2, signal_v4, signal_v8, signal_v9,
)                                                     # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("round_0004_train")

SESSION_DIR = REPO_ROOT / "logs" / "20260428_etf_meanrev_arbitrage"
TOP_K = 3


@dataclass(frozen=True)
class SweepCfg:
    variant: str
    window: int
    horizon: int
    slippage_bp: float
    quintile_n: int = 5
    threshold: float = 0.0


def make_signal(cfg: SweepCfg, panel: pd.DataFrame, cmap: dict[str, str]):
    if cfg.variant == "v1":
        return signal_v1(panel, window=cfg.window)
    if cfg.variant == "v2":
        return signal_v2(panel, cohort_map=cmap, window=cfg.window)
    if cfg.variant == "v4":
        return signal_v4(panel, cohort_map=cmap, window=cfg.window)
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
    logger.info("Train window: %s -> %s", TRAIN_START, TRAIN_END)
    panel = fetch_etf_panel(syms, TRAIN_START, TRAIN_END, field="close")
    panel = panel.dropna(how="all", axis=0).ffill(limit=5)
    coverage = panel.notna().sum()
    keep = coverage[coverage >= 200].index.tolist()
    panel = panel[keep]
    logger.info("Train panel: %s; columns: %s", panel.shape, list(panel.columns))

    cmap = {s: c for s, c in cohort_map().items() if s in panel.columns}
    cohort_sizes = {c: sum(1 for s, cc in cmap.items() if cc == c) for c in set(cmap.values())}
    logger.info("Cohort sizes: %s", cohort_sizes)

    sweep = []
    sweep += [SweepCfg("v2", w, h, s, q) for w in [20, 40, 60]
              for h in [5, 10, 15, 21] for s in [0.5, 1.0] for q in [3, 5]]
    sweep += [SweepCfg("v8", w, h, s, q, t) for w in [20, 40, 60]
              for h in [10, 21] for s in [0.5, 1.0]
              for q in [3, 5] for t in [1.0, 1.5, 2.0]]
    sweep += [SweepCfg("v9", w, h, s, q) for w in [20, 40, 60]
              for h in [10, 21] for s in [0.5, 1.0] for q in [3, 5]]
    sweep += [SweepCfg("v1", w, h, s, q) for w in [10, 20] for h in [5, 21]
              for s in [0.5, 1.0] for q in [3, 5]]
    sweep += [SweepCfg("v4", 20, h, s, q) for h in [5, 21]
              for s in [0.5, 1.0] for q in [3, 5]]

    logger.info("Sweep size: %d configs", len(sweep))

    rows: list[dict] = []
    audit_done: dict[tuple[str, int, float], bool] = {}

    for i, cfg in enumerate(sweep):
        akey = (cfg.variant, cfg.window, cfg.threshold)
        if akey not in audit_done:
            sig_fn = lambda p, c=cfg: make_signal(c, p, cmap)
            passed, _ = lookahead_invariance(signal_fn=sig_fn, prices=panel)
            audit_done[akey] = passed
            if not passed:
                logger.error("AUDIT FAIL %s w=%d t=%.1f", cfg.variant, cfg.window, cfg.threshold)
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
            logger.info("  %d/%d done", i + 1, len(sweep))

    df = pd.DataFrame(rows)
    df = df.sort_values("q5_sharpe_after_cost", ascending=False)
    out_csv = SESSION_DIR / "grid_search_round_0004_train.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s (%d rows)", out_csv, len(df))

    # Selection rule: top-K by Sh, but require worst-year-train ≥ 0
    # (positive in every Train year — basic stability filter before Validate).
    eligible = df[df["worst_year_q5_sharpe"] >= 0.0].copy()
    logger.info("Configs with worst-year-train ≥ 0: %d", len(eligible))

    top = eligible.head(TOP_K)
    if len(top) == 0:
        logger.warning("No config passed the worst-year-train ≥ 0 floor; falling back to top-K by Sh")
        top = df.head(TOP_K)

    show = ["variant", "window", "horizon", "slippage_bp", "quintile_n",
            "threshold", "q5_sharpe_after_cost", "worst_year_q5_sharpe",
            "best_year_out_q5_sharpe", "ic_at_delay_1"]
    logger.info("\nTop-%d candidates for Validate:\n%s",
                TOP_K, top[show].to_string(index=False))

    candidates = []
    for _, r in top.iterrows():
        candidates.append({k: (r[k].item() if hasattr(r[k], "item") else r[k])
                          for k in show})
    out_json = SESSION_DIR / "train_top_candidates.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"train_window": [str(TRAIN_START), str(TRAIN_END)],
                   "panel_columns": list(panel.columns),
                   "cohort_sizes": cohort_sizes,
                   "n_train_configs": len(rows),
                   "n_eligible_worst_year_geq_0": int(len(eligible)),
                   "candidates": candidates},
                  f, indent=2, ensure_ascii=False, default=str)
    logger.info("Wrote %s", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
