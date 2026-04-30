"""Round 0005 — Train sweep focused on v9 inverse-vol + controls.

Universe filter: only enabled cohorts (drops cy50, nasdaq100, sp500,
bond_treasury — all of which have insufficient data). Effective universe:
gold(4) + hs300(4) + zz500(3) + hang_seng(2) = 13 ETFs across 4 cohorts.

Hypothesis (per user 2026-04-28):
  v9's inverse-vol weighting is a portfolio-construction effect that should
  generalize across universes. With singleton/short-history cohorts removed,
  v9's edge should retain more of its measured value than v8's signal-design
  effect did in round 0004.

This script handles Train ONLY. Validate is run by run_round_0005_validate.py.
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
    signal_v2, signal_v8, signal_v9,
)                                                     # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("round_0005_train")

SESSION_DIR = REPO_ROOT / "logs" / "20260428_etf_meanrev_arbitrage"
TOP_K = 5


@dataclass(frozen=True)
class SweepCfg:
    variant: str
    window: int
    horizon: int
    slippage_bp: float
    quintile_n: int = 5
    threshold: float = 0.0
    vol_window: int = 0     # only used by v9 (0 means "= window")


def make_signal(cfg: SweepCfg, panel, cmap):
    if cfg.variant == "v2":
        return signal_v2(panel, cohort_map=cmap, window=cfg.window)
    if cfg.variant == "v8":
        return signal_v8(panel, cohort_map=cmap, window=cfg.window, threshold=cfg.threshold)
    if cfg.variant == "v9":
        vw = cfg.vol_window if cfg.vol_window > 0 else cfg.window
        return signal_v9(panel, cohort_map=cmap, window=cfg.window, vol_window=vw)
    raise ValueError(cfg.variant)


def main() -> int:
    if is_oos_unlocked():
        logger.error("OOS_UNLOCKED set; refusing.")
        return 2

    syms = all_symbols(enabled_only=True)
    logger.info("Train window: %s -> %s", TRAIN_START, TRAIN_END)
    logger.info("Enabled-only universe: %d symbols", len(syms))

    panel = fetch_etf_panel(syms, TRAIN_START, TRAIN_END, field="close")
    panel = panel.dropna(how="all", axis=0).ffill(limit=5)
    coverage = panel.notna().sum()
    keep = coverage[coverage >= 200].index.tolist()
    panel = panel[keep]
    logger.info("Train panel: %s; columns: %s", panel.shape, list(panel.columns))

    cmap = {s: c for s, c in cohort_map(enabled_only=True).items() if s in panel.columns}
    cohort_sizes = {c: sum(1 for s, cc in cmap.items() if cc == c) for c in set(cmap.values())}
    logger.info("Cohort sizes: %s", cohort_sizes)

    # ── v9 focused grid ─────────────────────────────────────────────
    sweep = []
    for w in [20, 40, 60, 120]:
        for vw in [20, 40, 60]:
            for h in [5, 10, 15, 21]:
                for s in [0.5, 1.0]:
                    for q in [3, 5]:
                        sweep.append(SweepCfg("v9", w, h, s, q, 0.0, vw))
    # ── v2 baseline (no inverse-vol) for direct comparison ─────────
    for w in [20, 40, 60]:
        for h in [5, 10, 15, 21]:
            for s in [0.5, 1.0]:
                for q in [3, 5]:
                    sweep.append(SweepCfg("v2", w, h, s, q))
    # ── v8 (round_0004 best) for sanity check ──────────────────────
    for w in [20, 40]:
        for h in [10, 21]:
            for s in [0.5, 1.0]:
                for q in [3, 5]:
                    for t in [1.5, 2.0]:
                        sweep.append(SweepCfg("v8", w, h, s, q, t))

    logger.info("Sweep size: %d configs (v9: %d, v2: %d, v8: %d)",
                len(sweep),
                sum(1 for c in sweep if c.variant == "v9"),
                sum(1 for c in sweep if c.variant == "v2"),
                sum(1 for c in sweep if c.variant == "v8"))

    rows: list[dict] = []
    audit_done: dict[tuple, bool] = {}

    for i, cfg in enumerate(sweep):
        akey = (cfg.variant, cfg.window, cfg.threshold, cfg.vol_window)
        if akey not in audit_done:
            sig_fn = lambda p, c=cfg: make_signal(c, p, cmap)
            passed, _ = lookahead_invariance(signal_fn=sig_fn, prices=panel)
            audit_done[akey] = passed
            if not passed:
                logger.error("AUDIT FAIL %s w=%d t=%.1f vw=%d",
                             cfg.variant, cfg.window, cfg.threshold, cfg.vol_window)
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
            "vol_window": cfg.vol_window if cfg.vol_window > 0 else cfg.window,
            **bt.metrics,
            "ic_at_delay_1": bt.ic_decay.get(1, float("nan")),
        }
        rows.append(row)

        if (i + 1) % 30 == 0:
            logger.info("  %d/%d done", i + 1, len(sweep))

    df = pd.DataFrame(rows).sort_values("q5_sharpe_after_cost", ascending=False)
    out_csv = SESSION_DIR / "grid_search_round_0005_train.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s (%d rows)", out_csv, len(df))

    # Selection: top-K by Sh among configs with worst-year-train ≥ 0
    eligible = df[df["worst_year_q5_sharpe"] >= 0.0].copy()
    logger.info("Configs with worst-year-train ≥ 0: %d", len(eligible))

    top = eligible.head(TOP_K)
    if len(top) == 0:
        logger.warning("No config passed the worst-year-train ≥ 0 floor; falling back to top-K by Sh")
        top = df.head(TOP_K)

    show = ["variant", "window", "vol_window", "horizon", "slippage_bp",
            "quintile_n", "threshold",
            "q5_sharpe_after_cost", "worst_year_q5_sharpe",
            "best_year_out_q5_sharpe", "ic_at_delay_1"]
    logger.info("\nTop-%d candidates for Validate:\n%s",
                TOP_K, top[show].to_string(index=False))

    candidates = []
    for _, r in top.iterrows():
        candidates.append({k: (r[k].item() if hasattr(r[k], "item") else r[k])
                          for k in show})
    out_json = SESSION_DIR / "train_top_candidates_r5.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"train_window": [str(TRAIN_START), str(TRAIN_END)],
                   "panel_columns": list(panel.columns),
                   "cohort_sizes": cohort_sizes,
                   "n_train_configs": len(rows),
                   "n_eligible_worst_year_geq_0": int(len(eligible)),
                   "candidates": candidates},
                  f, indent=2, ensure_ascii=False, default=str)
    logger.info("Wrote %s", out_json)

    # Best per variant for diagnostic
    best_per = df.loc[df.groupby("variant")["q5_sharpe_after_cost"].idxmax()]
    logger.info("\nBest per variant on Train:\n%s", best_per[show].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
