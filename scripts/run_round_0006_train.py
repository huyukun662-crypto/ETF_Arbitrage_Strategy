"""Round 0006 — Train sweep with NAV-based v6 (efinance-fueled).

Key changes vs round_0005:
- Added efinance adapter; cache hits unblock 510500 (zz500) and 159920
  (hang_seng) which AkShare had been blocking → potential 4-member zz500
  cohort + 3-member hang_seng cohort.
- Added v6 (NAV-based reference) — true fair-value-deviation reversion,
  using 单位净值 from efinance.fund.
- v9 retained as control (Round 0005 best Train mechanism, failed Validate).
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
from data.adapters.efinance_etf import (
    fetch_etf_panel as ef_fetch_panel,
    fetch_nav_panel as ef_fetch_nav,
)                                                     # noqa: E402
from data.splits import (
    TRAIN_END, TRAIN_START, is_oos_unlocked,
)                                                     # noqa: E402
from data.universe.build import all_symbols, cohort_map  # noqa: E402
from strategy.etf_mean_reversion.signals import (
    signal_v2, signal_v6, signal_v9,
)                                                     # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("round_0006_train")

SESSION_DIR = REPO_ROOT / "logs" / "20260428_etf_meanrev_arbitrage"
TOP_K = 5


@dataclass(frozen=True)
class SweepCfg:
    variant: str
    window: int
    horizon: int
    slippage_bp: float
    quintile_n: int = 5
    vol_window: int = 0


def main() -> int:
    if is_oos_unlocked():
        logger.error("OOS_UNLOCKED set; refusing.")
        return 2

    syms = all_symbols(enabled_only=True)
    logger.info("Train window: %s -> %s", TRAIN_START, TRAIN_END)

    # Fetch via efinance (with akshare cache fallback if needed).
    panel = ef_fetch_panel(syms, TRAIN_START, TRAIN_END, field="close")
    panel = panel.dropna(how="all", axis=0).ffill(limit=5)
    coverage = panel.notna().sum()
    keep = coverage[coverage >= 200].index.tolist()
    panel = panel[keep]
    logger.info("Train panel: %s; cols: %s", panel.shape, list(panel.columns))

    # Fetch NAV for v6 (only for the symbols we have prices for)
    nav = ef_fetch_nav(list(panel.columns), TRAIN_START, TRAIN_END)
    nav = nav.reindex(panel.index).ffill(limit=5)
    nav = nav[[c for c in panel.columns if c in nav.columns]]
    logger.info("Train NAV panel: %s; coverage: %s",
                nav.shape, dict(nav.notna().sum()))

    cmap = {s: c for s, c in cohort_map(enabled_only=True).items()
            if s in panel.columns}
    cohort_sizes = {c: sum(1 for s, cc in cmap.items() if cc == c)
                    for c in set(cmap.values())}
    logger.info("Cohort sizes: %s", cohort_sizes)

    sweep = []
    # v6 (focus: NAV-based)
    for w in [20, 40, 60, 120]:
        for h in [5, 10, 15, 21]:
            for s in [0.5, 1.0]:
                for q in [3, 5]:
                    sweep.append(SweepCfg("v6", w, h, s, q))
    # v9 (control)
    for w in [40, 60]:
        for vw in [40, 60]:
            for h in [10, 21]:
                for s in [0.5, 1.0]:
                    for q in [3, 5]:
                        sweep.append(SweepCfg("v9", w, h, s, q, vw))
    # v2 (baseline)
    for w in [40, 60]:
        for h in [10, 21]:
            for s in [0.5]:
                for q in [3, 5]:
                    sweep.append(SweepCfg("v2", w, h, s, q))

    logger.info("Sweep size: %d configs (v6: %d, v9: %d, v2: %d)",
                len(sweep),
                sum(1 for c in sweep if c.variant == "v6"),
                sum(1 for c in sweep if c.variant == "v9"),
                sum(1 for c in sweep if c.variant == "v2"))

    def make_signal(cfg: SweepCfg):
        if cfg.variant == "v2":
            return signal_v2(panel, cohort_map=cmap, window=cfg.window)
        if cfg.variant == "v6":
            return signal_v6(panel, nav=nav, cohort_map=cmap, window=cfg.window)
        if cfg.variant == "v9":
            vw = cfg.vol_window if cfg.vol_window > 0 else cfg.window
            return signal_v9(panel, cohort_map=cmap,
                              window=cfg.window, vol_window=vw)
        raise ValueError(cfg.variant)

    rows: list[dict] = []
    audit_done: dict[tuple, bool] = {}

    for i, cfg in enumerate(sweep):
        akey = (cfg.variant, cfg.window, cfg.vol_window)
        if akey not in audit_done:
            sig_fn = lambda p, c=cfg: make_signal(c)  # closure on outer panel/nav
            try:
                passed, _ = lookahead_invariance(signal_fn=sig_fn, prices=panel)
            except Exception as e:                          # noqa: BLE001
                logger.warning("Audit error %s: %r", cfg.variant, e)
                passed = False
            audit_done[akey] = passed
            if not passed:
                logger.error("AUDIT FAIL %s w=%d vw=%d",
                             cfg.variant, cfg.window, cfg.vol_window)
        if not audit_done[akey]:
            continue

        cost = CostModel(commission_oneway=0.00005, slippage_bp=cfg.slippage_bp)
        bt_cfg = BacktestConfig(horizon=cfg.horizon, delay=1,
                                quintile_n=cfg.quintile_n, min_universe_size=4)
        try:
            sig = make_signal(cfg)
            bt = run_backtest(signal=sig, prices=panel, variant=cfg.variant,
                              cost=cost, cfg=bt_cfg)
            row = {
                "variant": cfg.variant, "window": cfg.window,
                "horizon": cfg.horizon, "slippage_bp": cfg.slippage_bp,
                "quintile_n": cfg.quintile_n,
                "vol_window": cfg.vol_window if cfg.vol_window > 0 else cfg.window,
                **bt.metrics,
                "ic_at_delay_1": bt.ic_decay.get(1, float("nan")),
            }
            rows.append(row)
        except Exception as e:                              # noqa: BLE001
            logger.warning("Backtest error %s: %r", cfg.variant, e)

        if (i + 1) % 30 == 0:
            logger.info("  %d/%d done", i + 1, len(sweep))

    df = pd.DataFrame(rows).sort_values("q5_sharpe_after_cost", ascending=False)
    out_csv = SESSION_DIR / "grid_search_round_0006_train.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s (%d rows)", out_csv, len(df))

    eligible = df[df["worst_year_q5_sharpe"] >= 0.0].copy()
    logger.info("Configs with worst-year-train ≥ 0: %d", len(eligible))

    top = eligible.head(TOP_K)
    if len(top) == 0:
        logger.warning("No worst-year-positive config; falling back to top-K by Sh")
        top = df.head(TOP_K)

    show = ["variant", "window", "vol_window", "horizon", "slippage_bp",
            "quintile_n",
            "q5_sharpe_after_cost", "worst_year_q5_sharpe",
            "best_year_out_q5_sharpe", "ic_at_delay_1"]
    logger.info("\nTop-%d for Validate:\n%s",
                TOP_K, top[show].to_string(index=False))

    candidates = []
    for _, r in top.iterrows():
        candidates.append({k: (r[k].item() if hasattr(r[k], "item") else r[k])
                          for k in show})
    out_json = SESSION_DIR / "train_top_candidates_r6.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"train_window": [str(TRAIN_START), str(TRAIN_END)],
                   "panel_columns": list(panel.columns),
                   "cohort_sizes": cohort_sizes,
                   "n_train_configs": len(rows),
                   "n_eligible_worst_year_geq_0": int(len(eligible)),
                   "candidates": candidates},
                  f, indent=2, ensure_ascii=False, default=str)
    logger.info("Wrote %s", out_json)

    best_per = df.loc[df.groupby("variant")["q5_sharpe_after_cost"].idxmax()]
    logger.info("\nBest per variant on Train:\n%s",
                best_per[show].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
