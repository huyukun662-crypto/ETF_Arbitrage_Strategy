"""Run round_0001 backtest on v1 + v2, IS only.

Hard constraint: this script will REFUSE to load any data past IS_END
(2023-12-31). The OOS guard in `data/splits.py` enforces it at the
adapter level — if you try to bypass by editing this script, the
adapter will raise OOSAccessError unless you set OOS_UNLOCKED=true,
which should ONLY happen after IS results are recorded and approved.

Outputs:
    logs/20260428_etf_meanrev_arbitrage/grid_search_round_0001_is.csv
    logs/20260428_etf_meanrev_arbitrage/audit_round_0001.json
    logs/20260428_etf_meanrev_arbitrage/results_v1_is.csv
    logs/20260428_etf_meanrev_arbitrage/results_v2_is.csv
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

# Path bootstrap so script runs from repo root or from scripts/ dir.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

from backtest.audits import lookahead_invariance      # noqa: E402
from backtest.costs import CostModel                  # noqa: E402
from backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from data.adapters.akshare_etf import fetch_etf_panel # noqa: E402
from data.splits import IS_END, IS_START, is_oos_unlocked   # noqa: E402
from data.universe.build import COHORTS, all_symbols, cohort_map  # noqa: E402
from strategy.etf_mean_reversion.signals import signal_v1, signal_v2  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("round_0001_is")

SESSION_DIR = REPO_ROOT / "logs" / "20260428_etf_meanrev_arbitrage"
SESSION_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    if is_oos_unlocked():
        logger.error("OOS_UNLOCKED is set; refusing to run IS-only script with OOS unlock active.")
        return 2

    syms = all_symbols()
    logger.info("Universe: %d ETFs across %d cohorts", len(syms), len(COHORTS))
    logger.info("IS window: %s to %s", IS_START, IS_END)

    panel = fetch_etf_panel(syms, IS_START, IS_END, field="close")
    if panel.empty:
        logger.error("No data fetched. Aborting.")
        return 3

    panel = panel.dropna(how="all", axis=0)
    panel = panel.ffill(limit=5)
    logger.info("Panel shape: %s; date range: %s to %s",
                panel.shape, panel.index.min().date(), panel.index.max().date())

    # Drop ETFs with too little history (< 250 trading days)
    coverage = panel.notna().sum()
    keep = coverage[coverage >= 250].index.tolist()
    dropped = sorted(set(panel.columns) - set(keep))
    if dropped:
        logger.warning("Dropping ETFs with < 250 trading days: %s", dropped)
    panel = panel[keep]

    cmap = {s: c for s, c in cohort_map().items() if s in panel.columns}

    cost = CostModel()
    cfg = BacktestConfig(horizon=5, delay=1, quintile_n=5, min_universe_size=4)

    results: dict[str, dict] = {}
    audits: dict[str, dict] = {}

    for variant_name, signal_fn in [
        ("v1", lambda p: signal_v1(p, window=20)),
        ("v2", lambda p: signal_v2(p, cohort_map=cmap, window=60)),
    ]:
        logger.info("=" * 60)
        logger.info("Running %s ...", variant_name)

        passed, detail = lookahead_invariance(signal_fn=signal_fn, prices=panel)
        audits[variant_name] = {"lookahead_passed": passed, "lookahead_detail": detail}
        logger.info("  Look-ahead audit: passed=%s detail=%s", passed, detail)

        if not passed:
            logger.error("  -> SKIPPING %s (look-ahead audit failed)", variant_name)
            continue

        sig = signal_fn(panel)
        bt = run_backtest(signal=sig, prices=panel, variant=variant_name, cost=cost, cfg=cfg)

        results[variant_name] = {
            "metrics": bt.metrics,
            "ic_decay": bt.ic_decay,
            "per_year_ls": bt.per_year["ls"]["per_year"],
            "per_year_q5": bt.per_year["q5"]["per_year"],
        }

        bt.series.to_csv(SESSION_DIR / f"results_{variant_name}_is.csv")
        logger.info("  Metrics: %s", _fmt_metrics(bt.metrics))
        logger.info("  IC decay (delay -> mean IC): %s", {k: round(v, 4) for k, v in bt.ic_decay.items() if not np.isnan(v)})

    grid_df = pd.DataFrame({k: v["metrics"] for k, v in results.items()}).T
    grid_df.to_csv(SESSION_DIR / "grid_search_round_0001_is.csv")
    logger.info("\nIS Grid:\n%s", grid_df.to_string())

    with open(SESSION_DIR / "audit_round_0001.json", "w", encoding="utf-8") as f:
        out = {
            "round": "round_0001",
            "split": "is_only",
            "is_window": [str(IS_START), str(IS_END)],
            "universe": list(panel.columns),
            "n_trading_days": int(panel.shape[0]),
            "audits": audits,
            "results": results,
        }
        json.dump(out, f, indent=2, default=str, ensure_ascii=False)

    logger.info("Wrote: %s", SESSION_DIR / "grid_search_round_0001_is.csv")
    logger.info("Wrote: %s", SESSION_DIR / "audit_round_0001.json")
    return 0


def _fmt_metrics(m: dict) -> str:
    items = []
    for k, v in m.items():
        if isinstance(v, float):
            items.append(f"{k}={v:.3f}")
        else:
            items.append(f"{k}={v}")
    return " ".join(items)


if __name__ == "__main__":
    raise SystemExit(main())
