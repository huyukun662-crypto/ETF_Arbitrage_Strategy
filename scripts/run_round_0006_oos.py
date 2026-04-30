"""Round 0006 — OOS final test (2024-01-01 → 2026-04-28).

⚠ ONE-SHOT EVALUATION. NO PARAMETER TUNING ALLOWED HERE.

Reads `train_top_candidates_r6.json` and evaluates the PROMOTED candidates
(or specifically the champion + sister) on the OOS window.

Per the user's 2026-04-28 protocol:
  - OOS_UNLOCKED=true must be set before running
  - This script applies the same 4 promotion rules as Validate
  - If OOS passes → DEPLOY recommendation (paper trading first)
  - If OOS fails → STOP. No re-tune. No retry. Document and conclude.

Outputs:
    grid_search_round_0006_oos.csv
    oos_decision_r6.json     (final DEPLOY / STOP)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

from backtest.audits import (
    _sharpe, ic_decay_by_delay, lookahead_invariance, per_year_sharpe,
)                                                     # noqa: E402
from backtest.costs import CostModel                  # noqa: E402
from backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from data.adapters.efinance_etf import (
    fetch_etf_panel as ef_fetch_panel, fetch_nav_panel as ef_fetch_nav,
)                                                     # noqa: E402
from data.splits import (
    OOS_END, OOS_START, is_oos_unlocked,
)                                                     # noqa: E402
from data.universe.build import all_symbols, cohort_map  # noqa: E402
from strategy.etf_mean_reversion.signals import (
    signal_v2, signal_v6, signal_v9,
)                                                     # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("round_0006_oos")

SESSION_DIR = REPO_ROOT / "logs" / "20260428_etf_meanrev_arbitrage"


def main() -> int:
    if not is_oos_unlocked():
        logger.error("OOS_UNLOCKED is NOT set. Refusing to load OOS data.")
        logger.error("To unlock, run: export OOS_UNLOCKED=true (Linux/Mac)")
        logger.error("                  $env:OOS_UNLOCKED='true' (PowerShell)")
        logger.error("                  set OOS_UNLOCKED=true (cmd.exe)")
        return 2
    logger.info("=" * 60)
    logger.info("⚠ OOS UNLOCKED — running final test on %s → %s", OOS_START, OOS_END)
    logger.info("=" * 60)

    cands_path = SESSION_DIR / "train_top_candidates_r6.json"
    blob = json.load(open(cands_path, encoding="utf-8"))
    candidates = blob["candidates"]

    # Filter to ONLY candidates that PROMOTED in Validate (champion + sister)
    val_path = SESSION_DIR / "validate_decision_r6.json"
    val_blob = json.load(open(val_path, encoding="utf-8"))
    promoted_keys = {
        (r["variant"], r["window"], r.get("vol_window", r["window"]),
         r["horizon"], r["slippage_bp"], r["quintile_n"])
        for r in val_blob["all_results"] if r["PROMOTE"]
    }
    candidates_to_test = [
        c for c in candidates
        if (c["variant"], c["window"], c.get("vol_window", c["window"]),
            c["horizon"], c["slippage_bp"], c["quintile_n"]) in promoted_keys
    ]
    logger.info("Promoted candidates from Validate: %d", len(candidates_to_test))
    for c in candidates_to_test:
        logger.info("  %s w=%d vw=%d h=%d q=%d slip=%.1f",
                    c["variant"], c["window"], c.get("vol_window", c["window"]),
                    c["horizon"], c["quintile_n"], c["slippage_bp"])

    # Fetch panel: lookback buffer + OOS window
    LOOKBACK_DAYS = 250
    fetch_start = (pd.Timestamp(OOS_START) -
                   pd.Timedelta(days=int(LOOKBACK_DAYS * 1.6))).date()
    logger.info("Fetching panel from %s to %s", fetch_start, OOS_END)

    panel = ef_fetch_panel(all_symbols(enabled_only=True),
                           fetch_start, OOS_END, field="close")
    panel = panel.dropna(how="all", axis=0).ffill(limit=5)
    coverage = panel.notna().sum()
    keep = coverage[coverage >= 200].index.tolist()
    panel = panel[keep]
    nav = ef_fetch_nav(list(panel.columns), fetch_start, OOS_END)
    nav = nav.reindex(panel.index).ffill(limit=5)
    nav = nav[[c for c in panel.columns if c in nav.columns]]
    cmap = {s: c for s, c in cohort_map(enabled_only=True).items()
            if s in panel.columns}
    logger.info("Full panel: %s; nav cols: %d",
                panel.shape, nav.shape[1] if not nav.empty else 0)

    oos_mask = ((panel.index >= pd.Timestamp(OOS_START)) &
                (panel.index <= pd.Timestamp(OOS_END)))
    oos_dates = panel.index[oos_mask]
    logger.info("OOS window: %d trading days (%s to %s)",
                len(oos_dates),
                oos_dates.min().date(), oos_dates.max().date())

    def make_signal(v, w, vw, panel_, nav_, cmap_):
        if v == "v2":  return signal_v2(panel_, cohort_map=cmap_, window=w)
        if v == "v6":  return signal_v6(panel_, nav=nav_, cohort_map=cmap_, window=w)
        if v == "v9":  return signal_v9(panel_, cohort_map=cmap_,
                                          window=w, vol_window=vw)
        raise ValueError(v)

    rows: list[dict] = []
    for cand in candidates_to_test:
        v = cand["variant"]; w = int(cand["window"]); h = int(cand["horizon"])
        s = float(cand["slippage_bp"]); q = int(cand["quintile_n"])
        vw = int(cand.get("vol_window", w))

        passed, _ = lookahead_invariance(
            signal_fn=lambda p: make_signal(v, w, vw, p, nav, cmap),
            prices=panel,
        )
        if not passed:
            logger.error("AUDIT FAIL %s; skipping", cand)
            continue

        sig_full = make_signal(v, w, vw, panel, nav, cmap)
        cost = CostModel(commission_oneway=0.00005, slippage_bp=s)
        bt_full = run_backtest(signal=sig_full, prices=panel, variant=v,
                               cost=cost,
                               cfg=BacktestConfig(horizon=h, delay=1,
                                                  quintile_n=q,
                                                  min_universe_size=4))

        ser = bt_full.series.copy()
        oos_ser = ser.loc[(ser.index >= pd.Timestamp(OOS_START)) &
                          (ser.index <= pd.Timestamp(OOS_END))]
        oos_q5_net = oos_ser["q5_ret_net_daily"].dropna()
        oos_q5_gross = oos_ser["q5_ret_gross_daily"].dropna()
        oos_ls_net = oos_ser["ls_ret_net_daily"].dropna()

        sig_o = sig_full.loc[(sig_full.index >= pd.Timestamp(OOS_START)) &
                             (sig_full.index <= pd.Timestamp(OOS_END))]
        fwd_log = np.log(panel).diff(h).shift(-(1 + h))
        fwd = np.exp(fwd_log) - 1
        fwd_o = fwd.loc[sig_o.index.intersection(fwd.index)]
        ic_o = ic_decay_by_delay(signal=sig_o, forward_returns=fwd_o)
        py = per_year_sharpe(oos_q5_net)

        train_sh = float(cand["q5_sharpe_after_cost"])
        # Need validate_sh from the validate decision blob
        val_match = next(
            (r for r in val_blob["all_results"]
             if r["variant"] == v and r["window"] == w
             and r.get("vol_window", w) == vw and r["horizon"] == h
             and r["slippage_bp"] == s and r["quintile_n"] == q),
            None
        )
        validate_sh = val_match["validate_q5_sh_after_cost"] if val_match else float("nan")
        oos_sh = _sharpe(oos_q5_net)
        ic_d1 = ic_o.get(1, float("nan"))

        # Same 4 promotion rules, applied to OOS
        passes = {
            "oos_sh_geq_0p5": bool(oos_sh >= 0.5),
            "oos_ic_d1_geq_0": bool((not np.isnan(ic_d1)) and ic_d1 >= 0),
            "same_sign_train": bool((np.sign(train_sh) == np.sign(oos_sh))
                                    and oos_sh != 0),
            "no_60pct_degradation_vs_validate": bool(
                (validate_sh > 0) and (oos_sh >= 0.4 * validate_sh)
            ),
        }
        deploy = bool(all(passes.values()))

        # Compute drawdown
        cumret = (1 + oos_q5_net).cumprod()
        rolling_max = cumret.cummax()
        max_dd = float(((cumret - rolling_max) / rolling_max).min())

        # Win rate
        win_rate = float((oos_q5_net > 0).mean())

        row = {
            "variant": v, "window": w, "vol_window": vw,
            "horizon": h, "slippage_bp": s, "quintile_n": q,
            "train_sh": round(train_sh, 3),
            "validate_sh": round(validate_sh, 3),
            "oos_q5_sh_after_cost": round(oos_sh, 3),
            "oos_q5_sh_gross": round(_sharpe(oos_q5_gross), 3),
            "oos_ls_sh_after_cost": round(_sharpe(oos_ls_net), 3),
            "oos_ic_at_delay_1": round(ic_d1, 4),
            "oos_n_days": int(len(oos_q5_net)),
            "oos_max_dd": round(max_dd, 4),
            "oos_win_rate": round(win_rate, 3),
            "oos_per_year": {y: round(s_, 3) for y, s_ in py["per_year"].items()},
            **{f"pass_{k}": v_ for k, v_ in passes.items()},
            "DEPLOY": deploy,
        }
        rows.append(row)
        logger.info("=" * 60)
        logger.info("  %s w=%d vw=%d h=%d q=%d slip=%.1f", v, w, vw, h, q, s)
        logger.info("    Train Sh    : %.3f", train_sh)
        logger.info("    Validate Sh : %.3f", validate_sh)
        logger.info("    OOS Sh      : %.3f", oos_sh)
        logger.info("    OOS IC      : %.3f", ic_d1)
        logger.info("    OOS DD      : %.2f%%", max_dd * 100)
        logger.info("    OOS WinRate : %.1f%%", win_rate * 100)
        logger.info("    Per-year    : %s", row["oos_per_year"])
        logger.info("    Checks      : %s", passes)
        logger.info("    DEPLOY      : %s", deploy)

    df = pd.DataFrame(rows)
    df.to_csv(SESSION_DIR / "grid_search_round_0006_oos.csv", index=False)

    deployable = df[df["DEPLOY"]].sort_values("oos_q5_sh_after_cost", ascending=False)
    decision = {
        "oos_window": [str(OOS_START), str(OOS_END)],
        "n_candidates_evaluated": len(rows),
        "n_deployable": int(len(deployable)),
        "champion": (deployable.head(1).to_dict("records")[0]
                     if len(deployable) > 0 else None),
        "all_results": rows,
        "decision": ("DEPLOY (paper trading first)"
                     if len(deployable) > 0
                     else "STOP — OOS audit failed; do not re-tune"),
    }
    with open(SESSION_DIR / "oos_decision_r6.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False, default=str)

    logger.info("=" * 60)
    logger.info("=" * 60)
    logger.info("FINAL DECISION: %s", decision["decision"])
    logger.info("Deployable count: %d / %d", len(deployable), len(rows))
    if len(deployable) > 0:
        logger.info("Champion (OOS): %s", deployable.iloc[0].to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
