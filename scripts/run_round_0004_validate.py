"""Round 0004 — Validate gate (2022-07-01 → 2023-12-31).

Single-shot evaluation of the top-K candidates produced by
`run_round_0004_train.py`. NO parameter tuning happens here; we only
compute Validate metrics and apply promotion rules.

Promotion rules (any candidate that passes ALL = champion):
  1) Validate Q5 Sharpe ≥ 0.5
  2) Validate IC at delay=1 ≥ 0.0
  3) sign(Validate Sh) == sign(Train Sh)
  4) Validate Sh ≥ 0.4 × Train Sh   (no >60% degradation)

If a champion emerges, OOS becomes ELIGIBLE for unlock — but the script
still does NOT load OOS data. The user must explicitly approve.

Outputs:
    grid_search_round_0004_validate.csv
    validate_decision.json     (champion + decision rationale)
"""

from __future__ import annotations

import json
import logging
import sys
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
    TRAIN_END, VALIDATE_END, VALIDATE_START, is_oos_unlocked,
)                                                     # noqa: E402
from data.universe.build import all_symbols, cohort_map  # noqa: E402
from strategy.etf_mean_reversion.signals import (
    signal_v1, signal_v2, signal_v4, signal_v8, signal_v9,
)                                                     # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("round_0004_validate")

SESSION_DIR = REPO_ROOT / "logs" / "20260428_etf_meanrev_arbitrage"


def make_signal(variant, window, threshold, panel, cmap):
    if variant == "v1":  return signal_v1(panel, window=window)
    if variant == "v2":  return signal_v2(panel, cohort_map=cmap, window=window)
    if variant == "v4":  return signal_v4(panel, cohort_map=cmap, window=window)
    if variant == "v8":  return signal_v8(panel, cohort_map=cmap, window=window, threshold=threshold)
    if variant == "v9":  return signal_v9(panel, cohort_map=cmap, window=window, vol_window=window)
    raise ValueError(variant)


def main() -> int:
    if is_oos_unlocked():
        logger.error("OOS_UNLOCKED set; refusing.")
        return 2

    cands_path = SESSION_DIR / "train_top_candidates.json"
    if not cands_path.exists():
        logger.error("Missing %s — run round_0004 train first.", cands_path)
        return 3
    cands_blob = json.load(open(cands_path, encoding="utf-8"))
    candidates = cands_blob["candidates"]
    logger.info("Loaded %d candidates from Train phase", len(candidates))

    # IMPORTANT: signal computation needs lookback that crosses Train→Validate boundary.
    # Strategy: fetch [Train.start - lookback_buffer, Validate.end] panel,
    # compute signal on whole panel, then SLICE to Validate window for backtest.
    # This is fine because the look-ahead audit already guaranteed signal at
    # validate-day t uses only data ≤ t-1.
    LOOKBACK_BUFFER_DAYS = 250  # generous: covers any window/horizon combo
    fetch_start_date_str = (pd.Timestamp(VALIDATE_START) - pd.Timedelta(days=LOOKBACK_BUFFER_DAYS*1.6)).date()
    fetch_start = max(fetch_start_date_str, pd.Timestamp("2018-01-01").date())
    logger.info("Fetching panel from %s to %s (Validate end)", fetch_start, VALIDATE_END)
    panel_full = fetch_etf_panel(all_symbols(), fetch_start, VALIDATE_END, field="close")
    panel_full = panel_full.dropna(how="all", axis=0).ffill(limit=5)
    coverage = panel_full.notna().sum()
    keep = coverage[coverage >= 200].index.tolist()
    panel_full = panel_full[keep]
    cmap = {s: c for s, c in cohort_map().items() if s in panel_full.columns}
    logger.info("Full panel: %s; cols: %s", panel_full.shape, list(panel_full.columns))

    # Slice mask: rows in Validate window.
    valid_mask = (panel_full.index >= pd.Timestamp(VALIDATE_START)) & \
                 (panel_full.index <= pd.Timestamp(VALIDATE_END))
    valid_dates = panel_full.index[valid_mask]
    logger.info("Validate window has %d trading days (%s to %s)",
                len(valid_dates), valid_dates.min().date(), valid_dates.max().date())

    rows: list[dict] = []
    cost = CostModel(commission_oneway=0.00005)

    for cand in candidates:
        v = cand["variant"]; w = int(cand["window"]); h = int(cand["horizon"])
        s = float(cand["slippage_bp"]); q = int(cand["quintile_n"])
        t = float(cand["threshold"])

        passed, _ = lookahead_invariance(
            signal_fn=lambda p: make_signal(v, w, t, p, cmap),
            prices=panel_full,
        )
        if not passed:
            logger.error("AUDIT FAIL %s — skipping", cand)
            continue

        sig_full = make_signal(v, w, t, panel_full, cmap)

        # Run backtest on the FULL panel (need lookback) but only score on Validate window.
        cost_v = CostModel(commission_oneway=0.00005, slippage_bp=s)
        bt_full = run_backtest(signal=sig_full, prices=panel_full, variant=v,
                               cost=cost_v,
                               cfg=BacktestConfig(horizon=h, delay=1,
                                                  quintile_n=q, min_universe_size=4))

        # Slice the per-day return series to Validate window
        ser = bt_full.series.copy()
        v_ser = ser.loc[(ser.index >= pd.Timestamp(VALIDATE_START)) &
                        (ser.index <= pd.Timestamp(VALIDATE_END))]

        from backtest.audits import _sharpe, per_year_sharpe, ic_decay_by_delay
        v_q5_net = v_ser["q5_ret_net_daily"].dropna()
        v_q5_gross = v_ser["q5_ret_gross_daily"].dropna()
        v_ls_net = v_ser["ls_ret_net_daily"].dropna()

        # IC on Validate
        sig_v = sig_full.loc[(sig_full.index >= pd.Timestamp(VALIDATE_START)) &
                             (sig_full.index <= pd.Timestamp(VALIDATE_END))]
        # forward returns at validate dates
        fwd_log = np.log(panel_full).diff(h).shift(-(1 + h))
        fwd = np.exp(fwd_log) - 1
        fwd_v = fwd.loc[sig_v.index.intersection(fwd.index)]
        ic_decay_v = ic_decay_by_delay(signal=sig_v, forward_returns=fwd_v)

        py = per_year_sharpe(v_q5_net)

        train_sh = float(cand["q5_sharpe_after_cost"])
        validate_sh = _sharpe(v_q5_net)
        ic_v_d1 = ic_decay_v.get(1, float("nan"))

        # Promotion rules
        passes = {
            "validate_sh_geq_0p5": validate_sh >= 0.5,
            "validate_ic_d1_geq_0": (not np.isnan(ic_v_d1)) and ic_v_d1 >= 0.0,
            "same_sign_train": (np.sign(train_sh) == np.sign(validate_sh)) if validate_sh != 0 else False,
            "no_60pct_degradation": (validate_sh >= 0.4 * train_sh) if train_sh > 0 else False,
        }
        promote = all(passes.values())

        row = {
            "variant": v, "window": w, "horizon": h, "slippage_bp": s,
            "quintile_n": q, "threshold": t,
            "train_sh": round(train_sh, 3),
            "validate_q5_sh_after_cost": round(validate_sh, 3),
            "validate_q5_sh_gross": round(_sharpe(v_q5_gross), 3),
            "validate_ls_sh_after_cost": round(_sharpe(v_ls_net), 3),
            "validate_ic_at_delay_1": round(ic_v_d1, 4),
            "validate_n_days": int(len(v_q5_net)),
            "validate_per_year": {y: round(s, 3) for y, s in py["per_year"].items()},
            **{f"pass_{k}": v for k, v in passes.items()},
            "PROMOTE": promote,
        }
        rows.append(row)

        logger.info("  %s w=%d h=%d q=%d t=%.1f  Train Sh %.3f -> Validate Sh %.3f  IC %.3f  PROMOTE=%s",
                    v, w, h, q, t, train_sh, validate_sh, ic_v_d1, promote)
        logger.info("    Validate per-year: %s", row["validate_per_year"])
        logger.info("    Promotion checks: %s", passes)

    df = pd.DataFrame(rows)
    out_csv = SESSION_DIR / "grid_search_round_0004_validate.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s", out_csv)

    champions = df[df["PROMOTE"] == True].sort_values("validate_q5_sh_after_cost", ascending=False)
    decision = {
        "validate_window": [str(VALIDATE_START), str(VALIDATE_END)],
        "n_candidates_evaluated": len(rows),
        "n_promoted": int(len(champions)),
        "champion": (champions.head(1).to_dict("records")[0] if len(champions) > 0 else None),
        "all_results": rows,
        "oos_unlock_eligible": int(len(champions) > 0),
        "oos_unlock_status": "ELIGIBLE — pending user approval" if len(champions) > 0 else "BLOCKED — no candidate passed Validate gate",
    }
    with open(SESSION_DIR / "validate_decision.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False, default=str)

    logger.info("=" * 60)
    logger.info("PROMOTED candidates: %d", len(champions))
    if len(champions) > 0:
        logger.info("Champion: %s", champions.iloc[0].to_dict())
    else:
        logger.info("No candidate cleared the Validate gate. OOS remains LOCKED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
