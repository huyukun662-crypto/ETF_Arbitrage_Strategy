"""Round 0005 — Validate gate (2022-07-01 → 2023-12-31).

Reads candidates from `train_top_candidates_r5.json` and runs single-shot
Validate evaluation. Promotion rules unchanged from round_0004:
  1) Validate Q5 Sharpe ≥ 0.5
  2) Validate IC at delay=1 ≥ 0.0
  3) sign(Validate Sh) == sign(Train Sh)
  4) Validate Sh ≥ 0.4 × Train Sh   (no >60% degradation)

Outputs:
    grid_search_round_0005_validate.csv
    validate_decision_r5.json
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

from backtest.audits import (
    _sharpe, ic_decay_by_delay, lookahead_invariance, per_year_sharpe,
)                                                     # noqa: E402
from backtest.costs import CostModel                  # noqa: E402
from backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from data.adapters.akshare_etf import fetch_etf_panel # noqa: E402
from data.splits import (
    VALIDATE_END, VALIDATE_START, is_oos_unlocked,
)                                                     # noqa: E402
from data.universe.build import all_symbols, cohort_map  # noqa: E402
from strategy.etf_mean_reversion.signals import (
    signal_v2, signal_v8, signal_v9,
)                                                     # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("round_0005_validate")

SESSION_DIR = REPO_ROOT / "logs" / "20260428_etf_meanrev_arbitrage"


def make_signal(variant, window, threshold, vol_window, panel, cmap):
    if variant == "v2":  return signal_v2(panel, cohort_map=cmap, window=window)
    if variant == "v8":  return signal_v8(panel, cohort_map=cmap, window=window, threshold=threshold)
    if variant == "v9":  return signal_v9(panel, cohort_map=cmap, window=window, vol_window=vol_window)
    raise ValueError(variant)


def main() -> int:
    if is_oos_unlocked():
        logger.error("OOS_UNLOCKED set; refusing.")
        return 2

    cands_path = SESSION_DIR / "train_top_candidates_r5.json"
    if not cands_path.exists():
        logger.error("Missing %s — run round_0005 train first.", cands_path)
        return 3
    cands_blob = json.load(open(cands_path, encoding="utf-8"))
    candidates = cands_blob["candidates"]
    logger.info("Loaded %d candidates from r5 Train phase", len(candidates))

    LOOKBACK_BUFFER_DAYS = 250
    fetch_start = (pd.Timestamp(VALIDATE_START) -
                   pd.Timedelta(days=int(LOOKBACK_BUFFER_DAYS * 1.6))).date()
    fetch_start = max(fetch_start, pd.Timestamp("2018-01-01").date())
    logger.info("Fetching panel from %s to %s (validate end)",
                fetch_start, VALIDATE_END)

    panel_full = fetch_etf_panel(all_symbols(enabled_only=True),
                                 fetch_start, VALIDATE_END, field="close")
    panel_full = panel_full.dropna(how="all", axis=0).ffill(limit=5)
    coverage = panel_full.notna().sum()
    keep = coverage[coverage >= 200].index.tolist()
    panel_full = panel_full[keep]
    cmap = {s: c for s, c in cohort_map(enabled_only=True).items()
            if s in panel_full.columns}
    logger.info("Full panel: %s; cols: %s", panel_full.shape, list(panel_full.columns))

    valid_mask = ((panel_full.index >= pd.Timestamp(VALIDATE_START)) &
                  (panel_full.index <= pd.Timestamp(VALIDATE_END)))
    valid_dates = panel_full.index[valid_mask]
    logger.info("Validate window has %d trading days (%s to %s)",
                len(valid_dates),
                valid_dates.min().date(), valid_dates.max().date())

    rows: list[dict] = []
    cost = CostModel(commission_oneway=0.00005)

    for cand in candidates:
        v = cand["variant"]; w = int(cand["window"]); h = int(cand["horizon"])
        s = float(cand["slippage_bp"]); q = int(cand["quintile_n"])
        t = float(cand["threshold"])
        vw = int(cand.get("vol_window", w))

        passed, _ = lookahead_invariance(
            signal_fn=lambda p: make_signal(v, w, t, vw, p, cmap),
            prices=panel_full,
        )
        if not passed:
            logger.error("AUDIT FAIL %s — skipping", cand)
            continue

        sig_full = make_signal(v, w, t, vw, panel_full, cmap)
        cost_v = CostModel(commission_oneway=0.00005, slippage_bp=s)
        bt_full = run_backtest(signal=sig_full, prices=panel_full, variant=v,
                               cost=cost_v,
                               cfg=BacktestConfig(horizon=h, delay=1,
                                                  quintile_n=q,
                                                  min_universe_size=4))

        ser = bt_full.series.copy()
        v_ser = ser.loc[(ser.index >= pd.Timestamp(VALIDATE_START)) &
                        (ser.index <= pd.Timestamp(VALIDATE_END))]
        v_q5_net = v_ser["q5_ret_net_daily"].dropna()
        v_q5_gross = v_ser["q5_ret_gross_daily"].dropna()
        v_ls_net = v_ser["ls_ret_net_daily"].dropna()

        sig_v = sig_full.loc[(sig_full.index >= pd.Timestamp(VALIDATE_START)) &
                             (sig_full.index <= pd.Timestamp(VALIDATE_END))]
        fwd_log = np.log(panel_full).diff(h).shift(-(1 + h))
        fwd = np.exp(fwd_log) - 1
        fwd_v = fwd.loc[sig_v.index.intersection(fwd.index)]
        ic_decay_v = ic_decay_by_delay(signal=sig_v, forward_returns=fwd_v)
        py = per_year_sharpe(v_q5_net)

        train_sh = float(cand["q5_sharpe_after_cost"])
        validate_sh = _sharpe(v_q5_net)
        ic_v_d1 = ic_decay_v.get(1, float("nan"))

        passes = {
            "validate_sh_geq_0p5": bool(validate_sh >= 0.5),
            "validate_ic_d1_geq_0": bool((not np.isnan(ic_v_d1)) and ic_v_d1 >= 0.0),
            "same_sign_train": bool((np.sign(train_sh) == np.sign(validate_sh))
                                    and validate_sh != 0),
            "no_60pct_degradation": bool((train_sh > 0) and
                                         (validate_sh >= 0.4 * train_sh)),
        }
        promote = bool(all(passes.values()))

        row = {
            "variant": v, "window": w, "vol_window": vw, "horizon": h,
            "slippage_bp": s, "quintile_n": q, "threshold": t,
            "train_sh": round(train_sh, 3),
            "validate_q5_sh_after_cost": round(validate_sh, 3),
            "validate_q5_sh_gross": round(_sharpe(v_q5_gross), 3),
            "validate_ls_sh_after_cost": round(_sharpe(v_ls_net), 3),
            "validate_ic_at_delay_1": round(ic_v_d1, 4),
            "validate_n_days": int(len(v_q5_net)),
            "validate_per_year": {y: round(s_, 3) for y, s_ in py["per_year"].items()},
            **{f"pass_{k}": v_ for k, v_ in passes.items()},
            "PROMOTE": promote,
        }
        rows.append(row)

        logger.info("  %s w=%d vw=%d h=%d q=%d t=%.1f  "
                    "Train Sh %.3f -> Validate Sh %.3f  IC %.3f  PROMOTE=%s",
                    v, w, vw, h, q, t, train_sh, validate_sh, ic_v_d1, promote)
        logger.info("    Per-year: %s   checks: %s", row["validate_per_year"], passes)

    df = pd.DataFrame(rows)
    out_csv = SESSION_DIR / "grid_search_round_0005_validate.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s", out_csv)

    champions = df[df["PROMOTE"]].sort_values("validate_q5_sh_after_cost", ascending=False)
    decision = {
        "validate_window": [str(VALIDATE_START), str(VALIDATE_END)],
        "n_candidates_evaluated": len(rows),
        "n_promoted": int(len(champions)),
        "champion": (champions.head(1).to_dict("records")[0]
                     if len(champions) > 0 else None),
        "all_results": rows,
        "oos_unlock_eligible": int(len(champions) > 0),
        "oos_unlock_status": ("ELIGIBLE — pending user approval"
                              if len(champions) > 0
                              else "BLOCKED — no candidate passed Validate gate"),
    }
    with open(SESSION_DIR / "validate_decision_r5.json", "w", encoding="utf-8") as f:
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
