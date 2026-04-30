"""Round 0006 — Validate gate. Reads train_top_candidates_r6.json."""

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
from data.adapters.efinance_etf import (
    fetch_etf_panel as ef_fetch_panel, fetch_nav_panel as ef_fetch_nav,
)                                                     # noqa: E402
from data.splits import (
    VALIDATE_END, VALIDATE_START, is_oos_unlocked,
)                                                     # noqa: E402
from data.universe.build import all_symbols, cohort_map  # noqa: E402
from strategy.etf_mean_reversion.signals import (
    signal_v2, signal_v6, signal_v9,
)                                                     # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("round_0006_validate")

SESSION_DIR = REPO_ROOT / "logs" / "20260428_etf_meanrev_arbitrage"


def main() -> int:
    if is_oos_unlocked():
        logger.error("OOS_UNLOCKED set; refusing.")
        return 2

    cands_path = SESSION_DIR / "train_top_candidates_r6.json"
    if not cands_path.exists():
        logger.error("Missing %s", cands_path)
        return 3
    blob = json.load(open(cands_path, encoding="utf-8"))
    candidates = blob["candidates"]
    logger.info("Loaded %d candidates from r6 Train", len(candidates))

    LOOKBACK_DAYS = 250
    fetch_start = (pd.Timestamp(VALIDATE_START) -
                   pd.Timedelta(days=int(LOOKBACK_DAYS * 1.6))).date()
    fetch_start = max(fetch_start, pd.Timestamp("2018-01-01").date())
    logger.info("Fetching panel from %s to %s",
                fetch_start, VALIDATE_END)

    panel = ef_fetch_panel(all_symbols(enabled_only=True),
                           fetch_start, VALIDATE_END, field="close")
    panel = panel.dropna(how="all", axis=0).ffill(limit=5)
    coverage = panel.notna().sum()
    keep = coverage[coverage >= 200].index.tolist()
    panel = panel[keep]
    nav = ef_fetch_nav(list(panel.columns), fetch_start, VALIDATE_END)
    nav = nav.reindex(panel.index).ffill(limit=5)
    nav = nav[[c for c in panel.columns if c in nav.columns]]
    cmap = {s: c for s, c in cohort_map(enabled_only=True).items()
            if s in panel.columns}
    logger.info("Full panel: %s; nav cols: %d",
                panel.shape, nav.shape[1] if not nav.empty else 0)

    def make_signal(v, w, vw, panel_, nav_, cmap_):
        if v == "v2":  return signal_v2(panel_, cohort_map=cmap_, window=w)
        if v == "v6":  return signal_v6(panel_, nav=nav_, cohort_map=cmap_, window=w)
        if v == "v9":  return signal_v9(panel_, cohort_map=cmap_,
                                          window=w, vol_window=vw)
        raise ValueError(v)

    rows: list[dict] = []
    for cand in candidates:
        v = cand["variant"]; w = int(cand["window"]); h = int(cand["horizon"])
        s = float(cand["slippage_bp"]); q = int(cand["quintile_n"])
        vw = int(cand.get("vol_window", w))

        try:
            passed, _ = lookahead_invariance(
                signal_fn=lambda p: make_signal(v, w, vw, p, nav, cmap),
                prices=panel,
            )
        except Exception as e:                          # noqa: BLE001
            logger.warning("Audit err: %r", e)
            passed = False
        if not passed:
            logger.error("AUDIT FAIL %s; skipping", cand)
            continue

        sig_full = make_signal(v, w, vw, panel, nav, cmap)
        cost_v = CostModel(commission_oneway=0.00005, slippage_bp=s)
        bt_full = run_backtest(signal=sig_full, prices=panel, variant=v,
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
        fwd_log = np.log(panel).diff(h).shift(-(1 + h))
        fwd = np.exp(fwd_log) - 1
        fwd_v = fwd.loc[sig_v.index.intersection(fwd.index)]
        ic_v = ic_decay_by_delay(signal=sig_v, forward_returns=fwd_v)
        py = per_year_sharpe(v_q5_net)

        train_sh = float(cand["q5_sharpe_after_cost"])
        validate_sh = _sharpe(v_q5_net)
        ic_d1 = ic_v.get(1, float("nan"))

        passes = {
            "validate_sh_geq_0p5": bool(validate_sh >= 0.5),
            "validate_ic_d1_geq_0": bool((not np.isnan(ic_d1)) and ic_d1 >= 0),
            "same_sign_train": bool((np.sign(train_sh) == np.sign(validate_sh))
                                    and validate_sh != 0),
            "no_60pct_degradation": bool((train_sh > 0) and
                                         (validate_sh >= 0.4 * train_sh)),
        }
        promote = bool(all(passes.values()))

        row = {
            "variant": v, "window": w, "vol_window": vw,
            "horizon": h, "slippage_bp": s, "quintile_n": q,
            "train_sh": round(train_sh, 3),
            "validate_q5_sh_after_cost": round(validate_sh, 3),
            "validate_q5_sh_gross": round(_sharpe(v_q5_gross), 3),
            "validate_ls_sh_after_cost": round(_sharpe(v_ls_net), 3),
            "validate_ic_at_delay_1": round(ic_d1, 4),
            "validate_n_days": int(len(v_q5_net)),
            "validate_per_year": {y: round(s_, 3) for y, s_ in py["per_year"].items()},
            **{f"pass_{k}": v_ for k, v_ in passes.items()},
            "PROMOTE": promote,
        }
        rows.append(row)
        logger.info("  %s w=%d vw=%d h=%d q=%d  Train %.3f -> Validate %.3f  IC %.3f  PROMOTE=%s",
                    v, w, vw, h, q, train_sh, validate_sh, ic_d1, promote)
        logger.info("    Per-year: %s   checks: %s",
                    row["validate_per_year"], passes)

    df = pd.DataFrame(rows)
    df.to_csv(SESSION_DIR / "grid_search_round_0006_validate.csv", index=False)

    champions = df[df["PROMOTE"]].sort_values("validate_q5_sh_after_cost",
                                                ascending=False)
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
                              else "BLOCKED — no candidate passed"),
    }
    with open(SESSION_DIR / "validate_decision_r6.json", "w",
              encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False, default=str)

    logger.info("=" * 60)
    logger.info("PROMOTED: %d", len(champions))
    if len(champions):
        logger.info("Champion: %s", champions.iloc[0].to_dict())
    else:
        logger.info("No candidate cleared. OOS LOCKED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
