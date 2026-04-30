"""Generate publication-quality figures for the README.

Produces (under figures/):
    nav_full.png          — full 2018-2026 NAV curve with Train/Validate/OOS shading
    nav_oos.png           — OOS-only zoom
    drawdown.png          — drawdown curve over full window
    yearly_returns.png    — bar chart of annual excess returns
    sharpe_progression.png — Train→Validate→OOS Sharpe per round
    nav_vs_benchmark.png  — strategy vs cohort-equal-weight basket

Run with OOS_UNLOCKED=true to compute the full curve including OOS.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.costs import CostModel
from backtest.engine import BacktestConfig, run_backtest
from data.adapters.efinance_etf import (
    fetch_etf_panel as ef_fetch_panel, fetch_nav_panel as ef_fetch_nav,
)
from data.universe.build import all_symbols, cohort_map
from strategy.etf_mean_reversion.signals import signal_v9

FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)
RES_DIR = REPO_ROOT / "results"
RES_DIR.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
})

# Window boundaries
TRAIN_END = pd.Timestamp("2022-06-30")
VAL_END = pd.Timestamp("2023-12-31")


def main():
    if os.environ.get("OOS_UNLOCKED", "").lower() not in {"1", "true", "yes"}:
        print("WARNING: OOS_UNLOCKED not set. Plots will only cover IS window.")
        end_date = date(2023, 12, 31)
    else:
        end_date = date(2026, 4, 28)

    print(f"Fetching panel through {end_date} ...")
    panel = ef_fetch_panel(all_symbols(enabled_only=True),
                            date(2018, 1, 1), end_date)
    panel = panel.dropna(how="all", axis=0).ffill(limit=5)
    keep = panel.notna().sum()[panel.notna().sum() >= 200].index.tolist()
    panel = panel[keep]
    nav_panel = ef_fetch_nav(list(panel.columns), date(2018, 1, 1), end_date)
    nav_panel = nav_panel.reindex(panel.index).ffill(limit=5)
    cmap = {s: c for s, c in cohort_map(enabled_only=True).items()
            if s in panel.columns}

    # Run champion
    sig = signal_v9(panel, cohort_map=cmap, window=40, vol_window=40)
    bt = run_backtest(signal=sig, prices=panel, variant="v9_champion",
                      cost=CostModel(slippage_bp=0.5),
                      cfg=BacktestConfig(horizon=21, delay=1, quintile_n=5))

    s = bt.series["q5_ret_net_daily"].dropna()
    s.index = pd.to_datetime(s.index)
    nav = (1 + s).cumprod()
    bench_ret = bt.series["q5_ret_net_daily"] - bt.series["q5_ret_net_daily"]
    # Cohort-equal-weight benchmark (just for visual reference)
    bench_panel = panel.pct_change().mean(axis=1).dropna()
    bench_nav = (1 + bench_panel.loc[s.index.intersection(bench_panel.index)]).cumprod()

    # Save NAV CSV
    nav_df = pd.DataFrame({"strategy_excess_nav": nav, "benchmark_nav": bench_nav.reindex(nav.index)})
    nav_df.to_csv(RES_DIR / "nav_curves.csv")

    # ── Figure 1: NAV full window with shading ────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(nav.index, nav.values, color="#2F73C9", linewidth=1.6,
            label="v9 Champion · Q5 Long-only Excess NAV")
    ax.axhline(1.0, color="#888", linewidth=0.8, linestyle="--")
    # Window shading
    ax.axvspan(s.index.min(), TRAIN_END, alpha=0.07, color="#444",
               label="Train (2018-01 → 2022-06)")
    ax.axvspan(TRAIN_END, VAL_END, alpha=0.10, color="#E84D3D",
               label="Validate (2022-07 → 2023-12)")
    if s.index.max() > VAL_END:
        ax.axvspan(VAL_END, s.index.max(), alpha=0.08, color="#2A9D5F",
                   label="OOS (2024-01 → 2026-04)")
    # Annotate metrics
    ax.set_title("Q5 Long-only Excess NAV · v9 (w=40, vw=40, h=21, q=5, slip=0.5bp)",
                 fontweight="semibold")
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Excess Return (× initial)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    plt.savefig(FIG_DIR / "nav_full.png")
    plt.close()
    print(f"  Wrote {FIG_DIR / 'nav_full.png'}")

    # ── Figure 2: OOS zoom ────────────────────────────────────────────
    if s.index.max() > VAL_END:
        oos_s = s[s.index > VAL_END]
        oos_nav = (1 + oos_s).cumprod()
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(oos_nav.index, oos_nav.values, color="#2A9D5F",
                linewidth=1.8, label="OOS Excess NAV")
        ax.axhline(1.0, color="#888", linewidth=0.8, linestyle="--")
        sh = (oos_s.mean() / oos_s.std(ddof=1)) * np.sqrt(252)
        ann = (oos_nav.iloc[-1] ** (252 / len(oos_s)) - 1) * 100
        dd = ((oos_nav - oos_nav.cummax()) / oos_nav.cummax()).min() * 100
        ax.set_title(
            f"OOS (2024-01 → 2026-04) · Sharpe {sh:.2f} · Ann {ann:+.2f}% · MaxDD {dd:+.2f}%",
            fontweight="semibold")
        ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Excess Return")
        ax.grid(True, alpha=0.3)
        plt.savefig(FIG_DIR / "nav_oos.png")
        plt.close()
        print(f"  Wrote {FIG_DIR / 'nav_oos.png'}")

    # ── Figure 3: Drawdown ────────────────────────────────────────────
    rolling_max = nav.cummax()
    dd_series = (nav - rolling_max) / rolling_max * 100
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.fill_between(dd_series.index, dd_series.values, 0,
                    color="#E84D3D", alpha=0.4, linewidth=0)
    ax.plot(dd_series.index, dd_series.values, color="#B22D24", linewidth=0.9)
    ax.axvspan(s.index.min(), TRAIN_END, alpha=0.04, color="#444")
    ax.axvspan(TRAIN_END, VAL_END, alpha=0.07, color="#E84D3D")
    if s.index.max() > VAL_END:
        ax.axvspan(VAL_END, s.index.max(), alpha=0.05, color="#2A9D5F")
    ax.set_title(f"Drawdown · MaxDD {dd_series.min():+.2f}%", fontweight="semibold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    plt.savefig(FIG_DIR / "drawdown.png")
    plt.close()
    print(f"  Wrote {FIG_DIR / 'drawdown.png'}")

    # ── Figure 4: Yearly bar chart ────────────────────────────────────
    yearly = []
    for year, g in s.groupby(s.index.year):
        if len(g) < 2: continue
        cum = (1 + g).cumprod().iloc[-1]
        ann = (cum ** (252 / len(g)) - 1) * 100
        sh = (g.mean() / g.std(ddof=1)) * np.sqrt(252) if g.std(ddof=1) > 0 else float("nan")
        yearly.append((year, ann, sh, len(g)))
    yearly_df = pd.DataFrame(yearly, columns=["year", "ann_ret", "sharpe", "days"])
    yearly_df.to_csv(RES_DIR / "yearly_returns.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = []
    for y in yearly_df["year"]:
        if y <= 2022:   colors.append("#777")          # Train
        elif y == 2023: colors.append("#E84D3D")       # Validate
        else:           colors.append("#2A9D5F")       # OOS
    bars = ax.bar(yearly_df["year"].astype(str), yearly_df["ann_ret"],
                   color=colors, edgecolor="white")
    for bar, ret, sh in zip(bars, yearly_df["ann_ret"], yearly_df["sharpe"]):
        h = bar.get_height()
        offset = 0.4 if h >= 0 else -0.4
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                f"{ret:+.1f}%\nSh {sh:+.2f}",
                ha="center", va="bottom" if h >= 0 else "top",
                fontsize=9)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_title("Annualized Excess Return per Year (Q5 long-only after-cost)",
                 fontweight="semibold")
    ax.set_ylabel("Annualized Excess (%)")
    ax.grid(True, alpha=0.3, axis="y")
    # Custom legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#777", label="Train (2018-2022 H1)"),
        Patch(facecolor="#E84D3D", label="Validate (2022 H2-2023)"),
        Patch(facecolor="#2A9D5F", label="OOS (2024-2026)"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=9)
    plt.savefig(FIG_DIR / "yearly_returns.png")
    plt.close()
    print(f"  Wrote {FIG_DIR / 'yearly_returns.png'}")

    # ── Figure 5: Sharpe progression across rounds ────────────────────
    rounds = ["r1", "r2", "r3", "r4", "r5", "r6 Validate", "r6 OOS"]
    sharpes = [-0.69, 1.91, 2.95, 0.39, 0.01, 1.18, 1.36]
    notes = [
        "v1 baseline\n(true)",
        "v2 (had 1-bar leak)",
        "v8 (leak + Train\noverfit)",
        "v8 Validate\nfirst honest TVT",
        "v9 Validate\nfalsified",
        "v9 Validate\nchampion",
        "v9 OOS\nDEPLOY ✓",
    ]
    statuses = ["true", "leak", "leak+overfit",
                "no_promote", "no_promote", "promote", "deploy"]
    color_map = {
        "true": "#888", "leak": "#FFA500", "leak+overfit": "#E84D3D",
        "no_promote": "#888", "promote": "#3B97D6", "deploy": "#2A9D5F"
    }
    colors = [color_map[s] for s in statuses]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(rounds, sharpes, color=colors, edgecolor="white")
    for bar, sh, note in zip(bars, sharpes, notes):
        h = bar.get_height()
        offset = 0.08 if h >= 0 else -0.08
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                f"{sh:+.2f}\n{note}",
                ha="center", va="bottom" if h >= 0 else "top", fontsize=9)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.axhline(0.5, color="#2A9D5F", linewidth=1.0, linestyle="--",
               alpha=0.5, label="Promotion floor (Sh ≥ 0.5)")
    ax.set_title("Sharpe Progression Across 6 Rounds (Honest TVT Discipline)",
                 fontweight="semibold")
    ax.set_ylabel("Q5 Long-only Sharpe (after-cost)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=10)
    plt.savefig(FIG_DIR / "sharpe_progression.png")
    plt.close()
    print(f"  Wrote {FIG_DIR / 'sharpe_progression.png'}")

    # ── Figure 6: NAV vs benchmark ────────────────────────────────────
    if not bench_nav.empty:
        common = nav.index.intersection(bench_nav.index)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(common, nav.loc[common], color="#2A9D5F", linewidth=1.8,
                label=f"Strategy (Q5 long-only excess) · Final {nav.iloc[-1]:.3f}")
        bench_nav_norm = bench_nav.loc[common] / bench_nav.loc[common].iloc[0]
        ax.plot(common, bench_nav_norm, color="#888", linewidth=1.4,
                label=f"Cohort Equal-Weight Benchmark · Final {bench_nav_norm.iloc[-1]:.3f}",
                alpha=0.7)
        ax.axhline(1.0, color="#444", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_title("Strategy Excess NAV vs Cohort Equal-Weight Benchmark",
                     fontweight="semibold")
        ax.set_xlabel("Date"); ax.set_ylabel("Normalized NAV")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=10)
        plt.savefig(FIG_DIR / "nav_vs_benchmark.png")
        plt.close()
        print(f"  Wrote {FIG_DIR / 'nav_vs_benchmark.png'}")

    # ── Save the metrics summary ──────────────────────────────────────
    metrics_summary = {
        "Train":    {"start": "2018-01-02", "end": "2022-06-30",
                     "days": int((s.index <= TRAIN_END).sum())},
        "Validate": {"start": "2022-07-01", "end": "2023-12-31",
                     "days": int(((s.index > TRAIN_END) & (s.index <= VAL_END)).sum())},
        "OOS":      {"start": "2024-01-01", "end": "2026-04-28",
                     "days": int((s.index > VAL_END).sum())},
        "Full":     {"start": str(s.index.min().date()),
                     "end":   str(s.index.max().date()),
                     "days": int(len(s))},
    }
    for win, info in metrics_summary.items():
        if win == "Train":   sub = s[s.index <= TRAIN_END]
        elif win == "Validate": sub = s[(s.index > TRAIN_END) & (s.index <= VAL_END)]
        elif win == "OOS":   sub = s[s.index > VAL_END]
        else:                sub = s
        if len(sub) < 2: continue
        cum = (1 + sub).cumprod()
        ann = (cum.iloc[-1] ** (252 / len(sub)) - 1) * 100
        vol = sub.std(ddof=1) * np.sqrt(252) * 100
        sh  = ann / vol if vol > 0 else float("nan")
        rmax = cum.cummax(); dd = ((cum - rmax) / rmax).min() * 100
        win_rate = (sub > 0).mean() * 100
        info.update({"ann_ret_pct": round(ann, 3), "vol_pct": round(vol, 3),
                     "sharpe": round(sh, 3), "max_dd_pct": round(dd, 3),
                     "win_rate_pct": round(win_rate, 2)})
    pd.DataFrame(metrics_summary).T.to_csv(RES_DIR / "metrics_summary.csv")
    print(f"  Wrote {RES_DIR / 'metrics_summary.csv'}")
    print()
    print("All figures generated successfully.")


if __name__ == "__main__":
    main()
