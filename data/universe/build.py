"""ETF universe + cohort definitions for round_0001.

Hardcoded for round_0001 because AkShare's `fund_etf_spot_em` discovery
endpoint hangs in this environment. Production builds should replace
this with a discovery-based universe + filters from session_metadata.yml.

Cohort design rationale:
- Each cohort contains ≥3 ETFs tracking the same underlying (or same
  benchmark index). Within-cohort cross-sectional deviation is the
  fair-value reference for v2.
- Gold cohort uses true co-tracking ETFs (all AU9999 / SHAU).
- HS300 / ZZ500 cohorts use multiple issuers tracking the same equity index.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CohortDef:
    name: str
    benchmark: str
    asset_class: str            # "gold" | "equity_index" | "qdii_equity" | "bond"
    members: tuple[str, ...]
    enabled: bool = True        # set False to skip the cohort entirely


COHORTS: tuple[CohortDef, ...] = (
    CohortDef(
        name="gold",
        benchmark="AU9999/SHAU",
        asset_class="gold",
        members=("518880", "159934", "518800", "159937"),
    ),
    CohortDef(
        name="hs300",
        benchmark="CSI 300",
        asset_class="equity_index",
        members=("510300", "159919", "510310", "510330"),
    ),
    CohortDef(
        name="zz500",
        benchmark="CSI 500",
        asset_class="equity_index",
        members=("510500", "159922", "510510", "159967"),
    ),
    CohortDef(
        name="nasdaq100",
        benchmark="NASDAQ 100",
        asset_class="qdii_equity",
        members=("513100", "159834", "159632", "159509"),
        enabled=False,           # Round 0005: cached members all <500 rows
    ),
    CohortDef(
        name="sp500",
        benchmark="S&P 500",
        asset_class="qdii_equity",
        members=("513500", "159612", "513650", "513260"),
        enabled=False,           # Round 0005: 0 of 4 fetched
    ),
    CohortDef(
        name="hang_seng",
        benchmark="Hang Seng",
        asset_class="qdii_equity",
        members=("159920", "513660", "159607", "513600"),
    ),
    CohortDef(
        name="cy50",
        benchmark="ChiNext 50",
        asset_class="equity_index",
        members=("159949", "159952", "159682", "159781"),
        enabled=False,           # Round 0005: only 159781 cached, singleton
    ),
    # Round 0005: bond cohort attempted but ALL 4 symbols rate-limited by
    # AkShare. Disabled. Re-enable in future round if alternative data source
    # (e.g. TuShare with token) becomes available.
    CohortDef(
        name="bond_treasury",
        benchmark="国债",
        asset_class="bond",
        members=("511010", "511260", "511020", "511220"),
        enabled=False,
    ),
)


def all_symbols(enabled_only: bool = False) -> list[str]:
    if enabled_only:
        return [s for c in COHORTS if c.enabled for s in c.members]
    return [s for c in COHORTS for s in c.members]


def cohort_of(symbol: str) -> str | None:
    for c in COHORTS:
        if symbol in c.members:
            return c.name
    return None


def cohort_map(enabled_only: bool = False) -> dict[str, str]:
    if enabled_only:
        return {s: c.name for c in COHORTS if c.enabled for s in c.members}
    return {s: c.name for c in COHORTS for s in c.members}
