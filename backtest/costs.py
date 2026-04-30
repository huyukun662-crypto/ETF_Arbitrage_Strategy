"""Transaction cost model for ETF arbitrage backtests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission_oneway: float = 0.00005   # 0.5 bp negotiable retail rate
    slippage_bp: float = 5.0             # 5 bp impact assumption
    stamp_duty: float = 0.0              # ETF secondary exempt
    borrow_cost_apr: float = 0.08        # short leg (often unavailable)

    def round_trip_bps(self, short_leg: bool = False, hold_days: int = 5) -> float:
        """Total round-trip cost in basis points for one leg, one round trip.

        commission * 2 + slippage * 2 (in/out) + optional borrow accrual.
        """
        cost_bp = (self.commission_oneway * 2 + self.slippage_bp / 1e4 * 2) * 1e4
        if short_leg:
            # borrow cost accrues over hold period
            cost_bp += self.borrow_cost_apr * (hold_days / 252.0) * 1e4
        return cost_bp
