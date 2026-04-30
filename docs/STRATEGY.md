# Strategy Specification — v9 Cohort-Relative Inverse-Vol Mean Reversion

> Definitive reference for the production champion. Read this before paper-trading.

---

## 1. Mechanism (one-sentence)

Within a cohort of A-share ETFs that track the same benchmark, deviations from the cohort median price mean-revert on a ~21-day horizon; weighting positions by inverse realized volatility (per-asset risk parity) on the deviation z-score signal generates a stable long-only Q5 excess return over the cohort-equal-weight benchmark.

## 2. Universe

15 ETFs across 4 cohorts (all members ≥ 1086 trading days history, validated 2018-01 to 2026-04):

| Cohort | Benchmark | Members | Asset class |
|--------|-----------|---------|-------------|
| `gold` | AU9999 / SHAU | 518880, 159934, 518800, 159937 | Gold |
| `hs300` | CSI 300 | 510300, 159919, 510310, 510330 | Equity index |
| `zz500` | CSI 500 | 510500, 159922, 510510, 159967 | Equity index |
| `hang_seng` | Hang Seng | 159920, 513660, 513600 | Cross-border equity |

Cohort minimum size 3 is a **hard requirement** — below this, cohort-median demeaning is unstable.

Disabled cohorts (insufficient history or singleton):
- `nasdaq100`, `sp500`, `cy50`, `bond_treasury` — see `data/universe/build.py` for `enabled=False` flag

## 3. Signal Construction

### 3.1 Step-by-step

```
Inputs:
  prices: DataFrame[date × symbol]  (后复权 close prices, daily)
  cohort_map: dict[symbol → cohort name]

Step 1 (cohort-relative log-price):
  log_p = log(prices)
  cohort_relative[s] = log_p[s] - cohort_median(log_p[s.cohort])

Step 2 (rolling z-score, look-ahead-safe):
  past_relative = cohort_relative.shift(1)             # CRITICAL: t-1 ending
  mu = past_relative.rolling(40, min_periods=40).mean()
  sd = past_relative.rolling(40, min_periods=40).std(ddof=1)
  z = (cohort_relative - mu) / sd                       # uses today's value but past stats

Step 3 (inverse realized vol):
  log_returns = log_p.diff().shift(1)                   # past returns only
  realized_vol = log_returns.rolling(40, min_periods=40).std(ddof=1)
  inv_vol = 1.0 / realized_vol

Step 4 (signal = mean-reversion + risk parity):
  signal = -z × inv_vol                                  # negative for reversion

Step 5 (cross-sectional Q5 long-only):
  rank_pct = signal.rank(axis=1, pct=True)              # daily cross-sectional rank
  in_q5 = rank_pct > 0.8                                 # top quintile
  weights = in_q5 / in_q5.sum(axis=1)                   # equal-weight within Q5
```

### 3.2 Look-ahead safety

Every rolling window ends at `t-1`. Verified by `lookahead_invariance` audit: perturb the last 20 days of input, recompute signal, assert past signals are bit-identical (max_abs_diff = 0).

## 4. Execution Model

| Step | Action | Time |
|------|--------|------|
| Compute signal | After T-day market close | 15:00+ |
| Determine target Q5 basket | Same evening | 15:30+ |
| Place orders | T+1 day open | 09:30 |
| Hold | 21 trading days | ~1 month |
| Re-evaluate | T+21 day close → repeat | — |

`delay = 1` is mandatory and baked into the engine. `horizon = 21` is the empirical sweet spot found via 6-round grid search; do not modify.

## 5. Position Sizing

```
Within Q5 basket: equal weight per ETF
  N_q5 = ⌈ |universe| / 5 ⌉ ≈ 3 ETFs from 15
  weight_per_ETF = 1 / N_q5 ≈ 33%
```

Per-asset inverse-vol weighting is applied at the SIGNAL level (Step 3 above), then equal-weighted at the basket level. This is a deliberate two-layer design:

1. **Signal-level inverse-vol** decides which ETFs make Q5 (favors low-vol over high-vol when signal magnitudes are similar)
2. **Basket-level equal-weight** keeps execution simple and avoids further parameter overfitting at the sizing step

## 6. Cost Model (defaults — calibrate before paper trading)

| Parameter | Value | Note |
|-----------|-------|------|
| `commission_oneway` | 5e-5 (= 0.5 bp = 万 0.5) | retail rate negotiable; verify with broker |
| `slippage_bp` | 0.5 bp single-side | optimistic; champion + sister both pass at 1.0 bp |
| `stamp_duty` | 0.0 | ETF secondary market exempt |
| `borrow_cost_apr` | 0.08 | n/a for long-only; LS leg banned |

**Per round-trip total**: `(commission_oneway × 2 + slippage_bp / 1e4 × 2) × 1e4` = 2 bp at default.
**Annualized total cost** at 21-day rebalance: `(252/21) × 2 = 24 bp/year` ≈ 0.24%/year.

## 7. Decision Rules (TVT Audit)

The 4 promotion rules at each gate:

```python
PROMOTE if all([
    validate_q5_sh_after_cost   >= 0.5,
    validate_ic_at_delay_1      >= 0.0,
    sign(validate_sh) == sign(train_sh),
    validate_sh                  >= 0.4 × train_sh,    # ≤60% degradation
])
```

Round 0006 results:
- Train Sh 1.835 → Validate Sh 1.182 → **OOS Sh 1.359** (champion)
- Sister (slip=1.0): Train 1.758 → Validate 1.085 → OOS 1.281

**Both pass all 4 rules at ALL THREE gates.**

## 8. Risk Management

### 8.1 Suspension triggers (auto-halt new entries)

```python
if monthly_q5_excess_sharpe < -2.0:                     # bad month
    SUSPEND_NEW_ENTRIES
if quarterly_drawdown > 0.08:                           # 8% DD
    STOP_FULL_REVIEW
if effective_universe_size < 4_cohorts × 3_members:     # universe degradation
    SUSPEND
```

### 8.2 Universe edge cases

- ETF in Q5 hits 涨跌停 at T+1 open → defer 1-3 days; if blocked > 3 days, drop from this cycle
- ETF underlying has > 20% halted constituents on day t → drop from universe for that day
- ETF gets 限购 announcement (cross-border) → temporarily disable

### 8.3 Non-negotiable rules

1. **No leverage.** OOS Sh 1.36 assumes 1x exposure.
2. **No short leg.** LS Sh = -3.61 (OOS). Short leg is broken in A-share microstructure.
3. **No intraday rebalancing.** 21-day horizon is the cost-amortization sweet spot.
4. **No re-tuning after a losing month.** This is the classic overfit trap.
5. **No horizon extension beyond 21 days.** Empirical sweet spot from sweep.
6. **No Q5 → Q3 concentration.** Top-quintile is the audit-validated basket size.

## 9. Capital Sizing

| Tier | Notional | Rationale |
|------|----------|-----------|
| Minimum | 50,000 CNY | Each ETF ~16,700 CNY; covers ETF lot size + minimal slippage |
| Sweet spot | 300,000 - 1,000,000 CNY | Each ETF 100k+, slippage well-amortized |
| Capacity ceiling | ~50,000,000 CNY | Single-ETF (especially zz500/hang_seng) liquidity bound |

Above the capacity ceiling, slippage scales nonlinearly and the strategy alpha is consumed.

## 10. Audit Checklist (Pre-Production)

Before any real capital, verify:

- [ ] `pytest tests/ -v` returns 15/15 PASS
- [ ] `python scripts/run_round_0006_train.py` reproduces Train results
- [ ] `python scripts/run_round_0006_validate.py` reproduces Validate results
- [ ] `OOS_UNLOCKED=true python scripts/run_round_0006_oos.py` reproduces OOS Sh ≈ 1.36
- [ ] Inspect `figures/nav_full.png` — curve looks monotonic upward with shallow drawdowns
- [ ] Inspect `figures/yearly_returns.png` — every year is positive in Train/Validate; 2024+2025 positive in OOS
- [ ] Read `logs/20260428_etf_meanrev_arbitrage/round_0006_oos.yml` end-to-end
- [ ] Verify your broker's actual commission and slippage match assumptions (or re-run with realistic numbers)
- [ ] Run paper trading for ≥ 60 trading days; live Sharpe ≥ 0.5 before any real capital

## 11. Known Limitations

1. **2026 partial-year drawdown** (Jan-Apr 2026: -12.62% annualized over 53 days). Too small to interpret. Watch 2026 H2 closely. If 2026 H2 also negative → STOP and pivot mechanism.
2. **Cross-border QDII cohort under-represented** (only Hang Seng has 3 members; NASDAQ100 / S&P500 cohorts disabled due to insufficient history). Could be expanded if more issuers reach 4-year history.
3. **No intraday IOPV proxy** — daily-frequency NAV from efinance is a daily-EOD value; intraday IOPV (paid Level-2 feed) may yield a different / better v6 variant. Out of scope.
4. **Walk-forward not implemented** — Train and Validate are fixed windows. A walk-forward validation across ~6-month rolling windows could further harden the conclusion (deferred).
5. **Single-mechanism strategy** — does not combine with regime filters, cross-asset signals, or event-driven overlays. By design (workflow rule: one dominant mechanism per batch).

## 12. Why This Specifically (and Not Alternatives)

| Alternative | Tested? | Why not |
|-------------|---------|---------|
| v1 raw price z-score | r1 | weak; -0.69 IS Sh after-cost |
| v2 cohort-demean only | r2-r6 | works (r6 Train 1.81) but < v9 (Train 2.10) |
| v4 EG-cointegrated pair | r2 | look-ahead audit caught pair-selection leak; even fixed version weak |
| v6 NAV-based reference | r6 | unexpectedly worse than v2 (Train 0.71). NAV at daily EOD lags fair value |
| v7 monthly rebalance rank | r3-r4 | implementation issues; deferred |
| v8 asymmetric threshold |z|>2σ | r3-r4 | concentrated; failed Validate |
| **v9 cohort-z × inverse-vol** | r5-r6 | **passes Train + Validate + OOS audits** ✓ |

## 13. References

- Project research note: `research/etf_arbitrage_china_2026.md`
- Source audit trail: `research/sources.md`
- Per-round decision log: `logs/20260428_etf_meanrev_arbitrage/round_*.yml`
- Final OOS decision: `logs/20260428_etf_meanrev_arbitrage/oos_decision_r6.json`

## 14. License & Disclaimer

MIT (see [LICENSE](../LICENSE)). Research and educational use only. Not investment advice. Trading involves risk. The authors accept no liability for any losses.
