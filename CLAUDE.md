# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Build a research + backtest framework for **A-share ETF arbitrage strategies** that are realistically executable from a single retail/small-institutional account. The first session in this repo produced only research artifacts; **no production code exists yet**. The next coding session must scaffold the architecture below before writing any strategy.

The 2-3 strategies recommended for first implementation (full reasoning in `research/etf_arbitrage_china_2026.md` §7):

1. **Cross-border QDII-ETF premium/discount mean reversion (T+0, intraday)** — primary, lowest capital + reachable alpha.
2. **Same-benchmark multi-ETF pair statistical arbitrage** — secondary, market-neutral, low-capital.
3. **Primary-secondary 申购赎回 arbitrage** — only if the user later confirms ≥3M CNY capital + a broker-side 一级申赎 channel.

Explicitly out of scope until further notice: pure index-futures basis arbitrage (capital + current deep-discount makes it unviable for retail), gold internal/external arbitrage (needs offshore account), high-frequency intraday market-making (constrained by 2024 量化新规).

## Repository Layout (planned, not yet created)

```
ETF_Arbitrage/
├── research/                           # ✅ exists — read first
│   ├── etf_arbitrage_china_2026.md     # main research note (read this end-to-end before coding)
│   ├── sources.md                      # audit trail of every URL consulted
│   └── raw/, screenshots/              # evidence
├── data/                               # adapters & cached datasets
│   ├── adapters/                       # tushare/, akshare/, cninfo/  — one module each
│   ├── universe/                       # ETF master list, T+0 flag, 限购 status, PCF cache
│   ├── pcf/                            # 申购赎回清单 daily snapshots (per ETF, per date)
│   └── cache/                          # parquet caches (gitignored)
├── strategy/                           # one module per strategy in §7 of research note
│   ├── qdii_premium_meanrev/
│   ├── pair_statarb/
│   └── primary_secondary_arb/          # later
├── backtest/                           # event-driven engine (single source of truth)
│   ├── engine.py                       # respects T+0/T+1, 涨跌停, 停牌, 申赎 latency
│   ├── costs.py                        # 真实费率 + 冲击成本模型
│   └── reports/                        # IS/OOS/Full split + per-strategy CSV evidence
├── analysis/                           # Pareto plots, parameter-sensitivity, regime split
├── conf/                               # Hydra configs (one yaml per strategy + base.yaml)
└── outputs/                            # Hydra-managed run outputs (gitignored)
```

When you create new modules, follow the global coding-style rule (200-400 line files, factory + registry pattern for strategies/adapters, frozen dataclass for configs, full type hints).

## Critical Domain Rules (A-share ETF specific 暗礁)

These will silently break a backtest if violated. Read research note §8 for full list. Top hazards:

1. **T+0 is asymmetric**: stock-ETF "申赎 T+0" only means the ETF share appears same-day; **redeemed stocks settle T+1**. Never model the stock leg as T+0.
2. **Cross-border / gold / commodity / bond / money ETFs** are true T+0 on the secondary leg. Stock ETFs are T+1.
3. **IOPV distorted by 停牌**: any ETF with a halted constituent on day t is unusable for arbitrage signals on that day.
4. **限购墙**: cross-border ETF 申购 channel can be closed without warning; primary-leg arbitrage path silently dies. Maintain a daily 限购 status table from announcements.
5. **PCF 三上限字段** (effective 2025-11): 当日净申购上限 / 当日净赎回上限 / 单账户当日净申购上限. Order routing must parse these or be rejected.
6. **现金替代** flag in PCF (必须/允许/禁止) decides whether arbitrage is mechanically possible.
7. **No 印花税 on ETF secondary** but stock leg of 申赎 still pays it on the underlying — model accordingly.
8. **IS/OOS discipline** (from global CLAUDE.md): split is **IS = 2018-2023, OOS = 2024-2026**. OOS is **read-only**; reverse-tuning to fit a 2024 event is forbidden.

## Development Workflow

### Initial setup (first time only)

```bash
# Install uv if missing: https://docs.astral.sh/uv/getting-started/installation/
uv sync                                  # creates .venv + installs deps
cp .env.example .env                     # add TUSHARE_TOKEN, never commit
```

### Common commands (placeholders — module paths will exist after scaffold)

```bash
uv run python -m data.adapters.tushare.refresh_etf_universe
uv run python -m strategy.qdii_premium_meanrev.run --config-name=base
uv run python -m backtest.engine strategy=qdii_premium_meanrev split=is
uv run pytest                            # all tests
uv run pytest tests/test_costs.py -k impact   # single test
uv run ruff check .
uv run mypy data strategy backtest
```

### Backtest convention

Every grid-search script must emit three CSVs (per global CLAUDE.md 量化研发工程纪律):
- `grid_search_<round>_is.csv` — used for parameter selection
- `grid_search_<round>_oos.csv` — read-only, never used to rank
- `grid_search_<round>_full.csv` — final reporting only

A new champion only replaces the prior baseline if it **strictly dominates** on IS metrics. Single-variable iteration: change one dimension per round, freeze prior winner on the rest.

## Data Sources & Credentials

| Need | Source | Auth |
|------|--------|------|
| ETF daily OHLCV/NAV | TuShare Pro (`fund_daily`, `fund_nav`) | env `TUSHARE_TOKEN` |
| ETF universe + 限购 | AkShare (`fund_etf_category_sina`, etc.) | none |
| 申购赎回清单 (PCF) | Exchange / fund company crawl | none, but rate-limit friendly |
| 公告流（限购、临停、调仓、分红）| 巨潮资讯 `cninfo.com.cn` | none |
| Real-time IOPV (15s) | Level-2 data feed | broker subscription, **not free** |
| Stock-index futures | TuShare `fut_daily`, CFFEX | env `TUSHARE_TOKEN` |

Secrets go in `.env` (gitignored). Never hardcode tokens. Never commit `data/cache/`, `data/pcf/`, or `outputs/`.

## What This Repo Does NOT Have Yet

- No `pyproject.toml` deps installed (file exists but `uv sync` not run).
- No data adapters, no strategies, no backtest engine, no tests.
- No real PCF or IOPV data — research artifacts are text-only.

The next session should **start from `research/etf_arbitrage_china_2026.md` §10** to confirm with the user which strategy (A / B / C / D) to scaffold first, then build top-down from `data/adapters/` → `data/universe/` → `strategy/<chosen>/` → `backtest/`.
