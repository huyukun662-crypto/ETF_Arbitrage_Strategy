"""AkShare ETF daily-price adapter with parquet cache + OOS guard.

Why AkShare and not TuShare: TuShare requires a paid token for many ETF
endpoints; AkShare is free and provides `fund_etf_hist_em` which returns
EOD OHLCV per ETF symbol. Sufficient for daily-frequency strategies in
round_0001.

Caching: every (symbol, start, end) call is cached as parquet under
data/cache/akshare/{symbol}.parquet. Subsequent calls re-use the cache
and only fetch missing date ranges.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd

from data.splits import assert_is_only

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "akshare"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Map AkShare's Chinese column names to English.
_RENAME = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover_rate",
}


def _to_yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.parquet"


def fetch_etf_daily(symbol: str, start: date, end: date,
                    use_cache: bool = True) -> pd.DataFrame:
    """Fetch daily OHLCV for one ETF, applying OOS guard.

    Returns DataFrame indexed by date with columns:
        open, high, low, close, volume, amount, pct_change, turnover_rate

    Raises OOSAccessError if `end` would touch OOS territory without unlock.
    """
    assert_is_only(start, end)

    cache_path = _cache_path(symbol)
    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached.index = pd.to_datetime(cached.index).date
        # Cache hit if it covers the END of the requested window (start can be
        # a non-trading day before the real data; we accept that as a hit).
        # Accept cache if it covers within 5 days of end (covers weekend / holiday tails)
        if len(cached) > 0 and cached.index.max() >= end - timedelta(days=5):
            mask = (cached.index >= start) & (cached.index <= end)
            sliced = cached.loc[mask].copy()
            if len(sliced) > 0:
                logger.info("Cache hit %s (%d rows)", symbol, len(sliced))
                return sliced

    logger.info("Fetching %s from %s to %s via AkShare ...", symbol, start, end)
    raw = None
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            t0 = time.time()
            raw = ak.fund_etf_hist_em(
                symbol=symbol,
                period="daily",
                start_date=_to_yyyymmdd(start),
                end_date=_to_yyyymmdd(end),
                adjust="hfq",   # 后复权 — important for any returns calculation
            )
            elapsed = time.time() - t0
            logger.info("  -> %d rows in %.1fs (attempt %d)", len(raw), elapsed, attempt + 1)
            break
        except Exception as e:                          # noqa: BLE001
            last_err = e
            backoff = 2.0 * (attempt + 1)
            logger.warning("  attempt %d failed: %r; backing off %.1fs", attempt + 1, e, backoff)
            time.sleep(backoff)
    if raw is None:
        raise RuntimeError(f"All retries failed for {symbol}: {last_err!r}")

    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    df = raw.rename(columns=_RENAME).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.set_index("date").sort_index()

    # Persist full fetched range to cache
    if use_cache:
        df.to_parquet(cache_path)

    mask = (df.index >= start) & (df.index <= end)
    return df.loc[mask].copy()


def fetch_etf_panel(symbols: list[str], start: date, end: date,
                    field: str = "close", use_cache: bool = True,
                    sleep_between: float = 2.0) -> pd.DataFrame:
    """Fetch a wide panel of ETF prices.

    Returns DataFrame with date index, symbol columns, values = `field`.
    """
    assert_is_only(start, end)
    series_list: list[pd.Series] = []
    for sym in symbols:
        try:
            df = fetch_etf_daily(sym, start, end, use_cache=use_cache)
            if df.empty:
                logger.warning("No data for %s", sym)
                continue
            s = df[field].rename(sym)
            series_list.append(s)
            time.sleep(sleep_between)
        except Exception as e:                        # noqa: BLE001
            logger.error("Failed to fetch %s: %r", sym, e)
            continue
    if not series_list:
        return pd.DataFrame()
    panel = pd.concat(series_list, axis=1).sort_index()
    panel.index = pd.to_datetime(panel.index)
    return panel
