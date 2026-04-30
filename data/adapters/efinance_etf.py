"""efinance ETF adapter — Eastmoney via Micro-sheep/efinance.

Why this adapter exists alongside akshare_etf:
- AkShare's `fund_etf_hist_em` aggressively rate-limited; some ETFs can be
  blocked for hours.
- efinance hits the same eastmoney backend with different request patterns
  and recovers some symbols AkShare rejects.
- efinance also provides direct NAV access (`fund.get_quote_history`)
  without paid TuShare credentials — enables the v6 NAV-based reference
  variant that we couldn't build before.

Integration:
- Same OOS guard via `data.splits.assert_is_only`.
- Parquet cache parallel to akshare_etf, in `data/cache/efinance/`.
- Returns same column schema (renamed Chinese → English) so panel-builders
  can swap adapters without changes.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import efinance as ef
import pandas as pd

from data.splits import assert_is_only

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "efinance"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_RENAME_PRICE = {
    "日期": "date", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low",
    "成交量": "volume", "成交额": "amount",
    "振幅": "amplitude", "涨跌幅": "pct_change",
    "涨跌额": "change", "换手率": "turnover_rate",
    "股票名称": "name", "股票代码": "code",
}
_RENAME_NAV = {
    "日期": "date", "单位净值": "nav", "累计净值": "cum_nav",
    "涨跌幅": "pct_change",
}


def _cache_path(symbol: str, kind: str) -> Path:
    return CACHE_DIR / f"{symbol}_{kind}.parquet"


def fetch_etf_daily(symbol: str, start: date, end: date,
                    use_cache: bool = True) -> pd.DataFrame:
    """Fetch daily OHLCV via efinance.stock module (works for ETFs)."""
    assert_is_only(start, end)
    cache_path = _cache_path(symbol, "price")
    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if isinstance(cached.index, pd.DatetimeIndex):
            cached_max = cached.index.max().date()
        else:
            cached.index = pd.to_datetime(cached.index)
            cached_max = cached.index.max().date()
        # Accept cache covering within 5 days of end
        if len(cached) > 0 and cached_max >= end - timedelta(days=5):
            mask = (cached.index.date >= start) & (cached.index.date <= end)
            sliced = cached.loc[mask].copy()
            if len(sliced) > 0:
                logger.info("Cache hit %s price (%d rows)", symbol, len(sliced))
                return sliced

    logger.info("Fetching %s price via efinance ...", symbol)
    last_err = None
    for attempt in range(3):
        try:
            t0 = time.time()
            raw = ef.stock.get_quote_history(symbol)
            logger.info("  -> %d rows in %.1fs (attempt %d)",
                        len(raw), time.time() - t0, attempt + 1)
            break
        except Exception as e:                          # noqa: BLE001
            last_err = e
            logger.warning("  attempt %d failed: %r", attempt + 1, e)
            time.sleep(2.0 * (attempt + 1))
    else:
        raise RuntimeError(f"efinance failed for {symbol}: {last_err!r}")

    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    df = raw.rename(columns=_RENAME_PRICE).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    if use_cache:
        df.to_parquet(cache_path)

    mask = (df.index.date >= start) & (df.index.date <= end)
    return df.loc[mask].copy()


def fetch_etf_nav(symbol: str, start: date, end: date,
                  use_cache: bool = True) -> pd.DataFrame:
    """Fetch NAV (单位净值, 累计净值) via efinance.fund.get_quote_history."""
    assert_is_only(start, end)
    cache_path = _cache_path(symbol, "nav")
    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if not isinstance(cached.index, pd.DatetimeIndex):
            cached.index = pd.to_datetime(cached.index)
        cached_max = cached.index.max().date()
        if len(cached) > 0 and cached_max >= end - timedelta(days=5):
            mask = (cached.index.date >= start) & (cached.index.date <= end)
            sliced = cached.loc[mask].copy()
            if len(sliced) > 0:
                logger.info("Cache hit %s nav (%d rows)", symbol, len(sliced))
                return sliced

    logger.info("Fetching %s nav via efinance ...", symbol)
    last_err = None
    for attempt in range(3):
        try:
            t0 = time.time()
            raw = ef.fund.get_quote_history(symbol)
            logger.info("  -> %d rows in %.1fs (attempt %d)",
                        len(raw), time.time() - t0, attempt + 1)
            break
        except Exception as e:                          # noqa: BLE001
            last_err = e
            logger.warning("  attempt %d failed: %r", attempt + 1, e)
            time.sleep(2.0 * (attempt + 1))
    else:
        raise RuntimeError(f"efinance NAV failed for {symbol}: {last_err!r}")

    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    df = raw.rename(columns=_RENAME_NAV).copy()
    df["date"] = pd.to_datetime(df["date"])
    # Numeric columns can have '--' string sentinels mixed with floats →
    # coerce so pyarrow doesn't choke when caching.
    for col in ("nav", "cum_nav", "pct_change"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("date").sort_index()
    if use_cache:
        df.to_parquet(cache_path)

    mask = (df.index.date >= start) & (df.index.date <= end)
    return df.loc[mask].copy()


def fetch_etf_panel(symbols: list[str], start: date, end: date,
                    field: str = "close", use_cache: bool = True,
                    sleep_between: float = 1.5) -> pd.DataFrame:
    """Fetch wide panel of close prices.

    field: "close" for price, anything else looked up in price df.
    """
    assert_is_only(start, end)
    series_list = []
    for sym in symbols:
        try:
            df = fetch_etf_daily(sym, start, end, use_cache=use_cache)
            if df.empty:
                continue
            if field not in df.columns:
                logger.warning("Field %s missing for %s", field, sym)
                continue
            series_list.append(df[field].rename(sym))
            time.sleep(sleep_between)
        except Exception as e:                          # noqa: BLE001
            logger.error("Skip %s: %r", sym, e)
            continue
    if not series_list:
        return pd.DataFrame()
    panel = pd.concat(series_list, axis=1).sort_index()
    return panel


def fetch_nav_panel(symbols: list[str], start: date, end: date,
                    use_cache: bool = True,
                    sleep_between: float = 1.5) -> pd.DataFrame:
    """Fetch wide panel of unit NAV (单位净值)."""
    assert_is_only(start, end)
    series_list = []
    for sym in symbols:
        try:
            df = fetch_etf_nav(sym, start, end, use_cache=use_cache)
            if df.empty or "nav" not in df.columns:
                continue
            series_list.append(df["nav"].rename(sym))
            time.sleep(sleep_between)
        except Exception as e:                          # noqa: BLE001
            logger.error("NAV skip %s: %r", sym, e)
            continue
    if not series_list:
        return pd.DataFrame()
    return pd.concat(series_list, axis=1).sort_index()
