"""Benchmark series and the canonical trading calendar.

Benchmarks
----------
Every strategy in the eventual competition is scored against the same baselines, so
they belong in the data layer rather than in any one strategy's code:

  SPY   S&P 500 cap-weighted ETF - the "did you beat the index" bar
  RSP   S&P 500 equal-weight ETF - separates stock selection from size tilt
  ^GSPC S&P 500 price index level itself (no fees, no dividends)
  IWM   Russell 2000, for a size-factor reference
  ^VIX  volatility index, for regime tagging

The first backtest to run is buy-and-hold SPY. If that does not reproduce SPY's
actual total return to within a few basis points, the engine is wrong and nothing
downstream means anything.

Trading calendar
----------------
The calendar is derived EMPIRICALLY from SPY's own bars rather than from a holiday
rule set. SPY trades every session the NYSE is open, so its date index *is* the
calendar - and deriving it from the same feed as the price data means the calendar
can never disagree with the data it describes. A rules-based calendar would drift
whenever an unscheduled closure happened (Hurricane Sandy in 2012, the national day
of mourning in December 2018) and would silently mark those as missing bars.

This distinction matters downstream: a gap in a stock's history is a DATA QUALITY
problem worth investigating, whereas a gap on a non-trading day is nothing at all.
Without a trustworthy calendar the two are indistinguishable.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import get_settings
from ..storage import today_iso, write_bronze, write_silver
from .base import IngestResult

log = logging.getLogger(__name__)

SOURCE = "yfinance"
DATASET = "benchmarks"

BENCHMARKS = {
    "SPY":   "SPDR S&P 500 ETF (total return proxy, cap weighted)",
    "RSP":   "Invesco S&P 500 Equal Weight ETF",
    "^GSPC": "S&P 500 price index level",
    "IWM":   "iShares Russell 2000 ETF",
    "^VIX":  "CBOE Volatility Index",
}

CALENDAR_SOURCE = "SPY"


def derive_calendar(session_dates: list[str]) -> pd.DataFrame:
    """The trading calendar, from the dates one always-trading instrument printed on.

    `is_month_end` (and quarter, year) is true on the LAST session of the period, not
    on the calendar month-end - the 31st is a Sunday often enough that this matters.
    The final row compares against NaT and comes out True, which is correct: the most
    recent session is the latest we have for its month, quarter and year.
    """
    dates = sorted({str(d) for d in session_dates})
    if not dates:
        raise ValueError("cannot derive a calendar from no sessions")
    cal = pd.DataFrame({"date": dates})
    dt = pd.to_datetime(cal["date"])
    if dt.isna().any():
        raise ValueError("unparseable session dates")
    cal["year"] = dt.dt.year
    cal["month"] = dt.dt.month
    cal["day_of_week"] = dt.dt.dayofweek
    cal["is_month_end"] = dt.dt.to_period("M") != dt.shift(-1).dt.to_period("M")
    cal["is_quarter_end"] = dt.dt.to_period("Q") != dt.shift(-1).dt.to_period("Q")
    cal["is_year_end"] = dt.dt.year != dt.shift(-1).dt.year
    cal["session_index"] = range(len(cal))
    cal["calendar_source"] = CALENDAR_SOURCE
    return cal


def run(force: bool = False, start: str | None = None) -> IngestResult:
    import yfinance as yf

    res = IngestResult(source=SOURCE, dataset=DATASET)
    start = start or get_settings().price_start
    ingest_date = today_iso()

    from .prices_yfinance import _to_long

    syms = list(BENCHMARKS)
    raw = yf.download(syms, start=start, auto_adjust=False, actions=True,
                      progress=False, group_by="column", threads=True)
    df = _to_long(raw, syms)
    if df.empty:
        res.errors.append("no benchmark data returned")
        return res

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"])

    write_bronze(source=SOURCE, dataset=DATASET, filename="benchmarks.parquet",
                 content=df.to_parquet(index=False),
                 url=f"yfinance.download({syms}, start={start})",
                 ingest_date=ingest_date)
    res.bronze_files += 1

    df["description"] = df["ticker"].map(BENCHMARKS)
    df["source"] = SOURCE
    write_silver(df.reset_index(drop=True), "market/benchmarks")

    # ---- trading calendar, derived from the benchmark that trades every session --
    cal_dates = sorted(df.loc[df["ticker"] == CALENDAR_SOURCE, "date"].unique())
    if not cal_dates:
        res.errors.append(f"{CALENDAR_SOURCE} missing - cannot derive trading calendar")
        return res

    cal = derive_calendar(cal_dates)
    write_silver(cal, "reference/trading_calendar")

    per_year = cal.groupby("year").size()
    res.rows = len(df)
    res.notes = {
        "benchmarks": int(df["ticker"].nunique()),
        "trading_days": len(cal),
        "calendar_range": f"{cal['date'].min()} .. {cal['date'].max()}",
        "median_sessions_per_year": int(per_year.median()),
        "bars_per_benchmark": df.groupby("ticker").size().to_dict(),
    }
    return res
