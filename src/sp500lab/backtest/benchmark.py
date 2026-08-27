"""Total-return benchmark series.

The `benchmarks` silver table stores raw OHLCV plus discrete dividend and split events,
but no adjustment factors - `normalize/adjustments.py` runs over `daily_bars`, not over
this table. So the factors are computed here with the same function, which keeps one
implementation of the adjustment chain rather than two that can drift apart.

Why this matters more than it sounds
------------------------------------
SPY's price return over 2000-01-03..2026-08-26 is 6.43%/yr; its total return is
8.32%/yr. Comparing a total-return strategy against a price-return benchmark hands
every strategy 1.9pp/yr of free apparent alpha - enough to make a mediocre strategy
look like a good one for two decades running. Benchmarks are adjusted the same way the
strategy's holdings are, or the comparison means nothing.

Those two figures are also the engine's acceptance test (accept.py test 1). If a
buy-and-hold SPY run through the engine returns ~6.4%/yr the adjustment chain has
dropped dividends; ~10.2%/yr means they are being counted twice.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd

from ..normalize.adjustments import SPLIT_ADJUSTED, compute_factors
from ..query import connect

log = logging.getLogger(__name__)

#: yfinance pre-applies splits even with auto_adjust=False (ADR-007), and the
#: benchmarks table comes from the same feed, so the same convention applies.
BENCHMARK_CONVENTION = SPLIT_ADJUSTED


@lru_cache(maxsize=8)
def benchmark_total_return(ticker: str = "SPY") -> pd.Series:
    """Dividend-reinvested price series for one benchmark, indexed by date string.

    Anchored so the most recent bar equals its raw close, matching the convention in
    `adjustment_factors`: today's price is the one you would actually transact at.
    """
    con = connect()
    df = con.execute("""
        SELECT date, ticker, close, dividend, split_ratio
        FROM benchmarks WHERE ticker = ? ORDER BY date
    """, [ticker]).df()
    if df.empty:
        raise KeyError(f"no benchmark rows for {ticker!r}; "
                       "run `sp500lab ingest benchmarks`")

    df["security_id"] = f"BENCH_{ticker}"
    actions = []
    div = df.loc[df["dividend"] > 0, ["security_id", "date", "dividend"]]
    if len(div):
        div = div.rename(columns={"dividend": "value"})
        div["action_type"] = "dividend"
        actions.append(div)
    spl = df.loc[df["split_ratio"].fillna(0) > 0, ["security_id", "date", "split_ratio"]]
    spl = spl[spl["split_ratio"] != 1.0]
    if len(spl):
        spl = spl.rename(columns={"split_ratio": "value"})
        spl["action_type"] = "split"
        actions.append(spl)

    events = (pd.concat(actions, ignore_index=True) if actions
              else pd.DataFrame(columns=["security_id", "date", "value", "action_type"]))
    factors = compute_factors(df, events, convention=BENCHMARK_CONVENTION)
    merged = df.merge(factors, on=["security_id", "ticker", "date"], how="left")
    return pd.Series((merged["close"] * merged["adj_factor"]).to_numpy(),
                     index=merged["date"].to_numpy(), name=ticker)


def benchmark_price_return(ticker: str = "SPY") -> pd.Series:
    """Unadjusted close. For diagnostics only - never as a comparison benchmark."""
    df = connect().execute(
        "SELECT date, close FROM benchmarks WHERE ticker = ? ORDER BY date", [ticker]).df()
    return pd.Series(df["close"].to_numpy(), index=df["date"].to_numpy(), name=ticker)


def annualised(series: pd.Series) -> float:
    s = series.dropna()
    years = (pd.Timestamp(str(s.index[-1])) - pd.Timestamp(str(s.index[0]))).days / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1.0)


def available() -> list[str]:
    return connect().execute(
        "SELECT DISTINCT ticker FROM benchmarks ORDER BY ticker").df()["ticker"].tolist()
