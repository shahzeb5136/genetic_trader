"""The instrument the calendar lab trades: SPY, with opens adjusted the same as closes.

Why this file exists at all
----------------------------
`backtest/benchmark.py` adjusts SPY's CLOSE for total return, because that is all a
monthly engine needs. A close-to-open strategy needs the open adjusted by the same
factor - and it needs the factor applied consistently, because the whole overnight
anomaly lives in the seam between one close and the next open. The adjustment factor
steps at the ex-date, which is exactly where the dividend lands in the world: an
overnight return computed as `adj_open[t+1] / adj_close[t]` therefore contains the
dividend, and an intraday return `adj_close[t] / adj_open[t]` does not. Get that
convention backwards and the "overnight premium" grows by SPY's entire dividend yield.

One implementation of the adjustment chain, reused: the factors come from
`normalize/adjustments.compute_factors`, the same function the benchmark and the whole
panel use (ADR-006), so this series and the engine's SPY calibration cannot drift
apart. `timing_accept()` in engine.py asserts they have not.

The half-spread
---------------
SPY has no row in `gold_half_spread` (it is not an index constituent), and
Corwin-Schultz cannot resolve a spread this narrow anyway - the estimator's own
documentation says the tick floor is what binds for liquid names (ADR-020). So the
half-spread here IS the tick floor, under the same two-tick-wide convention as
spreads.py: `tick_size(date) / as-traded price`. That is ~0.8bp in 2007 and ~0.2bp
today - slightly wider than SPY's usual one-tick market, which errs on the side the
cost model should err on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from ..normalize.adjustments import SPLIT_ADJUSTED, compute_factors
from ..normalize.splits import tick_size
from ..query import connect

log = logging.getLogger(__name__)

#: The instrument. The lab could run RSP or IWM through the same machinery, but every
#: claim in strategies.py is documented on the S&P 500, so that is what gets tested.
DEFAULT_TICKER = "SPY"


@dataclass(frozen=True)
class TimingData:
    """Aligned per-session arrays for one instrument. All shapes (D,).

    dates        'YYYY-MM-DD' session strings, ascending; string compare is date compare
    adj_open     total-return adjusted opening price
    adj_close    total-return adjusted closing price
    raw_open     as-stored (split-adjusted) open - the broker's print for SPY, which
                 never split in this window
    raw_close    as-stored close
    half_spread  proportional half-spread (the tick floor; see module docstring)
    day_of_week  Monday=0 .. Friday=4
    month        calendar month 1..12
    vix          ^VIX close on the same session, NaN where unavailable. Strategies that
                 condition on it must lag it themselves and say so.
    """

    ticker: str
    dates: np.ndarray
    adj_open: np.ndarray = field(repr=False)
    adj_close: np.ndarray = field(repr=False)
    raw_open: np.ndarray = field(repr=False)
    raw_close: np.ndarray = field(repr=False)
    half_spread: np.ndarray = field(repr=False)
    day_of_week: np.ndarray = field(repr=False)
    month: np.ndarray = field(repr=False)
    vix: np.ndarray = field(repr=False)

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    def date_index(self, date: str, side: str = "exact") -> int:
        """Row of a date; 'prev'/'next' round to the nearest session inside the data."""
        i = int(np.searchsorted(self.dates, date))
        if side == "exact":
            if i >= len(self.dates) or self.dates[i] != date:
                raise KeyError(f"{date!r} is not a session for {self.ticker}")
            return i
        if side == "next":
            if i >= len(self.dates):
                raise KeyError(f"no session on or after {date!r}")
            return i
        if side == "prev":
            if i < len(self.dates) and self.dates[i] == date:
                return i
            if i == 0:
                raise KeyError(f"no session on or before {date!r}")
            return i - 1
        raise ValueError(f"side must be exact|prev|next, not {side!r}")

    @property
    def meta(self) -> dict:
        return {"instrument": self.ticker, "start": str(self.dates[0]),
                "end": str(self.dates[-1]), "n_dates": int(self.n_dates),
                "adjustment": "compute_factors, SPLIT_ADJUSTED convention (ADR-006/007)",
                "half_spread": "tick floor: tick_size(date) / as-traded price (ADR-020)"}


@lru_cache(maxsize=4)
def load_timing_data(ticker: str = DEFAULT_TICKER) -> TimingData:
    """Build the aligned arrays from the benchmarks table. Cached per process."""
    con = connect()
    df = con.execute("""
        SELECT date, ticker, open, close, dividend, split_ratio
        FROM benchmarks WHERE ticker = ? ORDER BY date
    """, [ticker]).df()
    if df.empty:
        raise KeyError(f"no benchmark rows for {ticker!r}; "
                       "run `sp500lab ingest benchmarks`")
    bad_open = ~np.isfinite(df["open"].to_numpy()) | (df["open"].to_numpy() <= 0)
    if bad_open.any():
        # An open backfilled from the close makes that session's overnight leg absorb
        # the whole day. One missing open in 6,700 is noise; a hundred would mean the
        # feed changed and this loader should refuse rather than quietly reprice legs.
        if bad_open.sum() > 20:
            raise ValueError(f"{ticker}: {int(bad_open.sum())} sessions have no usable "
                             "open; refusing to build a close-to-open series from them")
        log.warning("%s: %d missing open(s) backfilled from the close",
                    ticker, int(bad_open.sum()))
        df.loc[bad_open, "open"] = df.loc[bad_open, "close"]

    # The same event-frame shape benchmark.py builds, through the same factor chain.
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

    factors = compute_factors(df, events, convention=SPLIT_ADJUSTED)
    merged = df.merge(factors, on=["security_id", "ticker", "date"], how="left")

    dates = merged["date"].to_numpy().astype("<U10")
    factor = merged["adj_factor"].to_numpy(dtype=np.float64)
    raw_open = merged["open"].to_numpy(dtype=np.float64)
    raw_close = merged["close"].to_numpy(dtype=np.float64)

    d = pd.to_datetime(merged["date"])
    with np.errstate(divide="ignore", invalid="ignore"):
        half = tick_size(dates).astype(np.float64) / raw_close

    vix = _aligned_close(con, "^VIX", dates)

    return TimingData(
        ticker=ticker, dates=dates,
        adj_open=raw_open * factor, adj_close=raw_close * factor,
        raw_open=raw_open, raw_close=raw_close,
        half_spread=half,
        day_of_week=d.dt.dayofweek.to_numpy(dtype=np.int64),
        month=d.dt.month.to_numpy(dtype=np.int64),
        vix=vix,
    )


def _aligned_close(con, ticker: str, dates: np.ndarray) -> np.ndarray:
    """(D,) another benchmark's close on the instrument's own sessions, else NaN."""
    try:
        aux = con.execute(
            "SELECT date, close FROM benchmarks WHERE ticker = ? ORDER BY date",
            [ticker]).df()
    except Exception as exc:                                         # noqa: BLE001
        log.warning("%s unavailable: %s", ticker, exc)
        return np.full(len(dates), np.nan)
    s = pd.Series(aux["close"].to_numpy(dtype=np.float64),
                  index=aux["date"].to_numpy())
    return s.reindex(dates).to_numpy(dtype=np.float64)
