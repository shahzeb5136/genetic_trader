"""Macro and market-state features: one number per date, the same for every security.

These do not discriminate between stocks. They describe the world the cross-section is
sitting in, and they exist so a strategy can behave differently in different regimes -
hold less when credit is deteriorating, prefer low volatility when the term structure has
inverted, stop trading momentum when the market has just fallen 30%. Momentum crashes are
the canonical example: 12-1 momentum lost more than half its value in 2009 not because
the signal stopped working but because it was short the beaten-down names when they
rebounded, and that is a regime fact rather than a stock fact.

Only series that are never revised
-----------------------------------
7 of the 18 FRED series in this project are revised after publication (ADR-011): CPI,
GDP, payrolls, unemployment, industrial production, sentiment and the recession flag.
Using a revised series at face value is a lookahead leak, and a subtle one - today's
`UNRATE` for 2009-03 is not the number anyone saw in April 2009.

So this file uses only the daily MARKET series, which are prints rather than estimates
and are never restated: VIX, Treasury yields, the fed funds rate, the dollar index, oil,
and the ICE credit spreads. Vintage access for the revised series is TODO-7, and until it
exists they are deliberately absent rather than quietly wrong.

Publication lag
---------------
Even an unrevised daily series is published after the session it describes. Every macro
series is therefore lagged one session before it is sampled, so a rebalance on the last
day of the month reads the value from the day before. That costs a day of information and
removes an entire class of "the model knew today's close of VIX at today's close"
argument.

The credit spreads are truncated
---------------------------------
FRED's keyless endpoint returns only about three years of the licensed ICE series
(BAMLH0A0HYM2, BAMLC0A0CM). They are included because they are the best available
recession signal and they are the right thing to reach for once a key is configured, but
before roughly 2023 they are NaN, and a strategy that leans on them is a strategy with no
history. `FeaturePanel.coverage()` shows exactly how little there is.

Market state comes from the panel, not from FRED
-------------------------------------------------
Trend, drawdown and realised volatility of the market are computed from the same
equal-weighted point-in-time cross-section that `price.py` uses for betas. That keeps the
market definition consistent across the whole feature layer, and it means the market
series is survivorship-free in the same way the universe is.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..query import connect

log = logging.getLogger(__name__)

#: FRED series that are prints, not estimates. `revised` is False for all of these in
#: `fred_series`; the assertion is checked at build time rather than trusted.
UNREVISED_SERIES = {
    "vix": "VIXCLS",
    "term_spread": "T10Y2Y",
    "term_spread_3m": "T10Y3M",
    "ust10y": "DGS10",
    "fed_funds": "DFF",
    "hy_spread": "BAMLH0A0HYM2",
    "ig_spread": "BAMLC0A0CM",
    "dollar_index": "DTWEXBGS",
    "oil": "DCOILWTICO",
}

#: Sessions of lag applied to every macro series before sampling. One, not zero: a daily
#: series describing session t is published after session t closes.
PUBLICATION_LAG = 1


def compute(panel, rows: np.ndarray, *, data_cutoff: str | None = None) -> dict:
    """(R,) vectors per macro feature. `panel.py` broadcasts them across securities."""
    rows = np.asarray(rows, dtype=np.int64)
    out: dict[str, np.ndarray] = {}
    out.update(_fred(panel, rows, data_cutoff))
    out.update(_market_state(panel, rows))
    return out


# --------------------------------------------------------------------------
# FRED
# --------------------------------------------------------------------------

def _fred(panel, rows: np.ndarray, data_cutoff: str | None) -> dict:
    con = connect()
    ids = list(UNREVISED_SERIES.values())
    placeholders = ", ".join("?" for _ in ids)
    where = f"WHERE series_id IN ({placeholders})"
    params = list(ids)
    if data_cutoff:
        where += " AND date <= ?"
        params.append(data_cutoff)
    try:
        df = con.execute(
            f"SELECT series_id, date, value, revised FROM fred_series {where}",
            params).df()
    except Exception as exc:                                     # noqa: BLE001
        log.warning("macro features unavailable: %s", exc)
        return {}
    if df.empty:
        return {}

    revised = sorted(set(df.loc[df["revised"].astype(bool), "series_id"]))
    if revised:
        # Loud, not fatal: if a series this file assumed was a print turns out to be
        # flagged as revised, every feature derived from it is a lookahead leak and the
        # right response is to notice rather than to keep going quietly.
        log.error("macro: %s are flagged revised but are used as unrevised - "
                  "see ADR-011 and features/macro.py", revised)

    dates = panel.dates
    out: dict[str, np.ndarray] = {}
    for name, sid in UNREVISED_SERIES.items():
        s = df[df["series_id"] == sid].sort_values("date")
        if s.empty:
            continue
        aligned = _align(s["date"].to_numpy(), s["value"].to_numpy(float), dates)
        lagged = _lag(aligned, PUBLICATION_LAG)
        out[name] = lagged[rows]
        # Changes matter more than levels for most of these: a 10-year yield of 4% means
        # something different in 2007 and in 2023, but a 60-session rise of 100bp means
        # the same thing in both.
        out[f"{name}_chg_63d"] = (lagged[rows] - _shift_take(lagged, rows, 63))

    if "vix" in out:
        vix = _lag(_align(df.loc[df["series_id"] == "VIXCLS", "date"].to_numpy(),
                          df.loc[df["series_id"] == "VIXCLS", "value"].to_numpy(float),
                          dates), PUBLICATION_LAG)
        # VIX relative to its own year: the level that counts as "high" drifts, and a
        # ratio to the trailing median is stationary where the level is not.
        med = pd.Series(vix).rolling(252, min_periods=126).median().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            out["vix_relative"] = (vix / med)[rows]
    return out


def _align(src_dates: np.ndarray, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """(D,) a FRED series carried forward onto the trading calendar.

    Forward fill only. A macro series has holidays and gaps that are not the market's,
    and interpolating across one would invent an observation that never printed.
    """
    pos = np.searchsorted(src_dates, dates, side="right") - 1
    out = np.where(pos >= 0, values[np.maximum(pos, 0)], np.nan)
    return np.where(np.isfinite(out), out, np.nan)


def _lag(series: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return series
    out = np.full_like(series, np.nan)
    out[k:] = series[:-k]
    return out


def _shift_take(series: np.ndarray, rows: np.ndarray, back: int) -> np.ndarray:
    src = rows - back
    return np.where(src >= 0, series[np.maximum(src, 0)], np.nan)


# --------------------------------------------------------------------------
# Market state, from the panel's own cross-section
# --------------------------------------------------------------------------

def _market_state(panel, rows: np.ndarray) -> dict:
    """Trend, drawdown, realised volatility and breadth of the equal-weighted index."""
    close = panel.adj_close.astype(np.float64)
    member = panel.in_index & panel.has_price

    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.full_like(close, np.nan)
        rets[1:] = close[1:] / close[:-1] - 1.0
    rets[~np.isfinite(rets)] = np.nan

    r = np.where(member, rets, np.nan)
    counts = np.isfinite(r).sum(axis=1)
    mkt = np.where(counts >= 20, np.nansum(r, axis=1) / np.maximum(counts, 1), np.nan)

    level = np.cumprod(1.0 + np.nan_to_num(mkt))
    level[~np.isfinite(mkt) & (np.arange(len(mkt)) == 0)] = 1.0
    lv = pd.Series(level)

    with np.errstate(divide="ignore", invalid="ignore"):
        trend = (lv / lv.rolling(200, min_periods=100).mean() - 1.0).to_numpy()
        drawdown = (lv / lv.cummax() - 1.0).to_numpy()
    vol = (pd.Series(mkt).rolling(21, min_periods=10).std() * np.sqrt(252)).to_numpy()
    vol_252 = (pd.Series(mkt).rolling(252, min_periods=126).std()
               * np.sqrt(252)).to_numpy()

    # Breadth: the share of index members trading above their own 200-day average. It
    # falls before the index does, because the index is dominated by its largest names
    # long after the median name has rolled over.
    above = _breadth(close, member)

    return {
        "mkt_trend_200d": trend[rows],
        "mkt_drawdown": drawdown[rows],
        "mkt_vol_21d": vol[rows],
        "mkt_vol_252d": vol_252[rows],
        "mkt_breadth_200d": above[rows],
        # Realised volatility relative to its own year. High and rising is a different
        # regime from high and falling, and the ratio separates them.
        "mkt_vol_ratio": np.where(vol_252[rows] > 0, vol[rows] / vol_252[rows], np.nan),
    }


def _breadth(close: np.ndarray, member: np.ndarray) -> np.ndarray:
    ma = pd.DataFrame(close).rolling(200, min_periods=100).mean().to_numpy()
    with np.errstate(invalid="ignore"):
        above = (close > ma) & member & np.isfinite(ma)
    have = member & np.isfinite(ma)
    counts = have.sum(axis=1)
    return np.where(counts >= 20, above.sum(axis=1) / np.maximum(counts, 1), np.nan)
