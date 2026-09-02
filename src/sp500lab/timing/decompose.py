"""Per-ticker overnight/intraday decomposition, across the whole point-in-time index.

The single-instrument engine answers "does the overnight effect exist in the index".
This answers the finer question - WHICH names earn their return while the market is
closed - by splitting every member's close-to-close log return into its two legs over
exactly the sessions it was in the index. Membership-clipped, because a decomposition
of a name's post-index penny-stock years would be a statement about a different
security (invariant 3, HANDOFF).

This is a measurement, not a backtest: no costs, no execution, no portfolio. Its
gross-of-costs nature is the point - trading any single name close-to-open crosses
the spread ~500 times a year, which at a typical 2-5bp half-spread is a 10-25%/yr
bill before commission. The table exists to show where the anomaly lives, and the
`overnight_momentum` strategy in the monthly engine is the costed, tradable
expression of the same information.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..backtest.panel import Panel, build_panel

log = logging.getLogger(__name__)

#: A name needs at least this many in-index sessions before its split means anything.
MIN_SESSIONS = 500


def decompose_members(panel: Panel | None = None, *, start: str = "2007-04-01",
                      end: str = "2021-12-31",
                      min_sessions: int = MIN_SESSIONS) -> pd.DataFrame:
    """One row per security: annualised overnight vs intraday log return, in-index only.

    Columns: ticker, security_id, sessions, first, last, overnight_ann, intraday_ann,
    total_ann, overnight_share. `overnight_share` is overnight / (|overnight| +
    |intraday|), signed - 1.0 means the whole move happened overnight, -1.0 means
    overnight LOST what intraday made.
    """
    panel = panel or build_panel()
    lo = panel.date_index(start, side="next")
    hi = panel.date_index(end, side="prev")

    close = panel.adj_close[lo:hi + 1]
    open_ = panel.adj_open[lo:hi + 1]
    member = (panel.in_index & panel.has_price)[lo:hi + 1]
    dates = panel.dates[lo:hi + 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        on = np.log(open_[1:] / close[:-1])
        intra = np.log(close[1:] / open_[1:])
    # A leg only counts when the name was a member on BOTH ends of the session pair
    # and both prices exist - the same clipping a member-only trader would live.
    ok = member[1:] & member[:-1] & np.isfinite(on) & np.isfinite(intra)

    rows = []
    for s in range(panel.n_securities):
        col = ok[:, s]
        n = int(col.sum())
        if n < min_sessions:
            continue
        o = float(on[col, s].sum())
        i = float(intra[col, s].sum())
        years = n / 252.0
        idx = np.flatnonzero(col)
        denom = abs(o) + abs(i)
        rows.append({
            "ticker": str(panel.tickers[s]),
            "security_id": str(panel.security_ids[s]),
            "sessions": n,
            "first": str(dates[idx[0] + 1]),
            "last": str(dates[idx[-1] + 1]),
            "overnight_ann": float(np.expm1(o / years)),
            "intraday_ann": float(np.expm1(i / years)),
            "total_ann": float(np.expm1((o + i) / years)),
            "overnight_share": (o / denom) if denom > 1e-12 else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("overnight_ann", ascending=False)
    return df.reset_index(drop=True)


def summarise(df: pd.DataFrame) -> dict:
    """The cross-section in five numbers, for the report's stat row."""
    if df.empty:
        return {}
    return {
        "names": int(len(df)),
        "median_overnight_ann": float(df["overnight_ann"].median()),
        "median_intraday_ann": float(df["intraday_ann"].median()),
        "overnight_positive": float((df["overnight_ann"] > 0).mean()),
        "intraday_positive": float((df["intraday_ann"] > 0).mean()),
        "overnight_beats_intraday": float(
            (df["overnight_ann"] > df["intraday_ann"]).mean()),
    }
