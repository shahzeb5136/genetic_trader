"""Event features: index membership transitions and dividend behaviour.

These are the features that exist because of what this repo bothered to collect. Almost
nobody has point-in-time S&P 500 membership; almost everybody has adjusted prices. So the
membership features below cannot be computed from a normal price feed at all, and the
dividend features cannot be computed from an `adj_close` column - the adjustment has
already dissolved the dividends into the price by the time you see it.

Index membership as a signal
-----------------------------
Being added to the S&P 500 is a demand shock: index funds must buy, and they must buy
regardless of price. The classical result (Shleifer 1986; Harris & Gurel 1986) is a
several-percent pop into the effective date followed by partial reversal. The modern
result is that the effect has largely been arbitraged away since the early 2000s. Both
are interesting, and this project can actually test which applies in ITS window rather
than citing one.

Two honesty notes that decide how these can be used:

  * membership here is reconstructed at MONTHLY granularity from Wikipedia revisions
    (ADR-004), so `months_in_index` is accurate to a month and the pre-effective-date
    run-up is invisible. That rules out the announcement-window trade and leaves the
    post-inclusion drift, which is the part a monthly rebalancer could have traded
    anyway.
  * a name appears as "new" the first month it shows up in a snapshot. Because the
    snapshot is the last revision of that month, that is genuinely knowable at the
    month-end rebalance. It is not knowable a month earlier, and nothing here pretends
    otherwise.

Dividends
---------
`corporate_actions` holds dividends as discrete dated events, which is the only form in
which they can be used as a signal. The trailing yield is a classic value proxy that
needs no fundamentals at all - it is available from 2000 rather than from 2009, so it is
the one valuation-ish feature with full history. A dividend CUT is one of the strongest
negative corporate signals there is, and it is visible here as a fall in the trailing
sum.

The distributions are matched to their payment dates, so a dividend paid on the 15th is
in the trailing window from the 15th and not before.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..query import connect

log = logging.getLogger(__name__)

#: Calendar days in the trailing dividend window. 372 rather than 365 so that four
#: quarterly payments always land inside it even when a payment date drifts by a week -
#: a 365-day window intermittently catches three payments instead of four and produces a
#: 25% "dividend cut" that never happened.
DIV_WINDOW_DAYS = 372


def compute(panel, rows: np.ndarray, *, data_cutoff: str | None = None) -> dict:
    """(R, S) matrices for membership and dividend features."""
    rows = np.asarray(rows, dtype=np.int64)
    out: dict[str, np.ndarray] = {}
    out.update(_membership(panel, rows))
    out.update(_dividends(panel, rows, data_cutoff))
    return out


# --------------------------------------------------------------------------
# Membership
# --------------------------------------------------------------------------

def _membership(panel, rows: np.ndarray) -> dict:
    """Tenure in the index, and the transition flags around it.

    `months_in_index` counts consecutive month-ends of membership up to and including
    the current one, so it resets on a removal and re-entry rather than counting total
    lifetime membership. A name that left in 2009 and came back in 2015 is new in 2015,
    because the demand shock happened again.
    """
    member = panel.in_index[rows]                     # (R, S) bool
    r, s = member.shape

    tenure = np.zeros((r, s), dtype=np.float64)
    run = np.zeros(s, dtype=np.float64)
    for i in range(r):
        run = np.where(member[i], run + 1.0, 0.0)
        tenure[i] = run
    tenure[~member] = np.nan

    # A name is "new" for its first three month-ends. Three because the published
    # inclusion effect decays over roughly a quarter, and because at monthly granularity
    # a one-month window is a single observation per event.
    return {
        "months_in_index": tenure,
        "new_member": np.where(member, (tenure <= 3.0).astype(np.float64), np.nan),
        "log_tenure": np.where(member, np.log1p(tenure), np.nan),
    }


# --------------------------------------------------------------------------
# Dividends
# --------------------------------------------------------------------------

def _dividends(panel, rows: np.ndarray, data_cutoff: str | None) -> dict:
    """Trailing dividend yield and its year-on-year change, from discrete payments."""
    con = connect()
    where = "WHERE action_type = 'dividend'"
    params: list = []
    if data_cutoff:
        where += " AND date <= ?"
        params.append(data_cutoff)
    try:
        div = con.execute(
            f"SELECT security_id, date, value FROM corporate_actions {where}",
            params).df()
    except Exception as exc:                                     # noqa: BLE001
        log.warning("dividend features unavailable: %s", exc)
        return {}

    sid_pos = {s: i for i, s in enumerate(panel.security_ids.tolist())}
    div = div[div["security_id"].isin(sid_pos)]
    if div.empty:
        log.warning("no dividend events matched the panel's securities")
        return {}

    # Dividends are declared in AS-TRADED dollars per share, while the panel's share
    # count is in split-adjusted space. Summing raw per-share dividends across a split
    # would add pre-split and post-split dollars together, so each payment is divided by
    # the split ratio in force on its own date before anything is accumulated.
    col = div["security_id"].map(sid_pos).to_numpy(dtype=np.int64)
    date_pos = {d: i for i, d in enumerate(panel.dates.tolist())}
    row = div["date"].map(date_pos)
    keep = row.notna().to_numpy()
    row = row[keep].to_numpy(dtype=np.int64)
    col, amt = col[keep], div.loc[keep, "value"].to_numpy(dtype=np.float64)

    split = panel.cum_split[row, col]
    per_share = np.where(np.isfinite(split) & (split > 0), amt / split, np.nan)

    paid = np.zeros((panel.n_dates, panel.n_securities), dtype=np.float64)
    np.add.at(paid, (row, col), np.nan_to_num(per_share))

    cum = np.cumsum(paid, axis=0)
    trailing = _window_sum(cum, panel.dates, rows, DIV_WINDOW_DAYS)
    prior = _window_sum(cum, panel.dates, rows, DIV_WINDOW_DAYS,
                        offset_days=DIV_WINDOW_DAYS)

    # `raw_close`, not `adj_close`: the numerator has just been put into split-adjusted
    # share space, and `raw_close` is the price in that same space (ADR-007). Dividing by
    # the TOTAL-RETURN adjusted close would divide a per-share dividend by a price that
    # has already had every past dividend compounded out of it, and the resulting
    # "yield" would drift upward with age for reasons that have nothing to do with the
    # company.
    price = panel.raw_close[rows]
    with np.errstate(divide="ignore", invalid="ignore"):
        yield_ = np.where(price > 0, trailing / price, np.nan)
        growth = np.where(prior > 0, trailing / prior - 1.0, np.nan)

    payer = trailing > 0
    return {
        "div_yield": np.where(payer, yield_, np.nan),
        "div_growth_1y": np.where(payer & (prior > 0), growth, np.nan),
        # A cut is rare, informative and asymmetric, so it gets its own flag rather than
        # being left as the left tail of the growth feature - a strategy can then avoid
        # cutters without taking a view on the size of the cut.
        "div_cut": np.where(payer & (prior > 0),
                            (trailing < prior * 0.95).astype(np.float64), np.nan),
        "pays_dividend": payer.astype(np.float64),
        "div_due_1m": _div_due(panel, rows, row, col),
    }


def _div_due(panel, rows: np.ndarray, ev_row: np.ndarray, ev_col: np.ndarray
             ) -> np.ndarray:
    """(R, S) 1.0 where the payment cadence predicts an ex-dividend within ~a month.

    Hartzmark & Solomon's dividend-month premium is about PREDICTED dividends: prices
    drift up in months where a payment is expected. The prediction here uses nothing but
    the security's own past ex-dates - the median gap between its recent payments is its
    cadence, and a name is "due" when that cadence says the next payment lands within
    roughly the coming month. A real trader would know more (declarations precede
    ex-dates by weeks), so this is the conservative, point-in-time-safe version.

    NaN until three payments have been observed - a cadence needs two gaps before it is
    a cadence. 0.0 for an established payer that is not due. The gap bounds [20, 200]
    admit monthly through semiannual payers and refuse to extrapolate anything slower.
    """
    out = np.full((len(rows), panel.n_securities), np.nan)
    if not len(ev_row):
        return out

    dates_dt = pd.to_datetime(pd.Series(panel.dates.tolist())).to_numpy()
    grid = dates_dt[rows]
    day = np.timedelta64(1, "D")

    order = np.lexsort((ev_row, ev_col))
    r_sorted, c_sorted = ev_row[order], ev_col[order]
    bounds = np.searchsorted(c_sorted, np.arange(panel.n_securities + 1))

    for s in range(panel.n_securities):
        ev = np.unique(r_sorted[bounds[s]:bounds[s + 1]])
        if len(ev) < 3:
            continue
        ev_dates = dates_dt[ev]
        k = np.searchsorted(ev_dates, grid, side="right")
        known = k >= 3
        if not known.any():
            continue
        col_out = np.full(len(rows), np.nan)
        for j in np.flatnonzero(known):
            recent = ev_dates[max(0, k[j] - 5):k[j]]
            cadence = float(np.median(np.diff(recent) / day))
            since = float((grid[j] - recent[-1]) / day)
            due = cadence - since
            col_out[j] = float(20.0 <= cadence <= 200.0 and -10.0 <= due <= 31.0)
        out[:, s] = col_out
    return out


def _window_sum(cum: np.ndarray, dates: np.ndarray, rows: np.ndarray,
                days: int, offset_days: int = 0) -> np.ndarray:
    """(R, S) sum of a per-session quantity over a trailing calendar window.

    Differencing a cumulative sum, so the cost is two gathers rather than a rolling pass
    over 4,900 x 677 cells. `offset_days` shifts the whole window back, which is how the
    prior-year window for the growth comparison is built.
    """
    d = pd.to_datetime(pd.Series(dates.tolist()))
    at = d.iloc[rows].reset_index(drop=True)
    hi_date = at - pd.Timedelta(days=offset_days)
    lo_date = hi_date - pd.Timedelta(days=days)

    hi = np.searchsorted(d.to_numpy(), hi_date.to_numpy(), side="right") - 1
    lo = np.searchsorted(d.to_numpy(), lo_date.to_numpy(), side="right") - 1

    out = np.full((len(rows), cum.shape[1]), np.nan)
    ok = hi >= 0
    out[ok] = cum[hi[ok]] - np.where(lo[ok, None] >= 0, cum[np.maximum(lo[ok], 0)], 0.0)
    return out
