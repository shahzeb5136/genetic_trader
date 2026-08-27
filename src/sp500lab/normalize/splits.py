"""Recovering as-traded prices from a split-adjusted feed.

The problem, restated
---------------------
Our stored `close` is split-adjusted, because yfinance pre-applies splits even with
auto_adjust=False (ADR-007). It is therefore NOT the price that changed hands. NVDA's
2020 close is stored around $13 because of the 2021 4:1 and 2024 10:1 splits; what
actually traded was around $520.

Anything that reasons about the physical market rather than about returns needs the
as-traded price:

  * share counts, for a per-share commission (backtest/costs.py)
  * the minimum tick, for a spread floor (backtest/spreads.py)
  * shares outstanding x price, for point-in-time market cap (TODO-5)

`adj_factor_price` cannot do this job: under the `split_adjusted` convention it is 1.0
everywhere by construction, because there is nothing left for it to undo. The ratio has
to be rebuilt from the corporate-action events themselves.

    cum_split(t)       = product of every split ratio with ex-date STRICTLY AFTER t
    as_traded_price(t) = stored_close(t) * cum_split(t)

Strictly after, because a split's ex-date is the first session quoted at the new price.
Including it would divide that session's price by its own ratio.

Sanity check, and it is a real one: NVDA on 2024-06-06 must come back around $1,210,
not $121. Being wrong by a clean 10x means the product is inverted or the boundary is
off by one session.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def cumulative_split_ratio(
    dates: np.ndarray,
    sid_index: dict[str, int],
    splits: pd.DataFrame,
) -> np.ndarray:
    """(D, S) product of split ratios with ex-date strictly after each session.

    `dates` must be sorted ascending. `splits` needs security_id, date, value, where
    value is the ratio (10.0 for a 10:1 split). Securities with no splits get 1.0.
    """
    out = np.ones((len(dates), len(sid_index)), dtype=np.float64)
    if splits is None or splits.empty:
        return out
    for r in splits.itertuples(index=False):
        s = sid_index.get(r.security_id)
        if s is None or not np.isfinite(r.value) or r.value <= 0 or r.value == 1:
            continue
        hi = int(np.searchsorted(dates, r.date, side="left"))
        if hi > 0:
            out[:hi, s] *= float(r.value)
    return out


def load_splits(con) -> pd.DataFrame:
    """Every split event from silver, ready for cumulative_split_ratio()."""
    return con.execute("""
        SELECT security_id, date, value FROM corporate_actions
        WHERE action_type = 'split' AND value > 0 AND value <> 1
        ORDER BY date
    """).df()


#: NYSE and Nasdaq completed decimalisation on 2001-04-09. Before that the minimum
#: increment was a sixteenth of a dollar - 6.25x wider - which is a real regime
#: difference in trading costs, not a rounding detail.
DECIMALISATION_DATE = "2001-04-09"
TICK_PRE_DECIMAL = 0.0625
TICK_POST_DECIMAL = 0.01


def tick_size(dates: np.ndarray) -> np.ndarray:
    """(D,) minimum price increment per session."""
    return np.where(np.asarray(dates) < DECIMALISATION_DATE,
                    TICK_PRE_DECIMAL, TICK_POST_DECIMAL)
