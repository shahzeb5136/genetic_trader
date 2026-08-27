"""What happens to a position when the company stops being one.

The failure this prevents
-------------------------
A backtest holds a name, the price series ends, and the position quietly disappears
from the accounting. The strategy never books the outcome. This is survivorship bias
running in the opposite direction - it deletes the losses instead of the losers - and
it is worse than the usual kind because it is invisible: nothing errors, no row is
missing, the equity curve just does not contain the day Lehman went to zero.

The subtlety that makes this worse, not better, with a paid feed
----------------------------------------------------------------
Only a handful of cases are visible today because Yahoo carries almost no delisted
names - the bars simply are not there to be mishandled. Buying full coverage raises
that to hundreds. Fixing the coverage gap EXPOSES this problem rather than solving it,
which is why this exists before the EODHD migration rather than after it (TODO-3).

Three outcomes that are not the same event
------------------------------------------
    index_removal   still trading, just no longer in the index (market-cap decline,
                    a reconstitution). NOT a delisting. Sell at the next open at the
                    prevailing price. delist_return = 0.
    acquisition     bought for cash or stock. The position exits at deal terms, which
                    we do not have; approximate with the last traded price.
                    delist_return = 0, and the approximation is recorded.
    bankruptcy      the equity is wiped out. delist_return = -1.0.

Treating a removal as a bankruptcy fabricates catastrophic losses. Treating a
bankruptcy as a removal silently deletes them. Both are large; they do not cancel.

What this is not
----------------
It is not a delisting-return dataset. CRSP has one and it costs orders of magnitude
more than this project's entire budget. There is no free authoritative source. What
this produces is an **explicit, recorded assumption** per security - the `assumption`
column says in words what was assumed and why - so a reader can disagree with a
specific number instead of discovering a silent one.

Coverage honesty
----------------
`sp500_changes` is under-recorded before 2010 (ADR-010: ~7 events/year against the ~20
that actually happened), so an early-era name is likely to end up `unresolved`. That is
reported per run rather than defaulted away. `unresolved` behaves as `index_removal`,
which is the conservative choice for the common case and the wrong one for a
bankruptcy - and the engine tells you how much of the equity curve depended on it.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from ..query import connect
from ..storage import write_gold

log = logging.getLogger(__name__)

REASON_CATEGORIES = ("bankruptcy", "acquisition", "index_removal", "unresolved")

#: Order matters: the first pattern that matches wins. Bankruptcy is checked first
#: because "X filed for bankruptcy and was acquired out of it" must not read as a
#: clean acquisition.
_PATTERNS: tuple[tuple[str, str], ...] = (
    ("bankruptcy", r"bankrupt|chapter\s*(?:7|11)|liquidat|insolven|receivership|"
                   r"wound\s*up|winding\s*up|delisted.*(?:failure|collapse)"),
    ("acquisition", r"acquir|merg|bought by|taken over|takeover|purchas|"
                    r"go(?:es|ing)? private|take[- ]private|buyout|"
                    r"combin\w+ with|split[- ]off|exchange offer"),
    ("index_removal", r"market cap|reorganiz|reorganis|reconstitut|index|"
                      r"no longer|criteria|representation|rebalanc"),
)

#: A spin-off removes the PARENT from nothing - the parent keeps trading. These are
#: matched only to be excluded from the acquisition bucket above, which would
#: otherwise catch "completed the corporate spin-off of ...".
_SPINOFF = re.compile(r"spin[- ]?off", re.I)


def classify_reason(text: str | float | None) -> tuple[str, str]:
    """(category, assumption) from the free-text reason on an index change event."""
    if text is None or (isinstance(text, float) and np.isnan(text)) or not str(text).strip():
        return "unresolved", "no reason text recorded in sp500_changes"

    s = str(text).strip()
    low = s.lower()

    # A spin-off by the parent is a corporate action, not the parent's exit. If the
    # text is ONLY about a spin-off, this is a removal, not an acquisition.
    if _SPINOFF.search(low) and not re.search(_PATTERNS[0][1], low):
        if not re.search(r"acquir|merg|bought by", low):
            return "index_removal", f"spin-off by the removed name, still listed: {s[:120]}"

    for category, pattern in _PATTERNS:
        if re.search(pattern, low, flags=re.I):
            if category == "bankruptcy":
                note = f"reason text indicates failure; equity assumed worthless: {s[:120]}"
            elif category == "acquisition":
                note = ("deal terms unknown; exit approximated at the last traded price: "
                        f"{s[:120]}")
            else:
                note = f"still listed, removed from the index only: {s[:120]}"
            return category, note

    return "unresolved", f"reason text did not match any known pattern: {s[:120]}"


#: Terminal return applied to the last observed close, by category.
_RETURN_BY_CATEGORY = {
    "bankruptcy": -1.0,
    "acquisition": 0.0,
    "index_removal": 0.0,
    "unresolved": 0.0,
}


def build(write: bool = True) -> pd.DataFrame:
    """One row per security that leaves the panel, with an explicit exit assumption.

    Output `gold/backtest/delisting_returns`:

        security_id, ticker, last_bar_date, membership_end, delist_date,
        reason_category, reason_text, delist_return, source, assumption

    A security qualifies when its membership interval is closed - it left the index at
    some point. Names still in the index are excluded: they have not exited anything.
    """
    con = connect()

    ended = con.execute("""
        SELECT m.security_id, m.ticker, m.end_date AS membership_end,
               max(b.date) AS last_bar_date
        FROM sp500_membership_intervals m
        LEFT JOIN daily_bars_adjusted b ON b.security_id = m.security_id
        WHERE NOT m.end_is_open AND m.end_date IS NOT NULL
        GROUP BY 1, 2, 3
    """).df()
    if ended.empty:
        log.warning("delisting: no closed membership intervals found")
        return pd.DataFrame()

    changes = con.execute("""
        SELECT removed_ticker AS ticker, effective_date, reason
        FROM sp500_changes
        WHERE removed_ticker IS NOT NULL AND removed_ticker <> ''
    """).df()

    # Match on ticker + nearest effective date to the membership end. Ticker is a bad
    # join key in general (ADR-005) and sp500_changes carries no security_id, so the
    # date proximity is what disambiguates a reused symbol. A match more than a year
    # from the membership end is rejected rather than accepted loosely.
    rows = []
    for r in ended.itertuples(index=False):
        cand = changes[changes["ticker"] == r.ticker]
        reason_text, delist_date, source = None, None, "none"
        if len(cand):
            gap = (pd.to_datetime(cand["effective_date"])
                   - pd.to_datetime(r.membership_end)).abs()
            best = gap.idxmin()
            if gap.loc[best] <= pd.Timedelta(days=365):
                reason_text = cand.loc[best, "reason"]
                delist_date = cand.loc[best, "effective_date"]
                source = "sp500_changes"

        category, assumption = classify_reason(reason_text)
        if source == "none":
            assumption = ("no matching event in sp500_changes within 1 year of the "
                          "membership end; under-recorded before 2010 (ADR-010)")

        rows.append({
            "security_id": r.security_id,
            "ticker": r.ticker,
            "last_bar_date": r.last_bar_date,
            "membership_end": r.membership_end,
            "delist_date": delist_date or r.membership_end,
            "reason_category": category,
            "reason_text": (str(reason_text)[:300] if reason_text is not None else ""),
            "delist_return": _RETURN_BY_CATEGORY[category],
            "source": source,
            "assumption": assumption,
        })

    out = pd.DataFrame(rows)

    # A name whose bars run well past its membership end never actually delisted - it
    # was dropped from the index and kept trading. Force it to index_removal whatever
    # the text said, because the price series is stronger evidence than the wording.
    still = (pd.to_datetime(out["last_bar_date"])
             - pd.to_datetime(out["membership_end"])) > pd.Timedelta(days=90)
    reclassified = int((still & (out["reason_category"] == "bankruptcy")).sum())
    if reclassified:
        log.info("delisting: %d 'bankruptcy' rows still had bars 90+ days later; "
                 "reclassified as index_removal", reclassified)
    out.loc[still, "reason_category"] = "index_removal"
    out.loc[still, "delist_return"] = 0.0
    out.loc[still, "assumption"] = out.loc[still, "assumption"].str.replace(
        "^", "price series continues 90+ days past membership end, so still listed; ",
        regex=True)

    counts = out["reason_category"].value_counts().to_dict()
    log.info("delisting: %d securities - %s", len(out), counts)
    if write:
        write_gold(out, "backtest/delisting_returns")
    return out


def summarise(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Category counts by era. Read this before trusting any pre-2010 backtest."""
    if df is None:
        df = connect().execute("SELECT * FROM gold_delisting_returns").df()
    d = df.copy()
    d["era"] = np.where(d["membership_end"] < "2010-01-01", "pre-2010", "2010+")
    return (d.groupby(["era", "reason_category"]).size()
            .unstack(fill_value=0).reset_index())
