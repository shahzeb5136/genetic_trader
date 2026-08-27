"""Worked examples of querying the data layer correctly.

Run:  python scripts/explore.py

This is documentation-as-code. Each section demonstrates a query pattern that is easy
to get wrong, and shows the wrong version alongside the right one so the difference is
concrete. There are no strategies here - only correct data access.
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from sp500lab.query import (connect, fundamentals_asof,
                            prices_clipped_to_membership, universe_asof)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def hdr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    con = connect()

    # ------------------------------------------------------------------
    hdr("1. Survivorship bias, measured on this dataset")
    # WRONG: today's constituents, applied to a historical date.
    wrong = set(con.sql("SELECT ticker FROM sp500_current").df()["ticker"])
    # RIGHT: who was actually in the index on that date.
    right = set(universe_asof("2008-09-30", con)["ticker"])

    print(f"  today's constituent list          : {len(wrong)} names")
    print(f"  actual members on 2008-09-30      : {len(right)} names")
    print(f"  in the index then, absent today   : {len(right - wrong)}")
    print(f"  in today's list, not yet added    : {len(wrong - right)}")
    print("\n  Backtesting 2008 with today's list silently deletes those "
          f"{len(right - wrong)} companies -")
    print("  every one that failed, was acquired, or was demoted.")

    # ------------------------------------------------------------------
    hdr("2. The 2008 financial crisis, as the data sees it")
    print(con.sql("""
        SELECT ticker, start_date, end_date, end_is_open
        FROM sp500_membership_intervals
        WHERE ticker IN ('LEH','BSC','MER','FNM','FRE','WM','AIG')
        ORDER BY ticker, start_date
    """).df().to_string(index=False))
    print("\n  Note WM appears twice: Washington Mutual (failed 2008) and")
    print("  Waste Management (added 2009). Same ticker, different companies.")
    print("  This is why joins use security_id, never ticker.")

    # ------------------------------------------------------------------
    hdr("3. Point-in-time fundamentals: the look-ahead trap")
    for as_of in ("2023-10-15", "2023-11-05"):
        n = con.execute("""
            SELECT count(*) FROM xbrl_facts
            WHERE ticker='AAPL' AND period_end='2023-09-30' AND filed_date <= ?
        """, [as_of]).fetchone()[0]
        print(f"  facts about AAPL's FY2023 knowable on {as_of}: {n}")
    print("\n  Apple's fiscal year ended 2023-09-30 but the 10-K was filed 2023-11-03.")
    print("  Filtering on period_end alone hands a model those numbers 33 days early.")
    print("  Always use fundamentals_asof(), which filters on filed_date.")

    # ------------------------------------------------------------------
    hdr("4. Ticker recycling: clip prices to membership")
    # NB: `first` and `last` are reserved words in DuckDB - alias around them.
    raw = con.sql("""
        SELECT ticker, count(*) AS bars,
               min(date) AS first_bar, max(date) AS last_bar
        FROM daily_bars_adjusted WHERE ticker='CPWR' GROUP BY 1
    """).df()
    clipped = prices_clipped_to_membership(con)
    c = clipped[clipped["ticker"] == "CPWR"]
    print("  raw ticker join:")
    print(raw.to_string(index=False))
    if len(c):
        print(f"\n  clipped to membership: {len(c)} bars, "
              f"{c['date'].min()} .. {c['date'].max()}")
    print("\n  Compuware left the index in 2014; CPWR now belongs to an OTC shell")
    print("  trading 0-50 shares/day. The raw join splices them into one series.")

    # ------------------------------------------------------------------
    hdr("5. Adjusted vs raw prices across a split")
    print(con.sql("""
        SELECT date, round(close,2) raw_close, round(adj_factor,6) factor,
               round(adj_close,2) adj_close
        FROM daily_bars_adjusted
        WHERE ticker='NVDA' AND date BETWEEN '2024-06-06' AND '2024-06-12'
        ORDER BY date
    """).df().to_string(index=False))
    print("\n  yfinance pre-applies splits, so adj_factor here reflects dividends only.")
    print("  Applying split factors again would double-count by 10x (see ADR-007).")

    # ------------------------------------------------------------------
    hdr("6. Universe size over time")
    print(con.sql("""
        SELECT substr(snapshot_date,1,4) AS year,
               count(DISTINCT ticker) AS distinct_tickers,
               count(*)/count(DISTINCT snapshot_date) AS avg_per_snapshot
        FROM sp500_membership_snapshots
        WHERE substr(snapshot_date,6,2)='12'
        GROUP BY 1 ORDER BY 1
    """).df().to_string(index=False))

    # ------------------------------------------------------------------
    hdr("7. Macro series and their revision risk")
    print(con.sql("""
        SELECT series_id, description, revised, count(*) AS obs,
               min(date) AS first_obs, max(date) AS last_obs
        FROM fred_series GROUP BY 1,2,3 ORDER BY revised DESC, series_id
    """).df().to_string(index=False))
    print("\n  revised=true means the series is restated after publication.")
    print("  Using those values at face value in a backtest is a look-ahead leak.")

    print("\nDone. See docs/DATA_DICTIONARY.md for every table and column.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
