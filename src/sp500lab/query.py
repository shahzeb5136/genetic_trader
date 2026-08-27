"""DuckDB query layer over the parquet lake.

Opens a DuckDB connection with a view registered for every silver/gold dataset, so
the whole project is queryable with plain SQL and no loading step:

    from sp500lab.query import connect
    con = connect()
    con.sql("SELECT * FROM daily_bars WHERE ticker='AAPL' LIMIT 5").show()

DuckDB reads parquet directly off disk with predicate pushdown, so a filtered query
over 3.5M bars touches only the columns and row groups it needs. There is no server,
no import step, and the whole dataset stays a directory of files you can copy.

The two helpers below exist because they encode rules that are easy to get wrong and
catastrophic when wrong. Use them rather than hand-rolling the equivalent SQL.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from .paths import GOLD_DIR, SILVER_DIR

log = logging.getLogger(__name__)


def _view_name(path: Path, root: Path) -> str:
    """market/daily_bars/data.parquet -> daily_bars ; disambiguate on collision."""
    rel = path.parent.relative_to(root)
    parts = list(rel.parts)
    return parts[-1] if parts else path.stem


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with a view per dataset. Cheap - call it freely."""
    con = duckdb.connect(":memory:")
    registered: dict[str, Path] = {}

    for root, prefix in ((SILVER_DIR, ""), (GOLD_DIR, "gold_")):
        if not root.exists():
            continue
        for pq in sorted(root.rglob("*.parquet")):
            name = prefix + _view_name(pq, root)
            if name in registered:  # collision: qualify with parent folder
                name = prefix + "_".join(pq.parent.relative_to(root).parts)
            con.execute(
                f'CREATE OR REPLACE VIEW "{name}" AS '
                f"SELECT * FROM read_parquet('{pq.as_posix()}')")
            registered[name] = pq

    log.debug("registered %d views", len(registered))
    return con


def list_views() -> pd.DataFrame:
    """Every queryable dataset with row count and on-disk size."""
    rows = []
    for root, prefix in ((SILVER_DIR, ""), (GOLD_DIR, "gold_")):
        if not root.exists():
            continue
        for pq in sorted(root.rglob("*.parquet")):
            name = prefix + _view_name(pq, root)
            try:
                n = duckdb.sql(
                    f"SELECT count(*) FROM read_parquet('{pq.as_posix()}')").fetchone()[0]
            except Exception:  # noqa: BLE001
                n = -1
            rows.append({"view": name, "rows": n,
                         "mb": round(pq.stat().st_size / 1e6, 2),
                         "path": str(pq.relative_to(root.parent))})
    return pd.DataFrame(rows).sort_values("view").reset_index(drop=True)


# --------------------------------------------------------------------------
# Point-in-time helpers. Use these instead of hand-written SQL.
# --------------------------------------------------------------------------

def universe_asof(as_of: str, con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Index constituents as of a date, from the point-in-time membership intervals.

    This is the survivorship-free universe. Selecting from `sp500_current` instead
    would give today's members for every historical date - the single most common
    way to inflate a backtest.
    """
    con = con or connect()
    return con.execute("""
        SELECT security_id, ticker, start_date, end_date
        FROM sp500_membership_intervals
        WHERE start_date <= ?
          AND (end_is_open OR end_date >= ?)
        ORDER BY ticker
    """, [as_of, as_of]).df()


def fundamentals_asof(
    as_of: str,
    tags: list[str] | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Latest fundamental value per (security, tag, period) KNOWABLE on `as_of`.

    Enforces the bitemporal rule that makes fundamentals safe to backtest on:

      * filed_date <= as_of  - never see a filing before it was published
      * among the filings that qualify, take the most recently filed one, so a
        restatement is picked up only from the date it was actually restated

    Querying on period_end alone would hand a model February's 10-K numbers in
    January. That leak looks like exceptional alpha and vanishes in live trading.
    """
    con = con or connect()
    tag_filter, params = "", [as_of]
    if tags:
        placeholders = ", ".join("?" for _ in tags)
        tag_filter = f"AND tag IN ({placeholders})"
        params += list(tags)

    return con.execute(f"""
        WITH visible AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY security_id, tag, unit, period_end
                       ORDER BY filed_date DESC, accession DESC
                   ) AS rn
            FROM xbrl_facts
            WHERE filed_date <= ?
            {tag_filter}
        )
        SELECT security_id, ticker, cik, tag, unit, period_start, period_end,
               value, form, filed_date, accession
        FROM visible
        WHERE rn = 1
        ORDER BY ticker, tag, period_end
    """, params).df()


def prices_clipped_to_membership(
    con: duckdb.DuckDBPyConnection | None = None,
    warmup_days: int = 400,
) -> pd.DataFrame:
    """Adjusted bars restricted to each security's index-membership window.

    Guards against the ticker-recycling trap documented in quality/checks.py: a
    delisted member's symbol can be reassigned to an unrelated company, and a feed
    keyed on the bare ticker splices the two histories together seamlessly. Clipping
    to the membership interval discards the impostor's bars.

    `warmup_days` extends the window backwards so indicators have history before the
    first date a name is actually tradable. It does NOT extend forward - data after
    membership ends is exactly what we are trying to exclude.
    """
    con = con or connect()
    return con.execute("""
        SELECT b.*
        FROM daily_bars_adjusted b
        JOIN sp500_membership_intervals m
          ON b.security_id = m.security_id
         AND b.date >= CAST(CAST(m.start_date AS DATE) - INTERVAL (?) DAY AS VARCHAR)
         AND (m.end_is_open OR b.date <= m.end_date)
    """, [warmup_days]).df()
