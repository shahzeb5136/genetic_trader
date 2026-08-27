"""Wikipedia: current S&P 500 constituents + index change events.

Two pages, two datasets:

* `List of S&P 500 companies`            -> today's 503 constituents with GICS
                                            sector, CIK and date-added.
* `Historical components of the S&P 500` -> add/remove events with effective date
                                            and a free-text reason.

**Known coverage limitation** (measured, not assumed): the change-event table
carries ~22 events/year for 2010-2026, which matches the real rate of S&P 500
turnover, but only ~43 events for the whole of 2000-2009 - roughly a fifth of what
actually happened. Treat this table as reliable from ~2010 and increasingly
incomplete before it. The `changes_completeness` quality report quantifies this per
year, and the revision-history reconstruction (ingest/wikipedia_history.py) is the
independent cross-check.

Both pages are edited continuously, so we use a short TTL and keep every daily
snapshot in bronze - the accumulated snapshots become their own point-in-time
record over time.
"""

from __future__ import annotations

import io
import logging
import re

import pandas as pd

from ..http_cache import fetch
from ..registry import SecurityRegistry
from ..storage import today_iso, write_bronze, write_silver
from .base import IngestResult

log = logging.getLogger(__name__)

SOURCE = "wikipedia"
CURRENT_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"

# Both pages change on index events; 6h keeps us current without hammering.
TTL = 6 * 3600


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a MultiIndex header ('Added','Ticker') -> 'Added_Ticker'."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(dict.fromkeys(str(x) for x in col)).strip("_")
            for col in df.columns
        ]
    else:
        df.columns = [str(c) for c in df.columns]
    return df


def _clean_ticker(val) -> str | None:
    """Wikipedia cells carry footnote markers and occasional stray text."""
    if pd.isna(val):
        return None
    s = re.sub(r"\[.*?\]", "", str(val)).strip()
    s = s.split(",")[0].strip()
    if not s or s.lower() in {"nan", "none", "-", "—"}:
        return None
    return SecurityRegistry.normalize_ticker(s)


def run_current(force: bool = False) -> IngestResult:
    res = IngestResult(source=SOURCE, dataset="sp500_current")
    resp = fetch(CURRENT_URL, source=SOURCE, ttl_seconds=TTL, force=force)
    res.fetched += 0 if resp.from_cache else 1
    res.from_cache += 1 if resp.from_cache else 0

    write_bronze(source=SOURCE, dataset="sp500_current",
                 filename="list_of_sp500_companies.html",
                 content=resp.content, url=CURRENT_URL)
    res.bronze_files += 1

    tables = pd.read_html(io.StringIO(resp.text()))
    # The constituent table is the one with a Symbol column and ~500 rows.
    df = next(t for t in tables
              if any(str(c).strip().lower() == "symbol" for c in t.columns) and len(t) > 400)
    df = _flatten_columns(df)
    df = df.rename(columns={
        "Symbol": "ticker", "Security": "name", "GICS Sector": "gics_sector",
        "GICS Sub-Industry": "gics_sub_industry",
        "Headquarters Location": "headquarters", "Date added": "date_added",
        "CIK": "cik", "Founded": "founded",
    })
    df["ticker"] = df["ticker"].map(_clean_ticker)
    df = df[df["ticker"].notna()].copy()
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").fillna(0).astype("int64")
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce").dt.date.astype("string")
    df["snapshot_date"] = today_iso()

    reg = SecurityRegistry.load()
    df["security_id"] = reg.bulk_assign(df, seen_date=today_iso())
    reg.save()

    cols = ["snapshot_date", "security_id", "ticker", "name", "cik",
            "gics_sector", "gics_sub_industry", "headquarters", "date_added", "founded"]
    write_silver(df[cols].reset_index(drop=True), "reference/sp500_current")

    res.rows = len(df)
    res.notes = {"constituents": len(df),
                 "sectors": int(df["gics_sector"].nunique()),
                 "missing_cik": int((df["cik"] == 0).sum())}
    return res


def run_changes(force: bool = False) -> IngestResult:
    res = IngestResult(source=SOURCE, dataset="sp500_changes")
    resp = fetch(CHANGES_URL, source=SOURCE, ttl_seconds=TTL, force=force)
    res.fetched += 0 if resp.from_cache else 1
    res.from_cache += 1 if resp.from_cache else 0

    write_bronze(source=SOURCE, dataset="sp500_changes",
                 filename="historical_components_sp500.html",
                 content=resp.content, url=CHANGES_URL)
    res.bronze_files += 1

    tables = pd.read_html(io.StringIO(resp.text()))
    raw = max(tables, key=len)
    df = _flatten_columns(raw)

    date_col = next(c for c in df.columns if "effective" in c.lower() or c.lower() == "date")
    df = df.rename(columns={
        date_col: "effective_date",
        "Added_Ticker": "added_ticker", "Added_Security": "added_name",
        "Removed_Ticker": "removed_ticker", "Removed_Security": "removed_name",
        "Reason": "reason",
    })
    for c in ("added_ticker", "removed_ticker", "added_name", "removed_name", "reason"):
        if c not in df.columns:
            df[c] = None

    df["effective_date"] = pd.to_datetime(
        df["effective_date"], errors="coerce", format="mixed").dt.date.astype("string")
    df["added_ticker"] = df["added_ticker"].map(_clean_ticker)
    df["removed_ticker"] = df["removed_ticker"].map(_clean_ticker)
    for c in ("added_name", "removed_name", "reason"):
        df[c] = df[c].astype("string").str.replace(r"\[.*?\]", "", regex=True).str.strip()

    df = df[df["effective_date"].notna()]
    df = df[df["added_ticker"].notna() | df["removed_ticker"].notna()]
    df = df.drop_duplicates(
        subset=["effective_date", "added_ticker", "removed_ticker"]).reset_index(drop=True)
    df["snapshot_date"] = today_iso()

    cols = ["effective_date", "added_ticker", "added_name",
            "removed_ticker", "removed_name", "reason", "snapshot_date"]
    write_silver(df[cols], "reference/sp500_changes")

    yr = pd.to_datetime(df["effective_date"]).dt.year
    res.rows = len(df)
    res.notes = {
        "events": len(df),
        "date_range": f"{df['effective_date'].min()} .. {df['effective_date'].max()}",
        "events_2010_plus": int((yr >= 2010).sum()),
        "events_pre_2010": int((yr < 2010).sum()),
    }
    return res


def run(force: bool = False) -> IngestResult:
    """Run both Wikipedia datasets and merge the result summaries."""
    a = run_current(force=force)
    b = run_changes(force=force)
    return IngestResult(
        source=SOURCE, dataset="sp500_current+changes",
        rows=a.rows + b.rows,
        bronze_files=a.bronze_files + b.bronze_files,
        from_cache=a.from_cache + b.from_cache,
        fetched=a.fetched + b.fetched,
        errors=a.errors + b.errors,
        notes={"current": a.notes, "changes": b.notes},
    )
