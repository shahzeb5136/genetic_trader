"""SEC company_tickers_exchange.json -> CIK / ticker / exchange spine.

This is the identity backbone for everything else. CIK is the SEC's permanent
registrant number: it survives name changes, ticker changes and reincorporations,
which is exactly the property a security master needs.

Coverage caveat: this file lists *current* SEC registrants only. Companies that
delisted years ago keep their CIK in EDGAR but drop out of this file, so it cannot
by itself supply a survivorship-free universe. That gap is filled from the
Wikipedia revision history (ingest/wikipedia_history.py) and, later, from a paid
constituent dataset.

Source: https://www.sec.gov/files/company_tickers_exchange.json  (free, no key)
Refresh: daily-ish upstream. We use a 1-day TTL.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..http_cache import fetch
from ..registry import SecurityRegistry
from ..storage import today_iso, write_bronze, write_silver
from .base import IngestResult

log = logging.getLogger(__name__)

URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SOURCE = "sec"
DATASET = "company_tickers"


def run(force: bool = False) -> IngestResult:
    res = IngestResult(source=SOURCE, dataset=DATASET)

    resp = fetch(URL, source=SOURCE, ttl_seconds=86_400, force=force)
    res.fetched += 0 if resp.from_cache else 1
    res.from_cache += 1 if resp.from_cache else 0

    write_bronze(source=SOURCE, dataset=DATASET, filename="company_tickers_exchange.json",
                 content=resp.content, url=URL)
    res.bronze_files += 1

    payload = resp.json()
    df = pd.DataFrame(payload["data"], columns=payload["fields"])
    df = df.rename(columns={"name": "name"})
    df["cik"] = df["cik"].astype("int64")
    df["ticker"] = df["ticker"].astype(str).map(SecurityRegistry.normalize_ticker)
    df["exchange"] = df["exchange"].fillna("").astype(str)
    df["name"] = df["name"].fillna("").astype(str)
    df = df.drop_duplicates(subset=["cik", "ticker"]).reset_index(drop=True)

    # Mint stable internal IDs for every (cik, ticker) pair.
    reg = SecurityRegistry.load()
    df["security_id"] = reg.bulk_assign(df, seen_date=today_iso())
    reg.save()

    df["as_of"] = today_iso()
    write_silver(df[["security_id", "cik", "ticker", "name", "exchange", "as_of"]],
                 f"reference/{DATASET}")

    res.rows = len(df)
    res.notes = {"registrants": int(df["cik"].nunique()),
                 "securities": len(df),
                 "exchanges": df["exchange"].value_counts().head(6).to_dict()}
    return res
