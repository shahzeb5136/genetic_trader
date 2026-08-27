"""Point-in-time fundamentals from SEC EDGAR XBRL company facts.

Why this source
---------------
This is the primary record - the same filings that commercial fundamental vendors
parse and resell. It is free, it has no redistribution fee, and crucially every fact
carries the accession's **filing date**, which is what makes genuine point-in-time
analysis possible.

The bitemporal contract
-----------------------
Every row carries two dates, and conflating them is the classic way to leak future
information into a backtest:

  period_end    (effective_date) - when the fact was TRUE about the world.
                Q4 2023 revenue has period_end 2023-12-31.

  filed_date    (knowledge_date) - when you could first have KNOWN it.
                That same Q4 revenue was not public until the 10-K filed in
                February 2024.

A model trading on 2024-01-15 must see nothing with filed_date > 2024-01-15. Using
period_end alone would hand it February's numbers a month early - a leak that looks
like exceptional alpha and evaporates in live trading.

Restatements make this sharper: the same (tag, period_end) legitimately appears
several times with different filed_dates and different values. We keep every
version. `as_of` queries take the latest row with filed_date <= the query date,
which reproduces what was actually on the tape that day rather than today's
restated view.

Volume control
--------------
A large filer's companyfacts payload runs to several MB and thousands of tags. By
default we keep a curated set of line items sufficient for standard factor work
(see DEFAULT_TAGS); pass all_tags=True to retain everything. Raw JSON always lands
in bronze either way, so widening the tag list later never needs a re-download.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from ..http_cache import fetch
from ..storage import read_silver, silver_exists, today_iso, write_bronze, write_silver
from .base import IngestResult

log = logging.getLogger(__name__)

SOURCE = "sec"
DATASET = "companyfacts"
URL_TMPL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

#: Curated line items covering income statement, balance sheet, cash flow and share
#: counts - enough for value/quality/growth factors without storing every tag.
DEFAULT_TAGS = {
    # Income statement
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
    "CostOfRevenue", "CostOfGoodsAndServicesSold", "GrossProfit",
    "OperatingIncomeLoss", "OperatingExpenses",
    "ResearchAndDevelopmentExpense",
    "SellingGeneralAndAdministrativeExpense",
    "NetIncomeLoss", "ProfitLoss", "IncomeLossFromContinuingOperations",
    "EarningsPerShareBasic", "EarningsPerShareDiluted",
    "InterestExpense", "IncomeTaxExpenseBenefit",
    # Balance sheet
    "Assets", "AssetsCurrent", "Liabilities", "LiabilitiesCurrent",
    "LiabilitiesAndStockholdersEquity", "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "CashAndCashEquivalentsAtCarryingValue", "ShortTermInvestments",
    "InventoryNet", "AccountsReceivableNetCurrent", "AccountsPayableCurrent",
    "LongTermDebt", "LongTermDebtNoncurrent", "LongTermDebtCurrent",
    "DebtCurrent", "Goodwill", "IntangibleAssetsNetExcludingGoodwill",
    "PropertyPlantAndEquipmentNet", "RetainedEarningsAccumulatedDeficit",
    # Cash flow
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInInvestingActivities",
    "NetCashProvidedByUsedInFinancingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "DepreciationDepletionAndAmortization",
    "PaymentsOfDividendsCommonStock", "PaymentsForRepurchaseOfCommonStock",
    # Share counts
    "CommonStockSharesOutstanding", "CommonStockSharesIssued",
    "WeightedAverageNumberOfSharesOutstandingBasic",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
}


def _flatten_facts(payload: dict, cik: int, keep: set[str] | None) -> list[dict]:
    """Turn the nested companyfacts JSON into flat fact rows."""
    rows: list[dict] = []
    entity = payload.get("entityName", "")
    for taxonomy, tags in (payload.get("facts") or {}).items():
        for tag, body in tags.items():
            if keep is not None and tag not in keep:
                continue
            for unit, observations in (body.get("units") or {}).items():
                for ob in observations:
                    filed = ob.get("filed")
                    if not filed:
                        continue  # without a filing date the row is useless for PIT
                    rows.append({
                        "cik": cik,
                        "entity_name": entity,
                        "taxonomy": taxonomy,
                        "tag": tag,
                        "unit": unit,
                        "period_start": ob.get("start"),
                        "period_end": ob.get("end"),
                        "value": ob.get("val"),
                        "fy": ob.get("fy"),
                        "fp": ob.get("fp"),
                        "form": ob.get("form"),
                        "filed_date": filed,
                        "accession": ob.get("accn"),
                        "frame": ob.get("frame"),
                    })
    return rows


def universe_ciks(universe: str = "current", limit: int | None = None) -> pd.DataFrame:
    """(security_id, cik, ticker) for the companies to pull."""
    if universe == "ever" and silver_exists("reference/sp500_membership_intervals"):
        iv = read_silver("reference/sp500_membership_intervals")[["security_id", "ticker"]]
        master = read_silver("reference/security_master")[["security_id", "cik"]]
        df = iv.merge(master, on="security_id", how="left")
    else:
        df = read_silver("reference/sp500_current")[["security_id", "ticker", "cik"]]

    df = df[df["cik"].fillna(0).astype("int64") > 0]
    df = df.drop_duplicates(subset=["cik"]).sort_values("ticker").reset_index(drop=True)
    return df.head(limit) if limit else df


def run(
    force: bool = False,
    universe: str = "current",
    limit: int | None = None,
    all_tags: bool = False,
) -> IngestResult:
    res = IngestResult(source=SOURCE, dataset=DATASET)
    keep = None if all_tags else DEFAULT_TAGS
    ingest_date = today_iso()

    targets = universe_ciks(universe, limit)
    log.info("companyfacts for %d CIKs (universe=%s, all_tags=%s)",
             len(targets), universe, all_tags)

    frames: list[pd.DataFrame] = []
    failed: list[dict] = []

    for i, row in enumerate(targets.itertuples(index=False), 1):
        cik = int(row.cik)
        url = URL_TMPL.format(cik=cik)
        try:
            # Filings are append-only history; a 7-day TTL picks up new filings
            # without re-pulling multi-MB payloads on every run.
            resp = fetch(url, source=SOURCE, ttl_seconds=7 * 86_400, force=force)
        except Exception as exc:  # noqa: BLE001
            failed.append({"cik": cik, "ticker": row.ticker, "error": str(exc)[:200]})
            continue

        res.fetched += 0 if resp.from_cache else 1
        res.from_cache += 1 if resp.from_cache else 0

        if not resp.from_cache:
            write_bronze(source=SOURCE, dataset=DATASET,
                         filename=f"CIK{cik:010d}.json", content=resp.content,
                         url=url, ingest_date=ingest_date,
                         extra={"ticker": row.ticker})
            res.bronze_files += 1

        try:
            rows = _flatten_facts(resp.json(), cik, keep)
        except Exception as exc:  # noqa: BLE001
            failed.append({"cik": cik, "ticker": row.ticker, "error": f"parse: {exc}"[:200]})
            continue

        if rows:
            d = pd.DataFrame(rows)
            d["security_id"] = row.security_id
            d["ticker"] = row.ticker
            frames.append(d)

        if i % 50 == 0:
            log.info("  %d/%d CIKs  (%d net, %d cached)", i, len(targets),
                     res.fetched, res.from_cache)

    if not frames:
        res.errors.append("no facts parsed")
        return res

    facts = pd.concat(frames, ignore_index=True)
    for c in ("period_start", "period_end", "filed_date"):
        facts[c] = facts[c].astype("string")
    facts["value"] = pd.to_numeric(facts["value"], errors="coerce")

    # Same fact can be reported identically in several filings; keep one per
    # (security, tag, unit, period, filing) so restatements survive but noise does not.
    facts = facts.drop_duplicates(
        subset=["security_id", "taxonomy", "tag", "unit",
                "period_start", "period_end", "filed_date", "accession"])
    facts = facts.sort_values(["security_id", "tag", "period_end", "filed_date"])

    cols = ["security_id", "ticker", "cik", "entity_name", "taxonomy", "tag", "unit",
            "period_start", "period_end", "value", "fy", "fp", "form",
            "filed_date", "accession", "frame"]
    write_silver(facts[cols].reset_index(drop=True), "fundamentals/xbrl_facts")

    if failed:
        write_bronze(source=SOURCE, dataset=DATASET, filename="fetch_failures.json",
                     content=json.dumps(failed, indent=2).encode(),
                     url="(error report)", ingest_date=ingest_date)
        res.bronze_files += 1
        res.errors = [f"{f['ticker']}: {f['error'][:80]}" for f in failed[:20]]

    # How often is a fact restated? A useful sanity signal on the PIT machinery.
    dup = (facts.groupby(["security_id", "tag", "period_end"])["filed_date"]
           .nunique())
    res.rows = len(facts)
    res.notes = {
        "companies": int(facts["security_id"].nunique()),
        "distinct_tags": int(facts["tag"].nunique()),
        "filed_date_range": f"{facts['filed_date'].min()} .. {facts['filed_date'].max()}",
        "period_end_range": f"{facts['period_end'].min()} .. {facts['period_end'].max()}",
        "facts_restated_at_least_once": int((dup > 1).sum()),
        "failed_ciks": len(failed),
    }
    return res
