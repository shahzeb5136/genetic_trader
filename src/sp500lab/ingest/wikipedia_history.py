"""Point-in-time S&P 500 membership, reconstructed from Wikipedia revision history.

Why
---
The largest source of error in a naive S&P 500 backtest is survivorship bias:
downloading today's 503 constituents and testing them over 20 years silently
deletes every company that failed, was acquired, or was demoted. Published
comparisons of current-constituent vs point-in-time universes put the inflation at
roughly 1.5-2.0% annually. Lehman Brothers was in the index until September 2008;
a current-constituents backtest never sees it collapse.

Method
------
The Wikipedia article has been continuously edited since 2005 and its full revision
history is public. Taking the last revision of each month and parsing the
constituent table out of that revision's wikitext yields a monthly snapshot of who
was in the index *as believed at that time* - a genuine point-in-time record that
no amount of re-reading today's page can reproduce.

Performance: one old revision costs ~22s of fixed server latency, but the API
returns up to 50 revisions per call, so we batch and amortise it. Old revisions are
immutable and cached forever, making the job fully resumable - a crash at month 180
of 236 replays the first 179 from disk at zero network cost.

Accuracy and its limits
-----------------------
* Monthly sampling dates an add/remove to within one month. For daily-bar
  strategies rebalancing monthly or quarterly this is immaterial; the exact
  effective dates in `reference/sp500_changes` refine it where available.
* The table format changed repeatedly (plain ticker in 2008, a symbol template from
  ~2009, newline-separated cells from ~2020). The parser handles every observed
  variant and asserts a plausible row count, so a future format change fails loudly
  instead of silently yielding three tickers.
* Wikipedia is a volunteer-maintained secondary source. It is the best free option,
  not ground truth. Validate against a paid constituent set once acquired.
"""

from __future__ import annotations

import json
import logging
import re
from urllib.parse import quote

import pandas as pd

from ..config import get_settings
from ..http_cache import fetch
from ..registry import SecurityRegistry
from ..storage import today_iso, write_bronze, write_silver
from .base import IngestResult

log = logging.getLogger(__name__)

SOURCE = "wikipedia"
DATASET = "sp500_membership"
PAGE_TITLE = "List of S&P 500 companies"
API = "https://en.wikipedia.org/w/api.php"

REVIDS_PER_CALL = 50          # MediaWiki caps multi-revision content requests at 50
REV_LIST_TTL = 6 * 3600       # revision *metadata* is cheap; refresh a few times a day

# A snapshot outside this band means the format changed and the parser broke.
MIN_PLAUSIBLE = 350
MAX_PLAUSIBLE = 620

# Symbol templates: NyseSymbol|MMM, NASDAQ|AAPL, NYSE|X, BATS|.., NYSEARCA|.. and, since
# 2019-01, {{BZX link|CBOE}} for the one member listed on Cboe's own BZX exchange. That
# word was missing here until 2026-09-02, so CBOE "left the index" at the end of 2018 in
# every later snapshot (ADR-044). A template word this regex does not know drops the row
# silently; `quality`'s current-vs-intervals check is what makes the next one visible.
_TEMPLATE_TICKER = re.compile(
    r"\{\{\s*[A-Za-z0-9 _-]*(?:symbol|nyse|nasdaq|bats|bzx|cboe|arca|amex)[A-Za-z0-9 _-]*"
    r"\s*\|\s*([^}|\]]+?)\s*(?:\||\}\})", re.I)
_WIKILINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]|]+)\]\]")
_REF_TAG = re.compile(r"<ref.*?(?:/>|</ref>)", re.S | re.I)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_VALID_TICKER = re.compile(r"^[A-Z][A-Z0-9]{0,5}(?:\.[A-Z]{1,2})?$")

# Header and boilerplate words that are shaped exactly like valid tickers. Without
# this, a header cell that escapes row detection is silently ingested as a company:
# "SYMBOL" appeared in 10 snapshots (2023-11..2024-08) before this filter existed,
# and only surfaced when it failed to match anything in a vendor's symbol universe.
#
# What must NOT be here: any real ticker. "A" was on this list from the first commit -
# swept up with the "N/A" fragments - and Agilent Technologies, an index member since
# 2000 whose ticker is A, was therefore absent from every membership snapshot, every
# interval and every backtest until 2026-09-02 (ADR-044). "N/A" normalises to "N.A",
# which is what actually needs filtering; "N" and "NA" stay for the split fragments.
_NOT_TICKERS = {
    "SYMBOL", "TICKER", "SECURITY", "COMPANY", "CIK", "FOUNDED", "SECTOR",
    "GICS", "DATE", "ADDED", "HQ", "NOTES", "REPORTS", "N", "NA", "N.A",
}


# --------------------------------------------------------------------- revisions

def _api_url(params: dict[str, str]) -> str:
    return API + "?" + "&".join(f"{k}={quote(str(v), safe='|')}" for k, v in params.items())


def enumerate_revisions(force: bool = False) -> pd.DataFrame:
    """Every revision id + timestamp for the page. Metadata only, so it is cheap.

    Paginated with rvdir=newer, so the final page holds the most recent (and only
    changing) revisions. We use a uniform short TTL rather than trying to cache
    individual pages forever - the payload is small and correctness beats cleverness.
    """
    rows: list[dict] = []
    cont: dict[str, str] = {}
    page = 0
    while True:
        params = {
            "action": "query", "prop": "revisions", "titles": PAGE_TITLE,
            "rvprop": "ids|timestamp", "rvlimit": "500", "rvdir": "newer",
            "format": "json", "formatversion": "2",
        }
        params.update(cont)
        data = fetch(_api_url(params), source=SOURCE,
                     ttl_seconds=REV_LIST_TTL, force=force).json()
        for pg in data.get("query", {}).get("pages", []):
            for rev in pg.get("revisions", []):
                rows.append({"revid": rev["revid"], "timestamp": rev["timestamp"]})
        page += 1
        if "continue" not in data:
            break
        cont = dict(data["continue"])
        if page % 5 == 0:
            log.info("  enumerating revisions... %d so far (page %d)", len(rows), page)

    df = pd.DataFrame(rows).drop_duplicates(subset=["revid"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def select_month_end_revisions(revs: pd.DataFrame, start: str) -> pd.DataFrame:
    """Last revision within each calendar month, from `start` onward."""
    df = revs[revs["timestamp"] >= pd.Timestamp(start, tz="UTC")].copy()
    # tz-naive first: to_period() drops tz and warns otherwise
    df["month"] = df["timestamp"].dt.tz_convert(None).dt.to_period("M")
    picked = df.sort_values("timestamp").groupby("month", as_index=False).last()
    # end_time of a monthly Period is the last instant of that month.
    picked["snapshot_date"] = picked["month"].dt.end_time.dt.strftime("%Y-%m-%d")
    return picked[["snapshot_date", "revid", "timestamp"]].reset_index(drop=True)


def fetch_revision_contents(revids: list[int]) -> dict[int, str]:
    """Batch-fetch wikitext, REVIDS_PER_CALL at a time. Cached forever."""
    out: dict[int, str] = {}
    total = len(revids)
    for i in range(0, total, REVIDS_PER_CALL):
        chunk = revids[i:i + REVIDS_PER_CALL]
        ids = "|".join(str(r) for r in chunk)
        url = (f"{API}?action=query&prop=revisions&format=json&formatversion=2"
               f"&rvprop=ids|timestamp|content&rvslots=main&revids={ids}")
        data = fetch(url, source=SOURCE, ttl_seconds=None).json()
        for pg in data.get("query", {}).get("pages", []):
            for rev in pg.get("revisions", []):
                content = rev.get("slots", {}).get("main", {}).get("content")
                if content:
                    out[int(rev["revid"])] = content
        log.info("  wikitext %d/%d revisions", len(out), total)
    return out


# ----------------------------------------------------------------------- parsing

def _extract_ticker(cell: str) -> str | None:
    cell = _REF_TAG.sub("", _COMMENT.sub("", cell)).strip()
    if not cell:
        return None

    m = _TEMPLATE_TICKER.search(cell)
    cand = m.group(1) if m else None
    if cand is None:
        m = _WIKILINK.search(cell)
        cand = m.group(1) if m else None
    if cand is None:
        cand = re.sub(r"[\[\]{}'|]", " ", cell).strip()

    cand = cand.split("|")[0].strip().strip("'").strip()
    cand = SecurityRegistry.normalize_ticker(cand)
    if cand in _NOT_TICKERS:
        return None
    return cand if _VALID_TICKER.match(cand) else None


def _header_cells(table: str) -> list[str]:
    """Header cell texts, reading lines until the first data row."""
    cells: list[str] = []
    for line in table.split("\n"):
        s = line.strip()
        if s.startswith("!"):
            cells.extend(re.split(r"\s*!!\s*", s.lstrip("!").strip()))
        elif s.startswith("|") and not s.startswith(("|-", "|+", "|}")):
            break  # reached the first data row
    return cells


def _find_ticker_column(table: str) -> int:
    """Index of the ticker column, read from the header.

    The column order is NOT stable across the page's history: 2007 revisions put
    Company first and Ticker second, while 2008 onward lead with the ticker.
    Assuming column 0 silently harvested company names for the 2007 snapshots and
    produced ~4 valid tickers instead of 500. Reading the header makes the parser
    order-independent and robust to future reshuffles.
    """
    for i, cell in enumerate(_header_cells(table)):
        if re.search(r"ticker|symbol", cell, re.I):
            return i
    return 0


def parse_constituents(wikitext: str) -> list[str]:
    """Extract the ticker column of the constituent table from one revision."""
    start = wikitext.find("{|")
    if start == -1:
        return []
    end = wikitext.find("\n|}", start)
    table = wikitext[start:end if end != -1 else start + 400_000]

    col = _find_ticker_column(table)

    tickers: list[str] = []
    for row in table.split("|-"):
        row = row.strip()
        if not row or row.startswith("!"):
            continue
        parts = re.split(r"\|\||\n\s*\|", row)      # cells split on || or newline-|
        parts = [p for p in (x.lstrip("|").strip() for x in parts) if p]
        if len(parts) <= col:
            continue
        cell = parts[col]
        if cell.startswith("!"):
            continue
        tick = _extract_ticker(cell)
        if tick:
            tickers.append(tick)
    return list(dict.fromkeys(tickers))


# --------------------------------------------------------------------- intervals

def build_intervals(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Collapse monthly presence into contiguous membership intervals.

    `end_date` is None while a security is still in the newest snapshot;
    `end_is_open` marks that explicitly so a NULL is never mistaken for missing data.
    """
    snaps = sorted(snapshots["snapshot_date"].unique())
    pos = {d: i for i, d in enumerate(snaps)}
    last_idx = len(snaps) - 1

    rows = []
    for ticker, grp in snapshots.groupby("ticker", sort=True):
        idxs = sorted(pos[d] for d in grp["snapshot_date"].unique())
        run_start = prev = idxs[0]
        for i in idxs[1:]:
            if i == prev + 1:
                prev = i
                continue
            rows.append((ticker, snaps[run_start], snaps[prev], prev == last_idx))
            run_start = prev = i
        rows.append((ticker, snaps[run_start], snaps[prev], prev == last_idx))

    out = pd.DataFrame(rows, columns=["ticker", "start_date", "end_date", "end_is_open"])
    out["end_date"] = out["end_date"].astype("object")
    out.loc[out["end_is_open"], "end_date"] = None
    return out.sort_values(["ticker", "start_date"]).reset_index(drop=True)


# ----------------------------------------------------------------------- runner

def run(force: bool = False, start: str | None = None, limit: int | None = None) -> IngestResult:
    res = IngestResult(source=SOURCE, dataset=DATASET)
    start = start or get_settings().history_start

    log.info("enumerating revision history of '%s'...", PAGE_TITLE)
    revs = enumerate_revisions(force=force)
    log.info("  %d revisions, %s .. %s", len(revs),
             revs["timestamp"].min().date(), revs["timestamp"].max().date())

    picked = select_month_end_revisions(revs, start)
    if limit:
        picked = picked.tail(limit).reset_index(drop=True)
    log.info("selected %d month-end revisions from %s", len(picked), start)

    selection = [{"snapshot_date": r.snapshot_date, "revid": int(r.revid),
                  "rev_timestamp": r.timestamp.isoformat()} for r in picked.itertuples()]
    write_bronze(source=SOURCE, dataset=DATASET, filename="selected_revisions.json",
                 content=json.dumps(selection, indent=2).encode(),
                 url=f"{API} (revision selection)")
    res.bronze_files += 1

    contents = fetch_revision_contents(picked["revid"].astype(int).tolist())

    records, bad = [], []
    for r in picked.itertuples():
        text = contents.get(int(r.revid))
        if not text:
            bad.append({"snapshot_date": r.snapshot_date, "revid": int(r.revid),
                        "issue": "content missing"})
            continue
        tickers = parse_constituents(text)
        if not (MIN_PLAUSIBLE <= len(tickers) <= MAX_PLAUSIBLE):
            bad.append({"snapshot_date": r.snapshot_date, "revid": int(r.revid),
                        "issue": f"implausible count {len(tickers)}"})
            continue
        for t in tickers:
            records.append({"snapshot_date": r.snapshot_date, "revid": int(r.revid),
                            "rev_timestamp": r.timestamp.isoformat(), "ticker": t})

    snaps = pd.DataFrame(records)
    if snaps.empty:
        res.errors.append("no snapshots parsed")
        return res

    # Map tickers to stable IDs, preferring a CIK match from the SEC spine.
    reg = SecurityRegistry.load()
    tick2cik = (reg.df[reg.df["cik"] != 0]
                .drop_duplicates(subset=["ticker"])
                .set_index("ticker")["cik"].to_dict())
    uniq = pd.DataFrame({"ticker": sorted(snaps["ticker"].unique())})
    uniq["cik"] = uniq["ticker"].map(tick2cik).fillna(0).astype("int64")
    uniq["name"] = ""
    uniq["exchange"] = ""
    uniq["security_id"] = reg.bulk_assign(uniq, seen_date=today_iso())
    reg.save()
    snaps = snaps.merge(uniq[["ticker", "security_id"]], on="ticker", how="left")

    write_silver(snaps, "reference/sp500_membership_snapshots")

    intervals = build_intervals(snaps).merge(
        uniq[["ticker", "security_id"]], on="ticker", how="left")
    intervals["source"] = "wikipedia_revision_history"
    write_silver(intervals[["security_id", "ticker", "start_date", "end_date",
                            "end_is_open", "source"]],
                 "reference/sp500_membership_intervals")

    if bad:
        write_bronze(source=SOURCE, dataset=DATASET, filename="parse_failures.json",
                     content=json.dumps(bad, indent=2).encode(), url="(quality report)")
        res.bronze_files += 1
        res.errors = [f"{b['snapshot_date']}: {b['issue']}" for b in bad]

    ever = snaps["ticker"].nunique()
    current = set(snaps.loc[snaps["snapshot_date"] == snaps["snapshot_date"].max(), "ticker"])
    res.rows = len(snaps)
    res.notes = {
        "snapshots": int(snaps["snapshot_date"].nunique()),
        "date_range": f"{snaps['snapshot_date'].min()} .. {snaps['snapshot_date'].max()}",
        "unique_tickers_ever": int(ever),
        "current_members": len(current),
        "delisted_or_removed": int(ever - len(current)),
        "median_per_snapshot": int(snaps.groupby("snapshot_date").size().median()),
        "parse_failures": len(bad),
    }
    return res
