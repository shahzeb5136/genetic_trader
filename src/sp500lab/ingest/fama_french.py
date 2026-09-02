"""Daily factor returns from the Kenneth French data library. Free, keyless, since 1963.

Why these belong in the data layer
----------------------------------
Every strategy here is scored against SPY. That answers "did it beat the index" and
nothing else. The five Fama-French factors plus momentum answer the question that
comes next: *was it a factor bet in disguise?* A "low volatility" strategy that loads
0.9 on RMW and CMA has rediscovered quality, and a regression of its monthly returns on
these columns says so in one table. For a daily-rebalanced idea they are also the
cleanest available building block for factor-neutral construction: hedge out Mkt-RF
and SMB and what is left is the stock selection.

They are also a cross-source check on the price data. `Mkt-RF + RF` is the daily total
return of the whole CRSP market. SPY's daily total return tracks it at a correlation
above 0.98, and `quality.checks.check_ff_market_vs_spy` asserts exactly that - which
catches a percent/decimal mistake in THIS parser as surely as a shifted SPY series.

What is stored
--------------
One row per session, WIDE, in DECIMALS (the source publishes percent):

    date, mkt_rf, smb, hml, rmw, cma, rf, mom

`rf` is the one-month T-bill return for the day. `mom` comes from a separate file with
a longer history; sessions before the five-factor series begins (1963-07-01) carry NaN
for the five and a value for momentum. The library republishes monthly, usually with
small revisions to the most recent months as CRSP finalises delistings, so a 7-day TTL
is a compromise between freshness and the cache.

Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
The files are zipped CSVs with a free-text preamble and a copyright trailer; the
parser looks for the header row rather than trusting a line count.
"""

from __future__ import annotations

import io
import logging
import zipfile

import pandas as pd

from ..http_cache import fetch
from ..storage import today_iso, write_bronze, write_silver
from .base import IngestResult

log = logging.getLogger(__name__)

SOURCE = "fama_french"
DATASET = "factors_daily"
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

#: (zip name, columns the file carries, in the order the header names them)
FILES = {
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip":
        ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"),
    "F-F_Momentum_Factor_daily_CSV.zip": ("Mom",),
}
#: Source header -> silver column.
COLUMNS = {"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RMW": "rmw", "CMA": "cma",
           "RF": "rf", "Mom": "mom"}
TTL_SECONDS = 7 * 86_400


def parse_factor_csv(text: str, expect: tuple[str, ...]) -> pd.DataFrame:
    """The CSV inside one library zip -> a wide frame of DECIMAL daily returns.

    The preamble is free text of unknown length and the trailer is a copyright line, so
    the data block is found by its header: the first line whose fields, after the empty
    date field, are exactly the expected factor names (whitespace-insensitive). Rows
    are read from there until the first line that does not begin with an 8-digit date.
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        fields = [f.strip() for f in line.split(",")]
        if len(fields) == len(expect) + 1 and tuple(fields[1:]) == tuple(expect):
            start = i + 1
            break
    if start is None:
        raise ValueError(f"no header row with columns {expect} in the payload")

    rows = []
    for line in lines[start:]:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != len(expect) + 1 or not (fields[0].isdigit() and len(fields[0]) == 8):
            break
        rows.append(fields)
    if not rows:
        raise ValueError("header found but no data rows followed it")

    df = pd.DataFrame(rows, columns=("yyyymmdd",) + tuple(expect))
    out = pd.DataFrame({"date": pd.to_datetime(df["yyyymmdd"], format="%Y%m%d")
                        .dt.strftime("%Y-%m-%d")})
    for src in expect:
        # percent -> decimal. -99.99 is the library's missing marker.
        v = pd.to_numeric(df[src], errors="coerce")
        out[COLUMNS[src]] = v.where(v > -99.0) / 100.0
    return out


def _csv_from_zip(content: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"no CSV inside the zip ({z.namelist()})")
        return z.read(names[0]).decode("utf-8", errors="replace")


def run(force: bool = False) -> IngestResult:
    res = IngestResult(source=SOURCE, dataset=DATASET)
    ingest_date = today_iso()
    frames: list[pd.DataFrame] = []

    for fname, expect in FILES.items():
        url = BASE + fname
        try:
            resp = fetch(url, source=SOURCE, ttl_seconds=TTL_SECONDS, force=force)
        except Exception as exc:  # noqa: BLE001
            res.errors.append(f"{fname}: {exc}"[:160])
            continue
        res.fetched += 0 if resp.from_cache else 1
        res.from_cache += 1 if resp.from_cache else 0

        write_bronze(source=SOURCE, dataset=DATASET, filename=fname, content=resp.content,
                     url=url, ingest_date=ingest_date, extra={"columns": list(expect)})
        res.bronze_files += 1
        try:
            frames.append(parse_factor_csv(_csv_from_zip(resp.content), expect))
        except Exception as exc:  # noqa: BLE001
            res.errors.append(f"{fname}: parse: {exc}"[:160])

    if not frames:
        res.errors.append("no factor files parsed")
        return res

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)
    cols = ["date"] + [c for c in COLUMNS.values() if c in out.columns]
    out = out[cols]
    out["source"] = SOURCE
    write_silver(out, "factors/fama_french_daily")

    res.rows = len(out)
    res.notes = {
        "columns": cols[1:],
        "date_range": f"{out['date'].min()} .. {out['date'].max()}",
        "five_factor_from": str(out.loc[out["mkt_rf"].notna(), "date"].min())
        if "mkt_rf" in out else None,
        "annualised_mkt_rf_pct": round(float(out["mkt_rf"].mean() * 252 * 100), 2)
        if "mkt_rf" in out else None,
    }
    return res
