"""Macro series from FRED (Federal Reserve Bank of St. Louis).

Free and keyless: the `fredgraph.csv` endpoint returns a full series as CSV without
authentication, which keeps this source inside the $0 budget. Setting FRED_API_KEY
in .env unlocks the richer JSON API (vintages, ALFRED point-in-time revisions) but
nothing here requires it.

A revision caveat worth knowing before these feed a model
--------------------------------------------------------
Most macro series are REVISED after first publication. CPI, GDP and payrolls are
restated for months afterwards, so the value FRED shows today for March 2020 is not
what was on the screen in April 2020. Using today's values in a backtest is a
genuine look-ahead leak.

Market-based series (Treasury yields, VIX, spreads, the dollar index) are NOT
revised - a closing yield is final - so they are safe to use as-is. Each series here
is tagged `revised` accordingly. Treat revised=True series as indicative only until
they are re-pulled from ALFRED with proper vintage dates.
"""

from __future__ import annotations

import io
import logging

import pandas as pd

from ..http_cache import fetch
from ..storage import today_iso, write_bronze, write_silver
from .base import IngestResult

log = logging.getLogger(__name__)

SOURCE = "fred"
DATASET = "macro"
URL_TMPL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

#: series_id -> (description, revised)
SERIES = {
    # --- market-based: final on publication, safe for point-in-time use ---
    "DGS10":        ("10-Year Treasury constant maturity yield", False),
    "DGS2":         ("2-Year Treasury constant maturity yield", False),
    "DGS3MO":       ("3-Month Treasury constant maturity yield", False),
    "T10Y2Y":       ("10Y minus 2Y term spread", False),
    "T10Y3M":       ("10Y minus 3M term spread", False),
    "DFF":          ("Effective federal funds rate", False),
    "VIXCLS":       ("CBOE Volatility Index close", False),
    # NOTE: the two ICE BofA spreads return only ~3 years via the keyless CSV
    # endpoint - they are licensed third-party data and FRED caps bulk download.
    # Flagged by quality.checks.check_macro_history_depth. Do not use them for
    # pre-2023 regime tagging; a full history needs a FRED API key or another source.
    "BAMLH0A0HYM2": ("ICE BofA US High Yield option-adjusted spread", False),
    "BAMLC0A0CM":   ("ICE BofA US Corporate option-adjusted spread", False),
    "DTWEXBGS":     ("Trade-weighted US dollar index, broad goods & services", False),
    "DCOILWTICO":   ("WTI crude oil spot price", False),
    # --- revised after first release: look-ahead risk, see module docstring ---
    "CPIAUCSL":     ("CPI for all urban consumers, seasonally adjusted", True),
    "UNRATE":       ("Civilian unemployment rate", True),
    "INDPRO":       ("Industrial production index", True),
    "PAYEMS":       ("Total nonfarm payrolls", True),
    "UMCSENT":      ("University of Michigan consumer sentiment", True),
    "GDPC1":        ("Real gross domestic product", True),
    "USREC":        ("NBER recession indicator", True),
}


def parse_fred_csv(text: str, series: str, description: str, revised: bool) -> pd.DataFrame:
    """One `fredgraph.csv` payload -> the silver row shape.

    FRED marks a missing observation with a literal '.', which pandas would otherwise
    read as text and turn the whole column into strings. A row whose date will not
    parse is dropped; a row whose value will not parse keeps its date with a NaN value,
    because "no print that day" is information (holiday, suspended series) and a
    consumer that forward-fills needs the row to be there.
    """
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2:
        raise ValueError(f"{series}: expected a two-column CSV, got {list(df.columns)}")
    date_col, val_col = df.columns[0], df.columns[1]
    df = df.rename(columns={date_col: "date", val_col: "value"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
    df = df.dropna(subset=["date"])
    df["series_id"] = series
    df["description"] = description
    df["revised"] = revised
    return df[["series_id", "date", "value", "description", "revised"]].reset_index(drop=True)


def run(force: bool = False) -> IngestResult:
    res = IngestResult(source=SOURCE, dataset=DATASET)
    ingest_date = today_iso()
    frames: list[pd.DataFrame] = []

    for series, (desc, revised) in SERIES.items():
        url = URL_TMPL.format(series=series)
        try:
            resp = fetch(url, source=SOURCE, ttl_seconds=12 * 3600, force=force)
        except Exception as exc:  # noqa: BLE001
            res.errors.append(f"{series}: {exc}"[:160])
            continue

        res.fetched += 0 if resp.from_cache else 1
        res.from_cache += 1 if resp.from_cache else 0

        write_bronze(source=SOURCE, dataset=DATASET, filename=f"{series}.csv",
                     content=resp.content, url=url, ingest_date=ingest_date,
                     extra={"description": desc, "revised": revised})
        res.bronze_files += 1

        try:
            frames.append(parse_fred_csv(resp.text(), series, desc, revised))
        except Exception as exc:  # noqa: BLE001
            res.errors.append(f"{series}: parse: {exc}"[:160])

    if not frames:
        res.errors.append("no series downloaded")
        return res

    out = pd.concat(frames, ignore_index=True).sort_values(["series_id", "date"])
    write_silver(out.reset_index(drop=True), "macro/fred_series")

    res.rows = len(out)
    res.notes = {
        "series": int(out["series_id"].nunique()),
        "revised_series": sum(1 for _, r in SERIES.values() if r),
        "date_range": f"{out['date'].min()} .. {out['date'].max()}",
        "observations_per_series": int(out.groupby("series_id").size().median()),
    }
    return res
