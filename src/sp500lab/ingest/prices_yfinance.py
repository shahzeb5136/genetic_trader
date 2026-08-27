"""Daily OHLCV + corporate actions from Yahoo Finance (free tier, Phase 1/2).

Role in the plan
----------------
This is the *placeholder* price source. It is free, it covers current constituents
well, and it is good enough to build and debug the whole pipeline against. It is
NOT the long-term source, because Yahoo drops most delisted tickers - exactly the
names that survivorship-free backtesting needs. Phase 3 replaces this with a paid
EOD feed and re-runs the same backtests; the difference between the two runs is the
project's own measurement of survivorship bias.

Three rules this module follows
-------------------------------
1. **Raw prices, never vendor-adjusted only.** We request auto_adjust=False so the
   OHLC columns are as-traded, and capture Dividends / Stock Splits as separate
   event rows. Adjustment factors are computed by us (normalize/adjustments.py) and
   applied at query time. Vendor-adjusted series get silently rewritten on every
   dividend, which makes a backtest irreproducible - the same code returns different
   numbers next month with no change on your side.
2. **Bronze before parse.** Each chunk of tickers is written to bronze as returned,
   so a parser change never costs a re-download.
3. **Checkpointed and idempotent.** Work is chunked; a completed chunk is skipped on
   re-run unless --force. A crash at chunk 18 of 23 resumes at 18.

Note: yfinance manages its own HTTP session, so these calls bypass http_cache. The
fetch-once guarantee is preserved at the *artifact* level instead - if a chunk's
bronze file exists for today, it is not re-fetched.
"""

from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd

from ..config import get_settings
from ..paths import bronze_path
from ..registry import SecurityRegistry
from ..storage import read_silver, silver_exists, today_iso, write_bronze, write_silver
from .base import IngestResult

log = logging.getLogger(__name__)

SOURCE = "yfinance"
DATASET = "daily_bars"
CHUNK_SIZE = 40          # tickers per yfinance batch request
OHLC_COLS = ["open", "high", "low", "close", "volume"]


def resolve_universe(mode: str = "ever") -> pd.DataFrame:
    """Tickers to download.

    mode="ever"    every ticker that appears in any membership snapshot
                   (survivorship-free to the limit of our constituent history)
    mode="current" today's constituents only (deliberately biased; useful for
                   measuring how much the bias is worth)
    """
    if mode == "ever" and silver_exists("reference/sp500_membership_intervals"):
        iv = read_silver("reference/sp500_membership_intervals")
        return (iv[["security_id", "ticker"]]
                .drop_duplicates(subset=["ticker"])
                .sort_values("ticker").reset_index(drop=True))

    cur = read_silver("reference/sp500_current")
    return (cur[["security_id", "ticker"]]
            .drop_duplicates(subset=["ticker"])
            .sort_values("ticker").reset_index(drop=True))


def _to_long(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Reshape a yfinance batch frame into a tidy long table.

    yfinance returns a column MultiIndex (field, ticker) for multi-ticker requests
    and a flat frame for a single ticker. Normalise both to one shape.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        df = raw.stack(level=-1, future_stack=True).reset_index()
        df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "ticker"})
    else:
        df = raw.reset_index().rename(columns={raw.index.name or "index": "date"})
        df["ticker"] = tickers[0]

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"stock_splits": "split_ratio", "dividends": "dividend",
                            "adj_close": "adj_close_vendor"})

    keep = ["date", "ticker"] + [c for c in
            OHLC_COLS + ["adj_close_vendor", "dividend", "split_ratio"] if c in df.columns]
    df = df[keep]
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df[df["date"].notna()]
    # Drop all-NaN price rows (yfinance pads the calendar across a batch)
    price_cols = [c for c in OHLC_COLS if c in df.columns]
    df = df.dropna(subset=price_cols, how="all")
    return df.reset_index(drop=True)


def run(
    force: bool = False,
    universe: str = "ever",
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
) -> IngestResult:
    import yfinance as yf

    res = IngestResult(source=SOURCE, dataset=DATASET)
    start = start or get_settings().price_start
    ingest_date = today_iso()

    uni = resolve_universe(universe)
    if limit:
        uni = uni.head(limit)
    tickers = uni["ticker"].tolist()
    log.info("universe=%s  %d tickers  start=%s", universe, len(tickers), start)

    # yfinance wants '-' for share classes (BRK-B); our canonical form uses '.'.
    yf_symbol = {t: t.replace(".", "-") for t in tickers}
    inverse = {v: k for k, v in yf_symbol.items()}

    chunks = [tickers[i:i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]
    frames: list[pd.DataFrame] = []
    empty_tickers: list[str] = []

    for ci, chunk in enumerate(chunks, 1):
        # The filename encodes the REQUEST, not just the chunk index. Keying on the
        # index alone is unsafe: a smaller test run (different --limit/--start) writes
        # bars_chunk_001 for the same ingest_date, and a later full run then reuses
        # that file - silently dropping the tickers the test never requested and
        # truncating the history of the ones it did. That happened during development
        # and cost 32 securities before the membership-clipping check exposed it.
        req_key = hashlib.sha256(
            f"{start}|{end}|{','.join(chunk)}".encode()).hexdigest()[:10]
        fname = f"bars_chunk_{ci:03d}_{req_key}.parquet"
        bpath = bronze_path(SOURCE, DATASET, ingest_date, fname)

        if bpath.exists() and not force:
            cached = pd.read_parquet(bpath)
            # Defence in depth: the hash should guarantee this, but a mismatch here
            # means a stale or hand-edited artifact, and silently trusting it is how
            # the original bug survived.
            if set(cached["ticker"].unique()) - set(chunk):
                log.warning("chunk %d cache contents unexpected - re-fetching", ci)
            else:
                log.info("chunk %d/%d cached, skipping", ci, len(chunks))
                frames.append(cached)
                res.from_cache += 1
                continue

        syms = [yf_symbol[t] for t in chunk]
        try:
            raw = yf.download(syms, start=start, end=end, auto_adjust=False,
                              actions=True, progress=False, group_by="column",
                              threads=True)
        except Exception as exc:  # noqa: BLE001
            msg = f"chunk {ci} download failed: {exc}"
            log.error(msg)
            res.errors.append(msg)
            continue

        df = _to_long(raw, syms)
        if df.empty:
            res.errors.append(f"chunk {ci} returned no rows")
            continue

        df["ticker"] = df["ticker"].map(lambda s: inverse.get(s, s))
        got = set(df["ticker"].unique())
        empty_tickers.extend(t for t in chunk if t not in got)

        write_bronze(source=SOURCE, dataset=DATASET, filename=fname,
                     content=df.to_parquet(index=False),
                     url=f"yfinance.download({len(syms)} symbols, start={start})",
                     ingest_date=ingest_date,
                     extra={"tickers": chunk, "rows": len(df)})
        frames.append(df)
        res.bronze_files += 1
        res.fetched += 1
        log.info("chunk %d/%d  %d symbols -> %d rows", ci, len(chunks), len(syms), len(df))

    if not frames:
        res.errors.append("no data downloaded")
        return res

    allbars = pd.concat(frames, ignore_index=True)

    # ---- split into the two tables that must never be conflated -------------
    reg = SecurityRegistry.load()
    tick2sid = dict(zip(uni["ticker"], uni["security_id"]))
    allbars["security_id"] = allbars["ticker"].map(tick2sid)
    missing_sid = allbars["security_id"].isna()
    if missing_sid.any():
        for t in sorted(allbars.loc[missing_sid, "ticker"].unique()):
            tick2sid[t] = reg.resolve_or_assign(None, t, seen_date=ingest_date)
        reg.save()
        allbars["security_id"] = allbars["ticker"].map(tick2sid)

    allbars["date"] = allbars["date"].dt.strftime("%Y-%m-%d")
    allbars = (allbars.sort_values(["ticker", "date"])
               .drop_duplicates(subset=["ticker", "date"], keep="last"))

    bar_cols = ["security_id", "ticker", "date"] + \
               [c for c in OHLC_COLS if c in allbars.columns] + \
               [c for c in ["adj_close_vendor"] if c in allbars.columns]
    bars = allbars[bar_cols].copy()
    bars["source"] = SOURCE
    bars["ingest_date"] = ingest_date
    write_silver(bars, "market/daily_bars")

    # Corporate actions as discrete events, kept apart from prices.
    actions: list[pd.DataFrame] = []
    if "dividend" in allbars.columns:
        d = allbars.loc[allbars["dividend"].fillna(0) > 0,
                        ["security_id", "ticker", "date", "dividend"]].copy()
        d = d.rename(columns={"dividend": "value"})
        d["action_type"] = "dividend"
        actions.append(d)
    if "split_ratio" in allbars.columns:
        s = allbars.loc[(allbars["split_ratio"].fillna(0) > 0)
                        & (allbars["split_ratio"] != 1.0),
                        ["security_id", "ticker", "date", "split_ratio"]].copy()
        s = s.rename(columns={"split_ratio": "value"})
        s["action_type"] = "split"
        actions.append(s)

    if actions:
        ca = pd.concat(actions, ignore_index=True)
        ca["source"] = SOURCE
        ca = ca[["security_id", "ticker", "date", "action_type", "value", "source"]]
        ca = ca.sort_values(["ticker", "date", "action_type"]).reset_index(drop=True)
        write_silver(ca, "market/corporate_actions")
        res.notes["corporate_actions"] = len(ca)
        res.notes["dividends"] = int((ca["action_type"] == "dividend").sum())
        res.notes["splits"] = int((ca["action_type"] == "split").sum())

    if empty_tickers:
        write_bronze(source=SOURCE, dataset=DATASET, filename="no_data_tickers.json",
                     content=json.dumps(sorted(set(empty_tickers)), indent=2).encode(),
                     url="(coverage report)", ingest_date=ingest_date)
        res.bronze_files += 1

    res.rows = len(bars)
    res.notes.update({
        "tickers_requested": len(tickers),
        "tickers_with_data": int(bars["ticker"].nunique()),
        "tickers_no_data": len(set(empty_tickers)),
        "date_range": f"{bars['date'].min()} .. {bars['date'].max()}",
        "coverage_pct": round(100 * bars["ticker"].nunique() / max(len(tickers), 1), 1),
    })
    return res
