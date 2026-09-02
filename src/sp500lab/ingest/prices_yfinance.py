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

# --------------------------------------------------------------------------
# The integrity gate (ADR-043)
#
# On 2026-09-02 a routine refresh returned 34 delisted names Yahoo had never served
# before, and a third of them were garbage: Countrywide with 895 impossible bars,
# Titanium Metals at a $10,100 close and a +721,000% day, RadioShack missing half its
# sessions. It also returned nothing at all for Agilent, a live constituent, which
# silently vanished from silver. Everything downstream rebuilt without complaint and
# the equal-weight CAGR came out at 91%/yr. `doctor` caught it; this is what stops it
# reaching silver at all.
# --------------------------------------------------------------------------

#: Impossible OHLC rows (high < low, low > open, ...) tolerated in a LIVE member's series.
#: The four reviewed vendor print errors (ADR-040) are one per ticker, and a current
#: constituent with a couple of bad prints must still enter silver - rejecting it would
#: drop a live name, and on a first pull there is nothing to carry forward. Its bad rows
#: then show up as ERRORs in `quality` until a human reviews them into KNOWN_BAD_BARS,
#: which is the intended workflow. A DELISTED series gets no such tolerance: it was not
#: in the panel before, so rejecting it loses nothing, and the corrupt shells the gate
#: exists for arrive with exactly "a few" impossible rows as often as with hundreds.
MAX_BAD_OHLC_ROWS = 5
#: Largest one-day |close/close - 1| a series may show INSIDE its index-membership window,
#: on a day with no recorded split. Nothing in the S&P 500 has ever moved 400% in a
#: session; a post-delisting shell trading at pennies routinely does, which is why the
#: window matters - the panel clips to membership and never sees the shell.
MAX_ABS_DAILY_RETURN = 4.0
#: Share of sessions inside a series' own first..last span that must have a bar. A
#: six-bar stub or a series missing half its history is not a history. Live members are
#: exempt: a hole in a current constituent is reported by `quality`, never dropped -
#: dropping one is the survivorship bias this whole project exists to prevent.
MIN_SPAN_COVERAGE = 0.5


def series_integrity(bars: pd.DataFrame, intervals: pd.DataFrame | None,
                     calendar: pd.DataFrame | None) -> pd.DataFrame:
    """One row per ticker: what is wrong with its series, and whether it is rejected.

    Columns: ticker, bars, bad_ohlc, max_abs_ret, coverage, live, reject, reasons.
    Pure: reads nothing from disk, so it can be dry-run on any pull in bronze.
    """
    b = bars.sort_values(["ticker", "date"])
    lo = hi = live = None
    if intervals is not None and len(intervals):
        is_open = intervals["end_is_open"].astype(bool)
        span = intervals.assign(end=intervals["end_date"].where(~is_open, "9999-12-31"))
        span = span.groupby("ticker").agg(lo=("start_date", "min"), hi=("end", "max"))
        lo, hi = span["lo"], span["hi"]
        live = intervals.loc[is_open, "ticker"].unique()
    live = set(live) if live is not None else set()
    sessions = (pd.Index(calendar["date"].astype(str)) if calendar is not None
                and len(calendar) else None)
    has_split = "split_ratio" in b.columns

    # Reviewed vendor print errors (ADR-040) are the one allowlist the gate and the
    # quality battery share: a row a human has looked at and left in place must not
    # count against the series that carries it, or a delisted name with one harmless
    # bad print loses its whole index-era history to a rule meant for corrupt shells.
    from ..quality.checks import KNOWN_BAD_BARS

    rows = []
    for t, g in b.groupby("ticker", sort=False):
        impossible = ((g["high"] < g["low"]) | (g["high"] < g["open"]) | (g["high"] < g["close"])
                      | (g["low"] > g["open"]) | (g["low"] > g["close"]) | (g["close"] <= 0))
        reviewed = pd.Series([(t, d) in KNOWN_BAD_BARS for d in g["date"]], index=g.index)
        bad = int((impossible & ~reviewed).sum())
        inside = g
        if lo is not None and t in lo.index:
            inside = g[(g["date"] >= lo[t]) & (g["date"] <= hi[t])]
        ret = inside["close"].pct_change().abs()
        if has_split:
            split_day = inside["split_ratio"].fillna(0).gt(0) & inside["split_ratio"].ne(1.0)
            ret = ret[~split_day.to_numpy()]
        max_ret = float(ret.max()) if len(ret) and ret.notna().any() else 0.0
        if sessions is not None and len(g):
            expected = len(sessions[(sessions >= g["date"].min()) & (sessions <= g["date"].max())])
            coverage = len(g) / max(expected, 1)
        else:
            coverage = 1.0
        reasons = []
        if bad > (MAX_BAD_OHLC_ROWS if t in live else 0):
            reasons.append(f"{bad} impossible OHLC rows")
        if max_ret > MAX_ABS_DAILY_RETURN:
            reasons.append(f"a {max_ret:.0%} day inside membership")
        if coverage < MIN_SPAN_COVERAGE and t not in live:
            reasons.append(f"{coverage:.0%} of sessions in its own span")
        rows.append({"ticker": t, "bars": len(g), "bad_ohlc": bad,
                     "max_abs_ret": round(max_ret, 3), "coverage": round(coverage, 3),
                     "live": t in live, "reject": bool(reasons),
                     "reasons": "; ".join(reasons)})
    return pd.DataFrame(rows)


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
    ingest_date: str | None = None,
) -> IngestResult:
    """Fetch (or replay) every chunk, gate each series, and write silver.

    `ingest_date` names which bronze partition to build from. Default is today, which
    fetches anything not yet on disk. Passing an earlier date re-parses THAT pull with
    zero network - the way to roll silver back to a validated state after a refresh
    turns out to be bad, and the reason bronze is partitioned by fetch date at all.
    """
    import yfinance as yf

    res = IngestResult(source=SOURCE, dataset=DATASET)
    start = start or get_settings().price_start
    replay = ingest_date is not None
    ingest_date = ingest_date or today_iso()
    if replay:
        log.info("re-parsing the %s pull from bronze; nothing will be fetched", ingest_date)

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

    if replay:
        # Everything that pull wrote, whatever the request keys were. The universe may
        # have changed since, so the chunk boundaries and filenames need not match;
        # the partition IS the pull.
        partition = bronze_path(SOURCE, DATASET, ingest_date, "x").parent
        files = sorted(partition.glob("bars_chunk_*.parquet"))
        if not files:
            res.errors.append(f"no price chunks in bronze for {ingest_date} ({partition})")
            return res
        frames = [pd.read_parquet(p) for p in files]
        res.from_cache = len(files)
        log.info("replaying %d chunk(s) from %s", len(files), partition)
        chunks = []                       # nothing to fetch

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
        if replay:
            res.errors.append(f"chunk {ci}: no artifact for {ingest_date} - not fetching "
                              "during a replay")
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

    # ---- the integrity gate: a corrupt series never reaches silver ------------
    intervals = (read_silver("reference/sp500_membership_intervals")
                 if silver_exists("reference/sp500_membership_intervals") else None)
    calendar = (read_silver("reference/trading_calendar")
                if silver_exists("reference/trading_calendar") else None)
    verdict = series_integrity(allbars, intervals, calendar)
    rejected = verdict[verdict["reject"]]
    if len(rejected):
        for r in rejected.itertuples():
            log.warning("REJECTED %s: %s", r.ticker, r.reasons)
        write_bronze(source=SOURCE, dataset=DATASET, filename="rejected_tickers.json",
                     content=rejected.to_json(orient="records", indent=2).encode(),
                     url="(integrity gate, ADR-043)", ingest_date=ingest_date)
        res.bronze_files += 1
    good = allbars[~allbars["ticker"].isin(set(rejected["ticker"]))]

    bar_cols = ["security_id", "ticker", "date"] + \
               [c for c in OHLC_COLS if c in good.columns] + \
               [c for c in ["adj_close_vendor"] if c in good.columns]
    bars = good[bar_cols].copy()
    bars["source"] = SOURCE
    bars["ingest_date"] = ingest_date

    # Corporate actions as discrete events, kept apart from prices.
    actions: list[pd.DataFrame] = []
    if "dividend" in good.columns:
        d = good.loc[good["dividend"].fillna(0) > 0,
                     ["security_id", "ticker", "date", "dividend"]].copy()
        d = d.rename(columns={"dividend": "value"})
        d["action_type"] = "dividend"
        actions.append(d)
    if "split_ratio" in good.columns:
        s = good.loc[(good["split_ratio"].fillna(0) > 0) & (good["split_ratio"] != 1.0),
                     ["security_id", "ticker", "date", "split_ratio"]].copy()
        s = s.rename(columns={"split_ratio": "value"})
        s["action_type"] = "split"
        actions.append(s)
    ca = pd.concat(actions, ignore_index=True) if actions else pd.DataFrame(
        columns=["security_id", "ticker", "date", "action_type", "value"])
    ca["source"] = SOURCE

    # ---- carry-forward: a vendor hiccup never deletes a validated series -------
    # A ticker that was in silver last time and is absent (or rejected) this time keeps
    # its previous rows. Losing a live constituent to one bad response is survivorship
    # bias arriving through the back door; the rows it had already passed every check.
    # A replay is a rollback: the caller has decided the current silver is NOT to be
    # trusted, so nothing is carried forward from it. Silver becomes that pull alone.
    carried: list[str] = []
    if not replay and silver_exists("market/daily_bars"):
        prev = read_silver("market/daily_bars")
        wanted = set(tickers)
        have = set(bars["ticker"].unique())
        carry = sorted((set(prev["ticker"].unique()) & wanted) - have)
        if carry:
            bars = pd.concat([bars, prev[prev["ticker"].isin(carry)][bars.columns]],
                             ignore_index=True)
            if silver_exists("market/corporate_actions"):
                prev_ca = read_silver("market/corporate_actions")
                ca = pd.concat([ca, prev_ca[prev_ca["ticker"].isin(carry)][ca.columns]],
                               ignore_index=True)
            carried = carry
            log.warning("carried forward %d ticker(s) from the previous silver: %s",
                        len(carry), ", ".join(carry[:12]) + (" ..." if len(carry) > 12 else ""))
            write_bronze(source=SOURCE, dataset=DATASET,
                         filename="carried_forward_tickers.json",
                         content=json.dumps(carry, indent=2).encode(),
                         url="(carry-forward, ADR-043)", ingest_date=ingest_date)
            res.bronze_files += 1

    bars = (bars.sort_values(["ticker", "date"])
            .drop_duplicates(subset=["ticker", "date"], keep="first").reset_index(drop=True))
    write_silver(bars, "market/daily_bars")

    ca = ca[["security_id", "ticker", "date", "action_type", "value", "source"]]
    ca = (ca.sort_values(["ticker", "date", "action_type"])
          .drop_duplicates(subset=["security_id", "date", "action_type"], keep="first")
          .reset_index(drop=True))
    write_silver(ca, "market/corporate_actions")
    res.notes["corporate_actions"] = len(ca)
    res.notes["dividends"] = int((ca["action_type"] == "dividend").sum())
    res.notes["splits"] = int((ca["action_type"] == "split").sum())
    res.notes["rejected_tickers"] = {r.ticker: r.reasons for r in rejected.itertuples()}
    res.notes["carried_forward_tickers"] = carried

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
