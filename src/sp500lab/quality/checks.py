"""Data quality checks over the silver layer.

Philosophy
----------
The failure mode that hurts is not a job that errors loudly - it is a job that
quietly keeps running while the data goes wrong underneath it. These checks exist to
convert silent corruption into a visible report.

Every check returns rows describing what is WRONG, plus a severity. Nothing here
mutates data: the point is to surface issues for a human decision, because "fix"
usually means understanding a corporate action, not clamping a number.

Severities
----------
ERROR   Structurally impossible. A high below a low, a duplicate primary key, a
        negative volume. These indicate a bug or a corrupt feed and block use.
WARN    Suspicious but legitimately possible. A 60% single-day move is real for a
        biotech on trial results and bogus for a utility - it needs eyes.
INFO    Coverage and completeness reporting, not a defect.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..storage import read_silver, silver_exists, write_silver

log = logging.getLogger(__name__)

ERROR, WARN, INFO = "ERROR", "WARN", "INFO"

# A one-day move beyond this without a corporate action is almost always a bad print
# or an unrecorded split.
EXTREME_MOVE = 0.50
# Consecutive identical closes suggesting a stale/halted feed.
STALE_RUN = 5


def _finding(check: str, severity: str, entity: str, detail: str, **extra) -> dict:
    return {"check": check, "severity": severity, "entity": entity,
            "detail": detail, **extra}


def check_bar_integrity(bars: pd.DataFrame) -> list[dict]:
    """OHLC relationships that must hold by definition."""
    out: list[dict] = []
    need = {"open", "high", "low", "close"}
    if not need.issubset(bars.columns):
        return out

    b = bars.dropna(subset=list(need))
    violations = {
        "high < low": b["high"] < b["low"],
        "high < open": b["high"] < b["open"],
        "high < close": b["high"] < b["close"],
        "low > open": b["low"] > b["open"],
        "low > close": b["low"] > b["close"],
        "close <= 0": b["close"] <= 0,
    }
    for label, mask in violations.items():
        n = int(mask.sum())
        if n:
            sample = b.loc[mask, ["ticker", "date"]].head(3)
            where = ", ".join(f"{r.ticker}@{r.date}" for r in sample.itertuples())
            out.append(_finding("bar_integrity", ERROR, "market/daily_bars",
                                f"{n} rows where {label}", rows=n, sample=where))

    if "volume" in bars.columns:
        n = int((bars["volume"] < 0).sum())
        if n:
            out.append(_finding("bar_integrity", ERROR, "market/daily_bars",
                                f"{n} rows with negative volume", rows=n))
    return out


def check_primary_key(bars: pd.DataFrame) -> list[dict]:
    """(security_id, date) must be unique - a duplicate silently double-counts."""
    dup = bars.duplicated(subset=["security_id", "date"], keep=False)
    n = int(dup.sum())
    if not n:
        return []
    sample = bars.loc[dup, ["ticker", "date"]].head(3)
    where = ", ".join(f"{r.ticker}@{r.date}" for r in sample.itertuples())
    return [_finding("primary_key", ERROR, "market/daily_bars",
                     f"{n} duplicate (security_id, date) rows", rows=n, sample=where)]


def check_extreme_moves(bars: pd.DataFrame, actions: pd.DataFrame) -> list[dict]:
    """Large one-day moves not explained by a recorded corporate action.

    An unrecorded split is the usual culprit and it is poison for a model: the
    price gap looks like a real return and momentum signals load straight onto it.
    """
    if "close" not in bars.columns:
        return []
    b = bars.sort_values(["security_id", "date"]).copy()
    b["ret"] = b.groupby("security_id")["close"].pct_change()
    flag = b["ret"].abs() > EXTREME_MOVE

    if actions is not None and len(actions):
        known = set(zip(actions["security_id"], actions["date"]))
        is_known = pd.Series(
            [(s, d) in known for s, d in zip(b["security_id"], b["date"])],
            index=b.index)
        flag &= ~is_known

    hits = b[flag]
    if hits.empty:
        return []
    worst = hits.reindex(hits["ret"].abs().sort_values(ascending=False).index).head(5)
    sample = ", ".join(f"{r.ticker}@{r.date} {r.ret:+.0%}" for r in worst.itertuples())
    return [_finding("extreme_move", WARN, "market/daily_bars",
                     f"{len(hits)} one-day moves >{EXTREME_MOVE:.0%} with no recorded "
                     f"corporate action", rows=len(hits), sample=sample)]


def check_stale_prices(bars: pd.DataFrame) -> list[dict]:
    """Runs of identical closes, which usually mean a halted or stale feed."""
    if "close" not in bars.columns:
        return []
    b = bars.sort_values(["security_id", "date"]).copy()
    same = b["close"].eq(b.groupby("security_id")["close"].shift(1))
    grp = (~same).cumsum()
    runs = b.assign(_g=grp).groupby(["security_id", "_g"]).size()
    long_runs = runs[runs > STALE_RUN]
    if long_runs.empty:
        return []
    return [_finding("stale_prices", WARN, "market/daily_bars",
                     f"{len(long_runs)} runs of >{STALE_RUN} consecutive identical "
                     f"closes (longest {int(long_runs.max())} bars)",
                     rows=int(len(long_runs)))]


def check_calendar_gaps(bars: pd.DataFrame, calendar: pd.DataFrame) -> list[dict]:
    """Missing bars on real trading days, within each security's own active span.

    Only gaps INSIDE a security's first..last observed date count. Absence before
    listing or after delisting is expected, not a defect - conflating the two would
    flag every delisted company as broken.
    """
    if calendar is None or calendar.empty:
        return [_finding("calendar_gaps", INFO, "market/daily_bars",
                         "no trading calendar available - run ingest benchmarks first")]

    sessions = pd.Index(calendar["date"].astype(str))
    span = bars.groupby("security_id")["date"].agg(["min", "max", "count"])
    findings: list[dict] = []
    worst: list[tuple[str, int]] = []

    tick = bars.drop_duplicates("security_id").set_index("security_id")["ticker"]
    for sid, row in span.iterrows():
        expected = sessions[(sessions >= row["min"]) & (sessions <= row["max"])]
        missing = len(expected) - int(row["count"])
        if missing > 0:
            worst.append((str(tick.get(sid, sid)), missing))

    if worst:
        worst.sort(key=lambda x: -x[1])
        total = sum(m for _, m in worst)
        sample = ", ".join(f"{t}:{m}" for t, m in worst[:5])
        findings.append(_finding(
            "calendar_gaps", WARN, "market/daily_bars",
            f"{len(worst)} securities missing {total} bars on trading days inside "
            f"their active span", rows=total, sample=sample))
    return findings


def check_ticker_recycling(bars: pd.DataFrame) -> list[dict]:
    """Detect price history that continues long after index membership ended.

    Tickers get REUSED. Compuware (CPWR) was an S&P 500 member until 2014; the symbol
    was later reassigned to an unrelated OTC shell trading 0-50 shares a day. A price
    feed keyed on the bare ticker happily returns the shell's prices stitched onto the
    index member's history, and the join looks perfectly clean - same ticker, no gap,
    no error. The result is a fabricated series that a model will treat as one company.

    The membership intervals are what make this detectable: if a security's index
    membership ended years ago but "its" price series is still updating today, the
    recent bars almost certainly belong to a different company.

    Mitigation for downstream users: clip each security's price history to its
    membership interval (plus indicator warmup) rather than trusting the full series.
    """
    if not silver_exists("reference/sp500_membership_intervals"):
        return []
    iv = read_silver("reference/sp500_membership_intervals")
    open_now = set(iv.loc[iv["end_is_open"].astype(bool), "ticker"])
    # A name can leave the index and later re-enter (Hilton, Hyatt). Such a ticker has
    # both a closed and an open interval, and its ongoing price history is entirely
    # expected - only tickers with NO current membership can be recycled.
    closed = iv[(~iv["end_is_open"].astype(bool))
                & (~iv["ticker"].isin(open_now))].dropna(subset=["end_date"])
    if closed.empty:
        return []

    last_bar = bars.groupby("ticker")["date"].max()
    ends = closed.groupby("ticker")["end_date"].max()

    joined = pd.DataFrame({"end_date": ends}).join(
        last_bar.rename("last_bar"), how="inner").dropna()
    if joined.empty:
        return []

    overrun_days = (pd.to_datetime(joined["last_bar"])
                    - pd.to_datetime(joined["end_date"])).dt.days
    suspect = joined[overrun_days > 365].assign(overrun=overrun_days[overrun_days > 365])
    if suspect.empty:
        return []

    suspect = suspect.sort_values("overrun", ascending=False)
    sample = ", ".join(f"{t}(+{int(r.overrun)}d)" for t, r in suspect.head(6).iterrows())
    return [_finding(
        "ticker_recycling", WARN, "market/daily_bars",
        f"{len(suspect)} tickers have price bars >1yr after index membership ended - "
        f"likely symbol reuse by an unrelated company; clip prices to membership "
        f"intervals before use", rows=len(suspect), sample=sample)]


def check_universe_coverage(bars: pd.DataFrame) -> list[dict]:
    """How much of the survivorship-free universe actually has price data."""
    if not silver_exists("reference/sp500_membership_intervals"):
        return []
    iv = read_silver("reference/sp500_membership_intervals")
    wanted = set(iv["ticker"].unique())
    have = set(bars["ticker"].unique())
    missing = sorted(wanted - have)
    pct = 100 * len(have & wanted) / max(len(wanted), 1)

    sev = INFO if pct >= 95 else WARN
    return [_finding("universe_coverage", sev, "market/daily_bars",
                     f"{len(have & wanted)}/{len(wanted)} ({pct:.1f}%) of ever-members "
                     f"have price data; {len(missing)} missing",
                     rows=len(missing),
                     sample=", ".join(missing[:12]))]


def check_membership_sanity(snapshots: pd.DataFrame) -> list[dict]:
    """Constituent counts per snapshot should sit near 500."""
    if snapshots is None or snapshots.empty:
        return []
    counts = snapshots.groupby("snapshot_date").size()
    odd = counts[(counts < 480) | (counts > 520)]
    if odd.empty:
        return []
    sample = ", ".join(f"{d}:{c}" for d, c in odd.head(5).items())
    return [_finding("membership_sanity", WARN, "reference/sp500_membership_snapshots",
                     f"{len(odd)} snapshots with constituent count outside 480-520",
                     rows=len(odd), sample=sample)]


def check_macro_history_depth(min_years: float = 5.0) -> list[dict]:
    """Flag macro series returning far less history than they should have.

    Every series in the FRED set has decades of history at source, so a short one
    means the endpoint truncated it rather than the data not existing. The ICE BofA
    option-adjusted spread series (BAMLH0A0HYM2, BAMLC0A0CM) return only ~3 years via
    the keyless `fredgraph.csv` endpoint - they are licensed third-party data and FRED
    restricts bulk historical download.

    This matters because credit spreads are a regime indicator: a backtest that tags
    2008 or 2020 by high-yield spread will silently get nothing for those periods and
    quietly mislabel the regime rather than erroring.
    """
    if not silver_exists("macro/fred_series"):
        return []
    m = read_silver("macro/fred_series")
    span = m.groupby("series_id")["date"].agg(["min", "max", "count"])
    years = ((pd.to_datetime(span["max"]) - pd.to_datetime(span["min"])).dt.days / 365.25)
    short = span[years < min_years].assign(years=years[years < min_years].round(1))
    if short.empty:
        return []
    sample = ", ".join(f"{s}({r.years}y from {r['min']})" for s, r in short.iterrows())
    return [_finding(
        "macro_history_depth", WARN, "macro/fred_series",
        f"{len(short)} macro series have <{min_years:g}y of history - the endpoint "
        f"truncated them (licensed data), not an empty source; do not use for "
        f"pre-2023 regime tagging", rows=len(short), sample=sample)]


def check_changes_completeness() -> list[dict]:
    """Quantify the known incompleteness of the Wikipedia change-event table.

    Real S&P 500 turnover runs ~20-25 names/year. Years reporting far fewer events
    are under-recorded upstream, not quiet years. This is reported as INFO because
    it is a documented property of the source, not a bug we introduced - but it must
    be visible, because it bounds how far back event-driven analysis can be trusted.
    """
    if not silver_exists("reference/sp500_changes"):
        return []
    ch = read_silver("reference/sp500_changes")
    yr = pd.to_datetime(ch["effective_date"], errors="coerce").dt.year
    per_year = yr.value_counts().sort_index()
    modern = per_year[per_year.index >= 2010]
    sparse = per_year[(per_year.index >= 2000) & (per_year.index < 2010)]
    return [_finding(
        "changes_completeness", INFO, "reference/sp500_changes",
        f"change events/year: 2010+ median {int(modern.median()) if len(modern) else 0} "
        f"(plausible), 2000-2009 median {int(sparse.median()) if len(sparse) else 0} "
        f"(under-recorded upstream; prefer membership_intervals before 2010)",
        rows=len(ch))]


def run() -> pd.DataFrame:
    """Run every applicable check and persist the report to quality/data_quality."""
    findings: list[dict] = []

    if silver_exists("market/daily_bars"):
        bars = read_silver("market/daily_bars")
        actions = (read_silver("market/corporate_actions")
                   if silver_exists("market/corporate_actions") else pd.DataFrame())
        calendar = (read_silver("reference/trading_calendar")
                    if silver_exists("reference/trading_calendar") else pd.DataFrame())

        findings += check_primary_key(bars)
        findings += check_bar_integrity(bars)
        findings += check_extreme_moves(bars, actions)
        findings += check_stale_prices(bars)
        findings += check_calendar_gaps(bars, calendar)
        findings += check_ticker_recycling(bars)
        findings += check_universe_coverage(bars)
    else:
        findings.append(_finding("availability", INFO, "market/daily_bars",
                                 "not built yet"))

    if silver_exists("reference/sp500_membership_snapshots"):
        findings += check_membership_sanity(
            read_silver("reference/sp500_membership_snapshots"))
    findings += check_macro_history_depth()
    findings += check_changes_completeness()

    if not findings:
        findings.append(_finding("all_checks", INFO, "-", "no issues found"))

    rep = pd.DataFrame(findings)
    for col in ("rows", "sample"):
        if col not in rep.columns:
            rep[col] = None
    rep["checked_at"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    order = {ERROR: 0, WARN: 1, INFO: 2}
    rep = rep.sort_values("severity", key=lambda s: s.map(order)).reset_index(drop=True)
    write_silver(rep, "quality/data_quality")
    return rep
