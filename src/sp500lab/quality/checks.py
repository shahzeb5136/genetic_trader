"""Data quality checks over bronze, silver and gold.

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

What is checked, by layer
-------------------------
bronze      every artifact re-hashed against the manifest (`check_bronze_manifest`)
silver      a schema contract per dataset, then per-dataset invariants: bars,
            adjustment factors, benchmarks, the trading calendar, the security
            master, membership intervals, corporate actions, XBRL fundamentals,
            macro series
cross       two sources that measure the same thing must agree: FRED's VIX close
            against Yahoo's, SPY against the index it tracks. A silent unit change
            or a symbol swap upstream shows up here and nowhere else.
gold        spread estimates and delisting assumptions the engine consumes

The `run()` output schema is stable - `sp500lab status` and the honesty report read it.
"""

from __future__ import annotations

import logging
import re
from typing import Callable

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
#: A price series that keeps updating this long after index membership ended almost
#: certainly belongs to a different company that inherited the symbol. Matches the
#: 365-day window backtest/panel.py documents against.
RECYCLE_DAYS = 365
#: Daily macro series older than this, relative to the newest session we hold, have
#: stopped updating - a broken endpoint, not a quiet market.
MACRO_STALE_DAYS = 14

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{0,5}(?:\.[A-Z]{1,2})?$")

#: The longest gap between two consecutive sessions that is a closure rather than
#: missing data. A holiday on a long weekend is 4 days; Thanksgiving week is 5; the
#: September 2001 closure (Sept 11-14) is 7. Anything longer has never happened.
MAX_SESSION_GAP_DAYS = 7

#: Bars known to be wrong at the vendor, reviewed by hand, and left in place.
#:
#: Reviewed 2026-09-02 (ADR-040). Four rows out of 3.7M where `low > open` by a few
#: cents, plus one zero open. None of them can move a result: the engine executes at
#: the open and marks at the close, the zero open is filled from the close and counted,
#: and the spread estimator that reads high/low is a 21-session trailing median. They
#: stay ERROR-shaped in the data and WARN-shaped in the report, so `quality --strict`
#: can gate a pipeline on NEW vendor errors without these five blocking it forever.
#:
#: Add a row here only after looking at it. The point of the list is that everything
#: on it has been looked at.
KNOWN_BAD_BARS: dict[tuple[str, str], str] = {
    ("HUBB", "2021-05-05"): "low 196.21 > open 195.98; Yahoo print error, same day as UA",
    ("UA", "2021-05-05"): "low 21.00 > open 20.87; Yahoo print error, same day as HUBB",
    ("SAF", "2021-08-30"): "low 25.000 > open 24.995 on a 1,389-share session",
    ("NAVIV", "2014-04-30"): "open printed as 0.00; panel fills it from the close",
}


def _finding(check: str, severity: str, entity: str, detail: str, **extra) -> dict:
    return {"check": check, "severity": severity, "entity": entity,
            "detail": detail, **extra}


def _where(df: pd.DataFrame, mask, n: int = 3, key: str = "ticker") -> str:
    """'AAPL@2020-01-02, ...' for the first few offending rows."""
    hit = df.loc[mask]
    if key in hit.columns and "date" in hit.columns:
        return ", ".join(f"{r[key]}@{r['date']}" for _, r in hit.head(n).iterrows())
    if key in hit.columns:
        return ", ".join(str(v) for v in hit[key].head(n))
    return ""


# ==========================================================================
# Schema contracts
# ==========================================================================

#: Columns every consumer of a silver dataset is allowed to assume. A missing column
#: fails loudly here rather than as a KeyError three layers up. Dates are strings of
#: the form YYYY-MM-DD everywhere - the lake never stores datetime64, so a query can
#: compare them as text and DuckDB can push the predicate down.
SCHEMAS: dict[str, dict[str, tuple[str, ...]]] = {
    "market/daily_bars": {
        "required": ("security_id", "ticker", "date", "open", "high", "low", "close",
                     "volume", "source", "ingest_date"),
        "dates": ("date",), "numeric": ("open", "high", "low", "close", "volume"),
    },
    "market/daily_bars_adjusted": {
        "required": ("security_id", "ticker", "date", "open", "high", "low", "close",
                     "volume", "adj_factor", "adj_factor_price", "price_convention",
                     "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"),
        "dates": ("date",),
        "numeric": ("adj_factor", "adj_factor_price", "adj_open", "adj_close"),
    },
    "market/adjustment_factors": {
        "required": ("security_id", "ticker", "date", "adj_factor", "adj_factor_price",
                     "price_convention"),
        "dates": ("date",), "numeric": ("adj_factor", "adj_factor_price"),
    },
    "market/corporate_actions": {
        "required": ("security_id", "ticker", "date", "action_type", "value", "source"),
        "dates": ("date",), "numeric": ("value",),
    },
    "market/benchmarks": {
        "required": ("date", "ticker", "open", "high", "low", "close", "volume",
                     "dividend", "split_ratio", "source"),
        "dates": ("date",), "numeric": ("open", "high", "low", "close", "volume"),
    },
    "reference/trading_calendar": {
        "required": ("date", "year", "month", "day_of_week", "is_month_end",
                     "is_quarter_end", "is_year_end", "session_index"),
        "dates": ("date",), "numeric": ("session_index",),
    },
    "reference/security_master": {
        "required": ("security_id", "cik", "ticker", "name", "first_seen", "last_seen"),
        "dates": (), "numeric": (),
    },
    "reference/sp500_membership_intervals": {
        "required": ("security_id", "ticker", "start_date", "end_date", "end_is_open",
                     "source"),
        "dates": ("start_date",), "numeric": (),
    },
    "fundamentals/xbrl_facts": {
        "required": ("security_id", "ticker", "cik", "tag", "unit", "period_start",
                     "period_end", "value", "form", "filed_date", "accession"),
        "dates": ("period_end", "filed_date"), "numeric": ("value",),
    },
    "macro/fred_series": {
        "required": ("series_id", "date", "value", "revised"),
        "dates": ("date",), "numeric": ("value",),
    },
    "factors/fama_french_daily": {
        "required": ("date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf", "mom"),
        "dates": ("date",), "numeric": ("mkt_rf", "smb", "hml", "rmw", "cma", "rf", "mom"),
    },
}


def check_schema(dataset: str, df: pd.DataFrame) -> list[dict]:
    """Required columns present, date columns are YYYY-MM-DD text, numerics numeric."""
    spec = SCHEMAS.get(dataset)
    if spec is None:
        return []
    out: list[dict] = []
    missing = [c for c in spec["required"] if c not in df.columns]
    if missing:
        out.append(_finding("schema", ERROR, dataset,
                            f"missing required column(s): {', '.join(missing)}",
                            rows=len(missing), sample=", ".join(missing)))
    for col in spec["dates"]:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        if not pd.api.types.is_string_dtype(s) and not pd.api.types.is_object_dtype(s):
            out.append(_finding("schema", ERROR, dataset,
                                f"{col} is {s.dtype}, expected YYYY-MM-DD text",
                                rows=len(s)))
            continue
        probe = s.sample(min(len(s), 2000), random_state=0).astype(str)
        bad = ~probe.str.match(_DATE_RE)
        if bad.any():
            out.append(_finding("schema", ERROR, dataset,
                                f"{col} has values that are not YYYY-MM-DD",
                                rows=int(bad.sum()),
                                sample=", ".join(probe[bad].head(3))))
    for col in spec["numeric"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            out.append(_finding("schema", ERROR, dataset,
                                f"{col} is {df[col].dtype}, expected numeric"))
    return out


# ==========================================================================
# Daily bars
# ==========================================================================

def check_bar_integrity(bars: pd.DataFrame) -> list[dict]:
    """OHLC relationships that must hold by definition, and prints that cannot be."""
    out: list[dict] = []
    need = {"open", "high", "low", "close"}
    if not need.issubset(bars.columns):
        return out

    b = bars.dropna(subset=list(need))
    known = pd.Series(
        [(t, d) in KNOWN_BAD_BARS for t, d in zip(b["ticker"], b["date"])], index=b.index)
    violations = {
        "high < low": b["high"] < b["low"],
        "high < open": b["high"] < b["open"],
        "high < close": b["high"] < b["close"],
        "low > open": b["low"] > b["open"],
        "low > close": b["low"] > b["close"],
        "close <= 0": b["close"] <= 0,
        # A zero open is a vendor placeholder, not a price. The panel builder fills it
        # from the close and counts the fill; it must never be treated as a fill price.
        "open <= 0 on a priced bar": (b["open"] <= 0) & (b["close"] > 0),
    }
    for label, mask in violations.items():
        new = mask & ~known
        n = int(new.sum())
        if n:
            out.append(_finding("bar_integrity", ERROR, "market/daily_bars",
                                f"{n} rows where {label}", rows=n,
                                sample=_where(b, new)))
    reviewed = pd.Series(False, index=b.index)
    for mask in violations.values():
        reviewed |= mask & known
    n = int(reviewed.sum())
    if n:
        out.append(_finding("bar_integrity", WARN, "market/daily_bars",
                            f"{n} known vendor print errors, reviewed and left in place "
                            "(KNOWN_BAD_BARS, ADR-040)", rows=n,
                            sample=_where(b, reviewed, n=5)))

    if "volume" in bars.columns:
        n = int((bars["volume"] < 0).sum())
        if n:
            out.append(_finding("bar_integrity", ERROR, "market/daily_bars",
                                f"{n} rows with negative volume", rows=n))
    # Every bar must be a number. A NaN close is a hole; the engine handles holes at the
    # edges of a series (listing, delisting) but a hole in the middle is a data problem.
    inner = bars.sort_values(["security_id", "date"])
    nan_close = inner["close"].isna()
    if nan_close.any():
        out.append(_finding("bar_integrity", WARN, "market/daily_bars",
                            f"{int(nan_close.sum())} rows with no close at all",
                            rows=int(nan_close.sum()), sample=_where(inner, nan_close)))
    return out


def check_primary_key(bars: pd.DataFrame) -> list[dict]:
    """(security_id, date) must be unique - a duplicate silently double-counts."""
    dup = bars.duplicated(subset=["security_id", "date"], keep=False)
    n = int(dup.sum())
    if not n:
        return []
    return [_finding("primary_key", ERROR, "market/daily_bars",
                     f"{n} duplicate (security_id, date) rows", rows=n,
                     sample=_where(bars, dup))]


def check_identity_mapping(bars: pd.DataFrame) -> list[dict]:
    """One security_id <-> one ticker inside the bar table.

    The security master allows a ticker to be reused by a different issuer over time;
    that is exactly why security_id exists. But within the bar table each id must map
    to a single ticker and each ticker to a single id, or a join on either key will
    silently splice two companies together.
    """
    out = []
    sid_multi = bars.groupby("security_id")["ticker"].nunique()
    n = int((sid_multi > 1).sum())
    if n:
        out.append(_finding("identity", ERROR, "market/daily_bars",
                            f"{n} security_id(s) carry more than one ticker", rows=n,
                            sample=", ".join(sid_multi[sid_multi > 1].index[:4])))
    tick_multi = bars.groupby("ticker")["security_id"].nunique()
    n = int((tick_multi > 1).sum())
    if n:
        out.append(_finding("identity", ERROR, "market/daily_bars",
                            f"{n} ticker(s) map to more than one security_id", rows=n,
                            sample=", ".join(tick_multi[tick_multi > 1].index[:4])))
    return out


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
    runs = (b.assign(_g=grp).groupby(["security_id", "ticker", "_g"]).size()
            .rename("len").reset_index())
    long_runs = runs[runs["len"] > STALE_RUN]
    if long_runs.empty:
        return []
    top = long_runs.sort_values("len", ascending=False).head(5)
    sample = ", ".join(f"{r.ticker}:{int(r.len)}" for r in top.itertuples())
    return [_finding("stale_prices", WARN, "market/daily_bars",
                     f"{len(long_runs)} runs of >{STALE_RUN} consecutive identical "
                     f"closes (longest {int(long_runs['len'].max())} bars)",
                     rows=int(len(long_runs)), sample=sample)]


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


def check_off_calendar_bars(bars: pd.DataFrame, calendar: pd.DataFrame) -> list[dict]:
    """A bar on a date the calendar does not have is a bar the panel will drop."""
    if calendar is None or calendar.empty:
        return []
    sessions = set(calendar["date"].astype(str))
    off = ~bars["date"].astype(str).isin(sessions)
    n = int(off.sum())
    if not n:
        return []
    return [_finding("off_calendar", ERROR, "market/daily_bars",
                     f"{n} bars dated on non-sessions (the panel silently drops them)",
                     rows=n, sample=_where(bars, off))]


def check_ticker_recycling(bars: pd.DataFrame,
                           intervals: pd.DataFrame | None = None) -> list[dict]:
    """Detect price history that continues long after index membership ended.

    Tickers get REUSED. Compuware (CPWR) was an S&P 500 member until 2014; the symbol
    was later reassigned to an unrelated OTC shell trading 0-50 shares a day. A price
    feed keyed on the bare ticker happily returns the shell's prices stitched onto the
    index member's history, and the join looks perfectly clean - same ticker, no gap,
    no error. The result is a fabricated series that a model will treat as one company.

    The membership intervals are what make this detectable: if a security's index
    membership ended years ago but "its" price series is still updating today, the
    recent bars almost certainly belong to a different company.

    Two findings come out of this:

    * **recycled** - the symbol kept trading >1yr past membership. Counted in *bars*, so
      the size of the contamination is visible, not just the number of names.
    * **phantom** - the worst case: NO bar falls inside the membership window at all.
      Marshall & Ilsley (MI) left the index in 2011; the "MI" series on disk starts in
      2015. Every one of its bars is the impostor's, and the panel - which clips to
      membership - will hold nothing for it. The name is silently absent from every
      backtest while the coverage report counts it as priced.

    Mitigation for downstream users: clip each security's price history to its
    membership interval (plus indicator warmup). `query.prices_clipped_to_membership`
    does exactly that and the panel builder uses it.
    """
    if intervals is None:
        if not silver_exists("reference/sp500_membership_intervals"):
            return []
        intervals = read_silver("reference/sp500_membership_intervals")
    iv = intervals
    open_now = set(iv.loc[iv["end_is_open"].astype(bool), "ticker"])
    # A name can leave the index and later re-enter (Hilton, Hyatt). Such a ticker has
    # both a closed and an open interval, and its ongoing price history is entirely
    # expected - only tickers with NO current membership can be recycled.
    closed = iv[(~iv["end_is_open"].astype(bool))
                & (~iv["ticker"].isin(open_now))].dropna(subset=["end_date"])
    if closed.empty:
        return []

    span = (closed.groupby("ticker").agg(mem_start=("start_date", "min"),
                                         mem_end=("end_date", "max")))
    b = bars[bars["ticker"].isin(span.index)]
    if b.empty:
        return []
    b = b.merge(span, left_on="ticker", right_index=True, how="left")
    cutoff = (pd.to_datetime(b["mem_end"]) + pd.Timedelta(days=RECYCLE_DAYS))
    after = pd.to_datetime(b["date"]) > cutoff
    inside = (b["date"] >= b["mem_start"]) & (b["date"] <= b["mem_end"])

    findings = []
    recycled = (b[after].groupby("ticker").size().rename("impostor_bars")
                .sort_values(ascending=False))
    if len(recycled):
        sample = ", ".join(f"{t}({int(n)} bars)" for t, n in recycled.head(6).items())
        findings.append(_finding(
            "ticker_recycling", WARN, "market/daily_bars",
            f"{len(recycled)} tickers have {int(recycled.sum())} price bars >1yr after "
            f"index membership ended - likely symbol reuse by an unrelated company; "
            f"clip prices to membership intervals before use",
            rows=int(recycled.sum()), sample=sample))

    has_inside = b[inside].groupby("ticker").size()
    phantom = sorted(set(span.index) & set(b["ticker"]) - set(has_inside.index))
    if phantom:
        findings.append(_finding(
            "phantom_history", WARN, "market/daily_bars",
            f"{len(phantom)} tickers have price bars but NONE inside their index "
            f"membership window - the whole series belongs to a later company; the "
            f"panel holds nothing for them and coverage counts them as priced",
            rows=len(phantom), sample=", ".join(phantom[:8])))
    return findings


def check_universe_coverage(bars: pd.DataFrame,
                            intervals: pd.DataFrame | None = None) -> list[dict]:
    """How much of the survivorship-free universe actually has USABLE price data.

    "Usable" means at least one bar inside the name's membership window. A series
    that exists on disk but belongs entirely to a later company under the same symbol
    (see `check_ticker_recycling`, "phantom") is counted as missing here, because that
    is what it is to every backtest.
    """
    if intervals is None:
        if not silver_exists("reference/sp500_membership_intervals"):
            return []
        intervals = read_silver("reference/sp500_membership_intervals")
    iv = intervals
    wanted = set(iv["ticker"].unique())

    span = iv.assign(end=iv["end_date"].where(~iv["end_is_open"].astype(bool),
                                                "9999-12-31"))
    span = span.groupby("ticker").agg(lo=("start_date", "min"), hi=("end", "max"))
    b = bars[bars["ticker"].isin(wanted)].merge(span, left_on="ticker",
                                                 right_index=True, how="left")
    inside = (b["date"] >= b["lo"]) & (b["date"] <= b["hi"])
    usable = set(b.loc[inside, "ticker"].unique())
    on_disk = set(b["ticker"].unique())
    phantom = len(on_disk - usable)
    missing = sorted(wanted - usable)
    pct = 100 * len(usable) / max(len(wanted), 1)

    sev = INFO if pct >= 95 else WARN
    return [_finding("universe_coverage", sev, "market/daily_bars",
                     f"{len(usable)}/{len(wanted)} ({pct:.1f}%) of ever-members have "
                     f"price data inside their membership window; {len(missing)} missing "
                     f"({phantom} of those have bars on disk, all from a later company)",
                     rows=len(missing),
                     sample=", ".join(missing[:12]))]


# ==========================================================================
# Adjustment chain
# ==========================================================================

def check_adjustment_factors(factors: pd.DataFrame) -> list[dict]:
    """The factor table's own invariants, independent of the prices it scales.

    * every factor is a positive finite number
    * the NEWEST bar of every security has factor exactly 1.0 - that anchoring is what
      makes the series reproducible (ADR-006); a drift here means the cumulative
      product started from the wrong end
    * one price convention per table - mixing them is the 10x split bug (ADR-007)
    """
    out = []
    ent = "market/adjustment_factors"
    for col in ("adj_factor", "adj_factor_price"):
        bad = ~np.isfinite(factors[col]) | (factors[col] <= 0)
        n = int(bad.sum())
        if n:
            out.append(_finding("adjustment_factors", ERROR, ent,
                                f"{n} rows where {col} is not a positive finite number",
                                rows=n, sample=_where(factors, bad)))
    f = factors.sort_values(["security_id", "date"])
    last = f.groupby("security_id").tail(1)
    off = (last["adj_factor"] - 1.0).abs() > 1e-9
    n = int(off.sum())
    if n:
        out.append(_finding("adjustment_factors", ERROR, ent,
                            f"{n} securities whose newest bar has adj_factor != 1.0 "
                            "(the series must be anchored at the present)",
                            rows=n, sample=_where(last, off)))
    conv = set(factors["price_convention"].dropna().unique())
    if len(conv) > 1:
        out.append(_finding("adjustment_factors", ERROR, ent,
                            f"mixed price conventions in one table: {sorted(conv)}",
                            rows=len(conv)))
    return out


def check_adjusted_bars(adjusted: pd.DataFrame) -> list[dict]:
    """The adjusted table must be exactly raw x factor, and positive wherever raw is."""
    out = []
    ent = "market/daily_bars_adjusted"
    a = adjusted.dropna(subset=["close", "adj_close", "adj_factor"])
    expect = a["close"] * a["adj_factor"]
    off = (a["adj_close"] - expect).abs() > 1e-6 * expect.abs().clip(lower=1e-9)
    n = int(off.sum())
    if n:
        out.append(_finding("adjusted_bars", ERROR, ent,
                            f"{n} rows where adj_close != close * adj_factor",
                            rows=n, sample=_where(a, off)))
    neg = (a["close"] > 0) & ~(a["adj_close"] > 0)
    n = int(neg.sum())
    if n:
        out.append(_finding("adjusted_bars", ERROR, ent,
                            f"{n} rows with a positive close but a non-positive adj_close",
                            rows=n, sample=_where(a, neg)))
    if "adj_volume" in adjusted.columns and "volume" in adjusted.columns:
        v = adjusted.dropna(subset=["volume"])
        badv = v["adj_volume"].isna() | (v["adj_volume"] < 0)
        n = int(badv.sum())
        if n:
            out.append(_finding("adjusted_bars", ERROR, ent,
                                f"{n} rows where volume is present but adj_volume is "
                                "missing or negative", rows=n, sample=_where(v, badv)))
    return out


def check_adjustment_vs_vendor() -> list[dict]:
    """Surface the normalize step's return-level comparison against the vendor.

    normalize/adjustments.py writes one row per ticker with the median daily-return
    disagreement between our adjusted series and Yahoo's. A REVIEW flag means a split
    or dividend we got wrong (or they did); either way a human should look.
    """
    if not silver_exists("quality/adjustment_vs_vendor"):
        return [_finding("adjustment_vs_vendor", INFO, "quality/adjustment_vs_vendor",
                         "not built - run `normalize` to compare against the vendor")]
    rep = read_silver("quality/adjustment_vs_vendor")
    review = rep[rep["flag"] == "REVIEW"]
    if review.empty:
        return [_finding("adjustment_vs_vendor", INFO, "quality/adjustment_vs_vendor",
                         f"{len(rep)} tickers agree with the vendor on daily returns "
                         f"(median disagreement {rep['median_ret_diff'].median():.2e})",
                         rows=len(rep))]
    top = review.sort_values("median_ret_diff", ascending=False).head(5)
    sample = ", ".join(f"{r.ticker}({r.median_ret_diff:.1e})" for r in top.itertuples())
    return [_finding("adjustment_vs_vendor", WARN, "quality/adjustment_vs_vendor",
                     f"{len(review)} tickers disagree with the vendor's returns by more "
                     "than 1bp on a typical day - a missed split or a wrong ex-date",
                     rows=len(review), sample=sample)]


# ==========================================================================
# Benchmarks and the trading calendar
# ==========================================================================

def check_benchmarks(bench: pd.DataFrame, calendar: pd.DataFrame | None) -> list[dict]:
    """Each benchmark series must be as clean as the bars it is compared against."""
    out = []
    ent = "market/benchmarks"
    need = {"open", "high", "low", "close"}
    b = bench.dropna(subset=[c for c in need if c in bench.columns])
    bad = ((b["high"] < b["low"]) | (b["high"] < b["open"]) | (b["high"] < b["close"])
           | (b["low"] > b["open"]) | (b["low"] > b["close"]) | (b["close"] <= 0))
    n = int(bad.sum())
    if n:
        out.append(_finding("benchmarks", ERROR, ent, f"{n} rows with impossible OHLC",
                            rows=n, sample=_where(b, bad)))
    dup = bench.duplicated(subset=["ticker", "date"], keep=False)
    if dup.any():
        out.append(_finding("benchmarks", ERROR, ent,
                            f"{int(dup.sum())} duplicate (ticker, date) rows",
                            rows=int(dup.sum()), sample=_where(bench, dup)))
    if "dividend" in bench.columns:
        spy = bench[bench["ticker"] == "SPY"]
        if len(spy) and not (spy["dividend"].fillna(0) > 0).any():
            out.append(_finding("benchmarks", ERROR, ent,
                                "SPY carries no dividend events - the total-return "
                                "benchmark would be a price return (ADR-006)"))
        neg = bench["dividend"].fillna(0) < 0
        if neg.any():
            out.append(_finding("benchmarks", ERROR, ent,
                                f"{int(neg.sum())} negative dividend rows",
                                rows=int(neg.sum()), sample=_where(bench, neg)))
    if "split_ratio" in bench.columns:
        bad_split = bench["split_ratio"].notna() & (bench["split_ratio"] < 0)
        if bad_split.any():
            out.append(_finding("benchmarks", ERROR, ent,
                                f"{int(bad_split.sum())} negative split ratios",
                                rows=int(bad_split.sum()), sample=_where(bench, bad_split)))
    if calendar is not None and len(calendar):
        sessions = set(calendar["date"].astype(str))
        off = ~bench["date"].astype(str).isin(sessions)
        # Index levels (^VIX) can print on a day the ETFs did not trade; that is an
        # oddity of the vendor, not a calendar error, so it is reported not failed.
        if off.any():
            by = bench[off].groupby("ticker").size()
            sample = ", ".join(f"{t}:{int(n)}" for t, n in by.items())
            out.append(_finding("benchmarks", INFO, ent,
                                f"{int(off.sum())} benchmark rows on dates outside the "
                                "SPY-derived calendar (dropped by consumers)",
                                rows=int(off.sum()), sample=sample))
    return out


def check_trading_calendar(cal: pd.DataFrame, bars: pd.DataFrame | None) -> list[dict]:
    """The calendar is the spine every matrix is aligned to; it has to be flawless."""
    out = []
    ent = "reference/trading_calendar"
    c = cal.sort_values("date").reset_index(drop=True)
    dt = pd.to_datetime(c["date"], errors="coerce")
    if dt.isna().any():
        out.append(_finding("calendar", ERROR, ent,
                            f"{int(dt.isna().sum())} unparseable dates",
                            rows=int(dt.isna().sum())))
        c, dt = c[dt.notna()], dt[dt.notna()]
    weekend = dt.dt.dayofweek >= 5
    if weekend.any():
        out.append(_finding("calendar", ERROR, ent,
                            f"{int(weekend.sum())} sessions fall on a weekend",
                            rows=int(weekend.sum()),
                            sample=", ".join(c.loc[weekend, "date"].head(3))))
    dup = c["date"].duplicated(keep=False)
    if dup.any():
        out.append(_finding("calendar", ERROR, ent,
                            f"{int(dup.sum())} duplicate session dates",
                            rows=int(dup.sum())))
    si = c["session_index"].to_numpy()
    if len(si) and not np.array_equal(si, np.arange(si[0], si[0] + len(si))):
        out.append(_finding("calendar", ERROR, ent,
                            "session_index is not contiguous and ascending"))
    if len(si) and si[0] != 0:
        out.append(_finding("calendar", WARN, ent,
                            f"session_index starts at {si[0]}, not 0"))
    # is_month_end must be true exactly where the next session is in another month.
    nxt = dt.shift(-1)
    expect_me = (dt.dt.to_period("M") != nxt.dt.to_period("M"))
    wrong = c["is_month_end"].astype(bool) != expect_me
    if wrong.any():
        out.append(_finding("calendar", ERROR, ent,
                            f"{int(wrong.sum())} sessions where is_month_end disagrees "
                            "with the following session's month",
                            rows=int(wrong.sum()),
                            sample=", ".join(c.loc[wrong, "date"].head(3))))
    # No unexplained holes. The longest closure in the modern record is September
    # 2001 (7 days); anything longer is missing data, not a closure.
    gap_days = dt.diff().dt.days
    holes = gap_days > MAX_SESSION_GAP_DAYS
    if holes.any():
        out.append(_finding("calendar", ERROR, ent,
                            f"{int(holes.sum())} gaps of more than {MAX_SESSION_GAP_DAYS} "
                            "calendar days between consecutive sessions",
                            rows=int(holes.sum()),
                            sample=", ".join(
                                f"{c.loc[i - 1, 'date']}->{c.loc[i, 'date']}"
                                for i in c.index[holes][:3])))
    if bars is not None and len(bars):
        priced = set(bars["date"].astype(str).unique())
        empty = ~c["date"].astype(str).isin(priced)
        if empty.any():
            out.append(_finding("calendar", WARN, ent,
                                f"{int(empty.sum())} sessions have no bar for any security",
                                rows=int(empty.sum()),
                                sample=", ".join(c.loc[empty, "date"].head(3))))
    return out


# ==========================================================================
# Identity: the security master and membership
# ==========================================================================

def check_security_master(master: pd.DataFrame) -> list[dict]:
    out = []
    ent = "reference/security_master"
    dup = master["security_id"].duplicated(keep=False)
    if dup.any():
        out.append(_finding("security_master", ERROR, ent,
                            f"{int(dup.sum())} duplicate security_id rows",
                            rows=int(dup.sum()),
                            sample=", ".join(master.loc[dup, "security_id"].head(3))))
    pair_dup = master.duplicated(subset=["cik", "ticker"], keep=False) & master["cik"].notna()
    if pair_dup.any():
        out.append(_finding("security_master", ERROR, ent,
                            f"{int(pair_dup.sum())} rows share the same (cik, ticker) "
                            "under different ids", rows=int(pair_dup.sum()),
                            sample=_where(master, pair_dup)))
    no_cik = master["cik"].isna() | (master["cik"].fillna(0) <= 0)
    if no_cik.any():
        out.append(_finding("security_master", INFO, ent,
                            f"{int(no_cik.sum())} securities have no CIK (assigned from a "
                            "price feed, never matched to an SEC registrant)",
                            rows=int(no_cik.sum()),
                            sample=", ".join(master.loc[no_cik, "ticker"].head(6))))
    bad_t = ~master["ticker"].astype(str).str.match(_TICKER_RE)
    if bad_t.any():
        out.append(_finding("security_master", WARN, ent,
                            f"{int(bad_t.sum())} tickers do not match the canonical "
                            "pattern (share classes use a dot: BRK.B)",
                            rows=int(bad_t.sum()),
                            sample=", ".join(master.loc[bad_t, "ticker"].head(6))))
    return out


def check_referential_integrity(master: pd.DataFrame, tables: dict[str, pd.DataFrame]
                                ) -> list[dict]:
    """Every security_id anywhere in silver must exist in the master."""
    out = []
    known = set(master["security_id"])
    for name, df in tables.items():
        if df is None or "security_id" not in df.columns:
            continue
        orphan = sorted(set(df["security_id"].dropna()) - known)
        if orphan:
            out.append(_finding("referential_integrity", ERROR, name,
                                f"{len(orphan)} security_id(s) not in the security master",
                                rows=len(orphan), sample=", ".join(orphan[:5])))
    return out


def check_membership_intervals(iv: pd.DataFrame) -> list[dict]:
    out = []
    ent = "reference/sp500_membership_intervals"
    is_open = iv["end_is_open"].astype(bool)
    closed = iv[~is_open]
    inverted = closed["end_date"].notna() & (closed["end_date"] < closed["start_date"])
    if inverted.any():
        out.append(_finding("membership", ERROR, ent,
                            f"{int(inverted.sum())} intervals end before they start",
                            rows=int(inverted.sum()), sample=_where(closed, inverted)))
    no_end = ~is_open & iv["end_date"].isna()
    if no_end.any():
        out.append(_finding("membership", ERROR, ent,
                            f"{int(no_end.sum())} closed intervals have no end_date",
                            rows=int(no_end.sum()), sample=_where(iv, no_end)))
    multi_open = iv[is_open].groupby("security_id").size()
    n = int((multi_open > 1).sum())
    if n:
        out.append(_finding("membership", ERROR, ent,
                            f"{n} securities have more than one open interval", rows=n,
                            sample=", ".join(multi_open[multi_open > 1].index[:4])))
    # Overlapping intervals for one security would double-count it in universe_asof.
    overlaps = 0
    sample = []
    for sid, g in iv.sort_values("start_date").groupby("security_id"):
        if len(g) < 2:
            continue
        ends = g["end_date"].where(~g["end_is_open"].astype(bool), "9999-12-31")
        prev_end = ends.shift(1)
        clash = (g["start_date"] <= prev_end).fillna(False)
        if clash.any():
            overlaps += int(clash.sum())
            sample.append(str(g["ticker"].iloc[0]))
    if overlaps:
        out.append(_finding("membership", ERROR, ent,
                            f"{overlaps} overlapping intervals for the same security",
                            rows=overlaps, sample=", ".join(sample[:5])))
    bad_t = ~iv["ticker"].astype(str).str.match(_TICKER_RE)
    if bad_t.any():
        out.append(_finding("membership", WARN, ent,
                            f"{int(bad_t.sum())} rows with a non-canonical ticker",
                            rows=int(bad_t.sum()),
                            sample=", ".join(iv.loc[bad_t, "ticker"].head(6))))
    return out


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


# ==========================================================================
# Corporate actions
# ==========================================================================

#: Split ratios outside this are almost certainly a decimal-place error upstream.
#: 1:100 reverse splits happen; 1:1000 does not, in the S&P 500.
SPLIT_RATIO_RANGE = (0.005, 200.0)


def check_corporate_actions(actions: pd.DataFrame, calendar: pd.DataFrame | None,
                            bars: pd.DataFrame | None) -> list[dict]:
    out = []
    ent = "market/corporate_actions"
    kinds = set(actions["action_type"].unique()) - {"dividend", "split"}
    if kinds:
        out.append(_finding("corporate_actions", ERROR, ent,
                            f"unknown action_type(s): {sorted(kinds)}"))
    nonpos = ~(actions["value"] > 0)
    if nonpos.any():
        out.append(_finding("corporate_actions", ERROR, ent,
                            f"{int(nonpos.sum())} actions with a non-positive value",
                            rows=int(nonpos.sum()), sample=_where(actions, nonpos)))
    sp = actions[actions["action_type"] == "split"]
    odd = (sp["value"] < SPLIT_RATIO_RANGE[0]) | (sp["value"] > SPLIT_RATIO_RANGE[1])
    if odd.any():
        out.append(_finding("corporate_actions", WARN, ent,
                            f"{int(odd.sum())} split ratios outside "
                            f"{SPLIT_RATIO_RANGE} - check the decimal place",
                            rows=int(odd.sum()), sample=_where(sp, odd)))
    unity = sp["value"] == 1.0
    if unity.any():
        out.append(_finding("corporate_actions", WARN, ent,
                            f"{int(unity.sum())} splits with ratio 1.0 (a no-op that "
                            "should have been filtered)", rows=int(unity.sum())))
    dup = actions.duplicated(subset=["security_id", "date", "action_type"], keep=False)
    if dup.any():
        out.append(_finding("corporate_actions", ERROR, ent,
                            f"{int(dup.sum())} duplicate (security_id, date, type) rows",
                            rows=int(dup.sum()), sample=_where(actions, dup)))
    if calendar is not None and len(calendar):
        off = ~actions["date"].astype(str).isin(set(calendar["date"].astype(str)))
        if off.any():
            out.append(_finding("corporate_actions", WARN, ent,
                                f"{int(off.sum())} actions dated on non-sessions (an "
                                "ex-date is always a session; these will never match a "
                                "bar and the factor chain ignores them)",
                                rows=int(off.sum()), sample=_where(actions, off)))
    # A dividend larger than half the prior close is a special distribution or a
    # data error; either way the factor it produces is enormous and deserves eyes.
    if bars is not None and len(bars):
        dv = actions[actions["action_type"] == "dividend"]
        if len(dv):
            b = bars.sort_values(["security_id", "date"])
            b = b.assign(prev_close=b.groupby("security_id")["close"].shift(1))
            m = dv.merge(b[["security_id", "date", "prev_close"]],
                         on=["security_id", "date"], how="left")
            big = (m["value"] > 0.5 * m["prev_close"]) & m["prev_close"].notna()
            if big.any():
                out.append(_finding("corporate_actions", WARN, ent,
                                    f"{int(big.sum())} dividends exceed half the prior "
                                    "close - special distributions or bad prints",
                                    rows=int(big.sum()), sample=_where(m, big)))
    return out


# ==========================================================================
# Fundamentals
# ==========================================================================

#: The concepts the feature layer actually reads. Coverage below these is a broken
#: pull, not a quiet quarter.
CORE_XBRL_TAGS = ("NetIncomeLoss", "Assets", "StockholdersEquity",
                  "EarningsPerShareDiluted", "WeightedAverageNumberOfSharesOutstandingBasic")
#: Nothing in the S&P 500 reports a line item above ten trillion dollars.
XBRL_MAX_ABS_VALUE = 1e13


def check_fundamentals(xbrl: pd.DataFrame, universe: set[str] | None = None) -> list[dict]:
    """The bitemporal contract and basic hygiene of the XBRL fact table."""
    out = []
    ent = "fundamentals/xbrl_facts"
    null_filed = xbrl["filed_date"].isna()
    if null_filed.any():
        out.append(_finding("fundamentals", ERROR, ent,
                            f"{int(null_filed.sum())} facts have no filed_date - "
                            "unusable point-in-time", rows=int(null_filed.sum())))
    null_val = xbrl["value"].isna()
    if null_val.any():
        out.append(_finding("fundamentals", ERROR, ent,
                            f"{int(null_val.sum())} facts have no value",
                            rows=int(null_val.sum())))
    early = xbrl["filed_date"] < xbrl["period_end"]
    if early.any():
        # Happens: an entity-level fact (shares outstanding on the cover page) is dated
        # a few days AFTER the filing. Rare, and harmless for a filed_date <= as_of
        # query, but a large count would mean the two dates were swapped in the parser.
        sev = ERROR if early.mean() > 0.001 else WARN
        out.append(_finding("fundamentals", sev, ent,
                            f"{int(early.sum())} facts filed before their period ended "
                            "(cover-page facts do this; a large share means swapped dates)",
                            rows=int(early.sum()), sample=_where(xbrl, early)))
    key = ["security_id", "tag", "unit", "period_start", "period_end",
           "filed_date", "accession"]
    dup = xbrl.duplicated(subset=[k for k in key if k in xbrl.columns], keep=False)
    if dup.any():
        out.append(_finding("fundamentals", ERROR, ent,
                            f"{int(dup.sum())} exact duplicate facts (same filing, same "
                            "period, same tag)", rows=int(dup.sum())))
    huge = xbrl["value"].abs() > XBRL_MAX_ABS_VALUE
    if huge.any():
        out.append(_finding("fundamentals", WARN, ent,
                            f"{int(huge.sum())} facts with |value| > {XBRL_MAX_ABS_VALUE:.0e} "
                            "- a unit or scale error", rows=int(huge.sum()),
                            sample=_where(xbrl, huge)))
    present = set(xbrl["tag"].unique())
    missing_tags = [t for t in CORE_XBRL_TAGS if t not in present]
    if missing_tags:
        out.append(_finding("fundamentals", ERROR, ent,
                            f"core tag(s) absent from the pull: {missing_tags}",
                            rows=len(missing_tags)))
    if universe:
        have = set(xbrl["security_id"].unique())
        pct = 100 * len(have & universe) / max(len(universe), 1)
        out.append(_finding("fundamentals", INFO if pct >= 85 else WARN, ent,
                            f"{len(have & universe)}/{len(universe)} ({pct:.0f}%) of priced "
                            "securities have XBRL facts (XBRL begins 2009; older-only "
                            "names never filed any)", rows=len(universe - have)))
    return out


# ==========================================================================
# Macro
# ==========================================================================

#: Plausible bounds per series. Deliberately loose - these catch a unit change
#: (percent vs decimal, index rebasing) or a swapped series, not a bad month.
MACRO_RANGES: dict[str, tuple[float, float]] = {
    "DGS10": (0.0, 20.0), "DGS2": (0.0, 20.0), "DGS3MO": (-0.5, 20.0),
    "DFF": (0.0, 25.0), "T10Y2Y": (-5.0, 5.0), "T10Y3M": (-5.0, 6.0),
    "VIXCLS": (5.0, 100.0), "UNRATE": (2.0, 30.0), "USREC": (0.0, 1.0),
    "CPIAUCSL": (10.0, 500.0), "DCOILWTICO": (-50.0, 250.0),
    "DTWEXBGS": (50.0, 200.0), "BAMLH0A0HYM2": (1.0, 25.0), "BAMLC0A0CM": (0.3, 10.0),
    "GDPC1": (1000.0, 60000.0), "INDPRO": (1.0, 200.0), "PAYEMS": (20000.0, 250000.0),
    "UMCSENT": (30.0, 130.0),
}
#: Series observed every business day. Anything else is monthly or quarterly and is
#: not checked for staleness (a July CPI print in early September is normal).
MACRO_DAILY = ("DGS10", "DGS2", "DGS3MO", "DFF", "T10Y2Y", "T10Y3M", "VIXCLS",
               "DCOILWTICO", "DTWEXBGS", "BAMLH0A0HYM2", "BAMLC0A0CM")


def check_macro_integrity(macro: pd.DataFrame) -> list[dict]:
    out = []
    ent = "macro/fred_series"
    dup = macro.duplicated(subset=["series_id", "date"], keep=False)
    if dup.any():
        out.append(_finding("macro", ERROR, ent,
                            f"{int(dup.sum())} duplicate (series_id, date) rows",
                            rows=int(dup.sum()), sample=_where(macro, dup, key="series_id")))
    all_null = macro.groupby("series_id")["value"].apply(lambda s: s.isna().all())
    if all_null.any():
        out.append(_finding("macro", ERROR, ent,
                            f"{int(all_null.sum())} series have no values at all",
                            rows=int(all_null.sum()),
                            sample=", ".join(all_null[all_null].index)))
    for sid, (lo, hi) in MACRO_RANGES.items():
        s = macro.loc[macro["series_id"] == sid, "value"].dropna()
        if s.empty:
            continue
        bad = (s < lo) | (s > hi)
        if bad.any():
            out.append(_finding("macro", WARN, ent,
                                f"{sid}: {int(bad.sum())} values outside [{lo:g}, {hi:g}] "
                                f"(min {s.min():.4g}, max {s.max():.4g}) - unit change "
                                "or swapped series?", rows=int(bad.sum())))
    rec = macro.loc[macro["series_id"] == "USREC", "value"].dropna()
    if len(rec) and not rec.isin([0.0, 1.0]).all():
        out.append(_finding("macro", ERROR, ent, "USREC is not a 0/1 indicator"))
    # `revised` is a property of the series, so it must be constant within one.
    mixed = macro.groupby("series_id")["revised"].nunique()
    if (mixed > 1).any():
        out.append(_finding("macro", ERROR, ent,
                            f"{int((mixed > 1).sum())} series carry both revised=True and "
                            "revised=False rows", rows=int((mixed > 1).sum())))
    return out


def check_macro_staleness(macro: pd.DataFrame, as_of: str | None) -> list[dict]:
    """A daily series whose last print is weeks behind the newest bar has stopped."""
    if not as_of:
        return []
    last = macro.dropna(subset=["value"]).groupby("series_id")["date"].max()
    ref = pd.Timestamp(as_of)
    stale = {}
    for sid in MACRO_DAILY:
        if sid in last.index:
            lag = (ref - pd.Timestamp(last[sid])).days
            if lag > MACRO_STALE_DAYS:
                stale[sid] = lag
    if not stale:
        return []
    sample = ", ".join(f"{s}({d}d)" for s, d in sorted(stale.items(), key=lambda x: -x[1]))
    return [_finding("macro_staleness", WARN, "macro/fred_series",
                     f"{len(stale)} daily macro series are more than {MACRO_STALE_DAYS} "
                     f"days behind the newest price bar ({as_of})",
                     rows=len(stale), sample=sample)]


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


# ==========================================================================
# Cross-source agreement
# ==========================================================================

def check_vix_cross_source(macro: pd.DataFrame, bench: pd.DataFrame) -> list[dict]:
    """FRED's VIXCLS and Yahoo's ^VIX are the same index from two vendors.

    They must agree to the cent on almost every day. A systematic offset means one of
    them is a different contract; a burst of disagreement means one feed has stale or
    shifted dates. Either way the macro `vix` feature and the timing engine's VIX gate
    would be reading different worlds.
    """
    f = macro[(macro["series_id"] == "VIXCLS")].dropna(subset=["value"])
    y = bench[bench["ticker"] == "^VIX"].dropna(subset=["close"])
    if f.empty or y.empty:
        return []
    m = f.merge(y[["date", "close"]], on="date", how="inner")
    if len(m) < 100:
        return [_finding("cross_source_vix", WARN, "macro/fred_series",
                         f"only {len(m)} overlapping VIX dates between FRED and Yahoo")]
    diff = (m["value"] - m["close"]).abs()
    share = float((diff > 0.05).mean())
    med = float(diff.median())
    if med > 0.5:
        return [_finding("cross_source_vix", ERROR, "macro/fred_series",
                         f"FRED VIXCLS and Yahoo ^VIX differ by {med:.2f} on a typical "
                         "day - these are not the same series", rows=len(m))]
    if share > 0.01:
        return [_finding("cross_source_vix", WARN, "macro/fred_series",
                         f"FRED and Yahoo VIX disagree by >0.05 on {share:.1%} of "
                         f"{len(m)} days (max {diff.max():.2f})",
                         rows=int((diff > 0.05).sum()))]
    return [_finding("cross_source_vix", INFO, "macro/fred_series",
                     f"FRED VIXCLS and Yahoo ^VIX agree on {len(m)} days "
                     f"(median gap {med:.1e}, {int((diff > 0.05).sum())} days >0.05)",
                     rows=len(m))]


def check_spy_tracks_index(bench: pd.DataFrame) -> list[dict]:
    """SPY's daily price return must track ^GSPC almost perfectly.

    They are the same 500 stocks; the residual is the ETF's expense ratio and a few
    basis points of tracking noise. A correlation below 0.98 means a symbol was
    swapped or a series was shifted by a day - and every d_sharpe in the registry is
    measured against SPY.
    """
    s = bench[bench["ticker"] == "SPY"].set_index("date")["close"].sort_index()
    g = bench[bench["ticker"] == "^GSPC"].set_index("date")["close"].sort_index()
    common = s.index.intersection(g.index)
    if len(common) < 250:
        return []
    rs, rg = s[common].pct_change().dropna(), g[common].pct_change().dropna()
    corr = float(np.corrcoef(rs, rg)[0, 1])
    sev = ERROR if corr < 0.98 else INFO
    return [_finding("cross_source_spy", sev, "market/benchmarks",
                     f"SPY vs ^GSPC daily return correlation {corr:.4f} over "
                     f"{len(rs)} days" + (" - SPY is not tracking the index" if sev == ERROR
                                          else ""), rows=len(rs))]


def check_ff_market_vs_spy(ff: pd.DataFrame, bench: pd.DataFrame) -> list[dict]:
    """The Fama-French market factor and SPY are two vendors' view of one market.

    `mkt_rf + rf` is CRSP's value-weighted total return on every US stock; SPY is the
    largest 500 of them. Their daily returns correlate above 0.98. Anything lower means
    the factor parser lost a decimal (percent vs fraction shows up as a correlation of
    ~1.0 with the wrong SCALE, so the slope is checked too) or one series is shifted.
    """
    ent = "factors/fama_french_daily"
    need = {"date", "mkt_rf", "rf"}
    if not need.issubset(ff.columns):
        return []
    s = bench[bench["ticker"] == "SPY"].set_index("date")["adj_close_vendor"].sort_index()
    if s.dropna().empty:
        s = bench[bench["ticker"] == "SPY"].set_index("date")["close"].sort_index()
    f = ff.dropna(subset=["mkt_rf", "rf"]).set_index("date")
    common = s.index.intersection(f.index)
    if len(common) < 250:
        return []
    spy_ret = s[common].pct_change().dropna()
    mkt = (f.loc[common, "mkt_rf"] + f.loc[common, "rf"]).loc[spy_ret.index]
    corr = float(np.corrcoef(spy_ret, mkt)[0, 1])
    slope = float(np.polyfit(mkt, spy_ret, 1)[0])
    problems = []
    if corr < 0.98:
        problems.append(f"correlation {corr:.4f} < 0.98")
    if not 0.8 <= slope <= 1.2:
        problems.append(f"slope {slope:.3f} - a unit or scale error in one series")
    if problems:
        return [_finding("cross_source_ff", ERROR, ent,
                         "SPY does not track the Fama-French market factor: "
                         + "; ".join(problems), rows=len(spy_ret))]
    return [_finding("cross_source_ff", INFO, ent,
                     f"SPY vs Fama-French Mkt-RF+RF: correlation {corr:.4f}, slope "
                     f"{slope:.3f} over {len(spy_ret)} days", rows=len(spy_ret))]


def check_factor_sanity(ff: pd.DataFrame) -> list[dict]:
    """Daily factor returns are small numbers in decimals; anything else is a parse bug."""
    ent = "factors/fama_french_daily"
    out = []
    cols = [c for c in ("mkt_rf", "smb", "hml", "rmw", "cma", "mom") if c in ff.columns]
    for c in cols:
        v = ff[c].dropna()
        if v.empty:
            out.append(_finding("factors", ERROR, ent, f"{c} has no values"))
            continue
        # The worst day in the record is October 1987 at about -17%. A |return| above
        # 0.30 has never happened; above 1.0 the column is still in percent.
        big = (v.abs() > 0.30).sum()
        if big:
            sev = ERROR if v.abs().max() > 1.0 else WARN
            out.append(_finding("factors", sev, ent,
                                f"{c}: {int(big)} daily returns beyond +/-30% "
                                f"(max |{v.abs().max():.3f}|) - percent left undivided?",
                                rows=int(big)))
    if "rf" in ff.columns:
        rf = ff["rf"].dropna()
        if len(rf) and ((rf < -1e-6) | (rf > 0.001)).any():
            out.append(_finding("factors", WARN, ent,
                                "rf outside [0, 0.1%] per day - the T-bill rate is never "
                                "negative and never above ~25%/yr",
                                rows=int(((rf < -1e-6) | (rf > 0.001)).sum())))
    dup = ff["date"].duplicated(keep=False)
    if dup.any():
        out.append(_finding("factors", ERROR, ent, f"{int(dup.sum())} duplicate dates",
                            rows=int(dup.sum())))
    return out


# ==========================================================================
# Gold
# ==========================================================================

def check_half_spread(spreads: pd.DataFrame) -> list[dict]:
    from ..backtest.spreads import MAX_PLAUSIBLE_SPREAD
    out = []
    ent = "gold/backtest/half_spread"
    v = spreads["half_spread"]
    bad = v.notna() & ((v < 0) | (v > MAX_PLAUSIBLE_SPREAD))
    if bad.any():
        out.append(_finding("half_spread", ERROR, ent,
                            f"{int(bad.sum())} half-spreads outside [0, "
                            f"{MAX_PLAUSIBLE_SPREAD}]", rows=int(bad.sum())))
    kinds = set(spreads["binding"].dropna().unique()) - {"estimator", "tick_floor",
                                                          "fallback"}
    if kinds:
        out.append(_finding("half_spread", ERROR, ent, f"unknown binding value(s) {kinds}"))
    nan_share = float(v.isna().mean())
    if nan_share > 0.05:
        out.append(_finding("half_spread", WARN, ent,
                            f"{nan_share:.1%} of security-sessions have no spread estimate "
                            "(the cost model falls back to a flat 5bp there)",
                            rows=int(v.isna().sum())))
    return out


def check_delisting_returns(d: pd.DataFrame) -> list[dict]:
    from ..backtest.delisting import REASON_CATEGORIES
    out = []
    ent = "gold/backtest/delisting_returns"
    r = d["delist_return"]
    bad = r.notna() & ((r < -1.0) | (r > 1.0))
    if bad.any():
        out.append(_finding("delisting", ERROR, ent,
                            f"{int(bad.sum())} delisting returns outside [-1, 1]",
                            rows=int(bad.sum())))
    kinds = set(d["reason_category"].unique()) - set(REASON_CATEGORIES)
    if kinds:
        out.append(_finding("delisting", ERROR, ent, f"unknown reason category {kinds}"))
    bk = d[d["reason_category"] == "bankruptcy"]
    if len(bk) and not (bk["delist_return"] <= -0.99).all():
        out.append(_finding("delisting", ERROR, ent,
                            "a bankruptcy row does not book a total loss"))
    unresolved = int((d["reason_category"] == "unresolved").sum())
    if unresolved:
        out.append(_finding("delisting", INFO, ent,
                            f"{unresolved}/{len(d)} exits have an unresolved reason and "
                            "assume a 0% terminal return (docs/BACKTEST.md)",
                            rows=unresolved))
    return out


# ==========================================================================
# Bronze
# ==========================================================================

def check_bronze_manifest() -> list[dict]:
    """Re-hash every bronze and vault artifact against its recorded checksum."""
    from ..storage import verify_manifest
    v = verify_manifest()
    out = []
    if v["corrupt"]:
        out.append(_finding("bronze_manifest", ERROR, "bronze",
                            f"{v['corrupt']} artifact(s) fail their sha256 - bit-rot or a "
                            "hand edit", rows=v["corrupt"],
                            sample=", ".join(f["path"] for f in v["failures"]
                                             if f["issue"] == "checksum mismatch")[:200]))
    if v["missing"]:
        out.append(_finding("bronze_manifest", ERROR, "bronze",
                            f"{v['missing']} manifest entries point at files that no "
                            "longer exist (retire them with a tombstone if deliberate)",
                            rows=v["missing"],
                            sample=", ".join(f["path"] for f in v["failures"]
                                             if f["issue"] == "missing")[:200]))
    if not out:
        out.append(_finding("bronze_manifest", INFO, "bronze",
                            f"{v['ok']}/{v['artifacts']} artifacts match their checksums "
                            f"({v['retired']} retired)", rows=v["ok"]))
    return out


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


# ==========================================================================
# Runner
# ==========================================================================

def _safe(findings: list[dict], name: str, fn: Callable[[], list[dict]]) -> None:
    """A check that crashes is itself a finding, never a reason to skip the rest."""
    try:
        findings.extend(fn())
    except Exception as exc:  # noqa: BLE001
        log.exception("check %s crashed", name)
        findings.append(_finding(name, ERROR, "-",
                                 f"check crashed: {type(exc).__name__}: {exc}"[:200]))


def _load(dataset: str) -> pd.DataFrame | None:
    return read_silver(dataset) if silver_exists(dataset) else None


def run(*, verify_bronze: bool = True) -> pd.DataFrame:
    """Run every applicable check and persist the report to quality/data_quality."""
    findings: list[dict] = []

    bars = _load("market/daily_bars")
    actions = _load("market/corporate_actions")
    calendar = _load("reference/trading_calendar")
    bench = _load("market/benchmarks")
    master = _load("reference/security_master")
    iv = _load("reference/sp500_membership_intervals")
    macro = _load("macro/fred_series")
    xbrl = _load("fundamentals/xbrl_facts")
    factors = _load("market/adjustment_factors")
    adjusted = _load("market/daily_bars_adjusted")

    # -- schema contracts first: everything below assumes the columns exist
    for name, df in (("market/daily_bars", bars), ("market/corporate_actions", actions),
                     ("reference/trading_calendar", calendar), ("market/benchmarks", bench),
                     ("reference/security_master", master),
                     ("reference/sp500_membership_intervals", iv),
                     ("macro/fred_series", macro), ("fundamentals/xbrl_facts", xbrl),
                     ("market/adjustment_factors", factors),
                     ("market/daily_bars_adjusted", adjusted)):
        if df is not None:
            _safe(findings, "schema", lambda n=name, d=df: check_schema(n, d))

    if verify_bronze:
        _safe(findings, "bronze_manifest", check_bronze_manifest)

    if bars is not None:
        _safe(findings, "primary_key", lambda: check_primary_key(bars))
        _safe(findings, "identity", lambda: check_identity_mapping(bars))
        _safe(findings, "bar_integrity", lambda: check_bar_integrity(bars))
        _safe(findings, "extreme_move", lambda: check_extreme_moves(bars, actions))
        _safe(findings, "stale_prices", lambda: check_stale_prices(bars))
        _safe(findings, "calendar_gaps", lambda: check_calendar_gaps(bars, calendar))
        _safe(findings, "off_calendar", lambda: check_off_calendar_bars(bars, calendar))
        _safe(findings, "ticker_recycling", lambda: check_ticker_recycling(bars, iv))
        _safe(findings, "universe_coverage", lambda: check_universe_coverage(bars, iv))
    else:
        findings.append(_finding("availability", INFO, "market/daily_bars", "not built yet"))

    if factors is not None:
        _safe(findings, "adjustment_factors", lambda: check_adjustment_factors(factors))
    if adjusted is not None:
        _safe(findings, "adjusted_bars", lambda: check_adjusted_bars(adjusted))
    _safe(findings, "adjustment_vs_vendor", check_adjustment_vs_vendor)

    if bench is not None:
        _safe(findings, "benchmarks", lambda: check_benchmarks(bench, calendar))
        _safe(findings, "cross_source_spy", lambda: check_spy_tracks_index(bench))
    if calendar is not None:
        _safe(findings, "calendar", lambda: check_trading_calendar(calendar, bars))

    if master is not None:
        _safe(findings, "security_master", lambda: check_security_master(master))
        _safe(findings, "referential_integrity", lambda: check_referential_integrity(
            master, {"market/daily_bars": bars, "market/corporate_actions": actions,
                     "reference/sp500_membership_intervals": iv,
                     "fundamentals/xbrl_facts": xbrl}))
    if iv is not None:
        _safe(findings, "membership", lambda: check_membership_intervals(iv))
    snaps = _load("reference/sp500_membership_snapshots")
    if snaps is not None:
        _safe(findings, "membership_sanity", lambda: check_membership_sanity(snaps))

    if actions is not None:
        _safe(findings, "corporate_actions",
              lambda: check_corporate_actions(actions, calendar, bars))

    if xbrl is not None:
        universe = set(bars["security_id"].unique()) if bars is not None else None
        _safe(findings, "fundamentals", lambda: check_fundamentals(xbrl, universe))

    if macro is not None:
        _safe(findings, "macro", lambda: check_macro_integrity(macro))
        newest = str(bars["date"].max()) if bars is not None else None
        _safe(findings, "macro_staleness", lambda: check_macro_staleness(macro, newest))
        if bench is not None:
            _safe(findings, "cross_source_vix", lambda: check_vix_cross_source(macro, bench))
    _safe(findings, "macro_history_depth", check_macro_history_depth)
    _safe(findings, "changes_completeness", check_changes_completeness)

    ff = _load("factors/fama_french_daily")
    if ff is not None:
        _safe(findings, "schema", lambda: check_schema("factors/fama_french_daily", ff))
        _safe(findings, "factors", lambda: check_factor_sanity(ff))
        if bench is not None:
            _safe(findings, "cross_source_ff", lambda: check_ff_market_vs_spy(ff, bench))

    from ..paths import GOLD_DIR
    sp = GOLD_DIR / "backtest" / "half_spread" / "data.parquet"
    if sp.exists():
        _safe(findings, "half_spread", lambda: check_half_spread(pd.read_parquet(sp)))
    dl = GOLD_DIR / "backtest" / "delisting_returns" / "data.parquet"
    if dl.exists():
        _safe(findings, "delisting", lambda: check_delisting_returns(pd.read_parquet(dl)))

    if not findings:
        findings.append(_finding("all_checks", INFO, "-", "no issues found"))

    rep = pd.DataFrame(findings)
    for col in ("rows", "sample"):
        if col not in rep.columns:
            rep[col] = None
    rep["checked_at"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    order = {ERROR: 0, WARN: 1, INFO: 2}
    rep = rep.sort_values("severity", key=lambda s: s.map(order), kind="stable")
    rep = rep.reset_index(drop=True)
    write_silver(rep, "quality/data_quality")
    return rep


def summary(rep: pd.DataFrame) -> dict[str, int]:
    """{'ERROR': n, 'WARN': n, 'INFO': n} for a report frame."""
    counts = rep["severity"].value_counts()
    return {k: int(counts.get(k, 0)) for k in (ERROR, WARN, INFO)}
