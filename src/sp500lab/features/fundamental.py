"""Point-in-time fundamentals: what the filings said, as of when they said it.

The one rule
------------
A fundamental value enters this file on its `filed_date`, never on its `period_end`.
Apple's FY2023 ended 2023-09-30 and was not public until 2023-11-03 - 33 days of free
lookahead for anyone who filters on the wrong column. **60.8% of (security, tag, period)
combinations in this dataset have been restated at least once**, so a naive join also
hands the model numbers that did not exist in any form on the date it is pretending to
trade.

Everything below is therefore built as a STEP FUNCTION over filed_date and sampled with
an as-of join. Nothing is interpolated, nothing is back-filled, and a company that has
not filed yet simply has no value - which is different from having a value of zero and is
represented differently (NaN, which portfolio.py treats as "no opinion").

The restatement frontier
-------------------------
At any date, a company's "latest reported Assets" is the value for the most recent
`period_end` among the filings it had published by then. A restatement of an OLDER period
does not change that, and a re-filing of the CURRENT period does. So the state machine is:

    frontier row  <=>  period_end == running maximum of period_end so far

which is one cumulative max over filings sorted by filed_date. Rows that are not on the
frontier are restatements of the past: they leave the current value alone but they are
counted, because how often a company restates is itself one of the most interesting
things this dataset knows (see `restatement_rate`).

Which duration, and why not TTM
--------------------------------
XBRL carries the same tag at several durations: a quarter, a half, nine months, a year.
Flow quantities here use the ANNUAL duration and balance-sheet quantities use the instant.
That is a deliberate simplification with a real cost, and the cost is stated rather than
hidden:

  * a trailing-twelve-month figure would be fresher by up to three quarters, but building
    one correctly means reconstructing fiscal Q4 as `annual - nine-month YTD` for every
    company and fiscal calendar, and a single sign error there produces a feature that
    looks like alpha.
  * annual is what Sloan (1996) used for accruals and Novy-Marx (2013) used for gross
    profitability, so the features below match their published constructions rather than
    approximating them.

Quarterly EPS is the exception, because the whole point of an earnings surprise is that
it is quarterly. Fiscal Q4 is usually filed only as part of the annual figure and so has
no quarterly row at all - the surprise is simply NaN that quarter rather than being
fabricated from a subtraction.

Coverage, honestly
------------------
Fundamentals begin 2009-04 (the XBRL mandate) against a 2007-03 membership start, and
cover 649 of the 973 securities that have ever been in the index. Every feature here is
therefore NaN for the first two years of the backtest window and permanently NaN for a
third of the names - disproportionately the ones that were delisted. A strategy that
requires fundamentals is implicitly trading a survivor-biased subset ON TOP of the
coverage gap in ADR-023. `FeaturePanel.coverage()` reports it per feature; use it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..query import connect

log = logging.getLogger(__name__)

#: Balance-sheet tags: a level at a point in time. `period_start` is null for these.
INSTANT_TAGS = (
    "Assets", "StockholdersEquity", "CashAndCashEquivalentsAtCarryingValue",
    "AssetsCurrent", "LiabilitiesCurrent", "LongTermDebtNoncurrent",
)

#: Income-statement and cash-flow tags, taken at the ANNUAL duration. See the docstring.
ANNUAL_TAGS = (
    "NetIncomeLoss", "Revenues", "GrossProfit",
    "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsForRepurchaseOfCommonStock", "ResearchAndDevelopmentExpense",
    "PaymentsToAcquirePropertyPlantAndEquipment",
)

#: Share counts, for market capitalisation. Two tags because neither alone covers the
#: index: `CommonStockSharesOutstanding` is a balance-sheet line and
#: `EntityCommonStockSharesOutstanding` is the filing's cover page. The cover page is
#: preferred where both exist - it is dated closer to the filing.
SHARE_TAGS = ("EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding")

#: An annual period, in days. Fiscal years are not all 365 days and 52/53-week retail
#: calendars land at 364 or 371, so the band is wide enough to admit them and narrow
#: enough to exclude a nine-month cumulative figure.
ANNUAL_DAYS = (330, 400)
QUARTER_DAYS = (80, 100)

#: A restatement counts when the refiled value moves by more than this fraction. A
#: rounding change from 1,234.0 to 1,234.001 is not a restatement, and counting it would
#: make `restatement_rate` a measure of XBRL tooling rather than of accounting.
RESTATEMENT_TOLERANCE = 0.01

#: Market caps below this are treated as a data error rather than as a small company.
#: MEASURED: 0.79% of index-member observations compute to under $500M, and the 1st
#: percentile of everything above it is $2.2bn - so there is a clean gap rather than a
#: continuum, and the low tail is share counts that are wrong rather than companies that
#: are tiny. The S&P 500's own inclusion floor has ranged from $4bn to $18bn over this
#: window; a member worth $60 (FOX, on one filing) does not exist.
#:
#: The cause is multi-class and partial-context share tags: a filer reports one share
#: class, or a treasury-share context, on its cover page. Dropping these matters more
#: than it looks, because market cap is a DENOMINATOR - a 10,000-share Simon Property
#: produces a book-to-market of 300 and any value strategy buys it first.
MIN_MARKET_CAP = 5e8


def compute(panel, rows: np.ndarray, *, data_cutoff: str | None = None) -> dict:
    """(R, S) matrices for every fundamental feature, sampled at `rows`."""
    con = connect()
    rows = np.asarray(rows, dtype=np.int64)
    as_of = panel.dates[rows]
    sid_pos = {s: i for i, s in enumerate(panel.security_ids.tolist())}
    shape = (len(rows), panel.n_securities)

    facts = _load(con, data_cutoff)
    if facts.empty:
        log.warning("no XBRL facts available; fundamental features will be empty")
        return {}
    facts = facts[facts["security_id"].isin(sid_pos)].reset_index(drop=True)

    inst = _pit(facts, INSTANT_TAGS, "instant", as_of, sid_pos, shape, lag=4)
    ann = _pit(facts, ANNUAL_TAGS, "annual", as_of, sid_pos, shape, lag=1)
    shares, filed_ord = _shares(facts, as_of, sid_pos, shape)

    cap = _market_cap(panel, rows, shares, filed_ord)
    out: dict[str, np.ndarray] = {
        "log_market_cap": np.log(np.where(cap > 0, cap, np.nan)),
    }

    assets = inst.get("Assets")
    equity = inst.get("StockholdersEquity")
    ni = ann.get("NetIncomeLoss")
    cfo = ann.get("NetCashProvidedByUsedInOperatingActivities")

    out["book_to_market"] = _div(equity, cap, positive_denominator=True)
    out["earnings_yield"] = _div(ni, cap, positive_denominator=True)
    out["cf_yield"] = _div(cfo, cap, positive_denominator=True)
    out["buyback_yield"] = _div(ann.get("PaymentsForRepurchaseOfCommonStock"), cap,
                                positive_denominator=True)

    out["gross_profitability"] = _div(ann.get("GrossProfit"), assets,
                                      positive_denominator=True)
    out["roe"] = _div(ni, equity, positive_denominator=True)
    # Sloan (1996): earnings that are not cash are the part that reverses. A high number
    # is a company whose profit lives in receivables and inventory, and it predicts
    # NEGATIVELY - which is why the raw quantity is stored and the sign is the
    # strategy's decision, not the feature's.
    out["accruals"] = _div(_sub(ni, cfo), assets, positive_denominator=True)
    out["capex_intensity"] = _div(ann.get("PaymentsToAcquirePropertyPlantAndEquipment"),
                                  assets, positive_denominator=True)
    out["rnd_intensity"] = _div(ann.get("ResearchAndDevelopmentExpense"), assets,
                                positive_denominator=True)
    out["cash_ratio"] = _div(inst.get("CashAndCashEquivalentsAtCarryingValue"), assets,
                             positive_denominator=True)
    out["current_ratio"] = _div(inst.get("AssetsCurrent"), inst.get("LiabilitiesCurrent"),
                                positive_denominator=True)
    # Leverage as 1 - equity/assets rather than liabilities/assets: `Liabilities` is
    # tagged by only 489 of 649 companies, and the identity holds for all of them.
    equity_ratio = _div(equity, assets, positive_denominator=True)
    out["leverage"] = None if equity_ratio is None else 1.0 - equity_ratio
    out["debt_to_assets"] = _div(inst.get("LongTermDebtNoncurrent"), assets,
                                 positive_denominator=True)

    out["asset_growth"] = _growth(assets, inst.get("Assets__lag"))
    out["sales_growth"] = _growth(ann.get("Revenues"), ann.get("Revenues__lag"))
    out["earnings_growth"] = _growth_signed(ni, ann.get("NetIncomeLoss__lag"))

    out.update(_earnings_surprise(facts, as_of, sid_pos, shape))
    out.update(_filing_behaviour(facts, as_of, sid_pos, shape))

    return {k: v for k, v in out.items() if v is not None}


# --------------------------------------------------------------------------
# Loading and the point-in-time join
# --------------------------------------------------------------------------

def _load(con, data_cutoff: str | None) -> pd.DataFrame:
    """Every fact we might use, with its duration classified once.

    Loaded in one query rather than per tag: DuckDB reads the parquet with predicate
    pushdown, and 12 separate scans of 3.3M rows costs more than one scan and a groupby.
    """
    wanted = list(INSTANT_TAGS + ANNUAL_TAGS + SHARE_TAGS +
                  ("EarningsPerShareDiluted",))
    placeholders = ", ".join("?" for _ in wanted)
    cutoff = " AND filed_date <= ?" if data_cutoff else ""
    params = wanted + ([data_cutoff] if data_cutoff else [])
    df = con.execute(f"""
        SELECT security_id, tag, unit, period_start, period_end, filed_date,
               value, accession,
               CASE
                 WHEN period_start IS NULL THEN 'instant'
                 WHEN datediff('day', CAST(period_start AS DATE),
                                      CAST(period_end AS DATE))
                      BETWEEN {ANNUAL_DAYS[0]} AND {ANNUAL_DAYS[1]} THEN 'annual'
                 WHEN datediff('day', CAST(period_start AS DATE),
                                      CAST(period_end AS DATE))
                      BETWEEN {QUARTER_DAYS[0]} AND {QUARTER_DAYS[1]} THEN 'quarter'
                 ELSE 'other'
               END AS duration
        FROM xbrl_facts
        WHERE tag IN ({placeholders}) AND value IS NOT NULL{cutoff}
    """, params).df()
    return _add_ordinals(df)


def _add_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """Dates as integer day numbers, and identifiers as plain strings.

    Mechanical, and load-bearing. The parquet columns arrive as Arrow strings, which
    support neither `cummax` (the restatement frontier) nor `merge_asof` (the as-of
    join) - the two operations this entire module is built out of.
    """
    if df.empty:
        return df
    for col in ("security_id", "tag", "unit", "accession"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    df["pe_ord"] = _ordinal(df["period_end"])
    df["fd_ord"] = _ordinal(df["filed_date"])
    return df.dropna(subset=["pe_ord", "fd_ord"])


#: The epoch every date in this module is measured from.
EPOCH = pd.Timestamp("1970-01-01")


def _ordinal(dates) -> pd.Series:
    """A date column as whole days since the epoch. Comparable, sortable, joinable.

    Subtracting a Timestamp and taking `.dt.days`, NOT dividing the int64 view by
    86.4e12. pandas 2.x parses these strings to datetime64[**us**], not [ns], so the
    divisor is wrong by a factor of a thousand and every date in the dataset collapses
    onto one of about twenty integers. That does not raise - it silently produces an
    as-of join in which a decade of filings share a key.
    """
    parsed = pd.to_datetime(pd.Series(dates).astype(str), errors="coerce")
    return (parsed - EPOCH).dt.days.where(parsed.notna())


def _frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Filings that advanced (or refreshed) the newest period, in filed_date order.

    This is the whole point-in-time mechanism in three lines. Sort by when it became
    public; take the running maximum of the period it describes; keep the rows that
    match it. What survives is a step function whose value at any date is "the most
    recent period this company had reported by then", restatements of that period
    included and restatements of older periods correctly ignored.
    """
    df = df.sort_values(["security_id", "fd_ord", "accession"], kind="stable")
    frontier = df.groupby("security_id", sort=False)["pe_ord"].cummax()
    return df[df["pe_ord"].to_numpy() == frontier.to_numpy()]


def _with_lag(front: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Attach the value from `lag` periods earlier, AS IT WAS FIRST REPORTED.

    Taking the year-ago figure from its original filing rather than from the latest
    restatement of it is both cheaper and more correct: a growth rate computed against a
    number that was revised two years later is a growth rate nobody could have computed
    at the time.
    """
    firsts = (front.drop_duplicates(["security_id", "pe_ord"], keep="first")
                   .sort_values(["security_id", "pe_ord"], kind="stable")
                   .copy())
    firsts["lag_value"] = firsts.groupby("security_id", sort=False)["value"].shift(lag)
    return front.merge(firsts[["security_id", "pe_ord", "lag_value"]],
                       on=["security_id", "pe_ord"], how="left")


def _asof_grid(step: pd.DataFrame, as_of: np.ndarray, sid_pos: dict,
               shape: tuple, columns: tuple[str, ...]) -> dict:
    """Sample a per-security step function at every rebalance date.

    `merge_asof(direction='backward')` is the as-of join: for each (security, date) it
    takes the last row whose `filed_date` is <= that date, and nothing else. That single
    call is what makes every fundamental feature in this file point-in-time.
    """
    step = step.sort_values("fd_ord", kind="stable")
    securities = step["security_id"].unique()
    as_of_ord = _ordinal(pd.Series(as_of.tolist())).to_numpy(dtype=np.int64)
    grid = pd.DataFrame({
        "security_id": np.repeat(securities, len(as_of)),
        "as_of": np.tile(as_of_ord, len(securities)),
    }).sort_values("as_of", kind="stable")

    joined = pd.merge_asof(
        grid, step[["security_id", "fd_ord", *columns]],
        left_on="as_of", right_on="fd_ord", by="security_id", direction="backward")

    date_pos = {int(d): i for i, d in enumerate(as_of_ord.tolist())}
    r_idx = joined["as_of"].map(date_pos).to_numpy(dtype=np.int64)
    s_idx = joined["security_id"].map(sid_pos).to_numpy(dtype=np.int64)

    out = {}
    for col in columns:
        mat = np.full(shape, np.nan)
        vals = joined[col].to_numpy(dtype=np.float64)
        ok = np.isfinite(vals)
        mat[r_idx[ok], s_idx[ok]] = vals[ok]
        out[col] = mat
    return out


def _pit(facts: pd.DataFrame, tags: tuple[str, ...], duration: str,
        as_of: np.ndarray, sid_pos: dict, shape: tuple, lag: int) -> dict:
    """{tag: (R,S), tag__lag: (R,S)} - the latest reported value and its lagged twin."""
    out: dict[str, np.ndarray] = {}
    subset = facts[facts["duration"] == duration]
    for tag in tags:
        df = subset[subset["tag"] == tag]
        if df.empty:
            log.info("no %s facts at %s duration", tag, duration)
            continue
        step = _with_lag(_frontier(df), lag)
        got = _asof_grid(step, as_of, sid_pos, shape, ("value", "lag_value"))
        out[tag] = got["value"]
        out[f"{tag}__lag"] = got["lag_value"]
    return out


# --------------------------------------------------------------------------
# Market capitalisation - the feature everything else divides by
# --------------------------------------------------------------------------

def _shares(facts: pd.DataFrame, as_of: np.ndarray, sid_pos: dict,
            shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    """(R, S) shares outstanding as last reported, and (R, S) when it was reported.

    The filing DATE comes back with the value because the share count is expressed in
    the share basis of that date, and a split between then and now changes what it
    means. See `_market_cap` for the algebra.
    """
    out = np.full(shape, np.nan)
    df = facts[(facts["tag"].isin(SHARE_TAGS)) & (facts["unit"] == "shares")]
    if df.empty:
        log.warning("no shares-outstanding facts; market cap will be unavailable")
        return out, out.copy()
    # The LARGEST count reported on a filing, not the first by tag priority. A filer
    # with two share classes tags each class separately, and a filer with a treasury
    # context tags that too; the smaller number is always the wrong one, and picking it
    # understates the company by a factor of anything up to a thousand.
    df = (df.groupby(["security_id", "fd_ord", "pe_ord"], as_index=False)
            .agg(value=("value", "max"), accession=("accession", "max")))

    step = _frontier(df).copy()
    step["filed_ord"] = step["fd_ord"].astype(float)
    got = _asof_grid(step, as_of, sid_pos, shape, ("value", "filed_ord"))
    return got["value"], got["filed_ord"]


def _market_cap(panel, rows: np.ndarray, shares: np.ndarray,
                filed_ord: np.ndarray) -> np.ndarray:
    """(R, S) market capitalisation in dollars.

    The split trap, in full, because getting it backwards is a clean 4x error that looks
    plausible:

        reported_shares are in the share basis of the filing date f
        as-traded price at t  = raw_close[t] * cum_split[t]     (ADR-007)
        shares at t           = reported_shares * cum_split[f] / cum_split[t]
        market_cap(t)         = shares at t * as-traded price
                              = reported_shares * cum_split[f] * raw_close[t]

    `cum_split[t]` cancels exactly. What is left needs the split ratio at the FILING
    date, `f`, and that is the part the obvious formula misses. Using `cum_split[t]`
    instead is right except across a split that happened between the filing and now - a
    window of at most one quarter, but in that window the error is the whole split ratio.
    Apple's 4:1 in August 2020 would have quartered it for a quarter.

    Sanity check, and it is a real one because the failure mode is a clean multiple:
    Apple's market cap today comes out in trillions rather than in hundreds of billions.
    """
    if not np.isfinite(shares).any():
        return np.full(shares.shape, np.nan)

    price = panel.raw_close[rows]
    split = _split_at_filing(panel, rows, filed_ord)
    with np.errstate(invalid="ignore"):
        cap = shares * split * price
    cap = np.where(np.isfinite(cap) & (cap > 0), cap, np.nan)

    bad = np.isfinite(cap) & (cap < MIN_MARKET_CAP)
    if bad.any():
        log.info("market cap: %d of %d observations below $%.0fM discarded as bad "
                 "share counts (see MIN_MARKET_CAP)",
                 int(bad.sum()), int(np.isfinite(cap).sum()), MIN_MARKET_CAP / 1e6)
        cap = np.where(bad, np.nan, cap)
    return cap


def _split_at_filing(panel, rows: np.ndarray, filed_ord: np.ndarray) -> np.ndarray:
    """(R, S) the cumulative split ratio in force on each cell's own filing date.

    Falls back to the ratio at the rebalance date where the filing date is unknown, or
    where the security had no bar that day (a filing can land on a market holiday). The
    fallback is the old behaviour and is correct except across an intervening split.
    """
    fallback = panel.cum_split[rows]
    if not np.isfinite(filed_ord).any():
        return fallback

    panel_ord = _ordinal(pd.Series(panel.dates.tolist())).to_numpy(dtype=np.int64)
    safe = np.where(np.isfinite(filed_ord), filed_ord, panel_ord[-1])
    pos = np.searchsorted(panel_ord, safe.astype(np.int64), side="right") - 1
    pos = np.clip(pos, 0, len(panel_ord) - 1)

    at_filing = panel.cum_split[pos, np.arange(panel.n_securities)[None, :]]
    return np.where(np.isfinite(at_filing) & (at_filing > 0), at_filing, fallback)


# --------------------------------------------------------------------------
# Earnings surprise, and the filing behaviour features
# --------------------------------------------------------------------------

def _earnings_surprise(facts: pd.DataFrame, as_of: np.ndarray, sid_pos: dict,
                       shape: tuple) -> dict:
    """Standardised unexpected earnings, from quarterly diluted EPS.

    SUE, in its original random-walk form (Foster, Olsen & Shevlin 1984): this quarter's
    EPS minus the same quarter a year ago, divided by the standard deviation of the last
    eight such differences. The naive model - "earnings will be what they were a year
    ago" - is the point: the surprise is the part the model got wrong, and
    post-earnings-announcement drift is the market's slow response to it.

    No analyst estimates are involved, which is a limitation and also the reason this is
    computable at all on a $20/month budget.

    Fiscal Q4 is normally filed only inside the annual figure, so a quarter in four has
    no row and its surprise is NaN. Reconstructing it as `annual - nine months` is
    possible and is deliberately not done here; a sign error in that subtraction would
    manufacture an enormous fake surprise once a year, every year.
    """
    eps = facts[(facts["tag"] == "EarningsPerShareDiluted")
                & (facts["duration"] == "quarter")]
    if eps.empty:
        return {}

    front = _frontier(eps)
    firsts = (front.drop_duplicates(["security_id", "pe_ord"], keep="first")
                   .sort_values(["security_id", "pe_ord"], kind="stable").copy())
    g = firsts.groupby("security_id", sort=False)["value"]
    firsts["diff_yoy"] = firsts["value"] - g.shift(4)
    firsts["sue_sd"] = (firsts.groupby("security_id", sort=False)["diff_yoy"]
                        .rolling(8, min_periods=4).std()
                        .reset_index(level=0, drop=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        firsts["sue"] = firsts["diff_yoy"] / firsts["sue_sd"].where(firsts["sue_sd"] > 0)
    firsts["sue"] = firsts["sue"].clip(-10, 10)

    step = front.merge(firsts[["security_id", "pe_ord", "sue", "diff_yoy"]],
                       on=["security_id", "pe_ord"], how="left")
    got = _asof_grid(step, as_of, sid_pos, shape, ("sue", "diff_yoy"))
    return {"eps_surprise": got["sue"], "eps_change_yoy": got["diff_yoy"]}


def _filing_behaviour(facts: pd.DataFrame, as_of: np.ndarray, sid_pos: dict,
                      shape: tuple) -> dict:
    """How a company files, which is information about the company.

    Three features that exist only because this dataset kept `filed_date` alongside
    `period_end` instead of collapsing them:

      days_since_filing  staleness. Everything above it is this old.
      filing_lag_days    how long after the period closed the filing appeared. A company
                         drifting later than its own habit is a documented red flag -
                         late filings cluster with restatements and with bad news.
      restatement_rate   the share of this company's reported facts that it has since
                         revised, counted only from revisions that had already happened.
                         Accounting quality, measured rather than assumed, and not
                         computable at all from a single-vintage fundamentals feed.
    """
    events = (facts.groupby(["security_id", "fd_ord"], as_index=False)
              .agg(pe_ord=("pe_ord", "max")))
    events = events.sort_values(["security_id", "fd_ord"], kind="stable")
    events["filing_lag_days"] = (events["fd_ord"] - events["pe_ord"]).astype(float)
    events["filed_ordinal"] = events["fd_ord"].astype(float)

    revisions = _revision_counts(facts)
    step = events.merge(revisions, on=["security_id", "fd_ord"], how="left")
    step[["cum_facts", "cum_revised"]] = (
        step.groupby("security_id", sort=False)[["n_facts", "n_revised"]]
        .cumsum().fillna(0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        step["restatement_rate"] = np.where(
            step["cum_facts"] > 20, step["cum_revised"] / step["cum_facts"], np.nan)

    got = _asof_grid(step, as_of, sid_pos, shape,
                     ("filing_lag_days", "filed_ordinal", "restatement_rate"))

    as_of_ord = _ordinal(pd.Series(as_of.tolist())).to_numpy(dtype=float)[:, None]
    days_since = as_of_ord - got["filed_ordinal"]
    return {
        "days_since_filing": np.where(days_since >= 0, days_since, np.nan),
        "filing_lag_days": got["filing_lag_days"],
        "restatement_rate": got["restatement_rate"],
    }


def _revision_counts(facts: pd.DataFrame) -> pd.DataFrame:
    """Per (security, filed_date): new facts published, and old facts materially changed.

    A fact is "revised" when the same (tag, unit, period_end) is filed again with a value
    that moves by more than RESTATEMENT_TOLERANCE. Ordering is by filed_date, so a
    revision is counted on the date it became visible and never before.
    """
    df = facts.sort_values(["security_id", "tag", "unit", "pe_ord",
                            "fd_ord", "accession"], kind="stable")
    key = ["security_id", "tag", "unit", "pe_ord"]
    grouped = df.groupby(key, sort=False)["value"]
    prev = grouped.shift(1)
    seq = df.groupby(key, sort=False).cumcount()

    with np.errstate(divide="ignore", invalid="ignore"):
        moved = (prev.notna()
                 & ((df["value"] - prev).abs()
                    > RESTATEMENT_TOLERANCE * prev.abs().clip(lower=1e-9)))
    out = df.assign(_new=(seq == 0).astype(int), _rev=moved.astype(int))
    return (out.groupby(["security_id", "fd_ord"], as_index=False)
            .agg(n_facts=("_new", "sum"), n_revised=("_rev", "sum")))


# --------------------------------------------------------------------------
# Small arithmetic helpers that keep NaN meaning "no opinion"
# --------------------------------------------------------------------------

def _div(num, den, *, positive_denominator: bool = False):
    if num is None or den is None:
        return None
    d = np.where(den > 0, den, np.nan) if positive_denominator else den
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / d
    return np.where(np.isfinite(out), out, np.nan)


def _sub(a, b):
    return None if a is None or b is None else a - b


def _growth(now, before):
    """Year-on-year growth of a quantity that should be positive (assets, sales)."""
    if now is None or before is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(before > 0, now / before - 1.0, np.nan)
    return np.where(np.isfinite(out), out, np.nan)


def _growth_signed(now, before):
    """Growth of a quantity that can be negative - earnings. Scaled by |before|.

    A ratio is meaningless when the denominator changes sign, so this is the change
    divided by the magnitude of the base. A company going from -$1bn to +$1bn scores
    +2.0, which is a defensible reading; a plain ratio would score -1.0, which is not.
    """
    if now is None or before is None:
        return None
    base = np.abs(before)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(base > 0, (now - before) / base, np.nan)
    return np.clip(np.where(np.isfinite(out), out, np.nan), -10.0, 10.0)
