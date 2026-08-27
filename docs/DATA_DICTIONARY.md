# Data dictionary

Every silver table, keyed by its DuckDB view name. Query them via
`python -m sp500lab query "..."` or `sp500lab.query.connect()`.

Conventions used throughout:

- **Dates are `YYYY-MM-DD` strings**, not timestamps. Lexicographic ordering equals
  chronological ordering, comparisons work in SQL and pandas identically, and there is no
  timezone ambiguity on a daily bar.
- **`security_id`** is the join key everywhere. Never join on `ticker` (ADR-005).
- Tickers are normalized: share classes use `.` (`BRK.B`), never `-` or `/`.

---

## Reference

### `security_master`
Stable internal identity. Append-only — an ID is assigned once and never reused.

| Column | Type | Notes |
|---|---|---|
| `security_id` | str | Surrogate PK, `SID######` |
| `cik` | int64 | SEC registrant number; **0** when no SEC registrant |
| `ticker` | str | Normalized |
| `name` | str | Best-known name |
| `exchange` | str | Nasdaq / NYSE / OTC / CBOE / "" |
| `first_seen`, `last_seen` | str | Ingest dates bounding when we've observed it |

Natural key is `(cik, ticker)` — this keeps GOOG and GOOGL distinct (same CIK) and separates
Washington Mutual from Waste Management (same ticker, different CIK).

---

### `sp500_membership_intervals` ⭐
**The survivorship-free universe.** The most important table here.

| Column | Type | Notes |
|---|---|---|
| `security_id` | str | |
| `ticker` | str | |
| `start_date` | str | First monthly snapshot containing this name |
| `end_date` | str \| null | Last snapshot containing it; **null when still a member** |
| `end_is_open` | bool | Explicit "still in the index" flag |
| `source` | str | `wikipedia_revision_history` |

A ticker can have **multiple rows** — either because a company left and re-entered (Hilton), or
because the symbol was reassigned to a different company (WM).

Always read `end_is_open` rather than testing `end_date IS NULL`, so a null is never confused
with missing data. Use `query.universe_asof(date)`.

---

### `sp500_membership_snapshots`
The raw monthly observations the intervals are built from. 227 snapshots, 2007-03 → 2026-08.

| Column | Notes |
|---|---|
| `snapshot_date` | Month-end the snapshot represents |
| `revid` | Wikipedia revision ID — full provenance, re-checkable forever |
| `rev_timestamp` | Actual edit timestamp |
| `ticker`, `security_id` | |

---

### `sp500_current`
Today's 503 constituents with GICS classification. **Point-in-time unsafe by construction** —
using this for historical dates is exactly the survivorship-bias mistake.

`snapshot_date`, `security_id`, `ticker`, `name`, `cik`, `gics_sector`,
`gics_sub_industry`, `headquarters`, `date_added`, `founded`

---

### `sp500_changes`
407 index add/remove events with effective dates and free-text reasons.

`effective_date`, `added_ticker`, `added_name`, `removed_ticker`, `removed_name`, `reason`,
`snapshot_date`

⚠️ **Under-recorded before 2010** — ~7 events/year vs the ~20/year that actually occurred.
Prefer `sp500_membership_intervals` for that era (ADR-010).

---

### `trading_calendar`
6,702 NYSE sessions derived empirically from SPY (ADR-009).

`date`, `year`, `month`, `day_of_week`, `is_month_end`, `is_quarter_end`, `is_year_end`,
`session_index`, `calendar_source`

`session_index` is a dense 0-based counter — useful for "N sessions ago" without date
arithmetic. The `is_*_end` flags mark the last *trading* day of each period, not the calendar
date.

---

### `company_tickers`
SEC CIK↔ticker↔exchange spine. 10,388 current registrants.

---

## Market

### `daily_bars`
Raw OHLCV as received. 3.71M rows, 677 securities, 2000 → present.

| Column | Notes |
|---|---|
| `security_id`, `ticker`, `date` | PK is `(security_id, date)` |
| `open`, `high`, `low`, `close` | **See convention warning below** |
| `volume` | |
| `adj_close_vendor` | Vendor's adjusted close — kept **only** as a cross-check, never for analysis |
| `source` | `yfinance` |
| `ingest_date` | |

⚠️ **These are `split_adjusted`, not truly raw** — yfinance pre-applies splits to OHLC and
volume. Do not apply split factors again (ADR-007).

---

### `corporate_actions`
41,954 discrete events: 41,224 dividends, 730 splits.

| Column | Notes |
|---|---|
| `security_id`, `ticker`, `date` | `date` is the **ex-date** |
| `action_type` | `dividend` \| `split` |
| `value` | Cash amount per share, or split ratio (10.0 = 10:1) |
| `source` | |

Kept separate from prices on purpose: the events are the facts, adjustment factors are a
derived opinion about them.

---

### `adjustment_factors`
Cumulative back-adjustment factors, computed by us (ADR-006).

| Column | Notes |
|---|---|
| `security_id`, `ticker`, `date` | |
| `adj_factor` | Splits **+** dividends → use for **returns** (total return) |
| `adj_factor_price` | Splits only → use for price levels and to scale volume |
| `price_convention` | `as_traded` \| `split_adjusted` — what the input was |

The newest bar always has factor 1.0, so adjusted == raw today. Anchoring at the present is
deliberate: today's price is the one you'd actually transact at, and it makes the series
reproducible.

---

### `daily_bars_adjusted`
`daily_bars` ⨝ `adjustment_factors`, materialized. Adds `adj_open`, `adj_high`, `adj_low`,
`adj_close`, `adj_volume`.

Use `query.prices_clipped_to_membership()` rather than reading this directly — it excludes
bars belonging to a recycled symbol.

---

### `benchmarks`
SPY, RSP, ^GSPC, IWM, ^VIX. 32,577 rows. Same columns as `daily_bars` plus `description`.

The first backtest to run is buy-and-hold SPY. If it doesn't reproduce SPY's actual total
return to within a few basis points, the engine is wrong.

---

## Fundamentals

### `xbrl_facts` ⭐
Point-in-time SEC XBRL facts. **Bitemporal — read this carefully.**

| Column | Notes |
|---|---|
| `security_id`, `ticker`, `cik`, `entity_name` | |
| `taxonomy` | `us-gaap`, `dei`, … |
| `tag` | XBRL concept, e.g. `NetIncomeLoss` |
| `unit` | `USD`, `shares`, `USD/shares` |
| `period_start`, `period_end` | **effective_date** — when the fact was *true* |
| `value` | float |
| `fy`, `fp` | Fiscal year / period as the filer labelled them |
| `form` | `10-K`, `10-Q`, `8-K`, … |
| `filed_date` | **knowledge_date** — when it became *knowable* |
| `accession` | Filing accession number |
| `frame` | CY-frame label when assigned |

**The rule:** a model trading on date *T* may only see rows with `filed_date <= T`. Filtering
on `period_end` hands it next quarter's 10-K early — a leak that looks like alpha and vanishes
live.

**Restatements:** the same `(tag, period_end)` appears multiple times with different
`filed_date`s and values. All versions are kept. Use `query.fundamentals_asof(date)`, which
takes the latest row with `filed_date <= date`.

**Tag inconsistency:** revenue appears as `Revenues`,
`RevenueFromContractWithCustomerExcludingAssessedTax`, or `SalesRevenueNet` depending on filer
and era. Reconciling these into one series is Phase 3 work — don't assume one tag covers
everyone.

---

## Macro

### `fred_series`
18 series, 127,457 observations, long format.

| Column | Notes |
|---|---|
| `series_id` | e.g. `DGS10` |
| `date`, `value` | `value` is null where FRED reports `.` |
| `description` | |
| `revised` | ⚠️ **true = restated after publication** |

7 of 18 are flagged `revised` (CPI, GDP, payrolls, unemployment, industrial production,
sentiment, recession indicator). Using those at face value in a backtest is a look-ahead leak
(ADR-011). Market-based series are final on publication and safe.

---

## Quality

### `data_quality`
Output of `sp500lab quality`. `check`, `severity` (ERROR/WARN/INFO), `entity`, `detail`,
`rows`, `sample`, `checked_at`.

### `adjustment_vs_vendor`
Per-ticker agreement between our adjusted series and the vendor's, compared on **returns**
(invariant to the vendor's constant rescaling).

`ticker`, `bars`, `median_ret_diff`, `p99_ret_diff`, `max_ret_diff`, `flag`

Current state: median 1.3e-07 across 675 tickers, zero flagged.
