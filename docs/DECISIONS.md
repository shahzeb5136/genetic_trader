# Architecture Decision Records

Why things are the way they are. Each record states the decision, the reasoning, and — where
it applies — the **measurement** that settled it.

Status values: `accepted`, `superseded`, `revisit-at-phase-N`.

---

## ADR-001 — Point-in-time universe from Wikipedia revision history

**Status:** accepted

**Context.** Survivorship bias is the largest single source of error in a naive S&P 500
backtest. Building the universe from today's constituent list deletes every company that
failed, was acquired, or was demoted. Point-in-time constituent data is a paid product
(Norgate, Sharadar), and the budget is \$20/month.

**Decision.** Reconstruct membership from the *revision history* of the Wikipedia article.
Take the last edit of each month and parse the constituent table as it stood at that moment.

**Why this works.** The article has been edited continuously since 2005 and every revision is
public and immutable. A revision from May 2010 records who was believed to be in the index in
May 2010 — a genuine point-in-time record. Re-reading today's page cannot reproduce it.

**Measured result.** 227 monthly snapshots, 2007-03 → 2026-08. **973 unique tickers ever, 502
current, 471 removed.** Validated against known events: LEH ends 2008-08, BSC ends 2008-06,
FNM/FRE end 2008-08, MER ends 2009-02 — all correct.

**Costs.** Monthly granularity dates an add/remove to within a month. Wikipedia is a
volunteer-maintained secondary source. Both acceptable for daily-bar strategies rebalancing
monthly or quarterly.

**Revisit at Phase 3:** diff against a paid constituent set the day one is acquired.

---

## ADR-002 — Free sources now, one paid EOD feed later

**Status:** accepted

**Decision.** Build the entire pipeline on free sources first. Purchase price history only
after the pipeline works.

**Reasoning.** Historical data is a *purchase*, not a rent — AAPL's 2013 closes do not change.
The subscription is only needed for the tail. Building on free data first means the paid month
is spent downloading into a pipeline that already works, rather than debugging and paying
simultaneously.

There is a second benefit that is easy to miss: running the biased pipeline first and *then*
the unbiased one produces your own measurement of survivorship bias, on your own universe.
That number is worth more than any published estimate.

**What is deferred.** Intraday bars, tick/L2, analyst estimates, options. None are needed for
daily-bar strategies and all break the budget.

---

## ADR-003 — Medallion layers with an immutable bronze

**Status:** accepted

**Decision.** bronze (raw) → silver (normalized) → gold (analysis-ready). Bronze is
append-only, checksummed, and written before any parsing.

**Reasoning.** Under a burst-buy funding model, a corrupted raw file is a permanent loss, not
an inconvenience. Writing bytes before parsing means a parser bug costs a re-parse rather than
a re-download.

**This paid for itself during initial development.** The constituent parser was rewritten
twice — once for the 2007 column reordering, once for header-based column detection — and both
rewrites replayed 246 cached revisions at zero network cost.

---

## ADR-004 — Membership history starts 2007-03, not 2005

**Status:** accepted

**Context.** The plan assumed Wikipedia constituent history "degrades before ~2007". That was
a guess. It was checked.

**Measurement.** Revisions before 2007-03 contain **no constituent table at all** (`{|` count
= 0). The page was a bulleted list of company names:

```
*[[3M Company]]
*[[Abbott Labs]]
*[[ACE Limited]]
```

No ticker column exists, so ticker-level membership is unrecoverable from this source. The
2007-03 → 2008-01 revisions *do* have a table, but with **Company first and Ticker second** —
the reverse of later years.

**Decision.** `history_start = 2007-03-01`. The parser detects the ticker column from the
header rather than assuming position, which recovered the 2007 snapshots (500 tickers each)
and hardens against future reordering.

**Consequence.** ~19.5 years of history covering four distinct regimes: the GFC, the 2010s
bull, the 2020 crash, and the 2022 bear. Sufficient for walk-forward validation.

**Possible future work:** map 2005–2006 company names to tickers via fuzzy matching. Buys 15
months at real risk of mismatch error. Not worth it now.

---

## ADR-005 — Security identity is (cik, ticker), not ticker

**Status:** accepted

**Decision.** A surrogate `security_id` keyed on `(cik, ticker)`, assigned once, never reused.

**Reasoning.** Tickers fail as identifiers in three distinct ways — changes (FB→META), share
classes (GOOG/GOOGL share a CIK), and recycling (WM was Washington Mutual, now Waste
Management). The third is the dangerous one because it fabricates a single company out of two
with no gap and no error.

**Measured result.** WM correctly resolves to two intervals: 2007-03 → 2008-08 (Washington
Mutual) and 2009-08 → open (Waste Management). `check_ticker_recycling` flags 155 tickers
whose price history runs more than a year past membership end.

**Consequence.** Downstream code must use `prices_clipped_to_membership()` rather than joining
on a bare ticker.

---

## ADR-006 — Compute adjustment factors; never trust a vendor's adjusted close

**Status:** accepted

**Decision.** Store raw OHLCV and corporate actions separately. Derive adjustment factors
ourselves and store them versioned.

**The measurement that settled it.** Comparing our factors against yfinance's `Adj Close` over
2018–2024 gave a *constant* relative offset per ticker:

| Ticker | Offset | Dividend yield × 1.7yr |
|---|---|---|
| JNJ | 4.72% | ~4.7% |
| KO | 4.33% | ~4.3% |
| MMM | 3.32% | ~3.3% |
| AAPL | 0.73% | ~0.85% |
| NVDA | 0.15% | ~0.15% |

Identical for *every row* of each series. The cause: the vendor's history had been rewritten
by dividends paid **after** our window ended. A backtest run in January and re-run in June
against the same vendor returns different numbers with no code change.

Our series is anchored (newest bar has factor 1.0) and therefore reproducible.

**Validation method.** Levels cannot be compared directly, precisely because of that constant
offset. Comparing daily **returns** — invariant to a constant scale — gives a median
disagreement of **1.3e-07** across 675 tickers, with **zero flagged for review**. That is
float rounding.

---

## ADR-007 — Price convention must be declared per source

**Status:** accepted

**Context.** Discovered by a failed validation, not by reading documentation.

**The bug.** Applying split factors to yfinance data double-counted NVDA's 2024 10:1 split by
10×. Investigation showed yfinance returns NVDA's 2024-06-06 close as ~\$121, not the ~\$1,210
that actually traded — its OHLC is **already split-adjusted** even with `auto_adjust=False`.
Only dividends are left out.

**Decision.** Declare a convention per source in `SOURCE_CONVENTION`:

- `as_traded` — apply splits and dividends (EODHD raw, most paid feeds)
- `split_adjusted` — apply dividends only (yfinance)

Stamp the convention into the output table.

**Why this matters more than it looks.** The failure is silent and looks like signal: prices
jump by the split ratio at every split, and a momentum model loads straight onto it. Nothing
errors.

**Action required at Phase 3:** verify EODHD's actual convention against a known split before
trusting a single bar. Do not assume the table entry is right.

---

## ADR-008 — Quality checks report, never repair

**Status:** accepted

**Decision.** Checks emit findings with a severity. Nothing mutates data.

**Reasoning.** "Fixing" a bad price usually means understanding a corporate action, not
clamping a number. Automatic repair hides the signal that something upstream is wrong.

**Currently found:** 1 ERROR (4 rows with `low > open`), 5 WARN (306 extreme moves, 856 stale
runs, 24 calendar gaps, 147 recycled tickers, 69.6% coverage), 1 INFO.

The extreme moves are mostly the recycled tickers and unrecorded reverse splits — CPWR's
symbol now belongs to an OTC shell trading 0–50 shares/day.

---

## ADR-009 — Trading calendar derived from data, not from holiday rules

**Status:** accepted

**Decision.** Derive the calendar from SPY's own bars.

**Reasoning.** A rules-based calendar drifts on unscheduled closures (Hurricane Sandy 2012,
December 2018 day of mourning) and marks them as missing bars. Deriving from the same feed as
the price data guarantees the calendar cannot disagree with the data it describes.

**Result.** 6,702 sessions, 2000-01-03 → 2026-08-26, median 252/year.

---

## ADR-010 — Wikipedia change-event table is unreliable before 2010

**Status:** accepted

**Measurement.** Events per year in `sp500_changes`: median **20/year for 2010+** (matching
real S&P 500 turnover of ~20–25), but median **7/year for 2000–2009** — roughly a third of
what actually happened.

**Decision.** Treat the change table as reliable from ~2010. For earlier periods prefer
`sp500_membership_intervals`, which is derived independently from revision snapshots and does
not depend on anyone having recorded the event.

Reported as INFO by `check_changes_completeness` — a documented property of the source, not a
defect we introduced, but it must stay visible because it bounds how far back event-driven
analysis can be trusted.

---

## ADR-011 — FRED series tagged for revision risk

**Status:** accepted

**Decision.** Tag each macro series `revised: true/false`.

**Reasoning.** CPI, GDP, payrolls and unemployment are **restated** for months after first
publication. Today's value for March 2020 is not what was on the screen in April 2020, so
using it in a backtest is a genuine look-ahead leak. Market-based series (Treasury yields,
VIX, credit spreads, the dollar index) are final on publication and safe as-is.

7 of the 18 series are flagged.

**Measured 2026-08-27 with an ALFRED key** - the risk is real but narrower than the blanket
warning suggested, and is dominated by *rebasing* rather than genuine restatement:

| Series | period | first print | today | revision |
|---|---|---:|---:|---:|
| GDPC1 | 2022 Q1 | 19,731.1 | 21,932.7 | **+11.2%** |
| INDPRO | 2020-04 | 91.3 | 84.6 | **-7.4%** |
| PAYEMS | 2020-04 | 130,403 | 130,426 | +0.02% |
| CPIAUCSL | 2022-06 | 295.3 | 295.0 | -0.13% |
| UMCSENT | 2020-04 | 71.8 | 71.8 | 0.00% |

**Refined guidance.** Chain-weighted *level* series (GDPC1, INDPRO) are re-indexed to a new
base year periodically, which shifts the whole history by ~10% - a model reading levels sees
a number that never existed at the time. Growth rates computed from those series are largely
unaffected. Count and index series (PAYEMS, CPIAUCSL, UMCSENT) revise by well under 0.5%.

So: prefer growth rates over levels for revised series, and treat GDPC1/INDPRO levels as
unusable without vintages. A FRED API key is now present in .env and ALFRED vintages are
verified accessible (`realtime_start`/`realtime_end` on the observations endpoint), so
proper point-in-time macro is unblocked - it just is not built yet.

---

## ADR-012 — Stooq dropped as a cross-validation source

**Status:** accepted

**Context.** The plan called for cross-validating prices against Stooq, since two independent
sources catch vendor errors.

**Measurement.** `stooq.com/q/d/l/` now returns a JavaScript bot-detection interstitial rather
than CSV.

**Decision.** Dropped. Cross-validation currently relies on the vendor's own adjusted-close
column as an independent second opinion (ADR-006), which is weaker but real.

**Revisit at Phase 3:** once a paid feed exists, cross-validate paid vs yfinance on the ~655
overlapping tickers. That is a genuinely independent comparison and is the better check
anyway.

---

## ADR-013 — Cache keys must encode the request, not just the position

**Status:** accepted

**The bug.** Price downloads are chunked, and each chunk was written to bronze as
`bars_chunk_001.parquet`, keyed on the chunk **index** and ingest date alone. An earlier
8-ticker smoke test (`--limit 8 --start 2024-01-01`) wrote that filename. The subsequent full
973-ticker run found the file present, treated it as a cache hit, and reused it — **silently
dropping 32 securities and truncating 8 others to 2024 onward.**

Nothing errored. `verify` passed, because the artifact matched its own checksum perfectly. The
manifest was internally consistent and completely wrong.

**How it surfaced.** Not by any check aimed at it. Testing `prices_clipped_to_membership()`
showed AAPL with 665 bars starting 2024-01-02 instead of 6,702 starting 2000-01-03 — a number
that was obviously wrong to a human eye scanning output for a different purpose.

**Decision.** Cache filenames encode a hash of the request that produced them:
`bars_chunk_001_{sha256(start|end|tickers)[:10]}.parquet`. Different parameters yield a
different filename, so a partial run can never satisfy a fuller one. On cache hit the loaded
ticker set is additionally checked against the requested chunk.

**Cost of the bug:** 22 securities, ~162k bars. Coverage was under-reported as 67.3% when the
true figure was 69.6%.

**The general lesson.** A cache key must be derived from *everything that determines the
response*. Position, index, and date are attributes of the request but do not identify it.
Checksums verify that a file is what it was when written — they say nothing about whether it
is the right file, so integrity checking is not a substitute for correct keying.

---

## ADR-014 — EODHD free tier spent on universe metadata, not prices

**Status:** accepted

**Measured free-tier limits** (probed, not read off a pricing page):

| | |
|---|---|
| Daily calls | **20** (`dailyRateLimit`), plus a separate `extraLimit` of 500 |
| History depth | **~1 year.** `from=2000-01-01` returns 251 bars starting 12 months back |
| `/user` endpoint | **Free** — does not increment `apiRequests`, so budget can be polled at no cost |
| `/eod/{ticker}` | 1 call |
| `exchange-symbol-list` | 1 call |

**Decision.** Do not spend free-tier calls on price history. Twenty tickers x one year
is worthless beside the 677 tickers x 26 years already held from Yahoo. Spend them on
**universe metadata** instead: the active and delisted US symbol lists are one call
each and together describe 111,032 securities.

**Result — this is the purchase decision, quantified.** Of the 295 S&P 500 ever-members
Yahoo cannot supply, **281 (95.3%) are present in EODHD's delisted symbol list.** Two
calls answered whether the annual fee is worth paying, and the answer is yes.

The 14 genuinely absent are nearly all pre-2010 renames or non-common-stock artifacts
(WPO→Graham Holdings, FD→Macy's, MHFI→S&P Global, ACE→Chubb, KRFTV a when-issued line,
SGPPRB a preferred). Those are ticker-change cases to resolve via the security master,
not missing companies.

**Budget guard.** Every call routes through `_spend()`, which polls the free `/user`
endpoint and refuses rather than overruns, keeping one call in reserve. A cached
response costs nothing and skips the check entirely. This matters more under the paid
burst-buy plan than it does now: during that one month, a runaway loop is lost data.

**Still outstanding:** the price convention is UNVERIFIED. EODHD documents OHLC as raw
but volume as split-adjusted — a *mixed* convention that neither existing
`SOURCE_CONVENTION` value models. `verify_price_convention()` is written but returns
`inconclusive` on the free tier, because the NVDA 2024 split predates the 1-year
window. It must pass before EODHD volume is trusted anywhere.

---

## ADR-015 — Header words are not tickers

**Status:** accepted

**The bug.** `SYMBOL` was ingested as an S&P 500 constituent across 10 monthly snapshots
(2023-11 to 2024-08). The wikitext parser reads the ticker column positionally; when a
header row escaped row-detection, the literal word "SYMBOL" matched the ticker regex
(`^[A-Z][A-Z0-9]{0,5}$`) and entered the universe as a company.

**How it surfaced.** Not by any internal check — every internal check passed, because
"SYMBOL" is a structurally valid ticker. It appeared only when cross-referenced against
an external vendor's 111,032-symbol universe and matched nothing.

**Decision.** An explicit `_NOT_TICKERS` blocklist of header and boilerplate words.

**The general lesson.** Format validation cannot catch a value that is well-formed but
meaningless. Only comparison against an independent source can. This is a concrete
argument for holding two overlapping sources even when one is authoritative — the
second one's job is to disagree.

---

## ADR-016 - Strategy mandate: long-only, monthly, sub-$100k

**Status:** accepted (owner decision, 2026-08-27)

**Decision.** The backtest engine targets a long-only mandate, rebalanced monthly, at a
capital scale below $100k.

**Why each choice matters more than it looks:**

**Long-only.** Removes borrow cost, locate risk, and the need to ingest FINRA
short-interest data. Portfolio construction reduces to non-negative weights summing to
one. Most published equity factor research is replicable in this form.

**Monthly rebalance.** This is the load-bearing one. Our membership reconstruction dates
an index add/remove to within a *month* (ADR-001), which was the single largest accuracy
limit of the free constituent history. A strategy that only acts at month boundaries
cannot be harmed by sub-month dating error, so **the limitation stops mattering**.
Monthly also yields ~230 rebalance dates over 2007-2026 - enough observations for
walk-forward validation with purging.

**Sub-$100k.** Market impact on S&P 500 large caps is negligible at this size, so the
cost model reduces to commission plus half-spread and needs no ADV-scaled impact term or
capacity constraint. This matters because we have no quote data and cannot afford any:
the half-spread must be *estimated* from volatility and dollar volume. At institutional
size that estimate would be the dominant source of error; at retail size it is a rounding
detail.

**Consequence.** The cost model becomes tractable without quote data - which is the only
reason a credible backtest is possible on this budget at all.

**Still required despite the simplification:** delisting returns. A long-only monthly
strategy can still be holding a name when it is acquired or goes bankrupt mid-month, and
the position must resolve to a real outcome rather than silently vanishing. See HANDOFF
section 4.
