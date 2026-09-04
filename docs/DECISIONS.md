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

---

## ADR-017 — Leakage is prevented structurally, not by convention

**Status:** accepted (2026-08-27)

**Decision.** A strategy never receives the price panel. It receives a `PanelView` whose
numpy arrays are slices ending at the as-of session, and `Context` keeps no reference to
the parent panel.

**Why.** The alternative — hand over the full panel and document that strategies must
filter by date — makes correctness depend on every future author remembering a rule. That
is not a mechanism, and a genetic algorithm searching thousands of variants is precisely
the situation where one forgotten filter becomes a spectacular result nobody questions.

`panel.adj_close[:t+1]` is a view: O(1), no copy, and the array has exactly `t+1` rows.
Indexing row `t+1` raises `IndexError` because the memory is not in the object. The future
is not filtered out; it was never present.

**Three escape routes, all closed and tested:** indexing past the end raises `IndexError`;
`ctx.price_on()` on a future date raises `LookaheadError`; reaching for `ctx.view.panel`
raises `AttributeError`. Acceptance check 3 runs a deliberately cheating strategy for each.

**Cost.** `ctx.prices` (a pandas DataFrame) is offered for readability and is ~1000x
slower. It is documented as unusable inside a fitness function.

**Consequence.** Leakage becomes a class of bug that cannot be written, rather than one
that must be caught in review.

---

## ADR-018 — Buy-and-hold SPY is the engine's calibration instrument

**Status:** accepted (2026-08-27)

**Decision.** Before any strategy exists, the engine must reproduce SPY's actual total
return. Measured: **8.32%/yr over 2000-01-03..2026-08-26, matched to 0.2 basis points.**

**Why this specific test.** It is unusually diagnostic because the three ways the
adjustment chain can be wrong land on three separated numbers:

| Result | Diagnosis |
|---|---|
| ~6.43%/yr | dividends dropped — this is the price return |
| ~8.32%/yr | correct |
| ~10.2%/yr | dividends counted twice |

You cannot land on 8.32% by accident. A suite of unit tests on the accounting could all
pass while the adjustment chain was inverted; this cannot.

**Implementation note.** The `benchmarks` table stores raw OHLCV plus discrete dividend
and split events but no adjustment factors, because `normalize/adjustments.py` runs over
`daily_bars`. `backtest/benchmark.py` computes them with the *same* `compute_factors`
function rather than a second implementation, so the two cannot drift apart.

**Also load-bearing:** benchmarks are total-return adjusted. Comparing a total-return
strategy against a price-return benchmark hands every strategy 1.9pp/yr of free apparent
alpha.

---

## ADR-019 — Constant adjusted shares; no separate dividend accrual

**Status:** accepted (2026-08-27)

**Decision.** The engine holds a constant number of *adjusted* shares between rebalances
and never credits cash dividends.

**Why.** `adj_close` already reinvests dividends via `adj_factor` (ADR-006), so constant
adjusted shares reproduces the dividend-reinvested total return by construction. Adding
cash dividends on top double-counts them — worth ~1.9pp/yr on SPY, enough to make a bad
strategy look good.

**The trap this creates, and how it is handled.** "Shares" in the engine are therefore
notional, not the count on a brokerage statement. A per-share commission needs real share
counts, and our stored close is split-adjusted (ADR-007), so it is not the price that
traded. Real counts are recovered at execution through the cumulative split ratio:

    as_traded_price(t) = raw_close(t) x product of split ratios with ex-date > t

`adj_factor_price` cannot do this job — under the `split_adjusted` convention it is 1.0
everywhere by construction. `normalize/splits.py` holds the shared implementation, which
TODO-5 (point-in-time market cap) also needs.

**Guard.** Acceptance check 5 compares the engine's total return against the same
portfolio valued on a price-only series and requires a gap of one dividend stream
(measured: 2.26%/yr). A future "fix" that credits dividends would roughly double it.

---

## ADR-020 — Half-spread = max(Corwin-Schultz, tick floor)

**Status:** accepted (2026-08-27)

**Decision.** The proportional half-spread is the greater of a Corwin-Schultz (2012)
estimate and a floor derived from the minimum tick.

**Why not the estimator alone.** Corwin-Schultz was built for a market with 50-100bp
spreads; its sample is 1927-2006. For an S&P 500 mega cap in 2018 the true spread is
1-2bp, more than an order of magnitude below the estimator's resolution, so it correctly
returns ~zero. Zero is closer to right than the alternative but still wrong: a spread
cannot be narrower than one tick.

**The averaging convention is not a detail.** Roughly half the two-day estimates come out
negative for a liquid name. The common convention truncates each to zero and then
averages, which computes E[max(X,0)] — a pure positive bias when the true spread is near
zero. **Measured: that convention reports 36bp for AAPL in 2018-19 against a real quoted
spread of about 1bp.** This project averages the signed estimates and truncates the
average.

**The floor.**

    half_spread >= MIN_SPREAD_TICKS x tick_size(date) / 2 / as_traded_price

`tick_size` is $0.01 after decimalisation (2001-04-09) and $0.0625 before it, making the
pre-2001 cost regime genuinely different rather than assumed away. `MIN_SPREAD_TICKS =
2.0` is the single modelling assumption, stated in the module rather than buried.

**Verification.** AAPL 2024 0.52bp, MSFT 2024 0.24bp, KO 2024 1.68bp; 2008-09 wider than
2010-15; pre-decimalisation wider still. The floor binds 63% of the time overall, and the
estimator takes over where it can resolve — 2008-09 and the illiquid tail.

**Rejected alternative:** a flat basis-point assumption. It cannot express the 2008-09
widening or the decimalisation regime change, both of which are large and both of which
this reproduces.

**Bias direction, deliberately.** Where the two disagree the result errs high.
Overstating costs makes strategies look worse, which is the safe error for a research
platform.

---

## ADR-021 — Delisting outcomes are recorded assumptions, not measurements

**Status:** accepted (2026-08-27)

**Decision.** Build `gold/backtest/delisting_returns` by parsing the free-text `reason` in
`sp500_changes` into three categories, and record the assumption for every row in words.

| Category | Return | Rationale |
|---|---|---|
| `index_removal` | 0 | Still listed; sold at the next open |
| `acquisition` | 0 | Deal terms unknown; approximated at the last traded price |
| `bankruptcy` | -1.0 | Equity wiped out |

**Why it must exist.** Without it, a holding whose price series ends silently vanishes
from the accounting and the outcome is never booked. That is survivorship bias running
backwards — deleting the losses rather than the losers — and it is worse than the usual
kind because nothing errors and no row is missing.

**Why now rather than with the paid feed.** Only 4 cases are visible today because Yahoo
carries almost no delisted names. With full coverage there would be hundreds. Buying the
feed *exposes* this problem rather than solving it.

**Price beats prose.** Two rows classified as bankruptcy still had prices 90+ days later
and were reclassified as removals. PG&E's 2019 Chapter 11 is the instructive case: it
filed, and its equity was not wiped out. The price series is stronger evidence than the
wording.

**What this is not.** It is not a delisting-return dataset. CRSP has one and it costs
orders of magnitude more than this project's budget. 125 of 518 securities are
`unresolved` and default to an index removal — the conservative choice for the common case
and the wrong one for a bankruptcy — so the count is reported per run. Coverage of
`sp500_changes` is poor before 2010 (ADR-010), which is where most of them are.

The goal is an explicit assumption a reader can disagree with, not a silent one.

---

## ADR-022 — A 45-day exit buffer on the membership clip

**Status:** accepted (2026-08-27)

**Decision.** `prices_clipped_to_membership()` gains an `exit_days` parameter, default 0,
which the backtest panel sets to 45.

**Why a forward extension at all** — forward data being exactly what the clip exists to
exclude. A name dropped from the index in month M is still in `universe_asof(M-end)`, so a
monthly strategy only sells it at the M+1 rebalance, up to ~35 calendar days later. With
no buffer that bar does not exist and a live company is booked as a delisting.

**Why 45 specifically — measured, not chosen.** Of 207 closed membership intervals that
have prices, **203 keep trading past 45 days and exactly 4 stop.** Those 4 are the genuine
delistings Yahoo carries. So 45 days separates "removed from the index, still listed" from
"actually delisted" cleanly.

**Why it is safe against ticker recycling.** A reassigned symbol requires the old company
to delist and a new one to adopt it, which takes far longer than 45 days.
`quality/checks.py::check_ticker_recycling` uses a 365-day window; the buffer is an order
of magnitude below it. 187 of the 207 names have prices continuing past a year — that
population is what the clip is for, and the buffer does not touch it.

**Scope.** The buffer buys the sell fill and nothing else. Contexts still end at the as-of
date (ADR-017), so no strategy can see into it.

---

## ADR-023 — Price coverage is reported on every run

**Status:** accepted (2026-08-27)

**Decision.** Every backtest reports the fraction of the true point-in-time index that
actually had prices, measured against `sp500_membership_intervals` — not against the
panel's own columns.

**Why.** The point-in-time universe is essentially complete; the prices are not. Yahoo
does not carry most companies that later delisted, so a 2007 backtest trades a 273-name
subset of a 470-name index — **54.7% coverage**, rising to 100% today. 343 index members
have no usable price history at all.

That subset is not random. It is the survivors. So the coverage gap is a *second*
survivorship bias sitting underneath the point-in-time universe this project was built to
construct. The membership data is honest about who was in the index; the price data is not
honest about who could be bought.

**The denominator is the decision.** An earlier implementation counted panel columns
flagged as index members, which reported ~96% coverage in 2008 because securities with no
prices are not columns at all. It measured "of the names we have prices for, how many have
prices". Counting against the membership table makes the missing third visible.

**Consequence.** `--min-coverage` refuses to run below a threshold. It defaults to 0 —
reporting rather than refusing — because a 2007 start is often still the right choice; it
just must be a decision rather than a surprise. Closing the gap is TODO-8.

---

## ADR-024 — Ranking ties break on a stable hash, never on security_id

**Status:** accepted (2026-08-27)

**Decision.** When two securities have equal scores, `portfolio._top_k` orders them by a
blake2b hash of the `security_id`, not by the id itself.

**The bug this fixes, found by accident.** The first version broke ties lexicographically
by `security_id`. That is deterministic — which is what acceptance check 4 tests — and it
is also badly biased.

`security_id` is assigned by the registry in the order securities are first observed
(ADR-005), and that order correlates strongly with survival. Measured on the current
panel:

| Half of the security_id range | Still priced at the panel end |
|---|---:|
| Low | **99.0%** |
| High | **61.1%** |

So sorting ties by id hands every tie to a survivor.

**How it surfaced.** `EvolvedBlend` was first given a default genome of every gene's
midpoint. The midpoint of a signal weight bounded on [−1, 1] is **zero**, so all four
weights were zero, every score was identical, and the entire portfolio was chosen by the
tie-break. That strategy — which by construction has no opinion about anything — posted
**17.65%/yr at a Sharpe of 0.89**, beating every honest baseline and beating buy-and-hold
SPY.

It looked like a discovery. It was the tie-break selecting survivors.

**Why this is the archetypal failure for this project.** Nothing errored. The universe was
correctly point-in-time. The prices were correctly adjusted. Execution was at the next
open. Every acceptance check passed, including determinism — the run *was* perfectly
reproducible, it was just reproducibly wrong. Survivorship bias had been designed out of
the universe and walked back in through a sort key.

It is also the exact failure mode a genetic algorithm is built to find. A GA searching
over signal weights would drive them toward zero, discover that "no signal" scores
brilliantly, and converge on it.

**The fix.** `stable_tiebreak()` hashes each `security_id` with blake2b and orders on the
result. Deterministic across runs, machines and Python versions — the builtin `hash()`
is salted per process and would break reproducibility. Uncorrelated with listing order:
after the change the two halves survive at 81.2% and 79.0%.

Computed once per panel and carried through `Context.tiebreak`, so it costs nothing per
rebalance.

**Guarded by** three tests in `tests/test_backtest.py`: the hash is reproducible, its
ordering is uncorrelated with id order, and an all-tied score does not select the bottom
of the id range.

**The general lesson, which is bigger than the fix.** *Deterministic* is not the same as
*unbiased*, and a tie-break is a modelling choice. Anywhere this codebase imposes an
arbitrary order — a sort key, a `keep="first"`, a dictionary iteration that reaches data
— ask what that order correlates with. Here it correlated with survival, which is the one
thing the whole repository exists to keep out of the numbers.

---

## ADR-025 — 2022-01-01 onward is a holdout, and looking at it is recorded

**Status:** accepted (owner decision, 2026-08-27)

**Decision.** Everything from 2022-01-01 is reserved for the final test. Backtests default
to `holdout="exclude"` and stop the day before. Reaching into it requires an explicit
mode, and **every such run is appended to a ledger that cannot be disabled.**

**Why now rather than when the search starts.** Run thirty algorithms over 2007-2026, pick
the best, and the whole sample is spent - there is no clean data left to check the winner
against, and no later work recovers it. The cost of freezing the last stretch while still
building the engine is zero. The cost of not having frozen it is total and permanent.

**Why a constant and not a parameter.** A holdout you can move is not a holdout. Moving it
forward silently converts test data into training data, and the move would be invisible in
any result that quoted it.

**Why 2022-01-01.** Leaves ~176 months to research in - enough for walk-forward with
purging and an embargo - and ~56 months held out, enough to be a real test. It also
contains a regime the research window does not: the 2022 drawdown and the rate cycle.

**Three modes.** `exclude` (default, research window), `include` (full history), `only`
(the holdout alone - the final test). The last two are looks.

**The asymmetry.** Trial logging can be switched off for a scratch run
(`SP500LAB_REGISTRY=off`); the holdout ledger cannot. You may run something without
recording it as a trial. You may never look at the holdout without leaving a trace. Each
look degrades it, and the ledger is the only way to know how far.

**A bug this exposed immediately.** With `end` capped, the engine's final holding period
still ran to the end of the panel, so a `holdout="exclude"` run carried straight through
the reserved data anyway. Runs are now bounded by an `end_row`, and a rebalance on the
last session of a window is dropped rather than executed - its fill would be at the next
open, which is outside the window.

**Consequence.** Every documented baseline figure now covers 2007-05..2021-12 rather than
the full history, and the rankings change with it: over the research window `equal_weight`
beats SPY on return and `low_vol` beats it on Sharpe, neither of which was true over
2007-2026. That difference is the argument for the holdout, not against it.

---

## ADR-026 — Every backtest is logged as a trial, by default

**Status:** accepted (2026-08-27)

**Decision.** `run_backtest` appends to an append-only registry unless explicitly told not
to. Runs are grouped into named *studies*, and `n_trials` for the deflated Sharpe is the
count of **distinct configurations** within a study.

**Why it must be the default.** The deflated Sharpe (Bailey & Lopez de Prado, 2014) needs
two inputs that describe the search rather than the winner: how many configurations were
evaluated, and how much their Sharpes varied. Neither can be reconstructed afterwards. A
discarded idea leaves no trace anywhere else in this repo - not in git, not in the results
directory, nowhere. Opt-in logging would mean the numbers are missing precisely when the
search was informal, which is when the multiple-testing problem is least visible and most
dangerous.

An unlogged trial is not a small loss. It makes the reported Sharpe of the eventual winner
not conservative, not optimistic, but meaningless.

**Distinct configurations, not log lines.** Re-running one configuration is one hypothesis
tested twice; counting it twice would over-deflate and make a real result look worse than
it is. Runs are fingerprinted on strategy class, parameters, construction, dates, cost
model, capital, liquidity floor and seed.

The **data version is excluded** from that fingerprint. Re-running one configuration after
re-ingesting prices is still one hypothesis. The dataset is recorded separately as
`data_fingerprint`, which is for reproducing a run rather than for counting it.

**Studies are the scope, and the scope is the honesty.** The study boundary decides which
trials a winner must beat. Splitting one search across several study names understates
`n_trials` and flatters the result. That takes deliberate effort, which is the best a tool
can do.

**Monthly statistics are stored alongside the daily ones.** The headline Sharpe comes from
the daily equity curve; the deflated Sharpe cannot use it. Its derivation assumes
approximately independent observations, and a monthly-rebalanced portfolio holds a
constant share vector all month, so its daily returns are heavily autocorrelated. Treating
~4,900 daily points as independent when there are ~176 monthly ones would overstate every
Sharpe in the log and make the deflation far too generous. `deflate()` uses the monthly
figures only.

**Rejected alternative:** logging only runs the user chose to save. That is exactly the
winners, and the winners are the one population whose count the deflated Sharpe does not
need.

**Storage.** Append-only JSONL under `data/experiments/`, the same idiom as the bronze
manifest. `_append` heals a truncated final line before writing, because appending onto a
partial record would concatenate the two and destroy the good one along with the broken
one - the ADR-013 failure shape again. Not committed to git: a GA run appends thousands of
lines. Backed up with vault and bronze, because you cannot re-derive what you tried.

---

## ADR-027 — Month-end equity curves are stored with every run

**Status:** accepted (2026-08-28)

**Decision.** `registry.log()` also writes each run's month-end equity curve to
`data/experiments/curves.jsonl`, keyed by `run_id`, rebased to 1.0, with `nav_gross` and
`benchmark` alongside when they exist.

**The gap this closes.** The registry recorded summary statistics and nothing else, so a
run's equity curve survived only if `--save` was passed by hand. A report could therefore
show a leaderboard but could not plot any historical run without re-running it. This was
found by asking what a frontend would need, before building one - which is the argument
for designing the frontend early even when there is little to look at yet.

**Why monthly rather than daily, measured rather than guessed.**

| Stored per run | Size | 10,000 GA individuals |
|---|---|---|
| Daily curve | ~30 KB | ~300 MB |
| Month-end, three series | ~7 KB | ~75 MB |

The strategy only trades at month ends (ADR-016), so month-end is where the information
is. Nothing a comparison chart shows is lost, and it is the same frequency the deflated
Sharpe already uses for the reason given in ADR-026.

**Why a separate file.** `runs.jsonl` is the searchable index and `load()` parses all of
it. Putting curves inline would make every query parse ~7 KB per run it does not use. The
split keeps the index lean and loads curves only when something asks to draw one.

**Escape hatch.** `run_backtest(..., log_curve=False)` for a large search. Re-running a
winner afterwards recovers its curve, and because the fingerprint does not change, that
re-run is the same trial rather than a new one - so recovering a curve cannot inflate
`n_trials`.

**A gross curve identical to the net one is skipped** rather than stored twice; that is
every zero-cost run.

---

## ADR-028 — Reports are static self-contained HTML, and the view layer is pure

**Status:** accepted (owner decision, 2026-08-28)

**Decision.** `sp500lab report` writes one self-contained HTML file per report - inline
stylesheet, inline script, inline SVG, no network requests. Behind it, everything up to
and including `views.py` produces no markup.

**Why static rather than a served application.** The project's constraints are a $20/month
budget, one person, no cloud, and "the whole thing fits on a USB stick". A report that
needs a running process to be read does not survive those constraints, and neither does
one that fetches a charting library from a CDN. A file that opens offline in five years,
sits next to the commit that produced it, and can be emailed, does.

Interactivity is deliberately small - toggle a series, sort a column, hover for a value -
which is what a reader actually does with a backtest report. Anything beyond that wants a
real application, and that is a separate decision to make later rather than a feature to
creep into this one.

**Why hand-rolled SVG.** The chart vocabulary is five primitives of simple geometry, so
emitting SVG costs less than embedding hundreds of kilobytes of library in *every* report.
It also buys exact theme control: every colour is a CSS custom property, which is how one
stylesheet flips a report between light and dark without regenerating it.

**The part that matters more than the charts: the seam.** A frontend is the most likely
component to be rewritten. If it reaches into `BacktestResult` internals or parses JSONL by
hand, every rewrite risks quietly changing what a number means. So `series.py`,
`tables.py` and `views.py` are pure - registry rows in, dataclasses out - and `specs.py`
describes *what* to draw while `render/` decides *how*. Replacing the presentation layer
means reimplementing `render/` and nothing else.

The test suite is the evidence this holds: 47 tests, almost all asserting on numbers, and
only a handful touching markup at all.

**A bug this immediately caught.** `theme.direction` first inferred a metric's direction
from its name and concluded that lower drawdown is better. Drawdown is stored negative, so
the scoreboard highlighted -77.65% - the worst result in the table - as the best in its
column. It read plausibly in code and was visible only on screen. The direction map is now
explicit, unknown metrics return "no direction" rather than a guess, and
`test_worst_drawdown_is_never_the_best` pins it.

**Editorial rule, enforced in `views.py`.** Every report carries an honesty section, and
the deflation panel sits beside the scoreboard rather than behind a link. A report showing
a 12% CAGR that mentions on page four that half the index was untradable has already
misled its reader.

**Consequence.** `reports/` is gitignored - a report rebuilds from the registry in under a
second, and a 100 KB file per run would bloat the repo. The registry is the artifact worth
backing up; the report is a view of it.

---

## ADR-029 — Every order is written down, and unfilled orders are not charged

**Status:** accepted (2026-08-28)

**Decision.** `run_backtest` records an order-by-order ledger by default
(`result.trades`), carrying the AS-TRADED price and real share count alongside the
adjusted figures the accounting used. `sp500lab backtest trades <strategy>` exports it as
CSV together with the month-by-month holdings, and prints a reconciliation.

**Why.** An equity curve is a claim; a list of orders is the evidence for it. Somebody who
does not trust this engine cannot check a CAGR — there is nothing in it to check. They can
check "on 2007-05-01 this bought 20.06 shares of AAPL at $99.59", because that is a fact
about the world. Every other honesty mechanism in this repo makes a *failure mode* visible;
this one makes the *result* checkable by a third party, which is a different and stronger
property.

**Why two prices per row.** `price` is `raw_open x cum_split` — what a broker printed that
morning. `adj_price` is the total-return-adjusted number the NAV was computed from. Handing
an outside reader only the adjusted price guarantees a mismatch against any quote site that
means nothing; handing them only the as-traded price makes the ledger unreconcilable
against the curve. Verified against the source bars: APH on 2018-02-01 prints $91.40 where
the stored split-adjusted close is $22.85 (two later 2:1 splits), and HON prints $159.00
against $150.64 (the 2018 spin-offs). Both match the historical record.

**The identity that makes it an audit.** For every rebalance,
`cash_after = cash_before + sum(cash_flow)`, where `cash_flow` is net of the commission and
spread charged to that order. `trades.reconcile()` replays it against the run's own cash
column and reports the worst gap. Measured across all baselines and all twelve alpha
strategies: **0.0**.

**Two things this immediately exposed.**

*Costs charged to orders that did not exist.* Dropping sub-cent "dust" orders from the
ledger broke the cash identity by $1.26 on `equal_weight` — real money charged against
orders nobody could see. Every order the cost model prices now gets a row, however small,
because a cost with no order attached to it is exactly the failure this is meant to rule
out.

*Commission on unfilled orders.* Costs were priced before the fill check, so an order for a
name with no opening bar was charged a commission it never paid. The engine now resolves
fillability first. The effect on published numbers is under 0.01 basis points — the point
is not the magnitude, it is that the ledger made a wrong thing visible in an afternoon.

**Also corrected here.** `_as_traded_price` used the execution day's *close* to derive
share counts for the per-share commission; it now uses the *open*, which is the bar orders
actually fill in and the same price the ledger prints. Measured impact on `momentum_12_1`,
`equal_weight` and `low_vol`: **under 0.005 basis points of CAGR** — the $1 per-order
minimum dominates the per-share rate at this capital scale (ADR-016). One price, one story.

**Cost.** About 12,000 rows for a 50-name strategy over 232 rebalances. Recording is on by
default for ordinary runs and off inside the genetic algorithm; re-running a winner with it
on is the *same trial*, since the fingerprint does not include it.

---

## ADR-030 — A versioned feature layer, and a leakage test that can fail

**Status:** accepted (2026-08-28) — this is TODO-4

**Decision.** `data/gold/features/` holds an `(R, S, F)` float32 cube of 75 point-in-time
features on the month-end rebalance grid, built once and consumed by every strategy through
`ctx.feature(name)`. `sp500lab features check` rebuilds the whole matrix from a panel that
physically ends at a past date, with every filing published after that date deleted, and
asserts the earlier rows are **bit-identical**.

**Why one shared layer.** The stated goal is a competition between genetic algorithms,
neural nets and classical rules. If each competitor computes its own momentum, the
competition partly measures who wrote better feature code. It is also a hard performance
requirement for the GA: a fitness evaluation must never recompute a rolling regression.

**Why the month-end grid and not every session.** A daily grid of 75 features over 628
securities is 590 MB; the month-end grid is 17 MB. Nothing in a monthly-rebalanced strategy
can use the rows in between, and `at()` refuses a row it does not hold rather than
interpolating one — a silently interpolated feature is a lookahead bug wearing a
convenience API.

**The leakage test is the whole argument.** Two mechanisms fail differently: a price
feature can have a forward-looking window, and a fundamental can be joined on `period_end`
instead of `filed_date`. Truncating the panel catches the first; deleting later filings
catches the second. All 75 features currently pass bit-identically at a 2016-12-30 cut.

**Two bugs it caught, both silent.**

*The as-of join collapsed a decade into one key.* Dates were converted to integers by
dividing the int64 view by 86.4e12, which assumes nanosecond resolution. pandas 2.x parses
these strings to `datetime64[us]`, so every date in the dataset mapped onto one of about
twenty integers and `merge_asof` matched filings almost at random. Nothing raised;
fundamental coverage was 1.5% instead of 75% and would have read as "XBRL is sparse".

*The split basis belongs to the filing, not to today.* Reported shares are in the share
basis of the filing that carried them, so `market_cap(t) = shares × cum_split(filed) ×
raw_close(t)`; `cum_split(t)` cancels out of the algebra entirely. Using the as-of ratio
instead is correct except across a split between the filing and now — a window of at most
one quarter, in which the error is the whole split ratio. Apple's 4:1 in August 2020 would
have quartered it for a quarter, and the fix makes the series continuous across it
($1.82T July, $2.21T August, $1.98T September).

*Market cap, from partial share counts.* 0.79% of index-member observations computed to
under $500M — Simon Property at $1.7M, Fox at $62 — because a filer tags one share class or
a treasury context on its cover page. The 1st percentile of everything above the threshold
is $2.2bn, so there is a clean gap rather than a continuum. Those are now discarded and
counted. It matters because market cap is a *denominator*: a 10,000-share Simon Property
has a book-to-market of 300, and any value strategy buys it first.

**Deliberate simplifications, stated rather than hidden.** Flow quantities use the ANNUAL
XBRL duration rather than a reconstructed trailing twelve months — annual is what Sloan
(1996) and Novy-Marx (2013) used, and rebuilding fiscal Q4 as `annual - nine-month YTD` for
every fiscal calendar is a sign error away from a feature that looks like alpha. Only
unrevised daily macro series are used (ADR-011); the seven revised ones wait for TODO-7.

**Consequence.** Fundamental features begin 2010 and cover 649 of 973 historical index
members, disproportionately survivors. Strategies that need them declare a `min_date` and
run on a shorter, kinder window — which is why `backtest suite` scores every strategy
against the index over *its own* dates.

---

## ADR-031 — The genetic algorithm searches weighted rank blends, not expression trees

**Status:** accepted (2026-08-28)

**Decision.** The GA's search space is a weighted sum of cross-sectionally *ranked*
features plus a small portfolio-shape gene set: 19 genes for the price preset, 29 with
fundamentals. No expression trees, no evolved arithmetic, no products of indicators.

**Why the smallest space that can still express an idea.** An unconstrained search over
indicator combinations finds something that works beautifully in-sample every single time.
That is not a risk, it is arithmetic — the maximum of N draws grows with N whether or not
there is signal. Four structural defences, chosen because a statistical correction applied
to an unbounded search corrects a number that was never meaningful:

- **Ranks, not raw values.** One bad share count cannot dominate a portfolio; it is worth
  exactly "first place".
- **A dead zone.** Weights under ±0.10 are exactly zero, so "how many features does this
  use" has an answer, parsimony pressure has something to grip, and two behaviourally
  identical genomes deduplicate.
- **Curated presets.** 13 features, or 23 with fundamentals — not all 75. Every extra
  feature multiplies the space and the trial count the deflated Sharpe must discount.
- **Every individual is readable.** `describe_genome` prints a winner as sentences. A
  winning parameter vector nobody can read is a winning vector nobody can check.

**Fitness is not return.** Handed raw return, a long-only GA finds leverage as
concentration: `top_k` at its floor, ten names, a magnificent CAGR and a 70% drawdown. The
default objective is the **monthly** Sharpe — the same quantity `registry.deflate()` uses,
so the search and its own significance test look at the same number — aggregated across
folds as `mean - 0.5 x std`, with optional turnover and per-feature complexity penalties.

**The behavioural fingerprint is the trial count.** The evaluation cache and the registry's
`n_trials` key on the same rounded, dead-zoned vector. Counting behaviourally identical
individuals twice would over-deflate the winner; not deduplicating at all would report a
small trial count for a large search and under-deflate it.

**Speed.** ~0.15s per evaluation, so 1,500 individuals is about four minutes. Three things
buy it: the panel is memoised, feature ranks are precomputed once for the whole population
(the ranks depend on the date and the tradable mask, not on the genome), and each
individual is one backtest whose folds are sliced from the resulting curve rather than
re-run. The pre-ranked columns are named `<feature>__rank` so a strategy handed the wrong
panel fails at the first rebalance instead of silently summing raw ratios.

**What is deliberately absent.** Multi-objective (NSGA-II) selection, island models,
adaptive operator rates. All are reasonable and none address the failure mode that actually
threatens this project, which is not slow convergence — it is converging beautifully onto
noise.

---

## ADR-032 — Folds measure consistency, not out-of-sample performance

**Status:** accepted (2026-08-28)

**Decision.** GA fitness defaults to the metric computed on each of four contiguous
sub-periods of the research window, separated by a one-month embargo, aggregated as
`mean - 0.5 x std`. The folds are sliced from the single equity curve each individual
already produced.

**Why folds at all.** A strategy that made all its money in 2009 and nothing since has a
fine full-sample Sharpe. Fold consistency separates that from a strategy that worked in
every sub-period, and that distinction is most of what separates a discovery from a
coincidence on nineteen years of monthly data.

**Why contiguous with an embargo, never random K-fold.** Financial data is autocorrelated
and a shuffled split leaks across the boundary. The embargo is one month because that is
one holding period under ADR-016 — exactly the span over which a position opened in one
fold is still held in the next.

**What this is NOT, stated plainly because the name invites the confusion.** Every fold is
inside the research window and every individual is selected using all of them. A good fold
score is evidence of consistency, not of generalisation. The out-of-sample test is the 2022
holdout (ADR-025), it is looked at once, and looking is recorded.

**Why not a true walk-forward, refitting per fold.** Because an individual here is not
fitted to anything — it is a parameter vector, and *the search* is the fitting procedure. A
genuine walk-forward means re-running the entire GA inside each training window and
evaluating its winner on the next. That is the right next step and it costs one full search
per fold; it is deliberately not pretended at with a cheaper mechanism that would carry the
same name.

**Cost.** Slicing a curve is free. Four folds add no backtests, which is the only reason
the default can afford to be the robust option rather than the fast one.

---

## ADR-033 — A forward test is a pre-registered, paired comparison, not a later backtest

**Status:** accepted (2026-08-28)

**Decision.** Out-of-sample evaluation after the research window gets its own package
(`src/sp500lab/forward/`) and its own command tree. A forward test always runs **two**
backtests — the research window as the prediction, the forward window as the outcome —
reports the difference next to the standard error of that difference, and requires a
**seal**: a record of what was predicted, written before the look.

**Why not just `backtest run --holdout only`.** That already exists and it is the wrong
shape for the job in three ways.

*It reports a number, not a comparison.* "Sharpe 0.49 over 2022–2026" is not a result.
The result is that the research window said 0.69 and the forward window said 0.49, and
that the standard error on that difference is 0.56 — which is the whole story and none of
it is in a single-run summary.

*It has no memory of the prediction.* By the time anybody reads a forward figure, the
research figure it was supposed to beat is a row in a 4,000-line trial log.

*It cannot see the selection problem.* The holdout stops a strategy from being FITTED to
2022 onward. It does nothing about **choosing** what to test after seeing how it did, and
that is the failure that actually happens. Twenty honest `--holdout only` runs, three
reported, and the holdout is a second research window with no trace of how it became one.

**Pre-registration, with an escape hatch that is recorded rather than closed.** A seal
records the configuration fingerprint, the research-window prediction, the trial count
behind the candidate, and a free-text rationale. `forward seal` writes one and spends
nothing. Requiring it before every run would be the strict design and would be routed
around within a week, so `forward run` **auto-seals** what it has not seen — and records
`seal_mode` as `auto` rather than `declared`. The numbers in an auto seal are equally
clean; what it cannot prove is ordering. That distinction travels on every downstream
table, which is the best a tool can do.

The **earliest** line for a seal id binds. Ids hash the configuration rather than the
clock, so a seal rewritten after a disappointing look cannot replace the prediction it
failed to meet.

**Refutation, not confirmation, and the vocabulary says so.** The forward window holds 54
monthly observations. The standard error of a Sharpe over that span is about 0.47, so a
Sharpe of 1.0 carries a 95% band of roughly [0.06, 1.94] and the smallest difference from
the research window the test could resolve is about 1.35. A window this short can refute a
strategy that fails badly and cannot confirm one that does not. So the verdicts are
`failed` / `decayed` / `held` / `inconclusive`, `held` is documented everywhere as "not
refuted", and `windows.describe_power()` prints the band next to every verdict. Below 24
months no verdict is offered at all.

**The verdict is computed from the z, never from the raw drop.** A Sharpe falling from
1.32 to 0.71 looks decisive and is 1.1 standard errors. Reading the drop directly would
call noise a decay on this window roughly a third of the time.

**Data vintage, because out-of-sample data keeps arriving.** The forward window grows by a
month every month. A second look next year re-reads what it already saw *except* for the
new months, which are genuine fresh evidence. Every record stores the vintage it ran
against, and the next look reports `fresh_months`. The vintage is the last date the
previous look actually saw — the forward leg's own end, not the panel's — because using
the panel's end would retire months no look has ever covered.

**Multiple testing on the forward window itself.** `store.selection_bar()` applies
`metrics.expected_max_sharpe` to the spread of forward Sharpes observed, counted per
candidate rather than per run. Forward runs are also logged under a real study name, so
`experiments deflate forward-test` computes the correction with the registry's own
machinery, and `n_trials` there means the number of candidates the holdout has been asked
about.

**All three cost settings in one look.** costs.py insists all three always be reported,
and this is the one place where fetching a missing one later would cost a second look.
`look_number` counts per (candidate, cost model, mode), so three cost settings of one
strategy are one look at that candidate rather than three.

**Two modes, both one look.** `paired` runs the forward window from a fresh $100k with an
empty book — the honest simulation of reading the research and starting to trade in 2022,
and it pays entry costs the continuous path would not. `continuous` runs one unbroken
backtest and slices it — the honest simulation of having run it all along. Neither is more
correct, they differ by about a month of entry cost, and `mode` is on the record so
neither can be reported without the other being visible.

**Consequence.** `forward window`, `forward seals`, `forward seal` and `--dry-run` are all
free and answer most questions. Only `forward run` and `forward suite` read reserved data,
they announce it before doing so, and they land in the same unsilenceable ledger ADR-025
established.

---

## ADR-034 — The forward record is stored separately, and is enough to rebuild the result

**Status:** accepted (2026-08-28)

**Decision.** Forward tests write three append-only JSONL files under
`data/experiments/forward/` — `seals.jsonl`, `forward_runs.jsonl` and
`forward_curves.jsonl` — none of which `SP500LAB_REGISTRY=off` can silence. The record
carries both legs, the full comparison and the honesty diagnostics, and the stored curves
carry both windows, so a report built years later needs neither the panel nor a re-run.

**Why not reuse `runs.jsonl` and `curves.jsonl`.** Both legs *are* logged there and their
curves *are* stored there; this is not a rejection of ADR-026 or ADR-027. Three things the
trial log cannot do:

1. **It can be switched off.** That is correct for a scratch backtest and wrong for the
   one kind of run that consumes an irreplaceable resource. Same asymmetry as the holdout
   ledger.
2. **The record is a pair.** The quantity of interest is a difference between two runs
   plus the standard error of that difference. It does not fit a `RunRecord`, and
   reconstructing it later would mean re-deriving which two runs belonged together.
3. **The lifecycles differ by three orders of magnitude.** `runs.jsonl` holds 4,000 lines
   and grows by a thousand per genetic search. `forward_runs.jsonl` will hold tens, ever.
   Mixing them buries the second kind.

Measured cost: 5.6 KB per record, 10.4 KB per curve pair, 1.5 KB per seal. At tens
of records that is nothing, and the alternative is re-running a look that cannot be
re-run.

**Structured on disk, flat in a DataFrame.** The two legs are nested under `research` and
`forward` with identical field names, so code can rebuild a `compare.Leg` from either and
one comparison function serves both. `load()` flattens them to `research_*` / `forward_*`
columns because that is what a table wants. Only one of the two shapes is written down.

**The prediction and the re-measurement are both kept.** `research` is the sealed
prediction and is what the comparison uses — that is what pre-registration means.
`research_recomputed` is the research window as it measures at test time, and
`seal_drift_sharpe` is the gap. Non-zero means the data changed under the prediction (a
re-ingest, a restatement, a fixed bug). A large drift invalidates the comparison rather
than adjusting it, and there is no way to notice it without storing both.

**Curves keep the window's opening level, not just its month ends.** A forward window
opening 2022-01-01 first trades 2022-02-01 and its first month end is 2022-02-28. Rebasing
to that month end — which is what ADR-027 does, correctly, for a research run starting at
a month end — would silently drop February from the curve, from the stitched chart and
from the 2022 row of the annual table. That is about 2% of the evidence in a 54-month
window, and 2022 is the most informative year the holdout contains. So the stored grid is
month ends plus the run's own first session.

**Each window is rebased to its own start.** The two legs are independent runs and the
forward one really did begin from a fresh $100k; a shared base would imply a continuity
that did not happen. `stitched_curve()` re-imposes it for charting, marks the join in
`attrs["join_date"]`, and its docstring says plainly that it is a presentation rather than
a simulation.

**No reports in this package.** `store.py` exposes seven pure read functions — `load`,
`scoreboard`, `selection_bar`, `get`, `load_curves`, `stitched_curve`, `annual_table` —
returning DataFrames and dataclasses. The forward reports live in
`reporting/forward_views.py` on the same seam ADR-028 draws for everything else
(ADR-035), and they need nothing beyond those seven calls and nothing from the panel at
all.

**Three small promotions in the registry to avoid a second copy.** `append_jsonl`,
`read_jsonl`, `to_monthly`, `jsonable` and `panel_fingerprint` were private and are now
public. The append helper heals a truncated final line before writing (ADR-026), and two
implementations of a subtle recovery path is how one of them ends up wrong. `to_monthly`
being shared is what guarantees the forward curves and the trial-log curves land on the
same date grid.

**A statistic promoted too.** `metrics.sharpe_standard_error` and
`sharpe_confidence_interval` are new, and `probabilistic_sharpe` was rewritten as
`norm_cdf((SR − SR0) / SE)` over the first of them. The algebra is identical — the PSR
denominator already *was* that standard error — and writing it once means a confidence
interval can never disagree with the probability printed beneath it.

---

## ADR-035 — The forward report set, and a second rendering backend

**Status:** accepted (2026-08-28)

**Decision.** `sp500lab report forward` writes a directory rather than a file: an
executive summary, one technical report per candidate, a cross-sectional decay analysis,
an honesty page, Markdown copies of all of it, and the underlying CSVs. The views live in
`reporting/forward_views.py`, a sibling of `views.py`, and a new `render/markdown.py`
renders the same `Report` objects as text.

**Why a sibling module rather than more of `views.py`.** Size is the small reason.
The real one is that a forward report asks a different question. Every report in
`views.py` asks "what did this strategy do?"; every report here asks "did what it did out
of sample match what the research window predicted, and is the gap larger than the
sampling error of a 54-month sample?" That needs different tables, a different
scoreboard, a different scatter and a different editorial voice. Interleaving them would
blur both, and the file that composes eight reports does not need to compose twelve.

**One caveat outranks the rest and is printed on every page.** `views.py` establishes
that the caveats travel with the numbers. Here there is a single caveat that dominates:
54 monthly observations put a ±0.9 band around an annualised Sharpe of 1.0, so a forward
test can refute a strategy and cannot confirm one. `_power_note()` is therefore not
optional, not at the bottom, and pinned by a test. So is the null-hypothesis note:
`random_weight` encodes no forecast and also "held", which is the clearest available
demonstration that a verdict is a statement about matching a prediction rather than about
quality.

**The cross-sectional report exists because no per-strategy page can contain it.** The
question the whole exercise was run to answer is whether the research window's *ranking*
predicted the forward window's, and that is a property of the set. `rank_agreement()`
computes a Spearman correlation over the index-relative columns — rank rather than
Pearson, because the question is whether the ORDER survived and one outlier would
dominate a linear correlation on twenty points. It is hand-rolled: the project runs on
four libraries and this is a rank, a mean and a covariance.

**Markdown as a second backend, and why it is worth its 250 lines.** ADR-028 claims the
specs-and-views split lets a different frontend reuse every view untouched. Until now
that was an assertion. `render/markdown.py` is it being cashed: the same `Report` objects
become text, and not one line of any view changed. It pays for itself three ways — a
summary that pastes into an email or an issue, a format still readable when nothing
renders today's SVG, and the cheapest possible regression test for the seam, because a
view that started emitting markup would render fine in HTML and be visibly broken in
Markdown.

Charts are described rather than faked. A line chart has no honest Markdown form and an
ASCII sparkline would look like data while being an artefact of column width; bar charts
and heatmaps *are* small grids of numbers, so those are tabulated and lose nothing.

**The raw data ships with the argument.** `data/forward_tests.csv`,
`data/forward_curves.csv` and `data/seals.csv` sit beside the pages. A report is an
argument, and the numbers under it should be separable from it — otherwise the only way
to disagree is to rebuild the pipeline.

**Nothing in the report path runs a backtest or reads the panel.** Every figure comes out
of `data/experiments/forward/`. That is the ADR-034 guarantee being exercised rather than
asserted, and `test_the_stored_record_is_enough_to_rebuild_the_comparison` holds it.

**Only the extremes are labelled on the scatter.** Twenty-two names in one cloud overlap
into a smear, and the chart's job is the shape of the cloud rather than the identity of
every point. The scoreboard directly above it names all of them in order.

**Cost.** About 34 MB for 22 candidates, dominated by the embedded trade ledgers — the
same trade-off `views.MAX_EMBEDDED_TRADES` already makes, and `reports/` is gitignored
and rebuildable in seconds.

---

## ADR-036 — A separate leg engine for calendar rules, pinned to the monthly engine by identity

**Status:** accepted

**Context.** The overnight effect, the weekend effect, turn-of-month and the other
calendar anomalies are claims about WHEN the market pays, at sub-day granularity: a
position entered at one close and exited at the next open. The monthly engine cannot
express that, and its central invariant — a signal formed at the close fills at the NEXT
open (ADR-017) — exists to stop a signal trading on information from its own fill.
Loosening that invariant to admit close fills would re-open the exact hole it closes.

**Decision.** A second, deliberately small engine (`timing/`): every session has two
tradable legs (close→next open, open→close), a strategy is two boolean vectors over the
sessions, and the walk toggles one bit of state at two checkpoints per session. Close
fills are legitimate here for one stated reason: a calendar rule has no signal — the
schedule was knowable years in advance — so there is no information to leak. The one rule
that reads data (the VIX gate) conditions on the PRIOR session's close, and any future
rule that conditions on same-session data must move its entry to the next open.

**What is shared rather than rebuilt:** the adjustment chain (`compute_factors`, the same
function the benchmark uses, so the dividend lands in the overnight leg exactly where the
ex-date puts it), the cost model and its three settings, the metrics, the experiment
registry, the holdout clamp and its unsilenceable ledger, and `BacktestResult` itself —
so a calendar rule gets a registry row, a seal and a forward record through the same
machinery as every other competitor (the forward engine gained a `runner` parameter for
exactly this).

**The engine earns trust by identity, not review.** Two acceptance checks, both exact:
(1) both legs always on at zero cost reproduces the adjusted SPY series to **0.00 bp/yr**
— the same series ADR-018 calibrates the monthly engine against; (2) the overnight-only
and intraday-only strategies PARTITION buy-and-hold, so their NAV product must equal it
at every session — measured max error **6×10⁻¹⁵**, float noise. The second identity is
what turns "the overnight share of SPY's return" from an estimate into a measurement.

**Measured result, and the point of the whole exercise.** 2007-04→2021-12, gross:
overnight 8.31%/yr (Sharpe 0.71, maxDD −29%), intraday 2.21%/yr (0.22, −47%). Under
realistic costs the overnight rule falls to 3.66%/yr and pessimistic to 0.67%: ~500
round trips a year eat the anomaly at retail size. **The gap between gross and net is
the finding**, and the deflated Sharpe of the best calendar rule against the family's
own 27 trials is 0.69 — nothing in the family survives its own search. The tradable
expression of the overnight fact is therefore the monthly `overnight_momentum` strategy
(ADR-037), which trades twelve times a year instead of five hundred.

**Costs.** SPY has no row in `gold_half_spread`, and Corwin-Schultz cannot resolve a
spread this narrow anyway, so the half-spread IS the ADR-020 tick floor
(`tick_size/price`, ~0.8bp in 2007, ~0.2bp today) — slightly wider than SPY's usual
one-tick market, which errs the direction a cost model should.

---

## ADR-037 — The 2026-08 wave: new families after the holdout was spent, with the contamination written down

**Status:** accepted

**Context.** The reserved 2022+ period was spent on 2026-08-28 (ADR-033: 22 candidates,
one sealed set, 66 looks). The project then added new competitors: five cross-sectional
strategies (`strategies/frontier.py`), a shallow neural net (`learned.py`), nine calendar
rules (ADR-036), four features, and a third GA search. Every number those can produce on
2022-2026 is weaker evidence than the first test was, and no mechanism in the registry
can see why: the author had read the first forward test — knew 2022-2026 was one mega-cap
regime — before choosing what to build.

**Decision.** Build them anyway, and write the contamination down everywhere the numbers
appear rather than pretending the period is still clean. Each new candidate was sealed
with a rationale that discloses the contamination in its own text, forward-tested through
the standard harness, and reported with the caveat pinned beside the verdict. Their only
uncontaminated test is data that arrives after 2026-08 — `forward window` counts it, and
re-testing next year buys exactly the months that have accrued.

**What was added, and why each earns its trial.** Chosen to cover mechanisms the first
twelve hypotheses do not touch, mostly from the project's own ranked plan
(WHAT_TO_BUILD_NEXT.md — ensembling was rank 6, volatility capping rank 2, the shallow
net rank 8):

* `overnight_momentum` — 12-1 momentum computed from overnight legs only (Lou, Polk &
  Skouras 2019). Needs `adj_open` beside `adj_close`, which this panel keeps and most
  data layers throw away. Research window: **11.85%/yr, Sharpe 0.55** against
  `momentum_12_1`'s 6.39%/0.38 over the identical window and construction — the largest
  single-difference improvement in the file.
* `week52_breakout` — George & Hwang anchoring; the GA's price winner already weighted
  `high_52w_ratio` +0.38, so the standalone test was owed.
* `div_month` — Hartzmark & Solomon's dividend-month premium, off the new `div_due_1m`
  cadence feature; only buildable because dividends are stored as discrete events.
* `vol_managed` — the long-only half of Moreira & Muir: exposure `min(1, 1/vol_ratio²)`,
  floor 0.20. **Sharpe 0.67 vs SPY 0.59**, the best new hand-written entrant, and it
  cannot lever up so the published effect is deliberately halved.
* `ensemble_rank` — the rank-average of all twelve hypotheses. The cheapest idea in the
  plan and the honest bar for any future model: it posted 0.54, and several of its own
  members beat it, which is itself a finding about how correlated the twelve are.
* `shallow_mlp` — Gu-Kelly-Xiu shaped: features→32→16→1, seed-ensembled, refit yearly,
  trained only on features ≥90% populated over its own window, labels ending a full
  horizon before the as-of date (RollingRidge's discipline, plus a hard assert).
  Realistic Sharpe 0.38; its study's deflated Sharpe is **0.947 — below the 0.95 bar**,
  and it is reported that way.

**Features** (`FEATURE_VERSION` 2→3): `mom_on_12_1`, `mom_id_12_1`, `on_minus_id_252d`,
`div_due_1m`. All four pass the bit-identical leakage check; the overnight/intraday pair
compose back to `mom_12_1` by construction, and the composition is a test.

**The trial accounting.** Every run landed in named studies (`frontier-1`, `mlp-1`,
`timing-1`, `ga-night-1`) so the deflation has real denominators, and the forward records
raise `store.selection_bar()`'s candidate count for everyone — the honest price of
testing more ideas, paid in public.

**Postscript: the wave's own bug, caught by its own absurdity.** `shallow_mlp`'s first
forward record printed a 2.20 forward Sharpe against a 0.38 research run. Cause: the
model held its fitted nets on the instance, and the forward harness runs six backtests
(two legs × three costs) on one instance — so every leg after the first could score on
nets trained on another leg's future, including a research leg from 2007 scored on nets
fitted through 2026. Nothing errored; the number was just impossible, which is what
surfaced it. The fix is a state reset in `on_start` plus a backward-time refit guard,
pinned by a run-twice-identical test; the model carries a `revision` parameter so the
fix lands on a new fingerprint and a new seal, because the contaminated seal is
immutable by design and the way to retire it is to stop matching it. Revision 2 measures
0.33 forward against 0.38 research — held, mediocre both ways. Lesson for the HANDOFF §7
table: **a strategy that keeps fitted state must reset it per run, because instances
outlive runs** — determinism across seeds is not determinism across reuse.

---

## ADR-038 — Genome presets are immutable; new features mean a new preset

**Status:** accepted

**Context.** The four ADR-037 features are exactly what the GA should get to search —
that was half the point of building them. But an evolved winner is stored as a bare
float vector plus a preset NAME, and decodes by position against that preset's feature
tuple. Appending features to `PRICE_FEATURES` would silently mis-decode every genome in
every existing checkpoint: `ga-price-1-best` would reconstruct as a different strategy
with the same name, and nothing would error.

**Decision.** Presets are append-only as a SET: an existing preset's tuple is frozen the
moment a search has written a checkpoint against it. New inputs get a new preset —
`night` = the 13 price features + the four new ones, `PRESET_MIN_DATE` "" since all are
price/event derived.

**Measured result.** `ga-night-1`: 25 generations × 60, **1,404 distinct trials**, winner
**10.89%/yr, Sharpe 0.95 daily / 1.15 monthly, maxDD −17.9%** vs SPY 10.70%/0.60/−55%
over 2007-04→2021-12; survives pessimistic costs (9.99%); **deflated Sharpe 0.9828**
against the 1,404-trial bar of 0.55. It reads as sentences, and two of them are worth
recording: it ranked `div_due_1m` *negatively* (avoiding predicted-dividend names, the
opposite sign to `div_month`'s hypothesis), and inside the blend it weighted intraday
momentum +0.32 against overnight momentum −0.19 — the reverse of the standalone LPS
result. Partial weights in a 14-feature blend are not standalone claims, and the
disagreement between the search and the papers is exactly the kind of thing the next
years of forward data exist to adjudicate. ADR-032's caveat applies unchanged: fold
consistency is not generalisation, and both prior GA winners decayed out of sample.

---

## ADR-039 — The report data layer is a module; the registry is a package

**Status:** accepted

**Context.** Two modules had grown past the point where their name described them.

`reporting/cli.py` was 1,381 lines and was not a CLI. Roughly a dozen registry queries —
`_monthly_entries`, `_evolved_entries`, `_timing_entries`, `_ga_summary`,
`_study_deflation`, `_forward_lookup`, `_research_row`, `_deflation_from` — lived as
private helpers of command handlers, so the only way to reach the Algorithm Book's data
was to have an `argparse.Namespace`. `cmd_all` did exactly that:
`cmd_algorithms(Namespace(out=..., open_after=False))`, so one page could join the report
set. A command calling another command through its own argument parser is the shape a
missing module makes.

`backtest/registry.py` was 860 lines doing five jobs: JSONL plumbing, run logging, curve
storage, the deflated Sharpe, and the holdout ledger. `deflate` and `trial_sharpe_std`
are statistics, not storage, and they are what `reporting` and `evolve` import it for.

**Decision.** `reporting/queries.py` reads the registry, the forward store, the feature
panel and the strategy classes, and returns plain data — no chart specs, no printing, no
argparse. `algorithm_book()` and `calendar_lab()` gather each cross-cutting page in one
call, and `cli.py` composes them into a `Report` through `algorithms_page()` and
`calendar_lab_page()`, which `cmd_all` and the two commands share. This extends the
ADR-028 seam one step left: gather → spec → render, with a testable boundary at each
arrow. `build_strategy()` takes explicit keywords rather than a Namespace, so the report
set, the forward suite and the tests can all call it.

The registry becomes a package: `stats.py` (pure, no I/O), `store.py` (the append-only
logs), `deflation.py` (ADR-026), `holdout.py` (ADR-025). The public surface is unchanged.

**The one deliberate incompatibility.** The three log paths are *not* re-exported from
`registry/__init__.py`. `store` owns them and `holdout` reads them through it
(`store.HOLDOUT_LOG`, never a from-import). Re-exporting would have made
`monkeypatch.setattr(registry, "HOLDOUT_LOG", tmp_path / ...)` patch a copy while `store`
went on appending to the real ledger: green tests, polluted data. Both the trial log and
the holdout ledger are append-only and irreplaceable (ADR-025, ADR-026), and this is the
failure mode those ADRs exist to prevent. Because the name is absent from the package, a
patch aimed at the wrong module raises `AttributeError` instead of doing nothing
quietly. Three fixtures needed updating; one test caught the fourth case on its own.
`test_the_isolation_fixture_actually_isolates` now pins it, asserting both that the
paths are absent from the package and that the real logs are byte-identical after a run
that writes to both.

**Not done, and why.** A rename to disambiguate `Panel` from `FeaturePanel` was
considered and dropped: the convention is already consistent — `panel` is the price
matrix everywhere, and a `FeaturePanel` is `fp` or `features`. Two modules named
`panel.py` is friction when navigating, not a defect worth churning twenty files for.

**Result.** `cli.py` 1,381 → 800 lines; `registry.py` 860 → four modules of 64–557.
Four `slugify` implementations under three names, two `_gt`/`_lt` pairs and two
`_finite` copies collapsed into `reporting/util.py` — they had drifted, and
`views._gt(inf, 0)` disagreed with `forward_views._gt(inf, 0)`. 377 tests pass; the full
report set, the Algorithm Book and the Calendar Lab rebuild and every index link
resolves.

---

## ADR-040 — Reviewed vendor errors are allowlisted, not repaired and not ignored

**Status:** accepted (2026-09-02)

**Context.** Four bars out of 3.7 million are structurally impossible at the vendor: three
where `low > open` by a few cents (HUBB and UA on the same day, 2021-05-05; SAF on
2021-08-30) and one zero open (NAVIV, 2014-04-30). TODO-9 asked for a decision — drop,
repair, or leave — and until one was made `quality --strict` could not gate anything,
because it would fail forever on rows nobody was going to fix.

**Decision.** Leave the rows as received, list them in `quality.checks.KNOWN_BAD_BARS`
with a one-line reason each, and report them as WARN instead of ERROR. Any *new*
impossible bar is still an ERROR. Nothing is repaired: the engine executes at the open
and marks at the close, the zero open is filled from the close and counted, and the
spread estimator that reads high/low is a 21-session trailing median — none of the four
can move a result. Repairing them would put an invented number into silver, which is
worse than a documented wrong one.

**Consequence.** `sp500lab quality --strict` now exits 0 on the real lake and can gate a
pipeline. The allowlist is the contract: everything on it has been looked at, and adding
to it without looking is the one way to break the check without the check noticing.

---

## ADR-041 — Every strategy is checked, and every check runs from one command

**Status:** accepted (2026-09-02)

**Context.** The acceptance suite proved the engine: SPY total return, the equal-weight
identity, the leakage guard, determinism, dividends counted once. It proved nothing about
the twenty-five strategies plugged into it. A strategy that trained a model in
`on_start(panel)` on rows past its as-of date would pass all five — `on_start` sees the
whole panel by design, and `context.py` cannot bound it — and `shallow_mlp` did exactly
that in revision 1 (ADR-037 postscript). The checks also lived in five places, and it
was easy to run four of them.

**Decision.** Two checks over the roster in `backtest/accept.py`:

* **Check 6, contract.** Each strategy runs twice on 2012–2014 (~36 rebalances). The
  curve must be finite and positive, no weight row all-NaN, no negative weight or leverage
  past validation, both runs byte-identical, net NAV no higher than gross, turnover in
  [0, 1], and the trade ledger must reconcile.
* **Check 7, no lookahead.** Each strategy runs on the full panel and again on a copy
  physically truncated five sessions past the window with every later filing and macro
  print deleted (`truncate_panel` plus `build_features(data_cutoff=...)`). The target
  weights at every rebalance must be bit-identical — the same standard `check_leakage`
  holds features to. A difference means the strategy read data past its as-of date.

And one command, `sp500lab doctor`, that runs everything in cost order with one exit code:
the bronze re-hash, the silver battery strict on ERROR, the engine suite, the timing
engine's identities, the feature-leakage rebuild, and with `--roster` checks 6 and 7. A
stage that raises is a failed stage. Full run 2.5 minutes; `--fast` skips the two slow
stages for a commit hook.

**A bug this found.** The feature cache key read `n_dates` and `end` from `panel.meta`,
which `truncate_panel` does not update. A truncated panel therefore hashed to the same
key as the full one, and the engine's auto-load path could hand a truncated run the full
panel's cached features — the exact leak the truncation exists to detect. `check_leakage`
had sidestepped it with `use_cache=False`; check 7 could not. The key now reads the
arrays. One feature rebuild.

**Result.** All 25 strategies pass both checks. The learned family is opt-in
(`--include-learned`) because it trains a model per run.

---

## ADR-042 — Factor returns and a regime set join the data layer

**Status:** accepted (2026-09-02)

**Context.** Every strategy is scored against SPY, which answers "did it beat the index"
and nothing more. The next question — was it a factor bet in disguise — needs factor
returns, and a daily-rebalanced idea needs cross-asset and sector series the five
benchmarks did not carry. Both are free.

**Decision.** `ingest/fama_french.py` pulls the five Fama-French factors plus momentum,
daily since 1926, from the Ken French library — keyless zips with a free-text preamble,
parsed by finding the header row rather than trusting a line count — into
`factors/fama_french_daily`, wide, in decimals. `benchmarks.BENCHMARKS` grows from 5 to 29:
the eleven GICS sector SPDRs, the VIX 9-day and 3-month term structure, Treasuries at
three tenors, investment-grade and high-yield credit, gold, the dollar, commodities, the
Nasdaq-100, mid caps and the total market. The calendar is still derived from SPY alone.

**The check that came with it.** `Mkt-RF + RF` is CRSP's value-weighted total market
return; SPY's daily total return tracks it above 0.98 with a slope near 1. Asserting
both catches a lost decimal in the factor parser (percent left undivided gives
correlation ~1 with slope ~0.01) as surely as a shifted SPY series. It is the third
cross-source check, after VIX (FRED against Yahoo) and SPY against ^GSPC, and it is the
cheapest kind of check there is: two vendors who do not know about each other agreeing.

**What was measured on arrival.** 26,173 sessions, five factors from 1963-07-01, Mkt-RF
annualising to 7.32%, and the library lags about two months — data through 2026-06-30 on
2026-09-02. Re-ingesting benchmarks moved the calendar's last session from 2026-08-26 to
2026-09-01, which is why a benchmark refresh must be followed by a price refresh: the
calendar is the spine, and a session with no bars is a WARN the quality battery reports.

**Not done.** The 296 ever-members without prices stay missing. EODHD's delisted list
carries 277 of them, but the free tier returns one year of history per call at twenty
calls a day, which cannot backfill 2007. That remains the paid-plan purchase (TODO-8) and
the single largest limitation on every early-year number here.

---

## ADR-043 — The price ingester gates every series, and a rollback is a replay from bronze

**Status:** accepted (2026-09-02)

**Context.** The routine price refresh that followed ADR-042 came back different in
three ways at once, and `doctor` caught all three. Yahoo returned 34 delisted names it
had never served before, about a third of them corrupt — Countrywide with 895 impossible
bars and a +449,900% day, Titanium Metals at a $10,100 close, RadioShack missing half
its sessions, Merrill Lynch as two bars. It returned nothing for Agilent, a live
constituent, which silently vanished from silver. And it restated Amphenol and Leggett &
Platt end to end, which is the vendor rewriting history that ADR-006 exists for.
Everything downstream rebuilt without complaint. The equal-weight identity came out at
91%/yr against a reference of 88%; the dividend check reported 28 points of yield. The
676 previously-good tickers were otherwise byte-identical, which bronze proved — both
pulls were on disk, partitioned by fetch date, and the diff took one script.

**Decision.** Three things, all in `ingest/prices_yfinance.py`.

* **An integrity gate.** `series_integrity()` scores every returned series and rejects
  one that has more than five impossible OHLC rows, a one-day move above 400% inside its
  index-membership window on a day with no recorded split, or fewer than half its own
  sessions present. Rejected series never reach silver; they are written to bronze as
  `rejected_tickers.json` with the reason. The return rule is judged inside membership
  because that is where the panel looks — a shell's post-delisting pennies do not
  matter. The coverage rule exempts live members: a hole in a current constituent is a
  gap `quality` reports, never a series to drop, because dropping a live name is the
  survivorship bias this project exists to prevent. Dry-run on the validated pull: one
  rejection, a six-bar stub for a name that had just been acquired. On the refresh:
  seventeen, all delisted, all with a reason a human would agree with.
* **A carry-forward guard.** A ticker that was in silver last time and is absent or
  rejected this time keeps its previous rows, logged as a WARNING and written to bronze
  as `carried_forward_tickers.json`. Those rows had passed every check; one bad vendor
  response does not un-pass them.
* **Replay from bronze.** `ingest prices --from-bronze YYYY-MM-DD` rebuilds silver from
  that day's chunks with zero network. This is the rollback, and it is why bronze was
  partitioned by fetch date from the first commit. A replay skips the carry-forward,
  because a replay means the current silver is exactly what is not trusted.

**Consequence.** Silver was restored from the 2026-08-27 pull through the gate,
everything derived was rebuilt, and `doctor` passed. The refresh's chunks stay in bronze
as the record of what the vendor returned. The next refresh will run through the gate;
the names it accepts among the 34 are real coverage the project did not have before,
and the ones it rejects are the reason the gate exists.

**The lesson for the bug table.** Nothing errored. The universe was point-in-time, the
factors were computed, the panel clipped to membership, and five of the acceptance
checks passed. The one that failed was the accounting identity, because a fabricated
+721,000% day is a real return to an engine that trusts its inputs. A refresh is a
change to the data, and a change to the data gets the same gate as a change to the code.

**Postscript: the refresh re-run through the gate.** With the gate in place the same
chunks produced 693 tickers, bars to 2026-09-01, and a doctor that failed on two much
smaller things. Five impossible bars had entered on four accepted delisted names (the
gate tolerated up to five per series) plus two new ones Yahoo's restatement put on APH
and LEG; and the equal-weight identity landed at 10.9bp against its 10bp bound — from
0.1bp — because seventeen more names now genuinely delist inside the window, and the
reference renormalises their proceeds across survivors where the engine parks them in
cash. Its own docstring predicted that residual would grow with delisting coverage. The
gate was tightened (a delisted series with even one impossible bar is rejected; a live
member keeps the five-row tolerance, because on a first pull there is nothing to carry
forward and its bad rows go to review), and silver was restored to the validated pull
again. The identity question is TODO-10: whether the reference should model proceeds
the way the engine does before the bound is moved. Until it is settled, the committed
state is the validated pull, and the refresh is a command away.

---

## ADR-044 — A ticker blocklist is a universe decision

**Status:** accepted (2026-09-02)

**Context.** The refresh in ADR-043 "dropped" Agilent Technologies, ticker `A`, and the
new carry-forward guard did not carry it. It could not: `A` had never been requested.
The Wikipedia history parser keeps a small list of header words shaped like tickers
(`SYMBOL`, `CIK`, `GICS` ...) and, from the first commit, that list also held `N`, `A`
and `NA` — fragments of "N/A" boilerplate. Agilent has been an index member since June
2000. It was in today's constituent list, which a different parser builds, and absent
from every monthly snapshot, every membership interval, and therefore every backtest,
every feature panel and every forward test this project has run. The ever-member count
of 971 should have been 972. Nothing flagged it, because a name that is not in the
universe is not missing from anything. (CBOE, below, was counted - it had a closed
interval - which is a different failure with the same effect.)

**Decision.** `A` comes off the list. `N/A` normalises to `N.A`, which is what actually
needed filtering, so that goes on; `N` and `NA` stay for the split fragments. The parser
tests now pin `A`, `{{NYSE|A}}` and `[[Agilent Technologies|A]]` as the ticker `A`, and
`N/A`, `NA`, `N` as nothing. The membership history was re-parsed from the cached
revisions, the validated price pull replayed through the gate (it had carried Agilent's
bars all along), and everything derived was rebuilt.

**Why this ranks with the ticker-recycling and sort-key bugs.** A false positive on a
blocklist that is shaped like a ticker is survivorship bias with no error message: the
company never fails a check because it never enters one. Every entry on such a list has
to be checked against the constituent list, and the check that would have caught this
in 2026-08 is a one-liner — every ticker in `sp500_current` must appear in the
intervals as an open member. That check now lives in `quality` (`membership`, ERROR).

**It found a second one on its first run.** Cboe Global Markets (`CBOE`, a member since
2017) had a *closed* interval ending 2018-12-31. Cboe lists on its own BZX exchange, and
from 2019-01 the Wikipedia table writes its row as `{{BZX link|CBOE}}`; the template
regex knew `nyse|nasdaq|bats|arca|amex` and not `bzx`, so the row was dropped from
every later snapshot and CBOE spent seven years of backtests as an ex-member. `bzx` and
`cboe` are in the regex now, with the template pinned in the parser tests. Two live
constituents lost to two different parser gaps, neither visible from inside the
membership table — which is the argument for the cross-check in one sentence.

---

## ADR-045 — The report folder is sets and nothing else

**Status:** accepted (2026-09-03); title amended by ADR-047, which generalised "two
sets" to "one folder per lab". The decision below is unchanged.

**Context.** By 2026-08-30 `reports/` held 163 files and 254 MB: thirty strategy pages,
the feature layer, the registry, the honesty panel, the Algorithm Book, the Calendar Lab,
two study pages, a `trades/` folder of full ledgers plus the per-strategy sub-folders the
`backtest trades` command wrote there, and a `forward_tests/` folder with thirty-eight
candidate pages, a decay analysis, an honesty page, Markdown copies of everything, a
`data/` folder of CSVs, a README and a run log. Every file was defensible on its own. As
a folder it had stopped answering the question it exists for — *which algorithm do I
want to look at?* — because the answer was buried under the supporting material.

**Decision.** `reports/` holds two sets, `backtest/` and `forward/`, and each set holds
exactly two kinds of file: `index.html`, a scoreboard with every algorithm's headline
statistics (CAGR, volatility, Sharpe, drawdown, Sharpe against the index over its own
window, the deflated Sharpe where a search produced it), and one self-contained page per
algorithm, named by the algorithm — so `backtest/low-vol.html` and `forward/low-vol.html`
are the same strategy on either side of the boundary. The two sets share one roster,
defined once in `reporting.queries.roster()` and `ga_winners()`: every built-in
strategy, the `custom` group, and the winners of the best three genetic-algorithm
searches on disk, ranked by the research Sharpe of each search's best logged run
(`GA_WINNERS_SHOWN`). A rebuild prunes pages an earlier build left in the default
folder, so the folder always describes the last build; a folder chosen with `-o` is the
user's and is never pruned.

**Everything else moves out, not away.** The single-page commands — `strategy`,
`features`, `algorithms`, `timing`, `registry`, `honesty`, `study`, `run`, `compare`,
`trades` — still exist and default into `reports/extra/`. The forward decay analysis and
the forward honesty page remain as views in `forward_views.py` with no command of their
own. Trade ledgers as files are the engine's concern, not the reports': `backtest trades`
now writes to `results/trades/<strategy>/`, beside `results/forward/`, and the report set
no longer writes a sibling CSV — a page embeds its ledger up to `MAX_EMBEDDED_TRADES` and
names the command that writes the whole thing. The Markdown copies and the CSV exports of
the forward set are dropped; `render/markdown.py` stays as the second backend the seam is
tested against.

**What the roster changes.** `custom` strategies are in the report set by default. The
engine's `GROUPS["all"]` still excludes them, so `backtest suite all` and the forward
suite are unchanged: the scoreboard everyone is measured against is the engine's, and the
report set is where a strategy of your own is meant to be read next to it.

**What stays honest.** The forward index's multiple-testing bar is computed over every
forward record, not over the roster, and the page names the tested candidates it does not
show (the calendar rules, today — since ADR-047 it also links to them), because a
candidate that was looked at counts whether or not it has a page. Nothing in the record is deleted or hidden from the store; only the
folder got smaller. `tests/test_forward_reports.py` pins it.

**Consequence.** `report all` remains as an alias of `report backtest`. Three console
logs that had been parked in `reports/` (`ga-price-1.log`, `ga-full-2.log`,
`forward_tests/run.log`) were moved to `logs/`, since they are not regenerable and were
never reports. Everything else under `reports/` was deleted and rebuilt.

---

## ADR-046 — The genetic algorithm gets its own three pages

**Status:** accepted (2026-09-03)

**Context.** After ADR-045 the report folder answers "which algorithm do I want to look
at?" well and answers "where did the evolved ones come from?" not at all. An evolved
winner appears on both scoreboards as a row with a deflated Sharpe beside it, which is
the right amount of context in a table and nowhere near enough to judge the apparatus
that produced it. The search itself — a bounded genome, a fold-consistency objective,
six operators, four anti-overfitting defences, and three completed runs of ~1,400 trials
each — was documented only in `docs/EVOLUTION.md` and `docs/HOW_THE_GA_WORKS.md`, which
are for people who read the repository rather than for people who read the reports.

**Decision.** `sp500lab report genetic` writes `reports/genetic_algorithm/`, holding
three pages and no index:

| Page | Answers |
|---|---|
| `methodology.html` | the search space, the objective and every penalty, the operators, the four defences |
| `features.html` | the three presets, why they are short and frozen, and what each search converged on |
| `evolved-algorithms.html` | every search: settings, training history, winning genome, deflation, forward verdict |

**No index, deliberately.** A fourth page whose only content is three links is a file,
not navigation. Each page carries a link grid to the other two and back to the backtest
scoreboard, and both set indexes gained a card pointing here — the evolved winners are
rows on those scoreboards, and this is where they came from.

**The set runs nothing.** No search, no backtest, no panel. Every figure comes from the
checkpoints in `data/experiments/evolve/`, the trial log and the forward store, so the
pages rebuild in about a second. That is the same guarantee ADR-034 gives the forward
reports, and it matters more here: a search is thousands of evaluations and re-running
one to draw a chart of it would be absurd.

**The anatomy is read off the code, never restated.** `queries.genome_anatomy()` reports
the gene bounds from `alpha_genome`, the dead zone from `genome.py`, the BLX-α constant
from `operators.py` and the defaults from `EvolutionConfig`, so the methodology page
cannot drift from the search it describes.
`test_the_genome_anatomy_is_read_off_the_real_genome` pins that.

**Two editorial rules, both pinned by tests.** First, an evolved result never appears
without its trial count and its deflated Sharpe, because a searched Sharpe is the maximum
over every configuration evaluated and the maximum of N draws is high with or without
signal. Second, the deflated Sharpe is a *probability* and is never printed as `1.000`:
at or above 0.9995 it renders `>0.999`, because three decimals of rounding should not
manufacture a claim of certainty that 136 monthly observations cannot support.

**What the population agrees on outranks what the winner did.** The features page reports
the winner's weight on each feature *and* the share of the final 60-individual population
that put a live weight on it. The winner is one draw; the population share is the number
no single lucky genome can move, and it is the more honest description of what a search
found.

**A search with no checkpoint is named, not dropped.** `ga-full-1` is 1,005 trials in the
registry with no surviving checkpoint, so no winner can be decoded from it. It gets a row
saying exactly that, because those trials still count toward the deflated Sharpe of
anything logged in that study, and a reader comparing the page against
`sp500lab experiments studies` has to be able to see why one search has no winner.

**Consequence.** `reports/` now holds three folders. The genetic set is small (about
130 KB for three searches) because it embeds no trade ledgers. What it reports today, on
the three searches that have run: all three winners cleared the 0.95 deflated-Sharpe
convention on the research window, and all three decayed out of sample. The pages say so
in that order.

---

## ADR-047 — The calendar rules get a folder, not nine more scoreboard rows

**Status:** accepted (2026-09-04)

**Context.** The nine calendar rules were forward-tested through the same harness as
everything else and appeared in no report set. `reports/backtest/` and `reports/forward/`
are built from `queries.roster()`, which is the monthly strategy registry; a `tm_` rule is
not in it, so the forward index carried a note naming nine tested candidates it did not
show and pointing nowhere. Their only page, the Calendar Lab, had been on-demand since
ADR-045 and lived in `reports/extra/` — a folder that did not exist on disk, because
nobody had run the command since the reshape. Every number was in the store. None of it
was in the deliverable.

**Why not just add them to the two sets.** Three reasons, in order of how much they hurt.
The backtest index sorts on ΔSharpe against the index, and a rule that sits in cash 80% of
the time gets a structural Sharpe boost from not being in the market during the
volatility — `TurnOfMonth`'s own docstring says so. Sorted into a table of thirty fully
invested stock pickers it would rank near the top for a reason that is not skill, and the
comparison the set exists to make would quietly stop being valid. Second, the page shape
does not fit: `strategy_report` takes feature coverage, holdings and orders, and a
calendar rule has one instrument, no cross-section and no features, while its interesting
columns — exposure, cost drag, entries, the gross-to-net gap — have nowhere to go. Third,
`roster()` feeds two builders, and putting a second engine's names in it means every
consumer has to branch on which engine a name belongs to.

**Decision.** `sp500lab report timing` writes `reports/timing/`: an index and one page per
calendar rule, named by the rule, pruned on rebuild — the same grammar as the two monthly
sets. `reports/` is now one folder per lab, and a lab splits by window only when it is too
big for one page per algorithm. The monthly roster (thirty algorithms × two windows) is;
the calendar and genetic labs are not.

**Both windows on one page, which is the substantive choice.** A calendar rule's research
and forward numbers are one story: `tm_weekend` went from a 0.04 Sharpe in research to
−0.20 forward and is the single `failed` verdict in the whole 29-candidate forward set.
Filing the claim in one folder and its refutation in another would be the wrong shape for
a family this size.

**The forward half is not written twice.** `forward_views.outcome_sections()` is public
for exactly this: a rule page renders its paired comparison, its curve, its significance
arithmetic, its nine checks and its three cost settings with the forward set's own
sections, from the same stored record. A forward table computed in two places is a forward
table that will eventually disagree with itself, and the disagreement would be invisible —
both copies would look plausible.

**`entries` replaces a prose column.** Every page reports the round trips a rule makes over
the research window, counted off its own leg vectors: the legs are walked in time order
(intraday[t], overnight[t], intraday[t+1], …) and the rising edges of that interleaved
vector are the entries. It reproduces what `docs/TIMING.md` had been asserting in prose —
overnight ~3,700, weekend ~760, turn-of-month ~180, pre-holiday ~130, sell-in-May ~15 —
from the code, so the two cannot drift. It is the number that decides how to read every
other number on the page: `tm_sell_in_may` is invested across 1,806 sessions and enters
sixteen times, and sixteen is the sample. `test_the_entries_column_reproduces_the_documented_sample_sizes`
pins it.

**One printed number changed.** `time in market` used to be `(overnight | intraday).mean()`,
which called an overnight-only rule 100% invested because some part of every session was
held. It now counts half-sessions, so holding every night reads 50%. The old number
invited exactly the comparison against buy-and-hold that the column exists to inform.

**What stays where it was.** `roster()` and `forward_roster()` are untouched, so the
monthly sets are unchanged. The calendar rules remain outside the forward set's pages and
inside its multiple-testing bar — they were looked at, so they count — and its "not
everything tested is shown" note now links here instead of naming nine candidates and
stopping. `queries.claim_for()` resolves the timing registry as well, so anything handed a
`tm_` record renders it with the rule's own docstring rather than an empty claim.

**Consequence.** `reports/` holds four folders. The calendar set is about 600 KB for ten
pages; it is the only set that runs a backtest, because a rule with no research row yet
has to have one before it can be reported, and those fill-in runs are logged under the
study "reports" with the holdout untouched. What it reports today: nothing in the family
beats buy-and-hold on ΔSharpe under realistic costs, the best trial deflates to 0.69
against the family's own 27 trials, and the one rule that was refuted out of sample is on
its own page saying so.

---

## ADR-048 — The search space is nine prior-signed families, at most three at a time

**Status:** accepted (2026-09-04)

**Context.** Three genetic searches ran over the feature presets of ADR-031 — 13, 23 and
17 free weights, each in [−1, +1] — and every one cleared the 0.95 deflated-Sharpe
convention and decayed out of sample (`ga-price-1` 1.15 → 0.19, `ga-full-2` 1.36 → 0.67,
`ga-night-1` 0.95 → 0.20). The deflated Sharpe corrects for how many configurations were
*tried*; it cannot correct for a space wide enough that some configuration fits fifteen
years of monthly history by construction. Even thirteen free signs and magnitudes is that
wide: the searches disagreed with the literature and with each other on the sign of
book-to-market, residual momentum and the overnight decomposition, and each disagreement
was in-sample noise wearing a weight.

**Decision.** Two new presets, `families` and `families-price`, replace the free-weight
space as the default. The 22 features with a prior story are grouped into nine
economically motivated families — momentum, short-term reversal, low risk, illiquidity,
payout, value, quality, investment, earnings surprise — each a fixed composite of its
members' percentile ranks, every member signed by the literature (`vol_126d` low is good,
`gross_profitability` high is good). The genome carries one **non-negative** weight per
family and the preset caps how many may be live at once (`max_active = 3`), enforced when
the vector is decoded: `Genome.effective()` zeroes the dead zone and everything past the
cap, `decode()` returns that vector, and `fingerprint()` hashes it. The search picks
*which* stories to back and how hard. It cannot rank value backwards, and it cannot back
all nine at once. `families` carries all nine from 2010-07; `families-price` the five
visible from 2007-04 without a filing. 15 and 11 genes, against 19 to 29 before.

**Why families rather than fewer features.** A family is a hypothesis somebody could have
written down before looking: "back quality and low risk, two to one". A weight on
`accruals` is not — the sign is half the hypothesis, and a search that is free to choose
it is free to fit it. Fixing the sign inside the family removes 2^k hypotheses per k
features at no loss anyone can name, because the reversed hypothesis is one nobody would
defend in prose.

**Why the composite is `1 − rank` for a low-is-good member, not a negative weight.** Every
term stays in [0, 1], so a name missing a member is scored as average on it. With
negative weights a name missing a low-is-good feature would be quietly rewarded for
having no value, which is a survivorship channel in miniature.

**What was cut, and the reason is data.** Everything without a story: redundant horizons
(`mom_6_1`, `ret_12m`, `vol_21d`), the contested overnight decomposition, the size and
spread proxies, the index-entry flags, the ambiguous ratios, filing behaviour and every
macro series. `CUT_FEATURES` in `strategies/genome.py` records each cut beside its
reason, the features page prints it, and a future search that wants one back has to
argue with the reason rather than with an absence.

**Presets stay immutable (ADR-038).** A family's member tuple, its signs and its preset's
cap are part of the decode-by-position contract. `test_the_family_presets_are_frozen`
pins all of them; the three feature presets are untouched and their checkpoints decode
bit-identically (`test_a_feature_genome_still_decodes_exactly_as_before`).

**Consequence.** `EvolvedFamilies` is a subclass of `EvolvedAlpha` with the same engine
contract, so the family search costs the same ~0.1–0.2 s per evaluation and flows
through the same cache, registry, forward harness and report set. The first search over
it is `ga-families-1`.

---

## ADR-049 — Fitness is the worst quarter of random sub-periods, net of pessimistic costs, minus a charge per rule

**Status:** accepted (2026-09-04)

**Context.** ADR-032's objective scored an individual on four contiguous folds aggregated
as `mean − 0.5 × std`, under realistic costs, with turnover and complexity penalties that
defaulted to zero. It was better than the whole-window Sharpe and it was not enough: a
mean-minus-spread over four folds still rewards a rule that is excellent in two folds and
ordinary in two, and a search charged the realistic spread evolved 250–350%/yr of
turnover in every winner.

**Decision.** Four changes to the objective, all defaults, all recorded in every
checkpoint and every run's manifest.

1. **Random sub-periods, drawn once.** `Folds.random` draws twelve contiguous sub-periods
   of three to five years at random positions in the research window, from a private
   generator seeded by `fold_seed` — never from the search's own random state, so every
   individual and every seed of a search is scored on the same sub-periods and their
   fitnesses can be pooled. They overlap, and usually do; that is the point. The
   contiguous scheme stays available as `--fold-scheme contiguous`.
2. **The worst quarter decides.** `aggregate = quantile` at 0.25: an individual's fitness
   is the 25th percentile of its sub-period Sharpes. A rule that only works in one
   stretch is killed *during* evolution rather than surviving to the test set on the
   strength of that stretch. `min` is the harsher option and `mean_minus_std` the old one.
3. **Costs inside the fitness, at the pessimistic setting.** The curve being scored has
   always been net of costs; it is now net of commission plus *twice* the estimated
   half-spread. A rule that only works if the spread estimator is kind never scores well.
   The turnover penalty then charges turnover a second time (0.03 per 100%/yr), on
   purpose: it is a statement about model risk in the spread estimate, not about costs.
4. **Complexity, charged per rule.** 0.02 per family backed, 0.01 per feature read, and
   0.03 for switching the regime gate on. A three-family strategy with the gate pays about
   0.18 of fitness more than a one-family one without it, so it has to earn that in the
   worst-quarter statistic. The earlier searches showed why the gate is charged: the −16%
   drawdown that made `ga-price-1` look best was two parameters tuned on a window that
   contained 2008.

**What this is still not.** Sub-periods measure robustness. Every one is inside the
research window and every individual is selected using all of them, so a good score is
evidence of consistency and not of generalisation. The out-of-sample test remains the
2022 holdout, looked at once and recorded (ADR-025). The "not a shuffled K-fold" rule of
ADR-032 holds: a random sub-period is one contiguous stretch of history, random only in
where it starts and how long it is.

**Cost.** None to speak of. Twelve sub-periods are sliced from the single equity curve the
individual already produced, as the four folds were.

**Consequence.** The fitness a checkpoint records is no longer comparable with the three
earlier searches' fitnesses, and the searches page says which objective each search
used. `EvolutionConfig` moved to `evolve/config.py` so the command line can print the
engine's own defaults without importing pandas.

---

## ADR-050 — A search hands on an ensemble, not its champion

**Status:** accepted (2026-09-04)

**Context.** The single best individual of a search is the maximum over thousands of
draws — the most luck-contaminated object in the whole population — and it is exactly
what every earlier search handed to the forward test. Three for three decayed. The
population around the champion holds the same information with most of the luck averaged
out: what thirty survivors agree on is a finding, what one of them found alone is a
draw.

**Decision.** A search's deliverable is `EvolvedEnsemble`: the equal-weighted average of
the **beliefs** of its top `ensemble_size` (30) distinct individuals by fitness, pooled
across every seed it ran (`--seeds N` runs N independent populations sharing the study,
the objective and the evaluation cache). Precisely:

- each member's weighted sum of ranks, gate ignored, is re-ranked to [0, 1] within the
  tradable universe and the ensemble score is the mean over members with an opinion on
  the name, at least three of them;
- the regime gate is a vote — the ensemble steps aside when at least half its members
  would, investing the mean of those members' defensive exposure;
- the holding count and per-name cap are the medians of the members', the weighting
  scheme the mode.

**Beliefs, not portfolios.** Averaging thirty twelve-name portfolios gives a
two-hundred-name portfolio paying a dollar of commission minimum on every name at this
account size, and the result would measure the cost model rather than the signals.

**It is stored, logged and discovered like everything else.** The ensemble is written
beside the checkpoint as `<study>.ensemble.json`, backtested once at the end of the search
and logged into the same study as one more trial, so the deflated Sharpe of anything in
the study counts it. `evolve.winners()` hands over the ensemble as `<study>-ensemble`
wherever one exists and the champion as `<study>-best` where none does, so the forward
suite and both report sets receive the deliverable without knowing which kind it is.
`sp500lab evolve ensemble <study> --rebuild` builds one from any checkpoint, across all
its seeds, for a search that predates this decision or was interrupted.

**The champion is still shown.** The searches page decodes it beside the ensemble because
it is the one individual the ensemble is measured against, and because a reader who sees
only the average cannot check what the survivors converged toward.

**Consequence.** The three earlier searches keep handing over their champions — their
forward tests are spent and their record must not move — and every search from
`ga-families-1` on hands over an ensemble. Its honest test is the data that arrives after
2026-09, exactly as ADR-037 says of everything built after the holdout was read.

**Postscript, the same day.** `ga-families-1-ensemble` was forward-tested once, on the
user's decision, under all three cost settings (`forward run` now resolves an evolved
deliverable by name, so a single look no longer requires a suite that re-spends the
older champions'). 2022-02 → 2026-09: 5.73%/yr at a 0.46 Sharpe against the index's
13.52%/0.82; monthly Sharpe 1.42 → 0.55, −1.5σ. **Decayed, four for four.** It made
money, halved its drawdown and held its turnover — the redesign produced a more stable
strategy — and it did not beat the index in a mega-cap regime. Because the design was
chosen after the 2022–2026 results were read, the decay is a stronger refutation than a
hold would have been a confirmation. The walk-forward is the next piece of evidence, and
the months after 2026-09 the only clean one.
