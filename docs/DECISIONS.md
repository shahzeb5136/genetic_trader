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
