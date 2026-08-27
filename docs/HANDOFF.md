# Handoff

Written 2026-08-27 for whoever picks this up next — human or model. Read this before
touching anything. It assumes no prior context.

---

## 1. What this is

A research data platform for the S&P 500, built to a **$20/month** data budget by one
individual. The long-term goal is a **competition**: genetic algorithms and other
classical strategies against neural nets, all scored on one shared backtest harness.

**Current phase: the data layer is built. Nothing else is.** There are deliberately no
strategies, features, or backtests in this repo. That was the owner's explicit
instruction and it was the right call — every project of this kind dies on its data
layer, so the data got built and validated first.

Repo: `https://github.com/shahzeb5136/genetic_trader` (public, 2 commits)

---

## 2. State of play — verified numbers, not estimates

Every figure below was measured on 2026-08-27. Re-derive with `python -m sp500lab status`.

| Dataset | Rows | Notes |
|---|---:|---|
| `sp500_membership_intervals` | 1,019 | **Point-in-time universe.** The important one. |
| `sp500_membership_snapshots` | 113,731 | 227 monthly snapshots, 2007-03 → 2026-08 |
| `daily_bars` | 3,706,372 | 677 securities, 2000-01-03 → 2026-08-26 |
| `daily_bars_adjusted` | 3,706,372 | + our own adjustment factors |
| `corporate_actions` | 41,954 | 41,224 dividends, 730 splits |
| `xbrl_facts` | 3,346,513 | Point-in-time fundamentals, 649 companies |
| `fred_series` | 127,457 | 18 macro series |
| `eodhd_us_symbols` | 111,032 | 51,206 active + 59,826 delisted |
| `trading_calendar` | 6,702 | NYSE sessions, derived from SPY |
| `benchmarks` | 32,577 | SPY, RSP, ^GSPC, IWM, ^VIX |

**The headline: 971 tickers have been in the index since 2007-03. 501 are in it today.
470 are gone.** That gap is the entire reason this project exists.

Tests: 30 passing (`python -m pytest tests/ -q`). Bronze integrity: 700 artifacts,
all checksums verified, 25 tombstoned.

---

## 3. Non-negotiable invariants

Violating any of these produces a backtest that looks good and is wrong. They are not
style preferences. Each one is documented as an ADR in `docs/DECISIONS.md`.

1. **Never build a universe from `sp500_current`.** Use
   `query.universe_asof(date)`. The current list applied to a historical date is the
   survivorship-bias mistake, worth ~1.5–2.0%/yr of fake return.

2. **Never filter fundamentals on `period_end`.** Use
   `query.fundamentals_asof(date)`, which filters on `filed_date`. Apple's FY2023
   ended 2023-09-30 but was not public until 2023-11-03 — 33 days of free lookahead.
   **60.8% of (security, tag, period) combinations have been restated**, so this is
   not an edge case.

3. **Never join price data on a bare ticker.** Symbols get recycled. `WM` was
   Washington Mutual until it failed in 2008; Waste Management holds it now. Use
   `query.prices_clipped_to_membership()` — it cuts CPWR from 4,962 bars to its real
   1,279. 155 tickers are affected.

4. **Never use a vendor's `adj_close`.** It is rewritten on every dividend, which makes
   results irreproducible. Use `adj_factor` (splits + dividends) for returns and
   `adj_factor_price` (splits only) for levels and volume.

5. **Know the price convention of every source.** yfinance pre-applies splits even with
   `auto_adjust=False`; EODHD documents raw OHLC but split-adjusted volume — a *mixed*
   convention. Getting this wrong is silent and looks like alpha.

6. **7 of 18 FRED series are revised after publication** (CPI, GDP, payrolls,
   unemployment, industrial production, sentiment, recession flag). Using them at face
   value is a lookahead leak. The `revised` column flags them.

---

## 4. Are we ready to build algorithms?

**The data layer: yes, with four caveats below. The harness: it does not exist.**

`src/sp500lab/backtest`, `features`, and `costs` are all **ABSENT**. `data/gold/` is
**empty**. Full technical specifications for closing every gap are in **§5b (TO DO)**. So "start building GAs" really means "build the thing that scores GAs
first." Budget ~3 weeks before the first genetic algorithm runs.

### Gaps that will bite, by severity

**BLOCKING — fix before trusting any backtest number**

- **Delisting returns do not exist.** When a holding is delisted, the backtest books
  its last close and the position silently vanishes. The real outcome (merger cash, or
  ~−100% in a bankruptcy) is never realised. This inflates returns in the *opposite*
  direction from survivorship bias.
  **Critical subtlety: buying the paid feed makes this worse, not better.** Only 4
  cases are visible today because Yahoo carries almost no delisted names. With full
  coverage there will be ~500. Fixing the coverage gap *exposes* this one.
  There is no free source for delisting returns; the practical approximation is to read
  the reason text in `sp500_changes` (bankruptcy → −100%, acquisition → last price) and
  record the assumption explicitly.

- **EODHD price convention is unverified.** `verify_price_convention()` exists and
  returns `inconclusive` on the free tier because the NVDA 2024 split predates the
  1-year window. Run it the day the paid plan starts, before ingesting anything.
  Volume in particular would be wrong by the split ratio.

**IMPORTANT — needed for any factor or cap-weighted work**

- **No market cap series.** 329,298 shares-outstanding facts exist across 646 companies
  but are not reconciled into a point-in-time market cap. Size, value, and
  cap-weighting all need this.
- **Sectors are current-only.** `gics_sector` comes from today's snapshot. Any
  sector-neutral or sector-rotation strategy backtested with it leaks — a company
  reclassified in 2018 gets its 2026 sector in 2008.
- **No index weights.** The cap-weighted benchmark cannot be reproduced exactly; only
  equal-weight is available from our own data. Use the SPY series as the benchmark.

**MINOR**

- Fundamentals begin 2009-04 (XBRL mandate), leaving a 2-year hole against the
  2007-03 membership start. Price-only strategies can use the full window.
- 1 ERROR outstanding: 4 rows where `low > open` (HUBB, NAVIV, SAF) out of 3.7M.
- Membership is monthly, so an add/remove is dated to within a month.

---

## 4b. Mandate (decided 2026-08-27)

Long-only, monthly rebalance, sub-$100k capital. See ADR-016. This fixes the engine's
shape: non-negative weights summing to one, ~230 rebalance dates, and a cost model of
commission + estimated half-spread with no impact term. Critically, monthly rebalancing
means the monthly granularity of our membership reconstruction is no longer an accuracy
limit - a strategy that only acts at month boundaries cannot be harmed by sub-month
dating error.

---

## 5. Build order

Do not reorder 1 and 2. The engine is the GA's fitness function; features are its
inputs. Both must exist before any evolution happens.

**1. Backtest engine** — one interface both model families implement:
`(point-in-time view, current positions) -> target weights`. Vectorized first
(event-driven later). Must include a cost model: commission, half-spread, market impact
vs ADV, borrow cost, and execution at the **next** bar's open — never the close that
generated the signal. Make costs pluggable with optimistic/realistic/pessimistic
settings and report all three.

**Acceptance test before writing a single strategy:** buy-and-hold SPY through your own
engine must reproduce SPY's actual total return to within a few basis points. If it
doesn't, the engine is wrong and every downstream number is meaningless.

**2. Feature layer in `data/gold/`** — versioned, computed once, reused by every
competitor. Otherwise the competition measures who wrote better feature code.
Cross-sectional rank/z-score within each date is essential for the NN side.
Include a leakage test: recompute features with future data deleted and assert the
output is byte-identical.

**3. Baselines** — buy-and-hold SPY, equal-weight, 12-1 momentum, random-weight with
matched turnover. If a model can't beat 12-1 momentum after costs, it found nothing.

**4. Then** the genetic algorithms.

---

## 5b. TO DO — technical specifications

Ordered by dependency. TODO-1 is the starting point; nothing downstream is meaningful
until its acceptance tests pass.

Mandate constraints from ADR-016 apply throughout: **long-only, monthly rebalance,
sub-$100k capital.** These are not defaults to revisit casually — each one removes a
whole class of required work (borrow modelling, sub-month membership dating, market
impact respectively).

---

### TODO-1 — Backtest engine

**Why first:** it is the fitness function. A genetic algorithm cannot be evaluated
without it, and every metric anyone quotes comes out of it.

**Proposed layout**

```
src/sp500lab/backtest/
    __init__.py
    context.py     # point-in-time view handed to a strategy
    engine.py      # the rebalance loop + accounting
    costs.py       # cost model (see TODO-2)
    metrics.py     # performance statistics
```

**The interface both model families implement**

```python
class Strategy(Protocol):
    def target_weights(self, ctx: "Context") -> pd.Series:
        """Target portfolio weights indexed by security_id.

        Long-only mandate (ADR-016): every weight >= 0 and sum <= 1.0.
        The engine rejects violations rather than normalising them silently -
        a strategy that returns negative weights has a bug, not a short position.
        """
```

**Context — make leakage structurally impossible**

The single most important design decision in the engine. Do **not** hand the strategy
the full panel and trust it to filter by date; build the slice once, in the engine, so
future data is not reachable at all:

```python
@dataclass(frozen=True)
class Context:
    as_of: str              # rebalance date, a real trading session
    universe: pd.DataFrame  # query.universe_asof(as_of)
    prices: pd.DataFrame    # adjusted bars, date <= as_of ONLY
    features: pd.DataFrame  # gold features, knowledge_date <= as_of ONLY
    positions: pd.Series    # current shares by security_id
    cash: float
    nav: float
```

Build `prices` from `query.prices_clipped_to_membership()`, never from a raw ticker
join (invariant 3, §3).

**Rebalance schedule**

Month-end trading sessions: `SELECT date FROM trading_calendar WHERE is_month_end`.
That yields ~230 rebalance dates over 2007-03 → 2026-08, which is enough observations
for walk-forward with purging and an embargo.

**Execution timing — the classic bug**

Signals are computed from data up to and including the **close of date T**. Orders fill
at the **open of date T+1**. Filling at the close of T means trading on a price that
was not yet knowable when the signal was formed. Encode this in the engine, not in each
strategy, so no strategy can get it wrong.

**Accounting**

```
NAV_t = cash + Σ(shares_i × adj_close_i,t)
turnover_t = Σ|w_target − w_current| / 2
```

⚠️ **Do not add dividend cash separately.** `adj_close` already reinvests dividends via
`adj_factor` (splits + dividends). Adding cash dividends on top double-counts them and
inflates returns by roughly the dividend yield — for SPY that would be ~1.9pp/yr, which
is large enough to make a bad strategy look good. Use `adj_factor_price` (splits only)
if you ever need a price-return series.

**Acceptance tests — write these before any strategy**

1. **SPY total return.** Buy-and-hold SPY through the engine, zero costs, must
   reproduce SPY's actual total return within a few bp annualised. Reference figures
   already measured over 2000-01-03 → 2026-08-26: price return **6.43%/yr**, total
   return **8.32%/yr**. If the engine produces ~6.4% it is silently dropping dividends;
   if it produces ~8.3% the adjustment chain is correct.
2. **Equal-weight identity.** Zero costs and zero turnover on an equal-weight portfolio
   must equal the equal-weight index return computed directly from the panel.
3. **Leakage guard.** A strategy that attempts to read `ctx.prices` beyond `ctx.as_of`
   must raise, not silently succeed. Test it with a deliberately cheating strategy.
4. **Determinism.** Same inputs, same seed → byte-identical equity curve.

**Performance target.** A single full backtest over ~230 rebalances × ~500 names must
run in well under a second. GA fitness evaluation calls this 10,000+ times per run (§6);
anything slower makes evolution impractical. Vectorise over the whole panel — do not
loop per security.

---

### TODO-2 — Cost model

**Why it cannot be skipped:** turnover is where backtests lie. A monthly long-only
strategy at sub-$100k has a genuinely simple cost structure, so there is no excuse for
omitting it.

**What applies under ADR-016**

| Component | Treatment |
|---|---|
| Commission | Per-share with a minimum (e.g. IBKR ~$0.005/share, $1 min) |
| Half-spread | **Estimated** — we have no quote data |
| Market impact | **Omit.** Negligible for S&P 500 large caps below $100k |
| Borrow cost | **Omit.** Long-only |

**Estimating the spread without quote data.** Two published estimators use only daily
OHLC, which is exactly what we hold:

- Corwin & Schultz (2012), *A Simple Way to Estimate Bid-Ask Spreads from Daily High and
  Low Prices*, Journal of Finance — uses two-day high/low ranges.
- Abdi & Ranaldo (2017), *A Simple Estimation of Bid-Ask Spreads from Daily Close, High
  and Low Prices*, Review of Financial Studies.

Implement one, store the estimate per (security_id, date) in `gold/`, and sanity-check
that large caps land in single-digit basis points.

**Three settings, always report all three:** `optimistic` (commission only),
`realistic` (commission + half-spread), `pessimistic` (commission + 2× half-spread).
A strategy that only survives under `optimistic` is not a strategy.

---

### TODO-3 — Delisting returns  ⚠️ BLOCKING for correctness

**Why:** when a holding leaves the index, the engine currently books its last close and
the position vanishes. The real outcome is never realised. This inflates returns in the
*opposite* direction from survivorship bias.

**Read this before starting:** only 4 cases are visible today because Yahoo carries
almost no delisted names. **Buying the paid feed raises that to ~500.** Fixing coverage
exposes this problem rather than solving it, so build this before or alongside the
EODHD migration (TODO-8), not after.

**Critical distinction the table must encode** — these are not the same event:

| Case | What happened | Correct handling |
|---|---|---|
| Index removal, still listed | Market-cap decline, index rebalance | Not a delisting. Sell at next open. |
| Acquisition / merger | Company bought | Position exits at deal terms; approximate with last price |
| Bankruptcy / liquidation | Equity wiped out | ≈ **−100%** |

Treating case 1 as case 3 fabricates catastrophic losses; treating case 3 as case 1
silently deletes them.

**Proposed table** `market/delisting_returns`

```
security_id, ticker, delist_date, reason_category, delist_return,
source, assumption            # free text recording exactly what was assumed
```

**Approach.** Parse the free-text `reason` column in `reference/sp500_changes` (407
events, reliable from ~2010 per ADR-010) for bankruptcy/acquisition/rebalance keywords.
Coverage before 2010 is poor, so default conservatively and record the default in
`assumption`. There is no free authoritative source for delisting returns — CRSP has
them and costs far more than this project's budget. The goal is an explicit, documented
assumption, not a perfect number.

---

### TODO-4 — Feature layer in `data/gold/`

**Why:** computed once, versioned, reused by every competitor. Otherwise the
competition measures who wrote better feature code rather than who has better signal.
For GAs specifically this is also a hard performance requirement — features must never
be recomputed inside a fitness evaluation.

**Schema.** Wide format is friendlier for ML:
`security_id, date, feature_version, <f1>, <f2>, …`
Partition by year. Keep label definitions versioned alongside (forward returns at
several horizons, vol-scaled returns, triple-barrier).

**Cross-sectional normalisation** (rank or z-score **within each date**) is essential
for the NN side — raw price levels across 500 names carry no comparable information.

**Mandatory leakage test.** Recompute the whole feature matrix with all rows after date
T deleted; assert the output for dates ≤ T is byte-identical. Any feature that fails
this has a forward-looking window. Run it in CI.

---

### TODO-5 — Point-in-time market cap

**Why:** size, value, and any cap-weighted construction need it. 329,298
shares-outstanding facts already exist across 646 companies; they are simply not
reconciled into a series.

**Source.** `xbrl_facts` where `tag IN ('CommonStockSharesOutstanding',
'EntityCommonStockSharesOutstanding')` and `unit = 'shares'`, filtered via
`query.fundamentals_asof()` so `filed_date <= as_of`. Forward-fill between filings.

⚠️ **The split trap — read carefully.** Our stored `close` is **split-adjusted**
(yfinance convention, ADR-007), while shares outstanding in a filing is the
**as-reported** count at that time. Multiplying them directly understates market cap by
the cumulative split ratio — for a company that later did a 4:1 split, by 4×.

Note also that `adj_factor_price` is **1.0 everywhere** under the `split_adjusted`
convention, so it cannot be used to undo this. Derive the cumulative split ratio
directly from `market/corporate_actions` instead:

```
split_adjusted_shares(t) = reported_shares(t) × Π(split ratios with ex-date > t)
market_cap(t)            = adj_close(t) × split_adjusted_shares(t)
```

**Sanity check:** compute today's market cap for AAPL/MSFT/NVDA and compare against a
public quote. Being wrong by a clean 4× or 10× means the split adjustment is inverted.

---

### TODO-6 — Point-in-time GICS sectors

**Why:** `gics_sector` currently comes from today's snapshot only. A company
reclassified in 2018 gets its 2026 sector applied to 2008 — any sector-neutral or
sector-rotation strategy backtested against it leaks.

**This is cheap and needs zero network.** The Wikipedia revision wikitext for all 227
monthly snapshots is **already cached on disk** (`data/_cache/wikipedia/`), and
revisions from ~2008 onward carry a `GICS Sector` column. `parse_constituents()` in
`ingest/wikipedia_history.py` currently extracts only the ticker column; extend it to
return `(ticker, sector)` and emit a `reference/sp500_sector_history` table.

Use the same header-driven column detection already there (`_find_ticker_column`) —
column order is not stable across eras. 2007 revisions use an older free-text
`Industry` column rather than GICS; treat that era as unavailable rather than
force-mapping it.

---

### TODO-7 — ALFRED vintage macro

**Why:** 7 of 18 FRED series are revised. A key is now present in `.env` and vintage
access is **verified working**.

**Endpoint.** `https://api.stlouisfed.org/fred/series/observations` with
`realtime_start` / `realtime_end` set to the as-of date returns the series *as it stood
on that date*.

**Scope this by the measured evidence** (ADR-011, measured 2026-08-27): the problem is
concentrated in chain-weighted **level** series and is driven mostly by rebasing —
GDPC1 +11.2%, INDPRO −7.4%, versus PAYEMS +0.02%, CPIAUCSL −0.13%, UMCSENT 0.00%.

So: prioritise vintages for GDPC1 and INDPRO, prefer growth rates over levels
generally, and treat the low-revision series as safe as-is. This is a smaller job than
the original blanket warning implied.

---

### TODO-8 — EODHD paid migration

Full step-by-step in `docs/RUNBOOK.md` → *Migrating to paid data*. The order matters.

**Do not skip step 1.** `verify_price_convention()` exists and currently returns
`inconclusive` (free tier gives ~1 year; the NVDA 2024-06-10 split is outside it). Run
it the day the subscription starts. EODHD documents raw OHLC but **split-adjusted
volume** — a mixed convention neither existing `SOURCE_CONVENTION` value models, so
`normalize/adjustments.py` needs a separate volume convention before EODHD volume is
trusted anywhere.

Download into `data/vault/`, not `bronze/` — it is not re-fetchable after cancellation.

**Timing recommendation:** subscribe only after TODO-1 passes its acceptance tests.
A one-month subscription is a clock; spend it downloading into a working pipeline and
measuring the survivorship delta, not debugging an engine. Present coverage is already
97% (2024), 89% (2020), 79% (2016) — the paid feed mainly buys back 2007-2012.

---

### TODO-9 — Minor / cleanup

- **4 rows where `low > open`** (HUBB 2021-05-05, NAVIV 2014-04-30, SAF 2021-08-30) out
  of 3.7M. Genuine vendor errors. Decide: drop, or repair from adjacent bars. Record
  whichever in an ADR.
- **No index weights**, so the cap-weighted benchmark cannot be reproduced from our own
  data. Use the SPY series as the benchmark; equal-weight is reproducible directly.
- **Fundamentals begin 2009-04** (XBRL mandate) against a 2007-03 membership start.
  Price-only strategies can use the full window; fundamental strategies start 2009.
- **`sp500lab quality --strict`** exits non-zero on any ERROR — wire it into CI once
  the 4 bad rows are resolved, so new ERRORs block rather than accumulate.

---

## 6. Guidance specific to genetic algorithms

This is the part most likely to be got wrong, and it is where this project's stated goal
lives.

**Speed is a design constraint, not an optimization.** A GA with population 200 over 50
generations is 10,000 fitness evaluations. At 10s per backtest that is 28 hours per run.
Features **must** be precomputed in `gold/` and evaluation **must** be vectorized over
the whole panel. Never recompute an indicator inside the fitness function.

**GAs overfit more aggressively than almost anything else**, because the search is
explicitly optimizing the metric you report. Multiple-testing control is not optional
here:

- Log **every** individual evaluated, not just the winners. The trial count is the input
  to the deflated Sharpe ratio; without it your reported Sharpe is meaningless.
- Use walk-forward with purging and an embargo. Never random K-fold — financial data is
  autocorrelated and it leaks. López de Prado's *Advances in Financial Machine Learning*
  is the reference; read it before writing the validation loop.
- Hold out a final test period (last 2–3 years) and touch it **once**. Every look
  degrades it.
- Constrain the search space deliberately. An unconstrained GA over indicator
  combinations will find something that works beautifully in-sample every single time.

**Fitness should not be raw return.** Use a risk-adjusted, cost-inclusive measure, and
penalize turnover explicitly — GAs will happily discover a strategy that trades 400
times a year and dies on costs.

---

## 7. Bugs already found — do not reintroduce

Each cost real debugging time. All are fixed; the lessons generalize.

| Bug | Lesson |
|---|---|
| yfinance pre-applies splits (`auto_adjust=False` still split-adjusts) — 10× error on NVDA | Verify vendor semantics empirically; don't trust docs |
| Wikipedia's 2007 table put Company before Ticker | Read the header; never assume column position |
| Chunk cache keyed on index, not request — silently dropped 22 securities | A cache key must derive from *everything* that determines the response |
| `SYMBOL` ingested as a constituent for 10 months | Format validation can't catch well-formed nonsense; only an independent source can |
| `write_text(encoding=...)` truncates the file before it fails to encode | Verify before writing, not after |

The third and fourth are the important ones. `verify` passed the entire time the chunk
cache was corrupt, because the file matched its own checksum perfectly — **integrity
checking proves a file is unchanged, not that it is the right file.**

---

## 8. Operating it

```bash
python -m pip install -e .
```

```bash
python -m sp500lab status
```

```bash
python -m sp500lab ingest all
```

```bash
python -m sp500lab quality
```

Full command reference and troubleshooting: `docs/RUNBOOK.md`.
Table and column reference: `docs/DATA_DICTIONARY.md`.
Why anything is the way it is: `docs/DECISIONS.md` (15 ADRs).
Worked query examples showing the right and wrong way: `scripts/explore.py`.

**Secrets:** `.env` is gitignored and holds `EODHD_API_TOKEN`. The current token is a
free-tier key that was pasted into a chat session — rotate it before upgrading to paid.

**EODHD free tier:** 20 calls/day, ~1 year of history, `/user` is free. The budget guard
in `ingest/eodhd.py` refuses rather than overruns. Do not spend calls on price history;
the universe metadata is the valuable part.

---

## 9. The one thing to preserve

This codebase is opinionated about a single idea: **make the ways a backtest lies to you
detectable.** Survivorship bias, lookahead leakage, ticker recycling, vendor
rewriting, silent cache corruption — every design decision here exists to convert one of
those from invisible into visible.

That is worth more than any particular model. If a change makes the pipeline faster or
simpler but removes a check, it is a bad trade. The measured edge available on daily
bars over 500 large caps is small; most of what looks like edge in an early backtest is
one of the failure modes above.

Build the harness so it is honest with you, and you will learn more from it than from
any architecture you put inside it.
