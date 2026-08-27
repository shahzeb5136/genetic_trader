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
**empty**. So "start building GAs" really means "build the thing that scores GAs
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
