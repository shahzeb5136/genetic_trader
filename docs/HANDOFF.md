# Handoff

Written 2026-08-27 for whoever picks this up next — human or model. Read this before
touching anything. It assumes no prior context.

---

## 1. What this is

A research data platform for the S&P 500, built to a **$20/month** data budget by one
individual. The long-term goal is a **competition**: genetic algorithms and other
classical strategies against neural nets, all scored on one shared backtest harness.

**Current phase: the data layer and the backtest engine are built. The feature layer is
not.** The data got built and validated first — every project of this kind dies on its
data layer — and the engine came second because it is the fitness function every
competitor will be scored by. Features (TODO-4) are next, then walk-forward validation,
then the genetic algorithms.

Start here: `docs/BACKTEST.md`, then `python -m sp500lab backtest accept`.

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
| `gold_half_spread` | 3,706,372 | Estimated half-spread per (security, date) |
| `gold_delisting_returns` | 518 | Exit assumption per security, with its reasoning |
| `experiments/runs.jsonl` | per run | Trial log: 54 fields, append-only |
| `experiments/curves.jsonl` | per run | Month-end equity curves, for reports |

**The headline: 971 tickers have been in the index since 2007-03. 501 are in it today.
470 are gone.** That gap is the entire reason this project exists.

Tests: **168 passing** (`python -m pytest tests/ -q`) — 30 data-layer, 49 engine, 42 registry, 47 reporting.
Backtest acceptance: **6 of 6 passing** (`python -m sp500lab backtest accept`).
Bronze integrity: 700 artifacts, all checksums verified, 25 tombstoned.

**The number that calibrates everything:** buy-and-hold SPY through the engine reproduces
SPY's real total return to **0.2 basis points** (8.32%/yr, 2000-01-03 → 2026-08-26).

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

7. **Never credit cash dividends in the engine.** `adj_close` already reinvests them via
   `adj_factor`, so constant adjusted shares reproduces total return by construction.
   Adding dividends on top inflates returns by ~1.9pp/yr on SPY. ADR-019, and acceptance
   checks 1 and 5 exist to catch a regression.

8. **Never fill an order at the close that generated the signal.** Execution is at the
   NEXT session's open. Encoded once in `engine.py` so no strategy can get it wrong.

9. **Never quote an early-year backtest without its coverage number.** Price coverage of
   the point-in-time index is 54.7% in 2007. The traded subset is the survivors, which is
   a second survivorship bias underneath the first. ADR-023; reported on every run.

---

## 4. Are we ready to build algorithms?

**Yes. The engine exists, it passes acceptance, and the baselines are measured.**

`src/sp500lab/backtest/` is built. TODO-1 (engine), TODO-2 (cost model) and TODO-3
(delisting returns) are **done**. Full narrative in `docs/BACKTEST.md`; the design
decisions are ADR-017 through ADR-023.

```bash
python -m sp500lab backtest accept
```

Six checks, all passing. The headline: **buy-and-hold SPY through the engine reproduces
SPY's real total return to 0.2 basis points** (8.32%/yr). That single number calibrates
the whole adjustment chain — 6.43% would mean dividends are being dropped, 10.2% would
mean they are counted twice, and you cannot land on 8.32% by accident.

A full backtest — 232 rebalances over ~500 names — runs in **~0.17s**, which makes a
10,000-evaluation genetic algorithm about 28 minutes of fitness evaluation rather than
28 hours.

### The measured baselines — read these before writing anything

2007-05-01 → 2026-08-26, $100k, monthly, long-only, realistic costs:

| Strategy | CAGR | Vol | Sharpe | maxDD | Turnover | Cost drag |
|---|---:|---:|---:|---:|---:|---:|
| low_vol | 8.57% | 14.70% | 0.63 | −39.89% | 197% | 0.62% |
| equal_weight | 10.41% | 21.35% | 0.57 | −58.25% | 39% | 0.89% |
| random_weight | 8.87% | 21.47% | 0.50 | −61.23% | 1037% | 3.03% |
| momentum_12_1 | 8.85% | 23.34% | 0.48 | −55.53% | 354% | 1.43% |
| short_reversal | 5.20% | 29.02% | 0.32 | −77.65% | 1012% | 4.20% |
| **Buy-and-hold SPY** | **10.87%** | **19.69%** | **0.62** | **−55.19%** | — | — |

**Nothing beats the index.** That is the correct null result and it is the bar. If a
genetic algorithm or a neural net beats these by a wide margin on its first run, the
overwhelmingly likely explanation is a bug in the model, not an edge.

**Costs reorder the scoreboard.** Under `optimistic` costs, `random_weight` posts
11.17%/yr and the second-best Sharpe in the suite — beating 12-1 momentum. Under
`realistic` it falls to 8.87%, under `pessimistic` to 6.57%. A strategy ranked on
optimistic costs would conclude that randomness beats momentum. This is why all three
settings are always reported.

### Gaps that will bite, by severity

**BLOCKING for anything quoted as a headline**

- **Price coverage of the point-in-time index is 54.7% in 2007** and rises to 100% today.
  343 index members have no usable price history at all. A 2007 backtest trades a
  273-name subset of a 470-name index, and that subset is *the survivors* — a second
  survivorship bias sitting underneath the point-in-time universe this project was built
  to construct. The engine reports coverage on every run (ADR-023) and `--min-coverage`
  refuses below a threshold. Closing it is TODO-8; there is no way around it in code.

- **EODHD price convention is unverified.** `verify_price_convention()` returns
  `inconclusive` on the free tier because the NVDA 2024 split predates the 1-year window.
  Run it the day the paid plan starts, before ingesting anything. Volume in particular
  would be wrong by the split ratio.

**IMPORTANT — needed before the competition is meaningful**

- **No feature layer** (TODO-4). `Context` carries a `features` slot and slices it by
  knowledge date, but `data/gold/` has no feature matrices. Until it does, every
  competitor computes its own inputs and the competition partly measures who wrote better
  feature code. **This is the next thing to build.**
- **No walk-forward harness.** The engine runs one period. Purging, embargo and a
  touch-once holdout are the GA's scaffolding and do not exist yet.
- ~~No experiment registry.~~ **Built** (ADR-025/026, `docs/EXPERIMENTS.md`). Every run
  is logged as a trial, and 2022-01-01 onward is a holdout that backtests stop before.
  Looking at it takes an explicit flag and is permanently recorded.
- **No market cap series** (TODO-5). Size, value and cap-weighting all need it. The shared
  split-ratio machinery it depends on already exists in `normalize/splits.py`.
- **Sectors are current-only** (TODO-6). Any sector-neutral strategy leaks.
- **No index weights.** The cap-weighted benchmark cannot be reproduced exactly; use the
  SPY series, which the engine already does.

**MINOR**

- 125 of 518 delisting rows are `unresolved` and default to an index removal at the last
  price. Mostly pre-2010, where `sp500_changes` is under-recorded (ADR-010). Every run
  reports how many of its exits used one.
- Fundamentals begin 2009-04 (XBRL mandate) against a 2007-03 membership start.
  Price-only strategies get the full window.
- 1 ERROR outstanding: 4 rows where `low > open` out of 3.7M.

---

## 4b. Mandate (decided 2026-08-27)

Long-only, monthly rebalance, sub-$100k capital. See ADR-016. This fixes the engine's
shape: non-negative weights summing to one, ~232 rebalance dates, and a cost model of
commission plus estimated half-spread with no impact term. Critically, monthly
rebalancing means the monthly granularity of our membership reconstruction is no longer
an accuracy limit — a strategy that only acts at month boundaries cannot be harmed by
sub-month dating error.

**One consequence that only became visible once costs were charged:** at $100k with a
$1 per-order minimum, commission is effectively a flat $1 per name traded. The
`equal_weight` baseline holds 387 names — $258 per position — and pays 92.8bp of traded
notional, almost all of it minimums. **`top_k` is an economic decision, not a tuning
parameter.** Any strategy holding more than ~50 names at this capital scale is paying for
the privilege.

---

## 5. Build order

Steps 1 and 3 are **done**. The engine came first because it is the fitness function;
the baselines came next because they are the null hypotheses.

**1. Backtest engine — DONE.** `src/sp500lab/backtest/`. One interface both model
families implement: `(point-in-time view, current positions) -> target weights`.
Vectorized, ~0.17s per run. Costs are pluggable with optimistic/realistic/pessimistic
settings and all three are always reported. Execution is at the **next** bar's open.
Acceptance: buy-and-hold SPY reproduces SPY's real total return to 0.2bp.

**2. Feature layer in `data/gold/` — NEXT.** Versioned, computed once, reused by every
competitor. Otherwise the competition measures who wrote better feature code.
Cross-sectional rank/z-score within each date is essential for the NN side. Include a
leakage test: recompute features with future data deleted and assert the output is
byte-identical. See TODO-4.

**2b. Strategy templates for all three families — DONE.** `strategies/baselines.py`
(traditional), `strategies/evolvable.py` (genome encode/decode, bounded search space,
fitness function) and `strategies/learned.py` (a model refit at every rebalance with no
training label reaching the as-of date). The GA now only has to implement selection,
crossover and mutation.

**3. Baselines — DONE.** Buy-and-hold SPY, equal-weight, 12-1 momentum, low-vol,
short-reversal, random-weight, cash. Numbers in §4. If a model cannot beat 12-1 momentum
after costs it found nothing — and note that 12-1 momentum itself does not beat SPY over
this window.

**3b. Experiment registry and holdout — DONE.** Every backtest is logged as a trial;
2022-01-01 onward is reserved. `docs/EXPERIMENTS.md`, ADR-025/026. This had to land before
any searching because both mechanisms are lossy if added later.

**3c. Static HTML reports — DONE.** `sp500lab report study|run|registry|compare|honesty`
writes a self-contained file per report. `docs/REPORTS.md`, ADR-027/028. Building it
surfaced the gap it now fills: the registry had no equity curves.

**4. Populate algorithms.** Unblocked now. Write them, tag them with `--study`, let the
registry count the trials, and read them with `sp500lab report`. Building the feature
layer against what they actually recompute beats guessing at it.

**5. Walk-forward harness.** Purging and an embargo. Required before any *searched*
result means anything (§6).

**6. Then** the genetic algorithms.

---

## 5b. TO DO — technical specifications

Ordered by dependency. TODO-1, 2 and 3 are complete; their specifications are kept below
in condensed form because the reasoning still governs the code.

Mandate constraints from ADR-016 apply throughout: **long-only, monthly rebalance,
sub-$100k capital.**

---

### TODO-1 — Backtest engine ✅ DONE

Built at `src/sp500lab/backtest/`. Full documentation: `docs/BACKTEST.md`.
Decisions: ADR-017 (structural leakage guard), ADR-018 (SPY calibration),
ADR-019 (constant adjusted shares), ADR-022 (exit buffer), ADR-023 (coverage reporting),
ADR-024 (unbiased tie-breaks).

All four originally specified acceptance tests pass, plus two added during construction:

| Check | Result |
|---|---|
| 1. SPY total return | 8.32%/yr vs 8.32% expected — **0.2bp** |
| 1b. Dividend contribution | total − price = 1.88%/yr |
| 2. Equal-weight identity | engine 11.298% vs reference 11.122% — 17.6bp |
| 3. Leakage guard | 3 of 3 deliberate cheats blocked |
| 4. Determinism | same seed identical, different seed differs |
| 5. Dividends counted once | total − price = 2.26%/yr |

Performance target met: **~0.17s** per full backtest against the "well under a second"
requirement. 46 unit tests in `tests/test_backtest.py`.

**Things learned during construction that were not in the original spec:**

- An unfilled order is not an error. A name priced at the signal date can have no bar at
  the execution date; requiring one would be lookahead. The order simply does not fill.
- **A tie-break is a modelling choice.** Breaking ranking ties on `security_id` is
  deterministic and quietly selects survivors, because id order correlates with survival
  (99.0% vs 61.1%). It made a zero-signal strategy score 17.65%/yr. ADR-024.
- Ruin is a result, not a crash. A long-only portfolio can reach zero NAV. The engine
  flatlines the curve and reports it; only a *negative* NAV is an engine bug.
- The coverage denominator matters enormously — see ADR-023.

---

### TODO-2 — Cost model ✅ DONE

`backtest/costs.py` and `backtest/spreads.py`. Decision: ADR-020.

Commission is IBKR-shaped ($0.005/share, $1.00 minimum, 1% cap). The half-spread is
`max(Corwin-Schultz, tick floor)` — the estimator cannot resolve modern large-cap
spreads (it correctly returns ~0 where the truth is 1-2bp), and the tick floor supplies
the physical lower bound the estimator cannot see. `gold/backtest/half_spread` holds
3,706,372 estimates.

The averaging convention turned out to matter more than the estimator choice: truncating
each negative two-day estimate to zero *before* averaging reports **36bp for AAPL in
2018-19 against a real ~1bp spread**. Averaging the signed estimates and truncating the
average fixes it. Details in ADR-020.

Verified: AAPL 2024 0.52bp, MSFT 2024 0.24bp, KO 2024 1.68bp; 2008-09 wider than
2010-15; pre-decimalisation wider still.

---

### TODO-3 — Delisting returns ✅ DONE

`backtest/delisting.py` → `gold/backtest/delisting_returns` (518 securities).
Decision: ADR-021.

Three categories, never conflated: `index_removal` (0), `acquisition` (0, deal terms
approximated by the last price), `bankruptcy` (−1.0). Every row carries an `assumption`
column recording in words what was assumed.

**Still true and still important:** only 4 delisting cases are visible today because
Yahoo carries almost no delisted names. Buying the paid feed raises that to hundreds —
it *exposes* this problem rather than solving it. The mechanism is now in place to
absorb that.

125 of 518 rows are `unresolved`, mostly pre-2010 where `sp500_changes` is
under-recorded. Each backtest reports how many of its exits used one.

---

### TODO-4 — Feature layer in `data/gold/`  ⬅ NEXT

**Why:** computed once, versioned, reused by every competitor. Otherwise the competition
measures who wrote better feature code rather than who has better signal. For GAs
specifically this is also a hard performance requirement — features must never be
recomputed inside a fitness evaluation.

**The engine is already waiting for it.** `Context` has a `features` slot and
`ctx.feature(name)`, and `run_backtest(features=...)` accepts a feature panel. What is
missing is the panel itself. It needs `.at(t) -> (S, F)` returning only rows knowable at
session `t`, matching the slicing discipline in ADR-017.

**Schema.** Wide format is friendlier for ML:
`security_id, date, feature_version, <f1>, <f2>, …`
Partition by year. Keep label definitions versioned alongside (forward returns at
several horizons, vol-scaled returns, triple-barrier).

**Cross-sectional normalisation** (rank or z-score **within each date**) is essential for
the NN side — raw price levels across 500 names carry no comparable information.

**Mandatory leakage test.** Recompute the whole feature matrix with all rows after date T
deleted; assert the output for dates ≤ T is byte-identical. Any feature that fails this
has a forward-looking window. Run it in CI.

**Reuse what exists.** `backtest/panel.py` already builds the aligned (date × security)
matrices, the membership mask and the trailing dollar-volume series. The feature panel
should be built on the same date/security index so the two align without a join.

---

### TODO-5 — Point-in-time market cap

**Why:** size, value and any cap-weighted construction need it. 329,298
shares-outstanding facts already exist across 646 companies; they are simply not
reconciled into a series.

**Source.** `xbrl_facts` where `tag IN ('CommonStockSharesOutstanding',
'EntityCommonStockSharesOutstanding')` and `unit = 'shares'`, filtered via
`query.fundamentals_asof()` so `filed_date <= as_of`. Forward-fill between filings.

⚠️ **The split trap.** Our stored `close` is split-adjusted (ADR-007) while shares
outstanding in a filing is the as-reported count. Multiplying them directly understates
market cap by the cumulative split ratio.

**Half of this is already built.** `normalize/splits.py::cumulative_split_ratio()` was
written for the cost model and does exactly the required job:

    split_adjusted_shares(t) = reported_shares(t) x cum_split(t)
    market_cap(t)            = adj_close(t) x split_adjusted_shares(t)

**Sanity check:** compute today's market cap for AAPL/MSFT/NVDA against a public quote.
Being wrong by a clean 4x or 10x means the split adjustment is inverted.

---

### TODO-6 — Point-in-time GICS sectors

**Why:** `gics_sector` currently comes from today's snapshot only. A company reclassified
in 2018 gets its 2026 sector applied to 2008 — any sector-neutral or sector-rotation
strategy backtested against it leaks.

**This is cheap and needs zero network.** The Wikipedia revision wikitext for all 227
monthly snapshots is **already cached on disk** (`data/_cache/wikipedia/`), and revisions
from ~2008 onward carry a `GICS Sector` column. `parse_constituents()` in
`ingest/wikipedia_history.py` currently extracts only the ticker column; extend it to
return `(ticker, sector)` and emit a `reference/sp500_sector_history` table.

Use the same header-driven column detection already there (`_find_ticker_column`) — column
order is not stable across eras. 2007 revisions use an older free-text `Industry` column
rather than GICS; treat that era as unavailable rather than force-mapping it.

---

### TODO-7 — ALFRED vintage macro

**Why:** 7 of 18 FRED series are revised. A key is present in `.env` and vintage access is
**verified working**.

**Endpoint.** `https://api.stlouisfed.org/fred/series/observations` with `realtime_start`
/ `realtime_end` set to the as-of date returns the series *as it stood on that date*.

**Scope this by the measured evidence** (ADR-011, measured 2026-08-27): the problem is
concentrated in chain-weighted **level** series and is driven mostly by rebasing —
GDPC1 +11.2%, INDPRO −7.4%, versus PAYEMS +0.02%, CPIAUCSL −0.13%, UMCSENT 0.00%.

So: prioritise vintages for GDPC1 and INDPRO, prefer growth rates over levels generally,
and treat the low-revision series as safe as-is.

---

### TODO-8 — EODHD paid migration

Full step-by-step in `docs/RUNBOOK.md` → *Migrating to paid data*. The order matters.

**This is now the single largest limitation on results, not a nice-to-have.** Price
coverage of the point-in-time index is 54.7% in 2007 (ADR-023). Every early-year number
this engine produces is over a survivor subset until this is fixed.

**Do not skip step 1.** `verify_price_convention()` currently returns `inconclusive`.
Run it the day the subscription starts. EODHD documents raw OHLC but **split-adjusted
volume** — a mixed convention neither existing `SOURCE_CONVENTION` value models, so
`normalize/adjustments.py` needs a separate volume convention before EODHD volume is
trusted anywhere.

Download into `data/vault/`, not `bronze/` — it is not re-fetchable after cancellation.

**Timing.** TODO-1 now passes its acceptance tests, so the precondition is met. A
one-month subscription is a clock: spend it downloading into a working pipeline and
measuring the survivorship delta, not debugging an engine. Present coverage is 97%
(2024), 89% (2020), 79% (2016) by year; the paid feed mainly buys back 2007-2012.

**Rebuild order after ingesting:** `normalize` → `backtest build-spreads` →
`backtest build-delisting` → `backtest build-panel --rebuild` → `backtest accept`.
Expect the delisting count to jump from 4 visible cases to hundreds, and expect the
baselines to get *worse* — that is the survivorship bias being removed.

---

### TODO-9 — Minor / cleanup

- **4 rows where `low > open`** (HUBB 2021-05-05, NAVIV 2014-04-30, SAF 2021-08-30) out of
  3.7M. Genuine vendor errors. Decide: drop, or repair from adjacent bars. Record whichever
  in an ADR.
- **No index weights**, so the cap-weighted benchmark cannot be reproduced from our own
  data. The engine uses the SPY total-return series; equal-weight is reproducible directly.
- **Fundamentals begin 2009-04** (XBRL mandate) against a 2007-03 membership start.
- **`sp500lab quality --strict`** exits non-zero on any ERROR — wire it into CI alongside
  `backtest accept` once the 4 bad rows are resolved, so new ERRORs block rather than
  accumulate.

---

## 6. Guidance specific to genetic algorithms

This is the part most likely to be got wrong, and it is where this project's stated goal
lives.

**Speed is a design constraint, not an optimization.** A GA with population 200 over 50
generations is 10,000 fitness evaluations. At 10s per backtest that is 28 hours per run.

**This is already handled on the engine side: a full backtest runs in ~0.17s**, so 10,000
evaluations is about 28 minutes. Three things buy that, and a change to any of them will
cost it — the panel is built once and memoised, the loop is per rebalance rather than per
security, and Context slices are numpy views that allocate nothing. `ctx.prices` returns a
DataFrame and is ~1000x slower; never touch it in a fitness function.

The remaining half is TODO-4: features **must** be precomputed in `gold/`. Never recompute
an indicator inside the fitness function.

**GAs overfit more aggressively than almost anything else**, because the search is
explicitly optimizing the metric you report. Multiple-testing control is not optional
here:

- Log **every** individual evaluated, not just the winners. **This is now automatic** —
  `run_backtest` appends to the registry by default. Wrap a search in
  `with registry.study("ga-run-3"):` and every individual inside is tagged, then
  `sp500lab experiments deflate ga-run-3` does the rest. See `docs/EXPERIMENTS.md`.
- Hold out a final test period and touch it **once**. **This is now enforced** —
  2022-01-01 onward is excluded by default, and every look is permanently recorded in a
  ledger that cannot be disabled (ADR-025).
- Use walk-forward with purging and an embargo. Never random K-fold — financial data is
  autocorrelated and it leaks. López de Prado's *Advances in Financial Machine Learning*
  is the reference; read it before writing the validation loop.
- Constrain the search space deliberately. An unconstrained GA over indicator
  combinations will find something that works beautifully in-sample every single time.

**Fitness should not be raw return.** Use a risk-adjusted, cost-inclusive measure, and
penalize turnover explicitly — GAs will happily discover a strategy that trades 400 times
a year and dies on costs.

**The baselines already demonstrate this failure mode.** Under `optimistic` costs
`random_weight` posts 11.17%/yr and the second-best Sharpe in the suite, beating 12-1
momentum — on 1,037%/yr turnover. Charged realistically it drops to 8.87%, and
pessimistically to 6.57%. A GA scored on optimistic costs would evolve straight into that
trap. Run fitness on `realistic` at minimum, and report all three for anything you keep.

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
| Ranking ties broken by `security_id` selected survivors — a zero-signal strategy scored 17.65%/yr | *Deterministic* is not *unbiased*. A tie-break is a modelling choice (ADR-024) |

The third and fourth are the important ones. `verify` passed the entire time the chunk
cache was corrupt, because the file matched its own checksum perfectly — **integrity
checking proves a file is unchanged, not that it is the right file.**

The last one is the newest and the most instructive for what comes next. Nothing errored.
The universe was point-in-time, the prices were adjusted, execution was at the next open,
and every acceptance check passed — including determinism, because the run *was*
reproducible, just reproducibly wrong. Survivorship bias had been designed out of the
universe and walked back in through a sort key. Anywhere this codebase imposes an
arbitrary order, ask what that order correlates with.

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

```bash
python -m sp500lab backtest accept
```

```bash
python -m sp500lab backtest baselines
```

Full command reference and troubleshooting: `docs/RUNBOOK.md`.
The engine, end to end: `docs/BACKTEST.md`.
Table and column reference: `docs/DATA_DICTIONARY.md`.
Why anything is the way it is: `docs/DECISIONS.md` (28 ADRs).
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
