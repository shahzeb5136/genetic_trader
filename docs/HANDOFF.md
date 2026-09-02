# Handoff

Written 2026-08-27, extended 2026-08-28 and 2026-08-30, for whoever picks this up next —
human or model. Read this before touching anything. It assumes no prior context.

**What changed on 2026-08-28:** the trade ledger, the feature layer (TODO-4), twelve
strategies and the genetic algorithm. Jump to §4c for the summary; ADR-029 to ADR-032 for
the decisions.

**What changed on 2026-08-30:** the 2026-08 wave — four overnight/dividend features
(feature_version 3), five second-wave strategies, a shallow neural net, a daily leg
engine with nine calendar rules, a third GA search over the new features, the Algorithm
Book and Calendar Lab report pages, and a forward test of all sixteen new candidates
through the standard harness. Jump to §4d; ADR-036 to ADR-038 for the decisions, and
read the contamination note in ADR-037 before quoting any of the wave's forward numbers.

**What changed on 2026-09-02:** checks, and the data to check against. The quality
battery went from one table to the whole lake, with three cross-source agreements
(ADR-040 settles the four vendor print errors that had blocked `--strict`). Every
strategy is now run through a contract check and a no-lookahead check — the panel is
truncated just past the window and the weights must not change — and `sp500lab doctor`
runs every check across data and algorithms with one exit code (ADR-041). Daily
Fama-French factors and 24 more benchmark series (sectors, the VIX term structure, a
cross-asset regime set) joined the lake (ADR-042). A price refresh then came back
corrupt in three ways at once — `doctor` caught it, silver was rolled back to the
validated 2026-08-27 pull by replaying bronze, and the ingester gained an integrity gate
so it cannot happen again (ADR-043; the row at the bottom of §7). Bars therefore still
end 2026-08-26 while the calendar, from the refreshed benchmarks, runs to 2026-09-01;
the next `ingest prices` through the gate closes that. Nothing in the reporting layer
was touched.

---

## 1. What this is

A research data platform for the S&P 500, built to a **$20/month** data budget by one
individual. The long-term goal is a **competition**: genetic algorithms and other
classical strategies against neural nets, all scored on one shared backtest harness.

**Current phase: data layer, backtest engine, feature layer, twelve strategies and a
working genetic algorithm.** The data was built and validated first — every project of
this kind dies on its data layer — the engine came second because it is the fitness
function every competitor is scored by, and the feature layer third because a GA cannot
afford to recompute an indicator inside a fitness evaluation.

**The forward-testing harness is built and has been spent** (ADR-033/034/035,
`docs/FORWARD_TEST.md`). The whole roster — twenty registered strategies plus both
genetic-algorithm winners — was sealed as one set and tested on 2022-01 to 2026-08 under
all three cost settings. **66 looks are in the holdout ledger and the reserved period is
gone.** Anything built on it from here is research, not testing.

The result, under realistic costs: 16 held, 5 decayed, 1 failed; **2 of 22 beat the index
on risk-adjusted return**; Spearman rank correlation between the research ranking and the
forward ranking **−0.16**, with 0 of the research top five still in the forward top five.
Both genetic-algorithm winners decayed hardest — `ga-price-1-best` from a 1.15 Sharpe to
0.19 (−1.94σ), `ga-full-2-best` from 1.36 to 0.67 (−1.71σ) — which is exactly what the
multiple-testing literature predicts of a winner drawn from 1,400 individuals.

Read it with two cautions attached. `random_weight` also "held": a verdict says whether a
strategy matched its own prediction, not whether it is any good. And 2022–2026 is one
4.6-year regime dominated by mega-caps, against which every long-only equal-weighted
strategy here was structurally disadvantaged. `python -m sp500lab report forward` writes
the full set into `reports/forward_tests/`.

**What is still missing is a true walk-forward harness**, and it is the next thing worth
building. GA fitness currently measures fold *consistency* inside the research window,
which is evidence of robustness and not of generalisation (ADR-032). The forward harness
is not a substitute and never was: it evaluates a *fixed* candidate after the boundary,
where a walk-forward re-runs the *search* inside the research window. It was run before
one existed, which is the wrong order — a walk-forward would have narrowed the candidate
list first, and the set that went forward was therefore the whole roster rather than a
shortlist.

Start here: `docs/BACKTEST.md`, then `python -m sp500lab backtest accept`. If you want the
short version of what is new, `docs/TRADES.md`, `docs/EVOLUTION.md` and
`docs/FORWARD_TEST.md`.

Repo: `https://github.com/shahzeb5136/genetic_trader` (public, 2 commits)

---

## 2. State of play — verified numbers, not estimates

Every figure below was measured on 2026-09-02. Re-derive with `python -m sp500lab status`.

| Dataset | Rows | Notes |
|---|---:|---|
| `sp500_membership_intervals` | 1,020 | **Point-in-time universe.** The important one. |
| `sp500_membership_snapshots` | 114,050 | 227 monthly snapshots, 2007-03 → 2026-08 |
| `daily_bars` | 3,706,366 | 676 securities, 2000-01-03 → 2026-08-26, through the ingest gate (ADR-043) |
| `daily_bars_adjusted` | 3,706,366 | + our own adjustment factors |
| `corporate_actions` | 41,954 | 41,224 dividends, 730 splits |
| `xbrl_facts` | 3,346,513 | Point-in-time fundamentals, 649 companies |
| `fred_series` | 127,457 | 18 macro series |
| `fama_french_daily` | 26,173 | Five factors + momentum, daily since 1926, decimals (ADR-042) |
| `eodhd_us_symbols` | 111,032 | 51,206 active + 59,826 delisted |
| `trading_calendar` | 6,706 | NYSE sessions, derived from SPY, to 2026-09-01 |
| `benchmarks` | 171,135 | 29 series: the scoreboard, 11 sector SPDRs, the VIX term structure, a cross-asset regime set |
| `gold_half_spread` | 3,706,366 | Estimated half-spread per (security, date) |
| `gold_delisting_returns` | 518 | Exit assumption per security, with its reasoning |
| `data_quality` | 23 | The battery's findings: 0 ERROR, 14 WARN, 9 INFO |
| `experiments/runs.jsonl` | per run | Trial log: 54 fields, append-only |
| `experiments/curves.jsonl` | per run | Month-end equity curves, for reports |

**The headline: 972 tickers have been in the index since 2007-03. 501 are in it today.
471 are gone.** That gap is the entire reason this project exists. (972, not 971: Agilent
was missing from the universe until 2026-09-02, and Cboe had been wrongly closed out at
the end of 2018 — ADR-044.)

Tests: **627 passing** (`python -m pytest tests/ -q`, ~5 min with data) — data layer,
engine, registry, reporting, trade ledger, features, evolution, forward harness, the
timing leg engine and the frontier strategies, and (2026-09-02) every quality check on a
synthetic defect, the storage layer and the HTTP cache against a stubbed client, every
ingest parser and the price gate, the doctor's plumbing, and the strategy contract over
the whole roster.
Backtest acceptance: **5 of 5 on the engine, 50 of 50 over the roster** — every strategy
runs, and none changes its weights when the panel is truncated (`python -m sp500lab
backtest accept --strategies all`, ADR-041).
Timing acceptance: **both identities passing** (`python -m sp500lab timing accept`,
0.00bp/yr calibration, 6e-15 decomposition error).
Feature leakage: **79 of 79 bit-identical** (`python -m sp500lab features check`).
Data quality: **0 ERROR** across every silver table, the adjustment chain, both gold
tables, bronze and three cross-source checks (`python -m sp500lab quality --strict`).
Bronze integrity: 732 artifacts, all checksums verified, 25 tombstoned.
All of the above from one command: `python -m sp500lab doctor` (2.5 min, exit 0).

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

**They are built.** Twelve strategies in `strategies/alpha.py`, a 75-feature point-in-time
layer under `data/gold/features/`, and a genetic algorithm in `evolve/`. Three of twenty
strategies beat the index on risk-adjusted return over their own windows; the GA's first
run produced a winner with a deflated Sharpe of 0.9913. See §4c.

**The engine exists, it passes acceptance, and the baselines are measured.**

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

### 4c. What was built on 2026-08-28

**The trade ledger (ADR-029).** `run_backtest` records every order — side, real share
count, the AS-TRADED price a broker printed that morning, and that order's share of the
costs. `sp500lab backtest trades <strategy>` exports it as CSV; `sp500lab report trades`
publishes a page with the CSV embedded in it. Every export prints an audit: for each
rebalance `cash_after = cash_before + sum(cash_flow)`, and every dollar of cost lands on
exactly one order. Measured across all twenty strategies: **0.0 disagreement**.

*It found two things immediately.* Orders that never filled were being charged commission
(costs were priced before the fill check). And dropping sub-cent "dust" orders from the
ledger broke the cash identity by $1.26 on `equal_weight` — money charged against orders
nobody could see. Both fixed; both were invisible until somebody could look at the orders.

**The feature layer (ADR-030, this was TODO-4).** 75 features on the month-end grid, 17 MB,
versioned. `sp500lab features check` rebuilds the whole matrix from a panel that physically
ends at a past date with every later filing deleted, and asserts the earlier rows are
bit-identical. All 75 pass. Two silent bugs surfaced during the build and are documented in
the ADR — one of them (a date-ordinal conversion assuming nanosecond resolution where
pandas 2.x gives microseconds) made fundamental coverage read as 1.5% instead of 75% and
raised nothing at all.

**Twelve strategies (`docs/STRATEGIES.md`).** Four of them cannot be built without this
repository's unusual data: `pead_drift` and `restatement_averse` need `filed_date` beside
`period_end`, `index_entry_drift` needs point-in-time membership, and `dividend_grower`
needs dividends as discrete events. `sp500lab backtest suite` scores each against the index
over *its own* window, which matters more than it sounds — SPY returned 10.42%/yr from
2007-04 and 15.66%/yr from 2010-07, so a fundamentals strategy showing 17.4% is not beating
a price strategy showing 11.1%.

**The genetic algorithm (ADR-031, ADR-032).** ~0.15s per evaluation; 1,400 distinct
individuals in five minutes. First run: 12.40%/yr, Sharpe 1.15, −15.98% maximum drawdown,
deflated Sharpe **0.9913** against a bar of 0.64. Read `docs/EVOLUTION.md` including the
three reasons to be suspicious of it before quoting that anywhere.

### 4d. What was built on 2026-08-30 — the wave after the holdout

Everything below postdates the spend of the reserved period, so read ADR-037 first: the
author had seen the 2022-2026 results before choosing what to build. Nothing was fitted
to that period; the choice of what to implement was still informed by it; and the wave's
only clean test is data arriving after 2026-08.

**Four features** (feature_version 2→3, all pass the bit-identical leakage check):
`mom_on_12_1`, `mom_id_12_1`, `on_minus_id_252d` (the Lou-Polk-Skouras overnight/intraday
decomposition — buildable because the panel keeps adjusted opens) and `div_due_1m` (the
dividend calendar, from payment cadence).

**Six monthly strategies.** `strategies/frontier.py`: `overnight_momentum`,
`week52_breakout`, `div_month`, `vol_managed`, `ensemble_rank` (group `frontier`, folded
into `all`); plus `shallow_mlp` in `learned.py` — a Gu-Kelly-Xiu-shaped numpy net,
seed-ensembled, refit yearly, RollingRidge's label discipline plus a hard assert. Only
`vol_managed` beat the index in research (ΔSharpe +0.08). Studies: `frontier-1`, `mlp-1`.

**The calendar lab** (`timing/`, docs/TIMING.md, ADR-036). A leg engine — hold or don't,
at every close and open — pinned to the monthly engine by two exact identities
(calibration 0.00bp/yr; overnight × intraday = buy-and-hold to 6e-15). Nine rules, no
fitted parameters. Headline: SPY's overnight leg made 8.31%/yr gross at Sharpe 0.71
against intraday's 2.21%/0.22 — and ~500 round trips/yr of realistic costs cut the
overnight rule to 3.66%. The gap IS the finding. `sp500lab timing ...` is the CLI;
`report timing` the page; `timing decompose` the per-ticker split. Study: `timing-1`.

**A third GA search** (`ga-night-1`, preset `night`, ADR-038 — presets are immutable,
new features mean a new preset). 1,404 trials; winner 10.89%/yr, Sharpe 0.95, −17.9%
maxDD, deflated 0.9828. Forward: **decayed to 0.20** — the third GA winner in a row to
decay, now the project's most replicated result.

**Two report pages.** `report algorithms` — the Algorithm Book, every competitor
explained from its own docstring and scored on one page — and `report timing`. Both
regenerate inside `report all` and land on the index.

**The forward harness grew a `runner` parameter** so the leg engine's results flow
through the same seal → paired comparison → store pipeline as everything else. All
sixteen new candidates were sealed as one set (rationale discloses the contamination)
and tested under all three cost settings: `ga-night-1-best` and `vol_managed` decayed,
`tm_weekend` failed, the rest held. The holdout ledger now reads 117 looks; the
`selection_bar` counts 38 candidates and the bar rose to a 0.72 forward Sharpe —
anything below that is indistinguishable from the luckiest of 38.

**And one §7-grade bug, caught by its own absurdity** (ADR-037 postscript): revision 1
of `shallow_mlp` kept fitted nets across runs of one instance, and the forward harness
runs six backtests per instance — its first forward record printed an impossible 2.20
Sharpe, which is what exposed it. Fixed by a state reset in `on_start` plus a
backward-time guard, pinned by a run-twice-identical test, re-sealed as revision 2
(0.33 forward vs 0.38 research — held, mediocre both ways). The contaminated records
stay in the append-only ledgers, labelled by their own rationale.

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

- ~~No feature layer.~~ **Built** (ADR-030). 75 features, versioned, with a leakage test
  that passes bit-identically. Fundamental features begin 2010 and cover 649 of 973
  historical index members, which correlates with survival — a strategy needing them
  carries a survivorship bias *on top of* ADR-023, and runs on a shorter, kinder window.
- **No TRUE walk-forward harness.** GA fitness measures fold consistency inside the
  research window; every fold is data the winner was selected on. A real walk-forward
  re-runs the whole search inside each training window and evaluates its winner on the
  next. At 0.15s per evaluation that is about twenty minutes for five folds. **This is now
  the next thing to build**, and ADR-032 says plainly what the current mechanism is and is
  not.
- ~~No experiment registry.~~ **Built** (ADR-025/026, `docs/EXPERIMENTS.md`). Every run
  is logged as a trial, and 2022-01-01 onward is a holdout that backtests stop before.
  Looking at it takes an explicit flag and is permanently recorded.
- ~~No way to spend the holdout properly.~~ **Built and spent** (ADR-033/034/035,
  `docs/FORWARD_TEST.md`). `sp500lab forward` runs the reserved period as a
  pre-registered paired comparison and stores both legs, the decay, its standard error
  and the honesty diagnostics; `sp500lab report forward` publishes the set. **66 looks
  recorded** — the whole roster, one sealed set, all three cost settings.
- **THE HOLDOUT IS GONE.** There is no second reserved period and no way to make one.
  The only genuinely out-of-sample data this project will ever have again is the months
  that have not happened yet; `forward window` reports how many have accrued since each
  candidate was last tested. Any strategy chosen using the 2022–2026 results from here is
  being chosen in-sample, and no amount of later care undoes that.
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
- ~~1 ERROR outstanding: 4 rows where `low > open`.~~ **Resolved** (ADR-040): reviewed,
  allowlisted by row in `KNOWN_BAD_BARS`, reported as WARN. `quality --strict` exits 0
  on the real lake and any new impossible bar is still an ERROR.
- 48 tickers have price bars on disk but **none inside their membership window** — the
  whole series belongs to a later company under the same symbol. `quality` now reports
  them (`phantom_history`) and coverage counts them as missing: **625/971 usable**, not
  the 676 the old count claimed.

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

**6. Genetic algorithms — DONE.** `evolve/`, ADR-031 and ADR-032. Weighted sums of ranked
features rather than expression trees, every individual logged as a trial, fitness
measured on fold consistency, and the holdout untouched. `docs/EVOLUTION.md`.

**6b. Forward-testing harness — DONE.** `forward/`, ADR-033 and ADR-034. Pre-registered
seals, a paired research-versus-forward comparison with the standard error on the
difference, data-vintage tracking so a later look reports only what is genuinely new, and
the best-of-N correction applied to the forward window itself. `docs/FORWARD_TEST.md`.
The machine exists; the holdout is still unspent.

**6c. The forward test — DONE, and the holdout is spent.** The roster was sealed as one
set and run on 2022–2026 under all three cost settings; `report forward` publishes the
result into `reports/forward_tests/`. See the top of this document for what it found.
Testing *everything* is the honest version of running it before a walk-forward existed —
no selection was made on out-of-sample data — but it spends the period on twenty-two
candidates rather than one or two.

**7. NEXT: a true walk-forward — now for a different reason.** It can no longer decide
what to spend the holdout on, because there is nothing left to spend. What it can still
do is give future work an evaluation loop that consumes nothing: re-run the whole search
inside each training window and score its winner on the next. That is now the *only*
honest way to evaluate anything built from here, which makes it more important than it
was, not less.

**8. And then: let time pass.** The forward window grows by a month every month, and
those months are the only untainted evidence this project can still acquire.
`forward window` reports how many have accrued since each candidate was last tested, and
`fresh_months` on a later look says how much of it is new. Re-testing the same candidate
a year from now is worth exactly the twelve months of fresh data it adds.

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

### TODO-4 — Feature layer in `data/gold/` ✅ DONE

Built at `src/sp500lab/features/`, documented in `docs/FEATURES.md`, decided in ADR-030.
75 features on the month-end rebalance grid; `sp500lab features check` is the leakage test
and it passes bit-identically for all of them. The original specification is kept below
because the reasoning still governs the code.

**Departures from the original spec, and why.** Wide format and yearly partitions were
replaced by a single `(R, S, F)` float32 npz keyed by a content hash — the month-end grid
is 17 MB where a daily one is 590 MB, and nothing in a monthly-rebalanced strategy can use
the rows in between. Label definitions are not stored: no strategy here needs a label
matrix, `learned.py` builds its own inside the point-in-time view, and a versioned artifact
nobody reads rots. Cross-sectional normalisation happens at USE time
(`signals.rank_pct`) rather than at build time, because the eligibility mask is a
strategy-level choice — the exception is `features/ranked.py`, which precomputes ranks
against `panel.tradable()` purely as a GA speed optimisation and names the columns
differently so the two can never be confused.

#### Original specification

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

### TODO-5 — Point-in-time market cap ✅ PARTLY DONE

`log_market_cap` exists as a feature (`features/fundamental.py`), and `book_to_market`,
`earnings_yield`, `cf_yield` and `buyback_yield` divide by it. It is **not** written to a
standalone silver/gold table, so anything wanting market cap outside a backtest still has
to build it.

⚠️ **The split trap is real and the formula below is subtly wrong.** Reported shares are in
the share basis of the FILING date, so the correct expression is

    market_cap(t) = reported_shares * cum_split(filing) * raw_close(t)

`cum_split(t)` cancels; using `adj_close` (total-return adjusted) instead of `raw_close`
double-counts the dividend chain. Sanity check passes: NVDA $5.05T, AAPL $4.57T,
MSFT $3.69T on 2026-08-26.

⚠️ **0.79% of observations are wrong and are discarded.** Multi-class and treasury share
contexts produce a Simon Property worth $1.7M. Below $500M is treated as a data error;
the 1st percentile above it is $2.2bn, so there is a clean gap. See ADR-030.

#### Original specification

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

- ~~4 rows where `low > open`.~~ **Decided and recorded** (ADR-040): left in place,
  allowlisted by row with a reason each, reported as WARN. None can move a result.
- ~~Wire `quality --strict` into CI once the bad rows are resolved.~~ **Done, and
  broader:** `sp500lab doctor` runs the bronze re-hash, the strict data battery, the
  engine suite, the timing identities and the feature-leakage rebuild with one exit code;
  `--roster` adds every strategy. Put `doctor --fast` on a commit hook and `doctor
  --roster` before a release.
- **No index weights**, so the cap-weighted benchmark cannot be reproduced from our own
  data. The engine uses the SPY total-return series; equal-weight is reproducible directly.
- **Fundamentals begin 2009-04** (XBRL mandate) against a 2007-03 membership start.
- **Two junk tickers in the security master** (`NONE.`, `NE.WTA`), assigned from a price
  feed. Harmless — neither has membership — and flagged by `check_security_master`.
- **113 XBRL facts with |value| > 1e13** (AMD, ORCL, STX among them) — a unit or scale
  error at the filer. Flagged as WARN; the feature layer's winsorisation absorbs them, but
  a per-tag magnitude bound would be the clean fix.
- **The Fama-French library lags ~2 months.** Data through 2026-06-30 on 2026-09-02. Fine
  for research; anything reading the newest factor print should know it moves.

---

### TODO-10 — The next price refresh, and the equal-weight identity's residual

**Where it stands.** The committed data state is the validated 2026-08-27 price pull,
replayed through the ingest gate: 676 securities, bars to 2026-08-26. The 2026-09-02
refresh is in bronze and, run through the gate, gives 693 securities and bars to
2026-09-01 — seventeen delisted names Yahoo now serves cleanly (DDR, COL, MOLX, GR,
TLAB, WPI ...) are real coverage the project did not have. It is a command away:

```bash
python -m sp500lab ingest prices && python -m sp500lab normalize && python -m sp500lab backtest build-spreads && python -m sp500lab backtest build-delisting && python -m sp500lab backtest build-panel && python -m sp500lab features build --rebuild && python -m sp500lab doctor
```

**What stops it being accepted today.** `doctor` fails one check on that state: the
equal-weight identity (accept.py check 2) lands at **10.9bp** against a 10bp bound, up
from 0.1bp. The residual is the one its docstring names — proceeds from names that
delist inside the window are parked in cash by the engine and renormalised across
survivors by the reference — and it grew because seventeen more names now genuinely
delist inside 2007–2021. Two honest resolutions, and the bound must not simply be moved
without choosing one:

1. Teach `equal_weight_reference` to hold delisting proceeds in cash until the next
   execution date, the way the engine does. Then the identity is exact again and stays
   exact as coverage improves (the paid feed will add hundreds of delistings).
2. Decide the reference's renormalisation is the better model and change the engine.

Option 1 is the right one: the engine's behaviour is the mandate (a monthly rebalancer
cannot redeploy proceeds mid-month), and the reference is supposed to be the naive
restatement of it. Do it, re-run the refresh chain above, and expect check 2 back under
1bp. Also review the new impossible bars the refresh puts on APH and LEG (Yahoo's
restatement added a bad print on 2021-05-05 and 2023-06-05 to each) into
`KNOWN_BAD_BARS` or reject them — that is the ADR-040 workflow, not a bug.

---

## 6. Guidance specific to genetic algorithms

**Most of this section is now implemented rather than advisory — see `evolve/` and
`docs/EVOLUTION.md`.** It is kept because every claim below turned out to be right, and
because the next person will be tempted to relax one of them.

**Speed is a design constraint, not an optimization.** A GA with population 200 over 50
generations is 10,000 fitness evaluations. At 10s per backtest that is 28 hours per run.

**Handled: an evolved individual evaluates in ~0.15s**, so 10,000 evaluations is about 25
minutes. Four things buy that, and a change to any of them will cost it — the panel is
built once and memoised, the loop is per rebalance rather than per security, Context slices
are numpy views that allocate nothing, and **the feature ranks are precomputed once for the
whole population** (`features/ranked.py`; a rank depends on the date and the tradable mask,
not on the genome, so a 4,000-evaluation search was recomputing each one 4,000 times).
`ctx.prices` returns a DataFrame and is ~1000x slower; never touch it in a fitness
function.

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
  autocorrelated and it leaks. **Partly done:** fitness defaults to four contiguous folds
  with a one-month embargo, aggregated as `mean - 0.5*std`. That measures CONSISTENCY, not
  generalisation — every fold is inside the research window and the winner was selected
  using all of them. A true walk-forward re-runs the whole search per fold and is the next
  thing to build (ADR-032). López de Prado's *Advances in Financial Machine Learning* is
  still the reference.
- Constrain the search space deliberately. An unconstrained GA over indicator
  combinations will find something that works beautifully in-sample every single time.
  **Done:** weighted sums of cross-sectionally ranked features plus a portfolio shape —
  19 genes, no expression trees, and `describe_genome` prints any individual as sentences
  (ADR-031). If you widen this, widen it knowing that every added feature multiplies the
  trial count the deflation has to discount.

**Fitness should not be raw return.** Use a risk-adjusted, cost-inclusive measure, and
penalize turnover explicitly — GAs will happily discover a strategy that trades 400 times
a year and dies on costs. **Done:** the default objective is the MONTHLY Sharpe — the same
quantity `registry.deflate()` uses, so the search and its own significance test look at the
same number — with optional turnover and per-feature complexity penalties. Handed raw
return, a long-only GA finds leverage as concentration: `top_k` at its floor, ten names, a
magnificent CAGR and a 70% drawdown.

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
| Date ordinals computed by dividing the int64 view by 86.4e12 — pandas 2.x parses to microseconds, not nanoseconds, so a decade of filings collapsed onto one join key | Never assume a datetime's RESOLUTION. Subtract a Timestamp and take `.dt.days` |
| Unfilled orders were charged commission, because costs were priced before the fill check | A cost with no executed order behind it is invisible until somebody lists the orders (ADR-029) |
| Market cap from a cover-page share count that covered one share class — Simon Property at $1.7M | A wrong DENOMINATOR is worse than a wrong numerator: it puts the bad row at the top of a value ranking |
| `shallow_mlp` kept fitted nets across runs of one instance; the forward harness runs six backtests per instance, so a 2007 research leg scored on nets trained through 2026 — and printed a 2.20 forward Sharpe | A strategy that keeps fitted state must RESET it per run: instances outlive runs, and determinism across seeds is not determinism across reuse. The impossible number is what surfaced it (ADR-037 postscript) |
| A routine price refresh returned 34 delisted names Yahoo had never served, a third of them corrupt (a +721,000% day, 895 impossible bars), and restated two others end to end. Nothing errored; the equal-weight CAGR came out at 91%/yr | **A change to the data gets the same gate as a change to the code.** `doctor` caught it; the ingester now rejects a corrupt series before silver, carries a vanished ticker forward from the last validated rows, and `--from-bronze` replays any earlier pull — which is why bronze was partitioned by fetch date from the first commit (ADR-043) |
| `"A"` sat on the Wikipedia parser's not-a-ticker list from the first commit, swept up with the `N/A` fragments. Agilent Technologies — ticker `A`, an index member since 2000 — was therefore absent from every membership snapshot, every interval and every backtest. It surfaced only because the refresh above "dropped" it and the carry-forward guard could not carry a name that had never been requested. The cross-check written in response found a second one on its first run: Cboe (`CBOE`) had "left the index" at the end of 2018 because Wikipedia switched its row to a `{{BZX link\|...}}` template the regex did not know | **A filter list is a universe decision, and so is a regex.** Every entry on a blocklist that is shaped like a ticker must be checked against the constituent list, and every current constituent must be an open member of the intervals — because a false positive in either place is survivorship bias with no error message (ADR-044) |

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
python -m sp500lab doctor                 # every check, one exit code - run this first
```

```bash
python -m sp500lab doctor --roster        # ...and every strategy, before a release
```

```bash
python -m sp500lab quality                # the data battery on its own, with detail
```

```bash
python -m sp500lab backtest baselines
```

```bash
python -m sp500lab backtest suite all
```

```bash
python -m sp500lab backtest trades momentum_12_1
```

```bash
python -m sp500lab evolve run --study ga-1 && python -m sp500lab experiments deflate ga-1
```

Full command reference and troubleshooting: `docs/RUNBOOK.md`.
The engine, end to end: `docs/BACKTEST.md`.
The features and their leakage test: `docs/FEATURES.md`.
The strategies and the same-window scoreboard: `docs/STRATEGIES.md`.
The genetic algorithm: `docs/EVOLUTION.md`.
Exporting and auditing the orders: `docs/TRADES.md`.
Table and column reference: `docs/DATA_DICTIONARY.md`.
Why anything is the way it is: `docs/DECISIONS.md` (42 ADRs).
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
