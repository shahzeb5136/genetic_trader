# Runbook

Operating the data layer, the backtest engine, the experiment registry and the
reports: routine tasks, troubleshooting, and the paid-data migration.

---

## First-time setup

```bash
python -m pip install -e .
```

Copy `.env.example` to `.env` and set a real contact address — **SEC EDGAR returns 403 for
requests without a descriptive User-Agent**:

```bash
cp .env.example .env
```

Then create the layout and pull everything:

```bash
python -m sp500lab init
```

```bash
python -m sp500lab ingest all
```

Expect roughly 25–40 minutes on a cold cache, dominated by SEC fundamentals (~1.5 GB) and the
Wikipedia revision history. Safe to interrupt — everything is cached and resumable.

---

## Routine operations

Refresh prices and re-derive adjustments:

```bash
python -m sp500lab ingest prices && python -m sp500lab normalize
```

Every returned series goes through an integrity gate before silver (ADR-043): a series
with more than five impossible OHLC rows, a >400% day inside its membership window on a
non-split day, or under half its own sessions present is **rejected** and listed in
`bronze/yfinance/daily_bars/ingest_date=<today>/rejected_tickers.json`. A ticker that
was in silver last time and comes back empty or rejected keeps its previous rows
(`carried_forward_tickers.json`, and a WARNING in the log). Read both files after a
refresh; then rebuild the derived layers in the order below and run `doctor`.

If a refresh still turns out bad, roll silver back to any earlier pull with zero
network — bronze keeps every pull, partitioned by fetch date:

```bash
python -m sp500lab ingest prices --from-bronze 2026-08-27
```

A replay rebuilds silver from that pull alone (nothing is carried forward from the silver
you are replacing, because that is exactly what you do not trust).

Check what exists and how big it is:

```bash
python -m sp500lab status
```

Run every check across the data and the algorithms, with one exit code — the bronze
re-hash, the silver battery strict on ERROR, the engine acceptance suite, the timing
engine's identities, and the feature-leakage rebuild (about 2.5 minutes):

```bash
python -m sp500lab doctor
```

```bash
python -m sp500lab doctor --fast        # skip the two slow stages; a commit hook
```

```bash
python -m sp500lab doctor --roster      # also every strategy: contract + no-lookahead
```

Run the data-quality battery on its own (add `--strict` to exit non-zero on any ERROR):

```bash
python -m sp500lab quality
```

Re-hash every bronze artifact against its recorded checksum:

```bash
python -m sp500lab verify
```

Query with SQL:

```bash
python -m sp500lab query --list
```

```bash
python -m sp500lab query "SELECT ticker, count(*) n FROM daily_bars GROUP BY 1 ORDER BY n DESC LIMIT 10"
```

---

## Ordering constraints

`ingest all` runs in dependency order. Running pieces individually, respect this:

```
sec-tickers  ──►  wiki-current  ──►  wiki-history  ──►  prices  ──►  fundamentals
                                          │
                                     benchmarks (trading calendar)
                                     fred, fama-french (independent)
```

- `sec-tickers` seeds the security master; run it first or tickers won't map to CIKs.
- `wiki-history` builds the membership intervals that `prices --universe ever` reads.
- `benchmarks` produces the trading calendar; `quality` needs it to distinguish a data gap
  from a market holiday. **Refreshing benchmarks moves the calendar's end**, so follow it
  with `prices` or the newest sessions have no bars and `quality` says so.
- `fred` and `fama-french` depend on nothing and nothing depends on them; `quality` uses
  both for cross-source checks against the benchmarks.
- `normalize` must run after `prices`.

---

## Common tasks

**Test a change without a full run.** Every ingest command takes `--limit`:

```bash
python -m sp500lab ingest prices --universe current --limit 20
```

**Re-parse after fixing a parser.** Do *not* pass `--force` — the raw bytes are already
cached, and a plain re-run replays from disk at zero network cost. That is the entire point of
the bronze layer.

**Force a genuine re-fetch** (only when upstream actually changed):

```bash
python -m sp500lab ingest wiki-current --force
```

**Extend the history window.** Edit `history_start` in `config/settings.toml`, then re-run
`wiki-history`. Only newly-needed revisions hit the network.

**Add a FRED series.** Add it to `SERIES` in `src/sp500lab/ingest/fred.py` with a description
and — importantly — the correct `revised` flag.

**Move the data directory** (e.g. to an external SSD):

```bash
export SP500LAB_DATA_DIR=/Volumes/ssd/sp500lab-data
```

Then run `python -m sp500lab verify` to confirm nothing was corrupted in transit. Manifest
paths are stored relative to the data root precisely so this works.

---

## Backtesting

Full narrative in `docs/BACKTEST.md`. This section is the operational side.

### First-time setup, in order

The engine reads two gold tables, so build them before the first run:

```bash
python -m sp500lab backtest build-delisting
```

```bash
python -m sp500lab backtest build-spreads
```

Then the gate. **Do not trust any backtest number until this passes:**

```bash
python -m sp500lab doctor
```

That runs the engine's acceptance suite alongside every data check. The engine suite on
its own is `backtest accept`, ~15 seconds: the headline is check 1, buy-and-hold SPY
through the engine must reproduce SPY's real total return (8.32%/yr) to a few basis
points. It currently matches to 0.2bp.

Before a release, or after touching any strategy:

```bash
python -m sp500lab backtest accept --strategies all
```

Checks 6 and 7 over the whole roster — every strategy runs and produces a portfolio,
and none of them changes its weights when the panel is truncated just past the window
(the lookahead check `context.py` cannot make, because `on_start` sees the whole panel).
A few minutes; `--include-learned` adds the model-training family.

### Daily use

```bash
python -m sp500lab backtest baselines
```

```bash
python -m sp500lab backtest run momentum_12_1 --all-costs --annual
```

```bash
python -m sp500lab backtest run low_vol --top-k 30 --max-weight 0.06 --save results/lowvol30
```

```bash
python -m sp500lab backtest coverage --annual
```

`--all-costs` reports optimistic / realistic / pessimistic side by side and warns if the
strategy is only profitable under the optimistic setting. Use it for anything you intend
to quote.

### The calendar lab (daily legs on SPY - docs/TIMING.md)

```bash
python -m sp500lab timing accept
```

```bash
python -m sp500lab timing suite
```

```bash
python -m sp500lab timing run tm_overnight --all-costs
```

```bash
python -m sp500lab timing decompose --out results/overnight_by_ticker.csv
```

`timing accept` is the lab's gate, same as `backtest accept` is the engine's: buy-and-hold
through the leg engine must match the adjusted SPY series (currently 0.00bp/yr), and
overnight × intraday must multiply back to buy-and-hold exactly. `timing seal` and
`timing forward` run calendar rules through the standard forward harness — the second
one spends looks and is permanently recorded, like every other look.

### Rebuild order after changing upstream data

The panel caches aggressively, so this order matters:

```bash
python -m sp500lab normalize
```

```bash
python -m sp500lab backtest build-spreads
```

```bash
python -m sp500lab backtest build-delisting
```

```bash
python -m sp500lab backtest build-panel
```

```bash
python -m sp500lab features build --rebuild
```

```bash
python -m sp500lab doctor
```

`build-panel` rebuilds and re-caches (~11s). The cache key is a hash of the build
parameters, **not** of the underlying data, so silver changes do **not** invalidate it
automatically — you must rebuild explicitly. That is the one manual step in an otherwise
idempotent pipeline, and forgetting it means backtesting yesterday's data. The feature
cache is keyed on the panel's actual extent (ADR-041), so a rebuilt panel with a new end
date gets fresh features; `--rebuild` forces it regardless.

### Reading the diagnostics

Every result ends with a DIAGNOSTICS block. Lines prefixed `!!` are the ones that change
what a number means:

| Line | What it means |
|---|---|
| `price_coverage` | Priced share of the true index. **Always read this** for pre-2015 runs |
| `forced_exits` | Positions resolved outside a rebalance, by category |
| `!! unresolved_exits` | Exits with no recorded reason — treated as an index removal |
| `!! spread_fallback_orders` | The spread estimator had no value; a flat default was used |
| `!! half_spread` | The gold table is missing entirely — run `build-spreads` |
| `!! ruined` | NAV reached zero; every later rebalance is a no-op |
| `unfilled_orders` | No opening bar, so the order did not fill (correct, not an error) |

### Troubleshooting

**`no such table: gold_half_spread`** — run `backtest build-spreads`. The engine will still
run without it and charge a flat fallback spread, but it says so loudly in the diagnostics.

**`strategy allocated to N name(s) not tradable`** — the strategy put weight on a security
that was not in the index that month or had no price. This is the survivorship-bias guard
working. Filter on `ctx.tradable`.

**`negative weight ... under a long-only mandate`** — the engine never normalises this away
(ADR-016). Fix the strategy.

**`only N usable rebalance date(s)`** — the window is too short, or the strategy's `warmup`
consumes it. `momentum_12_1` needs 273 sessions before its first trade.

**Backtest suddenly slow** — something is allocating per rebalance, almost always
`ctx.prices` (a DataFrame, ~1000x slower than the array accessors) inside a hot loop.
`tests/test_backtest.py::test_backtest_is_fast_enough_for_a_genetic_algorithm` asserts under
1s per run.

**Results changed with no code change** — check whether the panel cache is stale (see
rebuild order above), then whether the strategy uses `np.random` instead of `ctx.rng`.
Acceptance check 4 catches the second case.

---

## Experiments: the trial log and the holdout

Full narrative in `docs/EXPERIMENTS.md`. This is the operational side.

### What happens without you doing anything

- **Every backtest is logged** to `data/experiments/runs.jsonl`
- **Every backtest stops on 2021-12-31.** 2022-01-01 onward is the holdout (ADR-025)

### Daily use

```bash
python -m sp500lab backtest run my_algo --study my-idea --notes "value tilt, 30 names"
```

```bash
python -m sp500lab experiments studies
```

```bash
python -m sp500lab experiments list --study my-idea --sort sharpe
```

```bash
python -m sp500lab experiments deflate my-idea
```

That last one is the one that matters. It answers "would the luckiest of N worthless
strategies have scored this well anyway?" Below ~0.95, the answer is yes.

### Reaching into the holdout

```bash
python -m sp500lab backtest run winner --holdout only --study final-test
```

`--holdout include` runs the full history; `--holdout only` runs the reserved period
alone. **Both are recorded permanently** and print a warning at the time. Check the ledger
before trusting a final number:

```bash
python -m sp500lab experiments holdout
```

### Turning trial logging off

```bash
SP500LAB_REGISTRY=off python -m sp500lab backtest run scratch_idea
```

Use it for genuinely throwaway runs. Note the asymmetry: this silences the *trial* log
only. A run that touches the holdout is still recorded, always.

### Troubleshooting

**"No runs logged yet"** — either nothing has been run, or `SP500LAB_REGISTRY` is set to
off in your shell. Check with `echo $SP500LAB_REGISTRY`.

**Trial count looks too low** — you probably split one search across several `--study`
names. That understates `n_trials` and makes the deflated Sharpe too generous. The study
should match the search you actually ran.

**Trial count looks too high** — check `experiments studies`: `runs` counts log lines,
`trials` counts distinct configurations. Only `trials` feeds the deflation.

**A number does not match an older note** — the default window changed when the holdout
landed. Runs now end 2021-12-31 (~176 rebalances) rather than at the data's end (~232).
Compare `start`/`end` in `experiments show <run_id>` before comparing figures.

**`git_dirty: true` on a run you care about** — it was produced from a working tree with
uncommitted changes and cannot be reproduced exactly. Commit, then re-run.

**Registry file ends mid-line** — a write was interrupted. Nothing to do: the next append
starts a new line and `load()` skips the broken one. Only that single record is lost.

### Backing it up

`data/experiments/` belongs in the backup set **with** `vault/` and `bronze/`. You cannot
re-derive what you tried: a discarded idea leaves no trace in git, in the results
directory, or anywhere else. Losing the log does not lose a result, it loses the ability
to know whether any result was real.

`data/experiments/forward/` is the same argument one step stronger. Losing a forward
record loses the only out-of-sample evidence the project has, and losing a *seal* loses
the proof that a prediction was written before its answer was known. Neither can be
re-derived by re-running anything. See [FORWARD_TEST.md](FORWARD_TEST.md).

---

## Reports

Full narrative in `docs/REPORTS.md`.

```bash
python -m sp500lab report backtest --open
```

```bash
python -m sp500lab report forward --open
```

```bash
python -m sp500lab report timing --open
```

```bash
python -m sp500lab report genetic --open
```

One folder per lab (ADR-045, ADR-047). `reports/backtest/` and `reports/forward/` are the
monthly roster on either side of the 2022 boundary, each holding an index and one page per
algorithm and nothing else. `reports/timing/` is the calendar rules, an index and one page
per rule with both windows on each. `reports/genetic_algorithm/` is three pages on the
search. `-o DIR` writes a set somewhere else; a folder you point at is yours and is never
pruned. Single pages on demand:

```bash
python -m sp500lab report study baselines --open
```

```bash
python -m sp500lab report run --study baselines
```

```bash
python -m sp500lab report registry
```

```bash
python -m sp500lab report compare momentum_12_1 low_vol equal_weight
```

```bash
python -m sp500lab report honesty
```

Single pages go to `reports/extra/` (gitignored, like all of `reports/`). `-o path.html`
overrides the destination, `--open` launches a browser. Each file is self-contained: no
server, no network.

### Troubleshooting

**"no runs logged"** - nothing has been run, or `SP500LAB_REGISTRY` is off in your shell.

**A run is in the tables but not the charts** - it has no stored equity curve. Curves are
written with every run by default; a run logged with `log_curve=False` (as a large search
should) needs re-running to recover one. The fingerprint is unchanged, so that re-run is
the same trial, not a new one.

**The report looks stale** - it is a snapshot. The footer carries the generation time.
Rebuild after new runs; it takes under a second.

**A drawdown chart looks too shallow** - it is. Charts are drawn from month-end curves, so
an intra-month trough is invisible. The `maxDD` column comes from the daily curve and is
the number to quote.

**`cash` is highlighted as best in several columns** - correct, if uninteresting. Holding
nothing genuinely has the lowest volatility, turnover, cost and drawdown.

---

## Backups

Priority order — these are **not** equally valuable:

1. **`data/vault/`** — paid-window downloads. Irreplaceable once a subscription lapses.
1. **`data/experiments/`** — the trial log and the holdout ledger. Equally
   irreplaceable, for a different reason: a discarded idea leaves no trace anywhere else,
   and without the trial count no searched result can be deflated (ADR-026). Tiny.
   Back up to two places.
1. **`data/experiments/forward/`** — the seals and the forward-test records. The most
   irreplaceable directory in the project and the smallest. Each line consumed a period
   of out-of-sample data that can never be un-consumed, and a seal is evidence of *when*
   a prediction was made, which nothing else records (ADR-033/034). Back up with the
   vault.
2. **`data/bronze/`** — raw artifacts. Free-tier data is re-fetchable in principle, but
   Wikipedia revisions and old filings are slow to re-pull. Worth backing up.
3. **`data/_cache/`** — convenience only. Disposable.
4. **`data/silver/`, `data/gold/`** — fully rebuildable. Do not bother.

`data/` is gitignored. Commit code and docs; never commit data.

---

## Troubleshooting

**`403` from SEC.** Your User-Agent is missing or generic. Set `SP500LAB_USER_AGENT` in `.env`
to something with a real contact address.

**Wikipedia requests take ~22 seconds each.** Expected — that is server-side latency on old
revisions, not a bug. Content requests are batched 50 at a time to amortise it. If you see one
request per revision, batching has broken.

**`implausible count N` in the wiki-history log.** The plausibility guard caught a snapshot
outside 350–620 constituents. Usually means the table format changed again. Inspect the raw
wikitext (it is cached) and check `_find_ticker_column`. Failures are written to
`bronze/wikipedia/sp500_membership/parse_failures.json`.

**`quality` reports extreme moves.** Mostly ticker recycling and unrecorded reverse splits.
Cross-check the ticker against `sp500_membership_intervals` — if price history continues well
past membership end, it is a recycled symbol. Use `prices_clipped_to_membership()`.

**Adjusted prices jump at splits.** The price convention for that source is wrong. See ADR-007
and `SOURCE_CONVENTION`. Validate with `validate_against_vendor()` — it compares *returns*, not
levels, so a constant vendor offset does not mask a real error.

**`FileNotFoundError: silver dataset '...' not built yet`.** Run the ingest step that produces
it; check the ordering diagram above.

**Fundamentals job looks stalled.** It pulls ~650 CIKs at 3–4 MB each. Watch progress with:

```bash
tail -f logs/sp500lab.log
```

---

## Migrating to paid data (Phase 3)

The sequence matters. Do not skip step 1.

**1. Verify the price convention before trusting anything.** Pull NVDA across 2024-06-10 and
check whether the pre-split close reads ~\$1,210 (`as_traded`) or ~\$121 (`split_adjusted`).
Set `SOURCE_CONVENTION` accordingly. Getting this wrong is silent and looks like alpha.

**2. Download into `data/vault/`, not `bronze/`.** Vault marks data that cannot be re-fetched
after the subscription ends. Use bulk/export endpoints — download whole tables and diff
locally rather than looping per ticker.

**3. Pull the full ever-member universe**, not just current constituents. The 296 missing
delisted tickers are the entire reason for the purchase.

**4. Cross-validate** against yfinance on the ~677 overlapping tickers. Compare returns, not
levels. Investigate any ticker whose median return difference exceeds ~1bp.

**5. Diff the constituent history** against the Wikipedia reconstruction. Disagreements tell
you how much to trust the free reconstruction for the pre-purchase era.

**6. Re-run the identical backtests** from the free-data era and record the difference. That
delta is *your* survivorship bias, measured on your own universe — a far more convincing
number than any published estimate.

**7. Cancel the subscription.** Confirm `verify` passes and vault is backed up in two places
first.

---

## The genetic algorithm

```bash
python -m sp500lab evolve run --study ga-1 --seeds 3
```

```bash
python -m sp500lab experiments deflate ga-1
```

```bash
python -m sp500lab evolve ensemble ga-1 --all-costs --trades results/trades/ga-1
```

A search over the nine prior-signed families (`--preset families`, from 2010-07; or
`families-price`, the five price-visible ones from 2007-04), scored on the worst quarter
of twelve random sub-periods net of pessimistic costs, three independent seeds pooled into
an ensemble of the 30 best survivors. A quarter of an hour. Every generation of every seed
is checkpointed to `data/experiments/evolve/<study>.jsonl` and the ensemble to
`<study>.ensemble.json`; `evolve ensemble <study> --rebuild` builds one from any
checkpoint, `evolve best` shows the champion for comparison, `evolve history` the
per-generation statistics. Give every search its own `--study`: it is what decides
`n_trials` for the deflated Sharpe, and the number a search prints is uncorrected until
`experiments deflate` has run. The ensemble, not the champion, is what `forward suite`
and `report backtest` pick up (ADR-050). See [EVOLUTION.md](EVOLUTION.md).

---

## What not to do

- **Do not commit `data/`.** It is large, and some of it is licensed.
- **Do not edit anything in `bronze/`.** It is the negative. Fix parsers instead.
- **Do not use vendor `adj_close` directly.** It is rewritten on every dividend and makes
  results irreproducible (ADR-006).
- **Do not join price data on a bare ticker.** Symbols get recycled (ADR-005).
- **Do not filter fundamentals on `period_end`.** Use `filed_date` via
  `query.fundamentals_asof()`, or you leak the future.
- **Do not treat revised macro series as point-in-time.** 7 of 18 FRED series are restated
  after publication (ADR-011).
