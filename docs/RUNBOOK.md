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

Check what exists and how big it is:

```bash
python -m sp500lab status
```

Run quality checks (add `--strict` to exit non-zero on any ERROR, for CI):

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
```

- `sec-tickers` seeds the security master; run it first or tickers won't map to CIKs.
- `wiki-history` builds the membership intervals that `prices --universe ever` reads.
- `benchmarks` produces the trading calendar; `quality` needs it to distinguish a data gap
  from a market holiday.
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
python -m sp500lab backtest accept
```

Six checks, ~15 seconds. The headline is check 1: buy-and-hold SPY through the engine must
reproduce SPY's real total return (8.32%/yr) to a few basis points. It currently matches to
0.2bp.

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
python -m sp500lab backtest accept
```

`build-panel` rebuilds and re-caches (~11s). The cache key is a hash of the build
parameters, **not** of the underlying data, so silver changes do **not** invalidate it
automatically — you must rebuild explicitly. That is the one manual step in an otherwise
idempotent pipeline, and forgetting it means backtesting yesterday's data.

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

---

## Reports

Full narrative in `docs/REPORTS.md`.

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

Output goes to `reports/` (gitignored). `-o path.html` overrides the destination,
`--open` launches a browser. Each file is self-contained: no server, no network, 15-100 KB.

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
