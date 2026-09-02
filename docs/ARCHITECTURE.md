# Architecture

## The shape of the thing

```
   SOURCES                INGEST              STORAGE              QUERY
   ───────                ──────              ───────              ─────
   SEC EDGAR      ┐                      ┌─ bronze/  raw bytes
   Wikipedia      │    http_cache        │           + sha256          DuckDB
   Yahoo Finance  ├──► fetch-once   ────►├─ silver/  normalized  ────► views over
   FRED           │    rate-limited      │           parquet           parquet
   Ken French     ┘    retried           ├─ gold/    features
                                         └─ vault/   paid-window
```

Three layers, each rebuildable from the one below it. Only bronze is irreplaceable.

---

## Why bronze is sacred

In a normal project a corrupted cache file is an annoyance — you re-fetch it. This project
is built around a **burst-buy** funding model: subscribe to a paid feed for one month, bulk
download everything, cancel. After that month, the API key is dead and the raw data cannot
be re-fetched at any price short of re-subscribing.

So bronze gets treated like a photographic negative:

- **Written before parsing.** Bytes hit disk first. A parser bug then costs a re-parse, never
  a re-download. This is not hypothetical — the constituent parser in this repo was rewritten
  twice (once for the 2007 column reordering, once for the header detection), and both
  rewrites cost zero network because the wikitext was already on disk.
- **Immutable and append-only.** Partitioned by `ingest_date`, never overwritten.
- **Checksummed.** Every artifact has a SHA-256 in an append-only manifest, plus a sidecar
  `.meta.json` with the URL, byte count, and fetch time. `sp500lab verify` re-hashes all of
  it — run it after moving the data directory or restoring from backup.
- **Backed up separately.** See RUNBOOK. Silver and gold are disposable; bronze is not.

---

## The vault / tail split

Two physically separate stores, because they have completely different risk profiles:

| | **vault/** | **bronze/** (tail) |
|---|---|---|
| Contains | Downloads made during a *paid* subscription window | Free-tier and incremental data |
| Re-fetchable? | **No** — the key is gone | Yes, any time |
| Backup priority | Critical | Convenient |

The query layer stitches them together. The point is that **your code keeps working the day
your API key dies** — which it will, deliberately, as part of the funding model.

`vault/` holds EODHD's two symbol lists today (free tier, but rate-limited hard enough
that re-fetching is a real cost). It fills properly when the first paid download happens.

---

## Fetch-once discipline

Every outbound HTTP request goes through `http_cache.fetch()`, which:

1. Hashes `(method, url, body)` into a cache key. Headers are deliberately excluded, so
   rotating a User-Agent does not invalidate the whole cache.
2. Serves from `data/_cache/<source>/<hash>.bin` when present and within TTL.
   `ttl_seconds=None` means cache forever — correct for immutable history like an old
   Wikipedia revision.
3. Rate-limits per host with a token bucket (SEC publishes a 10 req/s ceiling; we run at 8).
4. Retries 429/5xx with exponential backoff plus jitter.

Measured effect: a cached response resolves in ~12ms versus ~620ms over the network, and
the 246-revision Wikipedia history job replays entirely from disk on a re-run.

**One deliberate exception:** `yfinance` manages its own HTTP session and bypasses this
layer. The fetch-once guarantee is preserved at the *artifact* level instead — if a chunk's
bronze parquet already exists for today, the chunk is skipped.

---

## Identity: why tickers are not identifiers

Three separate failure modes, all silent:

1. **Ticker changes.** FB → META is the same company. Keying on ticker splits one history
   into two.
2. **Share classes.** GOOG and GOOGL are different securities under one CIK (1652044).
   Keying on CIK merges two into one.
3. **Ticker recycling.** WM was Washington Mutual until it failed in 2008; the symbol now
   belongs to Waste Management. Keying on ticker **fabricates a single company out of two**,
   with no gap and no error to notice.

The security master resolves this with a surrogate `security_id` keyed on **(cik, ticker)**,
assigned once and never reused. Securities with no SEC registrant get `cik = 0`, so they
still receive a stable ID. Ticker punctuation is normalized (`BRK-B`, `BRK/B` → `BRK.B`)
because vendors disagree.

`quality/checks.py::check_ticker_recycling` flags 155 tickers whose price history continues
more than a year past their index membership. Use `query.prices_clipped_to_membership()`
rather than joining on a bare ticker.

---

## Bitemporality: the two dates

Any fact derived from a filing or a revision carries two dates. Conflating them is the
classic way to leak the future into a backtest.

| | Meaning | Example |
|---|---|---|
| `period_end` / `effective_date` | When the fact was **true** | Q4-2023 revenue → 2023-12-31 |
| `filed_date` / `knowledge_date` | When it became **knowable** | published in the 10-K, Feb 2024 |

A model trading on 2024-01-15 must see nothing with `filed_date > 2024-01-15`. Filtering on
`period_end` alone hands it February's numbers a month early — which looks like extraordinary
alpha and disappears the moment it trades live.

Restatements make this sharper: the same `(tag, period_end)` legitimately appears several
times with different `filed_date`s and different values. **Every version is kept.**
`query.fundamentals_asof()` takes the latest row with `filed_date <= as_of`, reproducing
what was actually on the tape that day rather than today's restated view.

Always go through that helper rather than hand-writing the SQL.

---

## Prices: raw + factors, never pre-adjusted

Stored separately, and joined at query time:

- `daily_bars` — as-received OHLCV
- `corporate_actions` — dividends and splits as discrete dated events
- `adjustment_factors` — cumulative factors we computed
- `daily_bars_adjusted` — the join, materialized for convenience

Two factors, because they are not interchangeable:

- `adj_factor` — splits **and** dividends → use for **return** calculations
- `adj_factor_price` — splits only → use for price levels and to scale volume

### Vendor conventions differ, and getting it wrong is catastrophic

"Unadjusted close" does not mean the same thing everywhere:

- **`as_traded`** — truly as-traded. Splits *and* dividends must be applied. (EODHD raw, most
  paid feeds.)
- **`split_adjusted`** — splits are *already* applied to OHLC and volume; only dividends are
  missing. **yfinance behaves this way** even with `auto_adjust=False`.

This was discovered empirically here: NVDA's 2024-06-06 close comes back from yfinance as
~\$121, not the ~\$1,210 that actually traded. Applying split factors again double-counted by
10×. The symptom is a price series that jumps by the split ratio at every split — which a
momentum model reads as enormous alpha.

The convention is declared per source in `normalize/adjustments.py::SOURCE_CONVENTION` and
stamped into the output table. **Re-verify it for any new vendor before trusting a single
bar.**

---

## Trading calendar

Derived **empirically** from SPY's own bars, not from a holiday rule set. SPY trades every
session the NYSE is open, so its date index *is* the calendar — and deriving it from the same
feed as the price data means the calendar can never disagree with the data it describes.

A rules-based calendar drifts on unscheduled closures (Hurricane Sandy 2012, the national day
of mourning in December 2018) and marks them as missing bars.

This distinction is load-bearing: a gap in a stock's history is a **data quality problem**,
while a gap on a non-trading day is **nothing at all**. Without a trustworthy calendar the
two are indistinguishable. Result: 6,706 sessions, median 252/year.

---

## Storage sizing

The whole thing is small enough to be boring, which is the point — no cloud, no cluster.

| Layer | Size (2026-09-02) |
|---|---|
| bronze (raw, incl. SEC JSON and every price pull) | ~2.4 GB |
| silver (normalized parquet) | ~350 MB |
| vault (EODHD symbol lists) | ~15 MB |
| http cache | ~2.5 GB |

DuckDB reads parquet directly with predicate pushdown, so a filtered query over 3.5M bars
touches only the columns and row groups it needs. The entire project fits on a USB stick.

---

---

## Quality: every layer is checked, and one command runs all of it

```
   sp500lab doctor                          stage        what it proves
   ───────────────                          ─────        ──────────────
   storage.verify_manifest        ───►      bronze       every byte is the byte that was fetched
   quality.checks.run             ───►      silver       schema contracts, per-table invariants,
                                                         three cross-source agreements
   backtest.accept.run_all        ───►      engine       SPY to 0.2bp, the EW identity, the leakage
                                                         guard, determinism, dividends once
   timing.engine.timing_accept    ───►      legs         overnight x intraday == buy-and-hold
   features.check_leakage         ───►      features     rebuilt with the future deleted: identical
   accept.run_strategy_checks     ───►      roster       every strategy runs; none changes its
   (--roster)                                            weights when the panel is truncated
```

Three ideas carry the weight. **A crashing check is a finding, not a skipped stage.**
**Two vendors who do not know about each other agreeing** (FRED against Yahoo on the
VIX, SPY against ^GSPC, SPY against the Fama-French market) is the cheapest evidence
there is that a series is what it claims to be. And **reviewed defects are allowlisted
by row, never repaired** (ADR-040), so `--strict` fails only on something nobody has
looked at. Full run 2.5 minutes; the data battery alone is `sp500lab quality`.

---

## The backtest engine sits on top of this

`src/sp500lab/backtest/` turns the silver layer into a scoring harness. Its one structural
idea is worth stating here because it constrains the data layer: **a strategy never sees
the panel, only a slice of it that physically ends at the as-of date.**

```
   SILVER                    PANEL                       CONTEXT
   ──────                    ─────                       ───────
   daily_bars_adjusted  ┐                           ┌─ view.close = close[:t+1]
   membership_intervals ├─► (date x security)  ────►├─ universe, tradable
   corporate_actions    │   matrices, cached        ├─ positions, cash, nav
   gold/half_spread     │   once per process        └─ features (TODO-4)
   gold/delisting       ┘
```

`panel.adj_close[:t+1]` is a numpy view — O(1), no copy — with exactly `t+1` rows, so
indexing tomorrow raises `IndexError` rather than returning a price. Lookahead stops being
a rule people must remember and becomes a thing the object cannot express (ADR-017).

The panel is built once and cached, which is what makes 10,000 genetic-algorithm fitness
evaluations take 28 minutes instead of 28 hours. Full narrative: `docs/BACKTEST.md`.

---

## Research provenance, alongside data provenance

`data/_manifest/ingest_log.jsonl` records where every byte came from. `data/experiments/`
does the same for research: `runs.jsonl` logs every backtest as a trial, and
`holdout_log.jsonl` records every look at the reserved 2022-onward period.

The reasoning is identical to why bronze is sacred. **You cannot re-derive what you
tried.** A discarded strategy leaves no trace in git or anywhere else, and without the
trial count the deflated Sharpe cannot be computed — which makes a searched result
meaningless rather than merely uncertain. So both files are append-only, and the holdout
ledger cannot be switched off even when trial logging is. See `docs/EXPERIMENTS.md`.

---

## Reporting sits on the registry, not on the engine

```
   REGISTRY                    REPORTING                      OUTPUT
   ────────                    ─────────                      ──────
   runs.jsonl    ┐        ┌─ series.py     curves.jsonl  ├──────► ├─ tables.py   ) PURE - no markup ──► one self-contained
   holdout_log   ┘        └─ views.py   /                       .html file
                                 │
                                 ▼  specs: what to draw, never how
                            render/charts.py -> SVG
                            render/html.py   -> document
```

Reports never touch the engine or a `BacktestResult`. They read the registry, which means
a report can be rebuilt for a run made months ago without re-running anything - and it
means the presentation layer can be replaced by reimplementing `render/` alone.

Everything above the specs boundary produces numbers, not tags, so a change to how a chart
looks cannot quietly change what it shows. See ADR-028 and `docs/REPORTS.md`.

---

## What is deliberately absent

- **A walk-forward harness** with purging and an embargo. The forward-test harness
  evaluates a *fixed* candidate after 2022; a walk-forward re-runs the *search* inside the
  research window. The second is the only honest evaluation loop left now that the holdout
  is spent (HANDOFF §5), and it is the next thing to build.
- **Point-in-time sectors and market cap from a paid source** (TODO-5/6). Sector ETFs
  exist as benchmarks; constituent-level sectors are current-only.
- **The 296 delisted names Yahoo cannot supply** (TODO-8). Everything the pipeline does
  is built so that the day a paid feed lands, the same commands rebuild and the same
  checks say whether the result can be trusted.

Everything that was on this list when the engine shipped — the feature layer, the
genetic algorithm, the neural net, the forward harness — has since been built, and each
arrived with its own check: `features check`, the deflated Sharpe, the seal, and now
check 7. The order was deliberate: the fitness function existed before anything was
optimised against it, and the checks existed before anything was trusted.
