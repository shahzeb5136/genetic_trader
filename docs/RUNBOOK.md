# Runbook

Operating the data layer: routine tasks, troubleshooting, and the paid-data migration.

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

## Backups

Priority order — these are **not** equally valuable:

1. **`data/vault/`** — paid-window downloads. Irreplaceable once a subscription lapses.
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
