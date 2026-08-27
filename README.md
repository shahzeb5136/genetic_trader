# sp500lab

A survivorship-bias-aware research data platform for the S&P 500, built to a **$20/month
data budget**.

**Current phase: data layer + backtest engine.** The long-term goal is a competition
between genetic algorithms, neural nets and classical rules, all scored on one shared
harness. Every project of this kind dies on its data layer, so that got built and validated
first; the engine came second, because it is the fitness function everything else will be
measured by. Features (TODO-4) are next, then walk-forward validation, then the models.

The engine reproduces buy-and-hold SPY's real total return to **0.2 basis points**. Start
at [BACKTEST.md](docs/BACKTEST.md).

---

## What exists right now

Everything below was ingested from **free public sources** and validated. No paid
subscription is active.

| Dataset | Rows | What it is |
|---|---:|---|
| `sp500_membership_intervals` | 1,021 | **Point-in-time index membership** — who was in the index, when |
| `sp500_membership_snapshots` | 113,968 | 227 monthly constituent snapshots, 2007-03 → 2026-08 |
| `daily_bars` | 3,706,372 | Raw daily OHLCV, 677 securities, 2000 → present |
| `daily_bars_adjusted` | 3,706,372 | Same bars + **our own** adjustment factors |
| `corporate_actions` | 41,954 | 41,224 dividends + 730 splits, as discrete events |
| `adjustment_factors` | 3,706,372 | Split/dividend factors, computed not borrowed |
| `benchmarks` | 32,577 | SPY, RSP, ^GSPC, IWM, ^VIX |
| `trading_calendar` | 6,702 | NYSE sessions, derived empirically from SPY |
| `fred_series` | 127,457 | 18 macro series (rates, spreads, VIX, CPI, …) |
| `security_master` | 10,707 | Stable internal IDs surviving ticker changes |
| `sp500_changes` | 407 | Index add/remove events with effective dates |
| `xbrl_facts` | 3,346,513 | **Point-in-time** SEC fundamentals, 649 companies, 50 tags |
| `gold_half_spread` | 3,706,372 | Estimated half-spread per (security, date) — the cost model's input |
| `gold_delisting_returns` | 518 | What happened to each security that left the index |

### The number that matters

**973 tickers have been in the S&P 500 since March 2007. Only 502 are in it today.**

A backtest built on today's 503 constituents silently deletes 471 companies — every one
that failed, was acquired, or was demoted. Published estimates put the resulting inflation
at roughly 1.5–2.0% annually. This repo's whole reason for existing is to not do that.

Sanity check, straight from the data:

```
ticker  start_date   end_date     what actually happened
LEH     2007-03-31   2008-08-31   Lehman Brothers — removed Sept 2008
BSC     2007-03-31   2008-06-30   Bear Stearns — JPMorgan, May 2008
FNM     2007-03-31   2008-08-31   Fannie Mae — conservatorship
MER     2007-03-31   2009-02-28   Merrill Lynch — BofA
WM      2007-03-31   2008-08-31   Washington Mutual — failed
WM      2009-08-31   (open)       Waste Management — same ticker, different company
```

That last pair is why securities are keyed on an internal ID, not a ticker.

---

## Quickstart

```bash
python -m pip install -e .
```

```bash
python -m sp500lab init
```

```bash
python -m sp500lab ingest all
```

```bash
python -m sp500lab normalize
```

```bash
python -m sp500lab quality
```

```bash
python -m sp500lab status
```

Then query it with plain SQL — no server, no import step:

```bash
python -m sp500lab query "SELECT ticker, date, close FROM daily_bars WHERE ticker='AAPL' ORDER BY date DESC LIMIT 5"
```

Or from Python:

```python
from sp500lab.query import connect, universe_asof, fundamentals_asof

con = connect()
universe_asof("2008-09-30", con)          # survivorship-free constituents
fundamentals_asof("2024-01-15", con)      # only what was filed by that date
```

Then build the engine's inputs and run its acceptance checks:

```bash
python -m sp500lab backtest build-delisting && python -m sp500lab backtest build-spreads
```

```bash
python -m sp500lab backtest accept
```

```bash
python -m sp500lab backtest baselines
```

```bash
python -m sp500lab experiments studies
```

Or from Python:

```python
from sp500lab.backtest import run_backtest

res = run_backtest("momentum_12_1", start="2010-01-01", costs="realistic")
print(res.summary())
```

---

## Design rules

Five decisions do most of the work. All five are things that are cheap now and
impossible to retrofit later.

**1. Point-in-time universe, not today's list.** Membership is reconstructed from the
Wikipedia article's own revision history — the last edit of each month, parsed as it stood
that month. Re-reading today's page cannot reproduce this.

**2. Raw prices + our own adjustment factors.** Vendor "adjusted close" columns are
rewritten every time a dividend is paid, which makes backtests irreproducible. We store
as-traded OHLC and discrete corporate-action events, and derive factors ourselves. *This
was measured, not assumed* — see [ADR-006](docs/DECISIONS.md).

**3. Bitemporal fundamentals.** Every SEC fact carries `period_end` (when it was true) and
`filed_date` (when it became knowable). Querying on the former leaks the future.

**4. Bronze is immutable and checksummed.** Raw bytes land on disk before anything parses
them, with a SHA-256 in an append-only manifest. `sp500lab verify` re-hashes all of it.
Once a paid subscription lapses, a corrupted raw file is a permanent loss.

**5. Fetch once.** Every HTTP response is cached by request hash. Re-running any job
replays from disk. Free tiers make this feel unnecessary; a metered burst-buy month makes
it essential.

**6. Leakage is structural, not a rule.** A strategy never receives the price panel — it
receives a numpy view that physically ends at its as-of date, so indexing tomorrow raises
`IndexError` rather than returning a price. See [ADR-017](docs/DECISIONS.md).

**7. Every trial is counted, and the holdout is watched.** Backtests are logged
automatically and stop before a reserved 2022-onward holdout. Looking at that period takes
an explicit flag and is always recorded — trial logging can be switched off, the holdout
ledger cannot. Without the trial count, a searched Sharpe is not conservative or
optimistic; it is meaningless. See [EXPERIMENTS.md](docs/EXPERIMENTS.md).

---

## The backtest engine

```bash
python -m sp500lab backtest accept
```

Six acceptance checks. The gate is buy-and-hold SPY: the engine must reproduce SPY's real
total return, and the three ways the adjustment chain can be wrong land on three separated
numbers — 6.43%/yr means dividends were dropped, 10.2% means they were counted twice,
8.32% is correct. It currently matches to **0.2 basis points**.

```bash
python -m sp500lab backtest baselines
```

Every baseline, **2007-05 → 2021-12** (the research window; 2022 onward is a holdout),
$100k, monthly, long-only, realistic costs:

| Strategy | CAGR | Vol | Sharpe | maxDD | Turnover |
|---|---:|---:|---:|---:|---:|
| equal_weight | 11.10% | 22.67% | 0.58 | −58.25% | 39% |
| random_weight | 10.24% | 22.78% | 0.54 | −61.23% | 1026% |
| low_vol | 9.84% | 15.34% | **0.69** | −39.89% | 191% |
| momentum_12_1 | 6.39% | 23.24% | 0.38 | −55.53% | 354% |
| **Buy-and-hold SPY** | **10.42%** | **20.35%** | **0.59** | **−55.19%** | — |

Only `low_vol` (on Sharpe) and `equal_weight` (on return) clear the index — that is the
bar every model has to beat. Over the *full* 2007–2026 history nothing beat SPY at all;
2022–2026 was unusually kind to cap-weighted mega-caps. Same engine, different window,
different conclusion, which is precisely why that period is now held out.

And under *optimistic* costs `random_weight` posts the second-best Sharpe in the suite,
beating 12-1 momentum — which is why all three cost settings are always reported.

A full backtest runs in ~0.17s, so a 10,000-evaluation genetic algorithm is about 28
minutes of fitness evaluation rather than 28 hours.

All three model families implement the same interface and the engine cannot tell them
apart — `strategies/baselines.py` (traditional rules), `strategies/evolvable.py` (genome
encode/decode plus a bounded search space, ready for a GA) and `strategies/learned.py` (a
model refit at every rebalance, with no training label reaching the as-of date).

---

## Documentation

| Doc | Read it for |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layer design, vault/tail split, why bronze is sacred |
| [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) | Every table and column |
| [SOURCES.md](docs/SOURCES.md) | Every source: cost, limits, coverage, gotchas |
| [DECISIONS.md](docs/DECISIONS.md) | ADRs — *why* things are the way they are |
| [RUNBOOK.md](docs/RUNBOOK.md) | Operating it, troubleshooting, the paid-data migration |
| [BACKTEST.md](docs/BACKTEST.md) | The engine: design, writing a strategy, costs, baselines |
| [EXPERIMENTS.md](docs/EXPERIMENTS.md) | The trial log and the holdout — read before searching |
| [HANDOFF.md](docs/HANDOFF.md) | Project state and the remaining TODO list |

---

## Known limitations

These are measured, not guessed. Each is documented in full in `docs/`.

- **Price coverage of the point-in-time index is 54.7% in 2007**, rising to 100% today.
  343 index members have no usable price history at all — Yahoo drops delisted names. So a
  2007 backtest trades a 273-name subset of a 470-name index, and that subset is *the
  survivors*: a second survivorship bias sitting underneath the point-in-time universe this
  repo exists to construct. Every backtest reports its coverage ([ADR-023](docs/DECISIONS.md)).
  **This is the single largest known weakness**, and the exact gap the planned EODHD purchase
  closes.
- **Delisting returns are recorded assumptions, not measurements.** There is no free
  authoritative source. 125 of 518 securities are `unresolved` and default to an index
  removal at the last price — the wrong answer for a bankruptcy. Each carries its assumption
  in words ([ADR-021](docs/DECISIONS.md)).
- **Half-spreads are estimated**, because quote data costs more than this project's budget.
  Corwin-Schultz cannot resolve modern large-cap spreads, so a tick-size floor supplies the
  physical lower bound and binds 63% of the time ([ADR-020](docs/DECISIONS.md)).
- **No feature layer yet.** Until `data/gold/` has versioned feature matrices, each
  competitor computes its own inputs and the competition partly measures who wrote better
  feature code.
- **Membership history starts 2007-03.** Before that, Wikipedia listed constituents as
  bulleted company names with no ticker column at all, so ticker-level membership is
  unrecoverable from this source.
- **The index-change table is under-recorded before 2010** (~7 events/year recorded vs the
  ~20/year that actually occur). Prefer `sp500_membership_intervals` for that era.
- **155 tickers show likely symbol reuse** after leaving the index. Use
  `prices_clipped_to_membership()` rather than raw ticker joins.
- **Wikipedia is a volunteer-maintained secondary source.** It is the best free option, not
  ground truth. The first job after buying paid constituent data is to diff it against this.
- **Fundamentals start 2009** — XBRL only became mandatory then, so `filed_date` coverage
  begins 2009-04. Earlier filings exist on EDGAR but are not machine-readable this way.
- **Two credit-spread series are truncated to ~3 years** by FRED's keyless endpoint (licensed
  ICE data). Everything else has full history. Don't use them for pre-2023 regime tagging.

---

## Next steps

In order. Full specifications in [HANDOFF.md](docs/HANDOFF.md) §5b.

1. **Populate algorithms.** The engine takes them, every run is logged as a trial, and the
   holdout is protected. This is the part that is now unblocked.
2. **Feature layer** in `data/gold/`, versioned, with a byte-identical leakage test —
   built against what those algorithms actually recompute, rather than guessed at.
3. **Walk-forward harness** — purging and an embargo. Required before any *searched*
   result means anything.
4. **Backfill the price gap** with a paid EOD feed (~$17/mo annual billing), then re-run the
   identical pipeline and **measure the survivorship-bias delta yourself**. Expect the
   baselines to get *worse*: that is the bias being removed.
5. **Then** the genetic algorithms and the neural nets.
