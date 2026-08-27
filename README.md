# sp500lab

A survivorship-bias-aware research data platform for the S&P 500, built to a **$20/month
data budget**.

**Current phase: data collection only.** There are deliberately no strategies, models, or
backtests in this repo yet. The long-term goal is a competition between neural nets and
classical algorithms scored on the same backtest harness — but every one of those projects
dies on its data layer, so the data layer gets built and validated first.

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

---

## Documentation

| Doc | Read it for |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layer design, vault/tail split, why bronze is sacred |
| [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) | Every table and column |
| [SOURCES.md](docs/SOURCES.md) | Every source: cost, limits, coverage, gotchas |
| [DECISIONS.md](docs/DECISIONS.md) | ADRs — *why* things are the way they are |
| [RUNBOOK.md](docs/RUNBOOK.md) | Operating it, troubleshooting, the paid-data migration |

---

## Known limitations

These are measured, not guessed. Each is documented in full in `docs/`.

- **Price coverage is 69.6%** of ever-members (677 / 973). The missing 296 are almost all
  delisted names — Yahoo drops them. This is the exact gap the planned EODHD purchase
  closes, and it is the single largest known weakness.
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

Data layer only, in order:

1. Backfill the 33% price gap with a paid EOD feed (~$17/mo annual billing).
2. Re-run the identical pipeline and **measure the survivorship-bias delta yourself** —
   that number is worth more than any article about it.
3. Build the feature layer with leakage tests.

Only then: the backtest engine, and the competition.
