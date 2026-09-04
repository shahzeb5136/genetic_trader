# The feature layer

79 point-in-time features, computed once, versioned, and shared by every competitor.

```bash
python -m sp500lab features build
python -m sp500lab features list
python -m sp500lab features check      # <- the one that decides whether to trust it
python -m sp500lab report features --open   # all of it, as a page you can read
```

Every feature is documented as DATA in `features/catalog.py` — what it is, which end is
historically good, and where it came from — so the report explains itself and the test
suite fails if a feature exists without an entry.

---

## Why it exists

Two reasons, and the second is the harder constraint.

**The competition would otherwise measure the wrong thing.** The stated goal of this
project is genetic algorithms against neural nets against classical rules on one shared
harness. If each competitor computes its own momentum, its own volatility and its own
valuation ratio, the scoreboard partly ranks who wrote better feature code.

**A genetic algorithm cannot afford to recompute anything.** A fitness evaluation is a
full backtest. If each one re-derives a 252-day rolling regression across 628 securities,
a 1,500-individual search stops being a thing you run over lunch. With the panel
precomputed an evaluation is a gather and a dot product: **0.15 seconds**.

---

## The shape

`(R, S, F)` float32 — 320 month-end rebalance dates × 628 securities × 79 features,
17 MB on disk under `data/gold/features/`, keyed by a hash of the panel range and
`FEATURE_VERSION`.

```python
from sp500lab.features import build_features
from sp500lab.backtest import run_backtest

feats = build_features()
run_backtest("quality_value", features=feats)   # or just run_backtest("quality_value")
```

The second form works because a strategy declares `requires_features` and the engine loads
the panel itself. That is deliberate: forgetting to pass features does not fail loudly, it
hands the strategy a context with no features, every score comes back NaN, and the run
reports a flat cash curve as though that were a result.

### Month ends only

Features exist on the sessions a strategy can act on, which under ADR-016 is the ~232
month-end rebalances rather than the 4,900 sessions between them. Measured: a daily grid
would be **590 MB**, the month-end grid is **17 MB**, and nothing in a monthly-rebalanced
strategy can use the rows in between.

`at()` refuses a row it does not hold rather than interpolating one. A silently
interpolated feature is a lookahead bug wearing a convenience API.

---

## The leakage test

This is the reason to trust any of it.

```bash
python -m sp500lab features check --cut-at 2016-12-30
```

It rebuilds the entire matrix from a price panel that **physically ends** at the cut date,
with **every filing published after it deleted**, and asserts the rows on or before the cut
are **bit-identical**. Not close — identical.

Two mechanisms, because the two families fail differently:

* a **price** feature can have a forward-looking window. Truncating the panel leaves it
  nothing to look at, so a `center=True` or a negative shift changes the answer.
* a **fundamental** can be joined on `period_end` instead of `filed_date`. Deleting later
  filings removes exactly the values it was illegally reading.

All 79 currently pass. `tests/test_features.py` runs it on every commit, plus a fast
synthetic version that needs no ingested data.

---

## What is in it

### Price, volume, liquidity — 24 features, full 2007 window

`mom_12_1` `mom_6_1` `mom_1m` `rev_1m` `ret_3m` `ret_12m` `resid_mom_12_1`
`info_discreteness` `high_52w_ratio` `trend_200d` `vol_21d` `vol_126d` `vol_of_vol_252d`
`skew_252d` `max_ret_21d` `beta_252d` `corr_mkt_252d` `idio_vol_252d` `amihud_illiq`
`log_dollar_volume` `half_spread_bp` `mom_on_12_1` `mom_id_12_1` `on_minus_id_252d`

The last three (feature_version 3, ADR-037) split every close-to-close return at the
opening bell: `mom_on_12_1` is 12-1 momentum compounded from the overnight legs alone,
`mom_id_12_1` from the intraday legs, and `on_minus_id_252d` is the trailing-year
tug-of-war spread (Lou, Polk & Skouras 2019). Log legs, so the two momenta compose back
to `mom_12_1` exactly where both cover the same sessions — and a test asserts they do.
Buildable only because the panel keeps adjusted opens beside closes; the adjustment
factor steps at the ex-date, so the dividend lands in the overnight leg exactly as it
does in the world.

Chosen so a search over them chooses between *explanations* rather than between
parameterisations of one. `resid_mom_12_1` is momentum with beta stripped out;
`info_discreteness` is Da–Gurun–Warachka's measure of *how* a return arrived rather than
how big it was; `max_ret_21d` and `skew_252d` are two measurements of the same lottery
preference; `high_52w_ratio` is George–Hwang, which is not the same information as the
return that produced it.

**The market here is the equal-weighted point-in-time index, not SPY.** For a
cross-sectional strategy choosing among index members, the cross-section *is* what it
chooses from, and it is survivorship-free in the same way the universe is.

### Membership and dividends — 8 features

`months_in_index` `log_tenure` `new_member` `div_yield` `div_growth_1y` `div_cut`
`pays_dividend` `div_due_1m`

These cannot be computed from a normal price feed. Membership is the point-in-time
reconstruction (ADR-001); dividends are discrete dated events, because an `adj_close`
column has already dissolved them into the price.

`div_due_1m` (feature_version 3) is the dividend *calendar*: 1.0 where the security's
own payment cadence — the median gap between its recent ex-dates — predicts a payment
within roughly the coming month, NaN until three payments have established a cadence.
Predicted from past ex-dates only, so it is the conservative version of what a live
trader (who sees declarations weeks ahead) would know. `div_month` trades it;
Hartzmark & Solomon (2013) is the claim.

Dividends are divided by `cum_split` at their own payment date before being accumulated —
summing pre-split and post-split dollars per share adds two different things. The
denominator is `raw_close`, not `adj_close`: dividing a per-share dividend by a
total-return-adjusted price produces a "yield" that drifts upward with age for reasons
that have nothing to do with the company.

### Fundamentals — 22 features, from 2010

`log_market_cap` `book_to_market` `earnings_yield` `cf_yield` `buyback_yield`
`gross_profitability` `roe` `accruals` `capex_intensity` `rnd_intensity` `cash_ratio`
`current_ratio` `leverage` `debt_to_assets` `asset_growth` `sales_growth`
`earnings_growth` `eps_surprise` `eps_change_yoy` `days_since_filing` `filing_lag_days`
`restatement_rate`

Every one enters on its `filed_date`, never its `period_end`. **60.8% of (security, tag,
period) combinations in this dataset have been restated**, so a naive join hands a model
numbers that existed in no form on the date it is pretending to trade.

The mechanism is a *restatement frontier*: filings sorted by when they became public, with
the running maximum of the period they describe. A re-filing of the current period updates
the value; a restatement of an older one does not — but it is counted, because how often a
company restates is one of the most interesting things this dataset knows.

The last three are the ones a single-vintage fundamentals feed cannot produce at all.
`restatement_rate` is the share of a company's published facts it has since revised,
counted only from revisions that had already happened.

**Annual durations for flow quantities, instants for the balance sheet.** Not TTM. That is
a real cost — up to three quarters of staleness — and it is stated rather than hidden:
rebuilding fiscal Q4 as `annual âˆ’ nine-month YTD` across every fiscal calendar is one sign
error away from a feature that looks like alpha, and annual is what Sloan (1996) and
Novy-Marx (2013) actually used. Quarterly EPS is the exception, because a quarterly
surprise is the whole point of `eps_surprise`.

### Macro and market state — 25 features

`vix` `vix_relative` `term_spread` `term_spread_3m` `ust10y` `fed_funds` `hy_spread`
`ig_spread` `dollar_index` `oil`, each with a 63-session change, plus `mkt_trend_200d`
`mkt_drawdown` `mkt_vol_21d` `mkt_vol_252d` `mkt_vol_ratio` `mkt_breadth_200d`.

**Only unrevised series.** 7 of the 18 FRED series in this project are revised after
publication (ADR-011) — CPI, GDP, payrolls, unemployment, industrial production, sentiment,
the recession flag. Using today's `UNRATE` for 2009-03 is a lookahead leak, and a subtle
one. Vintage access is TODO-7; until it exists those seven are absent rather than quietly
wrong.

Every macro series is lagged **one session** before sampling. Even an unrevised daily
series is published after the session it describes.

Macro columns are one number broadcast across every security. That costs 0.6 MB each and
buys a single uniform accessor: `ctx.feature("vix")` works exactly like
`ctx.feature("mom_12_1")`, and no strategy has to know which kind it asked for.

---

## Coverage, honestly

```bash
python -m sp500lab features list
```

`overall` counts every (date, security) cell **including dates a name was not in the
index**, so ~80% is the practical ceiling. `recent` is the last twelve months, which is
what a live strategy would see.

- price features: ~80% recent — the ceiling.
- fundamentals: 72–95% recent, **0% before 2010-07**, and permanently NaN for a third of
  the 973 historical index members. Those missing names are disproportionately the ones
  that were delisted, so a strategy requiring fundamentals carries a survivorship bias
  **on top of** the price-coverage gap in ADR-023.
- `hy_spread` / `ig_spread`: **11%** — FRED's keyless endpoint returns about three years of
  the licensed ICE series. Do not use them for pre-2023 regime tagging.
- `gross_profitability`, `rnd_intensity`: ~45% — only 338 and 292 companies tag them.

This is why strategies declare a `min_date` and why `sp500lab backtest suite` scores every
strategy against the index over *its own* window. See [STRATEGIES.md](STRATEGIES.md).

---

## Two bugs the build surfaced

**The as-of join collapsed a decade onto one key.** Dates were converted to integers by
dividing the int64 view by `86.4e12`, which assumes nanosecond resolution. pandas 2.x
parses these strings to `datetime64[us]`, so every date in the dataset landed on one of
about twenty integers and `merge_asof` matched filings almost at random. Nothing raised.
Fundamental coverage came out at 1.5% instead of 75% and would have read as "XBRL is
sparse".

**The split basis is the FILING date, not today.** Reported shares are expressed in the
share basis of the filing that carried them, so

    market_cap(t) = reported_shares × cum_split(filing_date) × raw_close(t)

Using `cum_split(t)` instead is right except across a split between the filing and now — a
window of at most one quarter, in which the error is the whole split ratio. Apple's 4:1 in
August 2020 would have quartered it for a quarter. The series is now continuous across it:
$1.82T in July, $2.21T in August, $1.98T in September.

**Market cap from partial share counts.** 0.79% of index-member observations computed to
under $500M — Simon Property at $1.7M, Fox at $62 — because a filer tags one share class,
or a treasury context, on its cover page. The 1st percentile of everything above the
threshold is $2.2bn, so there is a clean gap rather than a continuum. Those are discarded
and counted. It matters because market cap is a **denominator**: a 10,000-share Simon
Property has a book-to-market of 300 and any value strategy buys it first.

Both in ADR-030.

---

## Versioning

`FEATURE_VERSION` is stamped into the cache key and into every run's config. Change a
feature definition and bump it: a backtest against v1 features and one against v2 are not
comparable, and the registry has no way to know that unless the version travels with the
run.

---

## Adding a feature

1. Write it in the right family module (`price.py`, `events.py`, `fundamental.py`,
   `macro.py`). Every window trailing; NaN means "no opinion", never zero.
2. Add an entry to `features/catalog.py`: family, what it is, which end is good, where it
   came from. **The test suite fails without one** — that is what keeps the feature report
   from degenerating into a list of column names.
3. Bump `FEATURE_VERSION`.
4. `python -m sp500lab features check` — it must be bit-identical.
5. `python -m sp500lab features list` — read its coverage before using it.

Adding it to a genome preset in `strategies/genome.py` is a separate decision: every
feature in a preset multiplies the search space and the trial count the deflated Sharpe
has to discount. Since ADR-048 the search reads **families**, not features: a new feature
joins a family only with a prior story and a sign the literature settled, and a feature
without one is cut with the reason recorded in `CUT_FEATURES`. Presets are immutable, so a
new member means a new preset (ADR-038).

