# sp500lab

A survivorship-bias-aware research data platform for the S&P 500, built to a **$20/month
data budget**.

**Current phase: the competition is running.** Thirty-seven algorithms across six
families — null baselines, twelve hand-written hypotheses, a second wave of five,
learned models including a small neural net, three genetic-algorithm winners, and nine
daily calendar rules on a separate leg engine — all scored on one shared harness, all
counted as trials, and all forward-tested on the post-2022 window. Every project of this
kind dies on its data layer, so that was built and validated first; the engine came
second, because it is the fitness function everything else is measured by.

The engine reproduces buy-and-hold SPY's real total return to **0.2 basis points**, and
every strategy's orders can be exported and checked against an independent source. Start
at [BACKTEST.md](docs/BACKTEST.md), or jump to [TRADES.md](docs/TRADES.md) if you would
rather see the receipts first.

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

```bash
python -m sp500lab features build && python -m sp500lab features check
```

```bash
python -m sp500lab report all --open
```

```bash
python -m sp500lab backtest suite all
```

```bash
python -m sp500lab timing accept && python -m sp500lab timing suite
```

```bash
python -m sp500lab evolve run --study ga-1 --generations 25 --population 60
```

```bash
python -m sp500lab experiments deflate ga-1
```

```bash
python -m sp500lab report study baselines --open
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

Results visualise into a single self-contained HTML file — no server, no CDN, no build
step — with equity curves, drawdowns, a sortable leaderboard and the deflated Sharpe:

```bash
python -m sp500lab report study baselines --open
```

All model families implement the same interface and the engine cannot tell them apart —
`strategies/baselines.py` (null hypotheses), `strategies/alpha.py` (twelve hypotheses),
`strategies/frontier.py` (the second wave), `strategies/evolvable.py` (the genome the GA
searches) and `strategies/learned.py` (models refit at every rebalance, with no training
label reaching the as-of date — a ridge and a small seed-ensembled MLP). The calendar
rules live in their own leg engine (`timing/`) because their granularity is sub-day, and
they share everything else: the cost model, the registry, the holdout ledger and the
forward harness.

---

## Everything as reports

```bash
python -m sp500lab report all --open
```

One self-contained HTML page per strategy, one for the feature layer, and an index that
links them together. No server, no build step, no Python needed to read any of it — the
folder is the deliverable.

Each strategy page carries, in this order: **what it claims** (taken from the strategy's
own docstring, so the report and the code cannot drift apart), the headline against the
index over its own window, the equity curve and drawdown, every calendar year, rolling
Sharpe and a monthly heatmap, all three cost settings, **every order as a downloadable
CSV embedded in the page**, what it would hold today, and finally every reason to distrust
the numbers above.

The scoreboard on the index says which strategies were **written** and which were
**evolved**, with the deflated Sharpe beside each — because a searched strategy's Sharpe is
the maximum over every configuration the search tried, and printing that in the same sorted
column as a hand-written one without saying so would be the most misleading thing this
project could do.

```bash
python -m sp500lab report strategy quality_value --open   # just one
python -m sp500lab report features --open                 # the 79 features, explained
python -m sp500lab report algorithms --open  # the Algorithm Book: every competitor,
                                             # explained in its own words and scored
python -m sp500lab report timing --open      # the Calendar Lab: overnight vs intraday
```

---

## Write your own

[`src/sp500lab/strategies/custom.py`](src/sp500lab/strategies/custom.py) is yours. Nothing
else writes to it, and it ships with one working example.

```python
@register("my_idea")
class MyIdea(FeatureStrategy):
    """One sentence somebody could argue with."""
    requires_features = ("gross_profitability", "vol_126d")
    construction = STANDARD

    def score(self, ctx):
        e = ctx.tradable
        return blend([rank_pct(self.f(ctx, "gross_profitability"), e),
                      rank_pct(-self.f(ctx, "vol_126d"), e)])
```

Add the name to `GROUPS["custom"]` and everything else follows automatically — the
registry logs it, the holdout guard protects it, the cost model charges it, the trade
ledger records its orders, and `report all custom` publishes it.

Return a **score**, not weights: `portfolio.py` handles the top-k cut, the per-name cap,
the long-only check and the unbiased tie-break, so the scoreboard compares your idea
against everyone else's rather than your position sizing against theirs. Full contract in
[ADDING_A_STRATEGY.md](docs/ADDING_A_STRATEGY.md).

---

## Show me the trades

An equity curve is a claim. A list of orders is the evidence for it.

```bash
python -m sp500lab backtest trades momentum_12_1
```

```bash
python -m sp500lab report trades momentum_12_1 --open
```

Every order, with the **as-traded** price a broker printed that morning and the real share
count — checkable against any quote source — alongside the adjusted figures the accounting
used:

```
signal_date  date        ticker  side  shares   price   notional  commission  spread_cost
2007-04-30   2007-05-01  AAPL    BUY   20.0601  99.59   1997.78    1.00        0.2008
```

Each export prints its own audit. For every rebalance,
`cash_after = cash_before + Σ cash_flow`, and every dollar of cost in the headline lands on
exactly one order. Measured across every baseline and all twelve strategies: **0.0
disagreement**. The HTML report embeds the CSV in the page itself, so a report sent on its
own still carries its evidence.

Two real bugs this exposed on the day it was built: costs charged to orders that never
filled, and a $1.26 cash break from "dust" orders that were charged but not shown. Full
detail in [TRADES.md](docs/TRADES.md) and ADR-029.

---

## The feature layer

79 point-in-time features, computed once, versioned, shared by every competitor:
momentum and residual momentum, the overnight/intraday decomposition of both, volatility
and idiosyncratic volatility, liquidity, membership tenure, dividend behaviour and the
dividend calendar, bitemporal fundamentals, and unrevised macro.

```bash
python -m sp500lab features check
```

That command is the reason to trust any of it. It rebuilds the whole matrix from a price
panel that **physically ends** at a past date, with **every filing published after it
deleted**, and asserts the earlier rows are **bit-identical**. All 79 pass.

Three of the features cannot be computed from an ordinary data feed at all —
`restatement_rate`, `filing_lag_days` and `months_in_index` — because they need
`filed_date` alongside `period_end`, and point-in-time membership. That is what the data
layer was built for. See [FEATURES.md](docs/FEATURES.md).

---

## Twelve strategies, and the only comparison that means anything

```bash
python -m sp500lab backtest suite all
```

Strategies here do not all run over the same window — anything built on XBRL fundamentals
starts in 2010 because XBRL does, and 2010-2021 was far kinder than 2007-2021. SPY returned
**10.42%** over the first window and **15.66%** over the second. So the suite puts every
strategy next to the index over *its own* dates:

| strategy | window | CAGR | Sharpe | maxDD | bench CAGR | bench Sharpe | ΔSharpe |
|---|---|---:|---:|---:|---:|---:|---:|
| low_vol | 2007-04 | 9.84% | 0.69 | −39.9% | 10.42% | 0.59 | **+0.10** |
| defensive_regime | 2007-04 | 9.21% | 0.65 | **−23.6%** | 10.42% | 0.59 | **+0.06** |
| lottery_averse | 2007-04 | 9.91% | 0.65 | −42.5% | 10.42% | 0.59 | **+0.06** |
| equal_weight | 2007-04 | 11.10% | 0.58 | −58.3% | 10.42% | 0.59 | −0.01 |
| illiquidity_carry | 2007-04 | 13.77% | 0.56 | −67.2% | 10.42% | 0.59 | −0.03 |
| accrual_quality | 2010-07 | **17.17%** | 0.86 | −45.5% | **15.66%** | **0.95** | −0.10 |
| momentum_12_1 | 2007-04 | 6.39% | 0.38 | −55.5% | 10.42% | 0.59 | −0.21 |

**Three of twenty beat the index on risk-adjusted return.** `accrual_quality` is the trap
the table exists to spring: 17.17%/yr is the best number in the file until you notice SPY
made 15.66% over the same dates at a higher Sharpe. A leaderboard sorted on CAGR would have
crowned it. Full set, and what each one claims, in [STRATEGIES.md](docs/STRATEGIES.md).

---

## The second wave: WHEN, not just WHICH

Five more hypotheses (`strategies/frontier.py`), chosen to cover mechanisms the first
twelve do not touch, plus a deliberately small neural net (`shallow_mlp`, Gu–Kelly–Xiu
shaped: features→32→16→1, seed-ensembled, refit yearly, no training label reaching the
as-of date). Research window, realistic costs, against SPY's 0.60 Sharpe:

| strategy | Sharpe | ΔSharpe | the claim |
|---|---:|---:|---|
| `vol_managed` | 0.67 | **+0.08** | volatility is forecastable, returns are not — scale exposure by inverse variance (Moreira & Muir, long-only half) |
| `overnight_momentum` | 0.55 | −0.05 | momentum's profits accrue while the market is closed — rank on the overnight component alone (Lou, Polk & Skouras) |
| `ensemble_rank` | 0.54 | −0.06 | the plain average of all twelve hypotheses |
| `week52_breakout` | 0.52 | −0.08 | anchoring to the 52-week high (George & Hwang) |
| `div_month` | 0.47 | −0.13 | the dividend-month premium, off a payment-cadence calendar (Hartzmark & Solomon) |
| `shallow_mlp` | 0.38 | −0.22 | interactions among the 55 well-populated features |

The cleanest experiment in the wave: `overnight_momentum` vs `momentum_12_1` — same
window, same construction, same costs, one difference — posts **11.85%/0.55 against
6.39%/0.38**. Splitting momentum at the opening bell carries real information (it needed
the adjusted *opens* this panel keeps and most data layers throw away). It still does not
beat the index. And the only new entrant that does, `vol_managed`, forecasts nothing
cross-sectional at all — it just owns less of the market when the market is turbulent.

**Everything in this wave was written after the 2022–2026 forward test was read**, and
that contamination is recorded in ADR-037 rather than waved away: nothing was fitted to
the period, but the author knew its character, and the wave's only clean test is the
months that arrive after 2026-08.

---

## The calendar lab: buy at close, sell at open

A second, deliberately small engine (`timing/`, [TIMING.md](docs/TIMING.md), ADR-036)
for claims the monthly engine cannot express: positions entered at one close and exited
at the next open. A session has two tradable legs — close→open and open→close — and any
calendar rule is two boolean vectors over the sessions. It earns trust by identity:
buy-and-hold through the leg engine reproduces the adjusted SPY series to **0.00 bp/yr**,
and the overnight and intraday strategies *partition* it — their NAV product equals it to
float precision at every session.

```bash
python -m sp500lab timing accept && python -m sp500lab timing suite
python -m sp500lab timing decompose          # per-ticker overnight/intraday split
python -m sp500lab report timing --open      # the whole lab as one page
```

The headline measurement, 2007-04 → 2021-12, gross: **SPY's overnight leg made 8.31%/yr
at Sharpe 0.71 with a −29% max drawdown; its intraday leg made 2.21%/yr at 0.22 with
−47%.** The market really does pay its night shift — and then the costs eat it: the
overnight rule trades ~500 times a year, so realistic costs cut it to 3.66%/yr and
pessimistic to 0.67%. **The gap between gross and net is the finding.** Nine rules run
through the family — overnight, intraday, weekend, turn-of-month, month-end drift,
pre-holiday, sell-in-May, and a VIX-gated overnight — and under realistic costs nothing
beats buy-and-hold; the family's best trial deflates to 0.69 against its own 27-trial
search. A per-ticker decomposition (`timing decompose`) shows where the anomaly lives
name by name, and the tradable expression of the same fact is `overnight_momentum` in
the monthly engine, twelve trades a year instead of five hundred.

---

## The genetic algorithm

```bash
python -m sp500lab evolve run --study ga-1 --generations 25 --population 60
```

1,400 distinct individuals in about five minutes — 0.15s per evaluation, because the panel
is memoised, the feature ranks are precomputed once for the whole population, and each
individual is one backtest whose folds are sliced from the resulting curve.

The first real search (`ga-price-1`) produced **12.40%/yr, Sharpe 1.15, −15.98% maximum
drawdown** against SPY's 10.42% / 0.59 / −55.19% over the same window, and it survives
under pessimistic costs (11.91%). Then the number that decides whether any of that means
anything:

```
n_trials                           1400
expected_max_sharpe_annualised     0.6395     <- what the luckiest of 1,400 would post
sharpe_annualised_monthly          1.3198
deflated_sharpe                    0.9913     >= 0.95, so it survives its own search
```

The winner reads as sentences, which is why the search space is bounded to weighted sums of
ranked features rather than expression trees:

```
    -1.00  div_yield              (LOW is good)
    -0.89  vol_126d               (LOW is good)
    -0.80  beta_252d              (LOW is good)
    +0.38  high_52w_ratio         (high is good)
    ...
Holds the top 12 by score, equal-weighted, capped at 6.0% per name.
De-risks to 24% invested when the index is below its 200-day average.
```

It rediscovered low volatility and low beta on its own. A second search with fundamentals
switched on (`ga-full-2`, 2010-07 onward because XBRL starts there) found **21.93%/yr,
Sharpe 1.36, information ratio 0.78 against SPY's 15.66% / 0.95**, deflated Sharpe 0.9994 —
and it independently rediscovered quality investing: high ROE, high gross profitability,
low accruals. Nobody told it Novy-Marx or Sloan exist.

**Be suspicious anyway.** `ga-price-1`'s −16% drawdown comes from a two-parameter regime
switch tuned on a window containing 2008 and 2020; it holds only 12 names; `ga-full-2`
trades a survivor-biased subset on a kinder decade; and every fold either was scored on
sits inside the research window. [EVOLUTION.md](docs/EVOLUTION.md).

A third search (`ga-night-1`, ADR-038) ran over a new preset: the price features plus the
overnight/intraday decomposition and the dividend calendar. 1,404 trials, winner
**10.89%/yr, Sharpe 0.95, −17.9% max drawdown**, deflated Sharpe **0.9828** — and inside
the blend it weighted *intraday* momentum positively and overnight momentum slightly
negatively, the reverse of the standalone result, which is exactly the kind of
disagreement between a search and the literature that forward data exists to settle.
Forward-tested the same week: **it decayed to a 0.20 Sharpe** — the third consecutive GA
winner to decay out of sample, which is by now the most replicated finding this project
has produced.

---

## Forward testing: after 2022

Everything above is measured on 2007-04 → 2021-12. Strategies were written, tuned, evolved
and ranked inside that window, so none of those numbers is out-of-sample. The reserved
period from 2022-01-01 is, and `sp500lab forward` is the machinery for spending it.

```bash
python -m sp500lab forward window                          # free: what exists, what it proves
python -m sp500lab forward seal low_vol --rationale "..."  # free: pre-register the prediction
python -m sp500lab forward run low_vol --dry-run           # free: what a look would cost
python -m sp500lab forward run low_vol                     # THIS SPENDS THE HOLDOUT
python -m sp500lab forward scoreboard                      # prediction against outcome
```

A forward test here is a **paired comparison**, never a number. The research window is the
prediction; the forward window is the outcome; the record carries both and the standard
error of the gap between them:

```
                             research      forward       change
  months                          175           54
  CAGR                          9.84%        5.43%       -4.41%
  Sharpe (monthly)               0.85         0.54        -0.31
  vs benchmark Sharpe            0.10        -0.34        -0.44
  max drawdown                -39.89%      -16.20%      +23.69%

  Sharpe 95% band        [-0.40, 1.48]  on 54 months of forward data
  decay significance     -0.56 sigma (p=0.289 that a drop this large is sampling noise)
  P(true SR > research)  0.257

  DECAYED: it beat the index over the research window and did not over the forward one
  (+0.10 then -0.34 Sharpe against the index). It still made money; it no longer made it
  better than buying the index.
```

Three ideas do the work, and each addresses something the holdout alone does not:

**54 months cannot confirm anything.** The standard error of a Sharpe over that span is
about 0.47, so a Sharpe of 1.0 has a 95% band of [0.06, 1.94] and the smallest difference
from the research window the test could resolve is ~1.35. A forward test here can *refute*
a strategy and cannot *confirm* one, so the verdicts are `failed` / `decayed` / `held` /
`inconclusive` and `held` means **not refuted**. The band is printed next to every one.

**Pre-registration, because the holdout stops fitting and not choosing.** Forward-test
twenty strategies and report the three that worked, and the reserved period has quietly
become a second research window with every individual run honest. So a candidate is
*sealed* first — configuration, prediction, trial count, and a written reason — and the
record says whether that seal was `declared` in advance or written at the moment of the
look. `store.selection_bar()` then applies the same best-of-N correction to the forward
window that the deflated Sharpe applies to a search.

**Out-of-sample data keeps arriving.** The window grows by a month every month. A second
look next year mostly re-reads what it already saw — except the new months, which are
genuine fresh evidence — so every record stores its data vintage and later looks report
`fresh_months`.

Nothing is spent by accident: `window`, `seals`, `seal` and `--dry-run` are all free, and
`run` prints what it is about to consume before consuming it. Every look lands in the same
unsilenceable ledger `backtest --holdout only` writes to.

### What happened when it was run

The whole roster — twenty registered strategies plus both genetic-algorithm winners —
was sealed as one set and tested on 2022-01 to 2026-08. Under realistic costs:

```
  22 candidates    16 held    5 decayed    1 failed
  2 of 22 beat the index on risk-adjusted return over the forward window
  Spearman rank correlation, research ranking vs forward ranking:  -0.16
  Top five by research still in the top five forward:               0 of 5
```

**The two most heavily searched candidates decayed the most.** `ga-price-1-best` went
from 12.40%/yr and a 1.15 Sharpe in research to 1.48%/yr and 0.19 forward (−1.94σ);
`ga-full-2-best` from 21.93% and 1.36 to 8.38% and 0.67 (−1.71σ). That is exactly what
the multiple-testing literature predicts of a winner selected from 1,400 individuals,
and it is the clearest result this project has produced.

Two cautions before reading any of that as settled. `random_weight`, which picks its
holdings at random, also "held" — a verdict is a statement about matching a prediction,
not about quality. And 2022–2026 was one 4.6-year regime dominated by a handful of
mega-caps, against which every long-only equal-weighted strategy here was structurally
disadvantaged whatever its signal.

**The 2026-08 wave went through the same machinery** — sixteen more candidates (the
second wave, the neural net, the night GA winner, the nine calendar rules) sealed as one
set and tested, with their ADR-037 contamination disclosed in the seal rationale itself.
The pattern repeated on cue: `ga-night-1-best`, the wave's most heavily searched
candidate (1,404 trials, deflated Sharpe 0.98 in research), **decayed from a 0.95 Sharpe
to 0.20**; `tm_weekend` failed outright; most fixed rules held, as predictions of nearly
nothing tend to.

And the wave produced one more bug for the §7 table. `shallow_mlp`'s first forward
record printed a 2.20 forward Sharpe against a 0.38 research run — a miracle, which per
this project's own rule meant a bug, and it was: the model kept its fitted nets on the
instance, and the forward harness runs six backtests on one instance, so later legs
scored on nets trained on earlier legs' *futures*. The absurd number is what surfaced
it. Revision 2 resets state per run, carries a new fingerprint and a new seal, and
measures **0.33 forward against 0.38 research — held, and mediocre both ways**. The
revision-1 records stay in the append-only ledgers as what they are: the honest record
of a leak the harness caught.

```bash
python -m sp500lab report forward --open
```

writes the full set into `reports/forward_tests/`: an executive summary, one technical
report per candidate, the cross-sectional decay analysis, an honesty page, Markdown
copies of all of it, and the raw CSVs. Nothing in that path runs a backtest — every
figure comes out of the stored record.

[FORWARD_TEST.md](docs/FORWARD_TEST.md) ·
[ADR-033, ADR-034 and ADR-035](docs/DECISIONS.md)

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
| [FORWARD_TEST.md](docs/FORWARD_TEST.md) | Testing after 2022: pre-registration, decay, and what 54 months can prove |
| [REPORTS.md](docs/REPORTS.md) | Static HTML reports: what they show, and the view-layer split |
| [ADDING_A_STRATEGY.md](docs/ADDING_A_STRATEGY.md) | **Write your own** — the contract, the helpers, and what will bite |
| [FEATURES.md](docs/FEATURES.md) | The 75 point-in-time features, and the leakage test |
| [STRATEGIES.md](docs/STRATEGIES.md) | The hypotheses — the twelve and the second wave — and the same-window scoreboard |
| [TIMING.md](docs/TIMING.md) | The calendar lab: the leg engine, the overnight decomposition, nine rules |
| [EVOLUTION.md](docs/EVOLUTION.md) | The genetic algorithm, and the four anti-overfitting defences |
| [HOW_THE_GA_WORKS.md](docs/HOW_THE_GA_WORKS.md) | The GA in five minutes — one page for an executive, one for an engineer |
| [TRADES.md](docs/TRADES.md) | Exporting every buy and sell, and the audit that ties it to the curve |
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
- **Fundamental features start 2010 and cover 649 of 973 historical index members**, which
  correlates with survival. Any strategy needing them carries a survivorship bias *on top
  of* the price-coverage gap above, and runs on a shorter, kinder window. Every one of them
  is scored against the index over its own dates for exactly this reason.
- **No true walk-forward yet.** GA fitness measures fold *consistency* inside the research
  window, which is evidence of robustness and not of generalisation. A real walk-forward
  re-runs the whole search inside each training window; it is the next thing to build.
  The forward-test harness is not a substitute — it evaluates a *fixed* candidate after
  2022, where a walk-forward re-runs the *search* inside the research window. The two are
  complements, and the second is what should narrow the candidates before the first is
  spent.
- **Two credit-spread features are ~11% populated**, because FRED's keyless endpoint
  returns about three years of the licensed ICE series.
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

1. **A true walk-forward harness.** Re-run the whole GA inside each training window and
   evaluate its winner on the next, with a purge and an embargo. This is the only way to
   get out-of-sample evidence without spending the holdout, and at 0.15s per evaluation it
   costs about twenty minutes for five folds.
2. **Spend the holdout — once.** `sp500lab experiments holdout` currently reads 0 looks.
   When there is a final candidate, pre-register it and run the forward test:
   `sp500lab forward seal <name> --rationale "..."` then `sp500lab forward run <name>`.
   Both legs, the decay and its standard error are recorded permanently
   ([FORWARD_TEST.md](docs/FORWARD_TEST.md)). Seal the candidates on a different day from
   the test — that gap is the only evidence the choice preceded the answer.
3. **Backfill the price gap** with a paid EOD feed (~$17/mo annual billing), then re-run the
   identical pipeline and **measure the survivorship-bias delta yourself**. Expect
   everything to get *worse*: that is the bias being removed.
4. **Point-in-time market cap and sectors** (TODO-5, TODO-6). Market cap now exists as a
   feature; sectors are still current-only, so any sector-neutral strategy leaks.
5. **The neural nets.** `strategies/learned.py` is the template and the feature layer is
   the input; what is missing is a model, not the plumbing.
