# The strategies

Twelve hypotheses on top of the six null hypotheses. Each one is a sentence somebody could
argue with, and they are chosen to disagree with each other.

```bash
python -m sp500lab backtest suite all
python -m sp500lab backtest suite alpha --costs pessimistic
python -m sp500lab backtest run quality_value --all-costs --annual
```

---

## Read the right column

Strategies here do **not** all run over the same window. Anything built on XBRL
fundamentals starts in 2010 because XBRL does, and 2010–2021 was a far kinder market than
2007–2021. Measured on the same engine:

| window | SPY CAGR | SPY Sharpe |
|---|---:|---:|
| 2007-04 → 2021-12 | 10.42% | 0.59 |
| 2010-07 → 2021-12 | 15.66% | 0.95 |
| 2012-01 → 2021-12 | 15.96% | 1.00 |

So `accrual_quality` at **17.40%** and `equal_weight` at **11.10%** are not ranked by those
numbers. `sp500lab backtest suite` puts each strategy next to the index over *its own*
dates and sorts by the difference. `excess` and `d_sharpe` are the only two columns that
compare like with like.

---

## The scoreboard

2007-04-30 → 2021-12-31 (or later where the data starts later), $100k, monthly, long-only,
**realistic** costs. `bench` is SPY over each row's own window.

| strategy | window | CAGR | Sharpe | maxDD | turnover | bench CAGR | bench Sharpe | excess | ΔSharpe |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| low_vol | 2007-04 | 9.84% | 0.69 | −39.9% | 191% | 10.42% | 0.59 | −0.58% | **+0.10** |
| defensive_regime | 2007-04 | 9.21% | 0.65 | **−23.6%** | 359% | 10.42% | 0.59 | −1.21% | **+0.06** |
| lottery_averse | 2007-04 | 9.91% | 0.65 | −42.5% | 492% | 10.42% | 0.59 | −0.51% | **+0.06** |
| equal_weight | 2007-04 | 11.10% | 0.58 | −58.3% | 39% | 10.42% | 0.59 | +0.68% | −0.01 |
| illiquidity_carry | 2007-04 | 13.77% | 0.56 | −67.2% | 401% | 10.42% | 0.59 | +3.35% | −0.03 |
| dividend_grower | 2007-04 | 10.90% | 0.55 | −58.9% | 489% | 10.42% | 0.59 | +0.48% | −0.04 |
| random_weight | 2007-04 | 10.24% | 0.54 | −61.2% | 1026% | 10.42% | 0.59 | −0.18% | −0.05 |
| multi_factor | 2010-07 | 13.35% | 0.88 | −36.7% | 269% | 15.66% | 0.95 | −2.31% | −0.07 |
| rolling_ridge | 2007-04 | 9.87% | 0.51 | −46.5% | 590% | 10.42% | 0.59 | −0.55% | −0.08 |
| accrual_quality | 2010-07 | 17.17% | 0.86 | −45.5% | 134% | 15.66% | 0.95 | +1.51% | −0.10 |
| index_entry_drift | 2007-04 | 8.80% | 0.46 | −69.4% | 170% | 10.42% | 0.59 | −1.63% | −0.13 |
| residual_momentum | 2007-04 | 7.89% | 0.46 | −61.3% | 342% | 10.42% | 0.59 | −2.53% | −0.13 |
| pead_drift | 2012-01 | 14.65% | 0.86 | −38.0% | 321% | 15.96% | 1.00 | −1.32% | −0.14 |
| restatement_averse | 2011-07 | 13.76% | 0.78 | −40.0% | 164% | 15.58% | 0.94 | −1.82% | −0.15 |
| frog_in_the_pan | 2007-04 | 6.88% | 0.41 | −56.2% | 373% | 10.42% | 0.59 | −3.54% | −0.18 |
| evolved_blend | 2007-04 | 6.46% | 0.40 | −51.7% | 450% | 10.42% | 0.59 | −3.96% | −0.19 |
| momentum_12_1 | 2007-04 | 6.39% | 0.38 | −55.5% | 354% | 10.42% | 0.59 | −4.03% | −0.21 |
| short_reversal | 2007-04 | 6.36% | 0.35 | −77.7% | 1004% | 10.42% | 0.59 | −4.07% | −0.23 |
| quality_value | 2010-07 | 12.16% | 0.63 | −51.0% | 154% | 15.66% | 0.95 | −3.51% | −0.33 |
| cash | 2007-04 | 0.00% | 0.00 | 0.0% | 0% | 10.42% | 0.59 | −10.42% | −0.59 |

**Three of twenty beat the index on risk-adjusted return.** Under *pessimistic* costs it is
two: `defensive_regime` falls to exactly flat and `lottery_averse` and `low_vol` survive.
That is the correct null result and it is the bar. If something beats these by a wide
margin on its first run, the overwhelmingly likely explanation is a bug.

### What is worth noticing

**`accrual_quality` is the trap this table exists to spring.** 17.17%/yr looks like the
best number in the file until you see SPY made 15.66% over the same dates with a *higher*
Sharpe. A leaderboard sorted on CAGR would have crowned it.

**`illiquidity_carry` earns +3.35%/yr and loses on Sharpe.** It is paid for holding what is
hard to sell, and it pays for that at every rebalance: 401% turnover, a −67% drawdown, and
the gap between its optimistic and pessimistic results is a direct measurement of how much
of the illiquidity premium a retail account keeps. No other strategy here produces that
number.

**`defensive_regime` has a −23.6% maximum drawdown** where everything else long-only ran to
−40% or worse. It is the only strategy that chooses *whether* to be invested rather than
only *what* to own — and it has two parameters set on a window containing 2008 and 2020,
which is exactly the criticism to make of it.

**`random_weight` beats seven real strategies.** It has no signal at all. Anything inside
its spread across seeds has demonstrated nothing.

---

## The twelve, and what each claims

### Under-reaction, three ways

**`residual_momentum`** — 12-1 momentum with market beta stripped out (Blitz, Huij &
Martens 2011). Raw momentum is partly a bet on beta: after a long rally the winners *are*
the high-beta names, so a momentum portfolio quietly becomes a leveraged market position
and is destroyed in the rebound. The direct comparison with `momentum_12_1` is the point —
same window, same construction, same costs, one difference. It wins that comparison
(+1.50pp CAGR, +0.08 Sharpe) and still loses to the index.

**`frog_in_the_pan`** — momentum, but only among names whose information arrived gradually
(Da, Gurun & Warachka 2014). Two stocks with identical 12-month returns drift differently
depending on whether the return came in many small pieces or a few jumps: continuous
information is individually negligible and gets under-reacted to; a 15% earnings gap is
impossible to ignore. Implemented as a conditional sort, not a blend — the hypothesis says
one signal is a *regime* for the other, and flattening that into a weighted sum tests a
different claim.

**`pead_drift`** — buy the largest recent earnings surprises. SUE against a random-walk
expectation: this quarter versus the same quarter a year ago, scaled by the volatility of
that difference. No analyst estimates, which is a limitation and also why it is computable
on this budget. Its universe is defined by an **event** rather than a ranking — only names
that filed within 100 days may be held — which is only possible because `filed_date` was
kept.

### Accounting

**`accrual_quality`** — Sloan (1996). Accruals are the gap between reported earnings and
operating cash flow, and the accrual component reverses while the cash component persists.
The market prices the two as if they were the same.

**`quality_value`** — cheap *and* good, which is rarer than either. Gross profitability,
ROE, low leverage, book-to-market, earnings yield, requiring at least three of five so a
name cannot qualify on one lucky ratio. It has the worst ΔSharpe in the table, which is
worth stating plainly: over 2010–2021 the intersection of cheap and good badly
underperformed an index led by expensive and good.

**`restatement_averse`** — own companies that report accurately and on time. This exists
because of a column most datasets throw away. `restatement_rate` is the share of a
company's published facts it has since revised, counted only from revisions that had
already happened; `filing_lag_days` is how long after the period closed the filing
appeared. Neither is computable from a single-vintage feed, because a single vintage *is*
the restatement.

*It is also the strategy most likely to be measuring the wrong thing.* A company with more
subsidiaries files more facts and has more to revise, so part of this may be a size and
complexity tilt wearing a governance costume. The honest test is whether it survives with
`log_market_cap` in the blend — which the genetic algorithm can settle better than an
argument can.

### Microstructure and index mechanics

**`illiquidity_carry`** — Amihud's measure: how far the price moves per dollar traded. See
above; the tension with the cost model *is* the experiment.

**`index_entry_drift`** — own whatever the index just added. Index funds must buy a new
constituent regardless of price. The classical estimates (Shleifer 1986; Harris & Gurel
1986) are 3–6%; the modern consensus is that it has been arbitraged away since ~2000, and
this project's window is entirely inside the modern era. Result: −1.63%/yr and a −69%
drawdown on a lumpy 19-name portfolio. Membership here is monthly (ADR-004), so the
pre-effective-date run-up — where most of the classical effect lives — is invisible.

### Preferences and payouts

**`lottery_averse`** — buy the boring ones. Low maximum daily return (Bali, Cakici &
Whitelaw 2011), low idiosyncratic volatility (Ang et al. 2006, priced *negatively*, which
no risk model predicts), low skew. Related to `low_vol` and deliberately not the same:
low_vol ranks on total volatility, which is mostly beta; this ranks on the shape of the
tail. Both beat the index on Sharpe, which is the closest thing to a positive result here.

**`dividend_grower`** — own the raisers, refuse the cutters. A dividend is the one
corporate statement that costs money to make. A cut is a **hard exclusion** rather than a
low score: a strategy that will buy a cutter at a sufficiently attractive yield is not the
strategy this is trying to be. Built from discrete dividend events, which is the only way
it can be built.

### Timing, and the bar

**`defensive_regime`** — changes *what* it owns and *how much*. Risk-on (index above its
200-day average): residual momentum, fully invested. Risk-off: lowest-volatility names, and
only half the account invested. Under a long-only mandate cash is the only defensive asset
there is. Momentum crashes are the motivation — in 2009 the losers momentum was avoiding
were the highest-beta names and they rebounded hardest, which no cross-sectional score can
see.

**`multi_factor`** — value, quality, momentum and low risk, equally weighted. Not an idea,
a **benchmark**: anything clever built on this feature layer has to beat the obvious thing
you get by averaging the four families with no tuning. The moment those weights are
optimised it stops being a benchmark and becomes another trial — which is what the genetic
algorithm is for.

---

## The second wave (2026-08): five mechanisms the twelve leave uncovered

`strategies/frontier.py`, group `frontier`, plus a second learned model in `learned.py`.
Added **after** the 2022-2026 forward test was spent — read ADR-037's contamination note
before treating any of their numbers as out-of-sample. Research window 2007-04→2021-12,
realistic costs, measured 2026-08-30; SPY made 10.70%/yr at Sharpe 0.60 over the same
dates, and ΔSharpe is against that. `sp500lab report algorithms` (→ `reports/extra/algorithms.html`)
carries the live version:

| strategy | CAGR | Sharpe | ΔSharpe | the claim in one line |
|---|---:|---:|---:|---|
| `vol_managed` | 10.18% | 0.67 | **+0.08** | volatility is forecastable and returns are not, so scale exposure by inverse variance (Moreira & Muir, long-only half) |
| `overnight_momentum` | 11.85% | 0.55 | −0.05 | momentum's profits accrue while the market is CLOSED — rank on the overnight component alone (Lou, Polk & Skouras) |
| `ensemble_rank` | 8.80% | 0.54 | −0.06 | the equal average of all twelve hypotheses beats backing any one of them |
| `week52_breakout` | 7.89% | 0.52 | −0.08 | investors anchor on the 52-week high and under-react near it (George & Hwang) |
| `div_month` | 8.62% | 0.47 | −0.13 | prices drift up in months where a dividend is predictable from the payment cadence (Hartzmark & Solomon) |
| `shallow_mlp` | 6.13% | 0.38 | −0.22 | a deliberately small neural net over the shared features, seed-ensembled (Gu, Kelly & Xiu) |

Worth noticing, in the same spirit as the first scoreboard:

* **`overnight_momentum` against `momentum_12_1` is the cleanest experiment in the
  file**: same window, same construction, same costs, one difference — and the overnight
  component posts 11.85%/0.55 where the whole return posts 6.39%/0.38. The decomposition
  carries real information. It still does not beat the index.
* **`vol_managed` is the only new entrant to clear the bar**, and it is the one that
  forecasts nothing cross-sectional — it owns the whole tradable universe and only
  decides how much. Consistent with this project's larger pattern: WHEN beats WHICH.
* **`ensemble_rank` lost to several of its own members**, which the ensembling
  literature says should be rare — and is itself evidence the twelve hypotheses are more
  correlated than their stories suggest.
* **`shallow_mlp`'s study deflates to 0.947, under the 0.95 bar.** Three seeds of the
  same architecture are three trials, and the best of three lucky draws would look like
  this. Printed as such.
* `div_month` pays 1,138%/yr turnover by construction — the dividend calendar rolls the
  eligible set monthly — and the premium does not cover the bill at retail size.

The calendar rules (`tm_*`) are the wave's third family and live at daily granularity in
their own engine and their own document: [TIMING.md](TIMING.md).

---

## What they share

A score, and nothing else. Position sizing, the top-k cut, the per-name cap and the
unbiased tie-break all come from `portfolio.py`, so the comparison measures signal quality
rather than who tuned their weighting (ADR-024). The default shape is 50 names, equal
weight, 5% cap — 50 because at $100k with a $1 per-order minimum, anything much wider pays
for the privilege (ADR-016).

Scores are **percentile ranks**, not z-scores. Cross-sectional fundamental data has tails
that are not merely fat but wrong: one bad share count produces a book-to-market of 300,
and a z-score would hand that name +25 and the entire portfolio.

A blend averages the components a name *has* rather than summing and letting one NaN
annihilate the row. The alternative silently shrinks the universe to names with complete
fundamental data — which, because coverage correlates with survival, is a survivorship
filter dressed up as an arithmetic convention.

---

## Honesty about what is being traded

The seven fundamentals-based strategies trade a narrower universe than the price-based
ones. Fundamental coverage is 649 of 973 historical index members and correlates with
survival, so they carry a survivorship bias **on top of** the price-coverage gap in
ADR-023 — which is itself 54.7% in 2007. Their `min_date` excludes the flat pre-data
stretch from their CAGR rather than quietly averaging it in.

Every run reports its coverage. Read it.

---

## Writing your own

`src/sp500lab/strategies/custom.py` is yours — nothing else in the project writes to it,
and it ships with one working example. The full contract, the scoring helpers and the
three things that will bite are in [ADDING_A_STRATEGY.md](ADDING_A_STRATEGY.md).

## The shape of one

```python
from sp500lab.backtest.strategy import FeatureStrategy, register
from sp500lab.strategies.signals import rank_pct

@register("my_idea")
class MyIdea(FeatureStrategy):
    """One sentence somebody could argue with."""
    requires_features = ("gross_profitability", "vol_126d")
    min_date = "2010-07-01"          # only if the inputs do not exist earlier

    def score(self, ctx):
        e = ctx.tradable
        return rank_pct(self.f(ctx, "gross_profitability"), e)
```

Add it to `GROUPS` in `strategies/__init__.py` so `backtest suite` picks it up. Then read
its number next to the index over its own window, and read
`sp500lab experiments deflate` before believing it — a strategy you wrote after looking at
this table is a trial too.
