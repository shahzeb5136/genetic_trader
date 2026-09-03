# The genetic algorithm

```bash
python -m sp500lab evolve run --study ga-1 --generations 25 --population 60
python -m sp500lab experiments deflate ga-1          # <- the number that decides
python -m sp500lab evolve best ga-1 --all-costs --trades results/trades/ga-1
python -m sp500lab report genetic --open             # all of it as three pages
```

`report genetic` writes `reports/genetic_algorithm/`: the methodology, the feature
presets and what each search converged on, and every search with its winner decoded, its
training history charted and its forward verdict attached. It reads the checkpoints and
re-runs nothing (ADR-046).

A loop over `run_backtest`. That is the whole trick, and it is why the engine was built
first: the fitness function **is** the backtest, so an evolved strategy is scored by
exactly the same accounting, the same next-open execution, the same costs and the same
survivorship-free universe as a hand-written one. The scoreboard cannot tell them apart.

---

## The result of the first real search

`ga-price-1`: 60 individuals × 25 generations, price preset, realistic costs, research
window only (2007-04 → 2021-12), turnover penalty 0.03, complexity penalty 0.01, seed 11.
**1,400 distinct individuals in about five minutes.**

| | winner | SPY, same window |
|---|---:|---:|
| CAGR | **12.40%** | 10.42% |
| Sharpe (daily) | **1.15** | 0.59 |
| Sharpe (monthly) | 1.32 | — |
| max drawdown | **−15.98%** | −55.19% |
| turnover | 352%/yr | — |
| names held | 12 | — |

Under all three cost settings: 12.89% optimistic, 12.40% realistic, **11.91%
pessimistic**. It does not depend on the spread estimator being kind.

### The correction

```
n_trials                           1400
trial_sharpe_std                   0.1909
expected_max_sharpe_annualised     0.6395     <- the bar the search itself set
sharpe_annualised_monthly          1.3198
deflated_sharpe                    0.9913     >= 0.95
```

The luckiest of 1,400 worthless strategies would have posted an annualised Sharpe of about
**0.64**. The winner posted **1.32**. The deflated Sharpe — the probability the result is
not that luck — is **0.9913**, which clears the 0.95 bar.

### What it evolved into, in its own words

```
Ranks 10 feature(s):
    -1.00  div_yield              (LOW is good)
    -0.89  vol_126d               (LOW is good)
    -0.80  beta_252d              (LOW is good)
    -0.79  trend_200d             (LOW is good)
    +0.38  high_52w_ratio         (high is good)
    +0.33  mom_12_1               (high is good)
    -0.27  idio_vol_252d          (LOW is good)
    +0.23  amihud_illiq           (high is good)
    -0.19  max_ret_21d            (LOW is good)
    -0.16  resid_mom_12_1         (LOW is good)
Holds the top 12 by score, equal-weighted, capped at 6.0% per name.
De-risks to 24% invested when the index is below its 200-day average or realised
volatility exceeds 2.47x its own year.
```

It is readable, which is the point of bounding the search space. It found **low
volatility, low beta, low idiosyncratic volatility and low lottery-payoff** — the same
place `low_vol` and `lottery_averse` live, arrived at independently. It kept a small
momentum tilt (`+mom_12_1`, `+high_52w_ratio`) while ranking `resid_mom_12_1` *negatively*,
which reads as "recent winners, but not the ones whose gains were idiosyncratic". It
avoids high dividend yield, which over 2007–2021 is largely avoiding financials and
energy.

### Three things to be suspicious of before believing any of it

**The drawdown is doing most of the work, and a tuned switch produced it.** −16% for a
long-only equity strategy through 2008 comes from the regime gate de-risking to 24%
invested — two parameters the search set on a window that contains 2008 and 2020. The
deflated Sharpe counts that as trials, which is partly why the correction is as large as it
is, but it is not the same as evidence that the switch generalises.

**It holds 12 names.** `top_k` evolved to near its floor. Concentration is how a long-only
search manufactures leverage, and the `min_names` guard stops the degenerate case rather
than the mild one.

**It is still in-sample.** Fold consistency was part of fitness, but every fold sits inside
the research window and the winner was selected using all of them. The only out-of-sample
evidence available is the 2022 holdout, which **has never been looked at** —
`sp500lab experiments holdout` says 0 — and which is worth exactly one look.

---

## The second search: `ga-full-2`, with fundamentals

Same shape, `full` preset, seed 21. 1,404 distinct individuals in **290 seconds**. It runs
on 2010-07 → 2021-12 because XBRL does, so it is **not comparable to `ga-price-1`** — a
different and considerably kinder market.

| | winner | SPY, same window |
|---|---:|---:|
| CAGR | **21.93%** | 15.66% |
| Sharpe (daily) | **1.36** | 0.95 |
| Sharpe (monthly) | 1.77 | — |
| max drawdown | −28.4% | −33.7% |
| information ratio vs SPY | **0.78** | — |
| turnover | 246%/yr | — |
| names held | 28 | — |

22.21% / 21.93% / 21.64% across optimistic, realistic and pessimistic costs — a cost drag
of 47 basis points, because 246% turnover across 28 large-cap names is cheap to run.

```
n_trials                           1404
expected_max_sharpe_annualised     0.6945
sharpe_annualised_monthly          1.7661
deflated_sharpe                    0.9994
```

### What it found

```
    -1.00  beta_252d              (LOW is good)
    -1.00  amihud_illiq           (LOW is good)
    -1.00  div_yield              (LOW is good)
    +0.84  roe                    (high is good)
    -0.70  accruals               (LOW is good)
    -0.61  vol_126d               (LOW is good)
    -0.60  info_discreteness      (LOW is good)
    -0.43  book_to_market         (LOW is good)
    +0.42  earnings_yield         (high is good)
    +0.39  gross_profitability    (high is good)
    ...
Holds the top 28 by score, equal-weighted, capped at 8.5% per name.
Always fully invested; the regime gate is switched off.
```

Three things worth reading in that list.

**It rediscovered quality, independently.** High ROE, high gross profitability, low
accruals — that is Novy-Marx and Sloan, arrived at by a search that had never been told
either exists. `accrual_quality` and `quality_value` are in this repository as hand-written
strategies and it did not copy them; it was handed 23 ranked columns and found that
combination.

**It is short value and long quality.** `book_to_market` is ranked NEGATIVELY (−0.43) while
`earnings_yield` is positive (+0.42). That is not a contradiction, it is the standard
distinction: cheap-on-assets was a bad bet over 2010–2021 and cheap-on-earnings was not.
`quality_value`, which blends both with positive weights, has the worst ΔSharpe in
[STRATEGIES.md](STRATEGIES.md) — the search found the same thing and split them.

**It turned the regime gate OFF**, where `ga-price-1` turned it hard on. Its window contains
no 2008. That is the clearest evidence in this project that these two searches found
different things because they looked at different periods, and it is the reason a result
from one preset says nothing about the other.

**And one that should bother you:** `+0.14 restatement_rate`, meaning it slightly prefers
companies that restate MORE. That is the opposite of `restatement_averse`'s claim, it is
inside the noise at that weight, and it is a good illustration of what a dead zone at ±0.10
buys — three more features fell below it and were dropped entirely.

### The caveat that matters most here

This window is not just kinder, it is **narrower**. Fundamental coverage is 649 of 973
historical index members and correlates with survival, so this strategy trades a
survivor-biased subset on top of the price-coverage gap. The run reports it: 74.2% median
coverage, 62.1% at its worst. Neither of these two winners has been near the holdout.

---

## The four defences

A GA overfits more aggressively than almost anything, because it is *explicitly*
maximising the number you report. Each of these exists for that reason.

### 1. A small, bounded, readable search space

The genome is a weighted sum of cross-sectionally **ranked** features plus a portfolio
shape: 19 genes for the price preset, 29 with fundamentals.

No expression trees, no evolved arithmetic, no products of indicators. Far less expressive,
and every point in the space has an economic reading a human can argue with. A tree-based
GA on this data would find spectacular nonsense in its first generation.

- **Ranks, not raw values.** One bad share count is worth exactly "first place".
- **A dead zone** at ±0.10, so "how many features does this use" has an answer, parsimony
  pressure has something to grip, and behaviourally identical genomes deduplicate.
- **Curated presets** — 13 features, or 23 with fundamentals. Not all 75. Every extra
  feature multiplies the space and the trial count the deflation must discount.
- **`describe_genome` prints an individual as sentences.** A winning parameter vector
  nobody can read is a winning vector nobody can check.

### 2. Every individual is logged as a trial

Not just the winners. The deflated Sharpe needs the trial count and the spread of trial
Sharpes; neither can be recovered afterwards (ADR-026). `evolve run` logs by default and
`--no-log` prints a warning saying the result can no longer be corrected.

The registry's trial count and the GA's evaluation cache key on the **same** behavioural
fingerprint. Counting identical individuals twice over-deflates; not deduplicating at all
reports a small trial count for a large search and under-deflates.

### 3. Fitness is fold consistency, not full-sample Sharpe

Default: the **monthly** Sharpe — the same quantity `registry.deflate()` uses, so the
search and its own significance test look at the same number — computed on each of four
contiguous sub-periods separated by a one-month embargo, aggregated as `mean − 0.5 × std`.

A strategy that made all its money in 2009 and nothing since has a fine full-sample Sharpe
and a poor fold-consistency score, and that distinction is most of what separates a
discovery from a coincidence on nineteen years of monthly data.

Contiguous with an embargo, **never random K-fold**: financial data is autocorrelated and a
shuffled split leaks across the boundary. The embargo is one month because that is one
holding period under ADR-016.

> **What this is not.** Every fold is inside the research window and every individual is
> selected using all of them. A good fold score is evidence of *consistency*, not of
> generalisation. A genuine walk-forward means re-running the entire GA inside each
> training window and evaluating its winner on the next; that costs one full search per
> fold, it is the right next step, and it is deliberately not pretended at here. ADR-032.

Optional penalties: `--turnover-penalty` (double-counting on purpose — the half-spread
estimate is the weakest input in the chain, and 500%/yr turnover is a large bet on it) and
`--complexity-penalty` (per active feature, biasing the population toward explanations).

### 4. The holdout is untouched

The search stops the day before 2022-01-01. Testing the winner there is a separate,
deliberate, permanently recorded act (ADR-025), and doing it more than once destroys the
only out-of-sample evidence there is.

---

## Speed

**~0.15 seconds per evaluation**, so 1,500 individuals is about four minutes. Four things
buy that, and a change to any of them costs it:

1. the price panel is built once and memoised;
2. **feature ranks are precomputed once for the whole population** — a rank depends on the
   feature, the date and the tradable mask, and on nothing about the genome, so in a
   4,000-evaluation search each one was being recomputed 4,000 times. Measured: 0.29s →
   0.15s per evaluation;
3. each individual is **one** backtest whose folds are sliced from the resulting curve
   rather than re-run;
4. the evaluation cache, keyed on the behavioural fingerprint, so a converged population
   costs nothing to re-evaluate.

Pre-ranked columns are named `<feature>__rank`. Same values in the same place would let a
strategy expecting raw values silently receive ranks and produce a portfolio that is wrong
but still looks like a portfolio; the rename turns that into a `KeyError` at the first
rebalance.

`git_state()` is memoised for the process — two subprocesses per logged run is nothing for
one backtest and about thirty seconds of pure overhead across 1,500.

---

## Reading a run

```
BY GENERATION
 generation  best_fitness  mean_fitness  best_sharpe  best_n_active  diversity  scorable
          0        0.5242        0.1809       0.6867              1     0.2915        60
          6        0.6296        0.4955       0.8598             10     0.1842        60
         13        0.7056        0.5333       1.0219             10     0.1931        60
         20        0.8381        0.5791       1.1496             10     0.1700        60
         24        0.8381        0.5788       1.1496             10     0.1586        60
```

`diversity` is the mean normalised spread across genes. It is reported every generation
because **the most common way a GA fails is silently**: a run whose diversity collapses in
generation 4 spends the next forty generations re-evaluating one individual, and the
fitness curve looks like convergence rather than the stall it is. Below 0.05 the CLI says
so. Around 0.15–0.30 is healthy.

`best_n_active` starting at 1 is the seeded population working: generation 0's best is
`low_vol`, a single-feature individual, and evolution improved on it.

---

## Seeding

`--seed-baselines` (on by default) starts the population from genomes that reproduce
momentum, residual momentum, low volatility, lottery aversion, reversal, trend, value and
quality.

This is not a shortcut, it is the experiment. If a population that starts from 12-1
momentum and low volatility cannot evolve anything better than its seeds, that is a far
more informative result than a random start wandering to a mediocre optimum — and it is
the direct answer to "can the search improve on what a person wrote". In `ga-price-1` it
did: 0.5242 → 0.8381.

---

## Operators

**Tournament selection**, not fitness-proportionate: fitness is a Sharpe ratio, which can
be negative and whose scale is arbitrary, so roulette-wheel selection would need an offset
somebody invented. A tournament only needs an ordering.

**Blend crossover (BLX-α)**, not single-point: a genome of feature weights has no
meaningful gene *order*, so a cut point is arbitrary. BLX samples each gene from an
interval slightly wider than the parents span, which lets a converged population reach
outside itself.

**Two mutations.** A Gaussian nudge scaled to each gene's own range explores locally; a
rare full reset is the only operator that can reintroduce diversity a population has lost
entirely.

**Elitism plus immigration**, pulling in opposite directions on purpose: the best survive
untouched so the search cannot go backwards, and a few random newcomers arrive every
generation so it cannot finish converging.

Every operator takes an explicit `numpy.random.Generator`. Nothing touches global numpy
state. A search that cannot be replayed exactly cannot be audited.

### Deliberately absent

Multi-objective (NSGA-II) selection, island models, adaptive operator rates. All reasonable,
and none address the failure mode that actually threatens this project — which is not slow
convergence, it is converging beautifully onto noise.

---

## Resumability

Every generation is appended to `data/experiments/evolve/<study>.jsonl` with its full
population. `evolve history` and `evolve best` read a finished run without re-running it,
and Ctrl-C returns the state at the last completed generation rather than losing it.

---

## The two presets

| preset | features | window | genes |
|---|---:|---|---:|
| `price` | 13 | 2007-04 onward | 19 |
| `full` | 23 | **2010-07** onward | 29 |

`full` adds book-to-market, earnings yield, gross profitability, ROE, accruals, market cap,
asset growth, the earnings surprise, the restatement rate and leverage — and costs three
years of history plus a narrower, more survivor-biased universe. Runs on the two presets
are **not comparable**; they cover different periods. Compare each against the index over
its own window (`sp500lab backtest suite`).

---

## The next thing to build

A true walk-forward: re-run the whole GA inside each training window, take its winner, and
evaluate it on the following window with a purge and an embargo. That is the only
mechanism here that would produce genuinely out-of-sample evidence without spending the
holdout, and it costs one full search per fold — about twenty minutes for five folds at
the current speed, which is affordable.

Until it exists, 0.9913 and 0.9994 say each winner survives *the search that produced it*.
Neither says the winner will work.

And a note on the two together: `ga-price-1` and `ga-full-2` are separate studies precisely
so their trial counts do not mix. A third study looking at both winners and picking the
better one is itself a trial, and the honest trial count for that choice is 2,804 rather
than 1,404. The registry cannot know that unless somebody writes it down.
