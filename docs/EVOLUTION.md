# The genetic algorithm

```bash
python -m sp500lab evolve run --study ga-1 --seeds 3          # nine families, three seeds
python -m sp500lab experiments deflate ga-1                   # <- the number that decides
python -m sp500lab evolve ensemble ga-1 --all-costs --trades results/trades/ga-1
python -m sp500lab report genetic --open                      # all of it as three pages
```

`report genetic` writes `reports/genetic_algorithm/`: the methodology, the families and
what each search converged on, and every search with its champion decoded, its ensemble
described, its training history charted and its forward verdict attached. It reads the
checkpoints and re-runs nothing (ADR-046).

A loop over `run_backtest`. That is the whole trick, and it is why the engine was built
first: the fitness function **is** the backtest, so an evolved strategy is scored by
exactly the same accounting, the same next-open execution, the same costs and the same
survivorship-free universe as a hand-written one. The scoreboard cannot tell them apart.

---

## The 2026-09 redesign, in one table

Three searches ran over the original design. Every winner cleared the 0.95
deflated-Sharpe convention on the research window and **every winner decayed out of
sample** — 1.15 → 0.19, 1.36 → 0.67, 0.95 → 0.20. That replication, not any single
winner, was the most solid thing the search had produced, and it is what the redesign
answers. Five changes, in order of how much each one moves the needle:

| | before (ADR-031/032) | now (ADR-048/049/050) |
|---|---|---|
| **search space** | one free weight in [−1, +1] per feature; 13, 17 or 23 features | one **non-negative** weight per **family**; nine prior-signed families, **at most three live**, enforced at decode time |
| **fitness** | four contiguous folds, `mean − 0.5 × std` | **twelve random sub-periods of 3–5 years, the 25th percentile** — the worst quarter decides |
| **complexity** | per-feature penalty, default 0 | charged **per family, per feature, and for switching the gate on**, all non-zero by default |
| **deliverable** | the champion | the **ensemble**: the average signal of the top 30 survivors, pooled across seeds |
| **costs** | net of realistic costs | net of **pessimistic** costs (twice the estimated spread), turnover charged again on top |

Everything else — the engine, the registry, the fingerprint, the holdout rule, the
forward harness — is unchanged, and the three earlier searches decode exactly as before.

### 1. Shrink the space: families with a prior

The features with a story are grouped into nine families in `strategies/genome.py`, each
a fixed composite of its members' percentile ranks with the sign the literature settled:

| family | members (+ high is good, − low is good) | the prior |
|---|---|---|
| Momentum | +mom_12_1 +resid_mom_12_1 +high_52w_ratio −info_discreteness | under-reaction; Jegadeesh & Titman, Blitz et al., George & Hwang, Da et al. |
| Short-term reversal | +rev_1m | last month reverses; Jegadeesh 1990 |
| Low risk | −vol_126d −idio_vol_252d −beta_252d −max_ret_21d | the low-risk anomaly; Frazzini & Pedersen, Ang et al., Bali et al. |
| Illiquidity | +amihud_illiq | paid for holding what is hard to sell; Amihud 2002 |
| Payout | +div_yield +div_growth_1y +div_due_1m | cash returned is a costly signal; Lintner, Hartzmark & Solomon |
| Value | +book_to_market +earnings_yield +cf_yield | cheap on a fundamental; Basu, Fama & French |
| Quality | +gross_profitability +roe −accruals −leverage | profitable, cash-backed, unlevered; Novy-Marx, Sloan, AFP |
| Investment | −asset_growth | fast balance-sheet growth underperforms; Cooper et al., CMA |
| Earnings surprise | +eps_surprise | post-earnings-announcement drift; Bernard & Thomas |

A family composite is the mean of its members' prior-signed ranks over whichever members
a name has a value for — `rank` for a high-is-good member, `1 − rank` for a low-is-good
one, so every term is in [0, 1] and a missing member counts as average rather than as a
reward. The genome holds one weight in [0, 1] per family plus the six shape genes; the
preset caps live families at three, and `Genome.effective()` zeroes the dead zone and
everything past the cap before the vector is decoded or fingerprinted. An individual
therefore reads as "back quality and low risk, three to two", which is a hypothesis a
person could have written down — and it cannot rank value backwards, which the second
search did.

Two presets: `families` (all nine, from 2010-07 because four need XBRL; 15 genes) and
`families-price` (momentum, reversal, low risk, illiquidity, payout from 2007-04; 11
genes). The 57 features without a story were cut and the reason for each is recorded in
`CUT_FEATURES` and printed on the features page: redundant horizons, the contested
overnight decomposition, size proxies, index-entry flags, ambiguous ratios, filing
behaviour, macro context. ADR-048.

### 2. Robust fitness, not maximal

`Folds.random` draws twelve sub-periods of three to five years at random positions inside
the research window — from a private generator seeded by `--fold-seed`, so every
individual and every seed is scored on the same ones. The fitness is the **25th
percentile** of the sub-period monthly Sharpes (`--aggregate quantile --quantile 0.25`;
`min` is the harsher option). A rule that only works in one stretch is killed during
evolution rather than surviving to the test set on that stretch. The sub-periods are
sliced from the one equity curve the individual already produced, so twelve cost no more
than four did. ADR-049.

> **What this is not.** Every sub-period sits inside the research window and the winner
> was selected using all of them. A good score is evidence of *consistency*, not of
> generalisation. The out-of-sample test is the 2022 holdout, it is looked at once, and
> looking is recorded (ADR-025). And "random" means random in position and length; each
> sub-period is one contiguous stretch of history — never a shuffled K-fold.

### 3. Complexity, charged per rule

| charge | default | what it says |
|---|---:|---|
| per family backed | 0.02 | every family is a rule |
| per feature read | 0.01 | every feature is a dimension the search could find a coincidence in |
| regime gate on | 0.03 | two tuned parameters and a switch; the −16% drawdown that flattered `ga-price-1` was this gate, tuned on a window containing 2008 |
| per 100%/yr turnover | 0.03 | model risk in the spread estimate, on top of the cost model |

A three-family strategy with the gate pays about 0.18 of fitness more than a one-family
one without it, and has to earn that in the worst quarter.

### 4. Ensemble, not champion

The single best individual is the maximum over thousands of draws — the most
luck-contaminated object in the population. What a search hands on (`evolve.winners()`,
and through it the forward suite and both report sets) is `EvolvedEnsemble`: the
equal-weighted average of the **beliefs** of its top 30 distinct individuals, pooled
across every seed it ran. Each member's weighted sum of ranks is re-ranked to [0, 1]
before averaging so no member out-shouts the rest; the regime gate is a vote (step aside
when at least half the members would); the holding count and cap are the members'
medians. Beliefs rather than portfolios, because averaging thirty twelve-name portfolios
gives a two-hundred-name one paying a dollar of commission minimum per name at this
account size.

`--seeds 3` runs three independent populations that share the study, the objective and
the evaluation cache; the ensemble pools their best. The ensemble is backtested once at
the end, logged into the study as one more trial, written beside the checkpoint as
`<study>.ensemble.json`, and rebuilt from any checkpoint by
`evolve ensemble <study> --rebuild`. The three earlier searches keep handing over their
champions — their forward tests are spent. ADR-050.

### 5. Costs inside fitness

The curve being scored has always been net of costs; it is now net of the **pessimistic**
setting — commission plus twice the estimated half-spread — so a rule that only works if
the spread estimator is kind never scores well. The turnover penalty then charges turnover
a second time, deliberately: the half-spread is the weakest input in the chain (ADR-020),
and a strategy trading 350%/yr is a large bet that it is right. The three earlier winners
traded 246–352%/yr.

---

## The defaults

`EvolutionConfig` lives in `evolve/config.py` so the command line prints the engine's own
defaults. What `sp500lab evolve run` does with no flags:

| | value | flag |
|---|---|---|
| preset | `families` | `--preset` |
| population × generations | 60 × 25 per seed | `--population`, `--generations` |
| seeds | 1 (raise it) | `--seeds` |
| costs charged | pessimistic | `--costs` |
| sub-periods | 12 random, 3–5 years, fold seed 0 | `--fold-scheme`, `--folds`, `--fold-years`, `--fold-seed` |
| aggregate | 25th percentile | `--aggregate`, `--quantile` |
| penalties | turnover 0.03, feature 0.01, family 0.02, gate 0.03 | `--turnover-penalty`, `--complexity-penalty`, `--family-penalty`, `--gate-penalty` |
| ensemble | top 30 | `--ensemble-size` |
| holdout | exclude | `--holdout` |

---

## The first search over the new design: `ga-families-1`

Ran 2026-09-04 with the defaults and `--seeds 3`: preset `families`, 2010-07 → 2021-12,
60 × 25 per seed, pessimistic costs, twelve random sub-periods at the 25th percentile,
every penalty on. **4,500 evaluations, 4,179 distinct individuals, 20 minutes.**

| | champion | ensemble of 30 | SPY, same window |
|---|---:|---:|---:|
| CAGR, realistic costs | 17.79% | **17.75%** | 15.66% |
| Sharpe (daily) | 1.15 | **1.15** | 0.95 |
| max drawdown | −30.2% | −30.2% | −33.7% |
| turnover | 220%/yr | 220%/yr | — |
| names held | 35 | 35 | — |
| worst-quarter score (no penalties) | 1.15 | 1.08 | — |
| pessimistic / optimistic CAGR | 17.38% / 18.21% | 17.33% / 18.17% | — |

```
n_trials 4180   trial_sharpe_std 0.1915   expected_max_sharpe 0.6973
sharpe_monthly 1.5599 (the study's best run)   deflated_sharpe 0.9924
the ensemble itself, realistic costs:  sharpe_monthly 1.42   deflated_sharpe 0.987
```

### What it found

```
Backs 2 of 9 families, at most 3:
    0.66  Quality              (+gross_profitability, +roe, -accruals, -leverage)
    0.23  Low risk             (-vol_126d, -idio_vol_252d, -beta_252d, -max_ret_21d)
Holds the top 35 by score, equal-weighted, capped at 6.4% per name.
Always fully invested; the regime gate is switched off.
```

**Quality and low risk, and nothing else.** Every one of the 30 ensemble members backs
exactly those two families — quality about three to one over low risk — with the gate off
and 30 to 40 names. Seed 0 and seed 2 each converged on three-family champions (adding
momentum or payout) that scored 0.94 and 0.89; seed 1's two-family individual scored
0.97 and outranked both, and the 30 best individuals of 4,179 are all in its
neighbourhood. That is the family cap, the per-rule charges and the worst-quarter
statistic doing exactly what they were built to do: the third story never paid for
itself in the worst quarter of the window. Compare the second search's winner on the
same window, which put twelve free-signed weights on 23 features, including a preference
for companies that restate more.

**The ensemble is the champion, this time.** 17.33% against 17.38% at pessimistic costs,
the same drawdown, the same turnover. When the thirty best survivors of a search are
near-clones, the average of their beliefs is their belief, and the ensemble buys nothing
the champion did not already have. Read that as a fact about this search — three seeds
converged on one neighbourhood — rather than as the ensemble being redundant: on a
search whose survivors disagree, it is the disagreement being averaged out. The next
refinement, if this repeats, is diversity-aware membership (the best individual of each
distinct family set, rather than the thirty best overall).

### Then it was forward-tested, once — and it decayed

The look was spent the same day, on the user's decision, under all three cost settings
(`forward run ga-families-1-ensemble`, looks 118–120 in the ledger):

| realistic costs | research 2010-08 → 2021-12 | forward 2022-02 → 2026-09 |
|---|---:|---:|
| CAGR | 17.75% | **5.73%** |
| the index, same dates | 15.66% | 13.52% |
| Sharpe (daily) | 1.15 | **0.46** (index 0.82) |
| Sharpe (monthly) | 1.42 | 0.55 (−1.5σ, p = 0.07) |
| max drawdown | −30.2% | −15.3% |
| turnover | 220%/yr | 244%/yr |

**DECAYED, four for four.** It made money, it halved its drawdown, and it did not beat
the index — the same verdict at optimistic and pessimistic costs. The families, the
worst-quarter objective, the per-rule charges and the ensemble did not rescue the search
from the regime it was measured in: 2022–2026 was a cap-weighted mega-cap market, and a
35-name equal-weighted quality-plus-low-risk book is exactly the kind of portfolio it
left behind. Two things distinguish this decay from the three before it, and neither is
a comfort. The drawdown and turnover checks held, which the earlier champions' did not —
the redesign produced a more stable strategy, not a better one. And this test is weaker
than the first two: the design was chosen after the 2022–2026 results were read
(ADR-037), so a decay here is a stronger refutation than a hold would have been a
confirmation.

**Two things still to hold it loosely by.** The window is the kind decade with the
survivor-biased fundamentals coverage (74% median, 62% at worst), so 17.8% was never
comparable with a 2007 winner's 12%. And 55 forward months resolve a Sharpe difference
of about 1.3 at best; what they refuted is the claim of beating the index, not the claim
of a positive return (P(true Sharpe > 0) = 0.88).

---

## The three earlier searches, and what they taught

Kept in full because the redesign is a response to them. All three ran over the
free-weight presets with four contiguous folds, `mean − 0.5 × std`, realistic costs.

### `ga-price-1` (2007-04 → 2021-12, price preset)

60 × 25, turnover penalty 0.03, complexity penalty 0.01, seed 11. **1,400 distinct
individuals in about five minutes.**

| | winner | SPY, same window |
|---|---:|---:|
| CAGR | **12.40%** | 10.42% |
| Sharpe (daily) | **1.15** | 0.59 |
| max drawdown | **−15.98%** | −55.19% |
| turnover | 352%/yr | — |
| names held | 12 | — |

```
n_trials 1400   expected_max_sharpe 0.6395   sharpe_monthly 1.3198   deflated_sharpe 0.9913
```

It found low volatility, low beta, low idiosyncratic volatility and low lottery payoff —
the neighbourhood `low_vol` and `lottery_averse` occupy — with a small momentum tilt, and
turned the regime gate hard on: −16% through 2008 came from de-risking to 24% invested,
two parameters set on a window containing 2008 and 2020. **Forward: 1.15 → 0.19.**

### `ga-full-2` (2010-07 → 2021-12, full preset)

Seed 21, 1,404 distinct individuals in 290 seconds. Not comparable with the first —
a different and kinder market.

| | winner | SPY, same window |
|---|---:|---:|
| CAGR | **21.93%** | 15.66% |
| Sharpe (daily) | **1.36** | 0.95 |
| max drawdown | −28.4% | −33.7% |
| turnover | 246%/yr | — |

```
n_trials 1404   expected_max_sharpe 0.6945   sharpe_monthly 1.7661   deflated_sharpe 0.9994
```

It rediscovered quality — high ROE, high gross profitability, low accruals — without
being told Novy-Marx or Sloan exist, ranked book-to-market *negatively* while ranking
earnings yield positively, and turned the gate off. It also put +0.14 on
`restatement_rate`, preferring companies that restate more — inside the noise, and
exactly the kind of free sign the families now remove. **Forward: 1.36 → 0.67.**

### `ga-night-1` (2007-04 → 2021-12, night preset)

1,404 trials; winner 10.89%/yr, Sharpe 0.95, −17.9% max drawdown, deflated 0.9828. Inside
the blend it weighted *intraday* momentum positively and overnight momentum slightly
negatively — the reverse of the standalone literature result. **Forward: 0.95 → 0.20.**
Built after the first forward results were read; ADR-037 records that contamination.

### What the three say together

Every one cleared its deflated-Sharpe threshold and every one decayed. The deflated
Sharpe corrects for how many configurations were *tried*; it cannot correct for a space
wide enough that some configuration fits fifteen years of monthly history by
construction, nor for the search and the researcher having seen the same history. Each
of the five changes above targets one of those channels: fewer and prior-signed
hypotheses, a statistic a lucky stretch cannot carry, a price on every extra rule, an
average in place of a maximum, and a cost model the winner cannot lean on.

---

## The five defences

A GA overfits more aggressively than almost anything, because it is *explicitly*
maximising the number you report. Each of these exists for that reason.

### 1. A small, bounded, readable search space

Weighted sums of cross-sectionally **ranked** features, grouped into prior-signed
families with at most three live, plus a portfolio shape: 11 or 15 genes on the family
presets (19 to 29 on the older ones). No expression trees, no evolved arithmetic, no
products of indicators. Every point in the space has an economic reading, and
`describe_genome` prints it as sentences.

- **Ranks, not raw values.** One bad share count is worth exactly "first place".
- **A dead zone** at ±0.10, so "how many features does this use" has an answer, parsimony
  pressure has something to grip, and behaviourally identical genomes deduplicate. The
  family cap works the same way: a weight past the cap is zero and does not change the
  fingerprint.
- **Immutable presets.** A stored genome decodes by position against its preset's gene
  list; a family's members and signs are part of that contract (ADR-038, ADR-048).

### 2. Every individual is logged as a trial

Not just the winners. The deflated Sharpe needs the trial count and the spread of trial
Sharpes; neither can be recovered afterwards (ADR-026). `evolve run` logs by default and
`--no-log` prints a warning saying the result can no longer be corrected. The registry's
trial count and the GA's evaluation cache key on the **same** behavioural fingerprint.
The ensemble is one more logged trial in the same study.

### 3. Fitness is the worst quarter of many sub-periods, net of pessimistic costs, minus a charge per rule

Above. The **monthly** Sharpe — the same quantity `registry.deflate()` uses, so the search
and its own significance test look at one number — on each of twelve random sub-periods,
at the 25th percentile, with the penalties subtracted.

### 4. The holdout is untouched

The search stops the day before 2022-01-01. Testing a deliverable there is a separate,
deliberate, permanently recorded act (ADR-025), and doing it more than once destroys the
only out-of-sample evidence there is.

### 5. The deliverable is an ensemble

Above. What reaches the forward test is the average of the survivors, never the maximum.

---

## Speed

**~0.1–0.2 seconds per evaluation**, so a 60 × 25 seed is three to six minutes and a
three-seed search is under a quarter of an hour. Four things buy that, and a change to
any of them costs it:

1. the price panel is built once and memoised;
2. **feature ranks are precomputed once for the whole population** — a rank depends on
   the feature, the date and the tradable mask, and on nothing about the genome;
3. each individual is **one** backtest whose sub-periods are sliced from the resulting
   curve rather than re-run;
4. the evaluation cache, keyed on the behavioural fingerprint and shared across seeds, so
   a converged population — or a second seed landing on the same individual — costs
   nothing to re-evaluate.

Pre-ranked columns are named `<feature>__rank`, so a strategy expecting raw values fails
at the first rebalance instead of silently summing ranks.

---

## Reading a run

```
BY GENERATION
 seed  generation  best_fitness  mean_fitness  best_sharpe  best_n_active  best_n_families  diversity  scorable
    0           0        0.7335        0.4102       0.93              4                1     0.176        60
    0          12        0.9446        0.6293       1.14              8                2     0.170        60
    1           0        0.7335        0.4102       0.93              4                1     0.176        60
```

`diversity` is the mean normalised spread across genes, reported every generation because
**the most common way a GA fails is silently**: a run whose diversity collapses in
generation 4 spends the next twenty re-evaluating one individual, and the fitness curve
looks like convergence rather than the stall it is. Below 0.05 the CLI says so, per seed.

`best_n_families` is the cap at work: the champion of a family search backs one, two or
three stories, never more, and the per-family charge is why it is often fewer than three.

The fitness printed is the objective's score — the worst-quarter monthly Sharpe minus the
penalties — not a Sharpe you can quote. The champion's and the ensemble's headline Sharpe
and CAGR are printed beside it, under the search's cost setting.

---

## Seeding

`--seed-baselines` (on by default) starts each population from one individual per family,
backing that story alone — or, on a feature preset, from genomes reproducing momentum,
residual momentum, low volatility, lottery aversion, reversal, trend, value and quality.

This is not a shortcut, it is the experiment. If a population that starts from the
hand-written hypotheses cannot evolve anything better than its seeds, that is a far more
informative result than a random start wandering to a mediocre optimum — and it is the
direct answer to "can the search improve on what a person wrote".

---

## Operators

**Tournament selection**, not fitness-proportionate: fitness can be negative and its scale
is arbitrary, so roulette-wheel selection would need an offset somebody invented.

**Blend crossover (BLX-α)**, not single-point: a genome of weights has no meaningful gene
*order*, so a cut point is arbitrary. BLX samples each gene from an interval slightly
wider than the parents span, which lets a converged population reach outside itself.

**Two mutations.** A Gaussian nudge scaled to each gene's own range explores locally; a
rare full reset is the only operator that can reintroduce diversity a population has lost.

**Elitism plus immigration**, pulling in opposite directions on purpose.

**Independent seeds.** `--seeds N` runs N populations with different random seeds under
one study, one objective and one cache; the ensemble pools their survivors.

Every operator takes an explicit `numpy.random.Generator`. Nothing touches global numpy
state. A search that cannot be replayed exactly cannot be audited.

### Deliberately absent

Multi-objective (NSGA-II) selection, island models, adaptive operator rates, evolved
expression trees. All reasonable, and none address the failure mode that actually
threatens this project — which is not slow convergence, it is converging beautifully onto
noise.

---

## Resumability

Every generation of every seed is appended to `data/experiments/evolve/<study>.jsonl`
with its full population, and the ensemble is written beside it as
`<study>.ensemble.json`. `evolve history`, `evolve best` and `evolve ensemble` read a
finished run without re-running it; Ctrl-C returns the state at the last completed
generation, and `evolve ensemble <study> --rebuild` builds the ensemble a killed run did
not get to.

---

## The five presets

| preset | kind | signals | window | genes |
|---|---|---:|---|---:|
| `families` | families, at most 3 live | 9 families / 22 features | **2010-07** onward | 15 |
| `families-price` | families, at most 3 live | 5 families / 13 features | 2007-04 onward | 11 |
| `price` | one free weight per feature | 13 | 2007-04 onward | 19 |
| `full` | one free weight per feature | 23 | **2010-07** onward | 29 |
| `night` | one free weight per feature | 17 | 2007-04 onward | 23 |

Runs on different windows are **not comparable**; compare each against the index over its
own window (`sp500lab backtest suite`). All five are frozen: new features, new members or
a different cap mean a new preset (ADR-038).

---

## The next thing to build

A true walk-forward: re-run the whole search — families, worst quarter, ensemble — inside
each of several training windows, and evaluate each window's ensemble on the window that
follows, with a purge and an embargo. That is the only mechanism that would produce
genuinely out-of-sample evidence without spending the holdout, and it costs one full
search per fold — about a quarter of an hour each at the current speed.

Until it exists, the deflated Sharpe says a result survives *the search that produced
it*, the worst-quarter score says it survived every stretch of the research window, and
neither says it will work — `ga-families-1` cleared both and decayed on 2022–2026 like
the three before it. The only untainted evidence left for any of the four is the data
that arrives after 2026-09.
