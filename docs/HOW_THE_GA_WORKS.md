# How the genetic algorithm works

Two audiences, one page. **Part 1** assumes nothing. **Part 2** assumes you write software.
The full reference is [EVOLUTION.md](EVOLUTION.md); this is the five-minute version.

---

# Part 1 — For an executive

## The idea in one paragraph

We describe a trading strategy as a short list of numbers — how hard to back each of nine
well-known investment stories (momentum, low risk, value, quality, …), how many stocks to
hold, and whether to pull back when markets turn. Any such list can be tested against
eleven years of history in about a tenth of a second. The genetic algorithm generates
sixty of them, keeps the ones that performed most *consistently*, breeds and mutates those
into a new sixty, and repeats twenty-five times — then does the whole thing again from two
more random starts. It is directed trial and error at machine speed: a few thousand
distinct strategies tested in a quarter of an hour.

## The analogy, and where it breaks

Breeding: keep the best, cross them, mutate a little, repeat. Over generations the
population drifts toward what works.

Where it breaks — and this is the whole risk — is that **history does not push back**. A
farmer's crop either survives the winter or it does not. Our strategies are graded against
a fixed past, and if we test enough of them, one will look excellent by luck alone. Test
1,400 coin-flipping strategies and the best will have a beautiful track record and no
skill whatsoever.

Everything expensive in this system exists to tell those two cases apart.

## What the first three searches produced, and what happened to them

Three searches ran in 2026-08 over a wider version of the design — each of about twenty
company characteristics could be weighted freely, in either direction. All three ran only
on data to the end of 2021; 2022 onward was sealed off.

| | Search 1 (prices only) | Search 2 (adds financials) | Search 3 (adds overnight returns) |
|---|---:|---:|---:|
| Annual return, 2007/2010–2021 | 12.4% | 21.9% | 10.9% |
| The index, same period | 10.4% | 15.7% | 10.4% |
| Passes the luck correction? | yes (99.1%) | yes (99.9%) | yes (98.3%) |
| **Risk-adjusted score, 2022–2026** | **1.15 → 0.19** | **1.36 → 0.67** | **0.95 → 0.20** |

Every one passed the standard statistical correction for "how good would the luckiest of
1,400 worthless strategies have looked", and every one fell apart on the data it had never
seen. That is the most reliable thing the search has produced, and it is the right answer
to the question the project asked: **the correction is necessary and it is not
sufficient**. A space that wide will always contain something that fits the past.

## What changed in 2026-09

Five changes, each aimed at one way the earlier searches fooled themselves:

1. **A much smaller menu.** Instead of twenty freely weighted characteristics, nine
   *stories* the finance literature already settled — each a fixed bundle of
   characteristics with its direction fixed (low volatility is good, high profitability
   is good). The search picks at most three stories and how hard to back each. It cannot
   decide that expensive stocks are cheap. Everything without a story was removed, and
   the reason for every removal is written down.
2. **Graded on its worst stretch, not its average.** A strategy is scored on twelve
   random three-to-five-year slices of history and given the mark of its worst quarter.
   Something that only worked in 2009 dies during breeding instead of surviving to the
   test.
3. **Every extra rule costs points.** Each story backed, each characteristic read, and
   the "pull back in a crash" switch are all charged, so a simpler strategy that ties a
   complex one wins.
4. **The team, not the champion.** The single best strategy of a search is, almost by
   definition, the luckiest one. What the search hands on is the *average* of its thirty
   best survivors from three independent runs — what they agree on is a finding; what one
   found alone is a draw.
5. **Harsher trading costs during the search.** Strategies are charged twice the
   estimated cost of trading while they evolve, so a strategy that only works if trading
   is cheap never scores well. Frequent trading is charged again on top.

## Should you believe it?

No — and the machine now says so for a fourth time. The search over the new design
(`ga-families-1`) found a quality-plus-low-risk portfolio worth 17.8% a year at a 1.15
risk-adjusted score on 2010–2021, passing the luck correction at 99%. Tested once on
2022–2026 it returned 5.7% a year against the index's 13.5%: **it made money and it did
not beat the index**, the same verdict as the three searches before it. It was a more
*stable* strategy than they were — its worst loss halved and its trading did not blow up
— and it was not a better one. Note that this design was chosen *after* the 2022–2026
results had been read, which makes the decay a stronger refutation, not a weaker one.

The right next step is a **walk-forward test** — rerun the whole search inside each of
several past windows and grade each result on the window that followed — and the honest
sentence is: this is a more disciplined way of asking the question, and the answer, four
times now, is that a search over this history does not find something that beats the
index on the history that follows.

## What it costs to run

A quarter of an hour on one laptop. No cloud, no GPU, no market-data subscription — the
entire project runs on free public sources within a $20/month budget. Every one of the
thousands of strategies tested is permanently logged, which is what makes the statistical
correction possible at all: you cannot retroactively count the ideas you threw away.

---

# Part 2 — Technical

## The loop

```
for each of N seeds:
    population ← 60 genomes (9 seeded, one per family; the rest random)
    repeat 25 times:
        score every genome  → one full backtest each, cached by behavioural fingerprint
                              (the cache is shared across seeds)
        log every individual to the experiment registry as a trial
        keep the 4 best untouched (elitism)
        breed 52 children   → tournament select ×2, crossover, mutate
        inject 4 random immigrants
ensemble ← the top 30 distinct individuals across all seeds, beliefs averaged
score the ensemble once, log it as one more trial, store it beside the checkpoint
```

Fitness is `run_backtest`. That is the entire trick, and it is why the backtest engine was
built before any of this: an evolved strategy is scored by the same accounting, the same
next-open execution, the same cost model and the same survivorship-free universe as a
hand-written one. The scoreboard cannot tell them apart.

## The genome

A weighted sum of **prior-signed family composites** of cross-sectionally ranked
features, plus portfolio shape.

| genes | what |
|---:|---|
| 9 (or 5) | one weight per family, each in [0, 1]; at most 3 non-zero after decoding |
| 1 | `top_k` — how many names to hold (10–100) |
| 1 | `max_weight` — per-name cap (2%–20%) |
| 1 | weighting scheme — equal / score-rank / inverse-vol |
| 3 | regime gate — on/off, defensive exposure, volatility trigger |

15 genes for `families`, 11 for `families-price`. A family composite is the mean of its
members' percentile ranks, `rank` for a high-is-good member and `1 − rank` for a
low-is-good one, so every term is in [0, 1] and a missing member counts as average. The
cap is enforced by `Genome.effective()` — the dead zone and everything past the cap are
zeroed before the vector is decoded or fingerprinted — so a capped-out gene changes
neither the portfolio nor the trial count.

- **Ranks, not raw values.** One bad share count is worth "first place", not +25 standard
  deviations.
- **Signs are priors, not genes.** The search cannot back value backwards.
- **No expression trees.** Every point in the space reads as a sentence —
  `describe_genome()` prints one — because a winning parameter vector nobody can read is
  one nobody can check.
- **Presets are frozen.** The three older free-weight presets (`price`, `full`, `night`)
  still decode their checkpoints bit-identically; the families are new presets, not a
  rewrite.

## Fitness

The **monthly** Sharpe of the net-of-cost curve under the **pessimistic** cost setting,
computed on each of twelve random sub-periods of three to five years (drawn once from a
fixed seed, so every individual and every seed sees the same ones), aggregated as the
**25th percentile**, minus 0.02 per family backed, 0.01 per feature read, 0.03 if the
regime gate is on, and 0.03 per 100%/yr of turnover.

Four choices worth defending:

**Monthly, not daily.** Daily returns of a monthly-rebalanced portfolio are strongly
autocorrelated, so 4,861 daily observations carry roughly 176 independent ones. The
deflated Sharpe uses the monthly figure, so the search and its own significance test look
at the same number.

**The worst quarter, not the mean.** A mean-minus-spread over four folds still rewards a
rule that is excellent in two and ordinary in two. The 25th percentile of twelve
overlapping sub-periods does not.

**Costs inside, at the pessimistic setting.** If costs were applied after the fact the
search would evolve a high-turnover rule every time. Charging twice the spread while it
evolves, then charging turnover again on top, is the statement that the spread estimate
is the weakest input in the chain.

**Every rule has a price.** The −16% drawdown that made the first search's winner look
best was a two-parameter regime switch tuned on a window containing 2008. It now costs
0.03 of fitness to switch on.

> ⚠️ **Sub-periods measure consistency, not generalisation.** Every one sits inside the
> research window and the winner was selected using all of them. See ADR-032 and ADR-049.

## The ensemble

The deliverable is `EvolvedEnsemble`: the equal-weighted average of the **beliefs** of the
top 30 distinct individuals across every seed. Each member's weighted sum of ranks is
re-ranked to [0, 1] before averaging; the regime gate is a vote (step aside when at least
half the members would, at their mean defensive exposure); the holding count and cap are
the members' medians. Beliefs rather than portfolios, because averaging thirty twelve-name
portfolios gives a two-hundred-name one paying a dollar of commission minimum on every
name at this account size. The champion is still decoded and shown — it is the one
individual the ensemble is measured against. ADR-050.

## Operators

| | choice | why not the obvious alternative |
|---|---|---|
| Selection | tournament (size 3) | fitness can be negative, so roulette-wheel needs an invented offset |
| Crossover | BLX-α blend, 30% uniform | weights have no meaningful gene *order*, so a single cut point is arbitrary |
| Mutation | Gaussian nudge scaled per gene, plus rare full reset | the nudge alone cannot reintroduce diversity a population has lost |
| Survival | 4 elites + 4 random immigrants | pulling in opposite directions on purpose: cannot go backwards, cannot finish converging |
| Seeds | `--seeds N` independent populations, one study, one cache | one population converges on one neighbourhood; the pool does not |

Every operator takes an explicit `numpy.random.Generator`; nothing touches global state.
A search that cannot be replayed cannot be audited.

## Performance

**~0.1–0.2 s per individual**, so a 60 × 25 seed is three to six minutes. Four things buy
it: the price panel is built once and memoised; **feature ranks are precomputed once for
the whole population**; each individual is **one** backtest whose sub-periods are sliced
from the resulting curve; and the evaluation cache keys on the behavioural fingerprint and
is shared across seeds.

## The five defences against overfitting

1. **A small, bounded, readable space** — nine prior-signed families, at most three live.
2. **Every individual logged as a trial**, not just winners; the ensemble is one more.
3. **Worst-quarter fitness, net of pessimistic costs, minus a charge per rule.**
4. **The holdout is untouched.** Searches stop the day before 2022-01-01; reaching past it
   requires an explicit flag and is written to a ledger that cannot be disabled.
5. **The deliverable is an ensemble, not the champion.**

## Results, with the correction

The three free-weight searches, and what the reserved period did to them:

```
                                   ga-price-1        ga-full-2        ga-night-1
distinct trials                          1,403            1,407             1,407
expected max Sharpe (luck alone)          0.64             0.70              0.67
winner's monthly Sharpe                   1.36             1.78              1.20
deflated Sharpe                         0.9947           0.9995            0.9828
forward Sharpe (2022 onward)              0.19             0.67              0.20
```

The first search over the families (`ga-families-1`, three seeds, 4,179 distinct
individuals, ensemble of 30) converged on quality plus low risk and nothing else — every
one of the thirty members, gate off, 35 names — for 17.75%/yr at a 1.15 Sharpe under
realistic costs against SPY's 15.66%/0.95 on the same 2010–2021 window, deflated 0.992
over 4,180 trials. Because the survivors are near-clones, the ensemble is the champion
this time (17.33% against 17.38% at pessimistic costs). Forward-tested once, the same
day: **5.73%/yr at a 0.46 Sharpe on 2022-02 → 2026-09 against the index's 13.52%/0.82 —
decayed**, at all three cost settings, with the drawdown halved (−15.3%) and the
turnover check held. Four searches, four decays. The full account is in
[EVOLUTION.md](EVOLUTION.md) and on the searches page of `report genetic`.

## Running one

```bash
python -m sp500lab evolve run --study my-search --seeds 3
python -m sp500lab experiments deflate my-search
python -m sp500lab evolve ensemble my-search --all-costs --trades results/trades/my-search
python -m sp500lab evolve best my-search            # the champion, for comparison
python -m sp500lab report genetic --open
```

`--preset families-price` runs the five price-visible families from 2007; `--preset
price|full|night` runs the older free-weight spaces; `--fold-scheme contiguous
--aggregate mean_minus_std` reproduces the old objective. Every generation of every seed is
checkpointed to `data/experiments/evolve/<study>.jsonl`, the ensemble to
`<study>.ensemble.json`, and `evolve ensemble <study> --rebuild` builds one from any
checkpoint.

## What is deliberately absent

Multi-objective (NSGA-II) selection, island models, adaptive operator rates, expression
trees. All reasonable, and none address the failure mode that actually threatens this
project — which is not slow convergence, it is converging beautifully onto noise.

---

**Next:** [EVOLUTION.md](EVOLUTION.md) for the full detail · [STRATEGIES.md](STRATEGIES.md)
for what it is competing against · [EXPERIMENTS.md](EXPERIMENTS.md) for the trial log and
the holdout · ADR-031, ADR-032 and ADR-048 to ADR-050 in [DECISIONS.md](DECISIONS.md) for
the decisions.
