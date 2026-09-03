# How the genetic algorithm works

Two audiences, one page. **Part 1** assumes nothing. **Part 2** assumes you write software.
The full reference is [EVOLUTION.md](EVOLUTION.md); this is the five-minute version.

---

# Part 1 — For an executive

## The idea in one paragraph

We describe a trading strategy as a short list of numbers — how much to weight each of
about twenty company characteristics, how many stocks to hold, and whether to pull back
when markets turn. Any such list can be tested against twenty years of history in about a
sixth of a second. The genetic algorithm generates sixty of them, keeps the ones that
performed best, breeds and mutates those into a new sixty, and repeats twenty-five times.
It is directed trial and error at machine speed: roughly 1,400 distinct strategies tested
in five minutes.

## The analogy, and where it breaks

Breeding: keep the best, cross them, mutate a little, repeat. Over generations the
population drifts toward what works.

Where it breaks — and this is the whole risk — is that **history does not push back**. A
farmer's crop either survives the winter or it does not. Our strategies are graded against
a fixed past, and if we test enough of them, one will look excellent by luck alone. Test
1,400 coin-flipping strategies and the best will have a beautiful track record and no
skill whatsoever.

Everything expensive in this system exists to tell those two cases apart.

## What it produced

Three searches. All ran only on 2007–2021; the last four years of data were sealed off
before any of this started. (They have since been spent — see the postscript below —
which is why this section reads like a prediction: it was one.) The table shows the
first two; the third, over a feature set including the overnight/intraday decomposition,
is in the postscript.

| | Search 1 (price data only) | Search 2 (adds company financials) |
|---|---:|---:|
| Period | 2007–2021 | 2010–2021 |
| Annual return | **12.4%** | **21.9%** |
| The index, same period | 10.4% | 15.7% |
| Worst peak-to-trough loss | **−16%** | −28% |
| The index's worst loss | −55% | −34% |
| Survives being charged full costs? | Yes (11.9%) | Yes (21.6%) |

Search 1's headline is the drawdown, not the return: it lost a sixth of its value at the
worst point of 2008 where the index lost more than half. Search 2's headline is that it
independently rediscovered a well-known school of investing — buy profitable companies
whose earnings are backed by cash — without ever being told that school exists.

## Should you believe it?

There is a standard statistical correction for exactly this problem: given that we tested
1,400 strategies, how good would the *luckiest worthless one* have looked? The answer here
is a risk-adjusted score of about **0.64**. Our winners scored **1.36** and **1.78**.

The correction converts that into a probability that the result is not luck: **99.5%** and
**99.9%**. Both clear the conventional 95% bar comfortably.

That is a real and meaningful test. It is also not the same as proof, and there are three
specific reasons to hold this loosely:

1. **At the time this was written, no unseen data had been touched.** We reserved 2022
   onward for a single look. That look has now happened, and the postscript below says
   what it found — which is why the rest of this section is kept in its original tense:
   it documents what was believed before the answer was known.
2. **Search 1's low drawdown comes from a switch that was tuned on the crisis it is being
   praised for.** It learned "reduce exposure when markets fall" from a period containing
   2008 and 2020. Whether that generalises is unknown.
3. **Search 2 trades a narrower list of companies.** Company financial filings only exist
   in machine-readable form from 2010, and only for two-thirds of the companies that have
   passed through the index — disproportionately the ones that survived. Its period was
   also a much easier market.

## What it costs to run

A five-minute search on one laptop. No cloud, no GPU, no market-data subscription — the
entire project runs on free public sources within a $20/month budget. Every one of the
1,400 strategies tested is permanently logged, which is what makes the statistical
correction above possible at all: you cannot retroactively count the ideas you threw away.

## The decision this supports

Not "deploy capital". The right next step is a **walk-forward test**: re-run the entire
search inside each of several historical windows and grade each winner on the window that
followed. That produces genuinely out-of-sample evidence without spending the reserved
period, and it costs about twenty minutes of compute. After that — and only after that —
the reserved 2022–2026 period is worth its single look.

## Postscript: the look happened, and the machine was right to be suspicious

The reserved period was spent in 2026-08 (ADR-033). Both winners above **decayed** —
Search 1 from a 1.15 Sharpe to 0.19, Search 2 from 1.36 to 0.67 — which is precisely what
the multiple-testing arithmetic on this page predicts for the best of 1,400 draws, while
simpler hand-written strategies largely matched their predictions. A third search was
then run over a new feature set (the overnight/intraday decomposition and the dividend
calendar — preset `night`, ADR-038): 1,404 trials, winner 10.9%/yr at Sharpe 0.95 with a
−18% drawdown, deflated Sharpe 0.983 — and forward-tested the same week, **it decayed to
0.20**. Three searches, three excellent in-sample winners, three decays. That
replication, not any single winner, is the most solid thing the genetic algorithm has
produced: the correction on this page is not pessimism, it is a forecast that has now
verified three times. (The third test is weaker than the first two — it was built after
the first forward results were read; ADR-037 records that contamination.)

---

# Part 2 — Technical

## The loop

```
population ← 60 genomes (8 seeded from known strategies, rest random)
repeat 25 times:
    score every genome  → one full backtest each, cached by behavioural fingerprint
    log every individual to the experiment registry as a trial
    keep the 4 best untouched (elitism)
    breed 52 children   → tournament select ×2, crossover, mutate
    inject 4 random immigrants
```

Fitness is `run_backtest`. That is the entire trick, and it is why the backtest engine was
built before any of this: an evolved strategy is scored by the same accounting, the same
next-open execution, the same cost model and the same survivorship-free universe as a
hand-written one. The scoreboard cannot tell them apart.

## The genome

A weighted sum of **cross-sectionally ranked features**, plus portfolio shape.

| genes | what |
|---:|---|
| 13 or 23 | one weight per feature, each in [−1, +1] |
| 1 | `top_k` — how many names to hold (10–100) |
| 1 | `max_weight` — per-name cap (2%–20%) |
| 1 | weighting scheme — equal / score-rank / inverse-vol |
| 3 | regime gate — on/off, defensive exposure, volatility trigger |

19 genes for the price preset, 29 with fundamentals. Deliberately small:

- **Ranks, not raw values.** One bad share count is worth "first place", not +25 standard
  deviations.
- **A dead zone at ±0.10.** Weights below it are exactly zero, so "how many features does
  this use" has an answer, parsimony pressure has something to grip, and two behaviourally
  identical genomes deduplicate to one trial.
- **No expression trees.** No evolved arithmetic, no products of indicators. Every point in
  the space reads as a sentence — `describe_genome()` prints one — because a winning
  parameter vector nobody can read is one nobody can check.

## Fitness

Default: the **monthly** Sharpe, computed on each of four contiguous sub-periods separated
by a one-month embargo, aggregated as `mean − 0.5 × std`. Optional turnover and
per-feature complexity penalties.

Three choices worth defending:

**Monthly, not daily.** Daily returns of a monthly-rebalanced portfolio are strongly
autocorrelated, so 4,861 daily observations carry roughly 176 independent ones. The
deflated Sharpe uses the monthly figure, so the search and its own significance test look
at the same number.

**Not raw return.** Handed return, a long-only search finds leverage as concentration:
`top_k` at its floor, ten names, a magnificent CAGR and a 70% drawdown.

**Fold consistency, not full-sample.** A strategy that made everything in 2009 and nothing
since has a fine full-sample Sharpe and a poor fold score. Folds are contiguous with an
embargo, never random K-fold — financial data is autocorrelated and a shuffled split leaks
across the boundary. The folds are sliced from the single equity curve each individual
already produced, so four folds cost zero extra backtests.

> ⚠️ **Folds measure consistency, not generalisation.** Every fold sits inside the research
> window and the winner was selected using all of them. See ADR-032.

## Operators

| | choice | why not the obvious alternative |
|---|---|---|
| Selection | tournament (size 3) | fitness is a Sharpe — it can be negative, so roulette-wheel needs an invented offset |
| Crossover | BLX-α blend, 30% uniform | feature weights have no meaningful gene *order*, so a single cut point is arbitrary |
| Mutation | Gaussian nudge scaled per gene, plus rare full reset | the nudge alone cannot reintroduce diversity a population has lost |
| Survival | 4 elites + 4 random immigrants | pulling in opposite directions on purpose: cannot go backwards, cannot finish converging |

Every operator takes an explicit `numpy.random.Generator`; nothing touches global state. A
search that cannot be replayed cannot be audited.

## Performance

**~0.15 s per individual**, so 1,500 evaluations is about five minutes. Four things buy it:

1. the price panel is built once and memoised;
2. **feature ranks are precomputed once for the whole population** — a rank depends on the
   date and the tradable mask, not on the genome, so a 1,500-evaluation search was
   otherwise recomputing each one 1,500 times (0.29 s → 0.15 s);
3. each individual is **one** backtest whose folds are sliced from the resulting curve;
4. the evaluation cache keys on the behavioural fingerprint, so a converged population is
   free to re-evaluate.

## The four defences against overfitting

1. **A small, bounded, readable space** — above.
2. **Every individual logged as a trial**, not just winners. The deflated Sharpe needs the
   trial count and the spread of trial Sharpes, and neither is recoverable afterwards. The
   registry's trial count and the GA's evaluation cache key on the *same* behavioural
   fingerprint — counting clones twice over-deflates, not deduplicating at all
   under-deflates.
3. **Fold-consistency fitness** — above.
4. **The holdout is untouched.** Searches stop the day before 2022-01-01. Reaching past it
   requires an explicit flag and is written to a ledger that cannot be disabled.

## Results, with the correction

```
                                   ga-price-1        ga-full-2
distinct trials                          1,403            1,407
expected max Sharpe (luck alone)          0.64             0.70
winner's monthly Sharpe                   1.36             1.78
deflated Sharpe                         0.9947           0.9995
```

Both winners are readable. `ga-price-1` converged on low volatility, low beta, low
idiosyncratic volatility and low lottery-payoff — the neighbourhood `low_vol` and
`lottery_averse` occupy — and turned its regime gate hard on. `ga-full-2` found high ROE,
high gross profitability and low accruals (Novy-Marx and Sloan, independently), ranked
book-to-market *negatively* while ranking earnings yield positively, and turned the regime
gate off. Its window contains no 2008, which is the cleanest evidence in this project that
the two searches found different things because they looked at different periods.

## Running one

```bash
python -m sp500lab evolve run --study my-search --preset price --generations 25 --population 60
python -m sp500lab experiments deflate my-search
python -m sp500lab evolve best my-search --all-costs --trades results/trades/my-search
python -m sp500lab report genetic --open
```

Everything on this page is also generated as three linked HTML pages by
`sp500lab report genetic`, with the numbers filled in from the searches that have
actually run — see `reports/genetic_algorithm/`.

Every generation is checkpointed to `data/experiments/evolve/<study>.jsonl`, so a killed
run resumes and a finished one can be re-read without re-running. Ctrl-C returns the state
at the last completed generation.

## What is deliberately absent

Multi-objective (NSGA-II) selection, island models, adaptive operator rates. All reasonable,
and none address the failure mode that actually threatens this project — which is not slow
convergence, it is converging beautifully onto noise.

---

**Next:** [EVOLUTION.md](EVOLUTION.md) for the full detail · [STRATEGIES.md](STRATEGIES.md)
for what it is competing against · [EXPERIMENTS.md](EXPERIMENTS.md) for the trial log and
the holdout · ADR-031 and ADR-032 in [DECISIONS.md](DECISIONS.md) for the decisions.
