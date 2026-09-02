"""Selection, crossover, mutation. The evolution, and nothing about finance.

Everything here operates on float vectors inside a box. That separation is the point: the
operators cannot accidentally encode a market opinion, and the genome cannot accidentally
encode a search strategy.

Choices, and what each one is defending against
------------------------------------------------
**Tournament selection**, not fitness-proportionate. Fitness here is a Sharpe ratio, which
can be negative and whose absolute scale is arbitrary - roulette-wheel selection needs
positive weights and would have to be given an offset somebody invented. A tournament only
needs an ordering, and `tournament_size` maps directly onto how hard the search pushes.

**Blend crossover (BLX-alpha)**, not single-point. A genome of feature weights has no
meaningful gene ORDER, so a cut point is arbitrary. BLX samples each gene from an interval
slightly wider than the two parents span, which lets a population that has converged on
an interior region still reach outside it - the property that keeps a real-valued GA from
stalling.

**Two mutations, not one.** A Gaussian nudge scaled to each gene's own range explores
locally; a full reset, applied rarely, is the only operator that can reintroduce diversity
a population has lost entirely. With just the nudge, a converged population stays
converged.

**Duplicate suppression.** Genomes that behave identically waste an evaluation and, worse,
corrupt the deflated Sharpe: the registry counts distinct fingerprints as trials, so a
population full of clones would report a small trial count for a large search and deflate
the winner far too gently. Duplicates are mutated again rather than dropped, so the
population size stays fixed.

**Elitism plus immigration.** The best individuals survive untouched, so the search can
never go backwards; a few random newcomers arrive every generation, so it can never
finish converging either. The two pull in opposite directions on purpose.

Determinism
-----------
Every operator takes an explicit `numpy.random.Generator`. Nothing here calls the global
numpy random state. A search that cannot be replayed exactly cannot be audited, and a
search that cannot be audited is a story about a number.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

#: How far outside the interval spanned by two parents a child may be sampled, as a
#: fraction of that interval. 0.5 is the value Eshelman & Schaffer recommended and it is
#: still the usual default: enough to escape a converged region, not so much that
#: crossover degenerates into a random draw.
BLX_ALPHA = 0.5


def tournament_select(fitness: np.ndarray, rng: np.random.Generator,
                      size: int = 3) -> int:
    """Index of the winner of one tournament among `size` random individuals.

    Sampled with replacement, which is standard and matters at small population sizes:
    without it the last few selections of a generation would draw from a shrinking pool
    and the effective selection pressure would drift during the generation.
    """
    n = len(fitness)
    if n == 0:
        raise ValueError("cannot select from an empty population")
    picks = rng.integers(0, n, size=min(max(size, 1), n))
    return int(picks[int(np.argmax(fitness[picks]))])


def blend_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
                    alpha: float = BLX_ALPHA) -> np.ndarray:
    """One child, each gene drawn from the interval its parents span, widened by alpha."""
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    width = (hi - lo) * alpha
    return rng.uniform(lo - width, hi + width)


def uniform_crossover(a: np.ndarray, b: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    """One child taking each gene from one parent or the other, with equal probability.

    Kept alongside BLX because it preserves parent values exactly. On the categorical
    genes - the weighting scheme, the regime switch - a blended value is a value neither
    parent had, and for a two-way switch that means the child is neither parent.
    """
    take_a = rng.random(len(a)) < 0.5
    return np.where(take_a, a, b)


def mutate(vector: np.ndarray, genome, rng: np.random.Generator,
           rate: float = 0.15, sigma: float = 0.15,
           reset_rate: float = 0.02) -> np.ndarray:
    """Perturb some genes, resample a few outright, and return a valid individual.

    `sigma` is a fraction of each gene's own range, so one number governs a weight in
    [-1, 1] and a holding count in [10, 100] without either dominating. Clipping happens
    once, at the end, through the genome - a mutation that overshoots a bound produces a
    boundary individual rather than an exception mid-population.
    """
    v = np.asarray(vector, dtype=np.float64).copy()
    spans = genome.spans

    touched = rng.random(len(v)) < rate
    if touched.any():
        v[touched] += rng.normal(0.0, sigma, size=int(touched.sum())) * spans[touched]

    reset = rng.random(len(v)) < reset_rate
    if reset.any():
        fresh = genome.random(rng)
        v[reset] = fresh[reset]

    return genome.clip(v)


def breed(parents: list[np.ndarray], genome, rng: np.random.Generator, *,
          crossover_rate: float = 0.7, mutation_rate: float = 0.15,
          mutation_sigma: float = 0.15, reset_rate: float = 0.02,
          uniform_share: float = 0.3) -> np.ndarray:
    """Two parents in, one mutated child out."""
    a, b = parents[0], parents[1]
    if rng.random() < crossover_rate:
        child = (uniform_crossover(a, b, rng) if rng.random() < uniform_share
                 else blend_crossover(a, b, rng))
    else:
        child = a.copy()
    return mutate(child, genome, rng, rate=mutation_rate, sigma=mutation_sigma,
                  reset_rate=reset_rate)


def deduplicate(population: list[np.ndarray], genome, rng: np.random.Generator,
                seen: set[str], attempts: int = 8) -> tuple[list[np.ndarray], int]:
    """Re-mutate individuals whose behaviour is already in `seen`. Returns (pop, n_forced).

    Fingerprints are behavioural, not bitwise: two vectors that differ only inside the
    dead zone produce the same portfolio and count as one. That is also what the trial
    count in the experiment registry means, so keeping the two definitions identical is
    what makes the deflated Sharpe honest about how large the search really was.

    An individual that cannot be made novel in `attempts` tries is replaced with a random
    one. That happens when the population has genuinely converged, and a random immigrant
    is the correct response to it.
    """
    out, forced = [], 0
    for vector in population:
        v = vector
        for _ in range(attempts):
            if genome.fingerprint(v) not in seen:
                break
            v = mutate(v, genome, rng, rate=0.4, sigma=0.3, reset_rate=0.1)
            forced += 1
        else:
            v = genome.random(rng)
            forced += 1
        out.append(v)
        seen.add(genome.fingerprint(v))
    return out, forced


def diversity(population: list[np.ndarray], genome) -> float:
    """Mean normalised spread across genes: 0 is a clone army, ~0.3 is healthy.

    Reported every generation because the most common way a GA fails is silently. A run
    whose diversity collapses in generation 4 spent the next forty generations
    re-evaluating one individual, and the fitness curve looks like convergence rather
    than like the stall it is.
    """
    if len(population) < 2:
        return 0.0
    m = np.vstack(population)
    spans = np.where(genome.spans > 0, genome.spans, 1.0)
    return float(np.mean(m.std(axis=0) / spans))
