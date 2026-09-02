"""The genetic algorithm: selection, crossover, mutation, and an honest fitness function.

    python -m sp500lab evolve run --study ga-1 --generations 25 --population 60
    python -m sp500lab experiments deflate ga-1        # <- read this before believing it

Reading order
-------------
    ../strategies/genome.py   the search space, and what a point in it becomes
    fitness.py                what is being maximised, and everything subtracted from it
    operators.py              selection, crossover, mutation - no finance in here
    engine.py                 the loop, the cache, the checkpoints
    cli.py                    `sp500lab evolve ...`

The one thing to remember: the reported Sharpe of the winner of a 1,500-individual search
is an estimate of skill PLUS the selection effect, and the selection effect grows with the
number of trials. `sp500lab experiments deflate <study>` is the correction, it is already
wired up, and a result that does not survive it is not a result.

See docs/EVOLUTION.md, ADR-031 and ADR-032.
"""

from __future__ import annotations

from .engine import (EvolutionConfig, EvolutionResult, evolve, load_history,
                     load_population, seed_vectors, study_preset, winners)
from .fitness import Evaluation, Folds, Objective

__all__ = [
    "EvolutionConfig", "EvolutionResult", "evolve",
    "Folds", "Objective", "Evaluation",
    "load_history", "load_population", "seed_vectors",
    "study_preset", "winners",
]
