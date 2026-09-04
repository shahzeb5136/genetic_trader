"""The genetic algorithm: selection, crossover, mutation, and an honest fitness function.

    python -m sp500lab evolve run --study ga-1 --seeds 3
    python -m sp500lab experiments deflate ga-1        # <- read this before believing it
    python -m sp500lab evolve ensemble ga-1 --all-costs

Reading order
-------------
    ../strategies/genome.py   the search space: nine prior-signed families, capped
    config.py                 every knob, with the 2026-09 defaults and why
    fitness.py                what is being maximised, and everything subtracted from it
    operators.py              selection, crossover, mutation - no finance in here
    engine.py                 the loop, the cache, the checkpoints, the ensemble
    cli.py                    `sp500lab evolve ...`

Two things to remember. The reported Sharpe of the winner of a 1,500-individual search
is an estimate of skill PLUS the selection effect, and the selection effect grows with the
number of trials: `sp500lab experiments deflate <study>` is the correction, it is already
wired up, and a result that does not survive it is not a result. And the winner is not
the deliverable: the search hands on the average signal of its top N survivors, pooled
across seeds, because the single best individual is the most luck-contaminated object in
the population (ADR-050).

See docs/EVOLUTION.md, ADR-031, ADR-032 and ADR-048 to ADR-050.
"""

from __future__ import annotations

from .config import EvolutionConfig
from .engine import (EvolutionResult, build_ensemble, champion, ensemble_strategy,
                     evolve, load_ensemble, load_history, load_individuals,
                     load_population, seed_vectors, study_config, study_preset,
                     winners)
from .fitness import Evaluation, Folds, Objective

__all__ = [
    "EvolutionConfig", "EvolutionResult", "evolve",
    "Folds", "Objective", "Evaluation",
    "load_history", "load_population", "load_individuals", "seed_vectors",
    "study_preset", "study_config", "champion", "winners",
    "load_ensemble", "ensemble_strategy", "build_ensemble",
]
