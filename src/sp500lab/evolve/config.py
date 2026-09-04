"""Everything that determines a search, in one dataclass with no heavy imports.

Separate from `engine.py` so the command-line parser can read the defaults without
importing pandas and the backtest engine: `sp500lab --help` builds every parser, and the
flags below must show the same defaults the engine uses rather than a copy that drifts.

The defaults are the 2026-09 design (ADR-048, ADR-049, ADR-050) and each one is a
decision, not a constant somebody inherited:

    preset            `families` - nine prior-signed stories, at most three live at once
    costs             `pessimistic` - the search is charged twice the estimated spread
    fold_scheme       `random` - twelve sub-periods of three to five years, drawn once
    aggregate         `quantile` at 0.25 - the worst quarter of those sub-periods decides
    penalties         turnover, per feature, per family, and for switching the gate on
    n_seeds           1 - raise it; the ensemble is meant to be pooled across seeds
    ensemble_size     30 - the deliverable is the average of the best thirty, not the best

A population of 60 over 25 generations is 1,500 evaluations per seed, which is enough to
find structure if there is any and small enough that the deflated Sharpe still has a
chance of clearing 0.95. Ten thousand individuals sounds more serious and mostly buys a
larger correction to apply to the winner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class EvolutionConfig:
    """Everything that determines a search. Stored verbatim with its result."""

    study: str = "ga"
    preset: str = "families"
    population: int = 60
    generations: int = 25
    elite: int = 4
    immigrants: int = 4
    tournament_size: int = 3
    crossover_rate: float = 0.7
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.15
    reset_rate: float = 0.02
    seed: int = 0
    #: How many independent searches to run, seeds `seed, seed+1, ...`. They share the
    #: study, the objective and the evaluation cache, and the ensemble pools them all.
    n_seeds: int = 1
    seed_with_baselines: bool = True

    start: str = "2007-04-01"
    end: str | None = None
    #: The cost setting the search is CHARGED. Pessimistic by default (ADR-049): the
    #: half-spread estimate is the weakest number in the chain, and a rule that only
    #: works if that estimate is kind is not a rule.
    costs: str = "pessimistic"
    liquidity_floor: float = 0.0
    holdout: str = "exclude"
    initial_capital: float = 100_000.0

    metric: str = "sharpe_monthly"
    #: `random`: `n_folds` sub-periods of `fold_min_years`..`fold_max_years`, drawn once
    #: from `fold_seed` so every individual and every seed is scored on the same ones.
    #: `contiguous`: the ADR-032 folds, equal spans separated by an embargo.
    fold_scheme: str = "random"
    n_folds: int = 12
    fold_min_years: float = 3.0
    fold_max_years: float = 5.0
    fold_seed: int = 0
    embargo_days: int = 31
    #: `quantile` scores an individual at the `quantile` point of its sub-period
    #: metrics - the worst quarter of them by default. A rule that only works in one
    #: stretch is killed during evolution rather than surviving to the test set.
    aggregate: str = "quantile"
    quantile: float = 0.25
    dispersion_weight: float = 0.5

    #: Charged on top of the cost model, per 100% of annual turnover. Model risk, not
    #: costs: it says the spread estimate might be wrong.
    turnover_penalty: float = 0.03
    #: Per feature the individual reads.
    complexity_penalty: float = 0.01
    #: Per family it backs - each one is a rule.
    family_penalty: float = 0.02
    #: For switching the regime gate on - two more tuned parameters and one more rule.
    gate_penalty: float = 0.03

    #: The top N distinct individuals across every seed, averaged. 0 disables it.
    ensemble_size: int = 30

    log_runs: bool = True
    checkpoint: bool = True
    max_evaluations: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def seeds(self) -> list[int]:
        return [int(self.seed) + k for k in range(max(int(self.n_seeds), 1))]
