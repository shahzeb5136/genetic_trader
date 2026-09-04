"""The genetic algorithm: a population of genomes, scored by the backtest engine.

    from sp500lab.evolve import EvolutionConfig, evolve
    result = evolve(EvolutionConfig(study="ga-run-1", generations=20, population=60))
    print(result.summary())

What this actually is
---------------------
A loop over `run_backtest`. That is the whole trick, and it is why the engine came first:
the fitness function IS the backtest, so an evolved strategy is scored by exactly the
same accounting, the same next-open execution, the same costs and the same
survivorship-free universe as a hand-written one. The scoreboard cannot tell them apart,
which is the point of the competition (see backtest/strategy.py).

The defences, all of which are load-bearing
--------------------------------------------
A GA overfits more aggressively than almost anything else, because it is *explicitly*
maximising the number you report. Every one of these is here for that reason:

1. **A small, bounded, readable search space.** Weighted sums of ranked features - and,
   since ADR-048, at most three prior-signed FAMILIES of them - not evolved expression
   trees. See strategies/genome.py.
2. **Every individual is logged as a trial**, not just the winners. The deflated Sharpe
   needs the trial count and the spread of trial Sharpes, and neither can be recovered
   after the fact (ADR-026). This is on by default and turning it off is a decision.
3. **Fitness is the worst quarter of many sub-periods, net of pessimistic costs, minus a
   charge per rule.** An individual that made all its money in one 18-month stretch, or
   only under a kind spread estimate, or with nine features where three would do, scores
   badly however good its headline looks. See fitness.py and ADR-049.
4. **The holdout is untouched.** The search runs on the research window and stops the day
   before 2022-01-01. Testing the winner there is a separate, deliberate, recorded act
   (ADR-025), and doing it more than once destroys the only out-of-sample evidence there
   is.
5. **The deliverable is an ensemble, not the champion.** The best single individual is
   the maximum over thousands of draws - the most luck-contaminated object in the whole
   population - so what a search hands to the forward test is the average signal of its
   top N survivors, pooled across every seed it ran (ADR-050).

Speed, and where it comes from
-------------------------------
About 0.15 seconds per evaluation, so a 60 x 25 search is roughly 3 minutes of fitness
evaluation per seed. Three things buy that and a change to any of them will cost it: the
price panel is built once and memoised, the feature ranks are precomputed once for the
whole population (features/ranked.py), and each individual is a single backtest whose
sub-periods are sliced from the resulting curve rather than re-run.

The evaluation cache is the fourth. Two genomes that differ only inside the dead zone
(or past the family cap) produce the same portfolio, and the cache keys on that
behavioural fingerprint - so a converged population costs nothing to re-evaluate, and the
registry's trial count stays equal to the number of distinct hypotheses actually tested.
The cache is shared across the seeds of one search for the same reason.

Resumability
------------
Every generation of every seed is appended to `data/experiments/evolve/<study>.jsonl`
with the full population, and the ensemble is written beside it as
`<study>.ensemble.json`. A finished run can be re-read without re-running it, and the
ensemble can be rebuilt from the checkpoint alone. Interrupting with Ctrl-C returns the
result computed so far rather than losing it.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest import registry as reg
from ..backtest.engine import run_backtest
from ..backtest.panel import build_panel
from ..paths import EXPERIMENTS_DIR
from ..strategies.evolvable import EvolvedEnsemble, from_vector
from ..strategies.genome import (PRESET_MIN_DATE, REGIME_FEATURES, active_families,
                                 active_features, alpha_genome, describe_genome,
                                 preset_families, preset_features, preset_kind)
from . import operators as ops
from .config import EvolutionConfig
from .fitness import UNFIT, Evaluation, Folds, Objective, benchmark_monthly_sharpe

log = logging.getLogger(__name__)

EVOLVE_DIR = EXPERIMENTS_DIR / "evolve"

#: Extra raw columns the pre-ranked panel must carry: the regime gate reads macro levels,
#: and the defensive sleeve ranks volatility.
RANK_PASSTHROUGH = REGIME_FEATURES

__all__ = ["EvolutionConfig", "EvolutionResult", "evolve", "seed_vectors",
           "load_history", "load_population", "load_individuals", "champion",
           "study_preset", "study_config", "winners", "load_ensemble",
           "ensemble_strategy", "build_ensemble", "EVOLVE_DIR"]


@dataclass
class EvolutionResult:
    """What a search produced, and enough context to argue with it."""

    config: EvolutionConfig
    genome_name: str
    best: Evaluation | None
    hall_of_fame: list[Evaluation] = field(default_factory=list)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_evaluations: int = 0
    n_distinct: int = 0
    elapsed: float = 0.0
    interrupted: bool = False
    objective: dict = field(default_factory=dict)
    seeds: list[int] = field(default_factory=list)
    #: The stored ensemble record (see `_ensemble_record`), or None when none was built.
    ensemble: dict | None = None

    def summary(self) -> str:
        c = self.config
        lines = [
            "=" * 76,
            f"EVOLUTION  study={c.study}  preset={c.preset}",
            "=" * 76,
            f"  {self.n_evaluations} evaluations, {self.n_distinct} distinct "
            f"individuals, {len(self.history)} generation(s) over "
            f"{len(self.seeds)} seed(s) {self.seeds}, {self.elapsed:.0f}s",
            f"  objective: {self.objective.get('metric')} at the "
            f"{self.objective.get('aggregate')} of {self.objective.get('n_folds')} "
            f"{self.objective.get('fold_scheme')} sub-periods, {c.costs} costs",
            f"  charged: turnover {c.turnover_penalty}/100%, feature "
            f"{c.complexity_penalty}, family {c.family_penalty}, gate {c.gate_penalty}",
        ]
        if self.interrupted:
            lines.append("  !! interrupted; this is the state at the last completed "
                         "generation, and no ensemble was built - run "
                         f"`sp500lab evolve ensemble {c.study} --rebuild`")
        if self.best is None:
            lines.append("  no individual scored a finite fitness")
            return "\n".join(lines)

        b = self.best
        lines += [
            "",
            f"  CHAMPION  fitness {b.fitness:.4f}   CAGR {b.cagr * 100:.2f}%   "
            f"Sharpe {b.sharpe:.2f}   maxDD {b.max_drawdown * 100:.1f}%   "
            f"turnover {b.turnover * 100:.0f}%",
            "",
            describe_genome(alpha_genome(c.preset), b.vector),
        ]
        e = self.ensemble
        if e and e.get("evaluation"):
            ev = e["evaluation"]
            lines += [
                "",
                f"  ENSEMBLE of {e['size']}  robust score {_fmt(ev.get('fitness'))}   "
                f"CAGR {_fmt(ev.get('cagr'), 100, '%')}   "
                f"Sharpe {_fmt(ev.get('sharpe'))}   "
                f"maxDD {_fmt(ev.get('max_drawdown'), 100, '%', 1)}   "
                f"turnover {_fmt(ev.get('turnover'), 100, '%', 0)}",
                "",
                e.get("prose", ""),
                "",
                "  The ensemble, not the champion, is what the forward test gets "
                "(ADR-050).",
            ]
        lines += [
            "",
            "  These numbers have not been corrected for the search that produced them.",
            f"  Run `sp500lab experiments deflate {c.study}` before believing them.",
        ]
        return "\n".join(lines)

    def leaderboard(self, top: int = 10) -> pd.DataFrame:
        rows = [e.as_row() for e in self.hall_of_fame[:top]]
        return pd.DataFrame(rows)

    def best_strategy(self, pre_ranked: bool = False):
        """The champion as a strategy object, ready to re-run or export trades from."""
        if self.best is None:
            raise ValueError("this search produced no scorable individual")
        return from_vector(self.best.vector, self.config.preset, pre_ranked=pre_ranked)


def _fmt(x, scale: float = 1.0, suffix: str = "", dp: int = 2) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    return f"{v * scale:.{dp}f}{suffix}"


# --------------------------------------------------------------------------
# The search
# --------------------------------------------------------------------------

def evolve(config: EvolutionConfig) -> EvolutionResult:
    """Run a search. Every individual is backtested, scored and logged.

    With `n_seeds > 1` the loop below runs once per seed - independent populations,
    the same objective, one shared evaluation cache - and the ensemble at the end pools
    the best individuals of all of them.
    """
    t0 = time.perf_counter()
    genome = alpha_genome(config.preset)
    panel, ranked = _ranked_inputs(config.preset, config.liquidity_floor)

    start = max(config.start, PRESET_MIN_DATE[config.preset] or config.start)
    objective = _objective(config, start)
    log.info("evolve: %s, %d x %d x %d seed(s) over %s, %s sub-periods: %s",
             genome.name, config.population, config.generations, len(config.seeds),
             start, objective.folds.scheme if objective.folds else "no",
             objective.folds.describe() if objective.folds else "whole window")

    cache: dict[str, Evaluation] = {}
    history: list[dict] = []
    interrupted = False
    seeds_run: list[int] = []

    with reg.study(config.study):
        try:
            for seed in config.seeds:
                rng = np.random.default_rng(seed)
                seen: set[str] = set()
                population = _initial_population(genome, config, rng)
                seeds_run.append(seed)
                for generation in range(config.generations):
                    if (config.max_evaluations
                            and len(cache) >= config.max_evaluations):
                        log.info("evolve: evaluation budget reached")
                        break

                    scored = [_evaluate(v, genome, config, panel, ranked, objective,
                                        cache, start)
                              for v in population]
                    fitness = np.array([e.fitness for e in scored], dtype=np.float64)
                    row = _generation_row(generation, scored, fitness, population,
                                          genome)
                    row["seed"] = int(seed)
                    history.append(row)
                    _log_generation(row)
                    if config.checkpoint:
                        _checkpoint(config, seed, generation, scored, row)

                    if generation + 1 >= config.generations:
                        break
                    population = _next_generation(population, fitness, genome, config,
                                                  rng, seen)
        except KeyboardInterrupt:
            interrupted = True
            log.warning("evolve: interrupted - returning the state so far")

        ranking = sorted((e for e in cache.values() if e.ok),
                         key=lambda e: e.fitness, reverse=True)
        ensemble = None
        if (config.ensemble_size > 0 and len(ranking) >= 2 and not interrupted):
            try:
                ensemble = _ensemble_record(
                    config, seeds_run, ranking[:config.ensemble_size], ranking[0],
                    objective, panel, ranked, start)
                if config.checkpoint:
                    write_ensemble(ensemble)
            except KeyboardInterrupt:
                interrupted = True
                log.warning("evolve: interrupted while scoring the ensemble")
            except Exception as exc:                              # noqa: BLE001
                log.warning("evolve: could not build the ensemble: %s", exc)

    return EvolutionResult(
        config=config, genome_name=genome.name,
        best=ranking[0] if ranking else None,
        hall_of_fame=ranking[:max(25, config.ensemble_size)],
        history=pd.DataFrame(history),
        n_evaluations=sum(int(r["evaluated"]) for r in history),
        n_distinct=len(cache),
        elapsed=time.perf_counter() - t0,
        interrupted=interrupted,
        objective=objective.describe(),
        seeds=seeds_run,
        ensemble=ensemble,
    )


def _ranked_inputs(preset: str, liquidity_floor: float = 0.0):
    """The memoised panel and the pre-ranked feature panel a preset's strategies read."""
    from ..features import build_features
    from ..features.ranked import rank_panel

    panel = build_panel()
    features = build_features(panel=panel)
    # The defensive sleeve ranks vol_126d whether or not the preset scores it.
    names = tuple(dict.fromkeys(preset_features(preset) + ("vol_126d",)))
    ranked = rank_panel(features, panel, names,
                        keep_raw=RANK_PASSTHROUGH + ("vol_126d",),
                        liquidity_floor=liquidity_floor)
    return panel, ranked


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _evaluate(vector: np.ndarray, genome, config: EvolutionConfig, panel, ranked,
              objective: Objective, cache: dict, start: str) -> Evaluation:
    """One individual: backtest it, score it, remember it.

    The cache key is the BEHAVIOURAL fingerprint, so two genomes that differ only inside
    the dead zone (or past the family cap) are one evaluation and one trial. That keeps
    the registry's trial count equal to the number of distinct hypotheses tested, which
    is what the deflated Sharpe needs it to mean.
    """
    key = genome.fingerprint(vector)
    hit = cache.get(key)
    if hit is not None:
        return hit

    strategy = from_vector(vector, config.preset, pre_ranked=True)
    n_active = len(active_features(genome, vector))
    n_families = len(active_families(genome, vector))
    gate_on = genome.decode(vector)["use_regime"] == "on"
    try:
        result = run_backtest(
            strategy, panel=panel, features=ranked, start=start, end=config.end,
            costs=config.costs, initial_capital=config.initial_capital,
            liquidity_floor=config.liquidity_floor, holdout=config.holdout,
            seed=config.seed, study=config.study, log_run=config.log_runs,
            # Curves and trade ledgers are per-individual overhead a search cannot
            # afford: ~7 KB and ~12,000 rows each. Re-running a winner with both on is
            # the SAME trial - the fingerprint does not include them - so nothing is
            # lost by leaving them off here.
            log_curve=False, record_trades=False, benchmark=None,
            notes=f"ga:{config.study}")
    except Exception as exc:                                      # noqa: BLE001
        # A genome that produces an unrunnable strategy is a fact about the search
        # space, not a crash. It scores -inf and the run continues.
        evaluation = Evaluation(vector=vector, fingerprint=key, fitness=UNFIT,
                                n_active=n_active, n_families=n_families,
                                error=str(exc)[:200])
        cache[key] = evaluation
        log.debug("evolve: individual failed: %s", exc)
        return evaluation

    fitness, detail = objective.score(result, n_active=n_active,
                                      n_families=n_families, gate_on=gate_on)
    perf = result.performance
    evaluation = Evaluation(
        vector=np.asarray(vector, dtype=np.float64).copy(), fingerprint=key,
        fitness=fitness, detail=detail, run_id=result.config.get("run_id"),
        cagr=perf.cagr, sharpe=perf.sharpe, max_drawdown=perf.max_drawdown,
        turnover=float(perf.ann_turnover or 0.0), n_active=n_active,
        n_families=n_families)
    cache[key] = evaluation
    return evaluation


def _objective(config: EvolutionConfig, start: str) -> Objective:
    end = config.end or reg.research_end()
    if config.holdout == "exclude":
        end = min(end, reg.research_end())
    folds = None
    if config.aggregate != "whole":
        if config.fold_scheme == "random":
            folds = Folds.random(start, end, n=config.n_folds,
                                 min_years=config.fold_min_years,
                                 max_years=config.fold_max_years,
                                 seed=config.fold_seed)
        elif config.fold_scheme == "contiguous":
            folds = Folds.split(start, end, n=config.n_folds,
                                embargo_days=config.embargo_days)
        else:
            raise ValueError(f"fold_scheme must be 'random' or 'contiguous', "
                             f"got {config.fold_scheme!r}")
    bench = (benchmark_monthly_sharpe(start, end)
             if config.metric == "excess_sharpe" else 0.0)
    return Objective(metric=config.metric, aggregate=config.aggregate, folds=folds,
                     turnover_penalty=config.turnover_penalty,
                     complexity_penalty=config.complexity_penalty,
                     family_penalty=config.family_penalty,
                     gate_penalty=config.gate_penalty,
                     dispersion_weight=config.dispersion_weight,
                     quantile=config.quantile,
                     benchmark_sharpe=bench)


# --------------------------------------------------------------------------
# Population management
# --------------------------------------------------------------------------

def _initial_population(genome, config: EvolutionConfig,
                        rng: np.random.Generator) -> list[np.ndarray]:
    """Random individuals, optionally seeded with the hand-written ideas.

    Seeding is not a shortcut - it is the experiment. If a population that starts from
    12-1 momentum and low volatility cannot evolve anything better than its seeds, that
    is a far more informative result than a random start wandering to a mediocre optimum,
    and it is the direct answer to "can the search improve on what a person wrote".
    """
    pop = []
    if config.seed_with_baselines:
        pop.extend(seed_vectors(genome, config.preset))
    while len(pop) < config.population:
        pop.append(genome.random(rng))
    return pop[:config.population]


_SHAPE_SEED = {"top_k": 50, "max_weight": 0.05, "weighting": "equal",
               "use_regime": "off", "defensive_gross": 0.5, "vol_trigger": 1.5}


def seed_vectors(genome, preset: str) -> list[np.ndarray]:
    """Genomes that reproduce well-known strategies, as starting points.

    For a family preset that is one individual per family, backing that story alone:
    the seeds ARE the hand-written hypotheses, one at a time.
    """
    if preset_kind(preset) == "families":
        out = []
        families = preset_families(preset)
        for fam in families:
            params = {f"f_{f.name}": 0.0 for f in families}
            params[f"f_{fam.name}"] = 1.0
            out.append(genome.encode(params | _SHAPE_SEED))
        return out

    features = preset_features(preset)
    recipes = {
        "momentum": {"mom_12_1": 1.0},
        "residual_momentum": {"resid_mom_12_1": 1.0},
        "low_vol": {"vol_126d": -1.0},
        "lottery_averse": {"max_ret_21d": -1.0, "idio_vol_252d": -1.0},
        "reversal": {"rev_1m": 1.0},
        "trend": {"trend_200d": 1.0, "high_52w_ratio": 1.0},
        "value": {"book_to_market": 1.0, "earnings_yield": 1.0},
        "quality": {"gross_profitability": 1.0, "roe": 1.0, "accruals": -1.0},
    }
    out = []
    for weights in recipes.values():
        if not set(weights) <= set(features):
            continue                                # preset does not carry these
        params = {f"w_{f}": 0.0 for f in features}
        params.update({f"w_{f}": w for f, w in weights.items()})
        out.append(genome.encode(params | _SHAPE_SEED))
    return out


def _next_generation(population: list[np.ndarray], fitness: np.ndarray, genome,
                     config: EvolutionConfig, rng: np.random.Generator,
                     seen: set[str]) -> list[np.ndarray]:
    """Elites, immigrants, and children of tournament winners - in that order."""
    order = np.argsort(-fitness)
    elites = [population[i].copy() for i in order[:max(config.elite, 0)]]

    children: list[np.ndarray] = []
    target = config.population - len(elites) - config.immigrants
    while len(children) < max(target, 0):
        a = ops.tournament_select(fitness, rng, config.tournament_size)
        b = ops.tournament_select(fitness, rng, config.tournament_size)
        children.append(ops.breed(
            [population[a], population[b]], genome, rng,
            crossover_rate=config.crossover_rate,
            mutation_rate=config.mutation_rate,
            mutation_sigma=config.mutation_sigma,
            reset_rate=config.reset_rate))

    # Only the children are deduplicated. Elites must survive untouched or the search can
    # go backwards, and immigrants are random by construction.
    children, forced = ops.deduplicate(children, genome, rng, seen)
    if forced:
        log.debug("evolve: %d child(ren) re-mutated to stay novel", forced)

    immigrants = [genome.random(rng) for _ in range(config.immigrants)]
    return (elites + children + immigrants)[:config.population]


def _generation_row(generation: int, scored: list[Evaluation], fitness: np.ndarray,
                    population: list[np.ndarray], genome) -> dict:
    ok = np.isfinite(fitness)
    best = scored[int(np.argmax(fitness))] if ok.any() else None
    return {
        "generation": generation,
        "evaluated": len(scored),
        "scorable": int(ok.sum()),
        "best_fitness": float(fitness[ok].max()) if ok.any() else float("nan"),
        "mean_fitness": float(fitness[ok].mean()) if ok.any() else float("nan"),
        "median_fitness": float(np.median(fitness[ok])) if ok.any() else float("nan"),
        "best_sharpe": float(best.sharpe) if best else float("nan"),
        "best_cagr": float(best.cagr) if best else float("nan"),
        "best_n_active": int(best.n_active) if best else 0,
        "best_n_families": int(best.n_families) if best else 0,
        "diversity": ops.diversity(population, genome),
        "best_fingerprint": best.fingerprint if best else "",
    }


def _log_generation(row: dict) -> None:
    log.info("seed %d gen %2d  best %.4f  mean %.4f  (Sharpe %.2f, CAGR %.2f%%, "
             "%d features, %d families)  diversity %.3f  %d/%d scorable",
             row.get("seed", 0), row["generation"], row["best_fitness"],
             row["mean_fitness"], row["best_sharpe"], row["best_cagr"] * 100,
             row["best_n_active"], row["best_n_families"], row["diversity"],
             row["scorable"], row["evaluated"])


def _checkpoint(config: EvolutionConfig, seed: int, generation: int,
                scored: list[Evaluation], row: dict) -> None:
    """Append this generation to the search's own log. Never rewrites, only appends."""
    EVOLVE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVOLVE_DIR / f"{_slug(config.study)}.jsonl"
    payload = {
        "study": config.study, "seed": int(seed), "generation": generation,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config.as_dict(), "stats": row,
        "population": [{"vector": [round(float(x), 6) for x in e.vector],
                        "fitness": None if not np.isfinite(e.fitness) else
                                   round(float(e.fitness), 6),
                        "sharpe": _num(e.sharpe), "cagr": _num(e.cagr),
                        "run_id": e.run_id, "n_active": e.n_active,
                        "n_families": e.n_families, "error": e.error}
                       for e in scored],
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


# --------------------------------------------------------------------------
# The ensemble (ADR-050)
# --------------------------------------------------------------------------

def _ensemble_record(config: EvolutionConfig, seeds: list[int],
                     members: list[Evaluation], champion_eval: Evaluation,
                     objective: Objective, panel, ranked, start: str) -> dict:
    """Build the ensemble of `members`, score it once, and describe it for the record.

    The ensemble is one more backtest, logged into the study as one more trial: it is a
    hypothesis the search produced, and the deflated Sharpe of anything in the study
    should count it. Its fitness is the objective's ROBUST SCORE with no complexity
    penalties - the penalties are selection pressure on individuals, not a metric.
    """
    strategies = [from_vector(e.vector, config.preset, pre_ranked=True) for e in members]
    strategy = EvolvedEnsemble(strategies, study=config.study)
    strategy.name = f"{config.study}-ensemble"
    result = run_backtest(
        strategy, panel=panel, features=ranked, start=start, end=config.end,
        costs=config.costs, initial_capital=config.initial_capital,
        liquidity_floor=config.liquidity_floor, holdout=config.holdout,
        seed=config.seed, study=config.study, log_run=config.log_runs,
        log_curve=False, record_trades=False, benchmark=None,
        notes=f"ga:{config.study}:ensemble of {len(members)}")
    fitness, detail = objective.score(result)
    perf = result.performance
    genome = alpha_genome(config.preset)
    return {
        "study": config.study, "preset": config.preset,
        "size": len(members), "requested_size": int(config.ensemble_size),
        "seeds": [int(s) for s in seeds],
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config.as_dict(), "objective": objective.describe(),
        "members": [{"vector": [round(float(x), 6) for x in e.vector],
                     "fingerprint": e.fingerprint, "fitness": _num(e.fitness),
                     "sharpe": _num(e.sharpe), "cagr": _num(e.cagr),
                     "n_active": int(e.n_active), "n_families": int(e.n_families),
                     "families": active_families(genome, e.vector)}
                    for e in members],
        # `base` is the champion's robust score BEFORE its penalties - the number the
        # ensemble's own score is comparable with, since the ensemble is not charged.
        "champion": {"vector": [round(float(x), 6) for x in champion_eval.vector],
                     "fingerprint": champion_eval.fingerprint,
                     "fitness": _num(champion_eval.fitness),
                     "base": _num(champion_eval.detail.get("base")),
                     "penalty": _num(champion_eval.detail.get("penalty")),
                     "sharpe": _num(champion_eval.sharpe),
                     "cagr": _num(champion_eval.cagr)},
        "evaluation": {"fitness": _num(fitness), "sharpe": _num(perf.sharpe),
                       "cagr": _num(perf.cagr), "max_drawdown": _num(perf.max_drawdown),
                       "turnover": _num(perf.ann_turnover),
                       "avg_positions": _num(perf.avg_positions),
                       "run_id": result.config.get("run_id"), "costs": config.costs,
                       "detail": detail},
        "family_usage": strategy.family_usage,
        "feature_usage": strategy.feature_usage,
        "construction": vars(strategy.construction).copy(),
        "prose": strategy.explain(),
    }


def write_ensemble(record: dict) -> None:
    EVOLVE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVOLVE_DIR / f"{_slug(record['study'])}.ensemble.json"
    path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    log.info("evolve: ensemble of %d written to %s", record["size"], path)


def load_ensemble(study: str) -> dict | None:
    """The stored ensemble of a search, or None if the search never built one."""
    path = EVOLVE_DIR / f"{_slug(study)}.ensemble.json"
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("evolve: unreadable ensemble file %s", path)
        return None
    return record if record.get("members") else None


def ensemble_strategy(study: str, record: dict | None = None,
                      pre_ranked: bool = False) -> EvolvedEnsemble:
    """The stored ensemble of a search as a strategy object, named `<study>-ensemble`."""
    record = record or load_ensemble(study)
    if not record:
        raise KeyError(f"search {study!r} has no stored ensemble; run "
                       f"`sp500lab evolve ensemble {study} --rebuild`")
    preset = record.get("preset") or study_preset(study)
    members = [from_vector(np.asarray(m["vector"], dtype=float), preset,
                           pre_ranked=pre_ranked) for m in record["members"]]
    strategy = EvolvedEnsemble(members, study=study)
    strategy.name = f"{study}-ensemble"
    return strategy


def build_ensemble(study: str, size: int | None = None, evaluate: bool = True,
                   write: bool = True) -> dict:
    """Rebuild a search's ensemble from its checkpoint, across every seed it ran.

    For a search that finished before ensembles existed, or that was interrupted before
    building one. The members are the top `size` distinct individuals ever scored, by
    the fitness recorded in the checkpoint - the same ranking `evolve()` uses.
    """
    config_dict = study_config(study)
    if not config_dict:
        raise KeyError(f"no recorded search named {study!r}")
    known = {f.name for f in EvolutionConfig.__dataclass_fields__.values()}
    config = EvolutionConfig(**{k: v for k, v in config_dict.items() if k in known})
    if size is not None:
        config.ensemble_size = int(size)
    if config.ensemble_size <= 0:
        raise ValueError("an ensemble needs a positive size")

    genome = alpha_genome(config.preset)
    individuals = load_individuals(study)
    by_fp: dict[str, Evaluation] = {}
    seeds: list[int] = []
    for p in individuals:
        if p.get("fitness") is None:
            continue
        if p.get("seed") is not None and int(p["seed"]) not in seeds:
            seeds.append(int(p["seed"]))
        v = np.asarray(p["vector"], dtype=float)
        fp = genome.fingerprint(v)
        e = Evaluation(vector=v, fingerprint=fp, fitness=float(p["fitness"]),
                       run_id=p.get("run_id"),
                       cagr=float(p.get("cagr") if p.get("cagr") is not None
                                  else float("nan")),
                       sharpe=float(p.get("sharpe") if p.get("sharpe") is not None
                                    else float("nan")),
                       n_active=int(p.get("n_active") or 0),
                       n_families=int(p.get("n_families") or 0))
        if fp not in by_fp or e.fitness > by_fp[fp].fitness:
            by_fp[fp] = e
    ranking = sorted(by_fp.values(), key=lambda e: e.fitness, reverse=True)
    if len(ranking) < 2:
        raise ValueError(f"search {study!r} has fewer than two scorable individuals")
    members = ranking[:config.ensemble_size]

    start = max(config.start, PRESET_MIN_DATE[config.preset] or config.start)
    objective = _objective(config, start)
    if not evaluate:
        strategies = [from_vector(e.vector, config.preset) for e in members]
        strategy = EvolvedEnsemble(strategies, study=study)
        record = {"study": study, "preset": config.preset, "size": len(members),
                  "requested_size": config.ensemble_size, "seeds": seeds,
                  "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "config": config.as_dict(), "objective": objective.describe(),
                  "members": [{"vector": [float(x) for x in e.vector],
                               "fingerprint": e.fingerprint, "fitness": _num(e.fitness),
                               "sharpe": _num(e.sharpe), "cagr": _num(e.cagr),
                               "n_active": e.n_active, "n_families": e.n_families,
                               "families": active_families(genome, e.vector)}
                              for e in members],
                  "champion": {"vector": [float(x) for x in ranking[0].vector],
                               "fingerprint": ranking[0].fingerprint,
                               "fitness": _num(ranking[0].fitness),
                               "sharpe": _num(ranking[0].sharpe),
                               "cagr": _num(ranking[0].cagr)},
                  "evaluation": None,
                  "family_usage": strategy.family_usage,
                  "feature_usage": strategy.feature_usage,
                  "construction": vars(strategy.construction).copy(),
                  "prose": strategy.explain()}
    else:
        panel, ranked = _ranked_inputs(config.preset, config.liquidity_floor)
        with reg.study(study):
            # The checkpoint keeps the champion's penalised fitness but not its robust
            # score, so it is scored once more here - the same fingerprint, so the same
            # trial - to give the ensemble's score something comparable to sit beside.
            champion_eval = _evaluate(ranking[0].vector, genome, config, panel, ranked,
                                      objective, {}, start)
            if not champion_eval.ok:
                champion_eval = ranking[0]
            record = _ensemble_record(config, seeds, members, champion_eval, objective,
                                      panel, ranked, start)
    if write:
        write_ensemble(record)
    return record


# --------------------------------------------------------------------------
# Reading a search back
# --------------------------------------------------------------------------

def _records(study: str) -> list[dict]:
    path = EVOLVE_DIR / f"{_slug(study)}.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _seed_of(record: dict) -> int:
    if record.get("seed") is not None:
        return int(record["seed"])
    return int((record.get("config") or {}).get("seed", 0))


def load_history(study: str) -> pd.DataFrame:
    """Per-generation statistics of a past search, without re-running it.

    One row per (seed, generation); the `seed` column is what separates the seeds of a
    pooled search. Searches recorded before seeds were stored carry their config's.
    """
    rows = []
    for rec in _records(study):
        try:
            row = dict(rec["stats"])
        except (KeyError, TypeError):
            continue
        row.setdefault("seed", _seed_of(rec))
        rows.append(row)
    return pd.DataFrame(rows)


def load_population(study: str, generation: int | None = None,
                    seed: int | None = None) -> list[dict]:
    """The individuals of one generation - the last one recorded by default.

    `seed` narrows a pooled search to one of its seeds; without it, "the last
    generation" is the last one written, which is the final generation of the last seed.
    """
    best = None
    for rec in _records(study):
        if seed is not None and _seed_of(rec) != seed:
            continue
        if generation is None or rec.get("generation") == generation:
            best = rec
    return best["population"] if best else []


def load_individuals(study: str) -> list[dict]:
    """Every individual of every generation of every seed, with `seed` and `generation`."""
    out = []
    for rec in _records(study):
        seed = _seed_of(rec)
        for p in rec.get("population", []):
            out.append(dict(p) | {"seed": seed, "generation": rec.get("generation")})
    return out


def champion(study: str) -> dict | None:
    """The single best individual a search ever scored, across every seed.

    With elitism the best individual of the final generation is the best individual of
    the whole run, so for a single-seed search this is exactly the old "best of the
    final generation"; for a pooled search it is the best across seeds.
    """
    scored = [p for p in load_individuals(study) if p.get("fitness") is not None]
    if not scored:
        return None
    return max(scored, key=lambda p: p["fitness"])


def study_config(study: str) -> dict:
    """The configuration a recorded search ran with, from its first checkpoint."""
    for rec in _records(study):
        return dict(rec.get("config", {}))
    return {}


def study_preset(study: str) -> str:
    """Which preset a recorded search used. Defaults to 'price'."""
    return str(study_config(study).get("preset", "price") or "price")


def winners() -> list[dict]:
    """The deliverable of every search on disk, as a ready-to-run strategy.

    Discovered rather than configured: a search that ran left a checkpoint, and what it
    produced is a candidate like any other. A search with a stored ensemble hands over
    the ensemble, named `<study>-ensemble` (ADR-050); one without hands over its
    champion as `<study>-best`, which is what the three searches that predate ensembles
    do.

    Lives here rather than in a caller because two of them need it - the report set and
    the forward-test suite - and a genome decoded two slightly different ways would be
    two different strategies wearing one name.

    Each entry: `name`, `strategy`, `study`, `preset`, `kind` (`ensemble` or
    `champion`), `vector` (the champion's, for the decoded description), `n_population`,
    `n_individuals`, `n_members`, `seeds`.
    """
    if not EVOLVE_DIR.exists():
        return []
    out = []
    for path in sorted(EVOLVE_DIR.glob("*.jsonl")):
        study = path.stem
        best = champion(study)
        if best is None:
            continue
        individuals = load_individuals(study)
        seeds = sorted({int(p["seed"]) for p in individuals if p.get("seed") is not None})
        preset = study_preset(study)
        vector = np.array(best["vector"], dtype=float)
        base = {"study": study, "preset": preset, "vector": vector,
                "n_population": len(load_population(study)),
                "n_individuals": len({tuple(p["vector"]) for p in individuals
                                      if p.get("fitness") is not None}),
                "seeds": seeds}

        record = load_ensemble(study)
        if record:
            try:
                strategy = ensemble_strategy(study, record)
            except Exception as exc:                              # noqa: BLE001
                log.warning("could not decode the ensemble of %s (%s); falling back "
                            "to its champion", study, exc)
            else:
                out.append(base | {"name": strategy.name, "strategy": strategy,
                                   "kind": "ensemble",
                                   "n_members": len(record["members"])})
                continue
        try:
            strategy = from_vector(vector, preset)
        except Exception:                                         # noqa: BLE001
            log.warning("could not decode the winner of %s; skipping", study)
            continue
        strategy.name = f"{study}-best"
        out.append(base | {"name": strategy.name, "strategy": strategy,
                           "kind": "champion", "n_members": 1})
    return out


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, 6)


def _slug(text: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "-" for c in str(text))
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or "evolve"
