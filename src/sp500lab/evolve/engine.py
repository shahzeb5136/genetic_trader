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

The four defences, all of which are load-bearing
-------------------------------------------------
A GA overfits more aggressively than almost anything else, because it is *explicitly*
maximising the number you report. Every one of these is here for that reason:

1. **A small, bounded, readable search space.** Weighted sums of ranked features, not
   evolved expression trees. See strategies/genome.py.
2. **Every individual is logged as a trial**, not just the winners. The deflated Sharpe
   needs the trial count and the spread of trial Sharpes, and neither can be recovered
   after the fact (ADR-026). This is on by default and turning it off is a decision.
3. **Fitness is fold-consistency, not full-sample Sharpe.** An individual that made all
   its money in one 18-month stretch scores badly however good its headline looks. See
   fitness.py.
4. **The holdout is untouched.** The search runs on the research window and stops the day
   before 2022-01-01. Testing the winner there is a separate, deliberate, recorded act
   (ADR-025), and doing it more than once destroys the only out-of-sample evidence there
   is.

Speed, and where it comes from
-------------------------------
About 0.15 seconds per evaluation, so a 60 x 25 search is roughly 3 minutes of fitness
evaluation. Three things buy that and a change to any of them will cost it: the price
panel is built once and memoised, the feature ranks are precomputed once for the whole
population (features/ranked.py), and each individual is a single backtest whose folds are
sliced from the resulting curve rather than re-run.

The evaluation cache is the fourth. Two genomes that differ only inside the dead zone
produce the same portfolio, and the cache keys on that behavioural fingerprint - so a
converged population costs nothing to re-evaluate, and the registry's trial count stays
equal to the number of distinct hypotheses actually tested.

Resumability
------------
Every generation is appended to `data/experiments/evolve/<study>.jsonl` with the full
population. A killed run can be resumed from its last generation, and - more usefully -
the history of a finished run can be re-read without re-running it. Interrupting with
Ctrl-C returns the result computed so far rather than losing it.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..backtest import registry as reg
from ..backtest.engine import run_backtest
from ..backtest.panel import build_panel
from ..paths import EXPERIMENTS_DIR
from ..strategies.evolvable import from_vector
from ..strategies.genome import (PRESET_MIN_DATE, PRESETS, REGIME_FEATURES,
                                 active_features, alpha_genome, describe_genome)
from . import operators as ops
from .fitness import UNFIT, Evaluation, Folds, Objective, benchmark_monthly_sharpe

log = logging.getLogger(__name__)

EVOLVE_DIR = EXPERIMENTS_DIR / "evolve"

#: Extra raw columns the pre-ranked panel must carry: the regime gate reads macro levels,
#: and the defensive sleeve ranks volatility.
RANK_PASSTHROUGH = REGIME_FEATURES


@dataclass
class EvolutionConfig:
    """Everything that determines a search. Stored verbatim with its result.

    The defaults are deliberately modest. A population of 60 over 25 generations is 1,500
    evaluations, which is enough to find structure if there is any and small enough that
    the deflated Sharpe still has a chance of clearing 0.95. Ten thousand individuals
    sounds more serious and mostly buys a larger correction to apply to the winner.
    """

    study: str = "ga"
    preset: str = "price"
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
    seed_with_baselines: bool = True

    start: str = "2007-04-01"
    end: str | None = None
    costs: str = "realistic"
    liquidity_floor: float = 0.0
    holdout: str = "exclude"
    initial_capital: float = 100_000.0

    metric: str = "sharpe_monthly"
    aggregate: str = "mean_minus_std"
    n_folds: int = 4
    embargo_days: int = 31
    turnover_penalty: float = 0.0
    complexity_penalty: float = 0.0
    dispersion_weight: float = 0.5

    log_runs: bool = True
    checkpoint: bool = True
    max_evaluations: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


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

    def summary(self) -> str:
        c = self.config
        lines = [
            "=" * 76,
            f"EVOLUTION  study={c.study}  preset={c.preset}",
            "=" * 76,
            f"  {self.n_evaluations} evaluations, {self.n_distinct} distinct "
            f"individuals, {len(self.history)} generations, {self.elapsed:.0f}s",
            f"  objective: {self.objective.get('metric')} "
            f"aggregated as {self.objective.get('aggregate')} over "
            f"{self.objective.get('folds')}",
        ]
        if self.interrupted:
            lines.append("  !! interrupted; this is the state at the last completed "
                         "generation")
        if self.best is None:
            lines.append("  no individual scored a finite fitness")
            return "\n".join(lines)

        b = self.best
        lines += [
            "",
            f"  BEST  fitness {b.fitness:.4f}   CAGR {b.cagr * 100:.2f}%   "
            f"Sharpe {b.sharpe:.2f}   maxDD {b.max_drawdown * 100:.1f}%   "
            f"turnover {b.turnover * 100:.0f}%",
            "",
            describe_genome(alpha_genome(c.preset), b.vector),
            "",
            "  This number has not been corrected for the search that produced it.",
            f"  Run `sp500lab experiments deflate {c.study}` before believing it.",
        ]
        return "\n".join(lines)

    def leaderboard(self, top: int = 10) -> pd.DataFrame:
        rows = [e.as_row() for e in self.hall_of_fame[:top]]
        return pd.DataFrame(rows)

    def best_strategy(self, pre_ranked: bool = False):
        """The winner as a strategy object, ready to re-run or export trades from."""
        if self.best is None:
            raise ValueError("this search produced no scorable individual")
        return from_vector(self.best.vector, self.config.preset, pre_ranked=pre_ranked)


# --------------------------------------------------------------------------
# The search
# --------------------------------------------------------------------------

def evolve(config: EvolutionConfig) -> EvolutionResult:
    """Run a search. Every individual is backtested, scored and logged."""
    from ..features import build_features
    from ..features.ranked import rank_panel

    t0 = time.perf_counter()
    genome = alpha_genome(config.preset)
    rng = np.random.default_rng(config.seed)

    panel = build_panel()
    features = build_features(panel=panel)
    ranked = rank_panel(features, panel, PRESETS[config.preset],
                        keep_raw=RANK_PASSTHROUGH + ("vol_126d",),
                        liquidity_floor=config.liquidity_floor)

    start = max(config.start, PRESET_MIN_DATE[config.preset] or config.start)
    objective = _objective(config, start)
    log.info("evolve: %s, %d x %d over %s, folds %s", genome.name, config.population,
             config.generations, start, objective.folds.describe()
             if objective.folds else "none")

    cache: dict[str, Evaluation] = {}
    seen: set[str] = set()
    population = _initial_population(genome, config, rng)
    history: list[dict] = []
    interrupted = False

    with reg.study(config.study):
        try:
            for generation in range(config.generations):
                if (config.max_evaluations
                        and len(cache) >= config.max_evaluations):
                    log.info("evolve: evaluation budget reached")
                    break

                scored = [_evaluate(v, genome, config, panel, ranked, objective,
                                    cache, start)
                          for v in population]
                fitness = np.array([e.fitness for e in scored], dtype=np.float64)
                row = _generation_row(generation, scored, fitness, population, genome)
                history.append(row)
                _log_generation(row)
                if config.checkpoint:
                    _checkpoint(config, generation, scored, row)

                if generation + 1 >= config.generations:
                    break
                population = _next_generation(population, fitness, genome, config,
                                              rng, seen)
        except KeyboardInterrupt:
            interrupted = True
            log.warning("evolve: interrupted - returning the state so far")

    ranking = sorted((e for e in cache.values() if e.ok),
                     key=lambda e: e.fitness, reverse=True)
    return EvolutionResult(
        config=config, genome_name=genome.name,
        best=ranking[0] if ranking else None,
        hall_of_fame=ranking[:25],
        history=pd.DataFrame(history),
        n_evaluations=sum(int(r["evaluated"]) for r in history),
        n_distinct=len(cache),
        elapsed=time.perf_counter() - t0,
        interrupted=interrupted,
        objective=objective.describe(),
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _evaluate(vector: np.ndarray, genome, config: EvolutionConfig, panel, ranked,
              objective: Objective, cache: dict, start: str) -> Evaluation:
    """One individual: backtest it, score it, remember it.

    The cache key is the BEHAVIOURAL fingerprint, so two genomes that differ only inside
    the dead zone are one evaluation and one trial. That keeps the registry's trial count
    equal to the number of distinct hypotheses tested, which is what the deflated Sharpe
    needs it to mean.
    """
    key = genome.fingerprint(vector)
    hit = cache.get(key)
    if hit is not None:
        return hit

    strategy = from_vector(vector, config.preset, pre_ranked=True)
    n_active = len(active_features(genome, vector))
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
                                n_active=n_active, error=str(exc)[:200])
        cache[key] = evaluation
        log.debug("evolve: individual failed: %s", exc)
        return evaluation

    fitness, detail = objective.score(result, n_active=n_active)
    perf = result.performance
    evaluation = Evaluation(
        vector=np.asarray(vector, dtype=np.float64).copy(), fingerprint=key,
        fitness=fitness, detail=detail, run_id=result.config.get("run_id"),
        cagr=perf.cagr, sharpe=perf.sharpe, max_drawdown=perf.max_drawdown,
        turnover=float(perf.ann_turnover or 0.0), n_active=n_active)
    cache[key] = evaluation
    return evaluation


def _objective(config: EvolutionConfig, start: str) -> Objective:
    end = config.end or reg.research_end()
    if config.holdout == "exclude":
        end = min(end, reg.research_end())
    folds = (Folds.split(start, end, n=config.n_folds,
                         embargo_days=config.embargo_days)
             if config.aggregate != "whole" else None)
    bench = (benchmark_monthly_sharpe(start, end)
             if config.metric == "excess_sharpe" else 0.0)
    return Objective(metric=config.metric, aggregate=config.aggregate, folds=folds,
                     turnover_penalty=config.turnover_penalty,
                     complexity_penalty=config.complexity_penalty,
                     dispersion_weight=config.dispersion_weight,
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


def seed_vectors(genome, preset: str) -> list[np.ndarray]:
    """Genomes that reproduce well-known strategies, as starting points."""
    features = PRESETS[preset]
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
        params |= {"top_k": 50, "max_weight": 0.05, "weighting": "equal",
                   "use_regime": "off", "defensive_gross": 0.5, "vol_trigger": 1.5}
        out.append(genome.encode(params))
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
        "diversity": ops.diversity(population, genome),
        "best_fingerprint": best.fingerprint if best else "",
    }


def _log_generation(row: dict) -> None:
    log.info("gen %2d  best %.4f  mean %.4f  (Sharpe %.2f, CAGR %.2f%%, %d features)  "
             "diversity %.3f  %d/%d scorable",
             row["generation"], row["best_fitness"], row["mean_fitness"],
             row["best_sharpe"], row["best_cagr"] * 100, row["best_n_active"],
             row["diversity"], row["scorable"], row["evaluated"])


def _checkpoint(config: EvolutionConfig, generation: int, scored: list[Evaluation],
                row: dict) -> None:
    """Append this generation to the search's own log. Never rewrites, only appends."""
    EVOLVE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVOLVE_DIR / f"{_slug(config.study)}.jsonl"
    payload = {
        "study": config.study, "generation": generation,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config.as_dict(), "stats": row,
        "population": [{"vector": [round(float(x), 6) for x in e.vector],
                        "fitness": None if not np.isfinite(e.fitness) else
                                   round(float(e.fitness), 6),
                        "sharpe": _num(e.sharpe), "cagr": _num(e.cagr),
                        "run_id": e.run_id, "n_active": e.n_active,
                        "error": e.error}
                       for e in scored],
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


def load_history(study: str) -> pd.DataFrame:
    """Per-generation statistics of a past search, without re-running it."""
    path = EVOLVE_DIR / f"{_slug(study)}.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line)["stats"])
        except (json.JSONDecodeError, KeyError):
            continue
    return pd.DataFrame(rows)


def load_population(study: str, generation: int | None = None) -> list[dict]:
    """The individuals of one generation - the last one by default."""
    path = EVOLVE_DIR / f"{_slug(study)}.jsonl"
    if not path.exists():
        return []
    best = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if generation is None or rec.get("generation") == generation:
            best = rec
    return best["population"] if best else []


def study_preset(study: str) -> str:
    """Which feature preset a recorded search used. Defaults to 'price'."""
    path = EVOLVE_DIR / f"{_slug(study)}.jsonl"
    if not path.exists():
        return "price"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                return json.loads(line).get("config", {}).get("preset", "price")
            except json.JSONDecodeError:
                break
    return "price"


def winners() -> list[dict]:
    """The best individual of every search on disk, as a ready-to-run strategy.

    Discovered rather than configured: a search that ran left a checkpoint, and its
    winner is a candidate like any other. The alternative is registering evolved genomes
    as named strategies, which would fill the strategy registry with artifacts of
    particular runs.

    Lives here rather than in a caller because two of them now need it - the report set
    and the forward-test suite - and a genome decoded two slightly different ways would
    be two different strategies wearing one name.

    Each entry: `name`, `strategy`, `study`, `preset`, `vector`, `n_population`.
    """
    if not EVOLVE_DIR.exists():
        return []
    out = []
    for path in sorted(EVOLVE_DIR.glob("*.jsonl")):
        study = path.stem
        population = [p for p in load_population(study)
                      if p.get("fitness") is not None]
        if not population:
            continue
        winner = max(population, key=lambda p: p["fitness"])
        vector = np.array(winner["vector"], dtype=float)
        preset = study_preset(study)
        try:
            strategy = from_vector(vector, preset)
        except Exception:                                         # noqa: BLE001
            log.warning("could not decode the winner of %s; skipping", study)
            continue
        strategy.name = f"{study}-best"
        out.append({"name": strategy.name, "strategy": strategy, "study": study,
                    "preset": preset, "vector": vector,
                    "n_population": len(population)})
    return out


def _num(x) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def _slug(text: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "-" for c in str(text))
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or "evolve"
