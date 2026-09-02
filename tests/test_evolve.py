"""The genetic algorithm: does the search space hold, and does the search stay honest?

None of these tests assert that evolution finds a good strategy. That is not a property
code can have - it is a property of the data, and the deflated Sharpe is how it gets
decided. What is testable is everything the result depends on:

  * a mutated genome is still inside its bounds, whatever the mutation did
  * two vectors that produce the same portfolio count as ONE trial, because the trial
    count is the input to the deflated Sharpe and double-counting over-deflates
  * the same seed produces the same search, or a result cannot be audited
  * an individual with no opinion holds cash rather than falling through to the
    tie-break and collecting the survivors (ADR-024)
  * fitness rejects the degenerate cases instead of ranking them
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp500lab.evolve import operators as ops
from sp500lab.evolve.fitness import UNFIT, Folds, Objective
from sp500lab.strategies.genome import (DEAD_ZONE, PRESETS, active_features,
                                        alpha_genome, describe_genome)


@pytest.fixture
def genome():
    return alpha_genome("price")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --------------------------------------------------------------------------
# The search space
# --------------------------------------------------------------------------

def test_a_random_genome_is_inside_its_bounds(genome, rng):
    for _ in range(50):
        v = genome.random(rng)
        assert (v >= genome.lows - 1e-12).all()
        assert (v <= genome.highs + 1e-12).all()


def test_decode_encode_round_trips(genome, rng):
    v = genome.random(rng)
    assert genome.encode(genome.decode(v)) == pytest.approx(v)


def test_categorical_genes_decode_to_labels(genome, rng):
    p = genome.decode(genome.random(rng))
    assert p["weighting"] in ("equal", "score_rank", "inverse_vol")
    assert p["use_regime"] in ("off", "on")
    assert isinstance(p["top_k"], int)


def test_clipping_never_raises_on_an_out_of_bounds_vector(genome):
    """A mutation that overshoots must produce a boundary individual, not an exception."""
    wild = np.full(len(genome), 1e6)
    clipped = genome.clip(wild)
    assert (clipped == genome.highs).all()


def test_fingerprint_ignores_weights_inside_the_dead_zone(genome):
    """Two genomes that behave identically are ONE trial. The deflated Sharpe needs that."""
    a = np.zeros(len(genome))
    b = np.zeros(len(genome))
    b[0] = DEAD_ZONE / 2
    assert genome.fingerprint(a) == genome.fingerprint(b)


def test_fingerprint_separates_genomes_that_behave_differently(genome):
    a = np.zeros(len(genome))
    b = np.zeros(len(genome))
    b[0] = 0.9
    assert genome.fingerprint(a) != genome.fingerprint(b)


def test_active_features_counts_only_what_is_outside_the_dead_zone(genome):
    v = np.zeros(len(genome))
    v[0] = 0.5
    v[1] = DEAD_ZONE / 2
    assert active_features(genome, v) == [PRESETS["price"][0]]


def test_describe_genome_reads_as_sentences(genome):
    v = np.zeros(len(genome))
    v[0] = 0.8
    text = describe_genome(genome, v)
    assert PRESETS["price"][0] in text
    assert "high is good" in text
    assert "Holds the top" in text


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------

def test_mutation_stays_inside_the_box(genome, rng):
    v = genome.random(rng)
    for _ in range(200):
        v = ops.mutate(v, genome, rng, rate=1.0, sigma=2.0, reset_rate=0.5)
        assert (v >= genome.lows - 1e-12).all() and (v <= genome.highs + 1e-12).all()


def test_crossover_children_lie_near_their_parents(genome, rng):
    a, b = genome.random(rng), genome.random(rng)
    child = ops.blend_crossover(a, b, rng, alpha=0.0)
    assert (child >= np.minimum(a, b) - 1e-9).all()
    assert (child <= np.maximum(a, b) + 1e-9).all()


def test_uniform_crossover_takes_every_gene_from_a_parent(genome, rng):
    a, b = genome.random(rng), genome.random(rng)
    child = ops.uniform_crossover(a, b, rng)
    assert ((child == a) | (child == b)).all()


def test_tournament_prefers_higher_fitness():
    """Statistical, not absolute: a tournament is a sample, but a heavily biased one."""
    rng = np.random.default_rng(1)
    fitness = np.arange(20, dtype=np.float64)
    picks = [ops.tournament_select(fitness, rng, size=5) for _ in range(500)]
    assert np.mean(picks) > 14


def test_a_fit_individual_wins_every_tournament_it_is_drawn_into():
    """An unfit individual can still be selected - when the sample contained only unfit
    ones. That is correct: a tournament ranks what it drew. What must never happen is a
    -inf beating a finite fitness that was in the same sample."""
    rng = np.random.default_rng(2)
    fitness = np.array([UNFIT, UNFIT, 1.0])
    picks = [ops.tournament_select(fitness, rng, size=3) for _ in range(500)]
    share = np.mean(np.array(picks) == 2)
    # 1 - (2/3)^3 = 0.70 is the probability the fit one is drawn at all.
    assert 0.62 < share < 0.78, share


def test_deduplicate_forces_novelty(genome, rng):
    v = genome.random(rng)
    seen = {genome.fingerprint(v)}
    out, forced = ops.deduplicate([v.copy(), v.copy()], genome, rng, seen)
    assert forced >= 2
    assert len({genome.fingerprint(x) for x in out}) == 2


def test_diversity_is_zero_for_a_clone_army(genome, rng):
    v = genome.random(rng)
    assert ops.diversity([v, v.copy(), v.copy()], genome) == pytest.approx(0.0)
    assert ops.diversity([genome.random(rng) for _ in range(20)], genome) > 0.1


def test_operators_are_reproducible(genome):
    """A search that cannot be replayed cannot be audited."""
    a = ops.mutate(np.zeros(len(genome)), genome, np.random.default_rng(5))
    b = ops.mutate(np.zeros(len(genome)), genome, np.random.default_rng(5))
    assert a == pytest.approx(b)


# --------------------------------------------------------------------------
# Folds and the objective
# --------------------------------------------------------------------------

def test_folds_are_contiguous_and_separated_by_the_embargo():
    folds = Folds.split("2010-01-01", "2020-01-01", n=4, embargo_days=31)
    assert len(folds) == 4
    for (_, end), (start, _) in zip(folds.bounds, folds.bounds[1:]):
        gap = (pd.Timestamp(start) - pd.Timestamp(end)).days
        assert gap == 31, "a position held across the boundary would span both folds"


def test_folds_slice_an_equity_curve():
    dates = pd.date_range("2010-01-01", "2020-01-01", freq="B").strftime("%Y-%m-%d")
    equity = pd.Series(np.linspace(1.0, 2.0, len(dates)), index=dates)
    pieces = Folds.split("2010-01-01", "2020-01-01", n=4).slices(equity)
    assert len(pieces) == 4
    assert sum(len(p) for p in pieces) < len(equity), "the embargo must remove sessions"


class _FakeResult:
    """The smallest thing Objective.score reads. Beats building a real backtest."""

    def __init__(self, months=120, sharpe=1.0, turnover=1.0, names=50, ruined=False):
        idx = pd.date_range("2010-01-31", periods=months * 21, freq="B")
        rng = np.random.default_rng(0)
        step = rng.normal(0.0004, 0.01, size=len(idx))
        self.equity = pd.Series(100_000 * np.cumprod(1 + step),
                                index=idx.strftime("%Y-%m-%d"))
        self.diagnostics = {"!! ruined": "yes"} if ruined else {}
        self.performance = type("P", (), {
            "sharpe": sharpe, "calmar": 1.0, "information_ratio": 0.5,
            "ann_turnover": turnover, "avg_positions": names})()


def test_objective_rejects_a_run_that_is_too_short():
    obj = Objective(aggregate="whole", min_months=36)
    fitness, detail = obj.score(_FakeResult(months=12))
    assert fitness == UNFIT
    assert "months" in detail["reject"]


def test_objective_rejects_ruin():
    obj = Objective(aggregate="whole")
    fitness, detail = obj.score(_FakeResult(ruined=True))
    assert fitness == UNFIT
    assert "zero NAV" in detail["reject"]


def test_objective_rejects_a_portfolio_of_almost_nothing():
    """Concentration is how a long-only GA manufactures leverage."""
    obj = Objective(aggregate="whole", min_names=5)
    fitness, detail = obj.score(_FakeResult(names=2))
    assert fitness == UNFIT
    assert "names" in detail["reject"]


def test_turnover_and_complexity_penalties_reduce_fitness():
    plain = Objective(aggregate="whole")
    charged = Objective(aggregate="whole", turnover_penalty=0.1,
                        complexity_penalty=0.01)
    result = _FakeResult(turnover=4.0)
    base, _ = plain.score(result, n_active=10)
    reduced, detail = charged.score(result, n_active=10)
    assert reduced == pytest.approx(base - 0.1 * 4.0 - 0.01 * 10, abs=1e-9)
    assert detail["penalty"] == pytest.approx(0.5, abs=1e-9)


def test_fold_aggregation_punishes_inconsistency():
    """Two runs with the same full-sample statistic must not score the same."""
    folds = Folds.split("2010-01-31", "2019-12-31", n=4)
    obj = Objective(aggregate="mean_minus_std", folds=folds, dispersion_weight=1.0)
    fitness, detail = obj.score(_FakeResult())
    assert np.isfinite(fitness)
    assert detail["fold_std"] > 0
    # The detail fields are rounded for the log; the fitness is not.
    assert fitness == pytest.approx(detail["fold_mean"] - detail["fold_std"], abs=1e-5)


def test_objective_describes_itself_for_the_manifest():
    obj = Objective(aggregate="whole", metric="sharpe")
    d = obj.describe()
    assert d["metric"] == "sharpe" and d["folds"] is None


# --------------------------------------------------------------------------
# The strategy a genome becomes
# --------------------------------------------------------------------------

def test_an_individual_with_no_opinion_holds_cash(genome):
    """The ADR-024 failure: a zero-signal strategy must not collect the survivors."""
    from sp500lab.strategies.evolvable import from_vector

    strategy = from_vector(np.zeros(len(genome)), "price")
    assert strategy.active_features == []

    class _Ctx:
        close = np.zeros((10, 4))
        tradable = np.ones(4, dtype=bool)
        features = np.zeros((4, 1))
        feature_names = ("x",)

    assert np.isnan(strategy.score(_Ctx())).all()


def test_the_genome_decides_the_portfolio_shape(genome):
    from sp500lab.strategies.evolvable import from_vector

    params = {f"w_{f}": 0.0 for f in PRESETS["price"]}
    params |= {"w_mom_12_1": 1.0, "top_k": 25, "max_weight": 0.08,
               "weighting": "score_rank", "use_regime": "on",
               "defensive_gross": 0.4, "vol_trigger": 2.0}
    strategy = from_vector(genome.encode(params), "price")
    assert strategy.construction.top_k == 25
    assert strategy.construction.weighting == "score_rank"
    assert strategy.construction.max_weight == pytest.approx(0.08)
    assert strategy.active_features == ["mom_12_1"]


def test_pre_ranked_mode_asks_for_differently_named_columns(genome):
    """A mismatch must be a KeyError, not a plausible-looking wrong portfolio."""
    from sp500lab.strategies.evolvable import from_vector

    raw = from_vector(np.zeros(len(genome)), "price")
    ranked = from_vector(np.zeros(len(genome)), "price", pre_ranked=True)
    assert "mom_12_1" in raw.requires_features
    assert "mom_12_1__rank" in ranked.requires_features
    assert "mom_12_1" not in ranked.requires_features


def test_seed_vectors_reproduce_the_strategies_they_are_named_after(genome):
    from sp500lab.evolve import seed_vectors
    from sp500lab.strategies.evolvable import from_vector

    seeds = seed_vectors(genome, "price")
    assert len(seeds) >= 5
    named = {tuple(from_vector(v, "price").active_features) for v in seeds}
    assert ("mom_12_1",) in named
    assert ("vol_126d",) in named


# --------------------------------------------------------------------------
# End to end, on real data
# --------------------------------------------------------------------------

def _have_data() -> bool:
    try:
        from sp500lab.storage import silver_exists
        return silver_exists("market/daily_bars_adjusted")
    except Exception:                                             # noqa: BLE001
        return False


needs_data = pytest.mark.skipif(not _have_data(),
                                reason="silver layer not built; run `sp500lab ingest all`")


@needs_data
def test_a_tiny_search_runs_and_is_reproducible(tmp_path, monkeypatch):
    """Two searches with the same seed must produce the same winner, exactly."""
    from sp500lab.evolve import EvolutionConfig, engine, evolve

    monkeypatch.setattr(engine, "EVOLVE_DIR", tmp_path)
    config = EvolutionConfig(study="unit-test", population=8, generations=2,
                             seed=3, log_runs=False, checkpoint=False,
                             seed_with_baselines=True)
    a = evolve(config)
    b = evolve(config)

    assert a.best is not None
    assert a.best.fingerprint == b.best.fingerprint
    assert a.best.fitness == pytest.approx(b.best.fitness)
    assert len(a.history) == 2
    assert a.n_distinct <= a.n_evaluations, "the cache must not invent individuals"


@needs_data
def test_a_search_checkpoints_every_generation(tmp_path, monkeypatch):
    from sp500lab.evolve import EvolutionConfig, engine, evolve
    from sp500lab.evolve.engine import load_history, load_population

    monkeypatch.setattr(engine, "EVOLVE_DIR", tmp_path)
    evolve(EvolutionConfig(study="chk", population=6, generations=2, seed=1,
                           log_runs=False, checkpoint=True))
    monkeypatch.setattr("sp500lab.evolve.engine.EVOLVE_DIR", tmp_path)
    assert len(load_history("chk")) == 2
    assert len(load_population("chk")) == 6
