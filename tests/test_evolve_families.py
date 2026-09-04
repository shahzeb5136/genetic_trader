"""The 2026-09 redesign of the search: families, robust fitness, penalties, ensembles.

Same rule as test_evolve.py - nothing here asserts that evolution finds a good strategy.
What is testable is the structure the result depends on:

  * a family preset cannot express more than its cap, and cannot reverse a prior
  * two vectors that differ only past the cap or inside the dead zone are ONE trial
  * the frozen family presets decode by position, exactly as ADR-038 demands
  * random sub-periods are reproducible, inside the window, and the right length
  * the worst-quartile aggregate punishes a rule that only works in one stretch
  * every penalty reduces fitness by exactly what it says
  * an ensemble averages BELIEFS, votes on the gate, and describes itself as JSON
  * a multi-seed search pools its seeds into one ensemble and writes it beside the
    checkpoint, and `winners()` hands that ensemble over
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sp500lab.evolve.fitness import UNFIT, Folds, Objective
from sp500lab.strategies.genome import (CUT_FEATURES, DEAD_ZONE, FAMILIES,
                                        FAMILY_BY_NAME, FAMILY_PRESETS, PRESET_MIN_DATE,
                                        PRESETS, active_families, active_features,
                                        all_presets, alpha_genome, describe_genome,
                                        family_weights, preset_features, preset_kind)


@pytest.fixture
def genome():
    return alpha_genome("families")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _vector(genome, **weights) -> np.ndarray:
    params = {f"f_{f.name}": 0.0 for f in FAMILIES if f"f_{f.name}" in genome.names}
    params.update({f"f_{k}": v for k, v in weights.items()})
    params |= {"top_k": 40, "max_weight": 0.05, "weighting": "equal",
               "use_regime": "off", "defensive_gross": 0.5, "vol_trigger": 1.5}
    return genome.encode(params)


# --------------------------------------------------------------------------
# The families and their presets are frozen
# --------------------------------------------------------------------------

def test_the_family_presets_are_frozen():
    """ADR-038 applied to families: a stored genome decodes BY POSITION against its
    preset's family tuple, and each family's members and signs are part of that
    contract. These pins fail the moment anyone touches one; the fix is a new preset."""
    names = tuple(f.name for f in FAMILIES)
    assert names == ("momentum", "reversal", "low_risk", "liquidity", "payout",
                     "value", "quality", "investment", "earnings")
    assert FAMILY_PRESETS["families"].families == names
    assert FAMILY_PRESETS["families"].max_active == 3
    assert FAMILY_PRESETS["families"].min_date == "2010-07-01"
    assert FAMILY_PRESETS["families-price"].families == (
        "momentum", "reversal", "low_risk", "liquidity", "payout")
    assert FAMILY_PRESETS["families-price"].max_active == 3
    assert FAMILY_BY_NAME["momentum"].members == (
        ("mom_12_1", 1), ("resid_mom_12_1", 1), ("high_52w_ratio", 1),
        ("info_discreteness", -1))
    assert FAMILY_BY_NAME["low_risk"].members == (
        ("vol_126d", -1), ("idio_vol_252d", -1), ("beta_252d", -1), ("max_ret_21d", -1))
    assert FAMILY_BY_NAME["quality"].members == (
        ("gross_profitability", 1), ("roe", 1), ("accruals", -1), ("leverage", -1))
    assert FAMILY_BY_NAME["value"].members == (
        ("book_to_market", 1), ("earnings_yield", 1), ("cf_yield", 1))
    assert FAMILY_BY_NAME["payout"].members == (
        ("div_yield", 1), ("div_growth_1y", 1), ("div_due_1m", 1))
    for fam in FAMILIES:
        assert fam.story and fam.reference, fam.name
        assert all(s in (-1, 1) for _, s in fam.members)


def test_every_family_member_is_catalogued_and_nothing_is_both_kept_and_cut():
    from sp500lab.features.catalog import FEATURE_DOCS
    kept = {f for fam in FAMILIES for f in fam.features}
    cut = {f for _, names in CUT_FEATURES for f in names}
    assert kept <= set(FEATURE_DOCS)
    assert cut <= set(FEATURE_DOCS)
    assert not kept & cut, kept & cut
    assert len(kept) == 22


def test_the_feature_presets_did_not_move():
    """The three older presets are what the checkpoints on disk decode against."""
    assert tuple(PRESETS) == ("price", "full", "night")
    assert all_presets() == ("price", "full", "night", "families", "families-price")
    assert {preset_kind(p) for p in PRESETS} == {"features"}
    assert {preset_kind(p) for p in FAMILY_PRESETS} == {"families"}
    assert PRESET_MIN_DATE["families"] == "2010-07-01"
    assert PRESET_MIN_DATE["families-price"] == ""
    assert len(alpha_genome("price")) == 19
    assert len(alpha_genome("full")) == 29
    assert len(alpha_genome("night")) == 23
    assert len(alpha_genome("families")) == 15
    assert len(alpha_genome("families-price")) == 11


def test_preset_features_is_the_union_of_the_members_in_family_order():
    feats = preset_features("families-price")
    assert feats[:4] == ("mom_12_1", "resid_mom_12_1", "high_52w_ratio",
                         "info_discreteness")
    assert "rev_1m" in feats and "div_due_1m" in feats
    assert "book_to_market" not in feats
    assert len(feats) == len(set(feats))


# --------------------------------------------------------------------------
# The cap and the prior are structural
# --------------------------------------------------------------------------

def test_a_family_genome_cannot_back_more_than_its_cap(genome, rng):
    for _ in range(100):
        v = genome.random(rng)
        assert len(active_families(genome, v)) <= genome.max_active
        decoded = genome.decode(v)
        live = [k for k, x in decoded.items() if k.startswith("f_") and x > 0]
        assert len(live) <= 3


def test_the_cap_keeps_the_largest_weights_and_zeroes_the_rest(genome):
    v = _vector(genome, momentum=0.9, reversal=0.5, low_risk=0.05, liquidity=0.7,
                payout=0.3, value=0.2, quality=0.95)
    assert active_families(genome, v) == ["quality", "momentum", "liquidity"]
    assert family_weights(genome, v)[0] == ("quality", pytest.approx(0.95))
    eff = genome.effective(v)
    assert eff[genome.names.index("f_reversal")] == 0.0
    assert eff[genome.names.index("f_low_risk")] == 0.0     # dead zone as well


def test_family_weights_cannot_be_negative(genome):
    v = _vector(genome, value=-0.8)
    assert genome.clip(v)[genome.names.index("f_value")] == 0.0
    assert active_families(genome, v) == []


def test_capped_out_and_dead_zone_genes_do_not_change_the_fingerprint(genome):
    """Two vectors that build the same portfolio are ONE trial, whatever the bits say."""
    a = _vector(genome, momentum=0.9, quality=0.8, value=0.7, payout=0.2)
    b = _vector(genome, momentum=0.9, quality=0.8, value=0.7, payout=0.05)
    c = _vector(genome, momentum=0.9, quality=0.8, value=0.7, payout=0.6)
    assert genome.fingerprint(a) == genome.fingerprint(b)
    assert genome.fingerprint(a) == genome.fingerprint(c)   # payout capped out
    d = _vector(genome, momentum=0.9, quality=0.8, value=0.7, payout=0.75)
    assert genome.fingerprint(a) != genome.fingerprint(d)   # payout replaces value


def test_a_feature_genome_still_decodes_exactly_as_before():
    """No cap, and sub-dead-zone weights survive decode: stored winners must not move."""
    g = alpha_genome("price")
    v = np.zeros(len(g))
    v[0] = 0.05
    v[1] = 0.5
    assert g.max_active is None
    assert g.decode(v)["w_mom_12_1"] == pytest.approx(0.05)
    assert active_features(g, v) == ["resid_mom_12_1"]
    assert g.fingerprint(v).startswith("0.000,0.500,")


def test_active_features_of_a_family_genome_are_its_live_members(genome):
    v = _vector(genome, low_risk=1.0, earnings=0.4)
    assert active_families(genome, v) == ["low_risk", "earnings"]
    assert active_features(genome, v) == ["vol_126d", "idio_vol_252d", "beta_252d",
                                          "max_ret_21d", "eps_surprise"]


def test_describe_genome_reads_families_as_sentences(genome):
    text = describe_genome(genome, _vector(genome, quality=0.9, momentum=0.4))
    assert "Backs 2 of 9 families, at most 3" in text
    assert "Quality" in text and "-accruals" in text
    assert "Ignores 7 family(ies)" in text
    assert "Holds the top 40" in text


def test_seed_vectors_back_one_family_each(genome):
    from sp500lab.evolve import seed_vectors
    seeds = seed_vectors(genome, "families")
    assert len(seeds) == 9
    assert [active_families(genome, v) for v in seeds] == [[f.name] for f in FAMILIES]


# --------------------------------------------------------------------------
# The strategy a family genome becomes
# --------------------------------------------------------------------------

class _Ctx:
    """The smallest context a score() needs: ranks already in [0, 1]."""

    def __init__(self, names, values, tradable=None):
        self.feature_names = tuple(names)
        self.features = np.asarray(values, dtype=float)
        n = self.features.shape[0]
        self.close = np.zeros((10, n))
        self.tradable = (np.ones(n, dtype=bool) if tradable is None
                         else np.asarray(tradable, dtype=bool))


def _pre_ranked_ctx(n=12, seed=0):
    from sp500lab.strategies.evolvable import EvolvedAlpha
    rng = np.random.default_rng(seed)
    feats = preset_features("families") + ("vol_126d",)
    names = [f + EvolvedAlpha.RANK_SUFFIX for f in dict.fromkeys(feats)]
    names += ["mkt_trend_200d", "mkt_vol_ratio"]
    values = rng.uniform(0, 1, size=(n, len(names)))
    values[:, -2] = 0.05                                    # market above its average
    values[:, -1] = 1.0
    return _Ctx(names, values)


def test_a_family_strategy_scores_the_prior_signed_mean_of_its_members(genome):
    from sp500lab.strategies.evolvable import EvolvedFamilies, from_vector

    ctx = _pre_ranked_ctx()
    strat = from_vector(_vector(genome, low_risk=1.0), "families", pre_ranked=True)
    assert isinstance(strat, EvolvedFamilies)
    assert strat.active_families == ["low_risk"]
    score = strat.score(ctx)
    cols = [ctx.feature_names.index(f + "__rank") for f in
            ("vol_126d", "idio_vol_252d", "beta_252d", "max_ret_21d")]
    expected = (1.0 - ctx.features[:, cols]).mean(axis=1)
    assert score == pytest.approx(expected)


def test_two_families_are_weighted_by_their_genes(genome):
    from sp500lab.strategies.evolvable import from_vector

    ctx = _pre_ranked_ctx()
    strat = from_vector(_vector(genome, reversal=1.0, liquidity=0.5), "families",
                        pre_ranked=True)
    rev = ctx.features[:, ctx.feature_names.index("rev_1m__rank")]
    liq = ctx.features[:, ctx.feature_names.index("amihud_illiq__rank")]
    assert strat.score(ctx) == pytest.approx((1.0 * rev + 0.5 * liq) / 1.5)


def test_a_family_strategy_with_no_opinion_holds_cash(genome):
    from sp500lab.strategies.evolvable import from_vector

    strat = from_vector(np.zeros(len(genome)), "families")
    assert not strat.has_opinion
    assert np.isnan(strat.score(_pre_ranked_ctx())).all()


def test_a_missing_member_counts_as_average_not_as_a_bonus(genome):
    """The 1 - rank form: a name missing a low-is-good member is scored on the members
    it has, never rewarded for the absence."""
    from sp500lab.strategies.evolvable import from_vector

    ctx = _pre_ranked_ctx()
    ctx.features[0, ctx.feature_names.index("beta_252d__rank")] = np.nan
    strat = from_vector(_vector(genome, low_risk=1.0), "families", pre_ranked=True)
    score = strat.score(ctx)
    cols = [ctx.feature_names.index(f + "__rank") for f in
            ("vol_126d", "idio_vol_252d", "max_ret_21d")]
    assert score[0] == pytest.approx((1.0 - ctx.features[0, cols]).mean())


def test_from_vector_refuses_to_cross_the_kinds(genome):
    from sp500lab.strategies.evolvable import EvolvedAlpha, EvolvedFamilies
    with pytest.raises(KeyError):
        EvolvedAlpha(preset="families")
    with pytest.raises(KeyError):
        EvolvedFamilies(preset="price")


def test_registered_default_instances_construct_and_hold_cash():
    from sp500lab.backtest.strategy import get_strategy
    s = get_strategy("evolved_families")
    assert s.active_families == []
    assert s.min_date == "2010-07-01"


# --------------------------------------------------------------------------
# Random sub-periods and the worst-quartile objective
# --------------------------------------------------------------------------

def test_random_folds_are_reproducible_and_inside_the_window():
    a = Folds.random("2010-07-01", "2021-12-31", n=12, min_years=3, max_years=5, seed=0)
    b = Folds.random("2010-07-01", "2021-12-31", n=12, min_years=3, max_years=5, seed=0)
    c = Folds.random("2010-07-01", "2021-12-31", n=12, min_years=3, max_years=5, seed=1)
    assert a.bounds == b.bounds and a.bounds != c.bounds
    assert len(a) == 12 and a.scheme == "random" and a.embargo_days == 0
    for lo, hi in a.bounds:
        assert "2010-07-01" <= lo < hi <= "2021-12-31"
        days = (pd.Timestamp(hi) - pd.Timestamp(lo)).days
        assert 3 * 365 <= days <= 5 * 366
    assert list(a.bounds) == sorted(a.bounds)


def test_random_folds_refuse_a_window_shorter_than_one_sub_period():
    with pytest.raises(ValueError):
        Folds.random("2020-01-01", "2021-12-31", n=4, min_years=3, max_years=5)


def test_the_objective_records_the_scheme_for_the_manifest():
    obj = Objective(aggregate="quantile", quantile=0.25,
                    folds=Folds.random("2010-01-01", "2020-01-01", n=6))
    d = obj.describe()
    assert d["fold_scheme"] == "random" and d["n_folds"] == 6
    assert d["quantile"] == 0.25 and d["folds"].count("..") == 6


class _FakeResult:
    def __init__(self, equity, turnover=1.0, names=50):
        self.equity = equity
        self.diagnostics = {}
        self.performance = type("P", (), {
            "sharpe": 1.0, "calmar": 1.0, "information_ratio": 0.5,
            "ann_turnover": turnover, "avg_positions": names})()


def _curve(months, drift_by_month):
    """A daily curve whose monthly drift is given per month: one stretch can be good."""
    idx = pd.date_range("2010-01-04", periods=months * 21, freq="B")
    rng = np.random.default_rng(0)
    step = np.empty(len(idx))
    for m in range(months):
        step[m * 21:(m + 1) * 21] = rng.normal(drift_by_month[m], 0.008, size=21)
    return pd.Series(100_000 * np.cumprod(1 + step), index=idx.strftime("%Y-%m-%d"))


def test_the_worst_quartile_kills_a_rule_that_only_works_in_one_stretch():
    """Same mean drift over the whole window; one earned it in 30 months, the other
    every month. Whole-window Sharpe cannot tell them apart. The quantile can."""
    months = 120
    steady = _curve(months, [0.0006] * months)
    lumpy = _curve(months, [0.0024] * 30 + [0.0] * 90)
    start, end = str(steady.index[0]), str(steady.index[-1])
    folds = Folds.random(start, end, n=12, min_years=3, max_years=4, seed=0)
    robust = Objective(aggregate="quantile", quantile=0.25, folds=folds)
    f_steady, d_steady = robust.score(_FakeResult(steady))
    f_lumpy, d_lumpy = robust.score(_FakeResult(lumpy))
    assert np.isfinite(f_steady) and np.isfinite(f_lumpy)
    assert f_steady > f_lumpy
    assert d_lumpy["fold_quantile"] < d_lumpy["fold_mean"]
    assert d_steady["fold_min"] <= d_steady["fold_quantile"] <= d_steady["fold_mean"]


def test_min_and_quantile_at_zero_agree():
    curve = _curve(96, [0.0005] * 96)
    folds = Folds.random(str(curve.index[0]), str(curve.index[-1]), n=8, seed=3)
    worst, _ = Objective(aggregate="min", folds=folds).score(_FakeResult(curve))
    q0, _ = Objective(aggregate="quantile", quantile=0.0, folds=folds).score(
        _FakeResult(curve))
    assert worst == pytest.approx(q0)


def test_every_penalty_charges_exactly_what_it_says():
    curve = _curve(96, [0.0005] * 96)
    plain = Objective(aggregate="whole")
    charged = Objective(aggregate="whole", turnover_penalty=0.03,
                        complexity_penalty=0.01, family_penalty=0.02,
                        gate_penalty=0.05)
    result = _FakeResult(curve, turnover=3.0)
    base, _ = plain.score(result)
    fit, detail = charged.score(result, n_active=9, n_families=3, gate_on=True)
    assert fit == pytest.approx(base - 0.03 * 3.0 - 0.01 * 9 - 0.02 * 3 - 0.05, abs=1e-9)
    off, _ = charged.score(result, n_active=9, n_families=3, gate_on=False)
    assert off == pytest.approx(fit + 0.05, abs=1e-9)
    assert detail["n_families"] == 3 and detail["gate_on"] is True


def test_the_objective_still_rejects_the_degenerate_cases():
    obj = Objective(aggregate="quantile",
                    folds=Folds.random("2010-01-01", "2020-01-01", n=4, seed=0))
    short = _curve(12, [0.001] * 12)
    fitness, detail = obj.score(_FakeResult(short))
    assert fitness == UNFIT and "months" in detail["reject"]


def test_the_config_defaults_are_the_redesign():
    from sp500lab.evolve import EvolutionConfig
    c = EvolutionConfig()
    assert c.preset == "families" and c.costs == "pessimistic"
    assert c.fold_scheme == "random" and c.n_folds == 12
    assert (c.fold_min_years, c.fold_max_years) == (3.0, 5.0)
    assert c.aggregate == "quantile" and c.quantile == 0.25
    assert c.turnover_penalty > 0 and c.complexity_penalty > 0
    assert c.family_penalty > 0 and c.gate_penalty > 0
    assert c.ensemble_size == 30
    assert EvolutionConfig(seed=5, n_seeds=3).seeds == [5, 6, 7]


# --------------------------------------------------------------------------
# The ensemble
# --------------------------------------------------------------------------

def _members(genome):
    from sp500lab.strategies.evolvable import from_vector
    specs = [dict(low_risk=1.0), dict(quality=0.8, momentum=0.4), dict(value=1.0),
             dict(momentum=1.0, payout=0.3)]
    out = [from_vector(_vector(genome, **spec), "families", pre_ranked=True)
           for spec in specs]
    return out


def test_an_ensemble_averages_the_re_ranked_beliefs_of_its_members(genome):
    from sp500lab.strategies.evolvable import EvolvedEnsemble
    from sp500lab.strategies.signals import rank_pct

    members = _members(genome)
    ens = EvolvedEnsemble(members, study="unit")
    ctx = _pre_ranked_ctx(n=15)
    expected = np.mean([rank_pct(m.alpha(ctx), ctx.tradable) for m in members], axis=0)
    assert ens.score(ctx) == pytest.approx(expected)
    assert ens.min_members == 3


def test_an_ensemble_needs_enough_opinions_per_name(genome):
    from sp500lab.strategies.evolvable import EvolvedEnsemble

    members = _members(genome)
    ens = EvolvedEnsemble(members, study="unit", min_members=4)
    ctx = _pre_ranked_ctx(n=15)
    # Blind every value member on name 0: the value family is one member of four.
    for f in ("book_to_market", "earnings_yield", "cf_yield"):
        ctx.features[0, ctx.feature_names.index(f + "__rank")] = np.nan
    score = ens.score(ctx)
    assert np.isnan(score[0]) and np.isfinite(score[1:]).all()


def test_the_gate_is_a_vote(genome):
    from sp500lab.strategies.evolvable import EvolvedEnsemble, from_vector

    on = _vector(genome, low_risk=1.0)
    on[genome.names.index("use_regime")] = 1
    on[genome.names.index("defensive_gross")] = 0.4
    off = _vector(genome, quality=1.0)
    ctx = _pre_ranked_ctx()
    ctx.features[:, ctx.feature_names.index("mkt_trend_200d")] = -0.1   # falling market

    one_of_three = EvolvedEnsemble([from_vector(on, "families", pre_ranked=True),
                                    from_vector(off, "families", pre_ranked=True),
                                    from_vector(off, "families", pre_ranked=True)],
                                   study="unit")
    assert not one_of_three.is_defensive(ctx)
    two_of_three = EvolvedEnsemble([from_vector(on, "families", pre_ranked=True),
                                    from_vector(on, "families", pre_ranked=True),
                                    from_vector(off, "families", pre_ranked=True)],
                                   study="unit")
    share, gross = two_of_three.vote(ctx)
    assert two_of_three.is_defensive(ctx) and share == pytest.approx(2 / 3)
    assert gross == pytest.approx(0.4)


def test_an_ensemble_describes_itself_as_json_and_requires_the_union(genome):
    from sp500lab.strategies.evolvable import EvolvedEnsemble

    members = _members(genome)
    ens = EvolvedEnsemble(members, study="unit")
    d = ens.describe()
    json.dumps(d)                                             # must be serialisable
    assert d["params"]["n_members"] == 4
    assert len(d["params"]["member_fingerprints"]) == 4
    assert d["family_usage"] == {"low_risk": 1, "quality": 1, "momentum": 2,
                                 "value": 1, "payout": 1}
    needed = set(ens.requires_features)
    for m in members:
        assert set(m.requires_features) <= needed
    assert ens.min_date == "2010-07-01"
    assert ens.construction.top_k == 40 and ens.construction.weighting == "equal"
    assert "4 evolved individuals" in ens.explain()


def test_an_ensemble_refuses_mixed_presets(genome):
    from sp500lab.strategies.evolvable import EvolvedEnsemble, from_vector
    a = from_vector(_vector(genome, low_risk=1.0), "families")
    b = from_vector(np.zeros(19), "price")
    with pytest.raises(ValueError):
        EvolvedEnsemble([a, b])
    with pytest.raises(ValueError):
        EvolvedEnsemble([])


# --------------------------------------------------------------------------
# `forward run` can spend a look on a search's deliverable by name
# --------------------------------------------------------------------------

def test_forward_run_resolves_an_evolved_deliverable_by_name(genome, monkeypatch):
    """Without this, the only way to forward-test one ensemble was a suite that also
    re-spends a look on every older champion."""
    from argparse import Namespace

    from sp500lab.forward import cli as fcli
    from sp500lab.strategies.evolvable import EvolvedEnsemble, from_vector

    members = [from_vector(_vector(genome, quality=0.9), "families"),
               from_vector(_vector(genome, low_risk=0.7), "families")]
    ens = EvolvedEnsemble(members, study="ga-x")
    ens.name = "ga-x-ensemble"
    monkeypatch.setattr("sp500lab.evolve.engine.winners",
                        lambda: [{"name": "ga-x-ensemble", "strategy": ens,
                                  "study": "ga-x", "kind": "ensemble"}])

    args = Namespace(strategy="ga-x-ensemble", top_k=None, max_weight=None,
                     weighting=None, origin_study=None)
    assert fcli._resolve_strategy(args) is ens
    assert args.origin_study == "ga-x", "the search behind the candidate travels with it"

    explicit = Namespace(strategy="ga-x-ensemble", top_k=None, max_weight=None,
                         weighting=None, origin_study="ga-other")
    fcli._resolve_strategy(explicit)
    assert explicit.origin_study == "ga-other"

    with pytest.raises(SystemExit):
        fcli._resolve_strategy(Namespace(strategy="ga-nope-ensemble", top_k=None,
                                         max_weight=None, weighting=None,
                                         origin_study=None))


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
def test_a_pooled_search_writes_one_ensemble_and_winners_hands_it_over(tmp_path,
                                                                        monkeypatch):
    from sp500lab.evolve import EvolutionConfig, engine, evolve
    from sp500lab.evolve.engine import (build_ensemble, champion, load_ensemble,
                                        load_history, load_population, winners)
    from sp500lab.strategies.evolvable import EvolvedEnsemble

    monkeypatch.setattr(engine, "EVOLVE_DIR", tmp_path)
    config = EvolutionConfig(study="unit-pool", preset="families-price", population=6,
                             generations=2, n_seeds=2, ensemble_size=4, seed=7,
                             log_runs=False, checkpoint=True)
    result = evolve(config)

    assert result.seeds == [7, 8]
    assert result.ensemble is not None and result.ensemble["size"] == 4
    assert result.ensemble["seeds"] == [7, 8]
    assert np.isfinite(result.ensemble["evaluation"]["sharpe"])
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "unit-pool.ensemble.json", "unit-pool.jsonl"]

    hist = load_history("unit-pool")
    assert sorted(hist["seed"].unique()) == [7, 8] and len(hist) == 4
    assert len(load_population("unit-pool", seed=7)) == 6
    assert champion("unit-pool")["fitness"] == pytest.approx(result.best.fitness)
    assert len(load_ensemble("unit-pool")["members"]) == 4

    found = [w for w in winners() if w["study"] == "unit-pool"]
    assert len(found) == 1
    assert found[0]["kind"] == "ensemble" and found[0]["name"] == "unit-pool-ensemble"
    assert isinstance(found[0]["strategy"], EvolvedEnsemble)
    assert found[0]["n_members"] == 4 and found[0]["seeds"] == [7, 8]

    # Rebuilding from the checkpoint alone reproduces the same members.
    rebuilt = build_ensemble("unit-pool", evaluate=False, write=False)
    assert [m["fingerprint"] for m in rebuilt["members"]] == \
        [m["fingerprint"] for m in result.ensemble["members"]]


@needs_data
def test_a_search_without_an_ensemble_still_hands_over_its_champion(tmp_path,
                                                                     monkeypatch):
    from sp500lab.evolve import EvolutionConfig, engine, evolve
    from sp500lab.evolve.engine import winners

    monkeypatch.setattr(engine, "EVOLVE_DIR", tmp_path)
    evolve(EvolutionConfig(study="unit-champ", preset="families-price", population=6,
                           generations=1, ensemble_size=0, seed=2,
                           log_runs=False, checkpoint=True))
    found = [w for w in winners() if w["study"] == "unit-champ"]
    assert found and found[0]["kind"] == "champion"
    assert found[0]["name"] == "unit-champ-best"
