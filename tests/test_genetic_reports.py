"""The genetic-algorithm lab: does each page say the thing it exists to say?

A genetic algorithm is the easiest way in this project to publish a number that means
nothing, so these pages carry a specific set of claims and the tests are about those
claims rather than about rendering:

  * the trial count and the deflated Sharpe travel with every winner's result
  * a probability is never printed as certainty
  * the fitness folds are described as robustness, NOT as out-of-sample evidence
  * the feature pages report what the POPULATION converged on, not only the winner
  * a search whose checkpoint is gone is named rather than dropped

Everything asserts on spec objects. No HTML is parsed here except in the one end-to-end
test that checks the folder contains three pages and nothing else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp500lab.reporting import genetic_views as GV
from sp500lab.reporting import queries as Q
from sp500lab.reporting.specs import LineChart, LinkGrid, Note, StatRow, TableBlock


# --------------------------------------------------------------------------
# Fixtures: a synthetic search, so the tests do not need a real one on disk
# --------------------------------------------------------------------------

def _history(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "generation": range(n),
        "best_fitness": np.linspace(0.5, 0.9, n),
        "mean_fitness": np.linspace(0.1, 0.6, n),
        "diversity": np.linspace(0.30, 0.12, n),
        "best_n_active": [3] * n,
    })


def _search(study="ga-demo", preset="price", *, forward=None, dsr=0.99,
            n_trials=1400) -> dict:
    return {
        "study": study, "preset": preset,
        "config": {"population": 60, "generations": 25, "elite": 4, "immigrants": 4,
                   "tournament_size": 3, "crossover_rate": 0.7, "mutation_rate": 0.15,
                   "mutation_sigma": 0.15, "turnover_penalty": 0.03,
                   "complexity_penalty": 0.01, "dispersion_weight": 0.5, "seed": 11,
                   "costs": "realistic", "holdout": "exclude",
                   "seed_with_baselines": True, "n_folds": 4},
        "history": _history(),
        "winner_name": f"{study}-best",
        "winner_fitness": 0.9,
        "weights": [("vol_126d", -0.89), ("mom_12_1", 0.33)],
        "n_ignored": 11,
        "active": ["vol_126d", "mom_12_1"],
        "portfolio": {"top_k": 12, "weighting": "equal", "max_weight": 0.06,
                      "use_regime": "on", "defensive_gross": 0.24, "vol_trigger": 2.47},
        "prose": "Ranks 2 features.",
        "usage": {"vol_126d": 60, "mom_12_1": 41, "div_yield": 58},
        "n_population": 60,
        "deflation": {"n_trials": n_trials, "deflated_sharpe": dsr,
                      "trial_sharpe_std": 0.19, "n_months": 175,
                      "sharpe_annualised_monthly": 1.37,
                      "expected_max_sharpe_annualised": 0.64},
        "research": {"cagr": 0.124, "sharpe": 1.155, "maxdd": -0.16, "turnover": 3.5,
                     "cost_drag": 0.006},
        "champion_research": {"cagr": 0.124, "sharpe": 1.155, "maxdd": -0.16,
                              "turnover": 3.5, "cost_drag": 0.006},
        "window": "2007-04 → 2021-12",
        "forward": forward,
        # The 2026-09 additions. A feature-preset search predates families and
        # ensembles, so it carries the empty forms of both.
        "kind": "features",
        "objective": {"metric": "sharpe_monthly", "costs": "realistic",
                      "fold_scheme": "contiguous", "n_folds": 4,
                      "aggregate": "mean_minus_std", "quantile": None,
                      "dispersion_weight": 0.5, "turnover_penalty": 0.03,
                      "complexity_penalty": 0.01, "family_penalty": 0.0,
                      "gate_penalty": 0.0, "n_seeds": 1, "ensemble_size": 0},
        "seeds": [11],
        "n_individuals": n_trials,
        "families": [],
        "family_usage": {},
        "deliverable": f"{study}-best",
        "ensemble": None,
    }


def _family_search(study="ga-fam", *, ensemble_forward=None) -> dict:
    """A search over the `families` preset with a stored ensemble, as the lab reads it."""
    s = _search(study, "families", forward=None, dsr=0.97, n_trials=4200)
    s["kind"] = "families"
    s["objective"] = {"metric": "sharpe_monthly", "costs": "pessimistic",
                      "fold_scheme": "random", "n_folds": 12, "fold_min_years": 3.0,
                      "fold_max_years": 5.0, "aggregate": "quantile", "quantile": 0.25,
                      "dispersion_weight": 0.5, "turnover_penalty": 0.03,
                      "complexity_penalty": 0.01, "family_penalty": 0.02,
                      "gate_penalty": 0.03, "n_seeds": 3, "ensemble_size": 30}
    s["seeds"] = [0, 1, 2]
    h = pd.concat([_history().assign(seed=k) for k in (0, 1, 2)], ignore_index=True)
    s["history"] = h
    s["families"] = [("quality", 0.9), ("low_risk", 0.6)]
    s["family_usage"] = {"quality": 55, "low_risk": 40, "momentum": 12}
    s["weights"] = [("gross_profitability", 0.9), ("roe", 0.9), ("accruals", -0.9),
                    ("leverage", -0.9), ("vol_126d", -0.6), ("idio_vol_252d", -0.6),
                    ("beta_252d", -0.6), ("max_ret_21d", -0.6)]
    s["n_ignored"] = 13
    s["usage"] = {"gross_profitability": 55, "vol_126d": 40, "mom_12_1": 12}
    s["deliverable"] = f"{study}-ensemble"
    s["ensemble"] = {
        "name": f"{study}-ensemble", "size": 30, "seeds": [0, 1, 2],
        "built_at": "2026-09-04T12:00:00Z",
        "family_usage": {"quality": 24, "low_risk": 18, "momentum": 9, "payout": 3},
        "feature_usage": {"gross_profitability": 24, "vol_126d": 18},
        "construction": {"top_k": 42, "weighting": "equal", "max_weight": 0.06,
                         "min_names": 10, "gross": 1.0, "long_only": True},
        "prose": "Averages the signals of 30 evolved individuals.",
        "evaluation": {"fitness": 0.71, "sharpe": 1.05, "cagr": 0.151,
                       "max_drawdown": -0.29, "turnover": 2.1, "costs": "pessimistic"},
        "member_fitness": {"best": 0.78, "worst": 0.66},
        "champion_fitness": 0.78,
        "research": {"cagr": 0.158, "sharpe": 1.12, "maxdd": -0.28, "turnover": 2.1,
                     "cost_drag": 0.004},
        "deflation": {"n_trials": 4200, "deflated_sharpe": 0.97, "trial_sharpe_std": 0.2,
                      "n_months": 137, "sharpe_annualised_monthly": 1.3,
                      "expected_max_sharpe_annualised": 0.75},
        "forward": ensemble_forward,
    }
    return s


DECAYED = {"verdict": "decayed", "research_sharpe": 1.15, "forward_sharpe": 0.19,
           "forward_d_sharpe": -0.64, "decay_z": -1.94}


def _genome() -> dict:
    return Q.genome_anatomy()


def _hrefs() -> dict:
    return dict(Q.GENETIC_PAGES) | {"backtest": "../backtest/index.html"}


def _text(report) -> str:
    """Every string a report contains, flattened. Contents, never markup."""
    out = [report.title, report.subtitle]
    for section in report.sections:
        out += [section.title, section.blurb]
        for block in section.blocks:
            if isinstance(block, Note):
                out += [block.title, block.text]
            elif isinstance(block, StatRow):
                out += [s.label + s.value + s.note for s in block.stats]
            elif isinstance(block, TableBlock):
                out += [block.title, block.table.caption, *block.table.columns]
                out += [c.text for row in block.table.rows for c in row]
            elif isinstance(block, LinkGrid):
                out += [c.title + c.href + c.blurb for c in block.cards]
            else:
                out += [getattr(block, "title", ""), getattr(block, "caption", "")]
    return " ".join(str(o) for o in out)


def _tables(report) -> list:
    return [b.table for s in report.sections for b in s.blocks
            if isinstance(b, TableBlock)]


# --------------------------------------------------------------------------
# The anatomy comes from the code, not from prose
# --------------------------------------------------------------------------

def test_the_genome_anatomy_is_read_off_the_real_genome():
    """If the search space changes, the methodology page must change with it."""
    from sp500lab.strategies.genome import (DEAD_ZONE, FAMILY_PRESETS, all_presets,
                                            alpha_genome)

    g = _genome()
    assert g["dead_zone"] == DEAD_ZONE
    assert set(g["sizes"]) == set(all_presets())
    for preset in all_presets():
        assert g["sizes"][preset] == len(alpha_genome(preset))
    assert g["kinds"]["families"] == "families" and g["kinds"]["price"] == "features"
    assert g["caps"] == {n: fp.max_active for n, fp in FAMILY_PRESETS.items()}
    assert g["n_families"]["families"] == 9 and g["n_families"]["price"] == 0
    assert g["defaults"]["preset"] == "families"
    names = {gene["name"] for gene in g["shape_genes"]}
    assert names == {"top_k", "max_weight", "weighting", "use_regime",
                     "defensive_gross", "vol_trigger"}
    assert not any(n.startswith("w_") for n in names), "feature weights are not shape"


def test_every_shape_gene_carries_its_real_bounds():
    top_k = next(g for g in _genome()["shape_genes"] if g["name"] == "top_k")
    assert (top_k["low"], top_k["high"], top_k["integer"]) == (10, 100, True)


# --------------------------------------------------------------------------
# A probability is never printed as certainty
# --------------------------------------------------------------------------

def test_a_deflated_sharpe_is_never_rendered_as_one():
    """0.9995 at three decimals is '1.000', which claims certainty nothing supports."""
    assert GV._dsr(0.9995) == ">0.999"
    assert GV._dsr(1.0) == ">0.999"
    assert GV._dsr(0.9827) == "0.983"
    assert GV._dsr(None) == "—"
    assert GV._dsr(float("nan")) == "—"


def test_no_page_prints_a_probability_of_one():
    report = GV.searches_report([_search(dsr=0.9995, forward=DECAYED)], [],
                                hrefs=_hrefs())
    assert "1.000" not in _text(report)
    assert ">0.999" in _text(report)


# --------------------------------------------------------------------------
# Methodology
# --------------------------------------------------------------------------

def test_the_methodology_page_says_the_folds_are_not_out_of_sample():
    """The single most misreadable thing about this search. Pinned deliberately."""
    report = GV.methodology_report(_genome(), [_search()], hrefs=_hrefs())
    danger = [b for s in report.sections for b in s.blocks
              if isinstance(b, Note) and b.level == "danger"]
    text = " ".join(n.title + n.text for n in danger)
    assert "ROBUSTNESS" in text
    assert "not evidence of generalisation" in text


def test_the_methodology_page_lists_all_four_defences():
    report = GV.methodology_report(_genome(), [_search()], hrefs=_hrefs())
    text = _text(report)
    for defence in ("1.", "2.", "3.", "4."):
        assert defence in text
    assert "logged as a trial" in text
    assert "holdout is untouched" in text


def test_the_methodology_page_explains_the_dead_zone_and_the_ranks():
    text = _text(GV.methodology_report(_genome(), [_search()], hrefs=_hrefs()))
    assert "PERCENTILE RANKS" in text
    assert "dead zone" in text.lower()


def test_the_methodology_page_reports_that_the_winners_decayed():
    report = GV.methodology_report(
        _genome(), [_search("a", forward=DECAYED), _search("b", forward=DECAYED)],
        hrefs=_hrefs())
    assert "2 of 2 winners tested out of sample DECAYED" in _text(report)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def _catalog(name):
    from sp500lab.features.catalog import describe
    return describe(name)


def test_the_features_page_lists_every_feature_of_every_preset():
    presets = {"price": ("mom_12_1", "vol_126d"), "full": ("mom_12_1", "roe")}
    report = GV.features_report(presets, [_search()], _catalog, hrefs=_hrefs())
    catalog = next(t for t in _tables(report) if "presets" in t.columns)
    listed = {row[0].text for row in catalog.rows}
    assert listed == {"mom_12_1", "vol_126d", "roe"}


def test_a_feature_says_which_presets_carry_it():
    presets = {"price": ("mom_12_1",), "full": ("mom_12_1", "roe")}
    report = GV.features_report(presets, [], _catalog, hrefs=_hrefs())
    catalog = next(t for t in _tables(report) if "presets" in t.columns)
    by_name = {row[0].text: row[-1].text for row in catalog.rows}
    assert by_name["mom_12_1"] == "price, full"
    assert by_name["roe"] == "full"


def test_the_features_page_reports_the_population_not_only_the_winner():
    """One individual is one draw. What 60 survivors agree on is the stronger claim."""
    report = GV.features_report({"price": ("vol_126d",)}, [_search()], _catalog,
                                hrefs=_hrefs())
    converged = next(t for t in _tables(report)
                     if any("population" in c for c in t.columns))
    by_name = {row[0].text: row for row in converged.rows}
    assert by_name["vol_126d"][3].text == "100.00%"      # 60 of 60
    # A feature the winner dropped but the population kept is still shown.
    assert by_name["div_yield"][1].text == "in the dead zone"
    assert by_name["div_yield"][3].text == "96.67%"      # 58 of 60


def test_the_features_page_warns_that_a_preset_is_frozen():
    text = _text(GV.features_report({"price": ("mom_12_1",)}, [], _catalog,
                                    hrefs=_hrefs()))
    assert "FROZEN" in text
    assert "mis-decode" in text


def test_the_features_page_says_the_full_preset_costs_history():
    text = _text(GV.features_report({"full": ("roe",)}, [], _catalog, hrefs=_hrefs()))
    assert "2010" in text


# --------------------------------------------------------------------------
# The searches
# --------------------------------------------------------------------------

def test_every_winner_carries_its_trial_count_and_deflated_sharpe():
    """The one thing that must never be separated from an evolved result."""
    report = GV.searches_report([_search(n_trials=1403, dsr=0.95, forward=DECAYED)], [],
                                hrefs=_hrefs())
    text = _text(report)
    assert "1,403" in text
    assert "0.950" in text or ">0.999" in text


def test_a_searched_sharpe_is_labelled_as_a_maximum():
    report = GV.searches_report([_search()], [], hrefs=_hrefs())
    warn = next(b for s in report.sections for b in s.blocks
                if isinstance(b, Note) and "not a Sharpe" in (b.title or ""))
    assert "MAXIMUM" in warn.text


def test_the_training_history_becomes_three_charts():
    report = GV.searches_report([_search()], [], hrefs=_hrefs())
    charts = [b for s in report.sections for b in s.blocks
              if isinstance(b, LineChart)]
    titles = {c.title for c in charts}
    assert titles == {"Best fitness by generation", "Mean fitness by generation",
                      "Population diversity by generation"}
    assert all(len(c.series[0]) == 10 for c in charts)


def test_diversity_is_explained_as_the_silent_failure_mode():
    report = GV.searches_report([_search()], [], hrefs=_hrefs())
    chart = next(b for s in report.sections for b in s.blocks
                 if isinstance(b, LineChart) and "diversity" in b.title.lower())
    assert "silently" in chart.caption


def test_an_untested_winner_says_so_rather_than_being_silent():
    report = GV.searches_report([_search(forward=None)], [], hrefs=_hrefs())
    assert "has not been forward-tested" in _text(report)


def test_a_search_with_no_checkpoint_is_named_rather_than_dropped():
    """Its trials still count toward the deflated Sharpe of everything in its study."""
    report = GV.searches_report(
        [_search()], [{"study": "ga-lost", "runs": 1005, "trials": 1005,
                       "best_sharpe": 1.45, "deflation": {"deflated_sharpe": 0.998}}],
        hrefs=_hrefs())
    text = _text(report)
    assert "ga-lost" in text
    assert "1,005" in text
    assert "no surviving checkpoint" in text.lower()


def test_three_for_three_is_stated_when_every_winner_decayed():
    searches = [_search("a", forward=DECAYED), _search("b", forward=DECAYED),
                _search("c", forward=DECAYED)]
    report = GV.searches_report(searches, [], hrefs=_hrefs())
    assert "3 of 3" in _text(report)


def test_an_empty_lab_produces_a_page_rather_than_an_exception():
    report = GV.searches_report([], [], hrefs=_hrefs())
    assert "Nothing has been searched yet" in _text(report)


# --------------------------------------------------------------------------
# The set holds together
# --------------------------------------------------------------------------

@pytest.mark.parametrize("build", ["methodology", "features", "searches"])
def test_every_page_links_to_the_other_two_and_back_to_the_scoreboard(build):
    searches = [_search()]
    if build == "methodology":
        report = GV.methodology_report(_genome(), searches, hrefs=_hrefs())
    elif build == "features":
        report = GV.features_report({"price": ("mom_12_1",)}, searches, _catalog,
                                    hrefs=_hrefs())
    else:
        report = GV.searches_report(searches, [], hrefs=_hrefs())

    hrefs = {c.href for s in report.sections for b in s.blocks
             if isinstance(b, LinkGrid) for c in b.cards}
    expected = {v for k, v in Q.GENETIC_PAGES.items() if k != build}
    assert expected <= hrefs, f"{build} does not link to its siblings"
    assert "../backtest/index.html" in hrefs
    assert Q.GENETIC_PAGES[build] not in hrefs, "a page must not link to itself"


def test_the_page_filenames_are_defined_in_one_place():
    assert set(Q.GENETIC_PAGES) == {"methodology", "features", "searches"}
    assert all(name.endswith(".html") for name in Q.GENETIC_PAGES.values())


# --------------------------------------------------------------------------
# End to end, against whatever searches are actually on disk
# --------------------------------------------------------------------------

def _have_searches() -> bool:
    try:
        from sp500lab.evolve.engine import EVOLVE_DIR
        return EVOLVE_DIR.exists() and any(EVOLVE_DIR.glob("*.jsonl"))
    except Exception:                                             # noqa: BLE001
        return False


needs_searches = pytest.mark.skipif(not _have_searches(),
                                    reason="no genetic search on disk")


@needs_searches
def test_the_folder_holds_exactly_the_three_pages(tmp_path):
    from sp500lab.cli import main

    out = tmp_path / "ga"
    assert main(["report", "genetic", "-o", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(Q.GENETIC_PAGES.values())


@needs_searches
def test_the_lab_reads_real_checkpoints_without_running_a_search():
    lab = Q.genetic_lab()
    assert lab["searches"], "there are checkpoints on disk but none were read"
    for s in lab["searches"]:
        assert s["preset"] in lab["presets"]
        assert len(s["history"]) > 0
        assert s["n_population"] > 0
        assert s["kind"] in ("features", "families")
        assert s["objective"]["fold_scheme"] in ("contiguous", "random")
        # Every active weight is outside the dead zone, by construction.
        assert all(abs(w) >= lab["genome"]["dead_zone"] for _, w in s["weights"])
        if s["ensemble"]:
            assert s["deliverable"] == s["ensemble"]["name"]
            assert s["ensemble"]["size"] == len(s["ensemble"]["family_usage"]) or \
                s["ensemble"]["size"] >= max(s["ensemble"]["family_usage"].values())
    assert len(lab["families"]) == 9
    assert set(lab["family_presets"]) == {"families", "families-price"}
    assert lab["cut"] and all(names for _, names in lab["cut"])


# --------------------------------------------------------------------------
# The 2026-09 redesign: families, the robust objective, the ensembles
# --------------------------------------------------------------------------

def _lab_kwargs() -> dict:
    """The family material the features page takes, read off the real genome."""
    from sp500lab.strategies.genome import (CUT_FEATURES, FAMILIES, FAMILY_PRESETS,
                                            PRESET_MIN_DATE, all_presets, preset_kind)
    return {
        "families": [{"name": f.name, "label": f.label, "story": f.story,
                      "reference": f.reference, "members": list(f.members),
                      "presets": [n for n, fp in FAMILY_PRESETS.items()
                                  if f.name in fp.families]} for f in FAMILIES],
        "family_presets": {n: {"families": list(fp.families),
                               "max_active": fp.max_active, "min_date": fp.min_date,
                               "note": fp.note} for n, fp in FAMILY_PRESETS.items()},
        "cut": [(r, list(names)) for r, names in CUT_FEATURES],
        "min_dates": dict(PRESET_MIN_DATE),
        "preset_kinds": {n: preset_kind(n) for n in all_presets()},
    }


def test_the_features_page_tells_every_family_story_with_its_signs():
    report = GV.features_report({"families": ("mom_12_1",)}, [], _catalog,
                                hrefs=_hrefs(), **_lab_kwargs())
    table = next(t for t in _tables(report) if t.columns[0] == "family")
    labels = [row[0].text for row in table.rows]
    assert len(labels) == 9 and "Quality" in labels and "Low risk" in labels
    members = {row[0].text: row[2].text for row in table.rows}
    assert "−vol_126d" in members["Low risk"]
    assert "+gross_profitability" in members["Quality"] and "−accruals" in members["Quality"]
    text = _text(report)
    assert "at most 3" in text.lower() or "At most 3" in text
    assert "What was cut, and why" in text


def test_the_features_page_says_what_was_cut_and_why():
    report = GV.features_report({"families": ("mom_12_1",)}, [], _catalog,
                                hrefs=_hrefs(), **_lab_kwargs())
    cut = next(t for t in _tables(report) if t.columns == ["reason", "features", "which"])
    listed = " ".join(row[2].text for row in cut.rows)
    assert "mom_on_12_1" in listed and "restatement_rate" in listed
    assert "log_market_cap" in listed


def test_the_features_page_reports_family_convergence_for_a_family_search():
    s = _family_search()
    report = GV.features_report({"families": ("vol_126d",)}, [s], _catalog,
                                hrefs=_hrefs(), **_lab_kwargs())
    table = next(t for t in _tables(report) if t.columns[0] == "family"
                 and "champion" in t.columns[1])
    by_name = {row[0].text: row for row in table.rows}
    assert by_name["Quality"][1].text == "0.90"
    assert by_name["Quality"][2].text == "91.67%"          # 55 of 60
    assert by_name["Momentum"][1].text == "not backed"


def test_the_methodology_page_lists_five_defences_and_the_ensemble():
    report = GV.methodology_report(_genome(), [_family_search()], hrefs=_hrefs())
    text = _text(report)
    assert "5." in text and "ensemble, not the champion" in text
    assert "worst quarter" in text.lower()
    assert "costs inside the fitness" in text
    assert "What a search hands on" in text
    assert "1 of 1 searches on disk built an ensemble" in text


def test_the_methodology_page_names_the_objective_defaults():
    report = GV.methodology_report(_genome(), [], hrefs=_hrefs())
    stats = [s for sec in report.sections for b in sec.blocks
             if isinstance(b, StatRow) for s in b.stats]
    by_label = {s.label: s for s in stats}
    assert by_label["charged at"].value == "pessimistic"
    assert by_label["aggregate"].value == "quantile"
    assert by_label["sub-periods"].value == "12"
    assert "random" in by_label["sub-periods"].note


def test_a_searches_page_shows_the_ensemble_beside_the_champion():
    s = _family_search(ensemble_forward=DECAYED)
    report = GV.searches_report([s], [], hrefs=_hrefs())
    titles = [sec.title for sec in report.sections]
    assert "ga-fam: the champion" in titles
    assert "ga-fam: the ensemble it hands on" in titles
    text = _text(report)
    assert "ensemble of 30" in text
    assert "pooled across 3 seeds" in text
    # The headline verdict is the ENSEMBLE's, because that is what was tested.
    headline = next(t for t in _tables(report) if "hands on" in t.columns)
    row = headline.rows[0]
    assert row[2].text == "ensemble of 30"
    assert row[-1].text == "DECAYED"


def test_an_untested_ensemble_says_so():
    report = GV.searches_report([_family_search()], [], hrefs=_hrefs())
    assert "This ensemble has not been forward-tested." in _text(report)


def test_the_settings_table_carries_the_objective():
    report = GV.searches_report([_search(), _family_search()], [], hrefs=_hrefs())
    table = next(t for t in _tables(report) if t.columns[0] == "")
    rows = {row[0].text: [c.text for c in row[1:]] for row in table.rows}
    assert rows["sub-periods"] == ["contiguous", "random"]
    assert rows["aggregate"] == ["mean_minus_std", "quantile"]
    assert rows["charged at"] == ["realistic", "pessimistic"]
    assert rows["seeds run"] == ["1", "3"]
    assert rows["ensemble size"][1] == "30"


def test_a_pooled_search_draws_one_training_line_per_seed():
    report = GV.searches_report([_family_search()], [], hrefs=_hrefs())
    chart = next(b for sec in report.sections for b in sec.blocks
                 if isinstance(b, LineChart) and b.title == "Best fitness by generation")
    assert [ser.name for ser in chart.series] == ["ga-fam s0", "ga-fam s1", "ga-fam s2"]
    assert all(len(ser) == 10 for ser in chart.series)


def test_decay_is_counted_on_whatever_the_search_handed_over():
    searches = [_search("a", forward=DECAYED), _family_search("b", ensemble_forward=DECAYED),
                _family_search("c")]
    report = GV.methodology_report(_genome(), searches, hrefs=_hrefs())
    assert "2 of 2 winners tested out of sample DECAYED" in _text(report)
