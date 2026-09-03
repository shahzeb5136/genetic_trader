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
        "window": "2007-04 → 2021-12",
        "forward": forward,
    }


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
    from sp500lab.strategies.genome import DEAD_ZONE, PRESETS, alpha_genome

    g = _genome()
    assert g["dead_zone"] == DEAD_ZONE
    assert set(g["sizes"]) == set(PRESETS)
    for preset in PRESETS:
        assert g["sizes"][preset] == len(alpha_genome(preset))
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
        # Every active weight is outside the dead zone, by construction.
        assert all(abs(w) >= lab["genome"]["dead_zone"] for _, w in s["weights"])
