"""Every competitor honours the one contract the engine offers.

The engine cannot tell the families apart, which is the point of the competition - and
it means a broken strategy fails the same way as a working one: quietly, with a curve.
So the roster is checked as a roster. The offline half proves what can be proved
without data: every name resolves, describes itself, asks only for features that
exist, and sits in exactly one group. The data-backed half runs `backtest/accept.py`
checks 6 and 7 - every strategy runs, and none of them changes its mind when the
future is deleted.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from sp500lab.backtest import accept
from sp500lab.backtest.portfolio import WEIGHTING_SCHEMES
from sp500lab.backtest.strategy import BaseStrategy, get_strategy, list_strategies
from sp500lab.features.catalog import FEATURE_DOCS
from sp500lab.strategies import GROUPS
from sp500lab.strategies.genome import PRESET_MIN_DATE, PRESETS
from sp500lab.timing.strategies import TIMING_GROUPS, get_timing_strategy

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ROSTER = accept.contract_roster(include_learned=True)


# ------------------------------------------------------------------ the roster

def test_every_group_member_is_registered_and_answers_to_its_name():
    registered = set(list_strategies())
    for group, names in GROUPS.items():
        for name in names:
            assert name in registered, f"{group}: {name} is not registered"
            assert get_strategy(name).name == name


def test_all_is_exactly_the_union_of_the_competition_groups():
    expected = [n for g in ("baselines", "alpha", "frontier", "learned", "evolved")
                for n in GROUPS[g]]
    assert list(GROUPS["all"]) == expected
    assert "my_first_idea" not in GROUPS["all"], "custom stays out until promoted"


def test_no_strategy_sits_in_two_competition_groups():
    seen: dict[str, str] = {}
    for group in ("baselines", "alpha", "frontier", "learned", "evolved", "custom"):
        for name in GROUPS[group]:
            assert name not in seen, f"{name} is in both {seen[name]} and {group}"
            seen[name] = group


def test_the_contract_roster_covers_every_competitor():
    assert set(ROSTER) >= set(GROUPS["all"]) - {"spy_buy_hold"}
    assert "spy_buy_hold" not in ROSTER, "the benchmark is a column, not a row"
    assert len(ROSTER) == len(set(ROSTER))


# ------------------------------------------------------------- self-description

@pytest.mark.parametrize("name", ROSTER)
def test_a_strategy_describes_itself_in_json(name):
    """The experiment log stores describe(); anything unserialisable is lost silently."""
    strat = get_strategy(name)
    d = strat.describe() if hasattr(strat, "describe") else {"name": strat.name}
    text = json.dumps(d)
    assert d["name"] == name and name in text


@pytest.mark.parametrize("name", ROSTER)
def test_declared_inputs_are_well_formed(name):
    strat = get_strategy(name)
    # Five years of sessions is the most any model here trains on; beyond that a
    # strategy would sit in cash for a third of the research window.
    warmup = getattr(strat, "warmup", 0)
    assert isinstance(warmup, (int, np.integer)) and 0 <= warmup <= 1300, warmup
    min_date = getattr(strat, "min_date", "") or ""
    assert min_date == "" or _DATE.match(min_date), min_date
    if isinstance(strat, BaseStrategy):
        assert callable(strat.target_weights)


@pytest.mark.parametrize("name", ROSTER)
def test_every_required_feature_exists_and_is_documented(name):
    """A strategy asking for a feature nobody computes fails at run time with a clear
    message - but only if it asks for it by a name the catalog knows."""
    strat = get_strategy(name)
    for f in getattr(strat, "requires_features", ()) or ():
        assert f in FEATURE_DOCS, f"{name} reads {f!r}, which is not a catalogued feature"


@pytest.mark.parametrize("name", ROSTER)
def test_construction_is_inside_the_mandate(name):
    strat = get_strategy(name)
    c = getattr(strat, "construction", None)
    if c is None:
        return
    assert c.weighting in WEIGHTING_SCHEMES
    if getattr(c, "top_k", None) is not None:
        assert c.top_k > 0
    if getattr(c, "max_weight", None) is not None:
        assert 0 < c.max_weight <= 1.0


# ------------------------------------------------------------------- presets

def test_genome_presets_are_frozen():
    """ADR-038: an evolved winner is a float vector decoded BY POSITION against its
    preset's feature tuple. Reordering or extending a preset silently re-decodes every
    checkpoint on disk into a different strategy with the same name. These pins fail
    the moment anyone touches one; the fix is a new preset, never an edit."""
    assert len(PRESETS["price"]) == 13 and PRESETS["price"][0] == "mom_12_1"
    assert PRESETS["price"][-1] == "div_yield"
    assert len(PRESETS["full"]) == 23 and PRESETS["full"][:13] == PRESETS["price"]
    assert len(PRESETS["night"]) == 17 and PRESETS["night"][:13] == PRESETS["price"]
    assert PRESETS["night"][13:] == ("mom_on_12_1", "mom_id_12_1", "on_minus_id_252d",
                                     "div_due_1m")
    for name, feats in PRESETS.items():
        assert isinstance(feats, tuple) and len(set(feats)) == len(feats), name
        assert set(feats) <= set(FEATURE_DOCS), f"{name} names an uncatalogued feature"
        assert name in PRESET_MIN_DATE


# ---------------------------------------------------------------- timing rules

def test_every_timing_rule_is_registered_and_named():
    assert set(TIMING_GROUPS["timing"]) <= set(TIMING_GROUPS["all"])
    assert "tm_buy_hold" in TIMING_GROUPS["all"]
    for name in TIMING_GROUPS["all"]:
        strat = get_timing_strategy(name)
        assert getattr(strat, "name", name) == name
        assert callable(strat.legs)


# ------------------------------------------------------------- the real panel

def _have_panel() -> bool:
    try:
        from sp500lab.storage import silver_exists
        return silver_exists("market/daily_bars_adjusted") and silver_exists(
            "reference/trading_calendar")
    except Exception:  # noqa: BLE001
        return False


needs_data = pytest.mark.skipif(not _have_panel(), reason="silver layer not built")


@pytest.fixture(scope="module")
def panel():
    from sp500lab.backtest.panel import build_panel
    return build_panel()


@needs_data
def test_every_strategy_runs_and_produces_a_portfolio(panel):
    checks = accept.check_strategy_contract(panel)
    failed = [c.line() for c in checks if not c.passed]
    assert not failed, "\n".join(failed)
    assert len(checks) == len(accept.contract_roster())


@needs_data
def test_no_strategy_changes_its_mind_when_the_future_is_deleted(panel):
    """Check 7. A strategy whose weights differ between the full panel and one truncated
    just past the window is reading past its as-of date - through on_start, through a
    centred window, or through a feature joined on the wrong date."""
    checks = accept.check_no_lookahead(panel)
    failed = [c.line() for c in checks if not c.passed]
    assert not failed, "\n".join(failed)


@needs_data
def test_timing_rules_emit_valid_legs(panel):
    """Both leg vectors are boolean, session-aligned, and buy-and-hold is all True."""
    from sp500lab.timing.data import load_timing_data
    data = load_timing_data()
    n = len(data.dates) if hasattr(data, "dates") else None
    for name in TIMING_GROUPS["all"]:
        on, intra = get_timing_strategy(name).legs(data)
        assert on.dtype == bool and intra.dtype == bool, name
        assert on.shape == intra.shape, name
        if n is not None:
            assert len(on) == n, name
        if name == "tm_buy_hold":
            assert on.all() and intra.all()
        elif name == "tm_overnight":
            assert on.any() and not intra.any()
        elif name == "tm_intraday":
            assert intra.any() and not on.any()
