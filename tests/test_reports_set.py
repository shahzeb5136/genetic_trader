"""The report set: is the catalogue complete, and does a page say what it means?

These reports are meant to be the documentation for somebody who will not open the code,
which changes what is worth testing. A chart that renders is not the point; a chart
labelled "the real drawdown" that was computed from a monthly curve IS the point, and so
is a scoreboard that sorts an evolved winner above a hand-written one without saying which
is which.

So the tests here are about claims:

  * every feature in the built panel has a catalogue entry, or the feature report is a
    directory listing
  * a searched strategy is labelled searched and a written one is not
  * the honest-summary note counts what it says it counts
  * the trades download shrinks rather than producing an unopenable page

Everything asserts on the spec objects, never on markup - the same split that lets
`views.py` be rewritten without changing what any of it means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp500lab.features.catalog import FAMILIES, FEATURE_DOCS, by_family, describe
from sp500lab.reporting import tables as T
from sp500lab.reporting.specs import Download, LinkGrid, Note, TableBlock
from sp500lab.reporting.views import _headline_claim, _trades_download, index_report


# --------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------

def test_every_documented_feature_has_a_known_family():
    for name, doc in FEATURE_DOCS.items():
        assert doc.family in FAMILIES, f"{name} is in family {doc.family!r}"


def test_every_documented_feature_says_what_it_is_and_which_end_is_good():
    for name, doc in FEATURE_DOCS.items():
        assert len(doc.what) > 15, f"{name} has no real description"
        assert len(doc.reading) > 5, f"{name} does not say which end is good"
        assert doc.source, f"{name} does not say where it came from"


def test_by_family_returns_every_feature_exactly_once():
    seen = [n for names in by_family().values() for n in names]
    assert sorted(seen) == sorted(FEATURE_DOCS)


def test_an_unknown_feature_gets_a_loud_placeholder():
    """Silence would let a feature quietly appear in a report with no explanation."""
    doc = describe("no_such_feature")
    assert doc.family == "Undocumented"
    assert "catalog.py" in doc.what


# --------------------------------------------------------------------------
# The scoreboard's central distinction
# --------------------------------------------------------------------------

def _spec(name, **kw):
    base = {"name": name, "href": f"strategy-{name}.html", "claim": "Does a thing.",
            "window": "2007-04–2021-12", "cagr": 0.10, "sharpe": 0.6,
            "d_sharpe": 0.05, "evolved": False, "n_trials": None,
            "deflated_sharpe": None}
    return base | kw


def test_an_evolved_strategy_is_labelled_evolved():
    table = T.strategy_roster([_spec("ga-1-best", evolved=True, n_trials=1403,
                                     deflated_sharpe=0.99)])
    found_by = table.rows[0][2].text
    assert "evolved" in found_by and "1,403" in found_by


def test_a_written_strategy_is_never_labelled_searched():
    """The trap: every report run is logged into one study, so `n_trials` is large for
    a hand-written strategy too. Reading that as 'this was searched' would be a lie in
    the opposite direction from the one the column exists to prevent."""
    table = T.strategy_roster([_spec("low_vol", evolved=False, n_trials=60,
                                     deflated_sharpe=0.80)])
    assert table.rows[0][2].text == "written"


def test_the_deflated_sharpe_is_shown_for_everything_that_has_one():
    table = T.strategy_roster([_spec("x", n_trials=60, deflated_sharpe=0.8)])
    assert "0.8" in table.rows[0][-1].text


def test_a_long_claim_is_trimmed_rather_than_wrapping_the_table():
    long_claim = "Evolved, not written. " + "weight on a feature, " * 30
    table = T.strategy_roster([_spec("ga", claim=long_claim, evolved=True)])
    text = table.rows[0][1].text
    assert len(text) <= 90
    assert text.endswith("…")


def test_beating_the_index_is_emphasised_in_both_directions():
    good = T.strategy_roster([_spec("a", d_sharpe=0.10)]).rows[0][6]
    bad = T.strategy_roster([_spec("b", d_sharpe=-0.10)]).rows[0][6]
    assert good.emphasis == "good" and bad.emphasis == "bad"
    assert good.text.startswith("+")


# --------------------------------------------------------------------------
# The index
# --------------------------------------------------------------------------

def test_the_index_counts_what_it_claims_to_count():
    specs = [_spec("a", d_sharpe=0.2), _spec("b", d_sharpe=-0.2),
             _spec("c", d_sharpe=0.01)]
    report = index_report(specs)
    note = next(b for b in report.section("Start here").blocks
                if isinstance(b, Note) and "beat the index" in b.text)
    assert "2 of 3" in note.text
    assert "a" in note.text and "c" in note.text


def test_the_index_warns_when_a_strategy_was_evolved():
    report = index_report([_spec("ga-1-best", evolved=True), _spec("low_vol")])
    notes = [b for b in report.section("Start here").blocks if isinstance(b, Note)]
    warning = next(n for n in notes if n.level == "warn")
    assert "ga-1-best" in warning.text
    assert "MAXIMUM" in warning.text, "it must say why a searched Sharpe is inflated"
    assert "holdout" in warning.text


def test_the_index_says_nothing_about_searching_when_nothing_was_searched():
    report = index_report([_spec("low_vol"), _spec("momentum_12_1")])
    assert not [b for b in report.section("Start here").blocks
                if isinstance(b, Note) and b.level == "warn"]


def test_every_card_links_somewhere_real():
    specs = [_spec("a"), _spec("b")]
    report = index_report(specs, extra_cards=[{"title": "Features",
                                               "href": "features.html"}])
    grids = [b for s in report.sections for b in s.blocks if isinstance(b, LinkGrid)]
    hrefs = {c.href for g in grids for c in g.cards}
    assert {"strategy-a.html", "strategy-b.html", "features.html"} <= hrefs
    assert "" not in hrefs


def test_the_scoreboard_is_sorted_by_the_column_it_says_it_is():
    specs = [_spec("worst", d_sharpe=-0.3), _spec("best", d_sharpe=0.3),
             _spec("middle", d_sharpe=0.0)]
    report = index_report(sorted(specs, key=lambda s: -s["d_sharpe"]))
    block = next(b for b in report.section("The scoreboard").blocks
                 if isinstance(b, TableBlock))
    assert [r[0].text for r in block.table.rows] == ["best", "middle", "worst"]


# --------------------------------------------------------------------------
# Subtitles and downloads
# --------------------------------------------------------------------------

def test_a_useless_first_sentence_borrows_the_second():
    """'Evolved, not written.' is a true first sentence and a useless subtitle."""
    out = _headline_claim("Evolved, not written. The best of 60 individuals in a search.")
    assert "best of 60" in out


def test_a_full_first_sentence_is_left_alone():
    claim = ("Lowest trailing realised volatility, ranked across the point-in-time "
             "universe. Negated, because low is good here.")
    assert _headline_claim(claim).startswith("Lowest trailing realised volatility")
    assert "Negated" not in _headline_claim(claim)


class _FakeResult:
    strategy = "demo"

    def __init__(self, n):
        self.trades = pd.DataFrame({
            "date": ["2020-01-02"] * n, "ticker": ["AAA"] * n,
            "notional": np.ones(n), "cost": np.ones(n)})


def test_a_small_ledger_is_embedded_whole():
    block = _trades_download(_FakeResult(50), _FakeResult(50).trades)
    assert isinstance(block, Download)
    assert block.content.count("\n") >= 50
    assert "every buy and sell" in block.label


def test_a_huge_ledger_is_capped_and_says_where_the_rest_is():
    big = _FakeResult(60_000)
    block = _trades_download(big, big.trades, full_csv_href="trades/demo.csv")
    assert len(block.content) <= 4_000_000
    assert block.content.count("\n") < 60_000
    assert "trades/demo.csv" in block.note
    assert "60,000" in block.note, "it must say how many orders it is NOT showing"


def test_the_cap_never_produces_an_empty_download():
    huge = _FakeResult(200_000)
    block = _trades_download(huge, huge.trades)
    assert block.content.count("\n") > 100


# --------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------

def _have_data() -> bool:
    try:
        from sp500lab.storage import silver_exists
        return silver_exists("market/daily_bars_adjusted")
    except Exception:                                             # noqa: BLE001
        return False


needs_data = pytest.mark.skipif(not _have_data(), reason="silver layer not built")


@needs_data
def test_every_feature_in_the_built_panel_is_documented():
    """A feature with no catalogue entry turns the feature report into a column list."""
    from sp500lab.features import build_features
    from sp500lab.features.catalog import undocumented

    missing = undocumented(build_features().names)
    assert not missing, f"undocumented features: {missing}"


@needs_data
def test_a_strategy_report_carries_the_strategy_s_own_words():
    """The claim comes from the source, so the report and the code cannot drift."""
    from sp500lab.backtest import run_backtest
    from sp500lab.backtest.strategy import get_strategy
    from sp500lab.reporting import strategy_report
    from sp500lab.reporting.queries import claim_of

    strat = get_strategy("low_vol")
    result = run_backtest(strat, start="2015-01-01", log_run=False)
    report = strategy_report(result, claim=claim_of(strat))

    assert report.title == "low_vol"
    titles = [s.title for s in report.sections]
    assert titles[0] == "What this claims"
    assert titles[-1] == "What would make me distrust this"
    body = " ".join(b.text for s in report.sections for b in s.blocks
                    if isinstance(b, Note))
    assert "volatility" in body.lower()
    assert "coverage" in body.lower(), "coverage must travel with the numbers"


@needs_data
def test_a_feature_report_puts_the_leakage_check_before_the_catalogue():
    """A list of 75 columns means nothing until somebody has shown none read the future."""
    from sp500lab.features import build_features
    from sp500lab.reporting import feature_report

    report = feature_report(build_features(),
                            leakage={"ok": True, "cut_at": "2016-12-30",
                                     "rows_compared": 204, "securities": 628,
                                     "features": [{"feature": "x", "identical": True}],
                                     "failed": []})
    titles = [s.title for s in report.sections]
    assert titles.index("Can any of it be trusted?") < titles.index("Momentum & trend")
    assert "Macro" in titles
