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
from sp500lab.reporting import queries as Q
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


def test_the_roster_table_carries_the_headline_statistics():
    """The index has one table, and it has to be enough to decide which page to open."""
    table = T.strategy_roster([_spec("a", ann_vol=0.15, max_drawdown=-0.3)])
    assert {"CAGR", "vol", "Sharpe", "maxDD", "vs index", "deflated"} <= set(table.columns)
    row = dict(zip(table.columns, table.rows[0]))
    assert row["vol"].text == "15.00%"
    assert row["maxDD"].text == "-30.00%"


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
    col = T.strategy_roster([_spec("a")]).columns.index("vs index")
    good = T.strategy_roster([_spec("a", d_sharpe=0.10)]).rows[0][col]
    bad = T.strategy_roster([_spec("b", d_sharpe=-0.10)]).rows[0][col]
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
# The roster, and the folder (ADR-045)
# --------------------------------------------------------------------------

def test_the_roster_is_every_builtin_plus_the_custom_group():
    from sp500lab.strategies import GROUPS
    names = Q.roster("all")
    assert set(GROUPS["all"]) <= set(names)
    assert set(GROUPS["custom"]) <= set(names)
    assert len(names) == len(set(names))


def test_a_named_group_or_a_list_is_taken_literally():
    from sp500lab.strategies import GROUPS
    assert Q.roster("baselines") == tuple(GROUPS["baselines"])
    assert Q.roster("low_vol, cash") == ("low_vol", "cash")


def test_ga_winners_are_capped_and_ranked_by_research_sharpe(monkeypatch):
    found = [{"name": "ga-a-best", "study": "ga-a"}, {"name": "ga-b-best", "study": "ga-b"},
             {"name": "ga-c-best", "study": "ga-c"}, {"name": "ga-d-best", "study": "ga-d"}]
    sharpe = {"ga-a": 0.5, "ga-b": 1.2, "ga-c": 0.9, "ga-d": float("-inf")}
    monkeypatch.setattr(Q, "evolved_winners", lambda: list(found))
    monkeypatch.setattr(Q, "_best_sharpe", lambda study: sharpe[study])
    assert [w["name"] for w in Q.ga_winners(3)] == ["ga-b-best", "ga-c-best", "ga-a-best"]
    assert Q.ga_winners(0) == []
    assert len(Q.ga_winners(None)) == 4


def test_the_forward_roster_only_names_what_was_tested(monkeypatch):
    """The calendar rules were forward-tested; they are not on the roster, so no page."""
    monkeypatch.setattr(Q, "ga_winners", lambda limit=None: [{"name": "ga-x-best"}])
    records = pd.DataFrame({"strategy": ["low_vol", "tm_overnight", "ga-x-best"]})
    assert Q.forward_roster(records) == ["low_vol", "ga-x-best"]


def test_a_rebuild_removes_pages_an_earlier_build_left_behind(tmp_path):
    from sp500lab.reporting.cli import _prune
    (tmp_path / "stale.html").write_text("x")
    (tmp_path / "keep.html").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    _prune(tmp_path, [tmp_path / "keep.html"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.html", "notes.txt"]


def test_report_all_is_still_an_alias_for_the_backtest_set():
    from sp500lab.cli import build_parser
    from sp500lab.reporting.cli import cmd_backtest
    assert build_parser().parse_args(["report", "all"]).func is cmd_backtest
    assert build_parser().parse_args(["report", "backtest"]).func is cmd_backtest


# --------------------------------------------------------------------------
# The calendar set (ADR-047)
#
# A separate folder because these rules are a separate engine: one instrument, all-in
# or cash, on a fixed schedule. The tests here are about the two claims that separation
# is FOR - that a calendar rule is never ranked as if it were a stock picker, and that
# its sample size is its round trips rather than its sessions.
# --------------------------------------------------------------------------

class _FakeRule:
    """A calendar rule with legs chosen by hand, so the counts can be done by eye."""

    def __init__(self, overnight, intraday):
        self._legs = (np.asarray(overnight, dtype=bool),
                      np.asarray(intraday, dtype=bool))

    def legs(self, data):
        return self._legs


def _schedule(overnight, intraday):
    return Q.rule_schedule(_FakeRule(overnight, intraday), None, 0, len(overnight))


def test_a_rule_that_re_enters_every_night_counts_every_night():
    """Overnight-only: each night is separated by an unheld day, so entries = nights."""
    sched = _schedule([True] * 4, [False] * 4)
    assert sched["episodes"] == 4
    assert sched["sessions"] == 4
    assert sched["legs"] == "overnight only"


def test_a_rule_that_holds_through_is_one_entry_not_many():
    """The whole point of the column: four sessions held in a row is ONE observation."""
    sched = _schedule([True, True, True, False], [True, True, True, True])
    assert sched["episodes"] == 1
    assert sched["sessions"] == 4


def test_two_separate_holds_are_two_entries():
    sched = _schedule([True, False, False, True, False],
                      [True, True, False, True, True])
    assert sched["episodes"] == 2


def test_time_in_market_counts_half_sessions():
    """An overnight rule owns half the clock. Reading it as 100% invited the wrong
    comparison against buy-and-hold, which owns all of it."""
    assert _schedule([True] * 10, [False] * 10)["exposure"] == pytest.approx(0.5)
    assert _schedule([True] * 10, [True] * 10)["exposure"] == pytest.approx(1.0)


def _rule(name, **kw):
    base = {"name": name, "href": f"{name.replace('_', '-')}.html",
            "claim": "Holds the market on some days and not others.",
            "paragraphs": ["Holds the market on some days and not others."],
            "explain": [], "exposure": "50%", "window": "2007-04–2021-12",
            "schedule": {"sessions": 3715, "of_sessions": 3715, "exposure": 0.5,
                         "episodes": 3715, "legs": "overnight only"},
            "settings": {c: {"cagr": 0.05, "sharpe": 0.5, "maxdd": -0.2,
                             "turnover": None, "cost_drag": 0.01}
                         for c in ("optimistic", "realistic", "pessimistic")},
            "forward": None, "is_benchmark": False, "d_sharpe": 0.1}
    return base | kw


def test_the_benchmark_is_the_bar_and_is_never_ranked_against_itself():
    from sp500lab.reporting.timing_views import _ranked

    rules = [_rule("tm_weekend", d_sharpe=-0.5),
             _rule("tm_buy_hold", is_benchmark=True, d_sharpe=0.0),
             _rule("tm_overnight", d_sharpe=0.1)]
    assert [r["name"] for r in _ranked(rules)] == ["tm_buy_hold", "tm_overnight",
                                                   "tm_weekend"]


def test_every_rule_card_links_to_the_page_the_set_writes():
    """The index and the pages are written by different loops and must agree."""
    from sp500lab.reporting.specs import LinkGrid
    from sp500lab.reporting.timing_views import _rules_section

    rules = [_rule("tm_overnight"), _rule("tm_weekend")]
    grid = next(b for b in _rules_section(rules).blocks if isinstance(b, LinkGrid))
    assert {c.href for c in grid.cards} == {"tm-overnight.html", "tm-weekend.html"}
    assert all(c.blurb for c in grid.cards)


def test_a_rule_page_says_its_sample_is_entries_not_sessions():
    from sp500lab.reporting.timing_views import _entries_note

    long_holds = _entries_note(_rule("tm_sell_in_may"), 16, 1806, 14.7)
    assert "16 entries" in long_holds.text and "1,806 sessions" in long_holds.text
    assert long_holds.level == "warn", "a 16-event sample has to be flagged, not noted"

    # A rule whose entries and sessions coincide must NOT claim they differ.
    nightly = _entries_note(_rule("tm_overnight"), 3715, 3715, 14.7)
    assert "is not the" not in nightly.text


def test_a_rule_with_no_forward_record_says_so_rather_than_going_quiet():
    from sp500lab.reporting.specs import Note
    from sp500lab.reporting.timing_views import _rule_forward

    sections = _rule_forward(_rule("tm_overnight"), None)
    assert len(sections) == 1
    note = next(b for b in sections[0].blocks if isinstance(b, Note))
    assert note.level == "warn"
    assert "never sealed" in note.text or "Never tested" in note.title


def test_the_forward_index_points_at_the_set_that_does_show_the_calendar_rules():
    """Naming nine hidden candidates without saying where they are was the bug."""
    from sp500lab.reporting.forward_views import _hidden_note

    note = _hidden_note(["tm_overnight", "tm_weekend"], "../timing/index.html")
    assert "tm_overnight" in note.text
    assert "../timing/index.html" in note.text
    assert "multiple-testing bar" in note.text, "a hidden candidate still counts"
    assert _hidden_note([], "../timing/index.html") is None


def test_a_calendar_rule_is_never_rendered_with_an_empty_claim():
    """`report forward` can be handed a tm_ record; a blank claim would be a silent lie
    about a strategy that has a perfectly good docstring."""
    assert "market is closed" in Q.claim_for("tm_overnight").lower()
    assert Q.claim_for("no_such_strategy_at_all") == ""


def test_the_calendar_rules_are_not_on_the_monthly_roster():
    """If they ever are, they get sorted into a scoreboard of stock pickers."""
    names = set(Q.roster("all"))
    assert not [n for n in names if n.startswith("tm_")]


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
def test_the_backtest_folder_holds_an_index_and_one_page_per_algorithm(tmp_path):
    """Two kinds of file and nothing else - the whole point of ADR-045."""
    from sp500lab.cli import main

    out = tmp_path / "bt"
    rc = main(["report", "backtest", "cash,equal_weight", "--start", "2015-01-01",
               "--no-log", "--no-evolved", "-o", str(out)])
    assert rc == 0
    assert sorted(p.name for p in out.iterdir()) == ["cash.html", "equal-weight.html",
                                                     "index.html"]
    index = (out / "index.html").read_text(encoding="utf-8")
    assert 'href="cash.html"' in index and 'href="equal-weight.html"' in index


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
def test_the_timing_folder_holds_an_index_and_one_page_per_calendar_rule(tmp_path):
    """The ADR-047 shape, and the link between the two kinds of file."""
    from sp500lab.cli import main
    from sp500lab.timing.strategies import TIMING_GROUPS

    out = tmp_path / "cal"
    assert main(["report", "timing", "-o", str(out)]) == 0
    written = sorted(p.name for p in out.iterdir())
    expected = sorted(["index.html"]
                      + [f"{n.replace('_', '-')}.html" for n in TIMING_GROUPS["all"]])
    assert written == expected

    index = (out / "index.html").read_text(encoding="utf-8")
    for name in TIMING_GROUPS["all"]:
        assert f'href="{name.replace("_", "-")}.html"' in index, name


@needs_data
def test_a_calendar_rule_page_carries_both_windows_on_one_page():
    """The reason this is a set and not two: a rule's research and forward numbers are
    one story, and `tm_weekend`'s story is that the second refuted the first."""
    from sp500lab.reporting.cli import calendar_lab_pages

    _, pages = calendar_lab_pages()
    by_name = {name: report for name, _, report in pages}
    report = by_name["tm_weekend"]
    titles = [s.title for s in report.sections]
    assert "The research window" in titles
    assert "Out of sample: 2022 onward" in titles
    assert "Prediction against outcome" in titles, "reused from the forward set"
    assert titles[-2] == "What would make you distrust this"
    assert "FAILED" in report.subtitle


@needs_data
def test_the_entries_column_reproduces_the_documented_sample_sizes():
    """docs/TIMING.md's `independent sample` column is prose. This is where it comes
    from, and the two must not drift: sell-in-May is ~15 cycles, not ~1,800 sessions."""
    from sp500lab.reporting.cli import calendar_lab_pages

    index, _ = calendar_lab_pages()
    table = next(b for s in index.sections for b in s.blocks
                 if isinstance(b, TableBlock) and "entries" in b.table.columns).table
    col = table.columns.index("entries")
    counts = {r[0].text: int(r[col].text.replace(",", "")) for r in table.rows}
    assert 3_500 < counts["tm_overnight"] < 3_900
    assert 700 < counts["tm_weekend"] < 800
    assert 150 < counts["tm_turn_of_month"] < 200
    assert counts["tm_sell_in_may"] < 25


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
