"""Tests for the forward-test report set and the Markdown backend.

The point of the specs-and-views split (ADR-028) is that a report can be checked by
asserting on its CONTENTS rather than on its markup, and these tests are what that buys.
Every one of them builds a `Report` from a synthetic forward store and looks inside it -
no HTML is parsed anywhere in this file.

Two things are worth testing here beyond "it runs":

  * **The editorial guarantees.** `docs/FORWARD_TEST.md` promises that the sample-size
    caveat travels with every page and that `held` is never presented as a pass. Those
    are claims about content, they are the whole reason the reports exist, and a
    refactor that quietly dropped one would look fine.
  * **The second backend.** Markdown exists to prove the views emit no markup. A view
    that started returning HTML in a caption would still render in HTML and would be
    visibly broken here, which is the cheapest possible regression test for the seam.
"""

from __future__ import annotations

import pandas as pd
import pytest

import sp500lab.forward.seal as seal_module
import sp500lab.forward.store as store_module
from sp500lab.reporting import forward_views as FV
from sp500lab.reporting.render import html as html_render
from sp500lab.reporting.render import markdown as md_render
from sp500lab.reporting.specs import Note, Report, StatRow, TableBlock

from test_forward import curve, leg


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A forward store in tmp_path, populated with a small synthetic set."""
    monkeypatch.setattr(store_module, "FORWARD_LOG", tmp_path / "forward.jsonl")
    monkeypatch.setattr(store_module, "FORWARD_CURVE_LOG", tmp_path / "curves.jsonl")
    monkeypatch.setattr(seal_module, "SEAL_LOG", tmp_path / "seals.jsonl")

    spec = [
        # name,            research d_sharpe, forward sharpe, forward cagr, verdict
        ("great",          0.30, 1.20, 0.20, "held"),
        ("faded",          0.40, 0.30, 0.03, "decayed"),
        ("broken",         0.20, -0.40, -0.08, "failed"),
        ("random_weight", -0.05, 0.55, 0.08, "held"),
        ("plodder",       -0.10, 0.50, 0.07, "held"),
    ]
    for name, research_edge, fwd_sharpe, fwd_cagr, verdict in spec:
        _write_one(store_module, name, research_edge, fwd_sharpe, fwd_cagr, verdict)
    return store_module


def _write_one(store, name, research_edge, fwd_sharpe, fwd_cagr, verdict):
    bench_r, bench_f = 0.59, 0.83
    r = leg("research", n_months=175, sharpe=bench_r + research_edge,
            sharpe_monthly=bench_r + research_edge, cagr=0.11,
            bench_sharpe=bench_r, bench_cagr=0.1042,
            start="2007-05-01", end="2021-12-31")
    f = leg("forward", n_months=54, sharpe=fwd_sharpe, sharpe_monthly=fwd_sharpe,
            cagr=fwd_cagr, bench_sharpe=bench_f, bench_cagr=0.1371,
            start="2022-02-01", end="2026-08-26")
    for cost in ("optimistic", "realistic", "pessimistic"):
        fid = store.new_forward_id() + cost[:3]
        store.record(store.ForwardRecord(
            forward_id=fid, logged_at=f"2026-08-28T00:00:0{len(name) % 9}Z",
            batch_id="b", seal_id=f"seal-{name}", seal_mode="declared", strategy=name,
            cost_model=cost, mode="paired", data_end="2026-08-26",
            rationale="a synthetic candidate", n_trials=1400 if name == "great" else 0,
            study="ga-1" if name == "great" else None,
            coverage_min=0.95, coverage_median=0.99, n_rebalances=54, n_orders=600,
            research=r.as_dict(), research_recomputed=r.as_dict(), forward=f.as_dict(),
            comparison=FV.compare(r, f).as_flat_dict(),
            verdict=verdict, verdict_reason=f"synthetic {verdict} reason"))
        store.save_curves(fid, strategy=name, seal_id=f"seal-{name}", cost_model=cost,
                          research={"nav": curve("2007-05-01", 175, 0.008)},
                          forward={"nav": curve("2022-02-01", 54, fwd_cagr / 12),
                                   "benchmark": curve("2022-02-01", 54, 0.011)})


def _text(report: Report) -> str:
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
            else:
                out += [getattr(block, "title", ""), getattr(block, "caption", "")]
    return " ".join(str(o) for o in out)


# ==========================================================================
# The executive summary
# ==========================================================================

def test_the_index_counts_every_verdict(store):
    report = FV.forward_index_report(store.load())
    text = _text(report)
    assert "5 candidates were carried out of the research window" in text
    for word in ("held", "decayed", "failed"):
        assert word in text


def test_the_sample_size_caveat_is_on_the_page(store):
    """docs/FORWARD_TEST.md promises this travels with every number. Hold it here."""
    report = FV.forward_index_report(store.load())
    text = _text(report)
    assert "54 months of forward data" in text
    assert "REFUTE" in text and "cannot CONFIRM" in text


def test_held_is_never_presented_as_a_pass(store):
    report = FV.forward_index_report(store.load())
    text = _text(report)
    assert "not refuted" in text


def test_the_null_hypothesis_is_used_as_the_calibration(store):
    """`random_weight` holding is the sharpest available warning. Do not lose it."""
    report = FV.forward_index_report(store.load())
    text = _text(report)
    assert "random_weight" in text
    assert "encodes no forecast" in text


def test_the_multiple_testing_bar_is_reported(store):
    report = FV.forward_index_report(store.load())
    assert "luckiest of 5" in _text(report)


def test_the_scoreboard_ranks_on_the_index_relative_column(store):
    """Ranking on the raw forward Sharpe would rank the market, not the strategies."""
    report = FV.forward_index_report(store.load())
    table = next(b.table for s in report.sections for b in s.blocks
                 if isinstance(b, TableBlock))
    assert [row[0].text for row in table.rows][0] == "great"


def test_the_index_says_the_holdout_is_spent(store):
    assert "holdout is spent" in _text(FV.forward_index_report(store.load())).lower()


def test_an_empty_store_produces_a_page_rather_than_an_exception():
    report = FV.forward_index_report(pd.DataFrame(columns=store_module._EMPTY_COLUMNS))
    assert "Nothing to report" in _text(report)


# ==========================================================================
# One candidate
# ==========================================================================

def test_a_strategy_report_carries_both_legs_and_the_verdict(store):
    records = store.load()
    report = FV.forward_strategy_report(records[records["strategy"] == "faded"])
    text = _text(report)
    assert "DECAYED" in text
    assert "research 2007-2021" in text and "forward 2022-2026" in text
    assert "Synthetic decayed reason" in text        # capitalised into a sentence


def test_a_strategy_report_shows_all_three_cost_settings(store):
    records = store.load()
    report = FV.forward_strategy_report(records[records["strategy"] == "great"])
    section = report.section("Under all three cost settings")
    table = next(b.table for b in section.blocks if isinstance(b, TableBlock))
    assert {row[0].text for row in table.rows} == {"optimistic", "realistic",
                                                   "pessimistic"}


def test_a_searched_candidate_carries_its_trial_count(store):
    records = store.load()
    report = FV.forward_strategy_report(records[records["strategy"] == "great"])
    assert "1,400" in _text(report)


def test_an_unattributed_candidate_says_its_search_is_unknown(store):
    records = store.load()
    report = FV.forward_strategy_report(records[records["strategy"] == "plodder"])
    assert "Search context unknown" in _text(report)


def test_every_named_check_reaches_the_page(store):
    records = store.load()
    report = FV.forward_strategy_report(records[records["strategy"] == "faded"])
    section = report.section("The checks")
    table = next(b.table for b in section.blocks if isinstance(b, TableBlock))
    names = {row[0].text for row in table.rows}
    assert {"enough_data", "no_ruin", "made_money", "beat_benchmark",
            "decay_within_noise", "turnover_held"} <= names


def test_the_change_column_keeps_each_rows_own_unit(store):
    """A CAGR change of -8pp and a Sharpe change of -0.9 must not look alike."""
    records = store.load()
    report = FV.forward_strategy_report(records[records["strategy"] == "faded"])
    table = next(b.table for b in report.section("Prediction against outcome").blocks
                 if isinstance(b, TableBlock))
    changes = {row[0].text: row[3].text for row in table.rows}
    assert changes["CAGR"].endswith("%")
    assert not changes["Sharpe (monthly)"].endswith("%")


def test_the_annual_table_reaches_the_page(store):
    records = store.load()
    report = FV.forward_strategy_report(records[records["strategy"] == "great"])
    assert report.section("Year by year") is not None
    assert "2022" in _text(report)


# ==========================================================================
# The cross-section
# ==========================================================================

def test_rank_agreement_is_a_spearman_over_the_index_relative_columns(store):
    rows = FV.primary_rows(store.load())
    stats = FV.rank_agreement(rows)
    assert -1.0 <= stats["spearman_d_sharpe"] <= 1.0
    assert 0 <= stats["top5_overlap"] <= 5
    assert stats["n"] == len(rows)


def test_a_perfectly_preserved_ranking_scores_one():
    rows = pd.DataFrame({
        "strategy": list("abcde"),
        "research_d_sharpe": [0.5, 0.4, 0.3, 0.2, 0.1],
        "forward_d_sharpe": [0.9, 0.7, 0.5, 0.3, 0.1],
        "research_cagr": [0.2, 0.18, 0.16, 0.14, 0.12],
        "forward_cagr": [0.2, 0.18, 0.16, 0.14, 0.12],
        "decay_sharpe_monthly": [0.1, 0.1, 0.1, 0.1, 0.1],
    })
    stats = FV.rank_agreement(rows)
    assert stats["spearman_d_sharpe"] == pytest.approx(1.0)
    assert stats["top5_overlap"] == 5


def test_a_reversed_ranking_scores_minus_one():
    rows = pd.DataFrame({
        "strategy": list("abcde"),
        "research_d_sharpe": [0.5, 0.4, 0.3, 0.2, 0.1],
        "forward_d_sharpe": [0.1, 0.3, 0.5, 0.7, 0.9],
        "research_cagr": [0.2, 0.18, 0.16, 0.14, 0.12],
        "forward_cagr": [0.12, 0.14, 0.16, 0.18, 0.2],
        "decay_sharpe_monthly": [-0.1] * 5,
    })
    assert FV.rank_agreement(rows)["spearman_d_sharpe"] == pytest.approx(-1.0)


def test_the_decay_report_refuses_to_correlate_two_points(store):
    rows = store.load().head(3)
    report = FV.forward_decay_report(rows)
    assert "Not enough candidates" in _text(report)


def test_a_near_zero_correlation_is_reported_as_no_information(store):
    report = FV.forward_decay_report(store.load())
    text = _text(report)
    assert "Spearman rank correlation" in text
    assert "indistinguishable from zero" in text


def test_the_decay_report_refuses_to_blame_the_method_alone(store):
    """A low correlation on one 4.6-year regime is not proof research is worthless."""
    assert "unusual" in _text(FV.forward_decay_report(store.load())).lower()


# ==========================================================================
# Honesty
# ==========================================================================

def test_the_honesty_report_leads_with_sample_size(store):
    report = FV.forward_honesty_report(store.load())
    assert report.sections[0].title == "The one that outranks the rest"
    assert "54 months" in _text(report)


def test_the_honesty_report_says_there_is_no_second_holdout(store):
    assert "no second holdout" in _text(FV.forward_honesty_report(store.load())).lower()


# ==========================================================================
# The Markdown backend - the seam, tested
# ==========================================================================

def test_markdown_renders_every_report_without_a_view_change(store):
    records = store.load()
    for report in (FV.forward_index_report(records),
                   FV.forward_decay_report(records),
                   FV.forward_honesty_report(records),
                   FV.forward_strategy_report(
                       records[records["strategy"] == "great"])):
        text = md_render.render(report)
        assert text.startswith("# ")
        assert "|---" in text                      # at least one table survived
        assert "<script" not in text and "<div" not in text


def test_the_two_backends_agree_on_the_numbers(store):
    """Same specs, two renderers. A number in one must be in the other."""
    report = FV.forward_index_report(store.load())
    md = md_render.render(report)
    html = html_render.render(report)
    for token in ("54 months", "random_weight", "great"):
        assert token in md and token in html


def test_markdown_tables_escape_pipes():
    from sp500lab.reporting.tables import Cell, Table
    block = TableBlock(Table(["a"], [[Cell("x | y", "x")]]))
    out = md_render._block(block)
    assert r"x \| y" in out


def test_a_line_chart_is_described_rather_than_faked(store):
    """There is no honest Markdown time series, so it must say so and not draw one."""
    records = store.load()
    report = FV.forward_strategy_report(records[records["strategy"] == "great"])
    md = md_render.render(report)
    assert "(chart — see the HTML report)" in md


def test_a_bar_chart_becomes_a_real_table(store):
    """A bar chart IS a small grid of numbers, so nothing is lost by tabulating it."""
    md = md_render.render(FV.forward_index_report(store.load()))
    assert "Change in annualised monthly Sharpe" in md


def test_markdown_round_trips_to_disk(tmp_path, store):
    path = md_render.write(FV.forward_index_report(store.load()),
                           tmp_path / "sub" / "summary.md")
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("# Forward test")


def test_a_long_table_is_truncated_with_a_note():
    from sp500lab.reporting.tables import Cell, Table
    rows = [[Cell(str(i), i)] for i in range(md_render.MAX_TABLE_ROWS + 10)]
    out = md_render._block(TableBlock(Table(["n"], rows)))
    assert "further row(s) omitted" in out


# ==========================================================================
# The whole set, end to end
# ==========================================================================

def test_the_html_set_writes_and_is_self_contained(tmp_path, store):
    records = store.load()
    page = html_render.write(FV.forward_index_report(records), tmp_path / "index.html")
    text = page.read_text(encoding="utf-8")
    assert text.lstrip().startswith("<!doctype html")
    # Self-contained means no network: no CDN, no external stylesheet, no remote image.
    for forbidden in ("src=\"http", "href=\"http://", "cdn."):
        assert forbidden not in text


def test_every_candidate_link_points_at_a_page_the_set_contains(store):
    """A dead relative link is the one defect an index page can have and still look fine."""
    from sp500lab.reporting.specs import LinkGrid

    records = store.load()
    index = FV.forward_index_report(records)
    hrefs = {c.href for s in index.sections for b in s.blocks
             if isinstance(b, LinkGrid) for c in b.cards}
    expected = {FV.strategy_href(str(n)) for n in FV.primary_rows(records)["strategy"]}
    assert expected <= hrefs
