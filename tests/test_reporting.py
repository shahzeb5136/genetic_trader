"""Tests for the reporting layer.

The split the package is built around shows up directly in how these are written: almost
every test asserts on **numbers and dataclasses**, not on markup. `series.py`,
`tables.py` and `views.py` produce no HTML, so a test can check that a drawdown is
computed correctly or that the right column is highlighted without a single string of
tags. Only the last handful touch the renderer, and those check structure rather than
appearance.

That is the payoff of the seam: a change to how a chart looks cannot break these, and a
change to what it shows cannot slip past them.

The bug that motivated several of these
---------------------------------------
The first version of `theme.direction` inferred a metric's direction from its name and
concluded that lower drawdown is better. Drawdown is stored negative, so it highlighted
-77.65% as the best in the column - the worst result in the table, marked green. It read
plausibly in code and was only visible on screen. `test_worst_drawdown_is_never_the_best`
exists so it cannot come back.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sp500lab.reporting import series as S
from sp500lab.reporting import tables as T
from sp500lab.reporting import theme, views
from sp500lab.reporting.render import charts, html
from sp500lab.reporting.specs import (AreaChart, BarChart, Heatmap, LineChart, Note,
                                      Report, ScatterChart, Section, Stat, StatRow,
                                      TableBlock)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def curve(values: list[float], start: str = "2010-01-31") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="ME").strftime("%Y-%m-%d")
    return pd.Series(values, index=list(idx), dtype=float)


def growing(n: int = 60, rate: float = 0.01) -> pd.Series:
    return curve([(1 + rate) ** i for i in range(n)])


def run_row(**kw) -> pd.Series:
    base = {
        "run_id": "20260101T000000-abc123", "fingerprint": "ffff", "study": "s",
        "strategy": "demo", "cost_model": "realistic", "start": "2010-01-31",
        "end": "2021-12-31", "cagr": 0.10, "total_return": 2.0, "ann_vol": 0.18,
        "sharpe": 0.55, "sortino": 0.7, "max_drawdown": -0.42, "calmar": 0.24,
        "hit_rate": 0.55, "ann_turnover": 1.2, "avg_positions": 50.0,
        "cost_drag": 0.01, "information_ratio": 0.05, "beta": 1.0, "alpha": 0.0,
        "n_months": 143, "sharpe_monthly": 0.6, "skew_monthly": -0.2,
        "kurtosis_monthly": 1.1, "coverage_min": 0.55, "coverage_median": 0.77,
        "forced_exits": 3, "unresolved_exits": 0, "spread_fallback_orders": 0,
        "unfilled_orders": 0, "ruined": False, "total_cost": 1234.0,
        "commission": 900.0, "spread_cost": 334.0, "traded_notional": 1e6,
        "n_orders": 500, "git_commit": "deadbeef", "git_dirty": False,
        "n_rebalances": 143, "touched_holdout": False, "runtime_seconds": 0.2,
    }
    base.update(kw)
    return pd.Series(base)


def runs_frame(n: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(run_row(run_id=f"run{i}", strategy=f"strat{i}", fingerprint=f"fp{i}",
                            sharpe=0.4 + 0.1 * i, cagr=0.08 + 0.01 * i,
                            max_drawdown=-0.3 - 0.1 * i, ann_vol=0.15 + 0.02 * i))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# series.py
# --------------------------------------------------------------------------

def test_equity_rebases_to_one():
    s = S.equity(curve([50.0, 55.0, 60.0]), "x")
    assert s.y[0] == pytest.approx(1.0)
    assert s.y[-1] == pytest.approx(1.2)


def test_drawdown_is_zero_on_a_monotonic_curve():
    dd = S.drawdown(growing(24))
    assert max(dd.finite_y) == pytest.approx(0.0)
    assert min(dd.finite_y) == pytest.approx(0.0)


def test_drawdown_measures_from_the_running_peak():
    dd = S.drawdown(curve([1.0, 2.0, 1.0, 1.5]))
    assert dd.y[1] == pytest.approx(0.0)
    assert dd.y[2] == pytest.approx(-0.5)      # half off a peak of 2.0
    assert dd.y[3] == pytest.approx(-0.25)


def test_align_restricts_to_the_common_window_and_rebases_there():
    """A curve that starts earlier must not get credit for compounding longer."""
    a = curve([1.0, 2.0, 4.0, 8.0], start="2010-01-31")
    b = curve([1.0, 2.0], start="2010-03-31")
    out = S.align({"a": a, "b": b})
    assert len(out["a"]) == len(out["b"]) == 2
    assert out["a"].iloc[0] == pytest.approx(1.0)
    assert out["b"].iloc[0] == pytest.approx(1.0)
    assert out["a"].iloc[-1] == pytest.approx(2.0)   # 4 -> 8 over the shared window


def test_relative_rises_when_outperforming():
    strat = curve([1.0, 1.2, 1.5])
    bench = curve([1.0, 1.1, 1.2])
    rel = S.relative(strat, bench, "x")
    assert rel.y[0] == pytest.approx(1.0)
    assert rel.y[-1] > 1.0


def test_rolling_sharpe_annualises_from_months_not_days():
    """Twelve periods a year, not 252. Getting this wrong overstates by sqrt(21)."""
    rng = np.random.default_rng(0)
    c = curve(list(np.cumprod(1 + rng.normal(0.01, 0.03, 120))))
    rs = S.rolling_sharpe(c, window=36)
    assert len(rs) == 120 - 1 - 36 + 1
    # A monthly Sharpe of this data annualised by 252 would be implausibly large.
    assert max(abs(v) for v in rs.finite_y) < 12


def test_rolling_windows_are_trailing():
    """The first value appears only once a full window of history exists."""
    rs = S.rolling_sharpe(growing(60), window=36)
    assert rs.x[0] >= curve([0.0] * 37).index[36]


def test_annual_returns_include_the_first_partial_year():
    years, values = S.annual_returns(growing(30, rate=0.01))
    assert years[0] == "2010"
    assert len(years) == len(values) >= 2
    assert all(v > 0 for v in values)


def test_monthly_grid_has_twelve_months_plus_ytd():
    rows, cols, grid = S.monthly_grid(growing(30))
    assert cols[-1] == "YTD"
    assert len(cols) == 13
    assert all(len(r) == 13 for r in grid)


def test_monthly_grid_ytd_compounds_the_row():
    rows, cols, grid = S.monthly_grid(growing(26, rate=0.01))
    full = next(r for r in grid if all(v is not None for v in r[:12]))
    expected = math.prod(1 + v for v in full[:12]) - 1
    assert full[12] == pytest.approx(expected)


def test_risk_return_marks_the_benchmark():
    pts = S.risk_return({"a": growing(40), "SPY": growing(40, 0.005)},
                        benchmark_name="SPY")
    kinds = {p.label: p.kind for p in pts}
    assert kinds["SPY"] == "benchmark"
    assert kinds["a"] == "strategy"


def test_series_survive_an_empty_curve():
    empty = pd.Series(dtype=float)
    assert len(S.equity(empty, "x")) == 0
    assert len(S.drawdown(empty)) == 0
    assert S.align({"a": empty}) == {}
    assert S.summary_stats(empty) == {}


# --------------------------------------------------------------------------
# theme.py - metric direction
# --------------------------------------------------------------------------

def test_higher_is_better_for_return_metrics():
    for m in ("cagr", "sharpe", "calmar", "information_ratio"):
        assert theme.direction(m) == 1


def test_lower_is_better_for_cost_metrics():
    for m in ("ann_vol", "ann_turnover", "cost_drag"):
        assert theme.direction(m) == -1


def test_max_drawdown_is_higher_is_better_because_it_is_stored_negative():
    """-0.39 beats -0.78. Treating drawdown as 'lower is better' inverts the column."""
    assert theme.direction("max_drawdown") == 1


def test_position_count_has_no_better_direction():
    assert theme.direction("avg_positions") == 0


def test_unknown_metrics_are_not_guessed():
    assert theme.direction("some_new_metric") == 0


def test_formatters_handle_missing_values():
    for fn in (theme.pct, theme.num, theme.money, theme.multiple, theme.bp, theme.count):
        assert fn(None) == "—"
        assert fn(float("nan")) == "—"
    assert theme.pct(0.1041) == "10.41%"
    assert theme.bp(0.00176) == "17.6bp"


# --------------------------------------------------------------------------
# tables.py
# --------------------------------------------------------------------------

def test_worst_drawdown_is_never_the_best():
    """The regression test for the inverted-direction bug. See the module docstring."""
    runs = runs_frame(3)                      # drawdowns -0.30, -0.40, -0.50
    table = T.scoreboard(runs)
    col = table.columns.index("maxDD")
    highlighted = [r[col].text for r in table.rows if r[col].emphasis == "good"]
    assert highlighted == ["-30.00%"], "the shallowest drawdown is the best one"


def test_scoreboard_highlights_the_best_in_each_direction():
    runs = runs_frame(3)
    table = T.scoreboard(runs)
    def best(header):
        c = table.columns.index(header)
        return [r[c].text for r in table.rows if r[c].emphasis == "good"]
    assert best("Sharpe") == ["0.60"]          # highest
    assert best("vol") == ["15.00%"]           # lowest


def test_scoreboard_sort_keys_are_numbers_not_text():
    """Sorting on rendered text puts '9.84%' above '11.10%'."""
    runs = runs_frame(3)
    table = T.scoreboard(runs)
    col = table.columns.index("CAGR")
    keys = [r[col].sort_key for r in table.rows]
    assert all(isinstance(k, float) for k in keys)
    assert keys == sorted(keys)


def test_scoreboard_is_empty_not_broken_without_runs():
    table = T.scoreboard(pd.DataFrame())
    assert table.empty


def test_deflation_panel_flags_a_result_below_the_threshold():
    low = T.deflation_panel({"deflated_sharpe": 0.42, "n_trials": 500,
                             "n_months": 143, "psr_vs_zero": 0.99})
    dsr_row = low.rows[-1]
    assert dsr_row[1].emphasis == "bad"
    assert "not distinguishable" in low.caption


def test_deflation_panel_passes_a_result_above_the_threshold():
    high = T.deflation_panel({"deflated_sharpe": 0.99, "n_trials": 4,
                              "n_months": 143, "psr_vs_zero": 0.99})
    assert high.rows[-1][1].emphasis == "good"
    assert "Survives" in high.caption


def test_diagnostics_flags_low_coverage_and_a_dirty_tree():
    table = T.diagnostics(run_row(coverage_min=0.55, git_dirty=True))
    flat = {r[0].text: r[1] for r in table.rows}
    assert flat["price coverage (worst rebalance)"].emphasis == "bad"
    assert flat["built from a dirty working tree"].emphasis == "warn"


def test_diagnostics_stays_quiet_when_everything_is_fine():
    table = T.diagnostics(run_row(coverage_min=0.99, git_dirty=False,
                                  unresolved_exits=0, spread_fallback_orders=0))
    assert all(cell.emphasis in ("", "muted") for row in table.rows for cell in row)


def test_holdout_ledger_reads_as_good_news_when_empty():
    table = T.holdout_ledger(pd.DataFrame())
    assert table.empty
    assert "Never looked at" in table.caption


def test_exits_table_flags_a_bankruptcy():
    df = pd.DataFrame([{"date": "2008-09-15", "ticker": "LEH", "reason": "bankruptcy",
                        "delist_return": -1.0, "proceeds": 0.0}])
    table = T.exits(df)
    assert table.rows[0][2].emphasis == "bad"


# --------------------------------------------------------------------------
# views.py - composition, still no markup
# --------------------------------------------------------------------------

def test_comparison_report_has_the_expected_sections(monkeypatch):
    monkeypatch.setattr(views.registry, "load_curves",
                        lambda ids: {r: pd.DataFrame(
                            {"nav": growing(60).to_numpy(),
                             "benchmark": growing(60, 0.008).to_numpy()},
                            index=growing(60).index) for r in ids})
    report = views.comparison_report(runs_frame(3))
    titles = [s.title for s in report.sections]
    assert "Overview" in titles and "Scoreboard" in titles
    assert any("distrust" in t for t in titles), "the honesty section always travels along"


def test_every_report_carries_an_honesty_section(monkeypatch):
    """The editorial rule in views.py: caveats travel with the numbers."""
    monkeypatch.setattr(views.registry, "load_curves", lambda ids: {})
    monkeypatch.setattr(views.registry, "holdout_touches", lambda: pd.DataFrame())
    for report in (views.comparison_report(runs_frame(2)),
                   views.honesty_report(runs_frame(2))):
        assert any("distrust" in s.title for s in report.sections)


def test_report_says_plainly_when_the_holdout_was_touched(monkeypatch):
    monkeypatch.setattr(views.registry, "load_curves", lambda ids: {})
    runs = runs_frame(2)
    runs.loc[0, "touched_holdout"] = True
    report = views.comparison_report(runs)
    notes = [b for s in report.sections for b in s.blocks if isinstance(b, Note)]
    assert any(n.level == "danger" and "holdout" in n.text.lower() for n in notes)


def test_report_is_not_empty_but_warns_when_no_runs_match():
    report = views.comparison_report(pd.DataFrame())
    assert len(report.sections) == 1
    assert isinstance(report.sections[0].blocks[0], Note)


def test_missing_curves_are_reported_not_hidden(monkeypatch):
    monkeypatch.setattr(views.registry, "load_curves", lambda ids: {})
    report = views.comparison_report(runs_frame(3))
    notes = [b for s in report.sections for b in s.blocks if isinstance(b, Note)]
    assert any("no stored equity curve" in n.text for n in notes)


def test_flat_strategies_get_no_drawdown_panel(monkeypatch):
    """`cash` never draws down; a full-width panel of a flat line is noise."""
    flat = pd.Series([1.0] * 60, index=growing(60).index)
    monkeypatch.setattr(views.registry, "load_curves",
                        lambda ids: {r: pd.DataFrame({"nav": flat}) for r in ids})
    runs = runs_frame(1)
    runs.loc[0, "strategy"] = "cash"
    report = views.comparison_report(runs)
    dd = report.section("Drawdown")
    if dd is not None:
        assert not any(isinstance(b, AreaChart) for b in dd.blocks)


def test_section_add_ignores_none():
    section = Section("x")
    section.add(None).add(Note("kept"))
    assert len(section.blocks) == 1


def test_report_skips_empty_sections():
    report = Report("t")
    report.add(Section("empty")).add(Section("full", [Note("x")]))
    assert [s.title for s in report.sections] == ["full"]


def test_section_anchors_are_url_safe():
    assert Section("What would make me distrust this?").anchor == \
        "what-would-make-me-distrust-this"


# --------------------------------------------------------------------------
# render/ - structure, not appearance
# --------------------------------------------------------------------------

def test_line_chart_breaks_the_path_at_a_gap():
    """A gap drawn as a straight line is a claim about data that does not exist."""
    s = S.LineSeries("x", ["2010-01-31", "2010-02-28", "2010-03-31"],
                     [1.0, float("nan"), 1.2])
    svg = charts.line_chart(LineChart([s]))
    assert svg.count("M") >= 2, "a break must start a new subpath"


def test_charts_render_without_data():
    for spec in (LineChart([]), BarChart([], []), ScatterChart([]),
                 Heatmap([], [], [])):
        assert "no data" in charts.render(spec)


def test_every_chart_kind_renders_valid_svg():
    line = S.equity(growing(40), "a")
    specs = [
        LineChart([line]),
        AreaChart(S.drawdown(curve([1.0, 2.0, 1.0]))),
        BarChart(["2010", "2011"], [0.1, -0.05]),
        ScatterChart([S.Point(0.1, 0.2, "a")]),
        Heatmap(["2010"], ["Jan", "Feb"], [[0.01, -0.02]]),
    ]
    for spec in specs:
        out = charts.render(spec)
        assert out.startswith("<svg") or out.startswith('<div class="chart-empty"')
        assert out.count("<svg") == out.count("</svg>")


def test_unknown_spec_is_a_type_error_not_silent():
    with pytest.raises(TypeError):
        charts.render(object())


def test_html_is_self_contained():
    """No CDN, no external stylesheet, no build step."""
    report = Report("t", sections=[Section("s", [Note("hello")])])
    out = html.render(report)
    assert "<!doctype html>" in out.lower()
    assert "http://" not in out and "https://" not in out
    assert "<style>" in out and "<script>" in out


def test_html_escapes_user_content():
    report = Report("<script>alert(1)</script>",
                    sections=[Section("s", [Note("<img onerror=x>")])])
    out = html.render(report)
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out


def test_table_cells_carry_a_numeric_sort_key_into_the_markup():
    table = T.scoreboard(runs_frame(2))
    out = html.render(Report("t", sections=[Section("s", [TableBlock(table)])]))
    assert 'data-key="' in out


def test_stat_tiles_render():
    block = StatRow([Stat("CAGR", "10.00%", emphasis="good")])
    out = html.render(Report("t", sections=[Section("s", [block])]))
    assert "10.00%" in out and "stat good" in out


def test_write_produces_a_file(tmp_path):
    report = Report("t", sections=[Section("s", [Note("x")])])
    out = html.write(report, tmp_path / "sub" / "r.html")
    assert out.exists() and out.stat().st_size > 500


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def _have_data() -> bool:
    try:
        from sp500lab.storage import silver_exists
        return silver_exists("market/daily_bars_adjusted")
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _have_data(), reason="silver layer not built")
def test_a_real_run_round_trips_into_a_report(tmp_path, monkeypatch):
    """Backtest -> registry -> curve -> report, with nothing stubbed."""
    from sp500lab.backtest import registry, run_backtest
    monkeypatch.setattr(registry, "EXPERIMENT_LOG", tmp_path / "runs.jsonl")
    monkeypatch.setattr(registry, "CURVE_LOG", tmp_path / "curves.jsonl")
    monkeypatch.setattr(registry, "HOLDOUT_LOG", tmp_path / "holdout.jsonl")
    monkeypatch.setenv("SP500LAB_REGISTRY", "on")

    run_backtest("equal_weight", start="2015-01-01", study="report-test")
    runs = registry.load("report-test")
    assert len(runs) == 1

    report = views.comparison_report(runs, title="test")
    out = html.write(report, tmp_path / "r.html")
    text = out.read_text(encoding="utf-8")
    assert "equal_weight" in text
    assert "<svg" in text, "the stored curve should have produced a chart"
