"""Composing reports. Reads the registry, returns a `Report` — never markup.

Four reports, one per question:

    comparison_report  which of these is better, and by how much
    run_report         what did this one strategy actually do
    registry_report    what have I tried, and does the winner survive the search
    honesty_report     what would make me distrust the numbers above

Everything here is pure: registry rows and curves in, dataclasses out. That is what lets
`tests/test_reporting.py` assert on the contents of a section without parsing HTML, and
it is what would let a different frontend reuse all of this untouched.

An editorial rule
-----------------
The caveats travel with the numbers. Coverage, the deflated Sharpe, unresolved delisting
exits, holdout exposure and a dirty working tree are not appendices — a report that shows
a 12% CAGR and mentions on page four that half the index was untradable has already
misled its reader. So every report carries its honesty section, and the deflation panel
sits next to the scoreboard rather than behind a link.
"""

from __future__ import annotations

import time

import pandas as pd

from ..backtest import registry
from . import series as S
from . import tables as T
from . import theme
from .specs import (AreaChart, BarChart, Heatmap, LineChart, Note, Report,
                    ScatterChart, Section, Stat, StatRow, TableBlock)

#: Kept short because it doubles as a scatter-point label, where a long
#: name collides with its neighbours. The grey dashed styling already says
#: "benchmark" without spelling it out.
BENCHMARK_LABEL = "SPY"


# --------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())


def _curves_for(runs: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.Series | None]:
    """{label: nav curve} plus one benchmark, from stored month-end curves.

    Labels prefer the strategy name and fall back to the run id when a study holds
    several runs of one strategy - otherwise two lines would share a name and the legend
    toggle would hide both.
    """
    stored = registry.load_curves(list(runs["run_id"]))
    names = runs["strategy"].tolist()
    duplicated = {n for n in names if names.count(n) > 1}

    out: dict[str, pd.Series] = {}
    benchmark: pd.Series | None = None
    for _, r in runs.iterrows():
        df = stored.get(r["run_id"])
        if df is None or "nav" not in df:
            continue
        label = r["strategy"]
        if label in duplicated:
            label = f"{label} · {str(r['run_id'])[-6:]}"
        out[label] = df["nav"].dropna()
        if benchmark is None and "benchmark" in df:
            b = df["benchmark"].dropna()
            if len(b) > 1:
                benchmark = b
    return out, benchmark


def _missing_curves_note(runs: pd.DataFrame, drawn: int) -> Note | None:
    if drawn >= len(runs):
        return None
    return Note(
        f"{len(runs) - drawn} of {len(runs)} runs have no stored equity curve and are "
        "in the tables but not the charts. Curves are stored with each run; a run "
        "logged with curves disabled (as a large search should) can be re-run to "
        "recover one — the fingerprint is unchanged, so it is the same trial.",
        level="warn", title="Some curves are missing.")


def _stat_tiles(record: pd.Series) -> StatRow:
    dd = record.get("max_drawdown")
    return StatRow([
        Stat("CAGR", theme.pct(record.get("cagr")),
             emphasis="good" if _pos(record.get("cagr")) else "bad"),
        Stat("Sharpe", theme.num(record.get("sharpe"))),
        Stat("volatility", theme.pct(record.get("ann_vol"))),
        Stat("max drawdown", theme.pct(dd), note="from the daily curve",
             emphasis="bad" if _lt(dd, -0.4) else ""),
        Stat("turnover", theme.pct(record.get("ann_turnover"), 0), note="per year"),
        Stat("cost drag", theme.pct(record.get("cost_drag")), note="per year",
             emphasis="bad" if _gt(record.get("cost_drag"), 0.02) else ""),
    ])


def _window_note(runs: pd.DataFrame) -> Note | None:
    """Say plainly whether these numbers include the holdout."""
    if runs.empty:
        return None
    touched = bool(runs["touched_holdout"].any()) if "touched_holdout" in runs else False
    start = str(runs["start"].min())
    end = str(runs["end"].max())
    if touched:
        return Note(
            f"At least one run here saw data from {registry.HOLDOUT_START} onward. Every "
            "look costs the holdout some of its value as an independent test, and each "
            "one is recorded in the ledger.",
            level="danger", title="This report includes holdout data.")
    return Note(
        f"All runs cover {start} to {end}, stopping before the "
        f"{registry.HOLDOUT_START} holdout. Numbers here are in-sample by construction — "
        "they say how a strategy did on the data it was designed against.",
        level="info", title="Research window only.")


# --------------------------------------------------------------------------
# 1. Comparison
# --------------------------------------------------------------------------

def comparison_report(runs: pd.DataFrame, *, title: str = "Strategy comparison",
                      subtitle: str = "") -> Report:
    """Several runs side by side: curves, drawdown, scoreboard, risk/return."""
    report = Report(title=title, subtitle=subtitle, generated_at=_now(),
                    meta={"runs": str(len(runs)),
                          "strategies": str(runs["strategy"].nunique()) if len(runs) else "0",
                          "window": f"{runs['start'].min()} .. {runs['end'].max()}"
                                    if len(runs) else "—"})
    if runs.empty:
        report.add(Section("Nothing to compare",
                           [Note("No runs matched. Run a backtest first.", "warn")]))
        return report

    curves, benchmark = _curves_for(runs)
    aligned = S.align(curves)
    if benchmark is not None and aligned:
        with_bench = dict(curves)
        with_bench[BENCHMARK_LABEL] = benchmark
        aligned = S.align(with_bench)

    overview = Section("Overview", blurb=(
        "Curves are rebased to 1.0 on the first date all of them share, so none of them "
        "gets credit for having started earlier."))
    overview.add(_window_note(runs))
    overview.add(_missing_curves_note(runs, len(curves)))

    if aligned:
        lines = [S.equity(c, name,
                          kind="benchmark" if name == BENCHMARK_LABEL else "strategy")
                 for name, c in aligned.items()]
        overview.add(LineChart(
            lines, title="Growth of 1.0", y_format="multiple", log_y=True,
            height=340,
            subtitle="Log scale: equal vertical distances are equal percentage moves, "
                     "so a good 2009 does not visually swamp a good 2019.",
            caption="Click a legend entry to hide that series here and on every other "
                    "chart in this report."))
    report.add(overview)

    board = Section("Scoreboard", blurb=(
        "Best value in each column is highlighted, using the right direction for each "
        "metric — lowest is best for volatility, drawdown, turnover and cost drag."))
    board.add(TableBlock(T.scoreboard(runs.sort_values("sharpe", ascending=False)),
                         title="All runs"))
    if aligned:
        board.add(ScatterChart(
            S.risk_return(aligned, benchmark_name=BENCHMARK_LABEL),
            title="Risk and return", x_label="annualised volatility",
            x_format="pct", y_format="pct",
            caption="A table sorted on return hides that the winner took more risk. "
                    "Up and to the left is better."))
    report.add(board)

    if aligned:
        dd = Section("Drawdown", blurb=(
            "How far below its own peak each strategy sat, and for how long. Depth is "
            "one question; the flat stretches are the other one."))
        flat = []
        for name, curve in aligned.items():
            if name == BENCHMARK_LABEL:
                continue
            line = S.drawdown(curve, name)
            # A strategy that never draws down has no drawdown chart worth drawing -
            # `cash` would otherwise get a full-width panel of a flat line at zero.
            if min(line.finite_y or [0.0]) > -1e-9:
                flat.append(name)
                continue
            dd.add(AreaChart(line, title=name, height=150))
        if flat:
            dd.add(Note(f"No panel for {', '.join(flat)}: never below its own peak.",
                        level="info"))
        dd.add(Note(
            "Computed from month-end curves, so an intra-month trough is invisible here "
            "and these are shallower than the true drawdowns. The maxDD column in the "
            "scoreboard comes from the daily curve and is the number to quote.",
            level="warn"))
        report.add(dd)

        if benchmark is not None:
            rel = Section("Versus the benchmark", blurb=(
                "Cumulative ratio to SPY. Rising means outperforming. Over nineteen "
                "years two equity curves a few percent a year apart look identical; "
                "their ratio does not."))
            rel.add(LineChart(
                [S.relative(c, benchmark, n) for n, c in aligned.items()
                 if n != BENCHMARK_LABEL],
                title="Relative to SPY", y_format="multiple", zero_line=False,
                height=280))
            report.add(rel)

    report.add(_honesty_section(runs))
    return report


# --------------------------------------------------------------------------
# 2. Single-run deep dive
# --------------------------------------------------------------------------

def run_report(record: pd.Series, *, exits: pd.DataFrame | None = None) -> Report:
    """One strategy in detail."""
    rid = record["run_id"]
    curves = registry.load_curve(rid)
    nav = curves["nav"].dropna() if curves is not None and "nav" in curves else None
    bench = (curves["benchmark"].dropna()
             if curves is not None and "benchmark" in curves else None)
    gross = (curves["nav_gross"].dropna()
             if curves is not None and "nav_gross" in curves else None)

    report = Report(
        title=f"{record['strategy']}",
        subtitle=f"{record.get('start')} to {record.get('end')} · "
                 f"{record.get('cost_model')} costs · study “{record.get('study')}”",
        generated_at=_now(),
        meta={"run": str(rid), "fingerprint": str(record.get("fingerprint")),
              "commit": str(record.get("git_commit", ""))[:10],
              "rebalances": theme.count(record.get("n_rebalances"))})

    head = Section("Headline")
    head.add(_stat_tiles(record))
    head.add(_window_note(pd.DataFrame([record])))
    if nav is None:
        head.add(Note("No stored equity curve for this run, so the charts below are "
                      "omitted. Re-run it to record one.", "warn"))
    report.add(head)

    if nav is not None:
        perf = Section("Performance")
        lines = [S.equity(nav, record["strategy"])]
        if gross is not None:
            lines.append(S.equity(gross, "gross of costs", kind="gross"))
        if bench is not None:
            lines.append(S.equity(bench, BENCHMARK_LABEL, kind="benchmark"))
        perf.add(LineChart(lines, title="Growth of 1.0", y_format="multiple",
                           log_y=True, height=330,
                           caption="The dotted grey line is the same strategy with costs "
                                   "switched off; the gap between it and the solid line "
                                   "is what trading cost."))
        perf.add(AreaChart(S.drawdown(nav), title="Drawdown", height=170,
                           caption="Month-end, so shallower than the true intra-month "
                                   "trough."))
        years, values = S.annual_returns(nav)
        if years:
            perf.add(BarChart(years, values, title="Calendar year returns",
                              y_format="pct", height=230))
        report.add(perf)

        stability = Section("Stability", blurb=(
            "A single Sharpe over nineteen years hides whether it was earned steadily or "
            "in one stretch. These are trailing windows, so every point was knowable on "
            "its own date."))
        rs = S.rolling_sharpe(nav, window=36)
        if len(rs):
            stability.add(LineChart([rs], title="Rolling 3-year Sharpe", y_format="num",
                                    zero_line=True, height=230, legend=False))
        rv = S.rolling_vol(nav, window=12)
        if len(rv):
            stability.add(LineChart([rv], title="Rolling 1-year volatility",
                                    y_format="pct", height=210, legend=False))
        report.add(stability)

        rows, cols, grid = S.monthly_grid(nav)
        if rows:
            month = Section("Month by month", blurb=(
                "Reading across a row shows how a year was actually earned; reading down "
                "a column is the only honest way to look for seasonality."))
            month.add(Heatmap(rows, cols, grid, title="Monthly returns"))
            report.add(month)

    costs = Section("Costs", blurb=(
        "Turnover is where backtests lie, so this is reported next to the return rather "
        "than beneath it."))
    costs.add(TableBlock(T.costs(record)))
    report.add(costs)

    report.add(_honesty_section(pd.DataFrame([record]), exits=exits, single=True))
    return report


# --------------------------------------------------------------------------
# 3. Registry / leaderboard
# --------------------------------------------------------------------------

def registry_report(runs: pd.DataFrame, studies: pd.DataFrame,
                    deflations: dict[str, dict] | None = None) -> Report:
    """Everything tried, with the deflated Sharpe next to the leaderboard."""
    deflations = deflations or {}
    report = Report(
        title="Experiment registry",
        subtitle="Every backtest logged, and what the search cost each winner's "
                 "credibility.",
        generated_at=_now(),
        meta={"runs": str(len(runs)),
              "trials": str(runs["fingerprint"].nunique()) if len(runs) else "0",
              "studies": str(len(studies)),
              "holdout looks": str(registry.holdout_touch_count())})

    if runs.empty:
        report.add(Section("Empty", [Note(
            "No runs logged yet. Every backtest is logged automatically unless you pass "
            "--no-log or set SP500LAB_REGISTRY=off.", "warn")]))
        return report

    overview = Section("Studies", blurb=(
        "A study is one named search. `trials` counts distinct configurations, not log "
        "lines — re-running the same thing is one hypothesis tested twice, and counting "
        "it twice would over-deflate."))
    overview.add(TableBlock(T.studies(studies)))
    report.add(overview)

    for study_name, deflation in deflations.items():
        if not deflation:
            continue
        sec = Section(f"Does “{study_name}” survive its own search?", blurb=(
            "The deflated Sharpe asks whether the best result beats what the luckiest of "
            "N worthless strategies would have posted anyway. Read it as a probability, "
            "not a score."))
        sec.add(TableBlock(T.deflation_panel(deflation),
                           title=f"{deflation.get('strategy', '')} — best in study"))
        sec.add(Note(
            "n_trials is the count of distinct configurations in this study. If that "
            "undercounts what you actually tried — because a single search got split "
            "across several study names — the number above is too generous.",
            level="warn"))
        report.add(sec)

    board = Section("Leaderboard", blurb=(
        "Every run, sortable. Click a column header. Sorting is on the underlying "
        "number, not the rendered text."))
    board.add(TableBlock(T.scoreboard(runs.sort_values("sharpe", ascending=False),
                                      highlight_best=False)))
    report.add(board)

    report.add(_holdout_section())
    return report


# --------------------------------------------------------------------------
# 4. Honesty
# --------------------------------------------------------------------------

def _honesty_section(runs: pd.DataFrame, *, exits: pd.DataFrame | None = None,
                     single: bool = False) -> Section:
    """The diagnostics that change what the numbers above mean."""
    sec = Section("What would make me distrust this", blurb=(
        "Every backtest has a set of conditions under which its headline stops being "
        "true. These are this one's."))

    if len(runs) == 1:
        record = runs.iloc[0]
        sec.add(TableBlock(T.diagnostics(record), title="Diagnostics"))
        note = T.coverage_note(record)
        if note:
            sec.add(Note(note, level="warn", title="Price coverage."))
    else:
        worst = runs.loc[runs["coverage_min"].idxmin()] if \
            runs["coverage_min"].notna().any() else None
        if worst is not None:
            sec.add(Note(T.coverage_note(worst), level="warn",
                         title="Price coverage."))
        unresolved = int(pd.to_numeric(runs.get("unresolved_exits"),
                                       errors="coerce").fillna(0).sum())
        if unresolved:
            sec.add(Note(
                f"{unresolved} position(s) across these runs exited with no recorded "
                "reason and were treated as an index removal at the last price. That is "
                "the wrong answer for a bankruptcy. Coverage of the index-change table "
                "is poor before 2010.",
                level="warn", title="Unresolved delisting exits."))
        dirty = int(runs.get("git_dirty", pd.Series(dtype=bool)).sum()) if \
            "git_dirty" in runs else 0
        if dirty:
            sec.add(Note(
                f"{dirty} run(s) were produced from a working tree with uncommitted "
                "changes and cannot be reproduced exactly.",
                level="warn", title="Dirty working tree."))

    if single and exits is not None and len(exits):
        sec.add(TableBlock(T.exits(exits), title="Positions resolved outside a rebalance"))

    sec.add(Note(
        "Standing limitations that apply to every number in this report: price coverage "
        "of the point-in-time index falls to 54.7% in 2007, and the names that are "
        "missing are disproportionately the ones that later delisted. Half-spreads are "
        "estimated, not observed. Delisting outcomes are recorded assumptions, not "
        "measurements. See docs/BACKTEST.md.",
        level="info", title="Always true here."))
    return sec


def honesty_report(runs: pd.DataFrame) -> Report:
    """The diagnostics on their own, for when that is the question being asked."""
    report = Report(title="Honesty panel",
                    subtitle="Coverage, exits, costs and holdout exposure across every "
                             "logged run.",
                    generated_at=_now(), meta={"runs": str(len(runs))})
    if runs.empty:
        report.add(Section("Empty", [Note("No runs logged yet.", "warn")]))
        return report

    cov = Section("Price coverage", blurb=(
        "The share of the point-in-time index that actually had prices at the worst "
        "rebalance of each run. The untradable names are disproportionately the ones "
        "that later delisted, so a low number means the run traded a survivor subset."))
    ordered = runs.sort_values("coverage_min")
    cov.add(BarChart([str(s) for s in ordered["strategy"]],
                     [float(v) if pd.notna(v) else 0.0 for v in ordered["coverage_min"]],
                     title="Worst-rebalance coverage by run", y_format="pct",
                     diverging=False, height=240))
    report.add(cov)

    report.add(_honesty_section(runs))
    report.add(_holdout_section())
    return report


def _holdout_section() -> Section:
    ledger = registry.holdout_touches()
    sec = Section("Holdout ledger", blurb=(
        f"Everything from {registry.HOLDOUT_START} is reserved for the final test. "
        "Backtests stop before it by default; reaching in takes an explicit flag and is "
        "recorded permanently."))
    sec.add(TableBlock(T.holdout_ledger(ledger)))
    if ledger.empty:
        sec.add(Note("Untouched. Keep it that way until there is one final candidate — "
                     "a look cannot be undone.", level="info"))
    else:
        sec.add(Note(f"{len(ledger)} look(s) recorded. Any result quoted against the "
                     "holdout after this many looks is no longer a clean out-of-sample "
                     "test.", level="danger"))
    return sec


# --------------------------------------------------------------------------

def _pos(x) -> bool:
    return _gt(x, 0.0)


def _gt(x, threshold: float) -> bool:
    try:
        return float(x) > threshold
    except (TypeError, ValueError):
        return False


def _lt(x, threshold: float) -> bool:
    try:
        return float(x) < threshold
    except (TypeError, ValueError):
        return False
