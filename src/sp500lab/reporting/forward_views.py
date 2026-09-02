"""Composing the forward-test reports. Reads the forward store, returns a `Report`.

A sibling of `views.py` rather than more of it, for two reasons. The first is size -
`views.py` already composes eight reports. The second matters more: **a forward report
answers a different question from every other report in this project.** Everything in
`views.py` asks "what did this strategy do?". Everything here asks "did what it did
out of sample match what the research window predicted, and is the gap larger than the
sampling error of a 54-month sample?" Those need different tables, different charts and
a different editorial voice, and mixing them would blur both.

Same discipline as `views.py`: pure. Forward records and stored curves in, dataclasses
out. No markup, so `tests/` can assert on the contents of a section without parsing a
tag, and `render/` stays swappable (ADR-028).

Four reports, one per question
-------------------------------
``forward_index_report``     the executive summary: what held, what decayed, what failed,
                             and what a window this short is able to say at all
``forward_strategy_report``  one candidate in full - prediction, outcome, every check,
                             every year, the orders
``forward_decay_report``     the cross-section: did the research ranking predict the
                             forward ranking? This is the question the whole exercise
                             exists to answer, and it is not visible in any single report
``forward_honesty_report``   what would make you distrust all of the above

The editorial rule, inherited and sharpened
--------------------------------------------
`views.py` says the caveats travel with the numbers. Here there is one caveat that
outranks the rest and it is printed on every page: **54 months of monthly observations
put a +-0.9 band around an annualised Sharpe of 1.0.** A forward test on this data can
refute a strategy and cannot confirm one. Any page that shows a forward Sharpe without
that band next to it has already misled its reader, so `_power_note()` is not optional
and is not at the bottom.

See docs/FORWARD_TEST.md, ADR-033 and ADR-034.
"""

from __future__ import annotations

import math

import pandas as pd

from ..forward import store
from ..forward.compare import compare
from ..forward.windows import describe_power, sharpe_band
from . import series as S
from . import theme
from .specs import (
    AreaChart,
    BarChart,
    Download,
    Heatmap,
    LineChart,
    LinkCard,
    LinkGrid,
    Note,
    Report,
    ScatterChart,
    Section,
    Stat,
    StatRow,
    TableBlock,
)
from .tables import Cell, Table
from .util import finite as _finite
from .util import gt as _gt
from .util import lt as _lt
from .util import now as _now
from .util import slugify

BENCHMARK_LABEL = "SPY"

#: The cost setting a summary leads with. All three are always shown; one has to be the
#: headline, and `realistic` is the one costs.py argues is the honest default.
PRIMARY_COSTS = "realistic"

#: How a verdict reads on a card and in a table.
VERDICT_EMPHASIS = {"held": "good", "decayed": "warn", "failed": "bad",
                    "inconclusive": ""}

VERDICT_ORDER = ("failed", "decayed", "held", "inconclusive")

#: Largest embedded trade ledger, in rows. Same reasoning as `views.MAX_EMBEDDED_TRADES`:
#: base64 inflates a file by a third and a 20,000-order ledger makes a page no browser
#: opens pleasantly. Lower here because a forward window is 54 months, so a ledger that
#: overflows this is a strategy holding several hundred names.
MAX_EMBEDDED_TRADES = 20_000


# --------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------

def _power_note(n_months: int) -> Note:
    """The caveat that outranks every number on the page. Printed on all of them."""
    return Note(
        describe_power(n_months) + " A forward test on this data can REFUTE a strategy "
        "and cannot CONFIRM one: a large negative result is informative because it is "
        "large, and a modest positive one is indistinguishable from noise. Read "
        "“held” as “not refuted”.",
        level="warn", title="What 54 months can and cannot prove.")


def _verdict_tiles(rows: pd.DataFrame) -> StatRow:
    """Counts by verdict, plus how many still beat the index out of sample."""
    counts = rows["verdict"].value_counts().to_dict()
    beat = int((pd.to_numeric(rows["forward_d_sharpe"], errors="coerce") > 0).sum())
    return StatRow([
        Stat("candidates", theme.count(len(rows))),
        Stat("held", theme.count(counts.get("held", 0)), note="not refuted",
             emphasis="good" if counts.get("held") else ""),
        Stat("decayed", theme.count(counts.get("decayed", 0)),
             note="worked, lost ground", emphasis="warn" if counts.get("decayed") else ""),
        Stat("failed", theme.count(counts.get("failed", 0)), note="refuted",
             emphasis="bad" if counts.get("failed") else ""),
        Stat("beat the index", f"{beat} of {len(rows)}", note="forward, risk-adjusted",
             emphasis="good" if beat else "bad"),
    ], title="Out of sample, 2022 onward")


def _selection_note(bar: dict) -> Note | None:
    """The best-of-N correction for the forward window itself."""
    n = bar.get("n_forward_tests", 0)
    if n < 2:
        return None
    return Note(
        f"{n} candidates were looked at on the forward window. The luckiest of {n} "
        f"worthless ones would have posted an annualised Sharpe of {bar['bar']:.2f} "
        f"over this span with no skill at all; the best actually posted "
        f"{bar['best_sharpe_monthly']:.2f}. The holdout stops a strategy being FITTED "
        "to 2022 onward. It does not stop one being CHOSEN there, and that is the same "
        "correction the deflated Sharpe applies to a search "
        "(`sp500lab experiments deflate forward-test`).",
        level="warn", title="Multiple testing on the forward window itself.")


def _seal_note(rows: pd.DataFrame) -> Note:
    """Declared or auto, and what the difference is worth."""
    declared = int((rows["seal_mode"] == "declared").sum())
    if declared == len(rows):
        return Note(
            "Every candidate here was pre-registered before any forward result was "
            "seen: its configuration, its research-window prediction and the trial "
            "count behind it were written down first. That fixes the candidate SET, "
            "which is what stops “test twenty, report three”. It does not prove the "
            "candidates were chosen without any prior knowledge of the 2022-2026 "
            "period - only a seal written days earlier does that.",
            level="info", title="Pre-registered as one set.")
    return Note(
        f"{declared} of {len(rows)} candidates were pre-registered before the look; the "
        "rest were auto-sealed at run time. An auto seal's numbers are equally clean - "
        "the prediction is still measured on research data alone - but it cannot prove "
        "the candidate was chosen before its answer was known. Read a set of auto seals "
        "as a survey.",
        level="warn", title="Not all of these were pre-registered.")


#: Candidates that encode no forecast. `random_weight` picks names at random,
#: `equal_weight` holds the whole tradable universe and `cash` holds nothing. They are
#: in the suite as null hypotheses, and their forward verdicts are the calibration for
#: everything else on the page.
NULL_HYPOTHESES = ("random_weight", "equal_weight", "cash")


def _control_note(rows: pd.DataFrame) -> Note | None:
    """What a verdict means, read against the strategies that forecast nothing.

    The single most misreadable number on the summary is the count of `held`. A verdict
    is a statement about whether a strategy matched ITS OWN prediction - and a strategy
    that predicted mediocrity and delivered mediocrity held. `random_weight` picks its
    holdings at random and is in the suite precisely so that sentence has a worked
    example next to it.
    """
    present = rows[rows["strategy"].isin(NULL_HYPOTHESES)]
    if present.empty:
        return None
    parts = [f"`{r['strategy']}` {str(r['verdict']).upper()}"
             + (f" (forward Sharpe {r['forward_sharpe']:.2f} against the index's "
                f"{r['forward_sharpe'] - r['forward_d_sharpe']:.2f})"
                if _finite(r.get("forward_sharpe")) and _finite(r.get("forward_d_sharpe"))
                else "")
             for _, r in present.iterrows()]
    return Note(
        "A verdict says whether a strategy matched ITS OWN research-window prediction, "
        "not whether it is any good. A strategy that predicted mediocrity and delivered "
        "mediocrity held. The null hypotheses in this set are the calibration: "
        + "; ".join(parts) + ". "
        "`random_weight` picks its holdings at random and encodes no forecast at all, "
        "so read its verdict as the floor - anything that merely matches it has "
        "demonstrated nothing. The column to read for quality is `vs index (fwd)`.",
        level="warn", title="What “held” does not mean.")


#: How many candidates at each extreme get a drawn label on the scatter.
_LABELLED_EXTREMES = 3


def _decay_scatter(rows: pd.DataFrame) -> ScatterChart:
    """Research edge on the x axis, forward edge on the y. The whole exercise, one chart.

    Only the extremes are labelled. Twenty-two names in one cloud overlap into an
    unreadable smear, and the chart's job is the SHAPE of the cloud - whether it slopes
    upward - not the identity of every point. The scoreboard immediately above names
    all of them in order.
    """
    usable = rows[rows["research_d_sharpe"].map(_finite)
                  & rows["forward_d_sharpe"].map(_finite)]
    named = _notable(usable)
    points = []
    for _, r in usable.iterrows():
        name = str(r["strategy"])
        y = float(r["forward_d_sharpe"])
        points.append(S.Point(x=float(r["research_d_sharpe"]), y=y,
                              label=name if name in named else "",
                              kind="strategy" if y > 0 else "benchmark"))
    return ScatterChart(
        points, title="Research edge against forward edge",
        x_label="research Sharpe minus SPY, 2007-2021 (vertical axis: 2022-2026)",
        x_format="num", y_format="num", height=400,
        caption="Each point is one candidate. Both axes are measured against the index "
                "over the SAME dates as the strategy, because the two windows differ "
                "and a raw Sharpe would rank the market. A point above zero on the "
                "vertical axis beat the index out of sample; a point on the diagonal "
                "delivered exactly what research promised. If the research window "
                "predicted anything, the cloud slopes upward — see the rank correlation "
                f"in the decay report. Only the {_LABELLED_EXTREMES} best and worst on "
                "each axis are labelled; the table above names them all.")


def _notable(rows: pd.DataFrame) -> set[str]:
    """The names worth drawing on a crowded scatter: both extremes of both axes."""
    named: set[str] = set()
    for column in ("forward_d_sharpe", "research_d_sharpe"):
        ordered = rows.sort_values(column)
        named |= set(ordered["strategy"].head(_LABELLED_EXTREMES))
        named |= set(ordered["strategy"].tail(_LABELLED_EXTREMES))
    return {str(n) for n in named}


#: Bar labels longer than this overlap their neighbours at 22 bars across 900px.
_MAX_BAR_LABEL = 13


def _decay_bars(rows: pd.DataFrame) -> BarChart:
    ordered = rows.sort_values("decay_sharpe_monthly", ascending=False)
    return BarChart(
        labels=[_short(str(s)) for s in ordered["strategy"]],
        values=[float(v) if _finite(v) else 0.0
                for v in ordered["decay_sharpe_monthly"]],
        title="Change in annualised monthly Sharpe, research to forward",
        y_format="num", diverging=True, height=300,
        caption="Positive means it did BETTER out of sample than in research. The "
                "standard error on each of these bars is around 0.55, so most of what "
                "you are looking at is sampling noise - which is the finding, not a "
                "defect in the chart.")


def _short(name: str) -> str:
    """Trim a strategy name to fit under a bar. The full name is in every table."""
    return name if len(name) <= _MAX_BAR_LABEL else name[:_MAX_BAR_LABEL - 1] + "…"


def _scoreboard_table(rows: pd.DataFrame, hrefs: dict[str, str] | None = None) -> Table:
    """The paired table: prediction, outcome, gap, verdict. One row per candidate."""
    hrefs = hrefs or {}
    columns = ["strategy", "verdict", "research CAGR", "forward CAGR",
               "research Sharpe", "forward Sharpe", "vs index (res)",
               "vs index (fwd)", "ΔSharpe", "sigma", "P(>research)", "maxDD fwd"]
    out = []
    for _, r in rows.iterrows():
        verdict = str(r.get("verdict", ""))
        out.append([
            Cell(str(r["strategy"]), str(r["strategy"]).lower(),
                 href=hrefs.get(str(r["strategy"]), "")),
            Cell(verdict.upper(), VERDICT_ORDER.index(verdict)
                 if verdict in VERDICT_ORDER else 9,
                 emphasis=VERDICT_EMPHASIS.get(verdict, "")),
            _num_cell(r.get("research_cagr"), theme.pct),
            _num_cell(r.get("forward_cagr"), theme.pct,
                      emphasis="good" if _gt(r.get("forward_cagr"), 0) else "bad"),
            _num_cell(r.get("research_sharpe"), theme.num),
            _num_cell(r.get("forward_sharpe"), theme.num),
            _num_cell(r.get("research_d_sharpe"), lambda v: f"{v:+.2f}"),
            _num_cell(r.get("forward_d_sharpe"), lambda v: f"{v:+.2f}",
                      emphasis="good" if _gt(r.get("forward_d_sharpe"), 0) else ""),
            _num_cell(r.get("decay_sharpe_monthly"), lambda v: f"{v:+.2f}",
                      title="annualised monthly Sharpe, forward minus research"),
            _num_cell(r.get("decay_z"), lambda v: f"{v:+.1f}",
                      title="that change divided by its own standard error"),
            _num_cell(r.get("psr_vs_research"), theme.num,
                      title="probability the true forward Sharpe exceeds the research one"),
            _num_cell(r.get("forward_max_drawdown"), theme.pct),
        ])
    return Table(columns, out,
                 aligns=["left", "left"] + ["right"] * (len(columns) - 2), caption=(
        "Sorted by forward Sharpe against the index over the same dates. `ΔSharpe` is "
        "the change from research to forward and `sigma` is that change in units of its "
        "own standard error - anything inside ±1 is noise."))


def _checks_table(comparison) -> Table:
    rows = []
    for chk in comparison.checks:
        mark = {True: "pass", False: "FAIL", None: "n/a"}[chk.passed]
        emphasis = {True: "good", False: "bad", None: "muted"}[chk.passed]
        rows.append([Cell(chk.name, chk.name), Cell(mark, mark, emphasis=emphasis),
                     Cell(chk.detail, chk.detail.lower())])
    return Table(["check", "result", "detail"], rows,
                 aligns=["left", "left", "left"], sortable=False,
                 caption="A check abstains rather than guessing when a number is "
                         "missing, and `kept_its_edge` abstains for a strategy that "
                         "never had an edge to keep - that is not a failure.")


def _num_cell(value, fmt, *, emphasis: str = "", title: str = "") -> Cell:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return Cell("—", float("-inf"), "muted", title)
    try:
        return Cell(fmt(float(value)), float(value), emphasis, title)
    except (TypeError, ValueError):
        return Cell(str(value), str(value), emphasis, title)


def strategy_href(name: str) -> str:
    return f"forward-{slugify(name)}.html"


def primary_rows(records: pd.DataFrame, cost_model: str = PRIMARY_COSTS) -> pd.DataFrame:
    """One row per candidate, at one cost setting, newest look first.

    Newest rather than first: a second look at the same candidate is the current state
    of the evidence, and `look_number` travels with the row so a reader can see there
    was an earlier one.

    One row per STRATEGY, not per seal. A configuration revision (a bug fix that bumps
    a fingerprint - see shallow_mlp's `revision` and the ADR-037 postscript) creates a
    second seal for the same candidate, and the newest record is the current state of
    knowledge about it. The superseded seal's records stay in the store and in the
    ledger CSVs; a summary page listing a candidate twice, once with a number its own
    rationale calls void, would be noise wearing honesty's clothes.
    """
    if records.empty:
        return records
    rows = records[records["cost_model"] == cost_model]
    if rows.empty:
        rows = records
    rows = (rows.sort_values("logged_at")
            .drop_duplicates("seal_id", keep="last")
            .drop_duplicates("strategy", keep="last"))
    return rows.sort_values("forward_d_sharpe", ascending=False).reset_index(drop=True)


# ==========================================================================
# 1. The executive summary
# ==========================================================================

def forward_index_report(records: pd.DataFrame, *, cost_model: str = PRIMARY_COSTS,
                         extra_cards: list[dict] | None = None) -> Report:
    """The landing page: what survived 2022-2026, and how much that is worth knowing."""
    rows = primary_rows(records, cost_model)
    report = Report(
        title="Forward test: 2022 onward",
        subtitle="What the strategies actually did after the research window ended.",
        generated_at=_now(),
        meta={"candidates": theme.count(len(rows)),
              "cost model": cost_model,
              "window": f"{rows['forward_start'].min()} → {rows['forward_end'].max()}"
                        if len(rows) else "—",
              "looks recorded": theme.count(len(records))})

    if rows.empty:
        report.add(Section("Nothing to report", [Note(
            "No forward test has been run. `python -m sp500lab forward run <strategy>`.",
            "warn")]))
        return report

    months = int(pd.to_numeric(rows["forward_n_months"], errors="coerce").max())
    bar = store.selection_bar(records, cost_model=cost_model)

    # ---- the headline ----------------------------------------------------
    head = Section("The short version", blurb=_headline_prose(rows, cost_model))
    head.add(_verdict_tiles(rows))
    head.add(_control_note(rows))
    head.add(_power_note(months))
    head.add(_selection_note(bar))
    head.add(_seal_note(rows))
    report.add(head)

    # ---- the scoreboard --------------------------------------------------
    board = Section("Every candidate, prediction against outcome", blurb=(
        "The research window made a prediction; the forward window is the only place it "
        "can be checked. Both are here, next to the gap between them and the standard "
        "error of that gap."))
    board.add(TableBlock(_scoreboard_table(
        rows, {str(s): strategy_href(str(s)) for s in rows["strategy"]}),
        title=f"[{cost_model} costs]"))
    board.add(_decay_scatter(rows))
    board.add(_decay_bars(rows))
    report.add(board)

    # ---- the curves ------------------------------------------------------
    curves = _forward_curves(rows)
    if curves:
        chart = Section("Every forward curve at once", blurb=(
            "Growth of 1.0 from the first month each strategy could trade in the "
            "forward window. SPY is on the same axis and is the bar."))
        lines = [S.equity(c, n, kind="benchmark" if n == BENCHMARK_LABEL else "strategy")
                 for n, c in curves.items() if len(c) > 2]
        chart.add(LineChart(lines, title="Growth of 1.0, 2022 onward",
                            y_format="multiple", height=400,
                            caption="Not log-scaled: 4.6 years is too short for a log "
                                    "axis to buy anything, and a linear one makes the "
                                    "2022 drawdown legible."))
        report.add(chart)

    # ---- links -----------------------------------------------------------
    cards = [LinkCard(**c) for c in (extra_cards or [])]
    cards += [
        LinkCard(
            title=str(r["strategy"]), href=strategy_href(str(r["strategy"])),
            blurb=str(r.get("verdict_reason", ""))[:220],
            stats=[("forward CAGR", theme.pct(r.get("forward_cagr"))),
                   ("vs index", f"{r['forward_d_sharpe']:+.2f}"
                    if _finite(r.get("forward_d_sharpe")) else "—"),
                   ("verdict", str(r.get("verdict", "")).upper())],
            emphasis=VERDICT_EMPHASIS.get(str(r.get("verdict", "")), ""))
        for _, r in rows.iterrows()]
    report.add(Section("One report per candidate", blurb=(
        "Each opens the full technical report: the paired comparison, every check, the "
        "curve, every year, the cost sensitivity and the orders it placed."
    )).add(LinkGrid(cards)))

    report.add(_index_honesty(records, rows))
    return report


def _headline_prose(rows: pd.DataFrame, cost_model: str) -> str:
    counts = rows["verdict"].value_counts().to_dict()
    beat = int((pd.to_numeric(rows["forward_d_sharpe"], errors="coerce") > 0).sum())
    best = rows.iloc[0]["strategy"] if len(rows) else "—"
    return (
        f"{len(rows)} candidates were carried out of the research window into "
        f"2022-2026 under {cost_model} costs. "
        f"{counts.get('held', 0)} were not refuted, {counts.get('decayed', 0)} decayed, "
        f"{counts.get('failed', 0)} failed outright"
        + (f", {counts['inconclusive']} were inconclusive" if counts.get("inconclusive")
           else "")
        + f". {beat} of {len(rows)} beat the index on risk-adjusted return over the "
        f"forward window; the best on that measure was {best}. Every one of those "
        "statements carries a standard error you should read before acting on it.")


def _forward_curves(rows: pd.DataFrame) -> dict[str, pd.Series]:
    """{label: forward nav}, plus SPY once, from the stored curves."""
    stored = store.load_curves(list(rows["forward_id"]))
    out: dict[str, pd.Series] = {}
    benchmark: pd.Series | None = None
    for _, r in rows.iterrows():
        block = stored.get(r["forward_id"], {}).get("forward")
        if block is None or "nav" not in block:
            continue
        out[str(r["strategy"])] = block["nav"].dropna()
        if benchmark is None and "benchmark" in block:
            b = block["benchmark"].dropna()
            if len(b) > 2:
                benchmark = b
    if benchmark is not None:
        out[BENCHMARK_LABEL] = benchmark
    return out


def _index_honesty(records: pd.DataFrame, rows: pd.DataFrame) -> Section:
    sec = Section("What would make you distrust all of this", blurb=(
        "Not an appendix. Every one of these changes what the numbers above mean."))

    worst_cov = pd.to_numeric(rows["coverage_min"], errors="coerce").min()
    dirty = int(rows["git_dirty"].astype(bool).sum()) if "git_dirty" in rows else 0
    repeats = int((pd.to_numeric(rows["look_number"], errors="coerce") > 1).sum())
    stale = int((pd.to_numeric(rows["fresh_months"], errors="coerce") == 0).sum())
    unresolved = int(pd.to_numeric(rows.get("unresolved_exits", 0),
                                   errors="coerce").fillna(0).sum())

    lines = [
        Cell("price coverage, worst rebalance", "coverage"),
        Cell(theme.pct(worst_cov), float(worst_cov) if _finite(worst_cov) else 0.0,
             emphasis="bad" if _lt(worst_cov, 0.9) else "good"),
        Cell("the share of the point-in-time index that actually had a price. Coverage "
             "is far better after 2022 than in 2007, which means the forward window is "
             "measured on a more complete universe than the research window - a "
             "difference that flatters neither consistently.", ""),
    ]
    facts = [lines]
    facts.append([
        Cell("repeat looks", "repeat"),
        Cell(theme.count(repeats), repeats, emphasis="warn" if repeats else "good"),
        Cell("candidates looked at more than once. A second look at unchanged data is "
             "the same measurement printed twice, not a confirmation.", ""),
    ])
    facts.append([
        Cell("looks with no fresh data", "stale"),
        Cell(theme.count(stale), stale, emphasis="warn" if stale else "good"),
        Cell("results whose entire window had already been seen by an earlier look at "
             "the same candidate.", ""),
    ])
    facts.append([
        Cell("unresolved delisting exits", "exits"),
        Cell(theme.count(unresolved), unresolved,
             emphasis="warn" if unresolved else "good"),
        Cell("positions that left the universe with no recorded reason and were "
             "liquidated at the last price - the wrong answer for a bankruptcy "
             "(ADR-021).", ""),
    ])
    facts.append([
        Cell("runs from a dirty tree", "dirty"),
        Cell(theme.count(dirty), dirty, emphasis="bad" if dirty else "good"),
        Cell("a result produced with uncommitted changes cannot be reproduced exactly.",
             ""),
    ])
    sec.add(TableBlock(Table(["", "", "why it matters"], facts,
                             aligns=["left", "right", "left"], sortable=False),
                       title="Diagnostics"))

    sec.add(Note(
        "The forward window is not a clean experiment repeated. It is one 4.6-year "
        "stretch of one market: a rate cycle, a 2022 drawdown, and a mega-cap rally "
        "that dominated 2023-2025. A strategy that avoids mega-caps was going to look "
        "bad here whatever its merits, and one that concentrates in them was going to "
        "look good. The verdicts above are about whether each strategy matched its own "
        "prediction, not about whether the period was representative - and it was not.",
        level="warn", title="One period, not a sample of periods."))

    sec.add(Note(
        f"{len(records)} forward runs are recorded across "
        f"{records['strategy'].nunique()} strategies and "
        f"{records['cost_model'].nunique()} cost settings. Every one of them is in the "
        "holdout ledger (`sp500lab experiments holdout`) and none of them can be "
        "withdrawn. The reserved period is now spent; anything further built on this "
        "data is research, not testing.",
        level="danger", title="The holdout is spent."))
    return sec


# ==========================================================================
# 2. One candidate, in full
# ==========================================================================

def forward_strategy_report(records: pd.DataFrame, *, cost_model: str = PRIMARY_COSTS,
                            trades_csv: str | None = None,
                            index_href: str = "index.html",
                            claim: str = "") -> Report:
    """Everything recorded about one candidate. `records` is its rows, one per cost."""
    rows = records.sort_values("logged_at")
    primary = rows[rows["cost_model"] == cost_model]
    row = (primary.iloc[-1] if len(primary) else rows.iloc[-1])
    record = store.get(str(row["forward_id"]))
    if record is None:                                        # pragma: no cover
        return Report(title=str(row["strategy"]), sections=[
            Section("Missing", [Note("The stored record could not be read.", "danger")])])

    comparison = compare(record.research_leg(), record.forward_leg())
    name = record.strategy
    report = Report(
        title=name,
        subtitle=f"Forward test, {record.forward_leg().start} to "
                 f"{record.forward_leg().end} — {record.verdict.upper()}",
        generated_at=_now(),
        meta={"verdict": record.verdict,
              "costs": record.cost_model,
              "mode": record.mode,
              "look": f"#{record.look_number}",
              "forward id": record.forward_id})

    report.add(_claim_section(record, claim, index_href))
    report.add(_paired_section(record, comparison))
    report.add(_curve_section(record))
    report.add(_year_section(record))
    report.add(_significance_section(record, comparison))
    report.add(_checks_section(comparison))
    report.add(_cost_section(rows))
    report.add(_provenance_section(record, trades_csv))
    return report


def _claim_section(record, claim: str, index_href: str) -> Section:
    leg_r, leg_f = record.research_leg(), record.forward_leg()
    sec = Section("The claim, and what happened to it", blurb=(
        claim or "The research window predicted the figures on the left. The forward "
        "window is the only place that prediction can be checked."))
    sec.add(StatRow([
        Stat("forward CAGR", theme.pct(leg_f.cagr),
             note=f"research {theme.pct(leg_r.cagr)}",
             emphasis="good" if leg_f.cagr > 0 else "bad"),
        Stat("forward Sharpe", theme.num(leg_f.sharpe),
             note=f"research {theme.num(leg_r.sharpe)}"),
        Stat("vs SPY", f"{leg_f.d_sharpe:+.2f}" if _finite(leg_f.d_sharpe) else "—",
             note="Sharpe, same dates",
             emphasis="good" if _gt(leg_f.d_sharpe, 0) else "bad"),
        Stat("max drawdown", theme.pct(leg_f.max_drawdown),
             note=f"research {theme.pct(leg_r.max_drawdown)}",
             emphasis="bad" if _lt(leg_f.max_drawdown, -0.35) else ""),
        Stat("turnover", theme.pct(leg_f.ann_turnover, 0), note="per year"),
        Stat("verdict", record.verdict.upper(),
             emphasis=VERDICT_EMPHASIS.get(record.verdict, "")),
    ]))
    sec.add(Note(_sentence(record.verdict_reason),
                 level=_note_level(record.verdict),
                 title=f"{record.verdict.upper()}."))
    sec.add(_power_note(leg_f.n_months))
    if record.rationale:
        sec.add(Note(
            f"“{record.rationale}” — recorded on seal {record.seal_id} "
            f"({record.seal_mode}) before the look.",
            level="info", title="Why this candidate was tested."))
    sec.add(LinkGrid([LinkCard(title="← all forward tests", href=index_href,
                               blurb="The executive summary and every other candidate.")]))
    return sec


def _note_level(verdict: str) -> str:
    return {"failed": "danger", "decayed": "warn"}.get(verdict, "info")


def _sentence(text: str) -> str:
    """Capitalise, so a verdict reason reads as prose after its bolded label."""
    text = (text or "").strip()
    return text[:1].upper() + text[1:] if text else text


def _paired_section(record, comparison) -> Section:
    leg_r, leg_f = record.research_leg(), record.forward_leg()
    metrics = [
        ("window", f"{leg_r.start} → {leg_r.end}", f"{leg_f.start} → {leg_f.end}", None),
        ("months", str(leg_r.n_months), str(leg_f.n_months), None),
        ("CAGR", theme.pct(leg_r.cagr), theme.pct(leg_f.cagr), comparison.decay_cagr),
        ("total return", theme.pct(_total(leg_r)), theme.pct(_total(leg_f)), None),
        ("Sharpe (daily)", theme.num(leg_r.sharpe), theme.num(leg_f.sharpe),
         comparison.decay_sharpe),
        ("Sharpe (monthly)", theme.num(leg_r.sharpe_monthly),
         theme.num(leg_f.sharpe_monthly), comparison.decay_sharpe_monthly),
        ("volatility", theme.pct(leg_r.ann_vol), theme.pct(leg_f.ann_vol), None),
        ("max drawdown", theme.pct(leg_r.max_drawdown), theme.pct(leg_f.max_drawdown),
         comparison.decay_max_drawdown),
        ("hit rate", theme.pct(leg_r.hit_rate), theme.pct(leg_f.hit_rate), None),
        ("SPY CAGR, same dates", theme.pct(leg_r.bench_cagr), theme.pct(leg_f.bench_cagr),
         None),
        ("SPY Sharpe, same dates", theme.num(leg_r.bench_sharpe),
         theme.num(leg_f.bench_sharpe), None),
        ("excess CAGR", theme.pct(leg_r.excess), theme.pct(leg_f.excess), None),
        ("Sharpe vs SPY", theme.num(leg_r.d_sharpe), theme.num(leg_f.d_sharpe),
         comparison.decay_d_sharpe),
        ("turnover", theme.pct(leg_r.ann_turnover, 0), theme.pct(leg_f.ann_turnover, 0),
         None),
        ("cost drag", theme.pct(leg_r.cost_drag), theme.pct(leg_f.cost_drag), None),
        ("names held", theme.num(leg_r.avg_positions, 0),
         theme.num(leg_f.avg_positions, 0), None),
    ]
    # The change column takes the row's OWN unit. A CAGR that fell 4.41 percentage
    # points and a Sharpe that fell 0.31 both render as "-0.04" and "-0.31" in raw
    # floats, and a column that mixes the two invites the reader to compare them.
    percent_rows = {"CAGR", "max drawdown"}
    rows = []
    for label, a, b, delta in metrics:
        if delta is None:
            change = Cell("", "")
        else:
            fmt = ((lambda v: theme.pct(v, signed=True)) if label in percent_rows
                   else (lambda v: f"{v:+.2f}"))
            change = _num_cell(delta, fmt)
        rows.append([Cell(label, label.lower()), Cell(a, a), Cell(b, b), change])
    return Section("Prediction against outcome", blurb=(
        "Left is what the research window said. Middle is what happened. Right is the "
        "difference, and the section below says how much of it is signal."
    )).add(TableBlock(Table(
        ["", "research 2007-2021", "forward 2022-2026", "change"], rows,
        aligns=["left", "right", "right", "right"], sortable=False), title="Both legs"))


def _total(leg) -> float:
    """Total return implied by the CAGR over the leg's own months."""
    if not (_finite(leg.cagr) and leg.n_months):
        return float("nan")
    return float((1 + leg.cagr) ** (leg.n_months / 12.0) - 1)


def _curve_section(record) -> Section:
    curves = store.load_curves([record.forward_id]).get(record.forward_id, {})
    forward = curves.get("forward")
    sec = Section("The curve")
    if forward is None or "nav" not in forward:
        sec.add(Note("No stored curve for this run.", "warn"))
        return sec

    nav = forward["nav"].dropna()
    lines = [S.equity(nav, record.strategy)]
    if "benchmark" in forward:
        lines.append(S.equity(forward["benchmark"].dropna(), BENCHMARK_LABEL,
                              kind="benchmark"))
    if "nav_gross" in forward:
        lines.append(S.equity(forward["nav_gross"].dropna(), "before costs",
                              kind="gross"))
    sec.add(LineChart(lines, title="Forward window, growth of 1.0",
                      y_format="multiple", height=320,
                      subtitle=f"{record.forward_leg().start} onward, from a fresh "
                               "$100k with an empty book"
                               if record.mode == "paired" else
                               "sliced from one unbroken run across the boundary",
                      caption="The gap between the strategy and “before costs” is the "
                              "cost drag: what trading this actually took out."))

    stitched = store.stitched_curve(record.forward_id)
    if stitched is not None and len(stitched) > 3:
        join = stitched.attrs.get("join_date", "")
        sec.add(LineChart(
            [S.equity(stitched, record.strategy)],
            title="Research and forward, spliced", y_format="multiple", log_y=True,
            height=300,
            subtitle=f"The forward leg begins {join}.",
            caption="A presentation, not a simulation: the two legs are independent "
                    "runs and the forward one started from a fresh book, so this "
                    "splices rather than continues. Log scale, because nineteen years "
                    "of compounding on a linear axis hides everything before 2015."))

    dd = S.drawdown(nav, record.strategy)
    if min(dd.finite_y or [0.0]) < -1e-9:
        sec.add(AreaChart(dd, title="Drawdown, forward window", height=170,
                          caption="From month-end values, so an intra-month trough is "
                                  "invisible and this is shallower than the true one. "
                                  "The maxDD in the tables comes from the daily curve."))
    return sec


def _year_section(record) -> Section:
    table = store.annual_table(record.forward_id)
    sec = Section("Year by year", blurb=(
        "Where the forward result actually came from. The first year is partial - the "
        "window opens on 2022-01-01 and the first trade is at the next month end - and "
        "it is measured from the run's own opening level rather than dropped."))
    if table.empty:
        sec.add(Note("No stored curve to derive annual returns from.", "warn"))
        return sec

    sec.add(BarChart(
        labels=[str(y) for y in table["year"]],
        values=[float(v) for v in table["strategy"]],
        title="Calendar-year return", y_format="pct", diverging=True, height=240))
    rows = []
    for _, r in table.iterrows():
        excess = r.get("excess")
        rows.append([
            Cell(str(r["year"]), str(r["year"])),
            _num_cell(r.get("strategy"), theme.pct,
                      emphasis="good" if _gt(r.get("strategy"), 0) else "bad"),
            _num_cell(r.get("benchmark"), theme.pct),
            _num_cell(excess, lambda v: theme.pct(v, signed=True),
                      emphasis="good" if _gt(excess, 0) else "bad"),
        ])
    sec.add(TableBlock(Table(["year", "strategy", "SPY", "excess"], rows,
                             caption="SPY over exactly the same dates.")))

    curves = store.load_curves([record.forward_id]).get(record.forward_id, {})
    forward = curves.get("forward")
    if forward is not None and "nav" in forward:
        years, months, values = S.monthly_grid(forward["nav"].dropna())
        if years:
            sec.add(Heatmap(rows=years, cols=months, values=values,
                            title="Monthly returns", value_format="pct",
                            caption="Reading across a row shows how a year was earned; "
                                    "a single annual number hides it."))
    return sec


def _significance_section(record, comparison) -> Section:
    leg_f = record.forward_leg()
    lo, hi = sharpe_band(leg_f.sharpe_monthly, leg_f.n_months)
    sec = Section("Is the change real?", blurb=(
        "The tempting reading of a forward test is that a Sharpe fell from A to B, "
        "therefore the strategy decayed. On 54 monthly observations that inference is "
        "usually unavailable, and this section is where it is checked rather than "
        "assumed."))
    sec.add(StatRow([
        Stat("Sharpe change", f"{comparison.decay_sharpe_monthly:+.2f}"
             if _finite(comparison.decay_sharpe_monthly) else "—",
             note="annualised, monthly"),
        Stat("standard error", theme.num(comparison.decay_se),
             note="of that change"),
        Stat("sigma", f"{comparison.decay_z:+.2f}"
             if _finite(comparison.decay_z) else "—",
             note="change ÷ its own error",
             emphasis="bad" if _lt(comparison.decay_z, -1.96) else
                      ("warn" if _lt(comparison.decay_z, -1) else "good")),
        Stat("95% band", f"[{lo:.2f}, {hi:.2f}]" if _finite(lo) else "—",
             note="around the forward Sharpe"),
    ], title="The gap, and the error on it"))

    rows = [
        ["P(true Sharpe > 0)", comparison.psr_vs_zero,
         "did it make risk-adjusted money at all, out of sample?"],
        ["P(true Sharpe > SPY)", comparison.psr_vs_benchmark,
         "did it beat the index over the same dates?"],
        ["P(true Sharpe > research)", comparison.psr_vs_research,
         "did it live up to what the research window promised? This is the one this "
         "harness exists to compute."],
    ]
    sec.add(TableBlock(Table(
        ["probability", "value", "the question it answers"],
        [[Cell(a, a), _num_cell(b, theme.num,
                                emphasis="good" if _gt(b, 0.95) else
                                         ("bad" if _lt(b, 0.5) else "")),
          Cell(c, c)] for a, b, c in rows],
        aligns=["left", "right", "left"], sortable=False,
        caption="Probabilistic Sharpe ratios, corrected for skew and fat tails, with "
                "the benchmark on the right-hand side of the comparison changed. Read "
                "them as probabilities, not scores.")))

    if record.n_trials:
        sec.add(Note(
            f"This candidate came out of the search “{record.study}”, which evaluated "
            f"{record.n_trials:,} distinct configurations. Its research-window deflated "
            f"Sharpe was {theme.num(record.deflated_sharpe_research)} — the probability "
            "it survived the search that produced it. A forward test does not replace "
            "that number; it asks a different question of the same candidate.",
            level="warn" if record.n_trials > 100 else "info",
            title=f"Searched {record.n_trials:,} times before it got here."))
    else:
        sec.add(Note(
            "No earlier run of this exact configuration is in the trial log, so the "
            "search behind this candidate is unknown and the research Sharpe above is "
            "NOT corrected for it. Treat the research figure as an upper bound.",
            level="warn", title="Search context unknown."))
    return sec


def _checks_section(comparison) -> Section:
    failed = [c.name for c in comparison.checks if c.passed is False]
    return Section("The checks", blurb=(
        "Nine named conditions, each independently readable. The verdict is computed "
        "from them in a fixed precedence, so it can be argued with rather than just "
        "quoted."
        + (f" Failed here: {', '.join(failed)}." if failed else " All passed.")
    )).add(TableBlock(_checks_table(comparison)))


def _cost_section(rows: pd.DataFrame) -> Section:
    sec = Section("Under all three cost settings", blurb=(
        "Always all three. A strategy that only works under `optimistic` is a bet that "
        "the half-spread estimator is wrong in your favour, not a strategy — and the "
        "forward window was taken under all three in a single look, so no second look "
        "was needed to produce this table."))
    order = {"optimistic": 0, "realistic": 1, "pessimistic": 2}
    ordered = rows.sort_values("cost_model", key=lambda s: s.map(order).fillna(9))
    table_rows = []
    for _, r in ordered.iterrows():
        verdict = str(r.get("verdict", ""))
        table_rows.append([
            Cell(str(r["cost_model"]), order.get(str(r["cost_model"]), 9)),
            _num_cell(r.get("forward_cagr"), theme.pct,
                      emphasis="good" if _gt(r.get("forward_cagr"), 0) else "bad"),
            _num_cell(r.get("forward_sharpe"), theme.num),
            _num_cell(r.get("forward_d_sharpe"), lambda v: f"{v:+.2f}"),
            _num_cell(r.get("forward_cost_drag"), theme.pct),
            _num_cell(r.get("total_cost"), theme.money),
            Cell(verdict.upper(), order.get(verdict, 9),
                 emphasis=VERDICT_EMPHASIS.get(verdict, "")),
        ])
    sec.add(TableBlock(Table(
        ["costs", "forward CAGR", "Sharpe", "vs index", "cost drag", "charged",
         "verdict"], table_rows, sortable=False)))

    cagrs = pd.to_numeric(ordered["forward_cagr"], errors="coerce")
    if len(cagrs) == 3 and cagrs.iloc[0] > 0 and cagrs.iloc[2] <= 0:
        sec.add(Note(
            "Profitable out of sample under optimistic costs and not under pessimistic "
            "ones. That is a bet on the spread estimator, not a strategy.",
            level="danger", title="Only works if trading is free."))
    return sec


def _provenance_section(record, trades_csv: str | None) -> Section:
    sec = Section("Provenance and honesty", blurb=(
        "What it would take to reproduce this, and what would make you distrust it."))
    rows = [
        ("forward id", record.forward_id, ""),
        ("seal", f"{record.seal_id} ({record.seal_mode})",
         "declared means pre-registered before the look; auto means sealed at run time"),
        ("look number", f"#{record.look_number}",
         "how many times this candidate has been forward-tested at this cost setting"),
        ("fresh months", f"{record.fresh_months} of {record.forward_leg().n_months}",
         "months of this result that no earlier look had already seen"),
        ("data vintage", record.data_end,
         "the last session in the panel when this ran"),
        ("seal drift", theme.num(record.seal_drift_sharpe),
         "sealed research Sharpe minus the one re-measured at test time; non-zero means "
         "the data changed under the prediction"),
        ("mode", record.mode,
         "paired = a fresh $100k in 2022; continuous = one unbroken run, sliced"),
        ("rebalances", theme.count(record.n_rebalances), ""),
        ("orders", theme.count(record.n_orders), ""),
        ("price coverage", f"{theme.pct(record.coverage_median)} median, "
                           f"{theme.pct(record.coverage_min)} worst",
         "share of the point-in-time index that actually had a price"),
        ("forced exits", f"{record.forced_exits} ({record.unresolved_exits} unresolved)",
         "positions resolved outside a rebalance; unresolved ones default to a "
         "liquidation at the last price"),
        ("unfilled orders", theme.count(record.unfilled_orders),
         "orders with no opening bar to fill against"),
        ("spread fallbacks", theme.count(record.spread_fallback_orders),
         "orders charged the fallback half-spread because the estimator had no value"),
        ("holdout looks at the time", theme.count(record.holdout_looks_total),
         "how degraded the reserved period already was when this ran"),
        ("research run", record.research_run_id or "—", "in the trial log"),
        ("forward run", record.forward_run_id or "—", "in the trial log"),
        ("git", record.git_commit[:12] + (" DIRTY" if record.git_dirty else ""),
         "a result from a dirty tree is not reproducible"),
    ]
    sec.add(TableBlock(Table(
        ["", "", "what it means"],
        [[Cell(a, a), Cell(str(b), str(b),
                           emphasis="bad" if a == "git" and record.git_dirty else ""),
          Cell(c, c)] for a, b, c in rows],
        aligns=["left", "right", "left"], sortable=False)))

    if trades_csv:
        sec.add(_trades_download(record, trades_csv))
    return sec


def _trades_download(record, csv_path) -> Download | None:
    """The forward window's orders, embedded so the page stays self-contained."""
    from pathlib import Path

    path = Path(csv_path)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:                                           # pragma: no cover
        return None
    lines = text.splitlines()
    note = (f"{len(lines) - 1:,} orders over the forward window. `price` is the "
            "as-traded open — what a broker printed that morning — so it compares "
            "directly against a quote site.")
    if len(lines) - 1 > MAX_EMBEDDED_TRADES:
        text = "\n".join(lines[:MAX_EMBEDDED_TRADES + 1])
        note += (f" Truncated to the first {MAX_EMBEDDED_TRADES:,} for embedding; the "
                 f"complete file is at {path}.")
    return Download(filename=f"forward-{slugify(record.strategy)}-trades.csv",
                    content=text, label="Download every forward order", note=note)


# ==========================================================================
# 3. The cross-section: did research predict anything?
# ==========================================================================

def forward_decay_report(records: pd.DataFrame,
                         cost_model: str = PRIMARY_COSTS) -> Report:
    """Across all candidates: did the research ranking predict the forward ranking?

    The question no single-strategy report can answer, and the one that decides whether
    the research window is worth anything as a selection device. If the rank correlation
    is around zero, then picking the best strategy on research evidence was no better
    than picking one at random - which is a finding about the whole method, not about
    any strategy in it.
    """
    rows = primary_rows(records, cost_model)
    report = Report(
        title="Did the research window predict anything?",
        subtitle="The cross-section of predictions against outcomes.",
        generated_at=_now(),
        meta={"candidates": theme.count(len(rows)), "cost model": cost_model})
    if len(rows) < 3:
        report.add(Section("Not enough candidates", [Note(
            "At least three forward tests are needed before a cross-sectional "
            "correlation means anything.", "warn")]))
        return report

    stats = rank_agreement(rows)
    sec = Section("The headline", blurb=(
        "Every candidate was ranked twice: once by the research window, once by the "
        "forward window. If research selection works, the two rankings agree."))
    sec.add(StatRow([
        Stat("rank correlation", theme.num(stats["spearman_d_sharpe"]),
             note="research vs forward, Sharpe against the index",
             emphasis="good" if _gt(stats["spearman_d_sharpe"], 0.3) else
                      ("bad" if _lt(stats["spearman_d_sharpe"], 0) else "warn")),
        Stat("top-5 overlap", f"{stats['top5_overlap']} of 5",
             note="best five by research, still in the best five forward"),
        Stat("mean Sharpe change", f"{stats['mean_decay']:+.2f}",
             note="forward minus research",
             emphasis="bad" if stats["mean_decay"] < -0.2 else ""),
        Stat("candidates that improved", f"{stats['improved']} of {len(rows)}",
             note="higher Sharpe forward than in research"),
    ]))
    sec.add(Note(_rank_prose(stats, len(rows)), level="warn",
                 title="What the correlation means."))
    report.add(sec)

    report.add(Section("The scatter", blurb=(
        "Both axes are measured against SPY over the strategy's own dates, so neither "
        "is a ranking of the market."
    )).add(_decay_scatter(rows)).add(_decay_bars(rows)))

    report.add(Section("Ranked both ways", blurb=(
        "Research rank against forward rank, candidate by candidate. A large move is "
        "either a real change in behaviour or 54 months of noise, and the sigma column "
        "on the executive summary says which is more likely."
    )).add(TableBlock(_rank_table(rows))))

    report.add(Section("Regime, not skill", blurb="").add(Note(
        "Before reading a low correlation as proof that research selection is "
        "worthless, note what the forward window was: a 2022 drawdown, a rate cycle, "
        "and a 2023-2025 rally concentrated in a handful of mega-caps. Almost every "
        "strategy here is equal-weighted or tilts small, and against a cap-weighted "
        "index in that period they were structurally disadvantaged regardless of "
        "signal quality. A rank correlation measures agreement between two windows; it "
        "cannot separate “the research window was uninformative” from “the forward "
        "window was unusual”, and 4.6 years is nowhere near enough to try.",
        level="warn", title="One period, and an unusual one.")))
    return report


def rank_agreement(rows: pd.DataFrame) -> dict:
    """Spearman correlation and overlap between the research and forward rankings.

    Spearman rather than Pearson: the question is whether the ORDER survived, and a
    single outlier on either axis would dominate a linear correlation on twenty points.
    Hand-rolled rather than scipy - the project runs on four libraries and this is a
    rank, a mean and a covariance.
    """
    r = pd.to_numeric(rows["research_d_sharpe"], errors="coerce")
    f = pd.to_numeric(rows["forward_d_sharpe"], errors="coerce")
    ok = r.notna() & f.notna()
    r, f = r[ok], f[ok]

    top_r = set(rows.loc[r.sort_values(ascending=False).index[:5], "strategy"])
    top_f = set(rows.loc[f.sort_values(ascending=False).index[:5], "strategy"])
    decay = pd.to_numeric(rows["decay_sharpe_monthly"], errors="coerce").dropna()
    return {
        "spearman_d_sharpe": _spearman(r, f),
        "spearman_cagr": _spearman(pd.to_numeric(rows["research_cagr"],
                                                 errors="coerce"),
                                   pd.to_numeric(rows["forward_cagr"],
                                                 errors="coerce")),
        "top5_overlap": len(top_r & top_f),
        "mean_decay": float(decay.mean()) if len(decay) else float("nan"),
        "improved": int((decay > 0).sum()),
        "n": int(len(r)),
    }


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation, or NaN when one side has no spread to correlate.

    Guarded rather than left to pandas: a column where every candidate scores the same
    has zero rank variance, and dividing by it emits a numpy warning and returns NaN
    anyway. Returning NaN deliberately says "not computable" instead.
    """
    ok = a.notna() & b.notna()
    a, b = a[ok].rank(), b[ok].rank()
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.corr(b))


def _rank_prose(stats: dict, n: int) -> str:
    rho = stats["spearman_d_sharpe"]
    if not _finite(rho):
        return "Too few comparable candidates to compute a rank correlation."
    if rho > 0.5:
        reading = ("a strong agreement: the research window's ranking largely survived "
                   "into the forward window")
    elif rho > 0.2:
        reading = ("a weak positive agreement - the research ranking carried some "
                   "information, but not much")
    elif rho > -0.2:
        reading = ("no agreement worth the name. On this evidence, choosing a strategy "
                   "by its research-window ranking was close to choosing one at random")
    else:
        reading = ("an INVERSE agreement: the strategies research liked best did worst "
                   "out of sample. On a sample this small that is more likely to be "
                   "noise than a genuine reversal, but it is not evidence of skill")
    return (f"Spearman rank correlation of {rho:+.2f} across {stats['n']} candidates — "
            f"{reading}. With {n} points the standard error of a rank correlation is "
            f"roughly {1 / math.sqrt(max(n - 1, 1)):.2f}, so anything inside about "
            f"±{2 / math.sqrt(max(n - 1, 1)):.2f} is indistinguishable from zero.")


def _rank_table(rows: pd.DataFrame) -> Table:
    r = pd.to_numeric(rows["research_d_sharpe"], errors="coerce")
    f = pd.to_numeric(rows["forward_d_sharpe"], errors="coerce")
    rank_r = r.rank(ascending=False)
    rank_f = f.rank(ascending=False)
    out = []
    for i, (_, row) in enumerate(rows.iterrows()):
        move = float(rank_r.iloc[i] - rank_f.iloc[i])
        out.append([
            Cell(str(row["strategy"]), str(row["strategy"]).lower(),
                 href=strategy_href(str(row["strategy"]))),
            _num_cell(rank_r.iloc[i], lambda v: f"{int(v)}"),
            _num_cell(rank_f.iloc[i], lambda v: f"{int(v)}"),
            _num_cell(move, lambda v: f"{int(v):+d}",
                      emphasis="good" if move > 2 else ("bad" if move < -2 else "")),
            _num_cell(row.get("research_d_sharpe"), lambda v: f"{v:+.2f}"),
            _num_cell(row.get("forward_d_sharpe"), lambda v: f"{v:+.2f}"),
            Cell(str(row.get("verdict", "")).upper(), str(row.get("verdict", "")),
                 emphasis=VERDICT_EMPHASIS.get(str(row.get("verdict", "")), "")),
        ])
    return Table(["strategy", "research rank", "forward rank", "move",
                  "research vs index", "forward vs index", "verdict"], out,
                 caption="Rank 1 is the best. A positive `move` means it climbed.")


# ==========================================================================
# 4. Honesty
# ==========================================================================

def forward_honesty_report(records: pd.DataFrame) -> Report:
    """Everything that limits what the forward tests above are worth."""
    report = Report(
        title="Forward tests: what limits them",
        subtitle="Read this before quoting any number from the other reports.",
        generated_at=_now(),
        meta={"forward runs": theme.count(len(records)),
              "strategies": theme.count(records["strategy"].nunique()) if len(records)
                            else "0"})
    if records.empty:
        report.add(Section("Nothing to qualify", [Note("No forward test has been run.",
                                                       "warn")]))
        return report

    rows = primary_rows(records)
    months = int(pd.to_numeric(rows["forward_n_months"], errors="coerce").max())

    report.add(Section("The one that outranks the rest", blurb=(
        "If you read nothing else on this page, read this."
    )).add(_power_note(months)).add(TableBlock(_power_table())))

    report.add(Section("Multiple testing", blurb=(
        "The reserved period protects against fitting. It does not protect against "
        "choosing, and this set of reports is itself an exercise in choosing."
    )).add(_selection_note(store.selection_bar(records)) or Note("—"))
        .add(_seal_note(rows)))

    report.add(Section("The period", blurb="").add(Note(
        "2022-01 to 2026-08 is one 4.6-year stretch of one market. It contains a rate "
        "cycle, a 2022 drawdown and a rally concentrated in a handful of mega-caps. "
        "Every strategy in this project is long-only, monthly and roughly "
        "equal-weighted within its selection, so against a cap-weighted index in that "
        "period they were structurally disadvantaged whatever their signal quality. "
        "That cuts both ways: it makes a `failed` verdict here weaker evidence than it "
        "looks, and a `held` verdict stronger.",
        level="warn", title="A regime, not a sample.")))

    report.add(Section("Everything the engine already warns about", blurb=(
        "The forward window inherits every limitation of the backtest engine, and two "
        "of them behave differently after 2022."
    )).add(TableBlock(_inherited_table(rows))))

    report.add(Section("The ledger", blurb=(
        "Every look at the reserved period, and it cannot be undone."
    )).add(TableBlock(_ledger_table(records))).add(Note(
        "The reserved period is now spent. Anything built on it from here is research, "
        "not testing, and the only genuinely out-of-sample data this project will ever "
        "have again is the months that have not happened yet. `forward window` reports "
        "how many of those have accumulated since each candidate was last tested.",
        level="danger", title="There is no second holdout.")))
    return report


def _power_table() -> Table:
    rows = []
    for n in (24, 36, 54, 120, 240):
        lo, hi = sharpe_band(1.0, n)
        rows.append([
            Cell(f"{n} months", n),
            Cell(f"[{lo:.2f}, {hi:.2f}]", lo),
            Cell(f"±{(hi - lo) / 2:.2f}", (hi - lo) / 2,
                 emphasis="bad" if (hi - lo) / 2 > 0.7 else ""),
        ])
    return Table(["window", "95% band on a Sharpe of 1.0", "half-width"], rows,
                 sortable=False,
                 caption="The forward window is 54 months. Two independent estimates "
                         "carry √2 times one error, so the smallest research-to-forward "
                         "difference it could resolve is about 1.35 Sharpe — larger "
                         "than almost any real effect.")


def _inherited_table(rows: pd.DataFrame) -> Table:
    cov_min = pd.to_numeric(rows["coverage_min"], errors="coerce").min()
    unresolved = int(pd.to_numeric(rows.get("unresolved_exits", 0),
                                   errors="coerce").fillna(0).sum())
    fallbacks = int(pd.to_numeric(rows.get("spread_fallback_orders", 0),
                                  errors="coerce").fillna(0).sum())
    items = [
        ("price coverage", theme.pct(cov_min),
         "Worst rebalance in the forward window. Coverage is near-complete after 2022 "
         "and was 55% in 2007, so the forward leg trades a MORE complete universe than "
         "the research leg. That is a difference between the two windows which has "
         "nothing to do with any strategy (ADR-023)."),
        ("delisting assumptions", theme.count(unresolved),
         "Unresolved exits, liquidated at the last price. Far rarer after 2022 than "
         "before 2010, so this limitation binds mostly on the research leg (ADR-021)."),
        ("estimated spreads", theme.count(fallbacks),
         "Orders charged the fallback half-spread. Half-spreads are estimated "
         "throughout, never quoted, which is why all three cost settings are always "
         "reported (ADR-020)."),
        ("survivorship, second order", "—",
         "Anything using SEC fundamentals covers 649 of 973 historical members, and "
         "coverage correlates with survival. Those strategies run on a shorter, kinder "
         "research window, which makes their research figure the flattered one."),
        ("execution", "next open",
         "Signals form on a close and fill at the following open (ADR-019). No impact "
         "model: at $100k that is defensible, and it would not be at scale."),
    ]
    return Table(["", "", "what it does to these numbers"],
                 [[Cell(a, a), Cell(b, b), Cell(c, c)] for a, b, c in items],
                 aligns=["left", "right", "left"], sortable=False)


def _ledger_table(records: pd.DataFrame) -> Table:
    from ..backtest import registry

    df = registry.holdout_touches()
    if df.empty:
        return Table(["at", "strategy", "mode", "start", "end"], [], sortable=False)
    show = df.tail(200)
    rows = [[Cell(str(r.get("at", "")), str(r.get("at", ""))),
             Cell(str(r.get("strategy", "")), str(r.get("strategy", ""))),
             Cell(str(r.get("mode", "")), str(r.get("mode", ""))),
             Cell(str(r.get("start", "")), str(r.get("start", ""))),
             Cell(str(r.get("end", "")), str(r.get("end", "")))]
            for _, r in show.iterrows()]
    return Table(["at", "strategy", "mode", "start", "end"], rows,
                 caption=f"{len(df)} recorded look(s) at the reserved period"
                         + (f"; the most recent {len(show)} shown." if len(show) < len(df)
                            else "."))
