"""The Algorithm Book: every competitor explained, then scored, on one page.

Pure view, same contract as views.py: prepared data in, a `Report` out, no I/O and no
markup. The CLI gathers (registry rows, docstrings, benchmark stats, forward records)
and this file decides what a reader sees and in what order.

Editorial stance, because a page like this can mislead by layout alone:

  * Explanations come from each strategy's own docstring, the same source the strategy
    pages use, so the book and the code cannot drift apart.
  * The scoreboard sorts on Delta-Sharpe against the index over each strategy's OWN
    window - never on raw CAGR, which ranks windows rather than skill.
  * Searched and written strategies are labelled, with the deflated Sharpe and trial
    count beside anything searched; a calendar rule in cash most of the time is
    labelled too, because its Sharpe gets a structural boost nothing earned.
  * Forward (2022+) columns appear wherever a candidate has been tested, with the
    standing caveat that 54 months refutes and never confirms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import series as S
from . import theme
from .specs import (LineChart, LinkCard, LinkGrid, Note, Report, Section, Stat,
                    StatRow, TableBlock)
from .tables import Cell, Table, _cell, _text


@dataclass
class AlgorithmEntry:
    """Everything the book knows about one competitor. Assembled by the CLI."""

    name: str
    family: str                       # display label, e.g. "The twelve hypotheses"
    engine: str                       # "monthly" | "daily legs"
    origin: str                       # "written" | "searched" | "rule"
    claim: str = ""                   # first docstring paragraph
    explain: list[str] = field(default_factory=list)   # further paragraphs
    window: str = ""
    settings: dict = field(default_factory=dict)       # cost_model -> stats dict
    bench: dict | None = None                          # {"cagr","sharpe"} same window
    curve: pd.DataFrame | None = None                  # nav [+ benchmark, nav_gross]
    deflation: dict | None = None                      # study/n_trials/deflated_sharpe
    forward: dict | None = None                        # verdict/sharpes/decay
    href: str = ""                                     # per-strategy page, if any
    exposure: str = ""                                 # e.g. "19% of sessions"

    @property
    def realistic(self) -> dict:
        return self.settings.get("realistic", {})

    @property
    def d_sharpe(self) -> float | None:
        r, b = self.realistic.get("sharpe"), (self.bench or {}).get("sharpe")
        if r is None or b is None:
            return None
        return float(r) - float(b)


#: Family display order. Baselines first because they are the nulls everything else is
#: read against; the searched families last because their numbers need the most salt.
FAMILY_ORDER = (
    "The null hypotheses",
    "The twelve hypotheses",
    "The second wave",
    "Learned models",
    "Evolved by the genetic algorithm",
    "The calendar rules",
)

FAMILY_BLURBS = {
    "The null hypotheses": (
        "Baselines with little or no signal. They exist to calibrate everything below: "
        "a stock picker that loses to equal weight is not picking stocks, and any idea "
        "inside random_weight's spread across seeds has demonstrated nothing."),
    "The twelve hypotheses": (
        "Hand-written cross-sectional strategies, each a different economic claim - "
        "under-reaction, accounting quality, liquidity, index mechanics, payout "
        "behaviour, lottery preference. Chosen to disagree with each other, so if "
        "several work they should not all work for the same reason."),
    "The second wave": (
        "Five mechanisms the first twelve leave uncovered: WHEN returns accrue "
        "(overnight vs intraday), how much to be invested at all (volatility "
        "management), the dividend calendar, anchoring to the 52-week high, and the "
        "plain average of every other hypothesis. Written after the 2022-2026 forward "
        "test was read - see the honesty note below."),
    "Learned models": (
        "Models refit on a trailing window at every rebalance, with no training label "
        "reaching the as-of date. The discipline is the deliverable: swap the ridge "
        "solve for any model and the leakage guarantee is unchanged."),
    "Evolved by the genetic algorithm": (
        "Winners of registered genetic searches over weighted sums of ranked features. "
        "Every individual each search evaluated was logged, which is what makes the "
        "deflated Sharpe beside each winner meaningful - a searched Sharpe without its "
        "trial count is not a number."),
    "The calendar rules": (
        "Fixed schedules on SPY at daily granularity - overnight, weekend, turn of "
        "month, pre-holiday, Halloween, and a VIX-gated overnight. No parameters, no "
        "fitting; the question each asks is whether a documented calendar effect "
        "survives realistic retail costs. Sharpe ratios here get a structural boost "
        "from time in cash - read Delta-Sharpe and time-invested together."),
}


def algorithms_report(entries: list[AlgorithmEntry], *, ga: dict | None = None,
                      curves: dict[str, pd.Series] | None = None,
                      forward_context: dict | None = None,
                      generated_at: str = "", index_href: str = "index.html",
                      doc_links: list[dict] | None = None) -> Report:
    """The whole book. `entries` in any order; the view imposes the family order."""
    by_family: dict[str, list[AlgorithmEntry]] = {}
    for e in entries:
        by_family.setdefault(e.family, []).append(e)
    n_beat = sum(1 for e in entries if (e.d_sharpe or 0) > 0)
    n_forward = sum(1 for e in entries if e.forward)

    report = Report(
        title="The Algorithm Book",
        subtitle="Every competitor in the lab - what it claims, how it works, and "
                 "what it actually made",
        generated_at=generated_at,
        meta={"algorithms": theme.count(len(entries)),
              "families": theme.count(len(by_family)),
              "beat the index (research)": theme.count(n_beat),
              "forward-tested": theme.count(n_forward)},
    )

    report.add(_how_to_read(entries, n_beat))
    report.add(_scoreboard(entries))
    if curves:
        report.add(_every_curve(curves))
    for family in FAMILY_ORDER:
        if family in by_family:
            report.add(_family_section(family, by_family[family]))
    if ga:
        report.add(_ga_section(ga))
    report.add(_forward_section(entries, forward_context))
    report.add(_links(index_href, doc_links or []))
    return report


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

def _how_to_read(entries: list[AlgorithmEntry], n_beat: int) -> Section:
    s = Section(
        "How to read this",
        blurb="One page, every algorithm in the project: the six null baselines, the "
              "twelve hand-written hypotheses, the five second-wave mechanisms, the "
              "learned models, the genetic algorithm's winners, and the nine calendar "
              "rules. Each is explained in its own words below, then scored the only "
              "way that means anything - against buy-and-hold SPY over its own dates, "
              "under realistic costs, with the multiple-testing context beside it.")
    s.add(StatRow([
        Stat("The bar", "SPY, same window",
             "every row is scored against the index over its own dates"),
        Stat("Costs", "3 settings, always",
             "optimistic / realistic / pessimistic; tables show realistic"),
        Stat("Beat the index", f"{n_beat} of {len(entries)}",
             "on risk-adjusted return, research window",
             "good" if n_beat else "bad"),
        Stat("Research window", "2007-05 → 2021-12",
             "2022+ was a reserved holdout, spent 2026-08"),
    ]))
    s.add(Note(
        "Most of these do not beat the index, and that is the correct null result on "
        "daily bars over large caps with free data. The value of the losers is that "
        "they fail on an honest harness: same universe, same costs, same accounting, "
        "every trial counted.", level="info", title="Expect losers."))
    s.add(Note(
        "Everything added in the 2026-08 wave (the second wave, the calendar rules, "
        "the night-preset search) was written AFTER the 2022-2026 forward test was "
        "run and read. Nothing was fitted to that period, but the author knew its "
        "character when choosing what to build, and no statistic on this page can "
        "correct for that. Their only clean test is data that arrives after 2026-08.",
        level="warn", title="Contamination, recorded."))
    return s


def _scoreboard(entries: list[AlgorithmEntry]) -> Section:
    ordered = sorted(entries, key=lambda e: (e.d_sharpe if e.d_sharpe is not None
                                             else float("-inf")), reverse=True)
    rows = []
    for e in ordered:
        r = e.realistic
        d = e.d_sharpe
        fwd = e.forward or {}
        verdict = fwd.get("verdict", "")
        defl = (e.deflation or {}).get("deflated_sharpe")
        rows.append([
            Cell(e.name, sort_key=e.name, href=e.href, title=e.claim[:160]),
            _text(e.family.replace("The ", ""), emphasis="muted"),
            _text(e.window),
            _cell(r.get("cagr"), theme.pct),
            _cell(r.get("sharpe"), theme.num),
            _cell(r.get("maxdd"), theme.pct),
            _cell((e.bench or {}).get("sharpe"), theme.num),
            _cell(d, lambda v: theme.num(v, dp=2),
                  emphasis="good" if (d or 0) > 0 else ("bad" if d is not None else "")),
            _cell(r.get("cost_drag"), theme.pct),
            _cell(defl, lambda v: theme.num(v, dp=2)) if defl is not None
            else _text("—", emphasis="muted"),
            _text(verdict or "—",
                  emphasis={"held": "good", "decayed": "warn",
                            "failed": "bad"}.get(verdict, "muted")),
        ])
    table = Table(
        columns=["strategy", "family", "window", "CAGR", "Sharpe", "maxDD",
                 "SPY Sharpe", "ΔSharpe", "cost drag", "deflated", "forward"],
        rows=rows,
        caption="Realistic costs. ΔSharpe = strategy minus SPY over the strategy's "
                "own window - the sort key and the only column that ranks skill. "
                "'deflated' is the deflated Sharpe where the strategy came from a "
                "registered search; 'forward' is the 2022-2026 verdict where one "
                "exists (held means NOT refuted, nothing stronger).")
    s = Section("The scoreboard",
                blurb="Which ones made money, relative to the only free alternative - "
                      "buying the index and doing nothing.")
    s.add(TableBlock(table))
    return s


def _every_curve(curves: dict[str, pd.Series]) -> Section:
    aligned = S.align(curves)
    chart = LineChart(
        [S.equity(c, name, kind="benchmark" if name == "SPY" else "strategy")
         for name, c in aligned.items()],
        title="Growth of 1.0, common window, realistic costs",
        y_format="multiple", log_y=True, height=380,
        caption="Click a legend entry to dim a series everywhere on the page.")
    s = Section("Every curve at once",
                blurb="The whole competition on one chart, rebased to a common start.")
    s.add(chart)
    return s


def _family_section(family: str, members: list[AlgorithmEntry]) -> Section:
    s = Section(family, blurb=FAMILY_BLURBS.get(family, ""))

    rows = []
    for e in sorted(members, key=lambda e: (e.d_sharpe if e.d_sharpe is not None
                                            else float("-inf")), reverse=True):
        r = e.realistic
        d = e.d_sharpe
        rows.append([
            Cell(e.name, sort_key=e.name, href=e.href),
            _text(e.window),
            _cell(r.get("cagr"), theme.pct),
            _cell(r.get("sharpe"), theme.num),
            _cell(r.get("maxdd"), theme.pct),
            _cell(r.get("turnover"), lambda v: theme.pct(v, dp=0)),
            _cell(r.get("cost_drag"), theme.pct),
            _cell(d, lambda v: theme.num(v, dp=2),
                  emphasis="good" if (d or 0) > 0 else ("bad" if d is not None else "")),
        ])
    s.add(TableBlock(Table(
        columns=["strategy", "window", "CAGR", "Sharpe", "maxDD", "turnover",
                 "cost drag", "ΔSharpe"],
        rows=rows), title="Results, realistic costs"))

    for e in members:
        text = e.claim
        for para in e.explain[:2]:
            text += "\n\n" + para
        extra = []
        if e.exposure:
            extra.append(e.exposure)
        if e.deflation and e.deflation.get("n_trials"):
            extra.append(f"From a registered search of "
                         f"{e.deflation['n_trials']} trials; deflated Sharpe "
                         f"{theme.num(e.deflation.get('deflated_sharpe'), dp=2)}.")
        if extra:
            text += "\n\n" + " ".join(extra)
        s.add(Note(text, title=e.name))
    return s


def _ga_section(ga: dict) -> Section:
    """One write-up for the whole genetic family; the full story lives in the docs."""
    s = Section(
        "The genetic algorithm, in one panel",
        blurb="A strategy here is a short list of numbers - a weight per ranked "
              "feature, how many names to hold, whether to de-risk in a falling "
              "market. Any such list backtests in ~0.15s, so the search breeds a "
              "population of them: keep the best, cross and mutate, repeat. The "
              "search space is deliberately small and readable - weighted sums of "
              "ranked features, no expression trees - because a winning parameter "
              "vector nobody can read is one nobody can check.")
    s.add(Note(
        "History does not push back. Grade 1,400 random strategies against a fixed "
        "past and the best will look brilliant by luck alone, so every individual "
        "every search evaluates is logged as a trial, and each winner is reported "
        "with its deflated Sharpe: the probability its result exceeds what the "
        "luckiest of N worthless candidates would have posted. Below ~0.95, a "
        "searched result is indistinguishable from selection bias.",
        level="warn", title="Why the trial count is the whole game."))
    rows = []
    for run in ga.get("searches", []):
        rows.append([
            _text(run.get("study", "")),
            _text(run.get("preset", "")),
            _cell(run.get("n_trials"), theme.count),
            _text(run.get("window", "")),
            _cell(run.get("cagr"), theme.pct),
            _cell(run.get("sharpe_monthly"), theme.num),
            _cell(run.get("expected_max"), theme.num,
                  title="what the luckiest worthless trial would have posted"),
            _cell(run.get("deflated_sharpe"), lambda v: theme.num(v, dp=3),
                  emphasis="good" if (run.get("deflated_sharpe") or 0) >= 0.95
                  else "warn"),
            _text(run.get("forward", "") or "—",
                  emphasis={"held": "good", "decayed": "warn",
                            "failed": "bad"}.get(run.get("forward", ""), "muted")),
        ])
    if rows:
        s.add(TableBlock(Table(
            columns=["search", "features", "trials", "window", "best CAGR",
                     "best Sharpe (m)", "luck bar", "deflated", "forward"],
            rows=rows,
            caption="Each search's winner against the bar its own trial count set. "
                    "The 2022-2026 forward test is the column that matters: both "
                    "heavily-searched winners decayed hardest, exactly as the "
                    "multiple-testing literature predicts."),
            title="Every registered search"))
    if ga.get("night_winner"):
        s.add(Note(ga["night_winner"], title="What the newest search found",
                   level="info"))
    return s


def _forward_section(entries: list[AlgorithmEntry], ctx: dict | None) -> Section:
    ctx = ctx or {}
    s = Section(
        "After 2022: the forward record",
        blurb="Everything above is measured inside the research window the "
              "strategies were built in - none of it is out-of-sample. The reserved "
              "2022-2026 period was spent on 2026-08-28 testing the original "
              "twenty-two candidates as one pre-registered set; candidates added "
              "later were tested afterwards against the same machinery, with their "
              "contamination recorded.")
    if ctx.get("power_note"):
        s.add(Note(ctx["power_note"], level="warn",
                   title="What 54 months can and cannot prove."))
    tested = [e for e in entries if e.forward]
    if tested:
        rows = []
        for e in sorted(tested, key=lambda e: e.forward.get("forward_d_sharpe")
                        if e.forward.get("forward_d_sharpe") is not None
                        else float("-inf"), reverse=True):
            f = e.forward
            v = f.get("verdict", "")
            rows.append([
                Cell(e.name, sort_key=e.name, href=e.href),
                _text(f.get("seal_mode", ""), emphasis="muted"),
                _cell(f.get("research_sharpe"), theme.num),
                _cell(f.get("forward_sharpe"), theme.num),
                _cell(f.get("forward_d_sharpe"), lambda x: theme.num(x, dp=2),
                      emphasis="good" if (f.get("forward_d_sharpe") or 0) > 0
                      else "bad"),
                _cell(f.get("decay_z"), lambda x: theme.num(x, dp=2)),
                _text(v, emphasis={"held": "good", "decayed": "warn",
                                   "failed": "bad"}.get(v, "muted")),
            ])
        s.add(TableBlock(Table(
            columns=["strategy", "seal", "research Sharpe", "forward Sharpe",
                     "fwd ΔSharpe", "decay z", "verdict"],
            rows=rows,
            caption="Realistic costs. 'held' means the forward window failed to "
                    "refute the research-window prediction - random_weight also "
                    "held, so a verdict is not an endorsement. 'decay z' is how "
                    "many standard errors the Sharpe fell."),
            title="Forward outcomes"))
    if ctx.get("selection_note"):
        s.add(Note(ctx["selection_note"], level="warn"))
    if not tested:
        s.add(Note("No forward records found for these candidates yet.",
                   level="info"))
    return s


def _links(index_href: str, doc_links: list[dict]) -> Section:
    cards = [LinkCard("← all reports", index_href,
                      "The scoreboard, every strategy page, and the registry.")]
    cards += [LinkCard(**d) for d in doc_links]
    s = Section("Go deeper")
    s.add(LinkGrid(cards))
    return s
