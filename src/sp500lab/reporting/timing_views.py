"""The calendar set: the overnight decomposition, and one page per calendar rule.

Pure view. The CLI runs (or reads) the timing backtests, the decomposition, the
acceptance identities and the stored forward records, and hands everything here as plain
data.

Two reports, one per question
------------------------------
``timing_report``       the index: the leg engine and why to trust it, where SPY's return
                        actually happens, every rule costed three ways, and the
                        per-ticker decomposition
``timing_rule_report``  one rule in full: what it claims, the schedule it trades, the
                        research window under three cost settings, and - when it was
                        carried into the holdout - the forward test on the SAME page

**Both windows on one page, deliberately** (ADR-047). The monthly roster is thirty
algorithms and splits by window into `backtest/` and `forward/`; this family is nine
rules, and each one's research-to-forward arc is a single story. `tm_weekend` went from
a 0.04 Sharpe in research to -0.20 forward and was refuted: splitting that across two
folders would file the claim and its refutation in different rooms.

The forward half is not written here. `forward_views.outcome_sections()` builds it from
the stored record, so the calendar set and the forward set cannot drift into disagreeing
about what a paired comparison says.

The editorial rule, unchanged from the old single page: **gross and net are never shown
without each other.** The overnight anomaly is the textbook case of an effect that is
real, large, and mostly untradable at retail size - a page showing the gross curves
alone would be an advertisement, and one showing only the net would bury a genuine
structural fact about how equity returns arrive.

The second editorial rule is newer and is why `episodes` is on every page: **a calendar
rule's sample is its round trips, not its sessions.** `tm_sell_in_may` is invested across
1,806 sessions and enters sixteen times. Sixteen is the sample.
"""

from __future__ import annotations

import pandas as pd

from . import series as S
from . import theme
from .specs import (
    Download,
    LineChart,
    LinkCard,
    LinkGrid,
    Note,
    Report,
    Section,
    Stat,
    StatRow,
    TableBlock,
)
from .tables import Cell, Table, _cell, _text

#: Where a rule page links, relative to `reports/timing/`. One dict so the index and the
#: rule pages cannot disagree about the way out of the folder.
DEFAULT_HREFS = {
    "index": "index.html",
    "backtest": "../backtest/index.html",
    "forward": "../forward/index.html",
    "genetic": "../genetic_algorithm/methodology.html",
}

#: Deliberately NOT in the defaults above. The Algorithm Book is an on-demand page in
#: `reports/extra/` that may simply not have been built, and a set's index linking to a
#: file that is usually absent is the failure this set was created to fix. The CLI passes
#: `algorithms` when the page is on disk; the view stays pure and renders what it is given.
OPTIONAL_HREFS = ("algorithms",)

#: Verdict wording shared with the forward set, so a card reads the same in both places.
VERDICT_EMPHASIS = {"held": "good", "decayed": "warn", "failed": "bad",
                    "inconclusive": ""}


def _hrefs(hrefs: dict[str, str] | None) -> dict[str, str]:
    return {**DEFAULT_HREFS, **(hrefs or {})}


# ==========================================================================
# 1. The index
# ==========================================================================

def timing_report(*, accept: dict, rules: list[dict],
                  gross_curves: dict[str, pd.Series],
                  net_curves: dict[str, pd.Series],
                  members: pd.DataFrame, member_summary: dict,
                  members_csv: str = "", generated_at: str = "",
                  extra_cards: list[dict] | None = None,
                  hrefs: dict[str, str] | None = None,
                  index_href: str | None = None) -> Report:
    """The set's index: the machine, the decomposition, the scoreboard, the members.

    accept        timing_accept()'s report dict
    rules         one dict per rule: name, href, claim, explain, exposure, schedule, and
                  a settings dict {cost_model: {cagr, sharpe, maxdd, turnover,
                  cost_drag}}, plus d_sharpe vs buy-and-hold (realistic) and the stored
                  forward summary where there is one
    gross_curves  {label: monthly nav} for the decomposition chart, zero-cost
    net_curves    {label: monthly nav} same strategies under realistic costs
    members       per-ticker decomposition frame from timing.decompose
    members_csv   the full frame as CSV text, embedded as a download

    `index_href` is accepted for the page's old life in `reports/extra/`, where "back"
    meant the backtest scoreboard rather than a sibling in the same folder.
    """
    links = _hrefs(hrefs)
    if index_href:
        links["backtest"] = index_href
    n_rules = sum(1 for r in rules if not r.get("is_benchmark"))
    report = Report(
        title="The Calendar Lab",
        subtitle="When the market pays: overnight vs intraday, weekends, month "
                 "turns, holidays - each claim costed three ways on SPY",
        generated_at=generated_at,
        meta={"rules": theme.count(n_rules),
              "instrument": "SPY",
              "granularity": "daily legs (close→open, open→close)"},
    )
    report.add(_engine_section(accept, extra_cards))
    report.add(_decomposition_section(accept, gross_curves))
    report.add(_costed_section(rules, net_curves))
    report.add(_rules_section(rules))
    report.add(_members_section(members, member_summary, members_csv))
    report.add(_closing(links))
    return report


def _engine_section(accept: dict, extra_cards: list[dict] | None = None) -> Section:
    s = Section(
        "The machine, and why to trust it",
        blurb="A session has two tradable legs - close to next open, and open to "
              "close - and every rule here is just a schedule of which legs to hold. "
              "The engine that walks those legs is separate from the monthly one, so "
              "it earns trust the same way: identities that cannot pass by accident.")
    if extra_cards:
        s.add(LinkGrid([LinkCard(**c) for c in extra_cards]))
    s.add(StatRow([
        Stat("Calibration", f"{accept['calibration_bp_per_year']:.2f} bp/yr",
             "buy-and-hold through the leg engine vs the adjusted SPY series",
             "good" if accept["calibration_bp_per_year"] < 1 else "bad"),
        Stat("Decomposition", f"{accept['decomposition_max_rel_err']:.1e}",
             "max error of overnight × intraday = buy-and-hold, any session",
             "good" if accept["decomposition_max_rel_err"] < 1e-9 else "bad"),
        Stat("Execution", "at the scheduled price",
             "calendar rules carry no signal, so a close fill is not lookahead"),
        Stat("Costs", "shared cost model",
             "tick-floor half-spread + IBKR-shaped commission, 3 settings"),
    ]))
    s.add(Note(
        "The two legs multiply back to the close-to-close return by construction, so "
        "the overnight and intraday strategies PARTITION buy-and-hold: their product "
        "must equal it at every session, gross of costs. The engine asserts this to "
        "float precision on the real data, which is what makes 'the overnight share "
        "of SPY's return' a measurement rather than an estimate.",
        title="The identity doing the work."))
    return s


def _decomposition_section(accept: dict, gross_curves: dict[str, pd.Series]) -> Section:
    s = Section(
        "Where SPY's return actually happens",
        blurb="The headline fact this lab exists to show: split every close-to-close "
              "return at the opening bell and the two halves are not remotely alike.")
    s.add(StatRow([
        Stat("Overnight only", theme.pct(accept["overnight_cagr"]),
             "close → next open, every session, gross", "good"),
        Stat("Intraday only", theme.pct(accept["intraday_cagr"]),
             "open → close, every session, gross",
             "bad" if accept["intraday_cagr"] < accept["overnight_cagr"] else ""),
        Stat("Both (buy & hold)", theme.pct(accept["buy_hold_cagr"]),
             "the product of the two, exactly"),
        Stat("Window", accept["window"].replace("..", " → "), "research window only"),
    ]))
    if gross_curves:
        aligned = S.align(gross_curves)
        s.add(LineChart(
            [S.equity(c, name, kind="benchmark" if "hold" in name else "strategy")
             for name, c in aligned.items()],
            title="Growth of 1.0, zero costs - the decomposition itself",
            y_format="multiple", log_y=True, height=340,
            caption="Gross of all costs, which is the only setting in which these "
                    "three curves are a mathematical identity. The costed versions "
                    "are in the next section, and they tell a different story."))
    s.add(Note(
        "Candidate mechanisms - earnings land outside market hours, market-makers "
        "charge for inventory held through the close, retail flow buys at the open - "
        "disagree about whether this should persist. The lab does not adjudicate; it "
        "keeps the measurement standing so the next years of data can.",
        title="Why the market's night shift is paid."))
    return s


def _costed_section(rules: list[dict], net_curves: dict[str, pd.Series]) -> Section:
    s = Section(
        "The rules, costed three ways",
        blurb="Published calendar anomalies are gross of costs. These are not: every "
              "rule pays the tick-floor spread and a $1-minimum commission at every "
              "entry and exit, under the same three settings as every other "
              "competitor in the project.")
    rows = []
    for r in _ranked(rules):
        real = r["settings"].get("realistic", {})
        opt = r["settings"].get("optimistic", {})
        pes = r["settings"].get("pessimistic", {})
        sched = r.get("schedule") or {}
        d = r.get("d_sharpe")
        rows.append([
            Cell(r["name"], sort_key=r["name"], href=r.get("href", ""),
                 title=r.get("claim", "")[:160]),
            _text(r.get("exposure", "")),
            _cell(sched.get("episodes"), theme.count),
            _cell(opt.get("cagr"), theme.pct),
            _cell(real.get("cagr"), theme.pct),
            _cell(pes.get("cagr"), theme.pct),
            _cell(real.get("sharpe"), theme.num),
            _cell(real.get("maxdd"), theme.pct),
            _cell(real.get("cost_drag"), theme.pct),
            _cell(d, lambda v: theme.num(v, dp=2),
                  emphasis="good" if (d or 0) > 0 else ("bad" if d is not None
                                                        else "")),
        ])
    s.add(TableBlock(Table(
        columns=["rule", "time in market", "entries", "CAGR (opt)", "CAGR (real)",
                 "CAGR (pess)", "Sharpe", "maxDD", "cost drag", "ΔSharpe"],
        rows=rows,
        caption="ΔSharpe is against buy-and-hold SPY over the same window, realistic "
                "costs. A rule that is long ~19% of the time gets a structural "
                "Sharpe boost from sitting in cash; the ΔSharpe column is the one "
                "that already accounts for the alternative. `time in market` counts "
                "half-sessions, so an overnight-only rule held every night reads 50%, "
                "not 100%."),
        title="Every rule, all three cost settings"))
    if net_curves:
        aligned = S.align(net_curves)
        s.add(LineChart(
            [S.equity(c, name, kind="benchmark" if "hold" in name else "strategy")
             for name, c in aligned.items()],
            title="Growth of 1.0, realistic costs",
            y_format="multiple", log_y=True, height=340))
    s.add(_sample_note(rules))
    s.add(Note(
        "The overnight strategy is the family's designated cost casualty: ~500 round "
        "trips a year turns an 8%/yr gross anomaly into low single digits realistic "
        "and nothing pessimistic. That gap is not a failure of the test - it IS the "
        "result, and it is why the tradable expression of this fact in the monthly "
        "engine (overnight_momentum) trades twelve times a year instead.",
        level="warn", title="Costs are the finding."))
    return s


def _sample_note(rules: list[dict]) -> Note | None:
    """The `entries` column, said out loud. It is the column that decides the others."""
    counted = [(r["name"], (r.get("schedule") or {}).get("episodes"))
               for r in rules if not r.get("is_benchmark")]
    counted = [(n, e) for n, e in counted if e]
    if not counted:
        return None
    most, fewest = max(counted, key=lambda x: x[1]), min(counted, key=lambda x: x[1])
    years = _window_years(rules)
    return Note(
        "`entries` counts the round trips each rule makes over the research window - "
        "the rising edges of its own leg schedule, which is both what the cost model "
        "charges for and what an independent observation IS for a fixed calendar. It "
        f"runs from {most[1]:,} ({most[0]}) to {fewest[1]:,} ({fewest[0]}): a factor of "
        f"{most[1] / fewest[1]:,.0f}. Every rule in this table is measured over the "
        f"same {years:,.0f} years, so the windows are identical while the evidence "
        "behind them differs by orders of magnitude, and no shared column of Sharpe "
        "ratios can say so on its own.",
        level="warn", title="The rules do not carry equal evidence.")


def _window_years(rules: list[dict]) -> float:
    """Research-window length in years, off the session count every rule shares."""
    sessions = max(((r.get("schedule") or {}).get("of_sessions") or 0) for r in rules)
    return sessions / 252.0


def _rules_section(rules: list[dict]) -> Section:
    """One card per rule, ranked. The full claim now lives on the rule's own page."""
    s = Section(
        "One page per rule",
        blurb="Each card opens the rule in full: what it claims in its own words, the "
              "schedule it trades, all three cost settings on the research window, and "
              "the forward test where there is one - both windows on one page.")
    cards = []
    for r in _ranked(rules):
        real = r["settings"].get("realistic", {})
        fwd = r.get("forward") or {}
        stats = [("CAGR", theme.pct(real.get("cagr"))),
                 ("Sharpe", theme.num(real.get("sharpe"))),
                 ("vs buy & hold", f"{r['d_sharpe']:+.2f}"
                  if r.get("d_sharpe") is not None else "—")]
        if fwd.get("verdict"):
            stats.append(("forward", str(fwd["verdict"]).upper()))
        cards.append(LinkCard(
            title=r["name"], href=r.get("href", ""),
            blurb=_short_claim(r), stats=stats,
            emphasis=(VERDICT_EMPHASIS.get(str(fwd.get("verdict", "")), "")
                      if fwd else ("good" if (r.get("d_sharpe") or 0) > 0 else ""))))
    return s.add(LinkGrid(cards))


def _short_claim(rule: dict) -> str:
    """A card's line of prose: the rule's first sentence, or its first clause."""
    claim = (rule.get("claim") or "").strip()
    head = claim.split(". ")[0].strip()
    if len(head) < 25 and ". " in claim:
        head = ". ".join(claim.split(". ")[:2]).strip()
    return head if head.endswith(".") else head + "."


def _ranked(rules: list[dict]) -> list[dict]:
    """Benchmark first as the bar, then by ΔSharpe. The bar is never a ranked row."""
    bar = [r for r in rules if r.get("is_benchmark")]
    rest = sorted((r for r in rules if not r.get("is_benchmark")),
                  key=lambda r: (r.get("d_sharpe") if r.get("d_sharpe") is not None
                                 else float("-inf")), reverse=True)
    return bar + rest


def _members_section(members: pd.DataFrame, summary: dict, csv_text: str) -> Section:
    s = Section(
        "Which tickers earn their keep overnight",
        blurb="The same split, per security, over exactly the sessions each was in "
              "the point-in-time index. Gross of costs by design - trading any "
              "single name close-to-open crosses its spread ~500 times a year, so "
              "this table says where the effect LIVES, and the monthly "
              "overnight_momentum strategy is its costed, tradable expression.")
    if summary:
        s.add(StatRow([
            Stat("Names measured", theme.count(summary.get("names")),
                 "≥500 in-index sessions, 2007-2021"),
            Stat("Median overnight", theme.pct(summary.get("median_overnight_ann")),
                 "per year, close → open"),
            Stat("Median intraday", theme.pct(summary.get("median_intraday_ann")),
                 "per year, open → close"),
            Stat("Overnight wins", theme.pct(summary.get("overnight_beats_intraday"),
                                             dp=0),
                 "share of names whose overnight leg beat their intraday leg"),
        ]))
    if len(members):
        s.add(TableBlock(_members_table(members.head(15)),
                         title="Top 15 by overnight return"))
        s.add(TableBlock(_members_table(members.tail(15)),
                         title="Bottom 15 by overnight return"))
    if csv_text:
        s.add(Download(filename="overnight_decomposition.csv", content=csv_text,
                       label="Download the full per-ticker table",
                       note=f"{len(members)} securities, membership-clipped, gross "
                            "of costs."))
    return s


def _members_table(df: pd.DataFrame) -> Table:
    rows = []
    for _, r in df.iterrows():
        share = r["overnight_share"]
        rows.append([
            _text(str(r["ticker"])),
            _cell(int(r["sessions"]), theme.count),
            _cell(r["overnight_ann"], theme.pct,
                  emphasis="good" if r["overnight_ann"] > 0 else "bad"),
            _cell(r["intraday_ann"], theme.pct),
            _cell(r["total_ann"], theme.pct),
            _cell(share, lambda v: theme.num(v, dp=2)),
        ])
    return Table(
        columns=["ticker", "sessions", "overnight /yr", "intraday /yr", "total /yr",
                 "overnight share"],
        rows=rows)


def _closing(links: dict[str, str]) -> Section:
    s = Section("The honest footer")
    s.add(Note(
        "Sample sizes differ by orders of magnitude across this family and no shared "
        "table can say so loudly enough: the overnight split rests on ~3,700 daily "
        "observations, turn-of-month on ~180 monthly boundaries, pre-holiday on ~130 "
        "sessions, and sell-in-May on about fifteen winter/summer cycles - no sample "
        "at all. The project's own planning doc (WHAT_TO_BUILD_NEXT.md) warns that "
        "market-timing claims are sample-starved; the seasonal rules here are "
        "included partly to demonstrate that daily machinery cannot rescue a "
        "seasonal hypothesis.", level="warn", title="Sample sizes."))
    s.add(Note(
        "All of these rules were implemented in 2026-08, after the 2022-2026 holdout "
        "was spent on the original roster. Their research-window numbers are honest; "
        "their forward numbers, where present, carry the contamination recorded in "
        "ADR-037: the author knew what kind of market 2022-2026 was.",
        level="warn", title="When these were written."))
    cards = [
        LinkCard("← All algorithms", links["backtest"],
                 "The monthly scoreboard: every hand-written strategy and every "
                 "evolved winner, on the engine this family is separate from."),
        LinkCard("Forward tests", links["forward"],
                 "The 2022-2026 record for the monthly roster. The calendar rules "
                 "were tested through the same harness and are reported here."),
    ]
    if links.get("algorithms"):
        cards.append(LinkCard("The Algorithm Book", links["algorithms"],
                              "Every competitor explained and scored on one page."))
    s.add(LinkGrid(cards))
    return s


# ==========================================================================
# 2. One rule, in full
# ==========================================================================

def timing_rule_report(rule: dict, *, accept: dict,
                       curves: dict | None = None,
                       bench_curves: dict | None = None,
                       forward: dict | None = None,
                       generated_at: str = "",
                       hrefs: dict[str, str] | None = None) -> Report:
    """One calendar rule, both windows.

    rule          the entry `calendar_lab()` built for it
    curves        {"gross": nav, "net": nav} from the stored research curve
    bench_curves  the same for tm_buy_hold, so the rule is never drawn alone
    forward       `calendar_forward()`'s dict - record, comparison, rows - or None
    """
    links = _hrefs(hrefs)
    sched = rule.get("schedule") or {}
    verdict = ((forward or {}).get("record").verdict if forward else "")
    report = Report(
        title=rule["name"],
        subtitle=_subtitle(rule, verdict),
        generated_at=generated_at,
        meta={"schedule": sched.get("legs", "—"),
              "time in market": rule.get("exposure", "—"),
              "entries": theme.count(sched.get("episodes")),
              "window": rule.get("window", "—"),
              **({"forward": verdict} if verdict else {})},
    )
    report.add(_rule_claim(rule, links))
    report.add(_rule_schedule(rule, accept))
    report.add(_rule_research(rule, curves, bench_curves))
    for section in _rule_forward(rule, forward):
        report.add(section)
    report.add(_rule_honesty(rule, forward))
    report.add(_rule_links(rule, links))
    return report


def _subtitle(rule: dict, verdict: str) -> str:
    window = rule.get("window", "")
    if verdict:
        return (f"A calendar rule on SPY. Research {window}, forward-tested on "
                f"2022 onward — {verdict.upper()}.")
    return (f"A calendar rule on SPY. Research {window}; never carried into the "
            "holdout, so there is no out-of-sample evidence on this page.")


def _rule_claim(rule: dict, links: dict[str, str]) -> Section:
    """The rule in its own words - the whole docstring, not a trimmed first line."""
    real = rule["settings"].get("realistic", {})
    sched = rule.get("schedule") or {}
    s = Section("What this rule claims", blurb=rule.get("claim", ""))
    s.add(StatRow([
        Stat("CAGR", theme.pct(real.get("cagr")), "realistic costs",
             "good" if (real.get("cagr") or 0) > 0 else "bad"),
        Stat("Sharpe", theme.num(real.get("sharpe")), "realistic costs"),
        Stat("vs buy & hold", f"{rule['d_sharpe']:+.2f}"
             if rule.get("d_sharpe") is not None else "—",
             "Sharpe, same window",
             "good" if (rule.get("d_sharpe") or 0) > 0 else "bad"),
        Stat("max drawdown", theme.pct(real.get("maxdd")), "realistic costs"),
        Stat("entries", theme.count(sched.get("episodes")),
             "round trips - the sample this rule rests on"),
    ]))
    for para in rule.get("paragraphs", [])[1:]:
        s.add(Note(para))
    if rule.get("is_benchmark"):
        s.add(Note(
            "This is the bar, not a competitor. Both legs on at every session, which "
            "makes it the engine's calibration instrument: gross of costs it must "
            "reproduce the adjusted SPY series the monthly engine is calibrated to "
            "(ADR-018), and every ΔSharpe in this set is measured against it.",
            level="info", title="The benchmark twin."))
    return s


def _rule_schedule(rule: dict, accept: dict) -> Section:
    """What the rule actually does to a book, in sessions and round trips."""
    sched = rule.get("schedule") or {}
    episodes = sched.get("episodes") or 0
    sessions = sched.get("sessions") or 0
    of_sessions = sched.get("of_sessions") or 0
    years = (of_sessions / 252.0) if of_sessions else 0.0
    s = Section("The schedule it trades", blurb=(
        "A calendar rule has no signal. It is two boolean vectors over the sessions - "
        "hold from this close to the next open, hold from this open to this close - "
        "and the engine toggles one bit of state at two checkpoints a session, charging "
        "the cost model on the whole account at every transition."))
    s.add(StatRow([
        Stat("Legs held", sched.get("legs", "—"), "which half of the clock"),
        Stat("Sessions invested", f"{sessions:,} of {of_sessions:,}",
             "any part of the session"),
        Stat("Time in market", rule.get("exposure", "—"),
             "counting half-sessions, so 50% is every night"),
        Stat("Entries", theme.count(episodes),
             f"round trips, about {episodes / years:,.0f}/yr" if years else "round trips"),
    ]))
    s.add(_entries_note(rule, episodes, sessions, years))
    if not rule.get("is_benchmark"):
        s.add(Note(
            "Execution is at the scheduled price, and that is not the lookahead "
            "ADR-017 forbids everywhere else in this project. The monthly engine's "
            "next-open rule exists to stop a signal trading on information from its "
            "own fill; a calendar rule has no signal, and its schedule was knowable "
            "years in advance, so a market-on-close order placed at 15:50 fills at the "
            "price the rule was always going to trade. The one rule that reads data "
            "(tm_vix_overnight) conditions on the PRIOR session's close.",
            title="Why a close fill is honest here."))
    return s


def _entries_note(rule: dict, episodes: int, sessions: int, years: float) -> Note:
    """What `entries` means for THIS rule - the three cases read very differently.

    The benchmark buys once; an overnight rule re-enters every session it holds, so its
    entries and its sessions coincide; a seasonal rule holds long runs, and the gap
    between the two numbers is the whole warning.
    """
    if rule.get("is_benchmark"):
        return Note(
            f"One entry, held for {years:,.0f} years. That is what makes this the bar "
            "rather than a competitor: it pays the cost model once, so every cost drag "
            "in this set is measured against a rule that has almost none.",
            level="info", title="What the sample actually is.")
    common = ("It is counted off the rule's own leg vectors rather than written down "
              "beside them, which is what stops the page and the code disagreeing "
              "about how much evidence there is.")
    if episodes >= sessions:
        return Note(
            f"{episodes:,} entries across {sessions:,} sessions held - this rule "
            "re-enters every time it is invested, so its round trips and its holding "
            "days are the same number, and it pays the spread on every one of them. "
            f"Over {years:,.0f} years that is about {episodes / years:,.0f} round trips "
            "a year, so its cost bill scales with how OFTEN it holds rather than with "
            f"how long - which is what the three cost columns are there to price. "
            f"{common}",
            level="info", title="What the sample actually is.")
    return Note(
        f"{episodes:,} entries is the sample size of this rule, and it is not the "
        f"{sessions:,} sessions it was invested across: the sessions inside one "
        "holding period are one event, not many. The window is fixed at about "
        f"{years:,.0f} years for every rule in this family, so what separates a claim "
        "resting on thousands of independent observations from one resting on a dozen "
        f"is entirely this number. {common}",
        level="warn" if episodes < 100 else "info",
        title="What the sample actually is.")


def _rule_research(rule: dict, curves: dict | None,
                   bench_curves: dict | None) -> Section:
    s = Section("The research window", blurb=(
        f"{rule.get('window', '')}, under all three cost settings. Published calendar "
        "anomalies are quoted gross; the gap between the first column and the last is "
        "what trading this rule at retail size actually takes out."))
    order = ("optimistic", "realistic", "pessimistic")
    rows = []
    for cost in order:
        st = rule["settings"].get(cost)
        if not st:
            continue
        rows.append([
            _text(cost),
            _cell(st.get("cagr"), theme.pct,
                  emphasis="good" if (st.get("cagr") or 0) > 0 else "bad"),
            _cell(st.get("sharpe"), theme.num),
            _cell(st.get("maxdd"), theme.pct),
            _cell(st.get("cost_drag"), theme.pct),
        ])
    s.add(TableBlock(Table(
        columns=["costs", "CAGR", "Sharpe", "maxDD", "cost drag"], rows=rows,
        sortable=False,
        caption="Cost drag is the annualised gap between the gross and net curves. A "
                "rule that only works under `optimistic` is a bet that the half-spread "
                "estimator is wrong in your favour, not a rule.")))

    lines = {}
    if curves:
        if curves.get("gross") is not None:
            lines["before costs"] = curves["gross"]
        if curves.get("net") is not None:
            lines[rule["name"]] = curves["net"]
    if bench_curves and not rule.get("is_benchmark"):
        if bench_curves.get("net") is not None:
            lines["buy & hold"] = bench_curves["net"]
    if lines:
        aligned = S.align(lines)
        s.add(LineChart(
            [S.equity(c, name,
                      kind="benchmark" if name == "buy & hold" else
                           "gross" if name == "before costs" else "strategy")
             for name, c in aligned.items()],
            title="Growth of 1.0, research window", y_format="multiple", log_y=True,
            height=340,
            caption="The distance between “before costs” and the rule is the whole "
                    "cost argument, drawn. Buy-and-hold is on the same axis because a "
                    "rule that sits in cash most of the time can only be judged against "
                    "the alternative of not sitting in cash."))

    drag = (rule["settings"].get("pessimistic") or {}).get("cagr")
    opt = (rule["settings"].get("optimistic") or {}).get("cagr")
    if opt is not None and drag is not None and opt > 0 >= drag:
        s.add(Note(
            "Profitable under optimistic costs and not under pessimistic ones, on the "
            "window it was designed against. That is a bet on the spread estimator.",
            level="danger", title="Only works if trading is nearly free."))
    return s


def _rule_forward(rule: dict, forward: dict | None) -> list[Section]:
    """The out-of-sample half, or one honest section saying there isn't one."""
    if not forward:
        return [Section("Out of sample", blocks=[Note(
            f"{rule['name']} has no forward record: it was never sealed and carried "
            "into the 2022-2026 holdout, so everything above is measured on the window "
            "the rule was chosen against. `sp500lab timing seal " + rule["name"] +
            " --rationale \"...\"` pre-registers it; `timing forward` spends a look and "
            "is recorded forever.", level="warn", title="Never tested out of sample.")])]

    from .forward_views import outcome_sections    # circular only at import time

    record, comparison = forward["record"], forward["comparison"]
    leg_r, leg_f = record.research_leg(), record.forward_leg()
    head = Section("Out of sample: 2022 onward", blurb=(
        "The research window above made a prediction. This is the only place it can be "
        "checked, and the check was worth exactly one look."))
    head.add(StatRow([
        Stat("forward CAGR", theme.pct(leg_f.cagr),
             note=f"research {theme.pct(leg_r.cagr)}",
             emphasis="good" if leg_f.cagr > 0 else "bad"),
        Stat("forward Sharpe", theme.num(leg_f.sharpe),
             note=f"research {theme.num(leg_r.sharpe)}"),
        Stat("vs SPY", f"{leg_f.d_sharpe:+.2f}",
             note="Sharpe, same dates",
             emphasis="good" if leg_f.d_sharpe > 0 else "bad"),
        Stat("verdict", record.verdict.upper(),
             note=f"{record.cost_model} costs, look #{record.look_number}",
             emphasis=VERDICT_EMPHASIS.get(record.verdict, "")),
    ]))
    reason = (record.verdict_reason or "").strip()
    head.add(Note(reason[:1].upper() + reason[1:],
                  level={"failed": "danger", "decayed": "warn"}.get(record.verdict,
                                                                    "info"),
                  title=f"{record.verdict.upper()}."))
    if record.rationale:
        head.add(Note(
            f"“{record.rationale}” — recorded on seal {record.seal_id} "
            f"({record.seal_mode}) before the look.",
            level="info", title="Why this rule was tested."))
    return [head] + outcome_sections(record, comparison, forward["rows"])


def _rule_honesty(rule: dict, forward: dict | None) -> Section:
    sched = rule.get("schedule") or {}
    episodes = sched.get("episodes") or 0
    s = Section("What would make you distrust this", blurb=(
        "Not an appendix. Each of these changes what the numbers above mean."))
    if episodes and episodes < 300:
        s.add(Note(
            f"This rule enters {episodes:,} times over the whole research window. Daily "
            "machinery does not rescue a claim whose events are annual: the effective "
            f"sample is {episodes:,}, not the {sched.get('sessions', 0):,} sessions it "
            "was invested across, and a Sharpe estimated from that many independent "
            "episodes has an error bar wide enough to contain most of the interesting "
            "hypotheses. Read the number as a description of what happened, not as an "
            "estimate of what will.",
            level="warn", title="The sample is small enough to matter."))
    s.add(Note(
        "This family is nine fixed schedules, each at its paper's conventional "
        "definition with zero fitted parameters - so it is nine hypotheses, not nine "
        "hundred parameterisations. It is still a family: the best trial in it deflates "
        "against the whole family's trial count, not against one. A rule read on its "
        "own page, out of the set, will look better than it is.",
        level="warn", title="This rule was not tested alone."))
    s.add(Note(
        "Every rule here was written in 2026-08, after the 2022-2026 holdout had "
        "already been spent on the original roster. The research numbers are clean; "
        "any forward verdict on this page carries the contamination ADR-037 records - "
        "the author knew what kind of market 2022-2026 had been before choosing which "
        "calendar claims to test.",
        level="warn" if forward else "info", title="When this was written."))
    s.add(Note(
        "SPY only, and one instrument's calendar is not the calendar. The engine takes "
        "a symbol (`load_timing_data(ticker)`), so RSP for the equal-weight version of "
        "the overnight effect or IWM for small caps is a small change - the missing "
        "piece is the half-spread convention per instrument, not the machinery.",
        title="One instrument."))
    return s


def _rule_links(rule: dict, links: dict[str, str]) -> Section:
    return Section("The rest of the lab").add(LinkGrid([
        LinkCard("← All calendar rules", links["index"],
                 "The decomposition, every rule costed three ways, and the per-ticker "
                 "table."),
        LinkCard("The monthly scoreboard", links["backtest"],
                 "Thirty algorithms on the other engine, including "
                 "overnight_momentum - this family's one tradable expression."),
        LinkCard("Forward tests", links["forward"],
                 "The 2022-2026 record for the monthly roster, from the same harness "
                 "that produced any verdict above."),
    ]))
