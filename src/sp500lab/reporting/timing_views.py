"""The Calendar Lab page: the overnight decomposition and the calendar rules.

Pure view. The CLI runs (or reads) the timing backtests, the decomposition and the
acceptance identities, and hands everything here as plain data.

The page's one editorial rule: gross and net are never shown without each other. The
overnight anomaly is the textbook case of an effect that is real, large, and mostly
untradable at retail size - a page that showed the gross curves alone would be an
advertisement, and one that showed only the net would bury a genuine structural fact
about how equity returns arrive.
"""

from __future__ import annotations

import pandas as pd

from . import series as S
from . import theme
from .specs import (Download, LineChart, LinkCard, LinkGrid, Note, Report, Section,
                    Stat, StatRow, TableBlock)
from .tables import Cell, Table, _cell, _text


def timing_report(*, accept: dict, rules: list[dict],
                  gross_curves: dict[str, pd.Series],
                  net_curves: dict[str, pd.Series],
                  members: pd.DataFrame, member_summary: dict,
                  members_csv: str = "", generated_at: str = "",
                  index_href: str = "index.html") -> Report:
    """The whole page.

    accept        timing_accept()'s report dict
    rules         one dict per rule: name, claim, explain, exposure, and a
                  settings dict {cost_model: {cagr, sharpe, maxdd, turnover,
                  cost_drag}}, plus d_sharpe vs buy-and-hold (realistic)
    gross_curves  {label: monthly nav} for the decomposition chart, zero-cost
    net_curves    {label: monthly nav} same strategies under realistic costs
    members       per-ticker decomposition frame from timing.decompose
    members_csv   the full frame as CSV text, embedded as a download
    """
    n_rules = sum(1 for r in rules if r["name"] != "tm_buy_hold")
    report = Report(
        title="The Calendar Lab",
        subtitle="When the market pays: overnight vs intraday, weekends, month "
                 "turns, holidays - each claim costed three ways on SPY",
        generated_at=generated_at,
        meta={"rules": theme.count(n_rules),
              "instrument": "SPY",
              "granularity": "daily legs (close→open, open→close)"},
    )
    report.add(_engine_section(accept))
    report.add(_decomposition_section(accept, gross_curves))
    report.add(_costed_section(rules, net_curves))
    report.add(_rules_section(rules))
    report.add(_members_section(members, member_summary, members_csv))
    report.add(_closing(index_href))
    return report


def _engine_section(accept: dict) -> Section:
    s = Section(
        "The machine, and why to trust it",
        blurb="A session has two tradable legs - close to next open, and open to "
              "close - and every rule here is just a schedule of which legs to hold. "
              "The engine that walks those legs is separate from the monthly one, so "
              "it earns trust the same way: identities that cannot pass by accident.")
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
    ordered = sorted(rules, key=lambda r: (r.get("d_sharpe")
                                           if r.get("d_sharpe") is not None
                                           else float("-inf")), reverse=True)
    for r in ordered:
        real = r["settings"].get("realistic", {})
        opt = r["settings"].get("optimistic", {})
        pes = r["settings"].get("pessimistic", {})
        d = r.get("d_sharpe")
        rows.append([
            Cell(r["name"], sort_key=r["name"], title=r.get("claim", "")[:160]),
            _text(r.get("exposure", "")),
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
        columns=["rule", "time in market", "CAGR (opt)", "CAGR (real)",
                 "CAGR (pess)", "Sharpe", "maxDD", "cost drag", "ΔSharpe"],
        rows=rows,
        caption="ΔSharpe is against buy-and-hold SPY over the same window, realistic "
                "costs. A rule that is long ~19% of the time gets a structural "
                "Sharpe boost from sitting in cash; the ΔSharpe column is the one "
                "that already accounts for the alternative."),
        title="Every rule, all three cost settings"))
    if net_curves:
        aligned = S.align(net_curves)
        s.add(LineChart(
            [S.equity(c, name, kind="benchmark" if "hold" in name else "strategy")
             for name, c in aligned.items()],
            title="Growth of 1.0, realistic costs",
            y_format="multiple", log_y=True, height=340))
    s.add(Note(
        "The overnight strategy is the family's designated cost casualty: ~500 round "
        "trips a year turns an 8%/yr gross anomaly into low single digits realistic "
        "and nothing pessimistic. That gap is not a failure of the test - it IS the "
        "result, and it is why the tradable expression of this fact in the monthly "
        "engine (overnight_momentum) trades twelve times a year instead.",
        level="warn", title="Costs are the finding."))
    return s


def _rules_section(rules: list[dict]) -> Section:
    s = Section(
        "What each rule claims",
        blurb="In each rule's own words - the first paragraphs of its docstring, the "
              "same source the code runs from.")
    for r in rules:
        text = r.get("claim", "")
        for para in r.get("explain", [])[:1]:
            text += "\n\n" + para
        s.add(Note(text, title=r["name"]))
    return s


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


def _closing(index_href: str) -> Section:
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
    s.add(LinkGrid([LinkCard("← all reports", index_href,
                             "The scoreboard, every strategy page, the registry."),
                    LinkCard("The Algorithm Book", "algorithms.html",
                             "Every competitor explained and scored on one page.")]))
    return s
