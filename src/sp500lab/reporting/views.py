"""Composing reports. Reads the registry, returns a `Report` — never markup.

Four reports, one per question:

    index_report       the landing page: every strategy, one click away
    strategy_report    one strategy in full, from a live result - the main one
    feature_report     what every feature is, and whether it reads the future
    comparison_report  which of these is better, and by how much
    run_report         what did this one strategy actually do, from the registry
    trades_report      show me the orders, and let me take them away
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

import numpy as np
import pandas as pd

from ..backtest import registry
from . import series as S
from . import tables as T
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
from .util import gt as _gt
from .util import lt as _lt
from .util import now as _now
from .util import pos as _pos
from .util import slugify

#: Largest trade ledger embedded in a report, in rows AND in bytes of CSV. Whichever
#: binds first wins. base64 inflates a file by a third, and a 90,000-order equal-weight
#: ledger would produce a 20 MB page that no browser opens pleasantly.
#:
#: Measured: the embedded ledger is 91% of a strategy report's weight (3.65 MB of 4.00 MB
#: for `low_vol`); every chart on the page together is 0.31 MB. So this constant, and not
#: the charting, is what decides how large these files are.
MAX_EMBEDDED_TRADES = 25_000
MAX_EMBEDDED_BYTES = 4_000_000

#: Kept short because it doubles as a scatter-point label, where a long
#: name collides with its neighbours. The grey dashed styling already says
#: "benchmark" without spelling it out.
BENCHMARK_LABEL = "SPY"


# --------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------

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

# --------------------------------------------------------------------------
# Trades - the evidence for the curve
# --------------------------------------------------------------------------

def trades_report(result, *, reconciliation: dict | None = None) -> Report:
    """Everything an outside reader needs to check a strategy, in one file.

    The download block is the point of this report. Every other view in this module
    argues that a number is trustworthy; this one hands over the orders and invites the
    reader to disagree. So the CSV is embedded in the page rather than linked beside it -
    a report emailed on its own must still carry its evidence.

    The reconciliation table sits above the trade sample deliberately. "Here are the
    trades" and "here is the proof these are the trades that produced that curve" are
    different claims, and the second one is the one worth reading first.
    """
    from ..backtest.trades import reconcile

    trades = getattr(result, "trades", None)
    report = Report(
        title=f"Trades — {result.strategy}",
        subtitle=(f"{len(trades):,} orders, "
                  f"{result.config.get('start', '')} to {result.config.get('end', '')}, "
                  f"{result.config.get('cost_model', '')} costs"
                  if trades is not None and len(trades) else "no orders"),
        generated_at=_now(),
        meta={"strategy": result.strategy,
              "run": str(result.config.get("run_id", "")),
              "capital": theme.money(result.config.get("initial_capital", 0)),
              "commit": str(result.config.get("git_commit", ""))[:10]})

    if trades is None or not len(trades):
        report.add(Section("Trades", blurb="").add(Note(
            "This run recorded no orders. Either the strategy never traded, or it was "
            "run with record_trades=False.", level="warn")))
        return report

    audit = reconciliation if reconciliation is not None else reconcile(trades, result)
    p = result.performance

    report.add(Section("Summary", blurb=(
        "What the strategy did, and what it cost to do it."
    )).add(StatRow([
        Stat("CAGR", theme.pct(p.cagr), "after costs"),
        Stat("Sharpe", theme.num(p.sharpe)),
        Stat("max drawdown", theme.pct(p.max_drawdown), emphasis="bad"),
        Stat("orders", theme.count(len(trades))),
        Stat("traded", theme.money(float(trades["notional"].sum()))),
        Stat("cost", theme.money(float(trades["cost"].sum())),
             f"{result.costs.as_dict()['bps_of_traded']:.1f} bp of notional"),
    ])).add(TableBlock(T.trade_years(trades), title="By year")))

    report.add(Section("Take the orders", blurb=(
        "The file below is the whole ledger, one row per order. `date` is the execution "
        "session, `price` is the AS-TRADED open — the price a broker printed that "
        "morning, not an adjusted number — so every row can be checked against an "
        "independent quote source. `adj_price` and `adj_shares` are the "
        "total-return-adjusted figures the accounting actually used; the two views "
        "differ by the dividend and split chain and reconcile exactly."
    )).add(_trades_download(result, trades)).add(
        TableBlock(T.trade_reconciliation(audit), title="Does it add up?")).add(
        _audit_note(audit)))

    report.add(Section("What it traded", blurb=(
        "Concentration of activity. A strategy whose costs pile into a handful of names "
        "is making a bet on those names' liquidity as much as on their returns."
    )).add(TableBlock(T.trade_leaders(trades), title="Most traded"))
      .add(TableBlock(T.trade_sample(trades), title="Recent orders")))

    return report


def _trades_download(result, trades, full_csv_href: str | None = None) -> Download:
    """The whole ledger if it fits, a recent slice if it does not - and say which."""
    slug = slugify(result.strategy)
    csv = trades.to_csv(index=False)
    if len(trades) <= MAX_EMBEDDED_TRADES and len(csv) <= MAX_EMBEDDED_BYTES:
        return Download(
            filename=f"{slug}-trades.csv", content=csv,
            label="Download every buy and sell",
            note="Plain CSV, embedded in this page. Opens in any spreadsheet; no part "
                 "of it needs this repository to read.")

    keep = trades.tail(MAX_EMBEDDED_TRADES)
    sample = keep.to_csv(index=False)
    while len(sample) > MAX_EMBEDDED_BYTES and len(keep) > 500:
        keep = keep.tail(len(keep) // 2)
        sample = keep.to_csv(index=False)
    where = (f"The complete file is beside this page at {full_csv_href}."
             if full_csv_href else
             f"Write the whole file with `sp500lab backtest trades {result.strategy}`.")
    return Download(
        filename=f"{slug}-trades-recent.csv", content=sample,
        label="Download recent trades",
        note=(f"This ledger has {len(trades):,} orders — too many to embed without "
              f"making the page unopenable. The most recent {len(keep):,} are here. "
              + where))


def _audit_note(audit: dict) -> Note:
    if audit.get("ok"):
        return Note("The orders replay the cash account exactly, and every dollar of "
                    "cost in the headline is attributed to the order that incurred it. "
                    "The trade list and the equity curve are the same run.",
                    level="info", title="Reconciled.")
    return Note("The orders do NOT add up to the equity curve. One of the two is wrong "
                "and neither should be quoted until it is resolved.",
                level="danger", title="Reconciliation failed.")


# --------------------------------------------------------------------------
# 5. One strategy, in full - built from a live result rather than a registry row
# --------------------------------------------------------------------------

def strategy_report(result, *, results_by_cost=None, benchmark=None, claim: str = "",
                    feature_coverage=None, deflation: dict | None = None,
                    index_href: str = "index.html",
                    full_csv_href: str | None = None) -> Report:
    """Everything about one strategy, on one page, for a reader who will not open the code.

    `run_report` renders a row from the registry: whatever was logged, plus a stored
    month-end curve. This renders a live `BacktestResult`, which carries the daily curve,
    the trade ledger, the weights and the diagnostics - so it can show the drawdown at its
    true depth, the orders that produced it, and the portfolio it would hold today.

    The order of the sections is the argument the page is making, and it is deliberate:
    what this claims, then what happened, then what it cost, then the orders, then every
    reason to distrust the numbers above. A report that puts coverage on page four has
    already misled its reader.
    """
    from ..backtest.trades import reconcile

    cfg = result.config
    detail = cfg.get("strategy_detail", {})
    equity = result.equity.dropna()
    monthly = _to_monthly(equity)
    bench_curve = result.benchmark.dropna() if result.benchmark is not None else None

    from ..backtest.registry import git_state

    version = cfg.get("feature_version")
    report = Report(
        title=result.strategy,
        subtitle=_headline_claim(claim) or f"{cfg.get('start')} to {cfg.get('end')}",
        generated_at=_now(),
        meta={"window": f"{cfg.get('start')} → {cfg.get('end')}",
              "costs": str(cfg.get("cost_model", "")),
              "capital": theme.money(cfg.get("initial_capital", 0)),
              "rebalances": theme.count(cfg.get("n_rebalances")),
              "feature set": f"v{version}" if version else "none",
              "commit": git_state()[0][:10]})

    report.add(_claim_section(result, claim, detail, feature_coverage, index_href))
    report.add(_headline_section(result, benchmark))
    report.add(_performance_section(result, equity, monthly, bench_curve))
    report.add(_stability_section(monthly))
    report.add(_costs_section(result, results_by_cost))
    report.add(_orders_section(result, reconcile, full_csv_href))
    report.add(_result_honesty_section(result, deflation))
    return report


def _headline_claim(claim: str) -> str:
    """A subtitle from the claim: the first sentence, or two if the first is a fragment.

    "Evolved, not written." is a true first sentence and a useless subtitle, which is
    what happens when a rule about punctuation stands in for a rule about meaning.
    """
    if not claim:
        return ""
    parts = [p.strip() for p in claim.split(".") if p.strip()]
    if not parts:
        return ""
    out = parts[0]
    if len(out) < 45 and len(parts) > 1:
        out += ". " + parts[1]
    return out[:200].rstrip() + "."


def _claim_section(result, claim, detail, feature_coverage, index_href) -> Section:
    s = Section("What this claims", blurb=(
        "A strategy is a sentence somebody could argue with. This is the sentence, taken "
        "from the strategy's own source, so the report and the code cannot drift apart."))
    if claim:
        s.add(Note(claim, level="info"))
    construction = detail.get("construction") or {}
    s.add(StatRow([
        Stat("holds", theme.count(construction.get("top_k") or "everything"),
             "names, by score"),
        Stat("weighting", str(construction.get("weighting", "—"))),
        Stat("per-name cap", theme.pct(construction.get("max_weight"))
             if construction.get("max_weight") else "—"),
        Stat("warmup", f"{detail.get('warmup', 0)} sessions"),
        Stat("first tradable", str(detail.get("min_date") or result.config.get("start"))),
    ], title="How it builds the portfolio"))
    features = tuple(detail.get("features") or ())
    s.add(TableBlock(T.strategy_features(features, feature_coverage),
                     title="What it reads"))
    s.add(LinkGrid([LinkCard("← all strategies", index_href,
                             "The scoreboard, the other strategies, and the feature "
                             "layer they share.")]))
    return s


def _headline_section(result, benchmark) -> Section:
    p = result.performance
    s = Section("Headline")
    s.add(StatRow([
        Stat("CAGR", theme.pct(p.cagr), "after costs",
             emphasis="good" if _pos(p.cagr) else "bad"),
        Stat("Sharpe", theme.num(p.sharpe)),
        Stat("max drawdown", theme.pct(p.max_drawdown), "daily, peak to trough",
             emphasis="bad"),
        Stat("turnover", theme.pct(p.ann_turnover), "one-way, per year",
             emphasis="warn" if _gt(p.ann_turnover, 4.0) else ""),
        Stat("names", theme.num(p.avg_positions), "average held"),
        Stat("cost drag", theme.pct(p.cost_drag), "gross minus net"),
    ]))
    if benchmark is not None:
        s.add(TableBlock(T.versus_benchmark(result, benchmark),
                         title="Against the index, over exactly these dates"))
        s.add(Note(
            "Strategies in this project do NOT all cover the same window — anything "
            "built on SEC fundamentals starts in 2010 because XBRL does. SPY returned "
            "10.42%/yr from 2007-04 and 15.66%/yr from 2010-07, so a raw CAGR ranks "
            "windows rather than strategies. The `difference` column above is the "
            "comparison worth reading.",
            level="warn", title="Read the difference, not the level."))
    return s


def _performance_section(result, equity, monthly, bench_curve) -> Section:
    s = Section("What happened", blurb=(
        "Growth of a dollar, and the two ways it is usually flattered: a log axis that "
        "hides a drawdown, and an average that hides which years earned it."))
    lines = [S.equity(equity, result.strategy)]
    if result.gross_equity is not None:
        lines.append(S.equity(result.gross_equity.dropna(), "gross of costs",
                              kind="gross"))
    if bench_curve is not None:
        lines.append(S.equity(bench_curve, BENCHMARK_LABEL, kind="benchmark"))
    s.add(LineChart(lines, title="Growth of 1.0", y_format="multiple", log_y=True,
                    height=340,
                    caption="Log axis: equal vertical distances are equal percentage "
                            "moves. The dotted line is the same strategy with costs "
                            "switched off."))
    if bench_curve is not None:
        rel = S.relative(equity, bench_curve, f"vs {BENCHMARK_LABEL}")
        if len(rel):
            s.add(LineChart([rel], title=f"Cumulative ratio to {BENCHMARK_LABEL}",
                            y_format="multiple", height=230, legend=False,
                            caption="Rising means outperforming. Two near-identical "
                                    "equity curves are visually indistinguishable over "
                                    "fifteen years; their ratio is not."))
    s.add(AreaChart(S.drawdown(equity), title="Drawdown", height=190,
                    caption="Computed on the DAILY curve, so this is the real depth "
                            "rather than the shallower month-end shape."))
    years, values = S.annual_returns(equity)
    if years:
        s.add(BarChart(years, values, title="Calendar year returns", y_format="pct",
                       height=240))
    try:
        s.add(TableBlock(T.annual_returns(result.annual_table()),
                         title="Year by year against the index"))
    except Exception:                                             # noqa: BLE001
        pass
    return s


def _stability_section(monthly) -> Section:
    s = Section("Was it steady?", blurb=(
        "A single Sharpe over fifteen years hides whether it was earned evenly or in one "
        "stretch. Every window below is trailing, so each point was knowable on its own "
        "date."))
    if monthly is None or len(monthly) < 40:
        s.add(Note("Too few months to compute trailing statistics.", "warn"))
        return s
    rs = S.rolling_sharpe(monthly, window=36)
    if len(rs):
        s.add(LineChart([rs], title="Rolling 3-year Sharpe", y_format="num",
                        zero_line=True, height=240, legend=False,
                        caption="Below zero means three years of losing money. Most "
                                "strategies here spend time there."))
    rv = S.rolling_vol(monthly, window=12)
    if len(rv):
        s.add(LineChart([rv], title="Rolling 1-year volatility", y_format="pct",
                        height=210, legend=False))
    rows, cols, grid = S.monthly_grid(monthly)
    if rows:
        s.add(Heatmap(rows, cols, grid, title="Monthly returns",
                      caption="Across a row is how a year was earned; down a column is "
                              "the only honest way to look for seasonality."))
    return s


def _costs_section(result, results_by_cost) -> Section:
    s = Section("What it cost", blurb=(
        "Turnover is where backtests lie, so this sits next to the return rather than "
        "beneath it. All three cost settings, always — a strategy that only works under "
        "the optimistic one is a bet that the spread estimator is wrong in your favour."))
    if results_by_cost:
        s.add(TableBlock(T.cost_sensitivity(results_by_cost),
                         title="Under all three cost models"))
        cagrs = [r.performance.cagr for r in results_by_cost]
        if cagrs and cagrs[0] > 0 >= cagrs[-1]:
            s.add(Note("This strategy is profitable only under optimistic costs. That is "
                       "a bet on the half-spread estimator, not a strategy.",
                       level="danger", title="It does not survive being charged."))
    c = result.costs.as_dict()
    s.add(StatRow([
        Stat("total charged", theme.money(c["total"]),
             f"{c['bps_of_traded']:.1f} bp of notional"),
        Stat("commission", theme.money(c["commission"]),
             f"{c['n_min_commission']:,} at the $1 minimum"),
        Stat("spread", theme.money(c["spread"])),
        Stat("orders", theme.count(c["n_orders"])),
        Stat("traded", theme.money(c["traded_notional"])),
    ]))
    if c.get("n_spread_fallback"):
        s.add(Note(f"{c['n_spread_fallback']:,} orders had no spread estimate and used "
                   "the fallback. Their cost is an assumption, not a measurement.",
                   level="warn"))
    return s


def _orders_section(result, reconcile, full_csv_href=None) -> Section:
    trades = getattr(result, "trades", None)
    s = Section("The orders", blurb=(
        "The evidence for everything above. `price` is the AS-TRADED open — the price a "
        "broker printed that morning — so any row can be checked against an independent "
        "quote source."))
    if trades is None or not len(trades):
        s.add(Note("This run recorded no orders.", "warn"))
        return s
    audit = reconcile(trades, result)
    s.add(_trades_download(result, trades, full_csv_href))
    s.add(TableBlock(T.trade_reconciliation(audit), title="Does it add up?"))
    s.add(_audit_note(audit))
    s.add(TableBlock(T.trade_years(trades), title="Trading by year"))
    s.add(TableBlock(T.trade_leaders(trades, 15), title="Most traded"))
    s.add(TableBlock(T.holdings_snapshot(result), title="What it would hold today"))
    return s


def _result_honesty_section(result, deflation) -> Section:
    s = Section("What would make me distrust this", blurb=(
        "Every backtest in this project reports the ways it could be lying. These are "
        "measurements, not disclaimers."))
    diag = result.diagnostics
    rows = [[T._text(k.replace("!! ", "")), T._text(str(v),
             "bad" if k.startswith("!!") else "")]
            for k, v in diag.items()]
    s.add(TableBlock(T.Table(["diagnostic", "value"], rows,
                             aligns=["left", "left"], sortable=False),
                     title="Reported by the engine on every run"))
    coverage = str(diag.get("price_coverage", ""))
    if coverage:
        s.add(Note(
            f"Price coverage: {coverage}. The index members with no price history are "
            "disproportionately the ones that were delisted, so an early-year result is "
            "computed over a survivor subset sitting underneath the point-in-time "
            "universe this project exists to build (ADR-023).",
            level="warn", title="Coverage is a result, not a footnote."))
    if deflation and deflation.get("deflated_sharpe") is not None:
        dsr = deflation["deflated_sharpe"]
        s.add(TableBlock(T.deflation_panel(deflation),
                         title="Does it survive the search that produced it?"))
        s.add(Note(
            f"{deflation.get('n_trials')} distinct configurations were evaluated in this "
            f"study. The luckiest of that many worthless strategies would have posted an "
            f"annualised Sharpe of about "
            f"{deflation.get('expected_max_sharpe_annualised')}.",
            level="info" if (isinstance(dsr, float) and dsr >= 0.95) else "danger"))
    if result.config.get("touched_holdout"):
        s.add(Note("This run saw HOLDOUT data (2022 onward) and is permanently recorded "
                   "in the holdout ledger. Each look degrades the only out-of-sample "
                   "evidence this project has.", level="danger", title="Holdout touched."))
    return s


def _to_monthly(curve):
    s = curve.dropna().astype(float)
    if len(s) < 2:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(s.index))))
    m = pd.Series(s.to_numpy(), index=idx).resample("ME").last().dropna()
    m.index = m.index.strftime("%Y-%m-%d")
    return m


# --------------------------------------------------------------------------
# 6. The feature layer
# --------------------------------------------------------------------------

def feature_report(fp, *, leakage: dict | None = None, usage=None,
                   index_href: str = "index.html") -> Report:
    """What every feature is, where it comes from, and whether it can be trusted.

    The leakage check goes SECOND, immediately after the overview and before a single
    feature is listed. A catalogue of 75 columns is a directory listing until somebody
    has established that none of them read the future; putting the check at the bottom
    would be organising the page by how comfortable it is to read.
    """
    from ..features.catalog import FAMILY_NOTES, by_family

    coverage = fp.coverage()
    report = Report(
        title="The feature layer",
        subtitle=f"{fp.n_features} point-in-time features, "
                 f"{fp.meta.get('start')} to {fp.meta.get('end')}",
        generated_at=_now(),
        meta={"features": theme.count(fp.n_features),
              "rebalance dates": theme.count(len(fp.rows)),
              "securities": theme.count(len(fp.security_ids)),
              "version": str(fp.meta.get("feature_version"))})

    overview = Section("What this is", blurb=(
        "Every strategy in this project reads the same numbers. If each computed its own "
        "momentum and its own valuation ratio, the scoreboard would partly rank who "
        "wrote better feature code rather than who has better signal — and a genetic "
        "algorithm evaluating a thousand individuals cannot afford to recompute a "
        "rolling regression inside every one of them."))
    overview.add(StatRow([
        Stat("features", theme.count(fp.n_features)),
        Stat("families", theme.count(len(by_family()))),
        Stat("rebalance dates", theme.count(len(fp.rows))),
        Stat("securities", theme.count(len(fp.security_ids))),
        Stat("first date", str(fp.meta.get("start"))),
        Stat("version", str(fp.meta.get("feature_version")), "bump changes results"),
    ]))
    overview.add(TableBlock(T.feature_families(fp.names, coverage), title="By family"))
    overview.add(LinkGrid([LinkCard("← all reports", index_href,
                                    "The strategies that read these, and how each one "
                                    "scored.")]))
    report.add(overview)

    check = Section("Can any of it be trusted?", blurb=(
        "Features fail in two ways, and this catches both at once. A price feature can "
        "have a forward-looking window; a fundamental can be joined on the date a number "
        "became TRUE rather than the date it became KNOWN. So the whole matrix is rebuilt "
        "from a price panel that physically ends at a past date, with every filing "
        "published after it deleted, and the earlier rows must come out bit-identical."))
    check.add(TableBlock(T.leakage_summary(leakage or {})))
    if leakage and leakage.get("ok"):
        check.add(Note(
            "Deleting the future changed nothing about the past, for every feature. "
            "That is not proof the features are useful — only that they are honest.",
            level="info", title="Passed."))
    elif leakage:
        check.add(Note("At least one feature changed when later data was removed. It was "
                       "reading data it could not have had, and every backtest that used "
                       "it is void.", level="danger", title="FAILED."))
    report.add(check)

    cov = Section("How much of it actually exists", blurb=(
        "A feature that is 40% populated in 2010 and 95% populated in 2020 will make any "
        "strategy using it look like it improved over time when only the data did. "
        "Fundamentals begin in 2010 because the SEC's XBRL mandate does, and they cover "
        "649 of the 973 companies that have ever been in the index — which correlates "
        "with survival."))
    lines = _family_coverage_lines(fp)
    if lines:
        cov.add(LineChart(lines, title="Share of securities with a value, by family",
                          y_format="pct", height=300,
                          caption="Around 80% is the ceiling: the denominator counts "
                                  "every security in the panel, including ones that were "
                                  "not in the index on that date."))
    report.add(cov)

    for family, names in by_family().items():
        present = [n for n in names if fp.has(n)]
        if not present:
            continue
        sec = Section(family, blurb=FAMILY_NOTES.get(family, ""))
        sec.add(TableBlock(T.feature_catalog(present, coverage, family=family)))
        report.add(sec)

    if usage is not None and len(usage):
        who = Section("Who reads what", blurb=(
            "A strategy is only as available as its scarcest input."))
        who.add(TableBlock(usage))
        report.add(who)

    report.add(Section("Adding one", blurb=(
        "Four steps, in `docs/FEATURES.md`: write it in the family module with a trailing "
        "window, add a line to `features/catalog.py` so it appears here, bump "
        "`FEATURE_VERSION`, and run the leakage check. The test suite fails if a feature "
        "exists without a catalogue entry, which is the only way a page like this stays "
        "true.")).add(Note(
            "python -m sp500lab features build --rebuild && "
            "python -m sp500lab features check", level="info")))
    return report


def _family_coverage_lines(fp) -> list:
    """One line per family: the share of securities with a value, over time."""
    from ..features.catalog import describe

    groups: dict[str, list[int]] = {}
    for i, name in enumerate(fp.names):
        groups.setdefault(describe(name).family, []).append(i)

    finite = np.isfinite(fp.values)
    dates = [str(d) for d in fp.dates]
    out = []
    for family, cols in groups.items():
        if family in ("Macro", "Market state"):
            continue           # broadcast across every security; the line is a constant
        share = finite[:, :, cols].mean(axis=(1, 2))
        out.append(S.LineSeries(family, dates, [float(v) for v in share]))
    return out


# --------------------------------------------------------------------------
# 7. The index: one page that links to everything else
# --------------------------------------------------------------------------

def index_report(specs: list, *, curves=None, studies=None, extra_cards=None,
                 title: str = "sp500lab", subtitle: str = "") -> Report:
    """The landing page. Every other report is one click from here.

    Reports in this project are separate self-contained files, because that is what makes
    any one of them survive being sent on its own. The cost is that there is no
    navigation, and this is the fix — relative links between files in one folder, no
    server and no build step.
    """
    report = Report(title=title, subtitle=subtitle or
                    "Every strategy, what it claims, and what it actually did.",
                    generated_at=_now(),
                    meta={"strategies": theme.count(len(specs))})

    beat = [s for s in specs if _gt(s.get("d_sharpe"), 0)]
    searched = [s for s in specs if s.get("evolved")]
    start = Section("Start here", blurb=(
        "Each card opens a full report: the claim, the equity curve, every year, what it "
        "cost, and its orders as a downloadable CSV embedded in the page."))
    if extra_cards:
        start.add(LinkGrid([LinkCard(**c) for c in extra_cards]))
    start.add(Note(
        f"{len(beat)} of {len(specs)} strategies beat the index on risk-adjusted return "
        "over their own window"
        + (": " + ", ".join(s["name"] for s in beat) + "." if beat else ".")
        + " That is the correct null result and it is the bar. Anything that beats it by "
          "a wide margin on a first run is much more likely to be a bug than an edge.",
        level="info", title="The honest summary."))
    report.add(start)

    if searched:
        report.section("Start here").add(Note(
            ", ".join(s["name"] for s in searched)
            + " came out of a genetic algorithm. A searched strategy's Sharpe is the "
              "MAXIMUM over every configuration the search evaluated, and the maximum of "
              "N draws is high whether or not there is any signal in the data. The "
              "`deflated` column corrects for that and is the number to read first. "
              "Neither has been tested on the reserved 2022 holdout, which is the only "
              "out-of-sample evidence this project has and is worth exactly one look.",
            level="warn",
            title=f"{len(searched)} of these were not written by a person."))

    report.add(Section("The scoreboard", blurb=(
        "Every algorithm's headline statistics, sorted by Sharpe against the index over "
        "each strategy's OWN dates. Windows differ by design — anything using SEC "
        "fundamentals starts in 2010 — and SPY returned 10.42%/yr from 2007-04 against "
        "15.66%/yr from 2010-07, so a raw CAGR column ranks windows rather than "
        "strategies. Click a name for its page."
    )).add(TableBlock(T.strategy_roster(specs))))

    report.add(Section("All of them", blurb="One card per strategy."
                       ).add(LinkGrid([
        LinkCard(title=s["name"], href=s.get("href", ""), blurb=s.get("claim", ""),
                 stats=[("CAGR", theme.pct(s.get("cagr"))),
                        ("Sharpe", theme.num(s.get("sharpe"))),
                        ("vs index", f"{s['d_sharpe']:+.2f}"
                         if s.get("d_sharpe") is not None else "—")],
                 emphasis="good" if _gt(s.get("d_sharpe"), 0) else "")
        for s in specs])))

    if curves:
        report.add(Section("Every curve at once", blurb=(
            "Rebased to 1.0 at each strategy's own start, which is why the lines do not "
            "all begin together. Useful for shape, misleading for ranking — use the "
            "scoreboard for that."
        )).add(LineChart([S.equity(c, n) for n, c in curves.items() if len(c) > 2],
                         title="Growth of 1.0", y_format="multiple", log_y=True,
                         height=380)))

    if studies is not None and len(studies):
        report.add(Section("Searches", blurb=(
            "Every genetic-algorithm run is a study, and every individual it evaluated is "
            "logged as a trial. `trials` is the input to the deflated Sharpe: without it, "
            "the winner's Sharpe is not conservative or optimistic, it is meaningless."
        )).add(TableBlock(T.studies(studies))))

    return report
