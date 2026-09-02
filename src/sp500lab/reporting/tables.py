"""Tabular preparation: registry rows in, formatted `Table` out. No markup.

Same discipline as `series.py` - these functions return a small dataclass of already
formatted strings plus the raw values, and know nothing about HTML. A test asserts on
cell contents, not on tags.

Why cells carry both a formatted string and a raw value
-------------------------------------------------------
The renderer needs the string to display and the number to sort by. Sorting on the
displayed text would put "9.84%" above "11.10%", which is the kind of defect that is
obvious in a screenshot and invisible in code review. `Cell.sort_key` is always the
underlying float.

Colouring is by *direction*, not by sign
----------------------------------------
A large drawdown and a large return are both big numbers, and only one of them is good
news. `theme.direction()` knows which metrics improve downward, so `best`/`worst`
highlighting stays correct without every caller remembering that turnover and Calmar run
opposite ways.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from . import theme
from .util import gt as _gt
from .util import lt as _lt


@dataclass
class Cell:
    text: str
    sort_key: float | str = 0.0
    emphasis: str = ""        # "" | "good" | "bad" | "warn" | "muted"
    title: str = ""           # hover text, for a number that needs a caveat
    href: str = ""            # relative link to a sibling report, if this cell is one


@dataclass
class Table:
    columns: list[str]
    rows: list[list[Cell]]
    aligns: list[str] = field(default_factory=list)
    caption: str = ""
    sortable: bool = True

    def __post_init__(self) -> None:
        if not self.aligns:
            self.aligns = ["left"] + ["right"] * (len(self.columns) - 1)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def empty(self) -> bool:
        return not self.rows


def _cell(value: Any, fmt: Callable[[Any], str], *, emphasis: str = "",
          title: str = "") -> Cell:
    raw = value
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return Cell("—", float("-inf"), "muted", title)
    try:
        key = float(value)
    except (TypeError, ValueError):
        key = str(value)
        raw = value
    return Cell(fmt(raw), key, emphasis, title)


def _text(value: Any, emphasis: str = "", title: str = "") -> Cell:
    s = "" if value is None else str(value)
    return Cell(s, s.lower(), emphasis, title)


# --------------------------------------------------------------------------
# Scoreboard
# --------------------------------------------------------------------------

#: (registry column, header, formatter). Order is the column order.
SCOREBOARD_COLUMNS: tuple[tuple[str, str, Callable], ...] = (
    ("strategy", "strategy", str),
    ("cost_model", "costs", str),
    ("cagr", "CAGR", theme.pct),
    ("ann_vol", "vol", theme.pct),
    ("sharpe", "Sharpe", theme.num),
    ("max_drawdown", "maxDD", theme.pct),
    ("calmar", "Calmar", theme.num),
    ("ann_turnover", "turnover", lambda v: theme.pct(v, 0)),
    ("cost_drag", "cost drag", theme.pct),
    ("avg_positions", "names", lambda v: theme.num(v, 0)),
    ("information_ratio", "IR", theme.num),
)


def scoreboard(runs: pd.DataFrame, *, highlight_best: bool = True,
               caption: str = "") -> Table:
    """The comparison table. One row per run, best value in each column highlighted."""
    if runs.empty:
        return Table([c[1] for c in SCOREBOARD_COLUMNS], [], caption=caption)

    cols = [c for c in SCOREBOARD_COLUMNS if c[0] in runs.columns]
    best: dict[str, float] = {}
    if highlight_best:
        for key, header, _ in cols:
            d = theme.direction(key)
            if d == 0:
                continue
            vals = pd.to_numeric(runs[key], errors="coerce").dropna()
            if vals.empty:
                continue
            best[key] = float(vals.max() if d > 0 else vals.min())

    rows: list[list[Cell]] = []
    for _, r in runs.iterrows():
        row: list[Cell] = []
        for key, header, fmt in cols:
            value = r.get(key)
            emphasis = ""
            if key in best and value is not None:
                try:
                    if math.isclose(float(value), best[key], rel_tol=1e-12):
                        emphasis = "good"
                except (TypeError, ValueError):
                    pass
            row.append(_text(value, emphasis) if fmt is str
                       else _cell(value, fmt, emphasis=emphasis))
        rows.append(row)
    return Table([c[1] for c in cols], rows, caption=caption)


# --------------------------------------------------------------------------
# Deflation
# --------------------------------------------------------------------------

#: The threshold below which a searched result is not distinguishable from luck. Not a
#: law of nature - a convention, and the report says so rather than implying otherwise.
DSR_THRESHOLD = 0.95


def deflation_panel(deflation: dict) -> Table:
    """The multiple-testing panel: what the search cost the winner's credibility."""
    if not deflation or "deflated_sharpe" not in deflation:
        return Table(["", ""], [], caption="not enough observations to deflate")

    dsr = deflation.get("deflated_sharpe")
    survived = isinstance(dsr, (int, float)) and math.isfinite(dsr) and dsr >= DSR_THRESHOLD

    rows = [
        [_text("configurations tried (n_trials)"),
         _cell(deflation.get("n_trials"), theme.count)],
        [_text("spread of their Sharpes"),
         _cell(deflation.get("trial_sharpe_std"), theme.num)],
        [_text("monthly observations"),
         _cell(deflation.get("n_months"), theme.count)],
        [_text("Sharpe (annualised, daily curve)"),
         _cell(deflation.get("sharpe_annualised_daily"), theme.num)],
        [_text("Sharpe (annualised, monthly curve)"),
         _cell(deflation.get("sharpe_annualised_monthly"), theme.num,
               title="deflation uses this one - monthly returns are closer to independent")],
        [_text("bar set by the search"),
         _cell(deflation.get("expected_max_sharpe_annualised"), theme.num, emphasis="warn",
               title="the Sharpe the luckiest of n_trials worthless strategies would post")],
        [_text("probabilistic Sharpe vs zero"),
         _cell(deflation.get("psr_vs_zero"), lambda v: theme.num(v, 3))],
        [_text("DEFLATED SHARPE"),
         _cell(dsr, lambda v: theme.num(v, 3),
               emphasis="good" if survived else "bad",
               title=f"threshold {DSR_THRESHOLD}")],
    ]
    caption = ("Survives the search that produced it." if survived else
               "Below the 0.95 convention: not distinguishable from the best of "
               f"{deflation.get('n_trials', '?')} lucky draws.")
    return Table(["", "value"], rows, aligns=["left", "right"],
                 caption=caption, sortable=False)


# --------------------------------------------------------------------------
# Honesty
# --------------------------------------------------------------------------

def diagnostics(record: pd.Series) -> Table:
    """Per-run diagnostics that change what the headline number means."""
    def flag(value: float | int | None, bad_above: float) -> str:
        try:
            return "bad" if value is not None and float(value) > bad_above else ""
        except (TypeError, ValueError):
            return ""

    rows = [
        [_text("price coverage (worst rebalance)"),
         _cell(record.get("coverage_min"), theme.pct,
               emphasis="bad" if _lt(record.get("coverage_min"), 0.7) else "",
               title="share of the point-in-time index that actually had prices")],
        [_text("price coverage (median)"), _cell(record.get("coverage_median"), theme.pct)],
        [_text("forced exits"), _cell(record.get("forced_exits"), theme.count)],
        [_text("of which unresolved"),
         _cell(record.get("unresolved_exits"), theme.count,
               emphasis=flag(record.get("unresolved_exits"), 0),
               title="exited with no recorded reason; treated as an index removal")],
        [_text("orders using a fallback spread"),
         _cell(record.get("spread_fallback_orders"), theme.count,
               emphasis=flag(record.get("spread_fallback_orders"), 0))],
        [_text("orders that did not fill"),
         _cell(record.get("unfilled_orders"), theme.count)],
        [_text("rebalances"), _cell(record.get("n_rebalances"), theme.count)],
        [_text("touched the holdout"),
         _text("YES" if record.get("touched_holdout") else "no",
               "bad" if record.get("touched_holdout") else "")],
        [_text("built from a dirty working tree"),
         _text("YES" if record.get("git_dirty") else "no",
               "warn" if record.get("git_dirty") else "",
               title="a result from uncommitted code cannot be reproduced exactly")],
    ]
    return Table(["diagnostic", "value"], rows, aligns=["left", "right"], sortable=False)


def costs(record: pd.Series) -> Table:
    """Where the money went."""
    traded = record.get("traded_notional") or 0.0
    total = record.get("total_cost") or 0.0
    rows = [
        [_text("total charged"), _cell(total, theme.money)],
        [_text("commission"), _cell(record.get("commission"), theme.money)],
        [_text("spread"), _cell(record.get("spread_cost"), theme.money)],
        [_text("traded notional"), _cell(traded, theme.compact)],
        [_text("cost in bp of traded"),
         _cell((total / traded) if traded else None, lambda v: theme.bp(v, 1))],
        [_text("orders"), _cell(record.get("n_orders"), theme.count)],
        [_text("annual cost drag"),
         _cell(record.get("cost_drag"), theme.pct,
               emphasis="bad" if _gt(record.get("cost_drag"), 0.02) else "")],
    ]
    return Table(["cost", "value"], rows, aligns=["left", "right"], sortable=False)


def studies(df: pd.DataFrame) -> Table:
    """One row per search. `runs` counts log lines; `trials` counts distinct ideas."""
    if df.empty:
        return Table(["study", "runs", "trials", "best Sharpe", "best strategy"], [])
    rows = []
    for _, r in df.iterrows():
        rows.append([
            _text(r.get("study")),
            _cell(r.get("runs"), theme.count),
            _cell(r.get("trials"), theme.count,
                  title="distinct configurations - this is what the deflated Sharpe uses"),
            _cell(r.get("best_sharpe"), theme.num),
            _text(r.get("best_strategy")),
            _cell(r.get("touched_holdout"), theme.count,
                  emphasis="bad" if _gt(r.get("touched_holdout"), 0) else ""),
            _text(str(r.get("last", ""))[:10]),
        ])
    return Table(["study", "runs", "trials", "best Sharpe", "best strategy",
                  "holdout looks", "last run"], rows)


def holdout_ledger(df: pd.DataFrame) -> Table:
    """Every recorded look at the reserved period. Empty is the good state."""
    if df.empty:
        return Table(["when", "strategy", "study", "mode", "window"], [],
                     caption="Never looked at. Keep it that way until you have a final "
                             "candidate.")
    rows = []
    for _, r in df.iterrows():
        rows.append([
            _text(str(r.get("at", ""))[:19].replace("T", " ")),
            _text(r.get("strategy")),
            _text(r.get("study")),
            _text(r.get("mode"), "bad"),
            _text(f"{r.get('start', '')} .. {r.get('end', '')}"),
        ])
    return Table(["when", "strategy", "study", "mode", "window"], rows, sortable=False,
                 caption=f"{len(df)} look(s). Each one costs the holdout some of its "
                         "value as an independent test.")


def exits(df: pd.DataFrame, limit: int = 25) -> Table:
    """Positions resolved outside a rebalance - delistings, and gaps in the feed."""
    if df is None or df.empty:
        return Table(["date", "ticker", "reason", "return", "proceeds"], [],
                     caption="No position had to be resolved outside a rebalance.")
    d = df.sort_values("date")
    rows = []
    for _, r in d.head(limit).iterrows():
        reason = str(r.get("reason", ""))
        rows.append([
            _text(r.get("date")),
            _text(r.get("ticker")),
            _text(reason, "bad" if reason == "bankruptcy"
                  else "warn" if reason in ("unresolved", "price_gap") else ""),
            _cell(r.get("delist_return"), theme.pct),
            _cell(r.get("proceeds"), theme.money),
        ])
    caption = ""
    if len(d) > limit:
        caption = f"showing {limit} of {len(d)}"
    return Table(["date", "ticker", "reason", "return", "proceeds"], rows, caption=caption)


def coverage_note(record: pd.Series) -> str:
    """One sentence stating what a run's coverage means for its numbers."""
    lo = record.get("coverage_min")
    if lo is None or (isinstance(lo, float) and not math.isfinite(lo)):
        return ""
    return (f"At its worst rebalance this run could trade {theme.pct(lo, 1)} of the "
            "index. The names it could not trade are disproportionately the ones that "
            "later delisted, so the tradable subset is biased toward survivors.")


def trade_years(df: pd.DataFrame) -> Table:
    """Orders, notional and cost per year. The shape of the trading, at a glance."""
    from ..backtest.trades import summarise
    if df is None or df.empty:
        return Table(["year", "orders", "buys", "sells", "names", "notional",
                      "cost", "bp"], [], caption="This strategy placed no orders.")
    rows = []
    for _, r in summarise(df).iterrows():
        rows.append([
            _text(r["year"]),
            _cell(r["orders"], theme.count),
            _cell(r["buys"], theme.count),
            _cell(r["sells"], theme.count),
            _cell(r["names"], theme.count),
            _cell(r["notional"], theme.money),
            _cell(r["cost"], theme.money),
            _cell(r["cost_bps"], lambda v: f"{v:.1f}",
                  emphasis="warn" if _gt(r["cost_bps"], 20) else ""),
        ])
    return Table(["year", "orders", "buys", "sells", "names", "notional", "cost", "bp"],
                 rows,
                 caption="`bp` is cost as basis points of the notional traded that year.")


def trade_leaders(df: pd.DataFrame, limit: int = 20) -> Table:
    """The names a strategy actually spent its money - and its costs - on."""
    from ..backtest.trades import most_traded
    cols = ["ticker", "orders", "notional", "cost", "first", "last"]
    if df is None or df.empty:
        return Table(cols, [])
    rows = []
    for _, r in most_traded(df, limit).iterrows():
        rows.append([
            _text(r["ticker"]),
            _cell(r["orders"], theme.count),
            _cell(r["notional"], theme.money),
            _cell(r["cost"], theme.money),
            _text(r["first"]),
            _text(r["last"]),
        ])
    return Table(cols, rows,
                 caption=f"Top {limit} by notional traded over the whole run.")


def trade_sample(df: pd.DataFrame, limit: int = 40) -> Table:
    """The most recent orders, exactly as exported.

    A sample rather than the whole ledger: 12,000 rows of HTML is a slow page and nobody
    reads it. The full list is the CSV above, which is the artifact this section exists
    to hand over.
    """
    cols = ["date", "ticker", "side", "shares", "price", "notional", "cost", "reason"]
    if df is None or df.empty:
        return Table(cols, [])
    recent = df.sort_values(["date"]).tail(limit).iloc[::-1]
    rows = []
    for _, r in recent.iterrows():
        side = str(r["side"])
        reason = str(r.get("reason", ""))
        rows.append([
            _text(r["date"]),
            _text(r["ticker"]),
            _text(side, "good" if side == "BUY" else "bad"),
            _cell(r["shares"], lambda v: f"{v:,.2f}"),
            _cell(r["price"], theme.money),
            _cell(r["notional"], theme.money),
            _cell(r["cost"], theme.money),
            _text(reason, "warn" if reason != "rebalance" else "muted"),
        ])
    return Table(cols, rows,
                 caption=f"The last {len(rows)} orders. `price` is the AS-TRADED open - "
                         "the price a broker printed that morning - so it compares "
                         "directly against any quote source.")


def trade_reconciliation(report: dict) -> Table:
    """The audit: do the orders add up to the equity curve?"""
    rows = []
    labels = {
        "n_orders_ledger": "orders recorded",
        "n_orders_charged": "orders charged by the cost model",
        "cost_charged": "cost in the headline",
        "cost_in_ledger": "cost attributed to orders",
        "worst_cash_gap": "worst cash disagreement ($)",
        "cost_matches": "every dollar of cost lands on an order",
        "cash_reconciles": "cash flows replay the NAV path",
        "order_count_ok": "no order traded without being charged",
    }
    for key, label in labels.items():
        if key not in report:
            continue
        value = report[key]
        if isinstance(value, bool):
            rows.append([_text(label), _text("yes" if value else "NO",
                                             "good" if value else "bad")])
        elif isinstance(value, float):
            rows.append([_text(label), _cell(value, theme.money)])
        else:
            rows.append([_text(label), _cell(value, theme.count)])
    return Table(["check", "result"], rows, aligns=["left", "right"], sortable=False,
                 caption="Every one of these is arithmetic, not judgement. A 'NO' means "
                         "the trade list and the equity curve are not the same run.")


# --------------------------------------------------------------------------
# Strategy reports
# --------------------------------------------------------------------------

def cost_sensitivity(results: list) -> Table:
    """The same strategy under all three cost settings.

    Always all three, never one. The spread estimate is the weakest number in the chain
    (ADR-020), so a strategy that only works under `optimistic` is a bet that the
    estimator is wrong in your favour rather than a strategy. `random_weight` posts the
    second-best Sharpe in the whole suite under optimistic costs.
    """
    cols = ["costs", "CAGR", "vol", "Sharpe", "maxDD", "turnover", "cost drag", "names"]
    rows = []
    for r in results:
        p = r.performance
        setting = str(r.config.get("cost_model", ""))
        rows.append([
            _text(setting, "muted" if setting == "optimistic" else ""),
            _cell(p.cagr, theme.pct, emphasis="bad" if _lt(p.cagr, 0) else ""),
            _cell(p.ann_vol, theme.pct),
            _cell(p.sharpe, theme.num),
            _cell(p.max_drawdown, theme.pct),
            _cell(p.ann_turnover, theme.pct),
            _cell(p.cost_drag, theme.pct,
                  emphasis="warn" if _gt(p.cost_drag, 0.02) else ""),
            _cell(p.avg_positions, theme.num),
        ])
    return Table(cols, rows, sortable=False,
                 caption="Optimistic charges commission only; realistic adds one "
                         "estimated half-spread; pessimistic adds two.")


def versus_benchmark(result, bench, label: str = "SPY") -> Table:
    """A strategy against the index over EXACTLY its own dates.

    The only comparison in this project that means anything. Strategies here do not all
    cover the same window - anything using XBRL fundamentals starts in 2010 - and SPY
    returned 10.42%/yr from 2007-04 against 15.66%/yr from 2010-07.
    """
    p = result.performance
    pairs = [("CAGR", p.cagr, getattr(bench, "cagr", None), theme.pct, True),
             ("volatility", p.ann_vol, getattr(bench, "ann_vol", None), theme.pct, False),
             ("Sharpe", p.sharpe, getattr(bench, "sharpe", None), theme.num, True),
             ("max drawdown", p.max_drawdown,
              getattr(bench, "max_drawdown", None), theme.pct, True),
             ("hit rate", p.hit_rate, getattr(bench, "hit_rate", None), theme.pct, True)]
    rows = []
    for name, mine, theirs, fmt, higher_is_better in pairs:
        diff = (mine - theirs) if (theirs is not None and mine is not None) else None
        good = diff is not None and ((diff > 0) if higher_is_better else (diff < 0))
        rows.append([
            _text(name),
            _cell(mine, fmt),
            _cell(theirs, fmt),
            _cell(diff, fmt,
                  emphasis="good" if good else ("bad" if diff is not None else "")),
        ])
    return Table(["", "strategy", label, "difference"], rows, sortable=False,
                 aligns=["left", "right", "right", "right"],
                 caption=f"{label} measured over the strategy's own window, on the same "
                         "engine and the same total-return adjustment.")


def annual_returns(table_df) -> Table:
    """Year by year against the benchmark. Where a smooth curve stops flattering."""
    if table_df is None or not len(table_df):
        return Table(["year", "strategy", "benchmark", "excess"], [])
    has_bench = "benchmark" in table_df.columns
    cols = ["year", "strategy"] + (["benchmark", "excess"] if has_bench else [])
    rows = []
    for year, r in table_df.iterrows():
        row = [_text(year),
               _cell(r.get("strategy"), theme.pct,
                     emphasis="bad" if _lt(r.get("strategy"), 0) else "")]
        if has_bench:
            row.append(_cell(r.get("benchmark"), theme.pct))
            row.append(_cell(r.get("excess"), theme.pct,
                             emphasis="good" if _gt(r.get("excess"), 0) else "bad"))
        rows.append(row)
    return Table(cols, rows,
                 caption="Calendar years. A partial first or last year is shown as it "
                         "was actually earned, not annualised.")


def holdings_snapshot(result, limit: int = 25) -> Table:
    """The most recent target portfolio: what this strategy would own today."""
    w = getattr(result, "weights", None)
    if w is None or not len(w):
        return Table(["ticker", "weight"], [])
    last = w.iloc[-1]
    last = last[last > 0].sort_values(ascending=False)
    tick = {}
    trades = getattr(result, "trades", None)
    if trades is not None and len(trades):
        tick = dict(zip(trades["security_id"], trades["ticker"]))
    rows = [[_text(tick.get(sid, sid)), _cell(float(v), theme.pct)]
            for sid, v in last.head(limit).items()]
    caption = f"Target weights at the final rebalance, {w.index[-1]}."
    if len(last) > limit:
        caption += f" Showing {limit} of {len(last)} positions."
    return Table(["ticker", "weight"], rows, aligns=["left", "right"], caption=caption)


def strategy_features(names: tuple, coverage) -> Table:
    """Which features a strategy reads, and how often each is actually populated.

    A strategy is only as available as its scarcest input. `gross_profitability` is
    tagged by 338 of 649 companies, so anything requiring it trades a narrower universe
    than its headline coverage number suggests.
    """
    from ..features.catalog import describe
    cols = ["feature", "family", "what it is", "populated"]
    if not names:
        return Table(cols, [], aligns=["left", "left", "left", "right"],
                     caption="This strategy computes its own inputs from the price "
                             "panel and reads no shared features.")
    lookup = {}
    if coverage is not None and len(coverage):
        lookup = dict(zip(coverage["feature"], coverage["recent"]))
    rows = []
    for n in names:
        doc = describe(n)
        pct = lookup.get(n)
        rows.append([
            _text(n),
            _text(doc.family, "muted"),
            _text(doc.what),
            _cell(pct, theme.pct, emphasis="warn" if _lt(pct, 0.5) else ""),
        ])
    return Table(cols, rows, aligns=["left", "left", "left", "right"],
                 caption="`populated` is the share of panel securities with a value over "
                         "the last twelve months; ~80% is the practical ceiling because "
                         "it counts names outside the index too.")


# --------------------------------------------------------------------------
# The feature report
# --------------------------------------------------------------------------

def feature_catalog(names, coverage, family=None) -> Table:
    """Every feature with what it is, which end is good, and how often it exists."""
    from ..features.catalog import describe
    lookup, first = {}, {}
    if coverage is not None and len(coverage):
        lookup = dict(zip(coverage["feature"], coverage["recent"]))
        first = dict(zip(coverage["feature"], coverage["first_date"]))
    rows = []
    for n in names:
        doc = describe(n)
        if family is not None and doc.family != family:
            continue
        pct = lookup.get(n)
        rows.append([
            _text(n),
            _text(doc.what),
            _text(doc.reading, _reading_emphasis(doc.reading)),
            _text(str(first.get(n, ""))[:7], "muted"),
            _cell(pct, theme.pct, emphasis="warn" if _lt(pct, 0.5) else ""),
        ])
    return Table(["feature", "what it is", "which end is good", "from", "populated"],
                 rows, aligns=["left", "left", "left", "left", "right"])


def _reading_emphasis(reading: str) -> str:
    head = reading.split(".")[0].lower()
    if head.startswith("context") or "ambiguous" in head or "neither" in head:
        return "muted"
    if "avoid" in head:
        return "bad"
    return ""


def feature_families(names, coverage) -> Table:
    """One row per family: how many features, and how populated they are as a group."""
    from ..features.catalog import describe
    lookup = {}
    if coverage is not None and len(coverage):
        lookup = dict(zip(coverage["feature"], coverage["recent"]))
    groups = {}
    for n in names:
        groups.setdefault(describe(n).family, []).append(lookup.get(n, float("nan")))
    rows = []
    for fam, values in groups.items():
        finite = [v for v in values if v == v]
        mean = sum(finite) / len(finite) if finite else float("nan")
        rows.append([
            _text(fam),
            _cell(len(values), theme.count),
            _cell(mean, theme.pct, emphasis="warn" if _lt(mean, 0.5) else ""),
        ])
    return Table(["family", "features", "average populated"], rows,
                 aligns=["left", "right", "right"])


def leakage_summary(report: dict) -> Table:
    """The result of deleting the future and rebuilding. The check that decides the rest."""
    if not report:
        return Table(["check", "result"], [], sortable=False,
                     caption="Not run for this report. "
                             "`python -m sp500lab features check`")
    ok = bool(report.get("ok"))
    rows = [
        [_text("features compared"), _cell(len(report.get("features", [])), theme.count)],
        [_text("rebalance dates compared"),
         _cell(report.get("rows_compared"), theme.count)],
        [_text("securities"), _cell(report.get("securities"), theme.count)],
        [_text("pretended today was"), _text(str(report.get("cut_at", "")))],
        [_text("every feature bit-identical"),
         _text("yes" if ok else "NO", "good" if ok else "bad")],
    ]
    failed = report.get("failed") or []
    if failed:
        rows.append([_text("reading the future"), _text(", ".join(failed[:6]), "bad")])
    return Table(["check", "result"], rows, aligns=["left", "right"], sortable=False,
                 caption="Rebuilt from a price panel that physically ends at the cut "
                         "date, with every filing published after it deleted. Anything "
                         "that changes was reading data it could not have had.")


def strategy_roster(specs: list) -> Table:
    """Every strategy, what it claims, how it was found, and where its report is.

    `found by` is not decoration. A written strategy's Sharpe is one draw; a searched
    one's is the maximum of however many the search took, and the maximum of N draws is
    high whether or not there is any signal. Putting the two in one sorted table without
    saying which is which would be the single most misleading thing this project could
    print, so the column is there and the deflated Sharpe is beside it.
    """
    cols = ["strategy", "the claim", "found by", "window", "CAGR", "Sharpe",
            "vs index", "deflated"]
    rows = []
    for spec in specs:
        beat = spec.get("d_sharpe")
        trials = spec.get("n_trials")
        dsr = spec.get("deflated_sharpe")
        # Evolved, and ONLY evolved. A hand-written strategy logged into a study
        # alongside nineteen others is not a search result, and labelling it one because
        # the study happens to hold sixty runs would be a different lie in the opposite
        # direction.
        searched = bool(spec.get("evolved"))
        rows.append([
            Cell(spec["name"], spec["name"], href=spec.get("href", "")),
            _text(_short_claim(spec.get("claim", ""))),
            _text(f"evolved, {trials:,} trials" if searched and trials
                  else ("evolved" if searched else "written"),
                  "warn" if searched else "muted"),
            _text(spec.get("window", ""), "muted"),
            _cell(spec.get("cagr"), theme.pct),
            _cell(spec.get("sharpe"), theme.num),
            _cell(beat, lambda v: f"{v:+.2f}",
                  emphasis="good" if _gt(beat, 0) else "bad"),
            _cell(dsr, theme.num,
                  emphasis="good" if _gt(dsr, 0.95) else ("bad" if dsr is not None
                                                          else ""),
                  title="Probability this survives the number of configurations that "
                        "were evaluated before it was picked."),
        ])
    return Table(cols, rows,
                 aligns=["left", "left", "left", "left", "right", "right", "right",
                         "right"],
                 caption="`vs index` is the Sharpe difference against SPY over each "
                         "strategy's OWN window — the only comparison here that is "
                         "like-for-like. `deflated` corrects for how many configurations "
                         "were evaluated before this one was picked: for an evolved "
                         "strategy that is its whole search, and for a written one it is "
                         "the fact that all of these were run and the best was read. "
                         "Below 0.95, a result is not distinguishable from the luckiest "
                         "of that many draws. Click a name for its full report.")


def _short_claim(claim: str, limit: int = 88) -> str:
    """A claim trimmed to fit a table cell without a scrollbar."""
    text = " ".join(str(claim).split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;:") + "…"


def feature_usage(names) -> Table | None:
    """Which strategies read which shared features.

    Lives here rather than in the CLI because it is a table, and every other table in
    the report set is built in this module.
    """
    from ..backtest.strategy import get_strategy

    rows = []
    for name in names:
        try:
            strat = get_strategy(name)
        except Exception:                                         # noqa: BLE001
            continue
        needed = tuple(getattr(strat, "requires_features", ()) or ())
        rows.append([
            _text(name),
            Cell(str(len(needed)), len(needed)),
            _text(", ".join(needed) if needed
                  else "none — computes its own from the price panel",
                  "" if needed else "muted"),
        ])
    if not rows:
        return None
    return Table(["strategy", "features", "which"], rows,
                 aligns=["left", "right", "left"],
                 caption="A strategy with no shared features is not worse; it is older "
                         "than the feature layer.")
