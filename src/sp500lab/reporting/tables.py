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


@dataclass
class Cell:
    text: str
    sort_key: float | str = 0.0
    emphasis: str = ""        # "" | "good" | "bad" | "warn" | "muted"
    title: str = ""           # hover text, for a number that needs a caveat


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


def _lt(value: Any, threshold: float) -> bool:
    try:
        return float(value) < threshold
    except (TypeError, ValueError):
        return False


def _gt(value: Any, threshold: float) -> bool:
    try:
        return float(value) > threshold
    except (TypeError, ValueError):
        return False
