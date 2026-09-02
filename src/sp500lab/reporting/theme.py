"""Colours, typography and number formatting. The only file that knows what things look like.

Everything visual lives here so that changing the look is one edit rather than a search
through chart code. `series.py` and `tables.py` never import it - they produce numbers,
and numbers have no colour.

Two rules that are load-bearing rather than cosmetic:

**One meaning per colour.** The benchmark is always the same grey dashed line, drawdown
is always the same red fill, and a warning is always the same amber. A reader who learns
the palette once should not have to re-learn it per chart. Strategy series get their
colour by position in a fixed categorical ramp, so the same strategy keeps its colour
across every chart on a page.

**Formatting encodes precision, not decoration.** Returns are shown to two decimal places
because the fourth is noise; basis points get one because that is where the engine's
acceptance tolerances live; Sharpe gets two. A number printed to more precision than it
has is a small lie that compounds over a long document.

The palette is Tableau-10, which is widely tested for categorical separation and stays
distinguishable in the common forms of colour blindness. Dark-mode variants are lifted in
lightness rather than re-hued, so a chart read in one theme and then the other does not
change which series is which.
"""

from __future__ import annotations

from .util import finite as _finite

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------

#: Categorical ramp for strategy series. Assigned by position and reused across every
#: chart in a report, so a strategy keeps one colour throughout.
SERIES_COLORS: tuple[str, ...] = (
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#86BCB6",
)

SERIES_COLORS_DARK: tuple[str, ...] = (
    "#6BA3D0", "#FFA94D", "#74C46B", "#FF7A78", "#8FD6D1",
    "#F5D95B", "#CE9AC0", "#FFBAC2", "#BE9679", "#A4DAD4",
)

#: Reserved meanings. Never reuse these for a strategy series.
BENCHMARK_COLOR = "#8A8A8A"
POSITIVE = "#2E8B57"
NEGATIVE = "#C0392B"
WARNING = "#B8860B"
MUTED = "#9AA0A6"


def series_color(index: int) -> str:
    return SERIES_COLORS[index % len(SERIES_COLORS)]


def series_color_dark(index: int) -> str:
    return SERIES_COLORS_DARK[index % len(SERIES_COLORS_DARK)]


# --------------------------------------------------------------------------
# Number formatting
# --------------------------------------------------------------------------

def pct(value: float | None, dp: int = 2, signed: bool = False) -> str:
    """0.1041 -> '10.41%'. `signed` forces a leading + on positives."""
    if value is None or not _finite(value):
        return "—"
    s = f"{value * 100:+.{dp}f}%" if signed else f"{value * 100:.{dp}f}%"
    return s


def bp(value: float | None, dp: int = 1) -> str:
    """0.00176 -> '17.6bp'. Basis points are how the engine states its tolerances."""
    if value is None or not _finite(value):
        return "—"
    return f"{value * 1e4:.{dp}f}bp"


def num(value: float | None, dp: int = 2) -> str:
    if value is None or not _finite(value):
        return "—"
    return f"{value:.{dp}f}"


def money(value: float | None, dp: int = 0) -> str:
    if value is None or not _finite(value):
        return "—"
    return f"${value:,.{dp}f}"


def count(value: float | None) -> str:
    if value is None or not _finite(value):
        return "—"
    return f"{int(round(value)):,}"


def compact(value: float | None) -> str:
    """Large money, shortened: 3_706_372 -> '$3.7M'. For axis labels and tiles."""
    if value is None or not _finite(value):
        return "—"
    a = abs(value)
    for cut, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= cut:
            return f"${value / cut:,.1f}{suffix}"
    return f"${value:,.0f}"


def multiple(value: float | None, dp: int = 2) -> str:
    """A growth multiple: 4.15 -> '4.15x'."""
    if value is None or not _finite(value):
        return "—"
    return f"{value:.{dp}f}x"


# --------------------------------------------------------------------------
# Semantic direction
# --------------------------------------------------------------------------

#: Which way each metric improves. Explicit rather than inferred from the name, because
#: guessing gets it wrong and the failure is silent.
#:
#: ⚠️ **`max_drawdown` is stored NEGATIVE** (`equity / peak - 1`), so a *higher* value is
#: the better one: -0.39 beats -0.78. A rule of "drawdown is bad, therefore lower is
#: better" inverts it and highlights the worst drawdown in the table as the best - which
#: is exactly what happened on the first version of this file, and it looked plausible
#: enough in code to survive review. It only showed up on screen.
#:
#: 0 means neither direction is better. `avg_positions` is the case worth naming: a
#: concentrated portfolio is not better or worse than a diversified one, it is a
#: different bet, and colouring one of them green would be an opinion the table has no
#: business having.
METRIC_DIRECTION: dict[str, int] = {
    # higher is better
    "cagr": 1, "total_return": 1, "sharpe": 1, "sharpe_monthly": 1, "sortino": 1,
    "calmar": 1, "information_ratio": 1, "alpha": 1, "hit_rate": 1,
    "coverage_min": 1, "coverage_median": 1, "deflated_sharpe": 1, "psr_vs_zero": 1,
    "max_drawdown": 1,          # negative-valued: closer to zero is better
    # lower is better
    "ann_vol": -1, "volatility": -1, "ann_turnover": -1, "turnover": -1,
    "cost_drag": -1, "time_under_water": -1, "total_cost": -1,
    "unresolved_exits": -1, "spread_fallback_orders": -1, "unfilled_orders": -1,
    "var_95": 1, "cvar_95": 1,  # also negative-valued
    # neither
    "strategy": 0, "study": 0, "costs": 0, "cost_model": 0, "date": 0, "run_id": 0,
    "names": 0, "avg_positions": 0, "n_rebalances": 0, "n_trials": 0, "beta": 0,
    "n_orders": 0, "traded_notional": 0, "seed": 0, "n_months": 0,
    # forward testing (ADR-033): a decay is a change, and less decay is better. `decay_z`
    # is that change in standard errors, so it runs the same way. `fresh_months` and
    # `look_number` are facts about the evidence, not qualities of the strategy.
    "decay_sharpe": 1, "decay_sharpe_monthly": 1, "decay_cagr": 1, "decay_d_sharpe": 1,
    "decay_z": 1, "psr_vs_research": 1, "psr_vs_benchmark": 1,
    "forward_sharpe": 1, "forward_cagr": 1, "forward_d_sharpe": 1,
    "research_sharpe": 1, "research_cagr": 1, "research_d_sharpe": 1,
    "forward_max_drawdown": 1, "seal_drift_sharpe": 0,
    "fresh_months": 0, "look_number": 0, "verdict": 0, "seal_mode": 0,
    "holdout_looks_total": -1,
}


def direction(metric: str) -> int:
    """+1 if higher is better, -1 if lower is better, 0 if neither or unknown.

    Unknown metrics return 0 rather than guessing. A column with no highlight is a
    small loss; a column highlighted the wrong way round is a wrong answer.
    """
    return METRIC_DIRECTION.get(metric.strip().lower().replace(" ", "_"), 0)
