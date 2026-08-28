"""Turning equity curves into plot-ready numbers. Pure, and deliberately unaware of HTML.

Everything here takes a curve and returns arrays. No colours, no SVG, no markup - which
means every function is testable by asserting on numbers, and a change to how a chart
looks can never quietly change what it shows.

The frequency question, restated
--------------------------------
Curves arriving from the registry are **month-end** (ADR-027). That is the frequency the
strategy actually acts at, and the one at which its returns are approximately
independent. Statistics computed here therefore annualise from 12 periods per year, not
252. Mixing the two is how a rolling-Sharpe chart ends up quietly overstating itself by
sqrt(21).

Drawdown is the exception worth noting: a monthly curve understates the true peak-to-
trough, because an intra-month low is invisible to it. The headline max drawdown in the
registry comes from the *daily* curve and is the honest number; the drawdown chart here
is a shape, not a measurement. `drawdown()` says so in its docstring and the report
repeats it in the caption rather than letting a reader assume otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Month-end curves: twelve observations a year.
PERIODS_PER_YEAR = 12


@dataclass
class LineSeries:
    """One named line. `x` are date strings, `y` may contain NaN for gaps."""

    name: str
    x: list[str]
    y: list[float]
    kind: str = "strategy"          # strategy | benchmark | gross | reference
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.x)

    @property
    def finite_y(self) -> list[float]:
        return [v for v in self.y if v is not None and math.isfinite(v)]

    def last(self) -> float | None:
        vals = self.finite_y
        return vals[-1] if vals else None


def _clean(curve: pd.Series) -> pd.Series:
    return curve.dropna().astype(float)


# --------------------------------------------------------------------------
# Level series
# --------------------------------------------------------------------------

def equity(curve: pd.Series, name: str, *, rebase: float = 1.0,
           kind: str = "strategy") -> LineSeries:
    """Growth of `rebase` invested at the first observation."""
    s = _clean(curve)
    if s.empty:
        return LineSeries(name, [], [], kind)
    scaled = s / s.iloc[0] * rebase
    return LineSeries(name, [str(i) for i in scaled.index],
                      [float(v) for v in scaled.to_numpy()], kind)


def align(curves: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Reindex several curves onto their shared dates and rebase each to 1.0.

    Comparing curves that start on different dates is the most common way a chart
    misleads: whichever line starts earliest looks best purely from having compounded
    longer. Restricting to the common window and rebasing there removes that entirely.
    """
    cleaned = {k: _clean(v) for k, v in curves.items() if len(_clean(v)) > 1}
    if not cleaned:
        return {}
    common = None
    for s in cleaned.values():
        idx = pd.Index(s.index)
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) < 2:
        return {}
    common = common.sort_values()
    return {k: (s.reindex(common) / s.reindex(common).iloc[0]) for k, s in cleaned.items()}


def drawdown(curve: pd.Series, name: str = "drawdown") -> LineSeries:
    """Fraction below the running peak, as a non-positive series.

    ⚠️ Computed on a month-end curve, so an intra-month trough is invisible and the
    trough shown here is shallower than the true one. The registry's `max_drawdown`
    comes from the daily curve and is the number to quote.
    """
    s = _clean(curve)
    if s.empty:
        return LineSeries(name, [], [], "drawdown")
    dd = s / s.cummax() - 1.0
    return LineSeries(name, [str(i) for i in dd.index],
                      [float(v) for v in dd.to_numpy()], "drawdown")


def relative(curve: pd.Series, benchmark: pd.Series, name: str) -> LineSeries:
    """Cumulative ratio to the benchmark, rebased to 1.0. Rising = outperforming.

    Far more legible than two near-identical equity curves: over nineteen years two
    lines within a few percent a year of each other are visually indistinguishable, and
    their ratio is not.
    """
    a, b = _clean(curve), _clean(benchmark)
    common = pd.Index(a.index).intersection(pd.Index(b.index)).sort_values()
    if len(common) < 2:
        return LineSeries(name, [], [], "relative")
    ratio = (a.reindex(common) / a.reindex(common).iloc[0]) / \
            (b.reindex(common) / b.reindex(common).iloc[0])
    return LineSeries(name, [str(i) for i in ratio.index],
                      [float(v) for v in ratio.to_numpy()], "relative")


# --------------------------------------------------------------------------
# Rolling statistics
# --------------------------------------------------------------------------

def rolling_sharpe(curve: pd.Series, window: int = 36, name: str = "rolling Sharpe",
                   risk_free: float = 0.0) -> LineSeries:
    """Annualised Sharpe over a trailing window of months. Default 3 years.

    Trailing, so every point is knowable on its own date - the same discipline the engine
    applies to features. A centred window would look smoother and be a lookahead.
    """
    r = _returns(curve)
    if len(r) < window:
        return LineSeries(name, [], [], "rolling")
    excess = r - risk_free / PERIODS_PER_YEAR
    mean = excess.rolling(window).mean()
    sd = r.rolling(window).std(ddof=1)
    out = (mean / sd.replace(0.0, np.nan)) * math.sqrt(PERIODS_PER_YEAR)
    out = out.dropna()
    return LineSeries(name, [str(i) for i in out.index],
                      [float(v) for v in out.to_numpy()], "rolling")


def rolling_vol(curve: pd.Series, window: int = 12,
                name: str = "rolling volatility") -> LineSeries:
    r = _returns(curve)
    if len(r) < window:
        return LineSeries(name, [], [], "rolling")
    out = (r.rolling(window).std(ddof=1) * math.sqrt(PERIODS_PER_YEAR)).dropna()
    return LineSeries(name, [str(i) for i in out.index],
                      [float(v) for v in out.to_numpy()], "rolling")


def _returns(curve: pd.Series) -> pd.Series:
    return _clean(curve).pct_change().dropna()


# --------------------------------------------------------------------------
# Period aggregation
# --------------------------------------------------------------------------

def annual_returns(curve: pd.Series) -> tuple[list[str], list[float]]:
    """(years, returns). A partial first or last year is included and labelled as-is."""
    s = _clean(curve)
    if len(s) < 2:
        return [], []
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(s.index))))
    ser = pd.Series(s.to_numpy(), index=idx)
    yearly = ser.resample("YE").last()
    # Seed with the opening value so the first year is a real return, not a gap.
    first = pd.Series([ser.iloc[0]], index=[ser.index[0] - pd.Timedelta(days=1)])
    rets = pd.concat([first, yearly]).pct_change().dropna()
    return [str(i.year) for i in rets.index], [float(v) for v in rets.to_numpy()]


def monthly_grid(curve: pd.Series) -> tuple[list[str], list[str], list[list[float | None]]]:
    """(years, month labels, values[year][month]) plus a trailing YTD column.

    The classic monthly-returns table. Reading down a column shows seasonality; reading
    across a row shows how a year was actually earned, which a single annual number
    hides.
    """
    s = _clean(curve)
    if len(s) < 2:
        return [], [], []
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(s.index))))
    ser = pd.Series(s.to_numpy(), index=idx)
    m = ser.resample("ME").last().pct_change().dropna()
    if m.empty:
        return [], [], []

    years = sorted({d.year for d in m.index})
    months = [pd.Timestamp(2000, i, 1).strftime("%b") for i in range(1, 13)]
    lookup = {(d.year, d.month): float(v) for d, v in m.items()}

    values: list[list[float | None]] = []
    for y in years:
        row: list[float | None] = [lookup.get((y, mo)) for mo in range(1, 13)]
        present = [v for v in row if v is not None]
        row.append(float(np.prod([1 + v for v in present]) - 1) if present else None)
        values.append(row)
    return [str(y) for y in years], months + ["YTD"], values


# --------------------------------------------------------------------------
# Cross-sectional points
# --------------------------------------------------------------------------

@dataclass
class Point:
    x: float
    y: float
    label: str
    kind: str = "strategy"


def risk_return(curves: dict[str, pd.Series],
                benchmark_name: str | None = None) -> list[Point]:
    """(volatility, CAGR) per curve - the scatter that ranks by both at once.

    A table sorted on return hides that the winner took twice the risk. This does not.
    """
    out: list[Point] = []
    for name, curve in curves.items():
        s = _clean(curve)
        if len(s) < 3:
            continue
        r = s.pct_change().dropna()
        years = len(r) / PERIODS_PER_YEAR
        if years <= 0:
            continue
        cagr = float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1)
        vol = float(r.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR))
        kind = "benchmark" if name == benchmark_name else "strategy"
        out.append(Point(x=vol, y=cagr, label=name, kind=kind))
    return out


def summary_stats(curve: pd.Series) -> dict:
    """CAGR, vol, Sharpe, max drawdown from a monthly curve.

    Used only where a registry record is unavailable. Prefer the record: its statistics
    come from the daily curve and its drawdown is the true one.
    """
    s = _clean(curve)
    if len(s) < 3:
        return {}
    r = s.pct_change().dropna()
    years = len(r) / PERIODS_PER_YEAR
    vol = float(r.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR))
    dd = float((s / s.cummax() - 1.0).min())
    return {
        "cagr": float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) if years > 0 else float("nan"),
        "ann_vol": vol,
        "sharpe": float(r.mean() / r.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR))
                  if r.std(ddof=1) > 0 else 0.0,
        "max_drawdown": dd,
        "total_return": float(s.iloc[-1] / s.iloc[0] - 1),
        "n_months": len(r),
        "start": str(s.index[0]),
        "end": str(s.index[-1]),
    }
