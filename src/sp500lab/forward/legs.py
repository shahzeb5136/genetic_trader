"""Turning a `BacktestResult` into a `compare.Leg`, whole or sliced.

A thin seam, and it earns its own module for one reason: `compare.py` is deliberately
pure - it must not import the engine, the panel or a result type, or the verdict rules
could not be tested against hand-built inputs. Something still has to bridge the two,
and putting that bridge in `seal.py` would have made the pre-registration module the
owner of curve arithmetic it has no business owning.

So: `compare.py` defines what a leg IS, this module builds one from a run, and both
`seal.py` and `engine.py` depend on it rather than on each other.

Whole or sliced
---------------
`leg_from_result` reduces an entire run. `leg_from_slice` reduces a date range of one,
which is what makes `mode="continuous"` possible - a single unbroken backtest across the
holdout boundary, cut into a research leg and a forward leg afterwards. Every statistic
is recomputed on the slice; none of them can be recovered by arithmetic on the whole-run
figures, and pretending otherwise is how a "2022-2026 Sharpe" ends up being the
2007-2026 Sharpe with a different label.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..backtest import registry
from .compare import Leg

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..backtest.results import BacktestResult


def leg_from_result(result: "BacktestResult", label: str) -> Leg:
    """Reduce a whole `BacktestResult` to the numbers a paired comparison needs."""
    return leg_from_slice(result, label=label)


def leg_from_slice(result: "BacktestResult", label: str, start: str = "",
                   end: str = "") -> Leg:
    """The same reduction over a sub-window of one run. Empty bounds mean "all of it".

    Slicing rather than re-running is what makes `mode="continuous"` possible: one
    unbroken backtest across the holdout boundary, cut into a research leg and a forward
    leg afterwards. The statistics are recomputed on each slice - a Sharpe over
    2022-2026 is not recoverable from a Sharpe over 2007-2026 - and the per-rebalance
    ledger is sliced the same way, so turnover and cost drag belong to the right window.

    The monthly statistics come from `registry.monthly_stats`, the same resampling the
    deflated Sharpe uses, so the standard errors in `compare.py` and the deflation in
    `registry.deflate()` are computed over the same observations rather than over two
    different notions of "a return" (ADR-026).

    The benchmark comes from `result.benchmark`, which the engine already aligned to the
    strategy's own sessions and adjusted for total return. Rebasing does not move a CAGR
    or a Sharpe, so slicing it gives the index's statistics over exactly these dates -
    the comparison `benchmark.over_window` exists to make, without a second read of the
    price table.
    """
    from ..backtest import metrics

    equity = slice_curve(result.equity, start, end)
    if equity is None or len(equity.dropna()) < 3:
        return Leg(label=label, start=start, end=end)

    gross = slice_curve(result.gross_equity, start, end)
    bench = slice_curve(result.benchmark, start, end)
    ledger = _slice_ledger(result.rebalances, start, end)

    usable_bench = bench is not None and len(bench.dropna()) > 2
    perf = metrics.compute(
        equity,
        benchmark=bench if usable_bench else None,
        turnover=(ledger.set_index("date")["turnover"] if len(ledger) else None),
        positions=(ledger.set_index("date")["n_positions"] if len(ledger) else None),
        gross_equity=gross)
    monthly = registry.monthly_stats(equity)

    bench_perf = None
    if usable_bench:
        try:
            bench_perf = metrics.compute(bench.dropna())
        except ValueError:                                   # pragma: no cover
            bench_perf = None

    return Leg(
        label=label,
        start=str(equity.index[0]),
        end=str(equity.index[-1]),
        n_months=int(monthly["n_months"]),
        cagr=float(perf.cagr),
        sharpe=float(perf.sharpe),
        sharpe_monthly=float(monthly["sharpe"]),
        skew_monthly=float(monthly["skew"]),
        kurtosis_monthly=float(monthly["kurtosis"]),
        ann_vol=float(perf.ann_vol),
        max_drawdown=float(perf.max_drawdown),
        ann_turnover=_or_nan(perf.ann_turnover),
        cost_drag=_or_nan(perf.cost_drag),
        hit_rate=float(perf.hit_rate),
        avg_positions=_or_nan(perf.avg_positions),
        bench_cagr=float(bench_perf.cagr) if bench_perf else float("nan"),
        bench_sharpe=float(bench_perf.sharpe) if bench_perf else float("nan"),
        ruined="!! ruined" in result.diagnostics,
    )


def leg_from_dict(payload: dict, label: str = "") -> Leg:
    """Rebuild a `Leg` from stored JSON, ignoring fields a newer writer added."""
    known = {f for f in Leg.__dataclass_fields__}
    data = {k: v for k, v in (payload or {}).items() if k in known}
    if label:
        data["label"] = label
    return Leg(**data)


def slice_curve(series, start: str, end: str):
    """Date-string slice of a curve, tolerant of None and of empty bounds.

    Boolean-mask rather than `.loc[a:b]` because these curves are indexed by 'YYYY-MM-DD'
    STRINGS, not timestamps, and a label slice on a string index raises the moment a
    bound is not itself a session in the index. The dates are ISO, so string ordering is
    date ordering and the comparison is exact.
    """
    if series is None or not len(series):
        return series
    idx = np.asarray([str(d) for d in series.index])
    mask = np.ones(len(idx), dtype=bool)
    if start:
        mask &= idx >= start
    if end:
        mask &= idx <= end
    return series[mask]


def _slice_ledger(ledger: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Rebalance rows whose SIGNAL date falls in the window.

    The signal date rather than the execution date, so a rebalance decided on the last
    session of the research window is charged to the research window even though it
    filled on the first session of the forward one. That is the same boundary the engine
    uses to decide what a run may see (ADR-025).
    """
    if not len(ledger) or not (start or end):
        return ledger
    dates = ledger["date"].astype(str)
    keep = pd.Series(True, index=ledger.index)
    if start:
        keep &= dates >= start
    if end:
        keep &= dates <= end
    return ledger[keep]


def _or_nan(x) -> float:
    return float(x) if x is not None else float("nan")
