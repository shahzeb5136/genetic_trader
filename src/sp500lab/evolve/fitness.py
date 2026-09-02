"""What the genetic algorithm is actually maximising, and why it is not return.

A GA optimises the number you hand it, exactly and without mercy. So the objective
function is not a detail of the search - it IS the search, and every term below exists
because of a specific way an unconstrained one goes wrong.

Not return
----------
Handed raw return, a GA finds leverage in whatever form the constraints still allow.
Under this project's long-only, fully-invested mandate that form is concentration: it
drives `top_k` to its floor, holds ten names, and reports a magnificent CAGR next to a
70% drawdown. Sharpe closes that door.

The MONTHLY Sharpe, not the daily one
--------------------------------------
The headline Sharpe in every report is computed from the daily equity curve. That is the
right number to read and the wrong number to optimise. Daily returns of a
monthly-rebalanced portfolio are strongly autocorrelated within each holding period, so
4,861 daily observations carry roughly 176 independent ones, and a search that maximises
the daily figure is partly maximising an artefact of the sampling. The deflated Sharpe in
`registry.deflate()` uses the monthly statistics for exactly this reason (ADR-026), so
optimising the monthly Sharpe means the search and its own significance test are looking
at the same quantity.

Turnover, charged twice on purpose
-----------------------------------
The engine already charges costs, so a turnover penalty is double-counting - deliberately.
The cost model's weakest input is the estimated half-spread (ADR-020), and a strategy that
trades 500%/yr is making a large bet on that estimate being right. The penalty is a
statement about model risk, not about costs. It defaults to zero so that it is a choice
somebody made rather than a constant somebody inherited.

Parsimony
---------
Every additional feature an individual uses is another dimension the search could have
found a coincidence in. `complexity_penalty` charges per active feature, which biases the
population toward explanations rather than toward fits. A five-feature strategy that ties
a twelve-feature one is the better answer.

Fold consistency - the important one
-------------------------------------
`aggregate="mean_minus_std"` scores an individual on how it did in EVERY sub-period, minus
how much those sub-periods disagreed. A strategy that made all its money in 2009 and
nothing since has a fine full-sample Sharpe and a terrible fold-consistency score, and
that distinction is most of what separates a discovery from a coincidence here.

The folds are contiguous and separated by an embargo, never random K-fold. Financial data
is autocorrelated and a shuffled split leaks across the boundary; López de Prado's
*Advances in Financial Machine Learning* is the reference. The embargo defaults to one
month, which is one holding period under ADR-016 - the exact span over which a position
opened in one fold can still be held in the next.

**What this is not.** Folds here measure robustness, not out-of-sample performance. Every
fold is inside the research window and every individual is selected using all of them, so
a good fold score is evidence of consistency and not evidence of generalisation. The
out-of-sample test is the 2022 holdout, it is looked at once, and looking is recorded
(ADR-025).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest import metrics
from ..backtest.registry import monthly_stats

log = logging.getLogger(__name__)

#: Fitness for an individual that cannot be scored: a ruined portfolio, too short a
#: history, a degenerate curve. -inf rather than NaN so a sort never silently reorders
#: around a missing value, and so a failure can never win a tournament.
UNFIT = -np.inf

METRICS = ("sharpe_monthly", "sharpe", "excess_sharpe", "calmar", "information_ratio")
AGGREGATES = ("whole", "mean", "min", "mean_minus_std")


@dataclass(frozen=True)
class Folds:
    """Contiguous evaluation windows with an embargo between them.

    Not a train/test split: the individual is not fitted to anything, so there is nothing
    to hold out WITHIN the search. These are consistency windows, and the embargo exists
    because a position opened at the end of one fold is still held into the next.
    """

    bounds: tuple[tuple[str, str], ...]
    embargo_days: int = 31

    @classmethod
    def split(cls, start: str, end: str, n: int = 4, embargo_days: int = 31) -> "Folds":
        """`n` equal calendar spans between two dates, each shortened by the embargo."""
        if n < 1:
            raise ValueError("need at least one fold")
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        edges = pd.date_range(lo, hi, periods=n + 1)
        out = []
        for i in range(n):
            a = edges[i] + (pd.Timedelta(days=embargo_days) if i else pd.Timedelta(0))
            b = edges[i + 1]
            if b > a:
                out.append((str(a.date()), str(b.date())))
        return cls(tuple(out), embargo_days=embargo_days)

    def __len__(self) -> int:
        return len(self.bounds)

    def slices(self, equity: pd.Series) -> list[pd.Series]:
        idx = np.asarray([str(d) for d in equity.index])
        out = []
        for a, b in self.bounds:
            mask = (idx >= a) & (idx <= b)
            if mask.sum() >= 40:
                out.append(equity[mask])
        return out

    def describe(self) -> str:
        return " | ".join(f"{a}..{b}" for a, b in self.bounds)


@dataclass(frozen=True)
class Objective:
    """The number the search maximises, and everything subtracted from it.

    Deliberately a value object: it goes into the run's manifest verbatim, so the
    question "what was this search actually optimising" always has an answer.
    """

    metric: str = "sharpe_monthly"
    aggregate: str = "mean_minus_std"
    folds: Folds | None = None
    turnover_penalty: float = 0.0
    complexity_penalty: float = 0.0
    dispersion_weight: float = 0.5
    min_months: int = 36
    min_names: float = 5.0
    benchmark_sharpe: float = 0.0

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        if self.aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {AGGREGATES}")
        if self.aggregate != "whole" and self.folds is None:
            raise ValueError(f"aggregate={self.aggregate!r} needs folds")

    def describe(self) -> dict:
        d = {k: v for k, v in vars(self).items() if k != "folds"}
        d["folds"] = self.folds.describe() if self.folds else None
        return d

    # ------------------------------------------------------------- scoring

    def score(self, result, n_active: int = 0) -> tuple[float, dict]:
        """(fitness, detail). Detail is logged so a generation can be explained."""
        perf = result.performance
        detail: dict = {"n_active": int(n_active)}

        monthly = monthly_stats(result.equity)
        detail["n_months"] = int(monthly["n_months"])
        if monthly["n_months"] < self.min_months:
            detail["reject"] = (f"only {monthly['n_months']} months, "
                                f"need {self.min_months}")
            return UNFIT, detail
        if "!! ruined" in result.diagnostics:
            detail["reject"] = "portfolio reached zero NAV"
            return UNFIT, detail
        if (perf.avg_positions or 0) < self.min_names:
            detail["reject"] = f"held {perf.avg_positions:.1f} names on average"
            return UNFIT, detail

        whole = self._metric(result, monthly)
        detail["whole"] = _round(whole)
        if not np.isfinite(whole):
            detail["reject"] = f"{self.metric} is not finite"
            return UNFIT, detail

        base = whole
        if self.aggregate != "whole":
            per_fold = self._per_fold(result)
            detail["folds"] = [_round(x) for x in per_fold]
            if len(per_fold) < 2:
                detail["reject"] = "fewer than two usable folds"
                return UNFIT, detail
            arr = np.asarray(per_fold, dtype=np.float64)
            if not np.isfinite(arr).all():
                detail["reject"] = "a fold produced no usable statistics"
                return UNFIT, detail
            if self.aggregate == "mean":
                base = float(arr.mean())
            elif self.aggregate == "min":
                base = float(arr.min())
            else:
                base = float(arr.mean() - self.dispersion_weight * arr.std(ddof=1))
            detail["fold_mean"] = _round(float(arr.mean()))
            detail["fold_std"] = _round(float(arr.std(ddof=1)))

        turnover = float(perf.ann_turnover or 0.0)
        penalty = (self.turnover_penalty * turnover
                   + self.complexity_penalty * float(n_active))
        detail["base"] = _round(base)
        detail["penalty"] = _round(penalty)
        detail["turnover"] = _round(turnover)

        fitness = base - penalty
        detail["fitness"] = _round(fitness)
        return float(fitness), detail

    # ------------------------------------------------------------- internals

    def _metric(self, result, monthly: dict) -> float:
        perf = result.performance
        if self.metric == "sharpe_monthly":
            return float(monthly["sharpe"])
        if self.metric == "sharpe":
            return float(perf.sharpe)
        if self.metric == "excess_sharpe":
            return float(monthly["sharpe"]) - float(self.benchmark_sharpe)
        if self.metric == "calmar":
            return float(perf.calmar) if np.isfinite(perf.calmar) else UNFIT
        if self.metric == "information_ratio":
            ir = perf.information_ratio
            return float(ir) if ir is not None and np.isfinite(ir) else UNFIT
        raise ValueError(self.metric)

    def _per_fold(self, result) -> list[float]:
        """The metric recomputed on each fold, from the curve the run already produced.

        One backtest per individual, not one per fold. The equity curve carries
        everything a fold statistic needs, so slicing it is free where re-running the
        engine four times would quadruple the cost of the entire search.
        """
        out = []
        for piece in (self.folds.slices(result.equity) if self.folds else []):
            rebased = piece / float(piece.iloc[0])
            stats = monthly_stats(rebased)
            if stats["n_months"] < 6:
                continue
            if self.metric in ("sharpe_monthly", "excess_sharpe"):
                value = float(stats["sharpe"])
                if self.metric == "excess_sharpe":
                    value -= float(self.benchmark_sharpe)
            else:
                try:
                    value = getattr(metrics.compute(rebased),
                                    "sharpe" if self.metric == "sharpe" else self.metric)
                except Exception:                                 # noqa: BLE001
                    continue
            out.append(float(value))
        return out


@dataclass
class Evaluation:
    """One individual, scored. What the population is made of."""

    vector: np.ndarray
    fingerprint: str
    fitness: float
    detail: dict = field(default_factory=dict)
    run_id: str | None = None
    cagr: float = float("nan")
    sharpe: float = float("nan")
    max_drawdown: float = float("nan")
    turnover: float = float("nan")
    n_active: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return np.isfinite(self.fitness)

    @property
    def short_id(self) -> str:
        """Eight characters standing in for the genome, for tables people read.

        The fingerprint is the whole vector - the right thing to key a cache on and the
        wrong thing to print nineteen times in a leaderboard. `evolve best` resolves a
        short id back to the individual.
        """
        import hashlib
        return hashlib.blake2b(self.fingerprint.encode(), digest_size=4).hexdigest()

    def as_row(self) -> dict:
        return {"id": self.short_id, "fitness": self.fitness,
                "cagr": self.cagr, "sharpe": self.sharpe,
                "max_drawdown": self.max_drawdown, "turnover": self.turnover,
                "n_active": self.n_active, "run_id": self.run_id,
                "error": self.error}


def _round(x) -> float:
    try:
        return round(float(x), 5)
    except (TypeError, ValueError):
        return float("nan")


def benchmark_monthly_sharpe(start: str, end: str, ticker: str = "SPY") -> float:
    """The index's own monthly Sharpe over a window - the bar `excess_sharpe` subtracts.

    Computed once per search rather than per individual. Returns 0.0 if the benchmark is
    unavailable, which turns `excess_sharpe` back into `sharpe_monthly` rather than
    silently scoring everything as -inf.
    """
    from ..backtest.benchmark import benchmark_total_return
    try:
        s = benchmark_total_return(ticker)
    except Exception as exc:                                      # noqa: BLE001
        log.warning("benchmark %s unavailable, excess is measured against zero: %s",
                    ticker, exc)
        return 0.0
    idx = np.asarray([str(d) for d in s.index])
    piece = s[(idx >= start) & (idx <= end)].dropna()
    if len(piece) < 40:
        return 0.0
    return float(monthly_stats(piece)["sharpe"])
