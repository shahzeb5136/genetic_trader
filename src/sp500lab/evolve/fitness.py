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

Costs are inside the fitness, and then charged again
-----------------------------------------------------
The fitness is the NET equity curve the engine produced under the search's cost setting,
which is `pessimistic` by default: commission plus twice the estimated half-spread
(ADR-049). A rule that only works if the spread estimator is kind never scores well. On
top of that, `turnover_penalty` charges turnover a second time, deliberately: the cost
model's weakest input is that estimate (ADR-020), and a strategy that trades 500%/yr is
making a large bet on it being right. The penalty is a statement about model risk, not
about costs.

Parsimony, charged per rule
---------------------------
Every additional feature an individual uses is another dimension the search could have
found a coincidence in; every family it backs is another rule; the regime gate is two
tuned parameters and a switch. `complexity_penalty` charges per feature,
`family_penalty` per live family and `gate_penalty` once for switching the gate on, so
the population is biased toward explanations rather than fits. A one-family strategy
that ties a three-family one is the better answer, and the objective says so.

Robustness, not the whole window - the important one
-----------------------------------------------------
An individual is scored on SEVERAL sub-periods of the research window, not on the whole
of it, and the number it is given is a low quantile of those sub-period scores - the
25th percentile by default, or the minimum. A strategy that made all its money in 2009
and nothing since has a fine full-sample Sharpe and a terrible score here, and that
distinction is most of what separates a discovery from a coincidence. The point of
scoring the worst quarter rather than the mean is that a rule which only works in one
stretch is killed DURING evolution, rather than surviving to the test set on the
strength of that stretch.

Two ways to cut the window. `Folds.random` draws `n` sub-periods of three to five years
at random positions - drawn ONCE per search from a fixed seed, so every individual and
every seed of the search is scored on the same sub-periods and their fitnesses are
comparable. `Folds.split` is the older scheme (ADR-032): contiguous equal spans separated
by an embargo of one holding period, never a shuffled K-fold, because financial data is
autocorrelated and a shuffled split leaks across the boundary. Both slice the single
equity curve the individual already produced, so neither costs a second backtest.

**What this is not.** Sub-periods here measure robustness, not out-of-sample performance.
Every one of them is inside the research window and every individual is selected using
all of them, so a good score is evidence of consistency and not evidence of
generalisation. The out-of-sample test is the 2022 holdout, it is looked at once, and
looking is recorded (ADR-025).
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
AGGREGATES = ("whole", "mean", "min", "mean_minus_std", "quantile")
FOLD_SCHEMES = ("contiguous", "random")


@dataclass(frozen=True)
class Folds:
    """Evaluation windows inside the research period.

    Not a train/test split: the individual is not fitted to anything, so there is nothing
    to hold out WITHIN the search. These are consistency windows. `split` makes them
    contiguous with an embargo between them; `random` draws overlapping sub-periods at
    random positions, which is what the default objective scores.
    """

    bounds: tuple[tuple[str, str], ...]
    embargo_days: int = 31
    scheme: str = "contiguous"

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
        return cls(tuple(out), embargo_days=embargo_days, scheme="contiguous")

    @classmethod
    def random(cls, start: str, end: str, n: int = 12, min_years: float = 3.0,
               max_years: float = 5.0, seed: int = 0) -> "Folds":
        """`n` sub-periods of `min_years`..`max_years`, at random positions in the window.

        Drawn from a private generator seeded by `seed`, never from the search's own
        random state: the sub-periods are part of the OBJECTIVE, so two seeds of one
        search must draw the same ones or their fitnesses cannot be pooled. They may
        overlap, and usually do - twelve blocks of three to five years inside fifteen
        cover every stretch of the window several times, which is the point: a rule has
        to work in most of them, not in the one that happens to hold 2009.
        """
        if n < 1:
            raise ValueError("need at least one fold")
        if min_years <= 0 or max_years < min_years:
            raise ValueError("need 0 < min_years <= max_years")
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        span = (hi - lo).days
        min_days = int(round(min_years * 365.25))
        max_days = min(int(round(max_years * 365.25)), span)
        if span < min_days:
            raise ValueError(f"the window {start}..{end} is shorter than one "
                             f"{min_years}-year sub-period")
        rng = np.random.default_rng(int(seed))
        out = []
        for _ in range(n):
            length = int(rng.integers(min_days, max_days + 1))
            offset = int(rng.integers(0, span - length + 1))
            a = lo + pd.Timedelta(days=offset)
            b = a + pd.Timedelta(days=length)
            out.append((str(a.date()), str(b.date())))
        out.sort()
        return cls(tuple(out), embargo_days=0, scheme="random")

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
    family_penalty: float = 0.0
    gate_penalty: float = 0.0
    dispersion_weight: float = 0.5
    quantile: float = 0.25
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
        if not 0.0 <= self.quantile <= 1.0:
            raise ValueError("quantile must be in [0, 1]")

    def describe(self) -> dict:
        d = {k: v for k, v in vars(self).items() if k != "folds"}
        d["folds"] = self.folds.describe() if self.folds else None
        d["fold_scheme"] = self.folds.scheme if self.folds else None
        d["n_folds"] = len(self.folds) if self.folds else 0
        return d

    # ------------------------------------------------------------- scoring

    def score(self, result, n_active: int = 0, n_families: int = 0,
              gate_on: bool = False) -> tuple[float, dict]:
        """(fitness, detail). Detail is logged so a generation can be explained.

        `n_active`, `n_families` and `gate_on` are what the complexity terms charge:
        features read, families backed, and whether the regime gate is switched on.
        """
        perf = result.performance
        detail: dict = {"n_active": int(n_active), "n_families": int(n_families),
                        "gate_on": bool(gate_on)}

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
            elif self.aggregate == "quantile":
                base = float(np.quantile(arr, self.quantile))
            else:
                base = float(arr.mean() - self.dispersion_weight * arr.std(ddof=1))
            detail["fold_mean"] = _round(float(arr.mean()))
            detail["fold_std"] = _round(float(arr.std(ddof=1)))
            detail["fold_min"] = _round(float(arr.min()))
            detail["fold_quantile"] = _round(float(np.quantile(arr, self.quantile)))

        turnover = float(perf.ann_turnover or 0.0)
        penalty = (self.turnover_penalty * turnover
                   + self.complexity_penalty * float(n_active)
                   + self.family_penalty * float(n_families)
                   + (self.gate_penalty if gate_on else 0.0))
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
        engine twelve times would multiply the cost of the entire search.
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
    n_families: int = 0
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
                "n_active": self.n_active, "n_families": self.n_families,
                "run_id": self.run_id, "error": self.error}


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
