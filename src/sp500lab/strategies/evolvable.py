"""Strategies whose entire behaviour is a parameter vector — the GA's substrate.

Why this exists before the genetic algorithm does
--------------------------------------------------
A GA needs three things: a genome encoding, a decoder that turns a genome into
behaviour, and a fitness function. The fitness function is `run_backtest`. This module
is the other two, so that when the GA arrives it only has to implement selection,
crossover and mutation — the parts that are actually about evolution.

Building it now also proves the claim the engine is built on: that a genome-driven
strategy and a hand-written rule are the same kind of object to the engine. `EvolvedBlend`
implements exactly the same `score(ctx)` interface as `momentum_12_1`, pays the same
costs, and is scored by the same metrics. The scoreboard cannot tell them apart.

The search space is constrained on purpose
-------------------------------------------
An unconstrained GA over indicator combinations will find something that works
beautifully in-sample every single time — that is what a maximum over thousands of draws
does, signal or no signal. `GENOME` therefore bounds every gene to a range with an
economic reading, and the blend is a weighted sum of z-scored signals rather than an
arbitrary expression tree. That is a deliberate trade: less expressive, far less prone to
discovering noise.

Read docs/HANDOFF.md section 6 before running a search. In particular: log every
individual evaluated, not just the winners. `metrics.deflate_result()` needs the trial
count and the spread of trial Sharpes, and without them the winner's Sharpe is not
conservative or optimistic — it is meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backtest.context import Context
from ..backtest.portfolio import Construction
from ..backtest.strategy import SignalStrategy, register


@dataclass(frozen=True)
class Gene:
    """One tunable parameter, with the bounds a mutation must respect.

    `integer` matters: a lookback of 126.4 sessions is not a thing, and rounding at use
    time rather than at mutation time makes two genomes that differ only below the
    rounding threshold score identically — which wastes a generation.
    """

    name: str
    low: float
    high: float
    integer: bool = False

    def clip(self, value: float) -> float:
        v = float(np.clip(value, self.low, self.high))
        return float(round(v)) if self.integer else v

    def sample(self, rng: np.random.Generator) -> float:
        return self.clip(rng.uniform(self.low, self.high))


#: The search space for EvolvedBlend. Every bound has an economic reading rather than
#: being a round number: lookbacks span one month to two years, the skip covers the
#: short-term reversal window, and the signal weights are bounded so no single term can
#: dominate by scale alone.
GENOME: tuple[Gene, ...] = (
    Gene("mom_lookback", 21, 504, integer=True),
    Gene("mom_skip", 0, 42, integer=True),
    Gene("mom_weight", -1.0, 1.0),
    Gene("rev_lookback", 5, 63, integer=True),
    Gene("rev_weight", -1.0, 1.0),
    Gene("vol_lookback", 21, 252, integer=True),
    Gene("vol_weight", -1.0, 1.0),
    Gene("trend_lookback", 21, 252, integer=True),
    Gene("trend_weight", -1.0, 1.0),
    Gene("top_k", 10, 100, integer=True),
    Gene("max_weight", 0.02, 0.20),
)


def random_genome(rng: np.random.Generator) -> dict:
    """A uniformly sampled individual. The GA's initial population comes from here."""
    return {g.name: g.sample(rng) for g in GENOME}


def decode(vector: np.ndarray | dict) -> dict:
    """Genome (array or dict) -> clipped parameter dict.

    Accepts an array so a GA can carry a plain float vector and never think about
    names, and clips on the way out so a mutation that overshoots a bound produces a
    valid individual instead of an exception mid-population.
    """
    if isinstance(vector, dict):
        return {g.name: g.clip(vector[g.name]) for g in GENOME}
    v = np.asarray(vector, dtype=np.float64).ravel()
    if len(v) != len(GENOME):
        raise ValueError(f"genome has {len(v)} genes, expected {len(GENOME)}")
    return {g.name: g.clip(v[i]) for i, g in enumerate(GENOME)}


def encode(params: dict) -> np.ndarray:
    """Parameter dict -> float vector, in GENOME order."""
    return np.array([params[g.name] for g in GENOME], dtype=np.float64)


def _zscore(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score over the eligible names only.

    Standardising within the date is what makes the signals addable: a momentum figure
    and a volatility figure have no common scale until they are both expressed in
    cross-sectional standard deviations. Standardising over ALL names rather than the
    eligible ones would let untradable names move the mean.
    """
    out = np.full_like(x, np.nan)
    vals = x[mask]
    good = np.isfinite(vals)
    if good.sum() < 3:
        return out
    mu = float(vals[good].mean())
    sd = float(vals[good].std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return out
    out[mask] = (x[mask] - mu) / sd
    return out


def _trend(close: np.ndarray, at: int, lookback: int) -> np.ndarray:
    """(S,) price relative to its own trailing mean, NaN where there is no history.

    np.nanmean over an all-NaN column warns and returns NaN; masking the empty columns
    out first keeps the result identical and the log clean. A warning that fires 232
    times per backtest trains people to ignore warnings.
    """
    win = close[max(0, at - lookback + 1):at + 1]
    n = np.isfinite(win).sum(axis=0)
    out = np.full(close.shape[1], np.nan)
    ok = n > 0
    if ok.any():
        with np.errstate(divide="ignore", invalid="ignore"):
            out[ok] = close[at, ok] / np.nanmean(win[:, ok], axis=0) - 1.0
    return out


@register("evolved_blend")
class EvolvedBlend(SignalStrategy):
    """A weighted blend of four z-scored signals, entirely determined by its genome.

    score = w_mom * z(momentum) + w_rev * z(reversal)
          + w_vol * z(-volatility) + w_trend * z(price / moving average - 1)

    Every weight may be negative, so the GA can discover "buy losers" as readily as
    "buy winners" — which matters, because those two cannot both be picking up
    structure, and a search that lands on either needs the deflated Sharpe applied
    before anyone believes it.

    Instantiate with a genome:

        EvolvedBlend(**decode(vector))
        run_backtest(EvolvedBlend(**decode(vector)), costs="realistic")
    """

    name = "evolved_blend"

    #: A reasonable starting individual, NOT a tuned one - roughly 12-1 momentum with a
    #: low-volatility tilt. Deliberately not the midpoint of every gene: the midpoint
    #: sets all four signal weights to zero, which makes every score identical and
    #: leaves the whole portfolio to the tie-break. That is how the survivorship bug in
    #: ADR-024 was found, and a default that hides behind a tie-break is a bad default.
    DEFAULT_GENOME = {
        "mom_lookback": 252, "mom_skip": 21, "mom_weight": 0.6,
        "rev_lookback": 21, "rev_weight": 0.1,
        "vol_lookback": 126, "vol_weight": 0.3,
        "trend_lookback": 200, "trend_weight": 0.0,
        "top_k": 50, "max_weight": 0.05,
    }

    def __init__(self, **params):
        p = decode({**self.DEFAULT_GENOME, **params})
        super().__init__(**p)
        self.warmup = int(max(p["mom_lookback"] + p["mom_skip"], p["vol_lookback"],
                              p["trend_lookback"], p["rev_lookback"]) + 5)
        self.construction = Construction(
            top_k=int(p["top_k"]), weighting="equal",
            max_weight=float(p["max_weight"]), min_names=5)

    def score(self, ctx: Context) -> np.ndarray:
        p = self.params
        elig = ctx.tradable
        if elig.sum() < 5:
            return np.full(ctx.close.shape[1], np.nan)

        mom = ctx.trailing_return(int(p["mom_lookback"]), skip=int(p["mom_skip"]))
        rev = -ctx.trailing_return(int(p["rev_lookback"]), skip=0)
        vol = -self.vol_for_weighting(ctx, lookback=int(p["vol_lookback"]))

        trend = _trend(ctx.close, len(ctx.close) - 1, int(p["trend_lookback"]))

        blend = (p["mom_weight"] * _zscore(mom, elig)
                 + p["rev_weight"] * _zscore(rev, elig)
                 + p["vol_weight"] * _zscore(vol, elig)
                 + p["trend_weight"] * _zscore(trend, elig))
        # A name missing any one component would drop out of the sum entirely. Treating
        # a missing component as "average" keeps it in the running on the strength of
        # the components it does have, rather than silently shrinking the universe.
        return np.where(np.isfinite(blend), blend, np.nan)


def fitness(result, turnover_penalty: float = 0.0) -> float:
    """Turn a BacktestResult into one number for the GA to maximise.

    Sharpe rather than return, because a GA handed raw return will find leverage in
    whatever form the constraints still allow — here, concentration. The turnover
    penalty is exposed rather than baked in because costs are already charged by the
    engine; use it only when you want to discourage trading *beyond* what it costs.

    Ruin and empty runs return -inf rather than nan, so a sort never silently reorders
    around a missing value.
    """
    p = result.performance
    if not np.isfinite(p.sharpe) or p.n_periods < 24:
        return -np.inf
    score = float(p.sharpe)
    if turnover_penalty and p.ann_turnover:
        score -= turnover_penalty * float(p.ann_turnover)
    return score
