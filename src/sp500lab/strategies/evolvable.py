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
from ..backtest.portfolio import Construction, build_weights
from ..backtest.strategy import FeatureStrategy, SignalStrategy, register
from .genome import (DEAD_ZONE, PRESET_MIN_DATE, PRESETS, REGIME_FEATURES,
                     alpha_genome, describe_genome)
from .signals import rank_pct


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


# --------------------------------------------------------------------------
# EvolvedAlpha - the genome the genetic algorithm actually searches
# --------------------------------------------------------------------------

@register("evolved_alpha")
class EvolvedAlpha(FeatureStrategy):
    """A weighted blend of ranked FEATURES, entirely determined by its genome.

    The successor to `EvolvedBlend` above, and the difference is where the numbers come
    from. `EvolvedBlend` recomputes four indicators inside every `score()` call, which is
    fine for four and fatal for twenty: a 10,000-individual search would spend most of
    its time recomputing the same rolling regressions. This reads the shared feature
    panel, so a fitness evaluation is a gather and a dot product.

        score = sum_k  w_k * percentile_rank(feature_k)

    Percentile rank rather than z-score, because cross-sectional fundamental data has
    tails that are not merely fat but wrong (see signals.py). Weights inside the dead
    zone contribute exactly nothing, so "how many features does this use" has an answer
    and parsimony pressure has something to grip.

    The regime gate is the only non-linearity: when it is on and the market is below its
    200-day average or realised volatility is far above its own year, the strategy holds
    the low-volatility end of its own score and invests only `defensive_gross` of the
    account. Under a long-only mandate, cash is the only defensive asset there is.

    Instantiate from a vector:

        g = alpha_genome("price")
        EvolvedAlpha(preset="price", **g.decode(vector))
    """

    name = "evolved_alpha"

    #: Suffix on a pre-ranked feature column. See features/ranked.py.
    RANK_SUFFIX = "__rank"

    def __init__(self, preset: str = "price", pre_ranked: bool = False, **params):
        genome = alpha_genome(preset)
        defaults = genome.decode(np.zeros(len(genome)))
        merged = {**defaults, **{k: v for k, v in params.items()
                                 if k in set(genome.names)}}
        decoded = genome.decode(genome.encode(merged))
        super().__init__(preset=preset, **decoded)

        self._genome = genome
        self._features = tuple(PRESETS[preset])
        self._weights = np.array(
            [self.params[f"w_{f}"] for f in self._features], dtype=np.float64)
        # The dead zone is applied ONCE, here, rather than inside score(). Otherwise
        # every rebalance re-derives the same mask 232 times per fitness evaluation.
        self._weights[np.abs(self._weights) < DEAD_ZONE] = 0.0
        self._used = np.flatnonzero(self._weights != 0.0)

        # Pre-ranked columns carry a different NAME, not just different values, so a
        # strategy handed the wrong panel fails at the first rebalance instead of
        # quietly summing raw book-to-market ratios as if they were ranks.
        self.pre_ranked = bool(pre_ranked)
        self._columns = tuple(f"{f}{self.RANK_SUFFIX}" if self.pre_ranked else f
                              for f in self._features)
        self._defensive_column = ("vol_126d" + self.RANK_SUFFIX if self.pre_ranked
                                  else "vol_126d")
        needed = list(self._columns) + list(REGIME_FEATURES) + [self._defensive_column]
        # Deduplicated while preserving order: the defensive sleeve ranks volatility,
        # which is usually already one of the scored columns.
        self.requires_features = tuple(dict.fromkeys(needed))
        # Feature name -> column, resolved once on the first scoring call. `ctx.feature`
        # does a linear scan of 75 names, and a fitness evaluation asks for twelve of
        # them at each of 232 rebalances.
        self._cols: dict[str, int] | None = None
        self.min_date = PRESET_MIN_DATE[preset]
        self.construction = Construction(
            top_k=int(self.params["top_k"]),
            weighting=str(self.params["weighting"]),
            max_weight=float(self.params["max_weight"]),
            min_names=10)

    # ------------------------------------------------------------- behaviour

    @property
    def active_features(self) -> list[str]:
        return [self._features[i] for i in self._used]

    def _col(self, ctx: Context, name: str) -> np.ndarray:
        if self._cols is None:
            self._cols = {n: i for i, n in enumerate(ctx.feature_names)}
        return ctx.features[:, self._cols[name]]

    def _macro(self, ctx: Context, name: str) -> float:
        """A macro feature's single value for this date.

        Macro columns are one number broadcast across every security, so the first
        finite entry IS the value. `np.nanmedian` would give the same answer, scan the
        whole cross-section to do it, and emit an all-NaN warning 232 times per run in
        the early years where the series does not exist yet.
        """
        col = self._col(ctx, name)
        finite = col[np.isfinite(col)]
        return float(finite[0]) if finite.size else float("nan")

    def is_defensive(self, ctx: Context) -> bool:
        if self.params["use_regime"] != "on":
            return False
        trend = self._macro(ctx, "mkt_trend_200d")
        ratio = self._macro(ctx, "mkt_vol_ratio")
        return ((np.isfinite(trend) and trend < 0.0)
                or (np.isfinite(ratio) and ratio > float(self.params["vol_trigger"])))

    def score(self, ctx: Context) -> np.ndarray:
        elig = ctx.tradable
        n = ctx.close.shape[1]
        if not len(self._used) or elig.sum() < 10:
            # An individual with every weight inside the dead zone has no opinion about
            # anything. It scores NaN and holds cash, rather than falling through to the
            # tie-break and collecting the survivors - which is exactly the failure that
            # made a zero-signal strategy post 17.65%/yr in ADR-024.
            return np.full(n, np.nan)

        if self.is_defensive(ctx):
            # Defensive months rank on low volatility rather than on the evolved score.
            # The score is what the individual believes; this is what it does when it
            # does not want to act on that belief.
            vol = self._col(ctx, self._defensive_column)
            # Ranks are a monotone transform, so ranking a rank is the identity - the
            # negation still has to happen, because low volatility is the good end.
            return rank_pct(-vol, elig) if not self.pre_ranked else np.where(
                elig & np.isfinite(vol), 1.0 - vol, np.nan)

        total = np.zeros(n, dtype=np.float64)
        present = np.zeros(n, dtype=np.float64)
        for i in self._used:
            w = self._weights[i]
            col = self._col(ctx, self._columns[i])
            r = col if self.pre_ranked else rank_pct(col, elig)
            ok = np.isfinite(r) & elig
            total[ok] += w * r[ok]
            present[ok] += abs(w)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = total / present
        return np.where((present > 0) & np.isfinite(out), out, np.nan)

    def target_weights(self, ctx: Context) -> np.ndarray:
        from dataclasses import replace
        c = self.construction
        if self.is_defensive(ctx):
            c = replace(c, gross=float(self.params["defensive_gross"]))
        s = np.asarray(self.score(ctx), dtype=np.float64)
        vol = (self.vol_for_weighting(ctx) if c.weighting == "inverse_vol" else None)
        return build_weights(s, self.eligible(ctx), c, tiebreak=ctx.tiebreak, vol=vol)

    def describe(self) -> dict:
        d = super().describe()
        d["preset"] = self.params["preset"]
        # Deliberately NOT in `params`: pre-ranking is an execution detail that produces
        # identical weights either way, and putting it in the fingerprint would make the
        # same hypothesis count as two trials and over-deflate the winner.
        d["pre_ranked"] = self.pre_ranked
        d["active_features"] = self.active_features
        d["n_active"] = len(self._used)
        return d

    def explain(self) -> str:
        """The individual in sentences. See genome.describe_genome."""
        return describe_genome(self._genome, self._genome.encode(self.params))


def from_vector(vector, preset: str = "price",
                pre_ranked: bool = False) -> EvolvedAlpha:
    """The GA's constructor: a float vector in, a scorable strategy out."""
    genome = alpha_genome(preset)
    return EvolvedAlpha(preset=preset, pre_ranked=pre_ranked,
                        **genome.decode(vector))
