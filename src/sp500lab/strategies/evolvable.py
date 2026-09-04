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

Four strategy classes, in the order they arrived
-------------------------------------------------
    EvolvedBlend      four indicators recomputed inside score(); the first substrate
    EvolvedAlpha      weighted ranked FEATURES from the shared panel; presets price/full/night
    EvolvedFamilies   weighted prior-signed FAMILY composites, capped (ADR-048)
    EvolvedEnsemble   the average signal of the N best individuals of a search (ADR-050)

`from_vector()` turns a vector into whichever of the middle two its preset calls for, and
`EvolvedEnsemble` is built from a list of those. The engine cannot tell any of them from a
hand-written strategy, which is the point.

Read docs/HANDOFF.md section 6 before running a search. In particular: log every
individual evaluated, not just the winners. `metrics.deflate_result()` needs the trial
count and the spread of trial Sharpes, and without them the winner's Sharpe is not
conservative or optimistic — it is meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..backtest.context import Context
from ..backtest.portfolio import Construction, build_weights
from ..backtest.strategy import FeatureStrategy, SignalStrategy, register
from .genome import (DEAD_ZONE, FAMILY_BY_NAME, PRESET_MIN_DATE, PRESETS,
                     REGIME_FEATURES, alpha_genome, describe_genome, preset_families,
                     preset_features, preset_kind)
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
        if preset not in PRESETS:
            raise KeyError(f"{type(self).__name__} takes a feature preset, one of "
                           f"{tuple(PRESETS)}; got {preset!r}")
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
        self._wire(preset, pre_ranked, self._features)

    def _wire(self, preset: str, pre_ranked: bool, features: tuple[str, ...]) -> None:
        """Everything the engine needs to know, shared by the feature and family forms.

        Pre-ranked columns carry a different NAME, not just different values, so a
        strategy handed the wrong panel fails at the first rebalance instead of
        quietly summing raw book-to-market ratios as if they were ranks.
        """
        self.pre_ranked = bool(pre_ranked)
        self._columns = tuple(f"{f}{self.RANK_SUFFIX}" if self.pre_ranked else f
                              for f in features)
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

    @property
    def has_opinion(self) -> bool:
        """False when every signal weight is inside the dead zone."""
        return bool(len(self._used))

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

    def defensive_score(self, ctx: Context) -> np.ndarray:
        """What the individual holds when it does not want to act on its belief.

        Defensive months rank on low volatility rather than on the evolved score. Ranks
        are a monotone transform, so ranking a rank is the identity - the negation still
        has to happen, because low volatility is the good end.
        """
        elig = ctx.tradable
        vol = self._col(ctx, self._defensive_column)
        return rank_pct(-vol, elig) if not self.pre_ranked else np.where(
            elig & np.isfinite(vol), 1.0 - vol, np.nan)

    def alpha(self, ctx: Context) -> np.ndarray:
        """The individual's belief, gate ignored: the weighted sum of ranks.

        Public because an ensemble averages beliefs, not portfolios (ADR-050): it wants
        every member's opinion on every date, including the dates a member would itself
        have stepped aside on.
        """
        elig = ctx.tradable
        n = ctx.close.shape[1]
        if not self.has_opinion or elig.sum() < 10:
            # An individual with every weight inside the dead zone has no opinion about
            # anything. It scores NaN and holds cash, rather than falling through to the
            # tie-break and collecting the survivors - which is exactly the failure that
            # made a zero-signal strategy post 17.65%/yr in ADR-024.
            return np.full(n, np.nan)

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

    def score(self, ctx: Context) -> np.ndarray:
        elig = ctx.tradable
        if not self.has_opinion or elig.sum() < 10:
            return np.full(ctx.close.shape[1], np.nan)
        if self.is_defensive(ctx):
            # The score is what the individual believes; this is what it does when it
            # does not want to act on that belief.
            return self.defensive_score(ctx)
        return self.alpha(ctx)

    def target_weights(self, ctx: Context) -> np.ndarray:
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
        d["n_active"] = len(self.active_features)
        return d

    def explain(self) -> str:
        """The individual in sentences. See genome.describe_genome."""
        return describe_genome(self._genome, self._genome.encode(self.params))


# --------------------------------------------------------------------------
# EvolvedFamilies - the capped, prior-signed search space (ADR-048)
# --------------------------------------------------------------------------

@register("evolved_families")
class EvolvedFamilies(EvolvedAlpha):
    """A weighted blend of prior-signed FAMILY composites, at most a few at a time.

    Same engine contract as `EvolvedAlpha`, a much smaller space. Each family is a fixed
    story - momentum, low risk, value, quality, ... - told by a handful of features whose
    direction the literature already settled, and the composite is the plain mean of
    those prior-signed percentile ranks over whichever members a name has a value for.
    The genome carries one NON-NEGATIVE weight per family and the preset caps how many
    may be live, so an individual reads as "back these three stories, in these
    proportions". It cannot rank value backwards and it cannot back all nine at once.

        family_k(name) = mean over members of  rank       (prior says high is good)
                                            or 1 - rank   (prior says low is good)
        score(name)    = sum_k w_k * family_k(name) / sum_k w_k   over present families

    The `1 - rank` form rather than a negative weight, deliberately: every term stays in
    [0, 1], so a name missing a member is scored as average on it rather than being
    quietly rewarded for having no value on a feature whose low end is good.

    Instantiate from a vector:

        g = alpha_genome("families")
        EvolvedFamilies(preset="families", **g.decode(vector))
    """

    name = "evolved_families"

    def __init__(self, preset: str = "families", pre_ranked: bool = False, **params):
        if preset_kind(preset) != "families":
            raise KeyError(f"{type(self).__name__} takes a family preset; got {preset!r}")
        genome = alpha_genome(preset)
        defaults = genome.decode(np.zeros(len(genome)))
        merged = {**defaults, **{k: v for k, v in params.items()
                                 if k in set(genome.names)}}
        decoded = genome.decode(genome.encode(merged))
        FeatureStrategy.__init__(self, preset=preset, **decoded)

        self._genome = genome
        self._families = preset_families(preset)
        self._family_weights = np.array(
            [float(self.params[f"f_{fam.name}"]) for fam in self._families],
            dtype=np.float64)
        # decode() already zeroed the dead zone and everything past the cap.
        self._live = [i for i, w in enumerate(self._family_weights) if w > 0.0]
        self._features = preset_features(preset)
        self._feature_index = {f: i for i, f in enumerate(self._features)}
        self._used = np.array(sorted({self._feature_index[f] for i in self._live
                                      for f in self._families[i].features}), dtype=int)
        self._wire(preset, pre_ranked, self._features)

    @property
    def active_families(self) -> list[str]:
        return [self._families[i].name for i in self._live]

    @property
    def family_weights(self) -> list[tuple[str, float]]:
        out = [(self._families[i].name, float(self._family_weights[i]))
               for i in self._live]
        out.sort(key=lambda kv: -kv[1])
        return out

    @property
    def has_opinion(self) -> bool:
        return bool(self._live)

    def alpha(self, ctx: Context) -> np.ndarray:
        elig = ctx.tradable
        n = ctx.close.shape[1]
        if not self.has_opinion or elig.sum() < 10:
            return np.full(n, np.nan)

        total = np.zeros(n, dtype=np.float64)
        present = np.zeros(n, dtype=np.float64)
        for i in self._live:
            w = self._family_weights[i]
            comp = np.zeros(n, dtype=np.float64)
            cnt = np.zeros(n, dtype=np.float64)
            for feature, sign in self._families[i].members:
                col = self._col(ctx, self._columns[self._feature_index[feature]])
                r = col if self.pre_ranked else rank_pct(col, elig)
                ok = np.isfinite(r) & elig
                comp[ok] += r[ok] if sign > 0 else 1.0 - r[ok]
                cnt[ok] += 1.0
            has = cnt > 0
            total[has] += w * comp[has] / cnt[has]
            present[has] += w
        with np.errstate(divide="ignore", invalid="ignore"):
            out = total / present
        return np.where((present > 0) & np.isfinite(out), out, np.nan)

    def describe(self) -> dict:
        d = super().describe()
        d["active_families"] = self.active_families
        d["n_families"] = len(self._live)
        return d


def from_vector(vector, preset: str = "price",
                pre_ranked: bool = False) -> EvolvedAlpha:
    """The GA's constructor: a float vector in, a scorable strategy out.

    A feature preset becomes an `EvolvedAlpha`, a family preset an `EvolvedFamilies`.
    Both are `EvolvedAlpha`s to the engine and to the ensemble.
    """
    genome = alpha_genome(preset)
    cls = EvolvedFamilies if preset_kind(preset) == "families" else EvolvedAlpha
    return cls(preset=preset, pre_ranked=pre_ranked, **genome.decode(vector))


# --------------------------------------------------------------------------
# EvolvedEnsemble - the average of many survivors, not the champion (ADR-050)
# --------------------------------------------------------------------------

class EvolvedEnsemble(SignalStrategy):
    """The average SIGNAL of the N best individuals a search produced.

    The single best individual of a search is the most luck-contaminated object in the
    whole population: it is the maximum over thousands of draws, and three searches in a
    row have shown what that maximum is worth out of sample. An equal-weighted average
    over the top N survivors - across every seed the search ran - keeps whatever the
    survivors agree on and cancels most of what each one found alone, which is exactly
    the part that was luck.

    What is averaged, precisely:

      * every member's BELIEF (`alpha`, its weighted sum of ranks, gate ignored) is
        re-ranked to [0, 1] within the tradable universe, so a member whose weights sum
        larger cannot out-shout the rest, and the ensemble score is the mean rank over
        the members that have an opinion on the name - at least `min_members` of them;
      * the regime gate is a VOTE: the ensemble steps aside on a date when at least half
        its members would, and invests the mean of those members' defensive exposure
        while it does. A member with the gate switched off always votes to stay in;
      * the portfolio shape is the median of the members' shapes - holding count and
        per-name cap - and the weighting scheme most of them chose.

    Signals are averaged rather than portfolios, deliberately. Averaging thirty
    twelve-name portfolios produces a two-hundred-name portfolio that pays a dollar of
    commission minimum on every one of them at this account size, and the result would
    say more about the cost model than about the signals.

    Not registered under a name: it exists only as the product of a particular search,
    and `evolve.ensemble_strategy(study)` is how one is obtained.
    """

    name = "evolved_ensemble"

    def __init__(self, members: list[EvolvedAlpha], study: str = "",
                 min_members: int | None = None, **params):
        if not members:
            raise ValueError("an ensemble needs at least one member")
        presets = {m.params["preset"] for m in members}
        if len(presets) != 1:
            raise ValueError(f"ensemble members must share a preset; got {presets}")
        pre_ranked = {bool(getattr(m, "pre_ranked", False)) for m in members}
        if len(pre_ranked) != 1:
            raise ValueError("ensemble members must all be pre-ranked or all raw")

        self.members = list(members)
        self.pre_ranked = pre_ranked.pop()
        self.min_members = int(min_members if min_members is not None
                               else min(3, len(self.members)))
        # The fingerprint the registry hashes is `params`, so the members' own
        # behavioural fingerprints go there: two ensembles are the same trial exactly
        # when they average the same individuals.
        genome = members[0]._genome
        member_ids = [genome.fingerprint(genome.encode(m.params)) for m in members]
        # `member_fingerprints`, not `members`: BaseStrategy sets every parameter as an
        # attribute, and the strategy list must survive that.
        super().__init__(study=study, preset=presets.pop(), n_members=len(members),
                         min_members=self.min_members, member_fingerprints=member_ids,
                         **params)

        needed: list[str] = []
        for m in self.members:
            needed += list(m.requires_features)
        self.requires_features = tuple(dict.fromkeys(needed))
        self.min_date = max((m.min_date or "") for m in self.members)
        self.warmup = max(int(getattr(m, "warmup", 0) or 0) for m in self.members)

        top_k = int(np.median([m.construction.top_k for m in self.members]))
        max_weight = float(np.median([m.construction.max_weight for m in self.members]))
        schemes = [m.construction.weighting for m in self.members]
        weighting = max(sorted(set(schemes)), key=schemes.count)
        self.construction = Construction(top_k=top_k, weighting=weighting,
                                         max_weight=max_weight, min_names=10)
        self._defensive_column = self.members[0]._defensive_column

    # ------------------------------------------------------------- behaviour

    def on_start(self, panel) -> None:
        for m in self.members:
            if hasattr(m, "on_start"):
                m.on_start(panel)

    def vote(self, ctx: Context) -> tuple[float, float]:
        """(share of members that would be defensive, their mean defensive gross)."""
        votes = [m.is_defensive(ctx) for m in self.members]
        share = float(np.mean(votes)) if votes else 0.0
        gross = [float(m.params["defensive_gross"]) for m, v in zip(self.members, votes)
                 if v]
        return share, (float(np.mean(gross)) if gross else 1.0)

    def is_defensive(self, ctx: Context) -> bool:
        return self.vote(ctx)[0] >= 0.5

    def alpha(self, ctx: Context) -> np.ndarray:
        """Mean re-ranked belief across the members with an opinion on each name."""
        elig = ctx.tradable
        n = ctx.close.shape[1]
        total = np.zeros(n, dtype=np.float64)
        count = np.zeros(n, dtype=np.float64)
        for m in self.members:
            s = m.alpha(ctx)
            r = rank_pct(s, elig)
            ok = np.isfinite(r)
            total[ok] += r[ok]
            count[ok] += 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            out = total / count
        return np.where((count >= self.min_members) & np.isfinite(out), out, np.nan)

    def score(self, ctx: Context) -> np.ndarray:
        if ctx.tradable.sum() < 10:
            return np.full(ctx.close.shape[1], np.nan)
        if self.is_defensive(ctx):
            return self.members[0].defensive_score(ctx)
        return self.alpha(ctx)

    def target_weights(self, ctx: Context) -> np.ndarray:
        c = self.construction
        share, gross = self.vote(ctx)
        if share >= 0.5:
            c = replace(c, gross=float(min(max(gross, 0.01), 1.0)))
        s = np.asarray(self.score(ctx), dtype=np.float64)
        vol = (self.vol_for_weighting(ctx) if c.weighting == "inverse_vol" else None)
        return build_weights(s, self.eligible(ctx), c, tiebreak=ctx.tiebreak, vol=vol)

    # ------------------------------------------------------------- reporting

    @property
    def family_usage(self) -> dict[str, int]:
        """How many members back each family. Empty for feature-preset members."""
        counts: dict[str, int] = {}
        for m in self.members:
            for fam in getattr(m, "active_families", []):
                counts[fam] = counts.get(fam, 0) + 1
        return counts

    @property
    def feature_usage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self.members:
            for f in m.active_features:
                counts[f] = counts.get(f, 0) + 1
        return counts

    def describe(self) -> dict:
        d = super().describe()
        d["preset"] = self.params["preset"]
        d["pre_ranked"] = self.pre_ranked
        d["n_members"] = len(self.members)
        d["family_usage"] = self.family_usage
        d["feature_usage"] = self.feature_usage
        return d

    def explain(self) -> str:
        """The ensemble in sentences: what its members agree on."""
        n = len(self.members)
        lines = [f"Averages the signals of {n} evolved individuals, equal-weighted, "
                 f"re-ranked before averaging; a name needs at least "
                 f"{self.min_members} opinions to be scored."]
        fams = self.family_usage
        if fams:
            lines.append("Families backed, by share of members:")
            for name, k in sorted(fams.items(), key=lambda kv: -kv[1]):
                label = FAMILY_BY_NAME[name].label if name in FAMILY_BY_NAME else name
                lines.append(f"    {k / n * 100:5.1f}%  {label}")
        else:
            feats = self.feature_usage
            lines.append("Features ranked, by share of members:")
            for name, k in sorted(feats.items(), key=lambda kv: -kv[1])[:12]:
                lines.append(f"    {k / n * 100:5.1f}%  {name}")
        gated = sum(1 for m in self.members if m.params["use_regime"] == "on")
        c = self.construction
        lines.append(f"Holds the top {c.top_k} by score, {c.weighting}-weighted, capped "
                     f"at {c.max_weight * 100:.1f}% per name (medians of the members).")
        lines.append(f"{gated} of {n} members carry the regime gate; the ensemble steps "
                     "aside only when at least half of them would.")
        return "\n".join(lines)
