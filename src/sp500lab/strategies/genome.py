"""The genome: a bounded search space, and the strategy that a point in it becomes.

This is the substrate the genetic algorithm operates on. `evolve/` implements selection,
crossover and mutation and knows nothing about finance; this file knows about finance and
nothing about evolution. The seam between them is a float vector.

Why the space is deliberately small
------------------------------------
An unconstrained search over indicator combinations finds something that works
beautifully in-sample every single time. That is not a risk, it is arithmetic: the maximum
of N draws from a zero-mean distribution grows with N whether or not there is any signal.
The defences here are structural rather than statistical, because a statistical correction
applied to an unbounded search is correcting a number that was never meaningful:

  * **A weighted sum of ranked features, not an expression tree.** No `if`, no products of
    indicators, no evolved arithmetic. Far less expressive, and every individual in the
    space has an economic reading a human can argue with. A tree-based GA on this data
    would find spectacular nonsense on its first generation.
  * **Percentile ranks, not raw values.** A book-to-market of 300 from one bad share count
    cannot dominate a portfolio, because it is worth exactly "first place".
  * **A dead zone on every weight.** Weights below `DEAD_ZONE` are treated as exactly
    zero, so an individual using three features genuinely uses three features. Without it
    every individual uses all of them a little, parsimony pressure has nothing to grip,
    and two genomes that behave identically look different to the deduplicator.
  * **Curated feature presets.** 13 features for the full 2007 window, 23 including
    fundamentals from 2010. Not 75. Every feature added to the preset multiplies the
    search space and the number of trials the deflated Sharpe has to discount.

The regime gate
---------------
Three genes decide whether the individual de-risks when the market is falling or unusually
volatile, and by how much. It is the only non-linearity in the space, and it is there
because momentum crashes are a real phenomenon that no cross-sectional score can see: in
2009 the losers momentum was avoiding were the highest-beta names, and they rebounded
hardest. Whether a search actually USES the gate is one of the more interesting things a
run reports.

Reading a genome
----------------
`describe_genome()` prints an individual as sentences. A winning parameter vector nobody
can read is a winning parameter vector nobody can check, and this project's entire
argument is that results should be checkable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: A feature weight whose magnitude is below this contributes nothing. See the docstring:
#: without a dead zone, "how many features does this use" has no answer.
DEAD_ZONE = 0.10

#: Features available for the whole research window (2007-04 onward). Price, volume,
#: liquidity, membership and dividends - everything that does not depend on XBRL.
PRICE_FEATURES: tuple[str, ...] = (
    "mom_12_1", "resid_mom_12_1", "rev_1m", "info_discreteness", "high_52w_ratio",
    "trend_200d", "vol_126d", "idio_vol_252d", "beta_252d", "max_ret_21d",
    "amihud_illiq", "log_dollar_volume", "div_yield",
)

#: Everything above, plus the fundamentals. Costs three years of history: XBRL starts
#: 2009-04 and the derived growth and surprise features need several quarters after that.
FUNDAMENTAL_FEATURES: tuple[str, ...] = (
    "book_to_market", "earnings_yield", "gross_profitability", "roe", "accruals",
    "log_market_cap", "asset_growth", "eps_surprise", "restatement_rate",
    "leverage",
)

#: Macro features the regime gate reads. Never revised, and lagged a session (macro.py).
REGIME_FEATURES: tuple[str, ...] = ("mkt_trend_200d", "mkt_vol_ratio")

#: The overnight/intraday decomposition and the dividend calendar (feature_version 3).
#: A separate preset rather than an extension of `price`, deliberately: stored winners
#: decode against their preset's feature list, so mutating an existing preset would
#: silently mis-decode every genome already in a checkpoint.
NIGHT_FEATURES: tuple[str, ...] = (
    "mom_on_12_1", "mom_id_12_1", "on_minus_id_252d", "div_due_1m",
)

PRESETS = {
    "price": PRICE_FEATURES,
    "full": PRICE_FEATURES + FUNDAMENTAL_FEATURES,
    "night": PRICE_FEATURES + NIGHT_FEATURES,
}

#: The date each preset's inputs actually exist from. `price` inherits the panel's own
#: start; `full` cannot begin before its fundamentals do.
PRESET_MIN_DATE = {"price": "", "full": "2010-07-01", "night": ""}

WEIGHTINGS = ("equal", "score_rank", "inverse_vol")


# --------------------------------------------------------------------------
# The generic machinery: genes and a genome specification
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Gene:
    """One tunable parameter and the bounds a mutation must respect.

    `integer` matters: a top_k of 37.4 is not a thing, and rounding at use time rather
    than at mutation time makes two genomes that differ below the rounding threshold
    score identically - which wastes a generation and corrupts the trial count, because
    the registry would see two fingerprints for one hypothesis.

    `choices` turns the gene categorical. The underlying value stays a float in [0, n) so
    that crossover and mutation need no special cases; only `decode` knows it is a label.
    """

    name: str
    low: float
    high: float
    integer: bool = False
    choices: tuple[str, ...] | None = None
    note: str = ""

    @property
    def span(self) -> float:
        return float(self.high - self.low)

    def clip(self, value: float) -> float:
        v = float(np.clip(value, self.low, self.high))
        return float(round(v)) if (self.integer or self.choices) else v

    def sample(self, rng: np.random.Generator) -> float:
        return self.clip(rng.uniform(self.low, self.high))

    def decode(self, value: float):
        v = self.clip(value)
        if self.choices:
            return self.choices[int(min(max(v, 0), len(self.choices) - 1))]
        return int(v) if self.integer else v


@dataclass(frozen=True)
class Genome:
    """An ordered set of genes: the search space, and the codec for a point in it."""

    genes: tuple[Gene, ...]
    name: str = "genome"

    def __len__(self) -> int:
        return len(self.genes)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.genes)

    @property
    def lows(self) -> np.ndarray:
        return np.array([g.low for g in self.genes], dtype=np.float64)

    @property
    def highs(self) -> np.ndarray:
        return np.array([g.high for g in self.genes], dtype=np.float64)

    @property
    def spans(self) -> np.ndarray:
        return self.highs - self.lows

    def random(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([g.sample(rng) for g in self.genes], dtype=np.float64)

    def clip(self, vector: np.ndarray) -> np.ndarray:
        """Bring a vector back inside the box. Never raises - a mutation that overshoots
        should produce a valid individual, not an exception halfway through a population."""
        v = np.asarray(vector, dtype=np.float64).ravel()
        if len(v) != len(self.genes):
            raise ValueError(f"genome has {len(v)} genes, expected {len(self.genes)}")
        return np.array([g.clip(x) for g, x in zip(self.genes, v)], dtype=np.float64)

    def decode(self, vector: np.ndarray) -> dict:
        """Vector -> the parameter dict a strategy is constructed from."""
        v = self.clip(vector)
        return {g.name: g.decode(x) for g, x in zip(self.genes, v)}

    def encode(self, params: dict) -> np.ndarray:
        out = []
        for g in self.genes:
            x = params[g.name]
            if g.choices:
                x = float(g.choices.index(x)) if isinstance(x, str) else float(x)
            out.append(float(x))
        return self.clip(np.array(out, dtype=np.float64))

    def fingerprint(self, vector: np.ndarray, decimals: int = 3) -> str:
        """A stable id for the BEHAVIOUR of a genome, not for its bits.

        Rounded, and with dead-zoned weights collapsed to zero, so two vectors that
        produce the same portfolio are one trial rather than two. The registry counts
        distinct fingerprints for the deflated Sharpe, and counting behaviourally
        identical individuals twice would over-deflate the winner.
        """
        v = self.clip(vector).copy()
        for i, g in enumerate(self.genes):
            if g.name.startswith("w_") and abs(v[i]) < DEAD_ZONE:
                v[i] = 0.0
        return ",".join(f"{x:.{decimals}f}" for x in v)


# --------------------------------------------------------------------------
# The alpha genome
# --------------------------------------------------------------------------

def alpha_genome(preset: str = "price") -> Genome:
    """The search space for `EvolvedAlpha`: feature weights plus portfolio shape."""
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; have {sorted(PRESETS)}")
    genes = [Gene(f"w_{f}", -1.0, 1.0, note=f"weight on the percentile rank of {f}")
             for f in PRESETS[preset]]
    genes += [
        Gene("top_k", 10, 100, integer=True,
             note="how many names to hold; an economic decision at $100k, not a knob"),
        Gene("max_weight", 0.02, 0.20, note="per-name cap"),
        Gene("weighting", 0, len(WEIGHTINGS) - 1, choices=WEIGHTINGS,
             note="how capital is split across the selected names"),
        Gene("use_regime", 0, 1, choices=("off", "on"),
             note="de-risk when the market is falling or unusually volatile"),
        Gene("defensive_gross", 0.2, 1.0,
             note="fraction of NAV invested while defensive; the rest is cash"),
        Gene("vol_trigger", 1.0, 3.0,
             note="realised vol relative to its own year that counts as a shock"),
    ]
    return Genome(tuple(genes), name=f"alpha:{preset}")


def describe_genome(genome: Genome, vector: np.ndarray) -> str:
    """An individual, in sentences. A winner nobody can read is a winner nobody can check."""
    p = genome.decode(vector)
    lines = []
    weights = [(n[2:], v) for n, v in p.items()
               if n.startswith("w_") and abs(v) >= DEAD_ZONE]
    weights.sort(key=lambda kv: -abs(kv[1]))
    if weights:
        lines.append(f"Ranks {len(weights)} feature(s):")
        for feature, w in weights:
            direction = "high is good" if w > 0 else "LOW is good"
            lines.append(f"    {w:+.2f}  {feature:<22s} ({direction})")
    else:
        lines.append("Ranks nothing: every weight is inside the dead zone, so this "
                     "individual holds an arbitrary slice of the universe.")
    dropped = sum(1 for n, v in p.items()
                  if n.startswith("w_") and abs(v) < DEAD_ZONE)
    lines.append(f"Ignores {dropped} feature(s) (weight inside the +/-{DEAD_ZONE} "
                 "dead zone).")
    lines.append(f"Holds the top {p['top_k']} by score, {p['weighting']}-weighted, "
                 f"capped at {p['max_weight'] * 100:.1f}% per name.")
    if p["use_regime"] == "on":
        lines.append(f"De-risks to {p['defensive_gross'] * 100:.0f}% invested when the "
                     "index is below its 200-day average or realised volatility exceeds "
                     f"{p['vol_trigger']:.2f}x its own year.")
    else:
        lines.append("Always fully invested; the regime gate is switched off.")
    return "\n".join(lines)


def active_features(genome: Genome, vector: np.ndarray) -> list[str]:
    """The features an individual actually uses. Parsimony pressure counts these."""
    p = genome.decode(vector)
    return [n[2:] for n, v in p.items()
            if n.startswith("w_") and abs(v) >= DEAD_ZONE]
