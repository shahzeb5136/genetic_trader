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

Two kinds of preset (ADR-031, ADR-048)
--------------------------------------
The first three presets - `price`, `full`, `night` - give the search one weight gene per
feature. Three searches ran over them, every winner cleared the deflated-Sharpe
convention, and every winner decayed out of sample. The diagnosis in ADR-048 is that
even 13 free signs and magnitudes is too much room on fifteen years of monthly data.

The `families` presets shrink the room. The features with a prior story are grouped into
nine economically motivated FAMILIES, each a fixed composite of prior-signed ranks, and an
individual carries one non-negative weight per family - it chooses WHICH stories to back
and how hard, never whether value means cheap-is-good. A preset also caps how many
families may be live at once (`max_active`), enforced at decode time, so the space is
"pick at most three of nine stories and weight them", which is a hypothesis a person
could have written down. Everything without a story - redundant horizons, contested
decompositions, size proxies, filing behaviour, macro context - is cut and the reason is
recorded in `CUT_FEATURES`.

Presets are immutable (ADR-038). Stored genomes decode BY POSITION against their preset's
gene list, and a family's member tuple and prior signs are part of that contract: change
one and every checkpoint on disk silently becomes a different strategy. New features, new
members or a different cap mean a NEW preset.

The regime gate
---------------
Three genes decide whether the individual de-risks when the market is falling or unusually
volatile, and by how much. It is the only non-linearity in the space, and it is there
because momentum crashes are a real phenomenon that no cross-sectional score can see: in
2009 the losers momentum was avoiding were the highest-beta names, and they rebounded
hardest. Whether a search actually USES the gate is one of the more interesting things a
run reports - and under ADR-049 switching it on is charged as one more rule.

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

#: The feature-weight presets: one gene per feature, each free in [-1, +1]. FROZEN.
PRESETS = {
    "price": PRICE_FEATURES,
    "full": PRICE_FEATURES + FUNDAMENTAL_FEATURES,
    "night": PRICE_FEATURES + NIGHT_FEATURES,
}

WEIGHTINGS = ("equal", "score_rank", "inverse_vol")


# --------------------------------------------------------------------------
# Families: the features with a prior story, grouped by the story (ADR-048)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Family:
    """A handful of features that tell ONE economic story, with the sign fixed by it.

    `members` are (feature, sign) pairs: +1 means the high end of the feature's
    percentile rank is the good end, -1 means the low end is. The composite is the plain
    mean of the prior-signed ranks over whichever members a name has a value for, so a
    family is one number in [0, 1] per name and the search cannot re-weight or re-sign
    its members. That is the point: the family IS the hypothesis, and the search only
    decides how much to back it.
    """

    name: str
    label: str
    story: str
    members: tuple[tuple[str, int], ...]
    reference: str = ""

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(f for f, _ in self.members)

    def sign(self, feature: str) -> int:
        for f, s in self.members:
            if f == feature:
                return s
        raise KeyError(f"{feature} is not a member of the {self.name} family")


FAMILIES: tuple[Family, ...] = (
    Family(
        "momentum", "Momentum",
        "Prices under-react to information and keep drifting for six to twelve months. "
        "The 12-1 return is the classic measure; the residual version strips out the "
        "beta bet that makes raw momentum crash on rebounds; nearness to the 52-week "
        "high is the anchor investors under-react against; and a return that arrived in "
        "many small steps drifts further than one that arrived in a jump.",
        (("mom_12_1", +1), ("resid_mom_12_1", +1), ("high_52w_ratio", +1),
         ("info_discreteness", -1)),
        reference="Jegadeesh & Titman 1993; Blitz, Huij & Martens 2011; George & Hwang "
                  "2004; Da, Gurun & Warachka 2014"),
    Family(
        "reversal", "Short-term reversal",
        "The last month's return reverses: liquidity provision is paid for, and the "
        "effect is the reason 12-1 momentum skips its most recent month. The most "
        "expensive story to trade in this file, because it turns the book every month.",
        (("rev_1m", +1),),
        reference="Jegadeesh 1990; Lehmann 1990"),
    Family(
        "low_risk", "Low risk",
        "The safest names have earned the best risk-adjusted returns, which no risk model "
        "predicts: low beta, low idiosyncratic volatility, low realised volatility and a "
        "small largest-daily-gain all point the same way. The most robust effect on this "
        "project's own scoreboard.",
        (("vol_126d", -1), ("idio_vol_252d", -1), ("beta_252d", -1),
         ("max_ret_21d", -1)),
        reference="Frazzini & Pedersen 2014; Ang, Hodrick, Xing & Zhang 2006; Bali, "
                  "Cakici & Whitelaw 2011"),
    Family(
        "liquidity", "Illiquidity",
        "Holders of what is hard to sell are paid for holding it. One member on purpose: "
        "dollar volume and market cap are size proxies whose premium has been absent for "
        "decades, and the cost model already charges the spread, so this is the one "
        "liquidity measure with a story that survives the costs.",
        (("amihud_illiq", +1),),
        reference="Amihud 2002"),
    Family(
        "payout", "Payout",
        "Cash returned to shareholders is a statement management cannot make for free. "
        "Yield is the income tilt, a raised dividend is the costliest signal a board can "
        "send, and prices drift up in the month a payment is expected. All three are "
        "visible from 2007, unlike anything that needs a filing.",
        (("div_yield", +1), ("div_growth_1y", +1), ("div_due_1m", +1)),
        reference="Lintner 1956; Miller & Rock 1985; Hartzmark & Solomon 2013"),
    Family(
        "value", "Value",
        "Cheap relative to a fundamental: earnings, book equity and operating cash flow. "
        "Three denominators because a wrong share count shows up in each differently, "
        "and because cheap-on-assets and cheap-on-earnings disagreed sharply over "
        "2010-2021 - the composite refuses to bet on which one is right.",
        (("book_to_market", +1), ("earnings_yield", +1), ("cf_yield", +1)),
        reference="Basu 1977; Fama & French 1992"),
    Family(
        "quality", "Quality",
        "Profitable businesses whose earnings are cash and whose balance sheets are not "
        "stretched. Gross profitability is the other side of value, accruals are the part "
        "of earnings that reverses, and leverage is the safety leg. The family the "
        "second search rediscovered without being told it exists.",
        (("gross_profitability", +1), ("roe", +1), ("accruals", -1), ("leverage", -1)),
        reference="Novy-Marx 2013; Sloan 1996; Asness, Frazzini & Pedersen 2019"),
    Family(
        "investment", "Investment",
        "Companies that grow their balance sheets fastest subsequently underperform. One "
        "member, measured against the value as ORIGINALLY reported so it is a growth "
        "rate somebody could have computed at the time.",
        (("asset_growth", -1),),
        reference="Cooper, Gulen & Schill 2008; Fama & French 2015"),
    Family(
        "earnings", "Earnings surprise",
        "Post-earnings-announcement drift: prices keep moving in the direction of an "
        "earnings surprise for weeks after it. The most robust anomaly after momentum, "
        "and a different mechanism from it - news rather than price.",
        (("eps_surprise", +1),),
        reference="Ball & Brown 1968; Bernard & Thomas 1989"),
)

FAMILY_BY_NAME: dict[str, Family] = {f.name: f for f in FAMILIES}

#: What the family presets deliberately leave out, and why. Recorded here rather than
#: implied by absence, because "we could not state a prior for it" is the decision and a
#: future search that wants one of these back should have to argue with the reason.
CUT_FEATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("A second measurement of a story a family already tells",
     ("mom_6_1", "mom_1m", "ret_3m", "ret_12m", "trend_200d", "vol_21d",
      "vol_of_vol_252d", "skew_252d", "eps_change_yoy", "debt_to_assets")),
    ("Contested: the project's own third search ranked it against the literature and "
     "decayed",
     ("mom_on_12_1", "mom_id_12_1", "on_minus_id_252d")),
    ("A size or trading-cost proxy: the premium has been absent for decades and the "
     "cost model already charges the spread",
     ("log_dollar_volume", "log_market_cap", "half_spread_bp")),
    ("A one-off event at a known date, not a ranking; expressed by a hand-written "
     "strategy instead",
     ("months_in_index", "new_member", "log_tenure", "div_cut", "pays_dividend")),
    ("Ambiguous by the catalogue's own reading, or mostly an industry classification "
     "in disguise",
     ("buyback_yield", "cash_ratio", "current_ratio", "capex_intensity",
      "rnd_intensity", "sales_growth", "earnings_growth", "corr_mkt_252d")),
    ("Filing behaviour: a plausible governance reading that may be a size and "
     "complexity tilt wearing a costume, and the second search ranked it the wrong way",
     ("days_since_filing", "filing_lag_days", "restatement_rate")),
    ("Context, not a cross-section: one number per date. The regime gate reads two of "
     "them as levels",
     ("vix", "vix_chg_63d", "vix_relative", "term_spread", "term_spread_chg_63d",
      "term_spread_3m", "term_spread_3m_chg_63d", "ust10y", "ust10y_chg_63d",
      "fed_funds", "fed_funds_chg_63d", "hy_spread", "hy_spread_chg_63d", "ig_spread",
      "ig_spread_chg_63d", "dollar_index", "dollar_index_chg_63d", "oil", "oil_chg_63d",
      "mkt_trend_200d", "mkt_drawdown", "mkt_vol_21d", "mkt_vol_252d", "mkt_vol_ratio",
      "mkt_breadth_200d")),
)


@dataclass(frozen=True)
class FamilyPreset:
    """Which families a search may back, and how many at once. FROZEN once searched."""

    name: str
    families: tuple[str, ...]
    max_active: int = 3
    min_date: str = ""
    note: str = ""

    @property
    def features(self) -> tuple[str, ...]:
        """Every feature any member family reads, in family order, deduplicated."""
        return tuple(dict.fromkeys(
            f for name in self.families for f in FAMILY_BY_NAME[name].features))


#: The family presets. `families` is the default search space from 2026-09 (ADR-048).
FAMILY_PRESETS: dict[str, FamilyPreset] = {
    "families": FamilyPreset(
        "families", tuple(f.name for f in FAMILIES), max_active=3,
        min_date="2010-07-01",
        note="all nine stories; starts 2010-07 because four of them need XBRL"),
    "families-price": FamilyPreset(
        "families-price", ("momentum", "reversal", "low_risk", "liquidity", "payout"),
        max_active=3, min_date="",
        note="the five stories visible from 2007-04 without a filing"),
}

#: The date each preset's inputs actually exist from. `price` inherits the panel's own
#: start; `full` cannot begin before its fundamentals do.
PRESET_MIN_DATE = {"price": "", "full": "2010-07-01", "night": ""} | {
    name: fp.min_date for name, fp in FAMILY_PRESETS.items()}


def all_presets() -> tuple[str, ...]:
    """Every preset name, feature-weight presets first."""
    return tuple(PRESETS) + tuple(FAMILY_PRESETS)


def preset_kind(preset: str) -> str:
    """'features' for one-weight-per-feature presets, 'families' for the capped ones."""
    if preset in PRESETS:
        return "features"
    if preset in FAMILY_PRESETS:
        return "families"
    raise KeyError(f"unknown preset {preset!r}; have {all_presets()}")


def preset_features(preset: str) -> tuple[str, ...]:
    """The feature columns a preset's strategies read - what the search has to rank."""
    if preset in PRESETS:
        return tuple(PRESETS[preset])
    return FAMILY_PRESETS[preset].features


def preset_families(preset: str) -> tuple[Family, ...]:
    """The families a preset carries, in gene order; empty for a feature preset."""
    if preset in FAMILY_PRESETS:
        return tuple(FAMILY_BY_NAME[n] for n in FAMILY_PRESETS[preset].families)
    return ()


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
    """An ordered set of genes: the search space, and the codec for a point in it.

    `weight_prefix` names the genes that carry signal weights (`w_` per feature, `f_` per
    family) and `max_active` caps how many of those may be live at once. The cap is part
    of the codec, not of the strategy: `decode()` zeroes everything past the cap, so a
    capped genome cannot express an individual that breaks it, and two vectors that
    differ only in capped-out genes are one fingerprint and one trial.
    """

    genes: tuple[Gene, ...]
    name: str = "genome"
    weight_prefix: str = "w_"
    max_active: int | None = None

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

    @property
    def weight_indices(self) -> tuple[int, ...]:
        return tuple(i for i, g in enumerate(self.genes)
                     if g.name.startswith(self.weight_prefix))

    def random(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([g.sample(rng) for g in self.genes], dtype=np.float64)

    def clip(self, vector: np.ndarray) -> np.ndarray:
        """Bring a vector back inside the box. Never raises - a mutation that overshoots
        should produce a valid individual, not an exception halfway through a population."""
        v = np.asarray(vector, dtype=np.float64).ravel()
        if len(v) != len(self.genes):
            raise ValueError(f"genome has {len(v)} genes, expected {len(self.genes)}")
        return np.array([g.clip(x) for g, x in zip(self.genes, v)], dtype=np.float64)

    def effective(self, vector: np.ndarray) -> np.ndarray:
        """The vector as it BEHAVES: clipped, dead-zoned, and cut to the active cap.

        Weights inside the dead zone become exactly zero. If the genome is capped, only
        the `max_active` largest weights survive and the rest become zero too - ties keep
        the earlier gene, so the outcome is deterministic. This is what `fingerprint()`
        hashes and, for a capped genome, what `decode()` returns.
        """
        v = self.clip(vector).copy()
        idx = list(self.weight_indices)
        for i in idx:
            if abs(v[i]) < DEAD_ZONE:
                v[i] = 0.0
        if self.max_active is not None and len(idx) > self.max_active:
            mags = np.array([abs(v[i]) for i in idx])
            order = np.argsort(-mags, kind="stable")
            for k in order[self.max_active:]:
                v[idx[int(k)]] = 0.0
        return v

    def decode(self, vector: np.ndarray) -> dict:
        """Vector -> the parameter dict a strategy is constructed from.

        Uncapped genomes decode the clipped vector as it is - the feature presets have
        always done that, and stored winners must keep decoding identically. Capped
        genomes decode the EFFECTIVE vector, so the cap is structural.
        """
        v = self.effective(vector) if self.max_active is not None else self.clip(vector)
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

        Rounded, and with dead-zoned (and capped-out) weights collapsed to zero, so two
        vectors that produce the same portfolio are one trial rather than two. The
        registry counts distinct fingerprints for the deflated Sharpe, and counting
        behaviourally identical individuals twice would over-deflate the winner.
        """
        v = self.effective(vector)
        return ",".join(f"{x:.{decimals}f}" for x in v)


# --------------------------------------------------------------------------
# The alpha genome
# --------------------------------------------------------------------------

def _shape_genes() -> list[Gene]:
    """The six genes that are not signal weights. Identical for every preset."""
    return [
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


def alpha_genome(preset: str = "price") -> Genome:
    """The search space for an evolved strategy: signal weights plus portfolio shape.

    A feature preset gets one free weight in [-1, +1] per feature. A family preset gets
    one NON-NEGATIVE weight in [0, 1] per family - the sign lives in the family's prior -
    and a cap on how many families may be live (ADR-048).
    """
    if preset in PRESETS:
        genes = [Gene(f"w_{f}", -1.0, 1.0, note=f"weight on the percentile rank of {f}")
                 for f in PRESETS[preset]]
        return Genome(tuple(genes + _shape_genes()), name=f"alpha:{preset}")
    if preset in FAMILY_PRESETS:
        fp = FAMILY_PRESETS[preset]
        genes = [Gene(f"f_{name}", 0.0, 1.0,
                      note=f"weight on the {FAMILY_BY_NAME[name].label} composite; "
                           "its direction is fixed by the family's prior")
                 for name in fp.families]
        return Genome(tuple(genes + _shape_genes()), name=f"alpha:{preset}",
                      weight_prefix="f_", max_active=fp.max_active)
    raise KeyError(f"unknown preset {preset!r}; have {all_presets()}")


def describe_genome(genome: Genome, vector: np.ndarray) -> str:
    """An individual, in sentences. A winner nobody can read is a winner nobody can check."""
    p = genome.decode(vector)
    lines = []
    if genome.weight_prefix == "f_":
        fams = family_weights(genome, vector)
        n_all = len(genome.weight_indices)
        if fams:
            cap = (f", at most {genome.max_active}" if genome.max_active else "")
            lines.append(f"Backs {len(fams)} of {n_all} families{cap}:")
            for name, w in fams:
                fam = FAMILY_BY_NAME[name]
                parts = ", ".join(f"{'+' if s > 0 else '-'}{f}" for f, s in fam.members)
                lines.append(f"    {w:.2f}  {fam.label:<20s} ({parts})")
        else:
            lines.append("Backs nothing: every family weight is inside the dead zone, so "
                         "this individual holds cash.")
        lines.append(f"Ignores {n_all - len(fams)} family(ies). Each family is the mean "
                     "of its members' prior-signed percentile ranks; '-' marks a member "
                     "whose LOW end is the good end.")
    else:
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
    """The features an individual actually reads. Parsimony pressure counts these.

    For a family genome that is every member of every live family, in family order and
    deduplicated - the composite reads all of them, whatever the family's weight.
    """
    p = genome.decode(vector)
    if genome.weight_prefix == "f_":
        out: dict[str, None] = {}
        for name, _ in family_weights(genome, vector):
            for f in FAMILY_BY_NAME[name].features:
                out[f] = None
        return list(out)
    return [n[2:] for n, v in p.items()
            if n.startswith("w_") and abs(v) >= DEAD_ZONE]


def family_weights(genome: Genome, vector: np.ndarray) -> list[tuple[str, float]]:
    """(family, weight) for every live family, largest first. Empty for a feature genome."""
    if genome.weight_prefix != "f_":
        return []
    p = genome.decode(vector)
    out = [(n[2:], float(v)) for n, v in p.items()
           if n.startswith("f_") and float(v) > 0.0]
    out.sort(key=lambda kv: -kv[1])
    return out


def active_families(genome: Genome, vector: np.ndarray) -> list[str]:
    """The families an individual backs. Empty for a feature genome."""
    return [name for name, _ in family_weights(genome, vector)]
