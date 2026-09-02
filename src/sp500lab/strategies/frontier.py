"""The second wave: five mechanisms the first twelve hypotheses do not cover.

`alpha.py` asks which stocks to own. Three of the five here ask something different -
WHEN returns happen (overnight_momentum), how much to own at all (vol_managed), and
whether averaging every disagreeing hypothesis beats backing any one of them
(ensemble_rank). The other two are cross-sectional claims the first twelve left on the
table: anchoring to the 52-week high, and the dividend calendar.

Each earns its place the same way the first twelve did - a sentence somebody could argue
with, and a mechanism the others do not share:

    overnight_momentum   momentum's profits accrue while the market is CLOSED
    week52_breakout      investors anchor on the 52-week high and under-react near it
    div_month            prices drift up in months where a dividend is predictable
    vol_managed          volatility is forecastable and returns are not, so scale by it
    ensemble_rank        the average of twelve disagreeing hypotheses beats most of them

Three of the five need nothing but prices and dividends, so they run over the full 2007
window - unlike most of `alpha.py`, which starts in 2010 with the fundamentals.

Why these are trials, not discoveries
--------------------------------------
Every strategy in this file was written AFTER the 2022-2026 forward test was run and
read. Nothing here was selected using that period - each is a standing result from the
published literature, implemented from its paper's recipe with conventional parameters -
but the author of this file knew, when choosing which papers to implement, that
2022-2026 was a mega-cap market. That is a contamination the registry cannot see and
the deflated Sharpe cannot correct. It is recorded here and in ADR-037 so nobody
mistakes these for pre-registered out-of-sample candidates: their honest test is the
months that arrive after 2026-08.
"""

from __future__ import annotations

import numpy as np

from ..backtest.context import Context
from ..backtest.portfolio import Construction, build_weights
from ..backtest.strategy import FeatureStrategy, get_strategy, register
from .alpha import STANDARD, _macro
from .signals import blend, rank_pct, require


@register("overnight_momentum")
class OvernightMomentum(FeatureStrategy):
    """Rank on the overnight component of 12-1 momentum, and only that component.

    A close-to-close return is two different trades glued together: the overnight leg,
    where earnings land and institutions reposition, and the intraday leg, where retail
    flow and market-making noise live. Lou, Polk & Skouras (2019) split every US stock
    return this way and found that momentum's profits accrue almost entirely overnight -
    the intraday component of a winner's past return predicts nothing, or reverses.

    So this is `momentum_12_1` with the noisy half of its own input amputated. The
    direct comparison against `momentum_12_1` over the same window is the entire point:
    same construction, same costs, one difference. If LPS's decomposition carries real
    information, this should be the better momentum; if it does not, the two curves
    should be indistinguishable.

    Buildable here because the panel keeps adjusted OPENS next to closes - a
    close-only data layer cannot express the idea at all.
    """

    name = "overnight_momentum"
    requires_features = ("mom_on_12_1",)
    warmup = 273  # the 252-session lookback plus the 21-session skip
    construction = STANDARD

    def score(self, ctx: Context) -> np.ndarray:
        return rank_pct(self.f(ctx, "mom_on_12_1"), ctx.tradable)


@register("week52_breakout")
class Week52Breakout(FeatureStrategy):
    """Buy what trades nearest its own 52-week high. Anchoring, not momentum.

    George & Hwang (2004): the ratio of price to its 52-week high predicts returns, and
    it is not the same information as the return that produced it - a stock can be near
    its high after a flat year or far from it after a strong one. The mechanism is
    anchoring: traders treat the 52-week high as a ceiling and under-react to good news
    near it, so the news gets priced slowly, and the drift is the slow pricing.

    The genetic algorithm's price-preset winner put a +0.38 weight on exactly this
    feature, which is the strongest reason to test it standing alone: if the effect is
    real it should survive without the twelve other weights around it, and if it only
    works inside the evolved blend, that is worth knowing too.
    """

    name = "week52_breakout"
    requires_features = ("high_52w_ratio",)
    warmup = 260
    construction = STANDARD

    def score(self, ctx: Context) -> np.ndarray:
        return rank_pct(self.f(ctx, "high_52w_ratio"), ctx.tradable)


@register("div_month")
class DividendMonth(FeatureStrategy):
    """Own names whose own payment cadence predicts a dividend this coming month.

    Hartzmark & Solomon (2013) call it the dividend-month premium: stocks earn
    abnormally high returns in months where a dividend is PREDICTED, driven by price
    pressure from dividend-seeking buyers into the ex-date - and the effect needs no
    information beyond the calendar, because a quarterly payer's next ex-date is
    knowable from its last few.

    The eligible set is therefore defined by an event clock, not a ranking: only names
    whose cadence says a payment lands within the coming month may be held (the
    `div_due_1m` feature, built from discrete ex-dates - an adjusted-close feed cannot
    express this). Within the eligible set, higher trailing yield ranks first, because a
    larger expected payment is a larger buying pressure.

    The honest caveat is turnover: the eligible set rolls over almost completely every
    month by construction, so this strategy pays the full spread bill twelve times a
    year. Whether the premium survives that bill at retail size is precisely what the
    three cost settings measure - published results are gross of costs.
    """

    name = "div_month"
    requires_features = ("div_due_1m", "div_yield")
    construction = Construction(top_k=50, weighting="equal", max_weight=0.05,
                                min_names=10)

    def score(self, ctx: Context) -> np.ndarray:
        due = self.f(ctx, "div_due_1m")
        elig = ctx.tradable & (due > 0)
        return rank_pct(self.f(ctx, "div_yield"), require(elig, due))


@register("vol_managed")
class VolManaged(FeatureStrategy):
    """Scale exposure by inverse variance: own less of the market when it is turbulent.

    Moreira & Muir (2017) showed that dividing exposure by recent realised variance
    raises Sharpe ratios across essentially every equity factor, because volatility is
    strongly forecastable at a one-month horizon and expected returns are not - so
    turbulent months deliver the same return at much higher risk, and an investor who
    de-risks into them keeps most of the return at a fraction of the volatility.

    The implementation is the long-only half of their rule: exposure is
    `min(1, (vol_252d / vol_21d)^2)` of the account in the equal-weighted point-in-time
    index, the rest in cash. Long-run volatility is the target, so there are no fitted
    constants - when the last month is calmer than the last year the account is simply
    fully invested, and it can never lever up (ADR-016). That halves the published
    effect, which came substantially from leverage in calm markets; what remains is the
    drawdown protection, and that is the claim being tested.

    Deliberately different from `defensive_regime`: that strategy flips discretely on a
    trend threshold and changes WHAT it owns; this one scales continuously on variance
    and never changes the portfolio, so the two cannot succeed for the same reason.
    A floor of 20% keeps the scaling from rounding the account to cash entirely in a
    2008, which would turn a risk overlay into a market-timing call.
    """

    name = "vol_managed"
    requires_features = ("mkt_vol_ratio",)
    construction = Construction(top_k=None, weighting="equal", min_names=5)

    #: Exposure floor. Conventional, not searched; see the docstring.
    MIN_GROSS = 0.20

    def score(self, ctx: Context) -> np.ndarray:
        return np.where(ctx.tradable, 1.0, np.nan)

    def target_weights(self, ctx: Context) -> np.ndarray:
        from dataclasses import replace
        ratio = _macro(self.f(ctx, "mkt_vol_ratio"))
        gross = 1.0
        if np.isfinite(ratio) and ratio > 0:
            gross = float(np.clip(1.0 / ratio ** 2, self.MIN_GROSS, 1.0))
        c = replace(self.construction, gross=gross)
        s = np.asarray(self.score(ctx), dtype=np.float64)
        return build_weights(s, self.eligible(ctx), c, tiebreak=ctx.tiebreak)


#: The hypotheses the ensemble averages: everything in alpha.py. Named here rather than
#: imported from GROUPS to avoid a circular import; test_frontier pins the two in sync.
ENSEMBLE_MEMBERS = (
    "residual_momentum", "frog_in_the_pan", "pead_drift", "accrual_quality",
    "quality_value", "restatement_averse", "illiquidity_carry", "index_entry_drift",
    "lottery_averse", "dividend_grower", "defensive_regime", "multi_factor",
)


@register("ensemble_rank")
class EnsembleRank(FeatureStrategy):
    """Average the ranks of all twelve hypotheses. Model uncertainty, not conviction.

    The twelve strategies in `alpha.py` were chosen to disagree - under-reaction,
    accounting quality, liquidity, index mechanics, payout behaviour, lottery
    preference. Nobody knows which of those stories is true in which decade, and an
    equal-weighted average over them is the textbook answer to exactly that ignorance:
    ensembles beat most of their members almost universally, because averaging cancels
    each model's idiosyncratic error while keeping whatever signal they share.

    Each member's score is re-ranked to [0, 1] before averaging so that a member that
    emits raw magnitudes (index_entry_drift scores in months of tenure) cannot
    out-shout the ones that emit percentiles, and a name is scored once at least three
    members have an opinion - early years carry only the price-based members, and that
    is fine. Only the SIGNALS are averaged: defensive_regime's de-risking overlay is
    deliberately not inherited, so this is always fully invested and measures signal
    averaging alone.

    This is the cheapest idea in the whole roster - no new data, no new features, no
    parameters - which is exactly why it has to be in the competition. Any future model
    that cannot beat the plain average of the existing hypotheses has found nothing.
    """

    name = "ensemble_rank"
    construction = STANDARD

    def __init__(self, min_components: int = 3, **kw):
        super().__init__(min_components=int(min_components), **kw)
        self._members = [get_strategy(n) for n in ENSEMBLE_MEMBERS]
        # Instance-level rather than class-level, because it is derived: the engine
        # loads the feature panel off this attribute, and a member added to the roster
        # must never be able to find its features missing.
        self.requires_features = tuple(sorted({
            f for m in self._members
            for f in getattr(m, "requires_features", ())}))

    def score(self, ctx: Context) -> np.ndarray:
        elig = ctx.tradable
        parts = [rank_pct(np.asarray(m.score(ctx), dtype=np.float64), elig)
                 for m in self._members]
        return blend(parts, min_components=self.min_components)

    def describe(self) -> dict:
        d = super().describe()
        d["members"] = list(ENSEMBLE_MEMBERS)
        return d
