"""Twelve strategies, each making a different claim about why a stock outperforms.

Why these twelve and not twelve momentum variants
--------------------------------------------------
The baselines in `baselines.py` are null hypotheses. These are hypotheses. Each one is a
sentence somebody could argue with, and they are chosen to disagree with each other:

    residual_momentum     under-reaction, with market beta removed
    frog_in_the_pan       under-reaction depends on HOW the news arrived
    pead_drift            under-reaction to a specific, dated event
    accrual_quality       earnings that are not cash reverse
    quality_value         cheap AND good, which is rarer than either
    restatement_averse    accounting you can trust is worth paying for
    illiquidity_carry     you are paid for holding what is hard to sell
    index_entry_drift     forced index demand moves prices
    lottery_averse        people overpay for the chance of a jackpot
    dividend_grower       a raised dividend is a credible signal, a cut is a confession
    defensive_regime      WHEN to take risk, not which risk to take
    multi_factor          all of the above at once - the bar for anything clever

If several of them work, they should not all work for the same reason, and if the genetic
algorithm later rediscovers one of them from scratch that is evidence the search is
finding structure rather than noise. That is what this file is really for: it gives the GA
something to be measured against that is neither a straw man nor a random walk.

Four of them cannot be built without this repository's unusual data
--------------------------------------------------------------------
`pead_drift` and `restatement_averse` need `filed_date` alongside `period_end` - a
single-vintage fundamentals feed has thrown that away. `index_entry_drift` needs
point-in-time membership. `dividend_grower` needs dividends as discrete events rather than
dissolved into an adjusted-close column. Those four are the reason the data layer was
built first.

What every one of them shares
------------------------------
A score, and nothing else. Position sizing, the top-k cut, the per-name cap and the
unbiased tie-break all come from `portfolio.py`, so the comparison between them measures
signal quality rather than who tuned their weighting scheme (ADR-024, and the module
docstring in portfolio.py).

Honesty about what is being traded
-----------------------------------
The seven fundamentals-based strategies start in 2010 or later, because XBRL does. They
also trade a narrower universe than the price-based ones: fundamental coverage is 649 of
973 historical index members and correlates with survival, so they carry a survivorship
bias ON TOP of the price-coverage gap in ADR-023. Their `min_date` is set so the flat
pre-data stretch is excluded from their CAGR rather than being quietly averaged in, and
their windows therefore differ from the baselines'. Compare them on the same window or do
not compare them.
"""

from __future__ import annotations

import numpy as np

from ..backtest.context import Context
from ..backtest.portfolio import Construction, build_weights
from ..backtest.strategy import FeatureStrategy, register
from .signals import blend, conditional, rank_pct, require


def _macro(column: np.ndarray) -> float:
    """A macro feature's single value for a date.

    Macro columns are one number broadcast across every security, so the first finite
    entry IS the value. `np.nanmedian` gets the same answer, scans the whole
    cross-section to do it, and warns on every early date where the series does not
    exist yet.
    """
    finite = column[np.isfinite(column)]
    return float(finite[0]) if finite.size else float("nan")

#: The default shape of a portfolio here: 50 names, equal weight, 5% cap. 50 because at
#: $100k with a $1 per-order commission minimum, anything much wider pays for the
#: privilege (ADR-016 and the note in costs.py) - top_k is an economic decision.
STANDARD = Construction(top_k=50, weighting="equal", max_weight=0.05, min_names=10)

#: Fundamentals become usable partway through 2009 (the XBRL mandate) and the derived
#: growth and surprise features need several more quarters on top. These are the dates
#: each family of feature is actually populated, not round numbers.
FUNDAMENTAL_START = "2010-07-01"
SURPRISE_START = "2012-01-01"
RESTATEMENT_START = "2011-07-01"


# --------------------------------------------------------------------------
# Under-reaction: three different stories about the same phenomenon
# --------------------------------------------------------------------------

@register("residual_momentum")
class ResidualMomentum(FeatureStrategy):
    """12-1 momentum with market beta stripped out (Blitz, Huij & Martens 2011).

    Raw momentum is partly a bet on beta: after a long rally the winners ARE the
    high-beta names, so a momentum portfolio quietly becomes a leveraged market position
    and gets destroyed in the rebound. Ranking on the residual from a trailing regression
    against the equal-weighted index removes that component and leaves the part that is
    specific to the company.

    The direct comparison with `momentum_12_1` is the point - same window, same
    construction, same costs, one difference.
    """

    name = "residual_momentum"
    requires_features = ("resid_mom_12_1",)
    construction = STANDARD

    def score(self, ctx: Context) -> np.ndarray:
        return rank_pct(self.f(ctx, "resid_mom_12_1"), ctx.tradable)


@register("frog_in_the_pan")
class FrogInThePan(FeatureStrategy):
    """Momentum, but only where the information arrived gradually (Da, Gurun & Warachka).

    The claim is subtle and testable: two stocks with identical 12-month returns drift
    differently depending on whether the return came in many small pieces or a few jumps.
    Continuous information is individually negligible, so it is under-reacted to; a 15%
    gap on an earnings day is impossible to ignore and gets priced immediately.

    Implemented as a conditional sort rather than a blend, deliberately. A blend would
    claim the two signals are additive; the hypothesis says one is a REGIME for the
    other, and those are different models that happen to use the same two inputs.
    """

    name = "frog_in_the_pan"
    requires_features = ("mom_12_1", "info_discreteness")
    construction = STANDARD

    def __init__(self, keep: float = 0.5, **kw):
        super().__init__(keep=float(keep), **kw)

    def score(self, ctx: Context) -> np.ndarray:
        elig = ctx.tradable
        mom = rank_pct(self.f(ctx, "mom_12_1"), elig)
        # LOW discreteness is the continuous case: for a winner that ground steadily
        # upward, the share of up days is high, so the measure is negative.
        return conditional(mom, self.f(ctx, "info_discreteness"), elig,
                           keep=self.keep, high=False)


@register("pead_drift")
class PostEarningsDrift(FeatureStrategy):
    """Buy the biggest recent earnings surprises. The most robust anomaly after momentum.

    Standardised unexpected earnings against a random-walk expectation - this quarter
    versus the same quarter last year, scaled by the volatility of that difference. No
    analyst estimates are involved, which is a limitation (the true surprise is relative
    to expectations, not to last year) and also the reason it is computable here at all.

    The eligibility filter is the strategy. A surprise from a filing eight months ago is
    not news, so only names that have filed within `max_age_days` may be held, and the
    portfolio is therefore a rolling window over whoever reported most recently. This is
    the one strategy in the file whose universe is defined by an EVENT rather than by a
    ranking, and it is only possible because `filed_date` was kept.
    """

    name = "pead_drift"
    requires_features = ("eps_surprise", "days_since_filing")
    min_date = SURPRISE_START
    construction = Construction(top_k=40, weighting="equal", max_weight=0.05,
                                min_names=10)

    def __init__(self, max_age_days: float = 100.0, **kw):
        super().__init__(max_age_days=float(max_age_days), **kw)

    def score(self, ctx: Context) -> np.ndarray:
        age = self.f(ctx, "days_since_filing")
        sue = self.f(ctx, "eps_surprise")
        fresh = ctx.tradable & np.isfinite(age) & (age <= self.max_age_days)
        return rank_pct(sue, require(fresh, sue))


# --------------------------------------------------------------------------
# Accounting: what the numbers are made of
# --------------------------------------------------------------------------

@register("accrual_quality")
class AccrualQuality(FeatureStrategy):
    """Buy companies whose profit is cash (Sloan 1996). Avoid the ones where it is not.

    Accruals are the gap between reported earnings and operating cash flow: revenue
    booked but not collected, costs deferred, inventory built. They are legitimate
    accounting and they are also where earnings management lives, and Sloan's result is
    that the accrual component of earnings reverses while the cash component persists.
    The market prices earnings as if the two were the same.

    Ranked negatively - low accruals is the good end - and combined with a cash-flow
    yield so that the strategy is not simply buying whoever capitalises least.
    """

    name = "accrual_quality"
    requires_features = ("accruals", "cf_yield")
    min_date = FUNDAMENTAL_START
    construction = STANDARD

    def score(self, ctx: Context) -> np.ndarray:
        acc = self.f(ctx, "accruals")
        elig = require(ctx.tradable, acc)
        return blend([rank_pct(-acc, elig),
                      rank_pct(self.f(ctx, "cf_yield"), elig)],
                     weights=[0.7, 0.3], min_components=1)


@register("quality_value")
class QualityValue(FeatureStrategy):
    """Cheap and good at once - the combination, not either half.

    Value on its own buys companies that are cheap because they deserve to be. Quality on
    its own buys companies everybody already knows are excellent, at the price that
    implies. The interesting portfolio is the intersection, and it is small: the blend
    below requires at least three of five components so a name cannot qualify on one
    lucky ratio.

    Components: gross profitability (Novy-Marx's "other side of value"), return on
    equity, low leverage, book-to-market, earnings yield. All ranked, never z-scored -
    a single bad share count produces a book-to-market of 300 and a z-score would hand
    it the whole portfolio.
    """

    name = "quality_value"
    requires_features = ("gross_profitability", "roe", "leverage", "book_to_market",
                         "earnings_yield")
    min_date = FUNDAMENTAL_START
    construction = STANDARD

    def score(self, ctx: Context) -> np.ndarray:
        e = ctx.tradable
        return blend(
            [rank_pct(self.f(ctx, "gross_profitability"), e),
             rank_pct(self.f(ctx, "roe"), e),
             rank_pct(-self.f(ctx, "leverage"), e),
             rank_pct(self.f(ctx, "book_to_market"), e),
             rank_pct(self.f(ctx, "earnings_yield"), e)],
            weights=[0.2, 0.2, 0.15, 0.225, 0.225], min_components=3)


@register("restatement_averse")
class RestatementAverse(FeatureStrategy):
    """Own companies that report accurately and on time. Nothing else.

    This one exists because of a column most datasets throw away. `restatement_rate` is
    the share of a company's published facts that it has since revised, counted only from
    revisions that had already happened by the rebalance date; `filing_lag_days` is how
    long after the period closed the filing appeared. Neither can be computed from a
    single-vintage fundamentals feed at all, because a single vintage IS the restatement.

    The claim is that accounting reliability is priced too cheaply: a company that
    restates habitually and files late is telling you something about its controls, and
    that information is public, boring, and slow to be acted on.

    It is also the strategy most likely to be measuring the wrong thing, and it is worth
    saying so: a company with more subsidiaries files more facts and has more to revise,
    so part of this may be a size and complexity tilt wearing a governance costume. The
    honest test is whether it survives once `log_market_cap` is in the blend, which the
    genetic algorithm can settle better than an argument can.
    """

    name = "restatement_averse"
    requires_features = ("restatement_rate", "filing_lag_days")
    min_date = RESTATEMENT_START
    construction = STANDARD

    def score(self, ctx: Context) -> np.ndarray:
        rate = self.f(ctx, "restatement_rate")
        elig = require(ctx.tradable, rate)
        return blend([rank_pct(-rate, elig),
                      rank_pct(-self.f(ctx, "filing_lag_days"), elig)],
                     weights=[0.65, 0.35], min_components=1)


# --------------------------------------------------------------------------
# Market microstructure and index mechanics
# --------------------------------------------------------------------------

@register("illiquidity_carry")
class IlliquidityCarry(FeatureStrategy):
    """Get paid for holding what is hard to sell - and pay for it at every rebalance.

    Amihud's measure is the average absolute return per dollar traded: how far the price
    moves when someone needs to get out. The premium is well documented and this is the
    strategy in the file most exposed to being an artefact of its own cost model, because
    the thing being harvested is EXACTLY the thing the cost model charges for.

    That tension is the reason to include it. Under `optimistic` costs it should look
    good; under `pessimistic` it should not; and the gap between the two is a direct
    measurement of how much of the illiquidity premium a retail account can actually
    keep. No other strategy here produces that number.

    The bottom decile of dollar volume is excluded even so. Within the S&P 500 those are
    still large companies, but a strategy that has to buy $2,000 of something is not
    tested by whether it CAN - it is tested by whether it should.
    """

    name = "illiquidity_carry"
    requires_features = ("amihud_illiq", "log_dollar_volume")
    construction = Construction(top_k=30, weighting="equal", max_weight=0.06,
                                min_names=10)

    def score(self, ctx: Context) -> np.ndarray:
        elig = ctx.tradable
        liquid_enough = rank_pct(self.f(ctx, "log_dollar_volume"), elig) >= 0.10
        return rank_pct(self.f(ctx, "amihud_illiq"), elig & liquid_enough)


@register("index_entry_drift")
class IndexEntryDrift(FeatureStrategy):
    """Own whatever the index just added. A pure demand-shock trade.

    Index funds must buy a new constituent, and must buy it regardless of price. The
    classical estimates (Shleifer 1986; Harris & Gurel 1986) put the effect at 3-6%; the
    modern consensus is that it has been largely arbitraged away since around 2000. This
    project's window is entirely inside the modern era, so the expected result is
    somewhere between "small" and "nothing" - which is a perfectly good thing to measure,
    and better than citing a 1986 paper about a market that no longer exists.

    Two honest limitations. Membership here is monthly (ADR-004), so the pre-effective
    -date run-up - where most of the classical effect lives - is invisible and only the
    post-inclusion drift is tradable. And the portfolio is small and lumpy: roughly 20
    names join per year, so in a quiet stretch this holds a dozen positions and the
    result is as much about those dozen as about the effect.
    """

    name = "index_entry_drift"
    requires_features = ("months_in_index",)
    construction = Construction(top_k=30, weighting="equal", max_weight=0.10,
                                min_names=6)

    def __init__(self, window_months: float = 12.0, **kw):
        super().__init__(window_months=float(window_months), **kw)

    def score(self, ctx: Context) -> np.ndarray:
        tenure = self.f(ctx, "months_in_index")
        recent = ctx.tradable & np.isfinite(tenure) & (tenure <= self.window_months)
        # Newest first. Not ranked: tenure is already an interpretable ordering, and the
        # portfolio is small enough that a rank would just relabel it.
        return np.where(recent, -tenure, np.nan)


# --------------------------------------------------------------------------
# Preferences and payouts
# --------------------------------------------------------------------------

@register("lottery_averse")
class LotteryAverse(FeatureStrategy):
    """Buy the boring ones: low maximum daily return, low idiosyncratic volatility.

    Three measurements of one behavioural claim - that investors overpay for a small
    chance of a large gain, so stocks that LOOK like lottery tickets are systematically
    expensive. Bali, Cakici & Whitelaw (2011) used the maximum daily return of the past
    month; Ang et al. (2006) found idiosyncratic volatility is priced negatively, which
    no risk model predicts; positive return skewness is the same preference measured a
    third way.

    Related to `low_vol` and deliberately not the same: low_vol ranks on total
    volatility, which is mostly beta. This ranks on the shape of the tail. If they
    produce the same portfolio, that is a finding.
    """

    name = "lottery_averse"
    requires_features = ("max_ret_21d", "idio_vol_252d", "skew_252d")
    construction = Construction(top_k=50, weighting="inverse_vol", max_weight=0.05,
                                min_names=10)

    def score(self, ctx: Context) -> np.ndarray:
        e = ctx.tradable
        return blend([rank_pct(-self.f(ctx, "max_ret_21d"), e),
                      rank_pct(-self.f(ctx, "idio_vol_252d"), e),
                      rank_pct(-self.f(ctx, "skew_252d"), e)],
                     weights=[0.4, 0.4, 0.2], min_components=2)


@register("dividend_grower")
class DividendGrower(FeatureStrategy):
    """Own the raisers, refuse the cutters.

    A dividend is the one corporate statement that costs money to make. Raising it
    commits future cash; cutting it is an admission that management has run out of
    alternatives, and it is one of the most reliably negative corporate events there is.
    So the cut is a hard exclusion rather than a low score - a strategy that will buy a
    cutter at a sufficiently attractive yield is not the strategy this is trying to be.

    Built from discrete dividend events rather than from an adjusted-close column, which
    is the only way it can be built: adjustment dissolves the payments into the price and
    the event stops existing. Available from 2007 rather than 2010, so this is the one
    payout-based strategy with the full research window.
    """

    name = "dividend_grower"
    requires_features = ("div_growth_1y", "div_yield", "div_cut", "pays_dividend")
    construction = STANDARD

    def score(self, ctx: Context) -> np.ndarray:
        cut = self.f(ctx, "div_cut")
        pays = self.f(ctx, "pays_dividend")
        elig = ctx.tradable & (pays > 0) & ~(cut > 0)
        growth = self.f(ctx, "div_growth_1y")
        return blend([rank_pct(growth, require(elig, growth)),
                      rank_pct(self.f(ctx, "div_yield"), elig)],
                     weights=[0.6, 0.4], min_components=1)


# --------------------------------------------------------------------------
# Timing, and the kitchen sink
# --------------------------------------------------------------------------

@register("defensive_regime")
class DefensiveRegime(FeatureStrategy):
    """Change WHAT you own and HOW MUCH, according to the state of the market.

    Every other strategy in this file is always fully invested and only chooses between
    stocks. This one also chooses between being invested and not, which under a long-only
    mandate is the only risk control available (ADR-016). Two switches, both from macro
    features that are never revised:

        risk-on   the equal-weighted index is above its 200-day average
                  -> residual momentum, fully invested
        risk-off  it is below, or realised volatility is far above its own year
                  -> lowest-volatility names, and only `defensive_gross` of the account

    Momentum crashes are the motivation. 12-1 momentum lost more than half its value in
    2009 - not because the signal stopped working, but because after a crash the losers
    it was avoiding were the highest-beta names, and they rebounded hardest. That is a
    regime fact, and no cross-sectional score can see it.

    The obvious objection is that this is two free parameters fitted to a window that
    contains 2008 and 2020, and the objection is correct. Both are set to conventional
    values rather than searched ones, and the deflated Sharpe in the registry is the
    place to argue about it.
    """

    name = "defensive_regime"
    requires_features = ("resid_mom_12_1", "vol_126d", "mkt_trend_200d",
                         "mkt_vol_ratio")
    construction = STANDARD

    def __init__(self, defensive_gross: float = 0.5, vol_ratio_trigger: float = 1.5,
                 **kw):
        super().__init__(defensive_gross=float(defensive_gross),
                         vol_ratio_trigger=float(vol_ratio_trigger), **kw)

    def is_defensive(self, ctx: Context) -> bool:
        """One boolean per rebalance. Both inputs are trailing and lagged a session."""
        trend = _macro(self.f(ctx, "mkt_trend_200d"))
        vol_ratio = _macro(self.f(ctx, "mkt_vol_ratio"))
        return (np.isfinite(trend) and trend < 0.0) or (
            np.isfinite(vol_ratio) and vol_ratio > self.vol_ratio_trigger)

    def score(self, ctx: Context) -> np.ndarray:
        e = ctx.tradable
        if self.is_defensive(ctx):
            return rank_pct(-self.f(ctx, "vol_126d"), e)
        return rank_pct(self.f(ctx, "resid_mom_12_1"), e)

    def target_weights(self, ctx: Context) -> np.ndarray:
        from dataclasses import replace
        c = self.construction
        if self.is_defensive(ctx):
            c = replace(c, gross=self.defensive_gross)
        s = np.asarray(self.score(ctx), dtype=np.float64)
        return build_weights(s, self.eligible(ctx), c, tiebreak=ctx.tiebreak)


@register("multi_factor")
class MultiFactor(FeatureStrategy):
    """Value, quality, momentum and low risk, equally weighted. The bar to beat.

    Not an idea - a benchmark. Any clever thing built on this feature layer, evolved or
    hand-written, has to beat the obvious thing you get by averaging the four factor
    families with no weights and no tuning. Searched strategies routinely fail to, and
    when that happens the search found overfitting rather than signal.

    Equal weights across the four sleeves specifically because they were not chosen. The
    moment those weights are optimised this stops being a benchmark and becomes another
    trial, which is what the genetic algorithm is for.
    """

    name = "multi_factor"
    requires_features = ("book_to_market", "earnings_yield", "gross_profitability",
                         "roe", "accruals", "resid_mom_12_1", "mom_12_1",
                         "vol_126d", "idio_vol_252d")
    min_date = FUNDAMENTAL_START
    construction = STANDARD

    def score(self, ctx: Context) -> np.ndarray:
        e = ctx.tradable
        value = blend([rank_pct(self.f(ctx, "book_to_market"), e),
                       rank_pct(self.f(ctx, "earnings_yield"), e)])
        quality = blend([rank_pct(self.f(ctx, "gross_profitability"), e),
                         rank_pct(self.f(ctx, "roe"), e),
                         rank_pct(-self.f(ctx, "accruals"), e)])
        momentum = blend([rank_pct(self.f(ctx, "resid_mom_12_1"), e),
                          rank_pct(self.f(ctx, "mom_12_1"), e)])
        low_risk = blend([rank_pct(-self.f(ctx, "vol_126d"), e),
                          rank_pct(-self.f(ctx, "idio_vol_252d"), e)])
        return blend([value, quality, momentum, low_risk], min_components=3)
