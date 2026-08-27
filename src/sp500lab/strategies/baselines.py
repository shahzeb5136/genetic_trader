"""The baselines every competitor has to beat.

These are not filler. They are the null hypotheses, and they are chosen so that each
one kills a different way of fooling yourself:

    spy_buy_hold      Can you beat just owning the index? Most things cannot.
                      Also the engine's acceptance test - see accept.py.
    equal_weight      Equal-weighting the point-in-time universe is a real strategy
                      with a real premium (small-cap tilt + rebalancing). A stock
                      picker that loses to it is not picking stocks, it is
                      accidentally equal-weighting with extra steps.
    momentum_12_1     The single most robust published cross-sectional anomaly. If a
                      model cannot beat 12-1 momentum after costs, it found nothing.
    random_weight     Matched turnover, no signal. Anything a strategy earns above
                      this that is not signal is the cost model being too kind.
    low_vol           Trailing-vol-sorted. Cheap, and historically hard to beat
                      risk-adjusted, so it catches a strategy that only looks good
                      because it is low beta.

random_weight deserves its place. Run it a dozen times with different seeds and the
spread of outcomes IS the noise floor for the whole competition. A strategy inside that
spread has not demonstrated anything, no matter how good its Sharpe looks in isolation.
"""

from __future__ import annotations

import numpy as np

from ..backtest.context import Context
from ..backtest.portfolio import Construction
from ..backtest.strategy import BaseStrategy, SignalStrategy, register


@register("spy_buy_hold")
class BuyAndHoldSPY(BaseStrategy):
    """100% SPY, bought once and never touched.

    Implemented against the benchmark series rather than the security panel, because
    SPY is an ETF and was never an S&P 500 constituent - it is not in the universe and
    the engine would rightly refuse to let a strategy buy it.

    This is the engine's calibration instrument. Zero costs, it must reproduce SPY's
    real total return: 8.32%/yr over 2000-01-03..2026-08-26. See accept.py test 1.
    """

    name = "spy_buy_hold"

    def target_weights(self, ctx: Context) -> np.ndarray:
        raise NotImplementedError(
            "spy_buy_hold is evaluated by accept.replicate_benchmark(), not through "
            "the security panel - SPY is not an index constituent.")


@register("equal_weight")
class EqualWeight(SignalStrategy):
    """Equal weight across every tradable index member. No view at all."""

    name = "equal_weight"
    construction = Construction(top_k=None, weighting="equal", min_names=5)

    def score(self, ctx: Context) -> np.ndarray:
        return np.where(ctx.tradable, 1.0, np.nan)


@register("momentum_12_1")
class Momentum12_1(SignalStrategy):
    """Classic cross-sectional momentum: 12-month return, skipping the last month.

    The skip is not decoration. Short-horizon returns reverse, so including the most
    recent month systematically dilutes the signal. Jegadeesh & Titman (1993) and
    everything after it drop it.
    """

    name = "momentum_12_1"
    warmup = 273  # 252 sessions of lookback + the 21-session skip
    construction = Construction(top_k=50, weighting="equal", max_weight=0.05)

    def __init__(self, lookback: int = 252, skip: int = 21, **kw):
        super().__init__(lookback=lookback, skip=skip, **kw)
        self.warmup = lookback + skip

    def score(self, ctx: Context) -> np.ndarray:
        return ctx.trailing_return(self.lookback, skip=self.skip)


@register("low_vol")
class LowVolatility(SignalStrategy):
    """Lowest trailing realised volatility. Negated, because low is good here."""

    name = "low_vol"
    warmup = 130
    construction = Construction(top_k=50, weighting="inverse_vol", max_weight=0.05)

    def __init__(self, lookback: int = 126, **kw):
        super().__init__(lookback=lookback, **kw)
        self.warmup = lookback + 5

    def score(self, ctx: Context) -> np.ndarray:
        return -self.vol_for_weighting(ctx, lookback=self.lookback)


@register("random_weight")
class RandomWeight(SignalStrategy):
    """Random scores. The noise floor, and the only honest control in the whole suite.

    Uses `ctx.rng`, the engine's seeded generator, so a given seed reproduces exactly.
    Run several seeds: the spread across them is the width of the null distribution,
    and any strategy inside it has demonstrated nothing.
    """

    name = "random_weight"
    construction = Construction(top_k=50, weighting="equal", max_weight=0.05)

    def score(self, ctx: Context) -> np.ndarray:
        rng = ctx.rng if ctx.rng is not None else np.random.default_rng(0)
        s = rng.random(ctx.close.shape[1])
        return np.where(ctx.tradable, s, np.nan)


@register("short_reversal")
class ShortTermReversal(SignalStrategy):
    """Buy last month's losers. The counterweight to momentum at a one-month horizon.

    Included because it and momentum disagree by construction: a search that finds
    "buy recent winners" and a search that finds "buy recent losers" cannot both be
    picking up structure, and having both baselines makes that obvious.
    """

    name = "short_reversal"
    warmup = 25
    construction = Construction(top_k=50, weighting="equal", max_weight=0.05)

    def __init__(self, lookback: int = 21, **kw):
        super().__init__(lookback=lookback, **kw)
        self.warmup = lookback + 4

    def score(self, ctx: Context) -> np.ndarray:
        return -ctx.trailing_return(self.lookback, skip=0)


@register("cash")
class Cash(BaseStrategy):
    """Hold nothing. The floor: any strategy that loses to cash is worse than nothing.

    Also the degenerate case the accounting has to survive without dividing by zero.
    """

    name = "cash"

    def target_weights(self, ctx: Context) -> np.ndarray:
        return ctx.empty_weights()
