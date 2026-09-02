"""Your strategies go here. This file is yours; nothing else in the project writes to it.

Adding one is three things: a class, a `score`, and a name.

    @register("my_idea")                      # the name the CLI will know it by
    class MyIdea(FeatureStrategy):
        \"\"\"One sentence somebody could argue with.\"\"\"   # <- this becomes the report
        requires_features = ("gross_profitability",)      # what it reads
        construction = STANDARD                           # 50 names, equal, 5% cap

        def score(self, ctx):
            return rank_pct(self.f(ctx, "gross_profitability"), ctx.tradable)

Then:

    python -m sp500lab backtest run my_idea --all-costs
    python -m sp500lab backtest trades my_idea          # every order, as CSV
    python -m sp500lab report strategy my_idea --open   # the full page

Four rules, and they are the whole contract
--------------------------------------------
1. **Return a SCORE, not weights.** Higher is better. `portfolio.py` turns it into a
   portfolio - the top-k cut, the per-name cap, the long-only check and the unbiased
   tie-break are shared, so the scoreboard compares your idea against everyone else's
   rather than your position sizing against theirs.
2. **NaN means "no opinion", and it is not zero.** A name scored NaN is passed over; a
   name scored 0.0 is ranked last. Those are different statements and only one of them
   should be able to earn a position.
3. **Only score names in `ctx.tradable`.** The engine refuses an allocation to anything
   that was not in the index and priced that day - that refusal is the survivorship
   guard, and it will raise rather than quietly let it through.
4. **You cannot see the future, structurally.** `ctx.close` is a numpy view that
   physically ends at the as-of date, so `ctx.close[len(ctx.close)]` is an IndexError
   rather than tomorrow's price. There is nothing to remember not to do (ADR-017).

Where the numbers come from
---------------------------
`self.f(ctx, "name")` reads the shared feature layer - 75 point-in-time features,
documented in `docs/FEATURES.md` and listed by `sp500lab features list`. Declaring them
in `requires_features` is what makes the engine load the panel for you and fail loudly if
a name is wrong, instead of handing you a context with no features and letting every score
come back NaN.

You can also compute your own from `ctx.close`, `ctx.window(n)` and
`ctx.trailing_return(n, skip=k)`. Both are fine. The feature layer exists so that two
strategies ranking on momentum rank on the *same* momentum.

Making it show up everywhere
-----------------------------
`GROUPS` in `__init__.py` decides which strategies the suite and the report set include.
`my_first_idea` below is in the `custom` group and deliberately NOT in `all`, so it does
not join the main scoreboard until you put it there:

    python -m sp500lab backtest suite custom
    python -m sp500lab report all custom --open

Read `docs/ADDING_A_STRATEGY.md` for the longer version, and `strategies/alpha.py` for
twelve worked examples that are more interesting than the one below.
"""

from __future__ import annotations

import numpy as np

from ..backtest.context import Context
from ..backtest.portfolio import Construction
from ..backtest.strategy import FeatureStrategy, register
from .signals import blend, rank_pct

#: 50 names, equal weight, 5% cap. Copied from alpha.py so this file stands alone.
#: 50 because at $100k with a $1 per-order commission minimum, anything much wider is
#: paying for the privilege - `top_k` is an economic decision, not a knob (ADR-016).
STANDARD = Construction(top_k=50, weighting="equal", max_weight=0.05, min_names=10)


@register("my_first_idea")
class MyFirstIdea(FeatureStrategy):
    """Cheap, profitable and calm - equally weighted, with no tuning at all.

    A worked example that runs, rather than a stub that does not. It is also a real
    (weak) hypothesis: that the three factor families most people agree on do something
    when combined without anybody choosing how much of each.

    Delete it, or edit it into whatever you actually want to test. Nothing else in the
    project imports it.
    """

    name = "my_first_idea"

    #: Everything this reads. `sp500lab features list` shows all 75 with their coverage.
    requires_features = ("earnings_yield", "gross_profitability", "vol_126d")

    #: The earliest date these inputs exist. XBRL fundamentals begin 2009-04 and the
    #: derived figures need a few more quarters, so starting earlier would mean sitting
    #: in cash for three years and reporting the flat stretch as performance.
    min_date = "2010-07-01"

    construction = STANDARD

    def score(self, ctx: Context) -> np.ndarray:
        eligible = ctx.tradable
        return blend([
            rank_pct(self.f(ctx, "earnings_yield"), eligible),        # cheap
            rank_pct(self.f(ctx, "gross_profitability"), eligible),   # profitable
            rank_pct(-self.f(ctx, "vol_126d"), eligible),             # calm
        ], min_components=2)
