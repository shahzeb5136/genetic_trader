"""The calendar lab: WHEN to hold the market, at daily granularity.

Every strategy in `backtest/` chooses WHICH stocks to hold and rebalances monthly.
This package tests a different family of claims entirely - that returns are not
uniform across the clock and the calendar. The overnight session is not the trading
day, Friday's close is not Monday's open, and the turn of the month is not the middle.
Each claim here is a standing published anomaly, implemented as a fixed rule with no
fitted parameters, on the most liquid instrument on earth.

The machinery is deliberately separate from the monthly engine because the engine's
central invariant - signals at the close fill at the NEXT open (ADR-017/BACKTEST) -
exists to stop a signal trading on information from its own fill. Calendar rules have
no signal: the schedule was knowable years in advance, so filling AT the close is not
lookahead, it is the strategy. A rule that does condition on data (the VIX variant)
conditions only on the PRIOR session's print, and says so.

What is shared is everything that keeps the numbers honest: the same adjustment chain
as the benchmark (`normalize/adjustments.compute_factors`), the same cost model
(`backtest/costs.CostModel`), the same metrics, the same experiment registry and
holdout ledger, and the same three cost settings on every result. See docs/TIMING.md
and ADR-036.
"""

from .data import TimingData, load_timing_data
from .engine import run_timing_backtest, timing_accept
from .strategies import TIMING_GROUPS, get_timing_strategy, list_timing_strategies

__all__ = [
    "TimingData", "load_timing_data",
    "run_timing_backtest", "timing_accept",
    "TIMING_GROUPS", "get_timing_strategy", "list_timing_strategies",
]
