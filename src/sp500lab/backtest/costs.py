"""What a trade actually costs, under the ADR-016 mandate.

Turnover is where backtests lie. A strategy that looks excellent gross and is a
disaster net is the normal case, not the pathological one, and a genetic algorithm
will find those faster than anything else because nothing in an unconstrained search
penalises trading. So costs are charged by the engine, not by the strategy, and every
result reports three of them.

What applies at sub-$100k, long-only, monthly (ADR-016)
-------------------------------------------------------
    commission      per-share with a minimum and a percentage cap - IBKR-shaped
    half-spread     estimated, because we have no quotes (see spreads.py)
    market impact   OMITTED. At this size on S&P 500 large caps it is noise.
    borrow cost     OMITTED. Long-only.

Dropping impact and borrow is a consequence of the mandate, not laziness, and each
omission is only valid while the mandate holds. Raise the capital and impact stops
being negligible; allow shorts and borrow stops being zero. Both are recorded here so
that a future change to the mandate lands on a comment explaining what breaks.

The commission detail that dominates at this size
--------------------------------------------------
IBKR fixed pricing is $0.005/share, minimum $1.00 per order, capped at 1% of trade
value. In a $100k portfolio holding 50 names, a full rebalance of one name is a ~$2k
trade of maybe 40 shares - $0.20 of per-share commission, so **the $1 minimum binds
almost every time**. That makes commission roughly a flat $1 per name traded, which is
about 5bp on a $2k trade and rises sharply as the portfolio gets more concentrated or
smaller. A strategy holding 200 names at $100k is paying 20bp a side in minimums
alone. This is the main reason `top_k` is a real decision and not a detail.

Because the minimum binds, the per-share rate rarely matters - but it is still
computed from as-traded share counts via `cum_split`, since our stored close is
split-adjusted (ADR-007) and would otherwise inflate share counts by the cumulative
split ratio.

Three settings, always all three
--------------------------------
    optimistic    commission only
    realistic     commission + 1x estimated half-spread
    pessimistic   commission + 2x estimated half-spread

The spread estimate is the weakest number in the chain, so pessimistic doubles it
rather than adding a different term. A strategy that survives only under `optimistic`
is not a strategy - it is a bet that the spread estimator is wrong in your favour.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .spreads import FALLBACK_HALF_SPREAD

log = logging.getLogger(__name__)

SETTINGS = ("optimistic", "realistic", "pessimistic")


@dataclass(frozen=True)
class CostModel:
    """Commission plus an estimated half-spread. No impact, no borrow (ADR-016).

    per_share            commission per share traded
    min_commission       per-order minimum - binds at retail size, see module docstring
    max_commission_pct   commission cap as a fraction of trade value
    spread_multiple      0 = ignore the spread, 1 = cross it once, 2 = pessimistic
    fallback_half_spread used where the estimator has no value; usage is counted
    fixed_bps            optional flat cost in bp of traded notional, for sensitivity
                         runs against a simple assumption
    """

    name: str = "realistic"
    per_share: float = 0.005
    min_commission: float = 1.00
    max_commission_pct: float = 0.01
    spread_multiple: float = 1.0
    fallback_half_spread: float = FALLBACK_HALF_SPREAD
    fixed_bps: float = 0.0

    def describe(self) -> dict:
        return vars(self).copy()

    def charge(
        self,
        traded_notional: np.ndarray,
        as_traded_price: np.ndarray,
        half_spread: np.ndarray,
    ) -> "CostBreakdown":
        """Cost of one rebalance.

        traded_notional  (S,) absolute value traded per security, in dollars
        as_traded_price  (S,) the price that actually changed hands, for share counts
        half_spread      (S,) proportional half-spread estimate; NaN -> fallback
        """
        traded = np.abs(np.asarray(traded_notional, dtype=np.float64))
        active = traded > 0
        if not active.any():
            return CostBreakdown()

        px = np.asarray(as_traded_price, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            shares = np.where(active & np.isfinite(px) & (px > 0), traded / px, 0.0)

        by_rate = shares * self.per_share
        floored = np.maximum(by_rate, self.min_commission)
        cap = traded * self.max_commission_pct
        per_order = np.minimum(floored, cap) if self.max_commission_pct > 0 else floored
        commission = float(per_order[active].sum())

        # Which of the three terms actually set the price. At retail size the minimum
        # dominates, and on very small drift trades the 1% cap dominates that - a
        # portfolio whose costs are mostly minimums is telling you it holds too many
        # names for its capital, which is a portfolio-construction result, not a
        # cost-model detail.
        at_min = active & (floored > by_rate) & (per_order >= self.min_commission - 1e-12)
        at_cap = active & (cap < floored)

        hs = np.asarray(half_spread, dtype=np.float64)
        missing = active & ~np.isfinite(hs)
        hs = np.where(np.isfinite(hs), hs, self.fallback_half_spread)
        spread = float((traded * hs).sum() * self.spread_multiple)

        fixed = float(traded.sum() * self.fixed_bps * 1e-4)

        return CostBreakdown(
            commission=commission, spread=spread, fixed=fixed,
            traded_notional=float(traded.sum()),
            n_orders=int(active.sum()),
            n_spread_fallback=int(missing.sum()),
            n_min_commission=int(at_min.sum()),
            n_cap_commission=int(at_cap.sum()),
        )


@dataclass
class CostBreakdown:
    """Where the money went at one rebalance. Summed across a run for the report."""

    commission: float = 0.0
    spread: float = 0.0
    fixed: float = 0.0
    traded_notional: float = 0.0
    n_orders: int = 0
    n_spread_fallback: int = 0
    n_min_commission: int = 0
    n_cap_commission: int = 0

    @property
    def total(self) -> float:
        return self.commission + self.spread + self.fixed

    def __iadd__(self, other: "CostBreakdown") -> "CostBreakdown":
        self.commission += other.commission
        self.spread += other.spread
        self.fixed += other.fixed
        self.traded_notional += other.traded_notional
        self.n_orders += other.n_orders
        self.n_spread_fallback += other.n_spread_fallback
        self.n_min_commission += other.n_min_commission
        self.n_cap_commission += other.n_cap_commission
        return self

    def as_dict(self) -> dict:
        d = vars(self).copy()
        d["total"] = self.total
        d["bps_of_traded"] = (round(self.total / self.traded_notional * 1e4, 2)
                              if self.traded_notional > 0 else 0.0)
        return d


# --------------------------------------------------------------------------
# The three named settings
# --------------------------------------------------------------------------

OPTIMISTIC = CostModel(name="optimistic", spread_multiple=0.0)
REALISTIC = CostModel(name="realistic", spread_multiple=1.0)
PESSIMISTIC = CostModel(name="pessimistic", spread_multiple=2.0)

#: For acceptance tests and gross-of-cost identities only. Never report a headline
#: number from this: a frictionless backtest is a description of a market that does
#: not exist.
FREE = CostModel(name="free", per_share=0.0, min_commission=0.0,
                 max_commission_pct=0.0, spread_multiple=0.0,
                 fallback_half_spread=0.0)

_BY_NAME = {m.name: m for m in (OPTIMISTIC, REALISTIC, PESSIMISTIC, FREE)}


def get_cost_model(name: str | CostModel) -> CostModel:
    if isinstance(name, CostModel):
        return name
    if name not in _BY_NAME:
        raise KeyError(f"unknown cost model {name!r}; available: {sorted(_BY_NAME)}")
    return _BY_NAME[name]


def all_settings() -> list[CostModel]:
    """The three that must always be reported together."""
    return [OPTIMISTIC, REALISTIC, PESSIMISTIC]
