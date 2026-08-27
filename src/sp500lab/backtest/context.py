"""The point-in-time view a strategy is allowed to see.

This is the most important file in the engine. Everything else is accounting.

The design rule
---------------
Do not hand a strategy the full panel and trust it to filter by date. Trust is not a
mechanism. Build the slice in the engine, hand over only the slice, and keep no
reference to the rest - then a lookahead bug is not something you have to remember not
to write, it is something the object cannot express.

Concretely, `PanelView.close` is `panel.adj_close[:t + 1]`. numpy slicing returns a
VIEW, so this costs nothing and copies nothing, but the resulting array has exactly
t+1 rows. `view.close[t + 1]` is an IndexError, not tomorrow's price. The future is
not filtered out of the object; it was never in it.

The three ways a strategy could still cheat, and what stops each
----------------------------------------------------------------
1. Index past the end of the array         -> IndexError. The rows do not exist.
2. Ask for a specific date                 -> `price_on()` raises LookaheadError.
3. Reach through to the engine's panel     -> Context holds no reference to it.

Test 3 in accept.py exercises all three with a deliberately cheating strategy. If any
of them stops raising, the acceptance suite fails and every downstream number is void.

Cost of convenience
-------------------
`ctx.prices` materialises a pandas DataFrame and is roughly 1000x slower than the
array accessors. It exists so a traditional strategy can be written readably, and it
is memoised per Context. Do not touch it from a GA fitness function - use
`ctx.close`, `ctx.window()` and `ctx.latest()`, which are plain numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class LookaheadError(RuntimeError):
    """A strategy asked for data it could not have known on its as-of date.

    Never catch this to keep a backtest running. It means the result is invalid.
    """


class PanelView:
    """Bounded view of the panel: every array physically ends at the as-of session.

    Constructed only by the engine. Holds no reference to the parent panel, so there
    is nothing to reach through to.
    """

    __slots__ = ("t", "as_of", "dates", "security_ids", "tickers", "tiebreak",
                 "close", "open",
                 "raw_close", "cum_split", "dollar_volume", "half_spread",
                 "in_index", "has_price", "_date_pos")

    def __init__(self, panel, t: int):
        # Every one of these is a numpy view ending at row t inclusive. O(1), no copy.
        self.t = t
        self.as_of = str(panel.dates[t])
        self.dates = panel.dates[:t + 1]
        self.security_ids = panel.security_ids
        self.tickers = panel.tickers
        self.tiebreak = panel.tiebreak
        self.close = panel.adj_close[:t + 1]
        self.open = panel.adj_open[:t + 1]
        self.raw_close = panel.raw_close[:t + 1]
        self.cum_split = panel.cum_split[:t + 1]
        self.dollar_volume = panel.dollar_volume[:t + 1]
        self.half_spread = panel.half_spread[:t + 1]
        self.in_index = panel.in_index[:t + 1]
        self.has_price = panel.has_price[:t + 1]
        self._date_pos = panel._date_pos

    def date_index(self, date: str) -> int:
        """Row index of a date, refusing anything the as-of date could not know."""
        if date > self.as_of:
            raise LookaheadError(
                f"asked for {date!r} from a context as of {self.as_of!r}. "
                "The engine builds contexts so this cannot happen by accident - "
                "a strategy hitting this has a lookahead bug.")
        try:
            return self._date_pos[date]
        except KeyError:
            raise KeyError(f"{date!r} is not a trading session") from None


@dataclass(frozen=True)
class Context:
    """What a strategy sees at one rebalance date.

    `target_weights(ctx)` is the entire strategy interface. A genetic algorithm, a
    neural net and a moving-average crossover all receive this same object and all
    return the same thing: non-negative weights that sum to at most 1.

    Attributes
    ----------
    as_of        rebalance date - a real NYSE session, always a month end
    t            row index of `as_of` in the panel (also `len(ctx.dates) - 1`)
    view         bounded price arrays; see PanelView
    universe     (S,) bool - index members on `as_of` (survivorship-free)
    tradable     (S,) bool - universe AND priced AND above the liquidity floor
    positions    (S,) float - adjusted shares currently held
    cash         uninvested cash
    nav          cash + marked-to-market positions, at the close of `as_of`
    features     optional (S, F) float - point-in-time features, knowable on `as_of`
    feature_names names of the feature columns
    params       strategy parameters (a GA genome lands here)
    rng          seeded generator; use this, never np.random, or runs stop reproducing
    """

    as_of: str
    t: int
    view: PanelView = field(repr=False)
    universe: np.ndarray = field(repr=False)
    tradable: np.ndarray = field(repr=False)
    positions: np.ndarray = field(repr=False)
    cash: float
    nav: float
    features: np.ndarray | None = field(default=None, repr=False)
    feature_names: tuple[str, ...] = ()
    params: dict = field(default_factory=dict, repr=False)
    rng: np.random.Generator | None = field(default=None, repr=False)

    # -------------------------------------------------------- fast accessors

    @property
    def security_ids(self) -> np.ndarray:
        return self.view.security_ids

    @property
    def tickers(self) -> np.ndarray:
        return self.view.tickers

    @property
    def tiebreak(self) -> np.ndarray:
        """(S,) stable ordering keys for breaking ranking ties without bias.

        Never rank on security_id - it correlates with survival. See portfolio.py.
        """
        return self.view.tiebreak

    @property
    def dates(self) -> np.ndarray:
        """Sessions up to and including as_of. `dates[-1] == as_of` by construction."""
        return self.view.dates

    @property
    def close(self) -> np.ndarray:
        """(t+1, S) adjusted closes. The last row is `as_of`."""
        return self.view.close

    def latest(self, matrix: np.ndarray | None = None) -> np.ndarray:
        """(S,) the as-of row of a matrix - closes by default."""
        m = self.view.close if matrix is None else matrix
        return m[-1]

    def window(self, lookback: int, matrix: np.ndarray | None = None) -> np.ndarray:
        """(lookback, S) the most recent `lookback` sessions, ending at as_of.

        Returns fewer rows near the start of the panel rather than padding, so a
        strategy can check `len(w) < lookback` and abstain instead of trading noise.
        """
        if lookback <= 0:
            raise ValueError("lookback must be positive")
        m = self.view.close if matrix is None else matrix
        return m[-lookback:]

    def price_on(self, date: str, field_: str = "close") -> np.ndarray:
        """(S,) prices on a specific past session. Raises LookaheadError if future."""
        i = self.view.date_index(date)
        return getattr(self.view, field_)[i]

    def trailing_return(self, lookback: int, skip: int = 0) -> np.ndarray:
        """(S,) total return over `lookback` sessions, ending `skip` sessions ago.

        `skip=21` gives the classic 12-1 momentum construction: it drops the most
        recent month to sidestep short-term reversal.
        """
        c = self.view.close
        end = len(c) - 1 - skip
        start = end - lookback
        if start < 0:
            return np.full(c.shape[1], np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            return c[end] / c[start] - 1.0

    def feature(self, name: str) -> np.ndarray:
        """(S,) one feature column by name."""
        if self.features is None:
            raise KeyError("this context carries no features; pass a FeaturePanel "
                           "to run_backtest(features=...)")
        try:
            return self.features[:, self.feature_names.index(name)]
        except ValueError:
            raise KeyError(f"unknown feature {name!r}; have {self.feature_names}") from None

    # ---------------------------------------------------- convenience (slow)

    @property
    def prices(self) -> pd.DataFrame:
        """Adjusted closes as a DataFrame, dates x security_id, ending at as_of.

        Convenience only. This allocates; the array accessors do not. Never call it
        inside a fitness evaluation.
        """
        cached = getattr(self, "_prices_df", None)
        if cached is None:
            cached = pd.DataFrame(self.view.close, index=self.view.dates,
                                  columns=self.view.security_ids)
            object.__setattr__(self, "_prices_df", cached)
        return cached

    @property
    def universe_ids(self) -> np.ndarray:
        """(n,) security_ids of index members on as_of."""
        return self.view.security_ids[self.universe]

    def weights_series(self, weights: np.ndarray) -> pd.Series:
        """Turn an (S,) weight vector into the Series the Strategy protocol returns."""
        nz = weights != 0
        return pd.Series(weights[nz], index=self.view.security_ids[nz], name=self.as_of)

    def empty_weights(self) -> np.ndarray:
        """(S,) zeros - the right starting point for building a weight vector."""
        return np.zeros(self.view.security_ids.shape[0], dtype=np.float64)
