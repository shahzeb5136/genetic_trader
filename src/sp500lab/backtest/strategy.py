"""The one interface every competitor implements.

    target_weights(ctx: Context) -> weights

That is the whole contract. A moving-average crossover, an evolved decision tree and a
trained neural net are all "a function from a point-in-time view to a set of weights",
and the engine cannot tell them apart. It should not be able to - the point of a
competition is that the scoreboard does not know who is playing.

Two levels of abstraction
-------------------------
`Strategy` is the raw protocol: return the weights yourself, however you like.

`SignalStrategy` is the level almost everything should use: produce a cross-sectional
**score** per security and let portfolio.py turn it into weights. Sharing the
construction means the competition measures signal quality rather than who tuned their
position sizing, and it makes the mandate constraints impossible to violate by
accident.

Returning weights vs returning a score
--------------------------------------
Return weights directly only when the portfolio construction IS the idea - a
risk-parity overlay, a benchmark replication, a fixed sleeve. If the idea is "these
stocks look better than those stocks", it is a score, and subclassing SignalStrategy
gets the caps, the top-k, the tie-break determinism and the long-only validation for
free.

Where the model families plug in
--------------------------------
    traditional   score() computes an indicator from ctx.window(...)
    genetic       params carries the genome; score() decodes and evaluates it
    neural        score() runs a forward pass over ctx.features

All three see the same Context, pay the same costs, and are scored by the same
metrics. See docs/BACKTEST.md for worked examples of each.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .context import Context
from .portfolio import Construction, build_weights

log = logging.getLogger(__name__)


@runtime_checkable
class Strategy(Protocol):
    """Anything the engine can backtest."""

    name: str

    def target_weights(self, ctx: Context) -> np.ndarray | pd.Series:
        """Target portfolio weights for the next holding period.

        Returns either an (S,) array aligned to `ctx.security_ids`, or a pandas
        Series indexed by security_id (missing names are treated as zero).

        Long-only mandate (ADR-016): every weight >= 0 and the sum <= 1.0. The engine
        rejects violations rather than normalising them silently - a strategy that
        returns a negative weight has a bug, not a short position.
        """
        ...


class BaseStrategy:
    """Optional base class: naming, parameters and a warmup guard.

    Not required - the engine accepts any object with `target_weights` and a `name`.
    Subclassing just saves boilerplate and makes a strategy self-describing in the
    experiment log.
    """

    name: str = "unnamed"

    #: Sessions of history needed before this strategy will trade. The engine skips
    #: rebalances before this, so a strategy never trades on a half-filled window.
    warmup: int = 0

    #: Earliest date this strategy's INPUTS exist. `warmup` counts sessions of price
    #: history; this is for data that simply did not exist before a date - XBRL
    #: fundamentals begin 2009-04 whatever the price panel contains. The engine moves
    #: `start` forward to it rather than letting the strategy sit in cash for three
    #: years and report the resulting flat stretch as part of its CAGR.
    min_date: str = ""

    #: Feature columns this strategy reads. When set, `run_backtest` loads the shared
    #: feature panel automatically and fails loudly if a name is missing, rather than
    #: handing the strategy a context with no features and letting it score NaN.
    requires_features: tuple[str, ...] = ()

    def __init__(self, **params: Any):
        self.params = dict(params)
        for k, v in params.items():
            setattr(self, k, v)

    def describe(self) -> dict:
        """Everything needed to reproduce this strategy, for the experiment log."""
        d = {"name": self.name, "class": type(self).__name__,
             "warmup": self.warmup, "params": _jsonable(self.params)}
        if self.requires_features:
            d["features"] = list(self.requires_features)
        if self.min_date:
            d["min_date"] = self.min_date
        return d

    def on_start(self, panel) -> None:
        """Hook called once before the first rebalance.

        Use it to precompute anything panel-wide - a neural net's whole prediction
        matrix, say. Anything computed here must still be point-in-time safe: it sees
        the full panel, so a rolling window computed here MUST be trailing-only.
        Prefer computing inside score() unless profiling says otherwise.
        """

    def target_weights(self, ctx: Context) -> np.ndarray:
        raise NotImplementedError


class SignalStrategy(BaseStrategy):
    """Score the cross-section; let the shared construction build the portfolio.

    Subclasses implement `score(ctx) -> (S,) array`, higher is better. NaN means "no
    opinion" and the name is passed over - which is not the same as scoring it zero.
    """

    construction: Construction = Construction()

    def __init__(self, construction: Construction | None = None, **params: Any):
        super().__init__(**params)
        if construction is not None:
            self.construction = construction

    def score(self, ctx: Context) -> np.ndarray:
        raise NotImplementedError

    def eligible(self, ctx: Context) -> np.ndarray:
        """Which names may be held. Override to add a strategy-specific filter."""
        return ctx.tradable

    def vol_for_weighting(self, ctx: Context, lookback: int = 63) -> np.ndarray:
        """(S,) trailing realised vol, for inverse-vol weighting."""
        w = ctx.window(lookback + 1)
        if len(w) < 3:
            return np.full(ctx.close.shape[1], np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = w[1:] / w[:-1] - 1.0
        # A name with no observations in the window has no volatility, not a
        # volatility of zero - NaN keeps it out of the portfolio instead of handing
        # it an infinite inverse-vol weight.
        n = np.isfinite(rets).sum(axis=0)
        out = np.full(rets.shape[1], np.nan)
        ok = n >= 2
        if ok.any():
            out[ok] = np.nanstd(rets[:, ok], axis=0) * np.sqrt(252.0)
        return out

    def target_weights(self, ctx: Context) -> np.ndarray:
        s = np.asarray(self.score(ctx), dtype=np.float64)
        if s.shape != (ctx.close.shape[1],):
            raise ValueError(
                f"{self.name}.score returned shape {s.shape}, expected "
                f"({ctx.close.shape[1]},) aligned to ctx.security_ids")
        vol = (self.vol_for_weighting(ctx)
               if self.construction.weighting == "inverse_vol" else None)
        return build_weights(s, self.eligible(ctx), self.construction,
                             tiebreak=ctx.tiebreak, vol=vol)

    def describe(self) -> dict:
        d = super().describe()
        d["construction"] = vars(self.construction).copy()
        return d


class FeatureStrategy(SignalStrategy):
    """A SignalStrategy whose inputs come from the shared feature panel.

    Subclasses declare `requires_features` and read them with `self.f(ctx, name)`. The
    point of the indirection is that no strategy computes an indicator: two strategies
    ranking on `mom_12_1` are ranking on the SAME numbers, so a difference in their
    results is a difference in their idea (see features/panel.py).
    """

    def f(self, ctx: Context, name: str) -> np.ndarray:
        """(S,) one feature column, aligned to `ctx.security_ids`."""
        return ctx.feature(name)

    def eligible(self, ctx: Context) -> np.ndarray:
        return ctx.tradable


class FunctionStrategy(BaseStrategy):
    """Wrap a bare function as a strategy. Handy for one-off experiments and tests."""

    def __init__(self, fn: Callable[[Context], np.ndarray], name: str = "fn", **params: Any):
        super().__init__(**params)
        self._fn = fn
        self.name = name

    def target_weights(self, ctx: Context) -> np.ndarray:
        return self._fn(ctx)


# --------------------------------------------------------------------------
# Registry - so the CLI, the GA and the experiment log can all name a strategy
# --------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[..., Strategy]] = {}


def register(name: str) -> Callable:
    """Decorator registering a strategy factory under a CLI-visible name."""
    def deco(obj):
        if name in _REGISTRY:
            raise ValueError(f"strategy {name!r} already registered")
        _REGISTRY[name] = obj
        if isinstance(obj, type) and getattr(obj, "name", "unnamed") == "unnamed":
            obj.name = name
        return obj
    return deco


def get_strategy(name: str, **kwargs: Any) -> Strategy:
    """Instantiate a registered strategy by name."""
    _load_builtin()
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(_REGISTRY)}")
    strat = _REGISTRY[name](**kwargs)
    if getattr(strat, "name", "unnamed") == "unnamed":
        strat.name = name
    return strat


def list_strategies() -> list[str]:
    _load_builtin()
    return sorted(_REGISTRY)


def _load_builtin() -> None:
    """Import the baseline strategies so they self-register. Idempotent."""
    from .. import strategies  # noqa: F401


def normalize_weights(w: np.ndarray | pd.Series, security_ids: np.ndarray) -> np.ndarray:
    """Accept either return shape from a strategy and produce an aligned (S,) array."""
    if isinstance(w, pd.Series):
        out = np.zeros(len(security_ids), dtype=np.float64)
        pos = {s: i for i, s in enumerate(security_ids.tolist())}
        unknown = [str(k) for k in w.index if k not in pos]
        if unknown:
            raise KeyError(
                f"strategy returned weights for {len(unknown)} security_id(s) not in the "
                f"panel, e.g. {unknown[:3]}. Weights must be indexed by security_id, "
                "not by ticker.")
        for k, v in w.items():
            out[pos[k]] = float(v)
        return out
    arr = np.asarray(w, dtype=np.float64)
    if arr.shape != (len(security_ids),):
        raise ValueError(f"strategy returned shape {arr.shape}, expected "
                         f"({len(security_ids)},) aligned to ctx.security_ids")
    return arr


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)
