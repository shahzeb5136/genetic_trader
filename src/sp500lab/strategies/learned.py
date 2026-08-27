"""Model-driven strategies — the neural-network path, demonstrated end to end.

Why a ridge regression and not a neural net
--------------------------------------------
The thing that is hard about putting a learned model into a backtest is not the model.
It is keeping the *training* point-in-time honest: a model fitted on the whole history
and then backtested over that same history has seen every label it is being asked to
predict, and it will look extraordinary. That failure is silent, it survives any amount
of validation-set discipline applied at the wrong level, and it is the single most
common way machine-learning backtests are wrong.

So this module solves that problem rather than the modelling one. `RollingRidge` refits
at every rebalance on a trailing window, using only data reachable through `ctx` — which
ends at the as-of date by construction (ADR-017). Swap the ridge solve for a forward pass
and the discipline is unchanged. That is the template.

It is also a working strategy, not a stub, so the pattern is testable rather than
aspirational. Ridge specifically because it needs no dependency beyond numpy, trains in
microseconds, and cannot quietly memorise the training set the way an over-parameterised
model can — which keeps the demonstration about leakage rather than about overfitting.

How the training set is built
------------------------------
At rebalance date T with a 21-session horizon and W training points:

    for each offset j in 1..W:
        t_j    = T - j * 21          # a past rebalance-like date
        X_j    = features at t_j     # all data <= t_j
        y_j    = return from t_j to t_j + 21

The last training label ends at `T - 21`, so **no label reaches the as-of date**. That
one line is the whole point of the module. Getting it wrong by a single horizon is
enough to produce a strategy that beats everything and works on nothing.

What is deliberately not here
------------------------------
Features come from the price panel, because `data/gold/` has no feature layer yet
(TODO-4). When it does, `_features()` becomes `ctx.features` and this file gets shorter.
The engine already accepts a feature panel and slices it by knowledge date; the plumbing
is waiting.
"""

from __future__ import annotations

import numpy as np

from ..backtest.context import Context
from ..backtest.portfolio import Construction
from ..backtest.strategy import SignalStrategy, register
from .evolvable import _trend

#: Sessions between rebalances. Matches the engine's monthly schedule (ADR-016) so a
#: training label spans exactly one holding period - the thing being predicted.
HORIZON = 21


def _zscore(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score over eligible names. See evolvable._zscore."""
    out = np.zeros_like(x)
    vals = x[mask]
    good = np.isfinite(vals)
    if good.sum() < 3:
        return out
    mu, sd = float(vals[good].mean()), float(vals[good].std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return out
    z = np.zeros_like(x)
    z[mask] = np.nan_to_num((x[mask] - mu) / sd, nan=0.0)
    return np.clip(z, -4.0, 4.0)   # winsorise; one bad print should not steer a fit


@register("rolling_ridge")
class RollingRidge(SignalStrategy):
    """Cross-sectional ridge regression, refit on a trailing window at every rebalance.

    Predicts each name's next-month return from four standardised price features, then
    hands the predictions to the shared portfolio construction like any other score.

    Parameters
    ----------
    train_points  past observation dates in the training window (default 36 = 3 years)
    alpha         ridge penalty; the intercept is never penalised
    horizon       label horizon in sessions; must not exceed the rebalance interval
    """

    name = "rolling_ridge"
    construction = Construction(top_k=50, weighting="equal", max_weight=0.05)

    #: Enough history for the longest feature (252) plus the whole training window plus
    #: one horizon. Set in __init__ because it depends on train_points.
    warmup = 252 + 36 * HORIZON + HORIZON

    FEATURES = ("mom_12_1", "reversal_1m", "vol_6m", "trend_200d")

    def __init__(self, train_points: int = 36, alpha: float = 10.0,
                 horizon: int = HORIZON, **kw):
        super().__init__(train_points=int(train_points), alpha=float(alpha),
                         horizon=int(horizon), **kw)
        if self.horizon > HORIZON:
            raise ValueError(
                f"horizon {self.horizon} exceeds the {HORIZON}-session rebalance "
                "interval; a label would then extend past the next decision date")
        self.warmup = 252 + self.train_points * self.horizon + self.horizon + 5
        # Features at panel row `at` depend only on `close[:at + 1]` and on the
        # membership mask at `at`, both of which are fixed for the whole run. So they
        # can be memoised across rebalances - and they overlap heavily, since
        # consecutive rebalances draw training points from an almost identical set of
        # rows. This is a ~10x speedup and it cannot leak: the cache key is the row,
        # and the computation it caches never reads past that row.
        self._feature_cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------- features

    def _features(self, close: np.ndarray, at: int, elig: np.ndarray) -> np.ndarray:
        """(S, F) standardised features computed from `close[:at + 1]` only.

        `at` indexes into the bounded view, so this cannot reach past the as-of date
        even when called for a historical training point.
        """
        cached = self._feature_cache.get(at)
        if cached is not None:
            return cached

        def ret(lb: int, skip: int = 0) -> np.ndarray:
            end, start = at - skip, at - skip - lb
            if start < 0:
                return np.full(close.shape[1], np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                return close[end] / close[start] - 1.0

        trend = _trend(close, at, 200)

        vol_win = close[max(0, at - 126):at + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = vol_win[1:] / vol_win[:-1] - 1.0
        n = np.isfinite(rets).sum(axis=0)
        vol = np.full(close.shape[1], np.nan)
        ok = n >= 2
        if ok.any():
            vol[ok] = np.nanstd(rets[:, ok], axis=0)

        cols = [ret(252, skip=21), -ret(21), -vol, trend]
        out = np.column_stack([_zscore(c, elig) for c in cols])
        self._feature_cache[at] = out
        return out

    # ------------------------------------------------------------- training

    def _fit(self, ctx: Context) -> np.ndarray | None:
        """Ridge coefficients from a trailing window, or None if there is too little.

        Every training label ends at or before `T - horizon`, so no label reaches the
        as-of date. That bound is the entire correctness argument of this class.
        """
        close = ctx.close                     # (t+1, S), last row IS as_of
        last = len(close) - 1
        h = self.horizon
        X_parts, y_parts = [], []

        for j in range(1, self.train_points + 1):
            t_j = last - j * h                # feature date
            label_end = t_j + h               # <= last - h. Never reaches as_of.
            if t_j - 252 - 21 < 0:
                break
            elig = ctx.view.in_index[t_j] & ctx.view.has_price[t_j]
            if elig.sum() < 20:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                y = close[label_end] / close[t_j] - 1.0
            X = self._features(close, t_j, elig)
            keep = elig & np.isfinite(y) & np.isfinite(X).all(axis=1)
            if keep.sum() < 20:
                continue
            X_parts.append(X[keep])
            # Standardise the label within its own date too, or a training point drawn
            # from March 2009 dominates the fit purely through market-wide magnitude.
            yk = y[keep]
            sd = float(yk.std(ddof=1))
            y_parts.append((yk - yk.mean()) / sd if sd > 0 else yk * 0.0)

        if len(X_parts) < 6:
            return None
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        X = np.column_stack([np.ones(len(X)), X])

        pen = np.eye(X.shape[1]) * self.alpha
        pen[0, 0] = 0.0                       # never penalise the intercept
        try:
            return np.linalg.solve(X.T @ X + pen, X.T @ y)
        except np.linalg.LinAlgError:
            return None

    # -------------------------------------------------------------- scoring

    def score(self, ctx: Context) -> np.ndarray:
        elig = ctx.tradable
        if elig.sum() < 20:
            return np.full(ctx.close.shape[1], np.nan)

        beta = self._fit(ctx)
        if beta is None:
            return np.full(ctx.close.shape[1], np.nan)

        X = self._features(ctx.close, len(ctx.close) - 1, elig)
        pred = beta[0] + X @ beta[1:]
        return np.where(elig & np.isfinite(pred), pred, np.nan)

    def describe(self) -> dict:
        d = super().describe()
        d["features"] = list(self.FEATURES)
        d["note"] = ("refit at every rebalance on a trailing window; no training label "
                     "reaches the as-of date")
        return d
