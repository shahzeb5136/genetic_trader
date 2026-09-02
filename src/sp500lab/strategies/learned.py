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


@register("shallow_mlp")
class ShallowMLP(SignalStrategy):
    """A deliberately small neural net over the shared feature layer, seed-ensembled.

    Gu, Kelly & Xiu (2020) is the reference, and its headline finding dictates the
    architecture: on cross-sectional equity data, two or three hidden layers performed
    best and deeper networks performed WORSE - the gains come from allowing feature
    interactions, not from depth. So: dense features -> 32 -> 16 -> 1, ReLU, and an
    ensemble over independent seeds, which they found matters more than the
    architecture. Pure numpy with Adam; at this size a framework would be a dependency
    without a benefit.

    The training discipline is RollingRidge's, inherited whole: refit on a trailing
    window of past rebalance dates, every label ends at least one horizon before the
    as-of date, and features come from the point-in-time panel rows - which by
    construction contain only what was knowable on their own date. The one addition is
    a hard assertion that no training row's index reaches the as-of row, because this
    model reads the shared feature panel directly and the guard should be structural,
    not remembered.

    What it reads adapts to what exists: at each refit it keeps the features that are
    at least 90% populated over its own training window and actually vary across the
    cross-section. That drops the sparse fundamentals (imputing a 30%-populated column
    invents 70% of a feature) and the macro columns (constant within a date, so a
    cross-sectional z-score sends them to zero), and it means the 2008 model runs on
    price features while the 2015 model also sees fundamentals - using what was
    available, which is what a live model would have done.
    """

    name = "shallow_mlp"
    construction = Construction(top_k=50, weighting="equal", max_weight=0.05)
    warmup = 252 + 36 * HORIZON + HORIZON + 5

    #: Feature-panel hygiene thresholds. Conventional, not searched.
    MIN_COVERAGE = 0.90
    HIDDEN = (32, 16)

    def __init__(self, train_points: int = 36, horizon: int = HORIZON,
                 n_seeds: int = 3, epochs: int = 40, lr: float = 1e-3,
                 l2: float = 1e-4, refit_sessions: int = 252, revision: int = 2,
                 **kw):
        # `revision` exists so a semantic fix lands on a NEW fingerprint and a new
        # seal_id. revision 1 had a stale-state bug: the fitted nets survived across
        # runs of one instance, so a research leg re-run from 2007 could score on nets
        # trained years later. Its sealed prediction is a measurement of that bug, and
        # the honest way to retire an immutable seal is to stop matching it.
        super().__init__(train_points=int(train_points), horizon=int(horizon),
                         n_seeds=int(n_seeds), epochs=int(epochs), lr=float(lr),
                         l2=float(l2), refit_sessions=int(refit_sessions),
                         revision=int(revision), **kw)
        if self.horizon > HORIZON:
            raise ValueError("a label longer than the rebalance interval would extend "
                             "past the next decision date")
        self.warmup = 252 + self.train_points * self.horizon + self.horizon + 5
        self._reset_state()

    def _reset_state(self) -> None:
        self._fp = None                 # the shared FeaturePanel, loaded in on_start
        self._nets: list[dict] | None = None
        self._cols: np.ndarray | None = None
        self._fit_row = -10 ** 9

    def on_start(self, panel) -> None:
        # The full feature panel, not the engine's per-date slice: training needs PAST
        # rows. Each stored row contains only what was knowable on its own date, so the
        # only possible leak is reading a row past the as-of one - _train_rows guards
        # that with an assertion rather than a convention.
        #
        # State is reset here, unconditionally. A strategy instance can be run more
        # than once (the forward harness runs research and forward legs, three cost
        # settings each, on one instance), and a net fitted in an earlier run is
        # training on the future relative to a later run's early rebalances. That is
        # not a hypothetical: revision 1's forward record shows what it does.
        from ..features import build_features
        self._reset_state()
        self._fp = build_features(panel=panel)

    # ------------------------------------------------------------- training

    def _train_rows(self, t: int) -> np.ndarray:
        """Panel rows of past rebalances whose labels are fully knowable at `t`."""
        rows = self._fp.rows
        ok = rows + self.horizon <= t
        picked = rows[ok][-self.train_points:]
        assert not len(picked) or picked.max() + self.horizon <= t, \
            "a training label would reach past the as-of date"
        return picked

    def _dense_columns(self, X: np.ndarray, elig: np.ndarray) -> np.ndarray:
        """Indices of features populated and cross-sectionally alive over the window.

        `X` is (rows, S, F); eligibility masks which (row, security) cells count. A
        feature must have a value for MIN_COVERAGE of eligible cells and must vary
        within a date - a macro column is constant across the cross-section, carries
        nothing a cross-sectional z-score can keep, and is dropped here rather than
        being carried as 25 columns of zero.
        """
        cells = np.maximum(elig.sum(), 1)
        keep = []
        for f in range(X.shape[-1]):
            col = X[..., f]
            if np.isfinite(col)[elig].sum() / cells < self.MIN_COVERAGE:
                continue
            stds = []
            for i in range(col.shape[0]):
                vals = col[i][elig[i]]
                vals = vals[np.isfinite(vals)]
                if len(vals) >= 10:
                    stds.append(float(vals.std()))
            if stds and float(np.median(stds)) > 1e-12:
                keep.append(f)
        return np.asarray(keep, dtype=np.int64)

    def _fit(self, ctx: Context) -> None:
        rows = self._train_rows(ctx.t)
        if len(rows) < 12:
            self._nets = None
            return
        close = ctx.close
        feats = np.stack([self._fp.at(int(r)) for r in rows])        # (R, S, F)
        elig = np.stack([ctx.view.in_index[r] & ctx.view.has_price[r] for r in rows])

        cols = self._dense_columns(feats, elig)
        if len(cols) < 4:
            self._nets = None
            return

        X_parts, y_parts = [], []
        for i, r in enumerate(rows):
            with np.errstate(divide="ignore", invalid="ignore"):
                y = close[r + self.horizon] / close[r] - 1.0
            Z = np.column_stack([_zscore(feats[i, :, f], elig[i]) for f in cols])
            keep = elig[i] & np.isfinite(y)
            if keep.sum() < 20:
                continue
            yk = y[keep]
            sd = float(yk.std(ddof=1))
            X_parts.append(np.nan_to_num(Z[keep], nan=0.0))
            y_parts.append((yk - yk.mean()) / sd if sd > 0 else yk * 0.0)

        if len(X_parts) < 6:
            self._nets = None
            return
        X = np.vstack(X_parts).astype(np.float64)
        y = np.concatenate(y_parts).astype(np.float64)
        self._cols = cols
        self._nets = [_train_mlp(X, y, self.HIDDEN, seed=s, epochs=self.epochs,
                                 lr=self.lr, l2=self.l2)
                      for s in range(self.n_seeds)]
        self._fit_row = ctx.t

    # -------------------------------------------------------------- scoring

    def score(self, ctx: Context) -> np.ndarray:
        S = ctx.close.shape[1]
        if self._fp is None:
            return np.full(S, np.nan)
        # Refit when the model is stale - and also if time ever runs BACKWARD relative
        # to the last fit, which means this instance is being driven by a new run and
        # the held nets know this run's future. Belt to on_start's braces.
        if ctx.t < self._fit_row or ctx.t - self._fit_row >= self.refit_sessions:
            self._fit(ctx)
        if not self._nets:
            return np.full(S, np.nan)

        elig = ctx.tradable
        feats = self._fp.at(ctx.t)
        Z = np.nan_to_num(np.column_stack(
            [_zscore(feats[:, f], elig) for f in self._cols]), nan=0.0)
        pred = np.mean([_forward(net, Z) for net in self._nets], axis=0)
        return np.where(elig & np.isfinite(pred), pred, np.nan)

    def describe(self) -> dict:
        d = super().describe()
        d["hidden"] = list(self.HIDDEN)
        d["note"] = ("seed-ensembled 2-layer MLP on the >=90%-populated features; "
                     "refit yearly on a trailing window; no training label reaches "
                     "the as-of date")
        return d


# --------------------------------------------------------------------------
# The numpy MLP - initialisation, Adam, forward pass. ~60 lines, no framework.
# --------------------------------------------------------------------------

def _train_mlp(X: np.ndarray, y: np.ndarray, hidden: tuple[int, ...], *, seed: int,
               epochs: int, lr: float, l2: float, batch: int = 4096) -> dict:
    """Fit the small net with Adam and return its weights.

    Mini-batch order reshuffles per epoch from a seeded generator, so a given seed
    reproduces exactly and different seeds give genuinely different nets - the ensemble
    the strategy averages over.
    """
    rng = np.random.default_rng(seed)
    sizes = (X.shape[1],) + tuple(hidden) + (1,)
    params = {}
    for i in range(len(sizes) - 1):
        params[f"W{i}"] = rng.normal(0.0, np.sqrt(2.0 / sizes[i]),
                                     size=(sizes[i], sizes[i + 1]))
        params[f"b{i}"] = np.zeros(sizes[i + 1])

    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
    n_layers = len(sizes) - 1

    for _ in range(epochs):
        order = rng.permutation(len(X))
        for lo in range(0, len(X), batch):
            idx = order[lo:lo + batch]
            xb, yb = X[idx], y[idx]

            acts = [xb]
            for i in range(n_layers):
                z = acts[-1] @ params[f"W{i}"] + params[f"b{i}"]
                acts.append(np.maximum(z, 0.0) if i < n_layers - 1 else z)
            grad_out = (acts[-1][:, 0] - yb)[:, None] * (2.0 / len(idx))

            grads = {}
            g = grad_out
            for i in range(n_layers - 1, -1, -1):
                grads[f"W{i}"] = acts[i].T @ g + l2 * params[f"W{i}"]
                grads[f"b{i}"] = g.sum(axis=0)
                if i:
                    g = (g @ params[f"W{i}"].T) * (acts[i] > 0)

            step += 1
            for k in params:
                m[k] = b1 * m[k] + (1 - b1) * grads[k]
                v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
                mh = m[k] / (1 - b1 ** step)
                vh = v[k] / (1 - b2 ** step)
                params[k] -= lr * mh / (np.sqrt(vh) + eps)

    return {"params": params, "n_layers": n_layers}


def _forward(net: dict, X: np.ndarray) -> np.ndarray:
    a = X
    for i in range(net["n_layers"]):
        z = a @ net["params"][f"W{i}"] + net["params"][f"b{i}"]
        a = np.maximum(z, 0.0) if i < net["n_layers"] - 1 else z
    return a[:, 0]
