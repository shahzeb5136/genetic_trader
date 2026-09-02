"""Turning features into cross-sectional scores, the same way for every strategy.

Every strategy in `alpha.py` is a sentence like "buy the names with a high X and a low Y,
among the ones that satisfy Z". This module is the grammar: the standardisation, the
blending and the conditioning that turn that sentence into a score vector. Sharing it
means the strategies differ in what they say rather than in how carefully their author
handled NaNs, which is the same argument `portfolio.py` makes one layer down.

Standardise within the date, always
------------------------------------
A raw feature is not comparable across dates or across names. Gross profitability of 0.35
means something different in 2009 and in 2021, and a momentum figure and a volatility
figure have no common scale at all. Both problems disappear by ranking or z-scoring
**within each rebalance date, across the eligible names only**. Standardising over all
panel columns instead would let names that were not in the index that month move the mean.

Rank, not z-score, by default
------------------------------
`rank_pct` is the workhorse rather than `zscore`. Cross-sectional fundamental data has
tails that are not merely fat but wrong - one bad share count produces a book-to-market of
300, and a z-score hands that name a score of +25 and the entire portfolio. A percentile
rank caps the damage at "first place". The z-score is still available, winsorised, for
signals whose magnitude genuinely carries information.

A missing component is not a zero
----------------------------------
`blend` averages the components a name actually has, rather than summing and letting a
NaN annihilate the row. The alternative silently shrinks the universe to names with
complete fundamental data, which - because coverage correlates with survival - is a
survivorship filter dressed up as an arithmetic convention. `min_components` is the honest
control: say how much information a name must have before it is allowed an opinion.
"""

from __future__ import annotations

import numpy as np

#: Z-scores are clipped here. One bad print should not be able to steer a portfolio, and
#: at 4 standard deviations the difference between "extreme" and "wrong" is not decidable
#: from the data.
Z_CLIP = 4.0


def rank_pct(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(S,) percentile rank in [0, 1] among eligible names. NaN elsewhere.

    Ties get the average of the ranks they span, so a feature that is constant across
    half the universe does not hand an arbitrary ordering to whoever sorts first. The
    tie-break that decides the actual portfolio lives in `portfolio.py` and is
    deliberately uncorrelated with survival (ADR-024).
    """
    out = np.full(x.shape, np.nan)
    ok = mask & np.isfinite(x)
    n = int(ok.sum())
    if n < 3:
        return out
    vals = x[ok]
    order = np.argsort(vals, kind="stable")
    sorted_vals = vals[order]

    # Average ties, vectorised. The obvious version walks the sorted array in Python,
    # which costs ~500 iterations per feature per rebalance - about 1.4 million
    # interpreter steps per backtest, and a genetic algorithm runs ten thousand
    # backtests. This does the same thing with three cumulative sums.
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    np.not_equal(sorted_vals[1:], sorted_vals[:-1], out=is_new[1:])
    group = np.cumsum(is_new) - 1
    starts = np.flatnonzero(is_new)
    ends = np.append(starts[1:], n)
    mean_rank = (starts + ends - 1) / 2.0

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = mean_rank[group]
    out[ok] = ranks / max(n - 1, 1)
    return out


def zscore(x: np.ndarray, mask: np.ndarray, clip: float = Z_CLIP) -> np.ndarray:
    """(S,) cross-sectional z-score among eligible names, winsorised. NaN elsewhere."""
    out = np.full(x.shape, np.nan)
    ok = mask & np.isfinite(x)
    if int(ok.sum()) < 3:
        return out
    vals = x[ok]
    sd = float(vals.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return out
    out[ok] = np.clip((vals - float(vals.mean())) / sd, -clip, clip)
    return out


def blend(components: list[np.ndarray], weights: list[float] | None = None,
          min_components: int = 1) -> np.ndarray:
    """Weighted average of the components each name actually has.

    Not a sum: a name with three of five signals should be judged on three signals, not
    penalised into last place for the two it is missing. The weights of the present
    components are renormalised, so the result stays on the same scale however many
    turned up.
    """
    if not components:
        raise ValueError("blend needs at least one component")
    w = np.asarray(weights if weights is not None else [1.0] * len(components),
                   dtype=np.float64)
    stack = np.vstack([np.asarray(c, dtype=np.float64) for c in components])
    present = np.isfinite(stack)
    contrib = np.where(present, np.nan_to_num(stack) * w[:, None], 0.0)
    denom = np.where(present, np.abs(w)[:, None], 0.0).sum(axis=0)
    count = present.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = contrib.sum(axis=0) / denom
    return np.where((count >= min_components) & (denom > 0) & np.isfinite(out),
                    out, np.nan)


def conditional(primary: np.ndarray, condition: np.ndarray, mask: np.ndarray,
                keep: float = 0.33, high: bool = True) -> np.ndarray:
    """Score on `primary`, but only among names in the top (or bottom) slice of `condition`.

    A double sort rather than a blend. The two say different things: a blend claims the
    signals are additive, a conditional sort claims one of them is a REGIME for the other.
    "Momentum works better when the news arrived gradually" is a conditional statement, and
    flattening it into a weighted sum would test a different hypothesis than the one the
    literature makes.
    """
    r = rank_pct(condition, mask)
    threshold = (1.0 - keep) if high else keep
    selected = (r >= threshold) if high else (r <= threshold)
    return np.where(mask & selected & np.isfinite(primary), primary, np.nan)


def winsorise(x: np.ndarray, mask: np.ndarray, lo: float = 1.0,
              hi: float = 99.0) -> np.ndarray:
    """Clip a feature to its own cross-sectional percentiles before it is used raw."""
    ok = mask & np.isfinite(x)
    if int(ok.sum()) < 10:
        return np.where(ok, x, np.nan)
    a, b = np.percentile(x[ok], [lo, hi])
    return np.where(ok, np.clip(x, a, b), np.nan)


def require(mask: np.ndarray, *features: np.ndarray) -> np.ndarray:
    """Narrow an eligibility mask to names that have every one of these features.

    Use it when a strategy genuinely cannot form an opinion without a value - and know
    what it costs. Fundamental coverage correlates with survival, so requiring a
    fundamental narrows the universe toward the names that lived. The engine reports
    coverage on every run precisely so this shows up.
    """
    out = np.asarray(mask, dtype=bool).copy()
    for f in features:
        out &= np.isfinite(f)
    return out
