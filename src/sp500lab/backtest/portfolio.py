"""Turning a cross-sectional score into a portfolio.

Why this is a separate layer
----------------------------
The stated goal of this project is a competition: genetic algorithms against neural
nets against classical rules. If each competitor writes its own code for "take my
scores and build a portfolio", the competition partly measures who wrote better
portfolio-construction code, which is not the question being asked.

So the split is: a strategy produces a **score per security**, and this module turns
scores into weights. Everyone shares the same construction, the same caps and the same
tie-breaks, and the only thing that differs between competitors is the score. That is
the comparison we actually want.

It also removes the most common way a strategy accidentally violates the mandate.
Long-only means non-negative weights summing to at most one (ADR-016); enforcing that
in one place beats enforcing it in fifty.

Breaking ties without breaking the backtest
--------------------------------------------
Two runs of the same strategy must produce byte-identical weights, so ranking ties need
a deterministic tie-break. The obvious choice - order by security_id - is deterministic
and WRONG, and it cost real debugging time here.

`security_id` is assigned in the order securities are first observed, which correlates
strongly with survival: **99.0% of the low half of the id range is still priced today
against 61.1% of the high half.** Sorting ties by id therefore hands ties to survivors.
Measured consequence: a strategy with all four signal weights set to zero - literally no
opinion about anything - scored 17.65%/yr at a Sharpe of 0.89, beating every honest
baseline, purely from the tie-break.

So ties are broken on a stable hash of the security_id instead. It is deterministic
across runs, machines and Python versions (blake2b, not the salted builtin `hash`), and
it has no relationship to listing order. See ADR-024.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

WEIGHTING_SCHEMES = ("equal", "score", "score_rank", "inverse_vol")


@dataclass(frozen=True)
class Construction:
    """How a score vector becomes a weight vector.

    top_k          hold the best k names; None means every eligible name
    weighting      how to split capital across the selected names
    max_weight     per-name cap, applied after weighting (0.05 = 5%)
    min_names      abstain to cash if fewer eligible names than this
    gross          fraction of NAV invested; 1.0 is fully invested
    long_only      reject negative weights rather than silently clipping (ADR-016)
    """

    top_k: int | None = 50
    weighting: str = "equal"
    max_weight: float = 1.0
    min_names: int = 5
    gross: float = 1.0
    long_only: bool = True

    def __post_init__(self) -> None:
        if self.weighting not in WEIGHTING_SCHEMES:
            raise ValueError(f"weighting must be one of {WEIGHTING_SCHEMES}")
        if not 0 < self.gross <= 1.0:
            raise ValueError("gross must be in (0, 1]; leverage is outside the mandate")
        if not 0 < self.max_weight <= 1.0:
            raise ValueError("max_weight must be in (0, 1]")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1 or None")


def stable_tiebreak(security_ids: np.ndarray) -> np.ndarray:
    """(S,) uint64 ordering keys with no relationship to listing order.

    blake2b rather than the builtin `hash`, which is randomly salted per process and
    would make two runs of the same backtest disagree.
    """
    return np.array(
        [int.from_bytes(hashlib.blake2b(str(s).encode(), digest_size=8).digest(), "big")
         for s in np.asarray(security_ids).tolist()], dtype=np.uint64)


def build_weights(
    scores: np.ndarray,
    eligible: np.ndarray,
    construction: Construction,
    *,
    tiebreak: np.ndarray | None = None,
    security_ids: np.ndarray | None = None,
    vol: np.ndarray | None = None,
) -> np.ndarray:
    """(S,) weights from (S,) scores.

    `eligible` is the tradable mask - in the index, priced, liquid enough. Anything
    outside it gets weight zero regardless of its score, which is what stops a
    strategy from buying a name that was not in the index that month.

    Higher score is better. NaN scores are treated as ineligible rather than as zero,
    because "I have no opinion" and "I score this at exactly zero" are different
    statements and only one of them should be able to earn a position.
    """
    s = np.asarray(scores, dtype=np.float64)
    ok = np.asarray(eligible, dtype=bool) & np.isfinite(s)
    n_ok = int(ok.sum())
    w = np.zeros(s.shape[0], dtype=np.float64)
    if n_ok < construction.min_names:
        return w  # abstain to cash rather than build a portfolio out of nothing

    if tiebreak is None and security_ids is not None:
        tiebreak = stable_tiebreak(security_ids)

    idx = np.flatnonzero(ok)
    if construction.top_k is not None and n_ok > construction.top_k:
        idx = _top_k(s, idx, construction.top_k, tiebreak)

    raw = _raw_weights(s, idx, construction, vol)
    raw = raw / raw.sum()
    if construction.max_weight < 1.0:
        raw = _cap(raw, construction.max_weight)

    w[idx] = raw * construction.gross
    return validate_weights(w, long_only=construction.long_only)


def _top_k(scores: np.ndarray, idx: np.ndarray, k: int,
           tiebreak: np.ndarray | None) -> np.ndarray:
    """The k highest scores among `idx`, with a deterministic, unbiased tie-break.

    np.argpartition is O(n) but its order among equal keys is unspecified, so on a score
    with many ties - a binary GA rule, say - the selected set would depend on array
    layout and the run would stop being reproducible. Sorting on
    (-score, tiebreak) costs O(n log n) on ~500 elements, which is nothing.

    `tiebreak` must be uncorrelated with anything economic. Passing `None` falls back to
    array position, which IS correlated with survival (see the module docstring) - it
    exists only so a caller with no security_ids still gets a reproducible answer, and
    it should never be the path a backtest takes.
    """
    sub = scores[idx]
    keys = idx if tiebreak is None else np.asarray(tiebreak)[idx]
    return idx[np.lexsort((keys, -sub))[:k]]


def _raw_weights(scores: np.ndarray, idx: np.ndarray, c: Construction,
                 vol: np.ndarray | None) -> np.ndarray:
    n = len(idx)
    if c.weighting == "equal":
        return np.ones(n)

    if c.weighting == "score":
        # Shift so the weakest selected name gets ~0 rather than a negative weight.
        v = scores[idx]
        v = v - v.min()
        return v + 1e-12 if v.sum() <= 0 else v

    if c.weighting == "score_rank":
        # Rank-weighting is scale-free, so a GA cannot game it by inflating scores.
        order = np.argsort(np.argsort(-scores[idx]))
        return (n - order).astype(np.float64)

    if c.weighting == "inverse_vol":
        if vol is None:
            raise ValueError("inverse_vol weighting needs a vol vector")
        v = np.asarray(vol, dtype=np.float64)[idx]
        v = np.where(np.isfinite(v) & (v > 0), v, np.nan)
        inv = 1.0 / v
        if not np.isfinite(inv).any():
            return np.ones(n)
        return np.nan_to_num(inv, nan=float(np.nanmedian(inv)))

    raise ValueError(f"unknown weighting {c.weighting!r}")


def _cap(w: np.ndarray, cap: float) -> np.ndarray:
    """Enforce a per-name cap, redistributing the excess proportionally.

    Iterative because redistributing can push another name over the cap. Converges in
    a handful of passes; bails out if the cap is infeasible (n * cap < 1), in which
    case equal weights are the only feasible answer.
    """
    n = len(w)
    if n * cap <= 1.0:
        return np.full(n, 1.0 / n)
    w = w.copy()
    for _ in range(64):
        over = w > cap
        if not over.any():
            return w
        excess = (w[over] - cap).sum()
        w[over] = cap
        room = ~over
        under_sum = w[room].sum()
        if under_sum <= 0:
            w[room] = excess / room.sum()
        else:
            w[room] += excess * w[room] / under_sum
    return w


def validate_weights(w: np.ndarray, *, long_only: bool = True,
                     tol: float = 1e-9) -> np.ndarray:
    """Reject a weight vector that breaks the mandate. Never silently repairs it.

    A strategy returning a negative weight has a bug, not a short position (ADR-016).
    Normalising it away would hide the bug and produce a number that looks fine.
    """
    w = np.asarray(w, dtype=np.float64)
    if not np.isfinite(w).all():
        bad = int((~np.isfinite(w)).sum())
        raise ValueError(f"weights contain {bad} non-finite value(s)")
    if long_only and (w < -tol).any():
        worst = float(w.min())
        raise ValueError(
            f"negative weight {worst:.6g} under a long-only mandate (ADR-016). "
            "The engine does not normalise this away - fix the strategy.")
    total = float(w.sum())
    if total > 1.0 + tol:
        raise ValueError(f"weights sum to {total:.6f} > 1; leverage is outside the mandate")
    return w


def turnover(w_target: np.ndarray, w_current: np.ndarray) -> float:
    """One-way turnover: sum of absolute weight changes, halved.

    Halved because buying 10% of one name and selling 10% of another is a single 10%
    turn of the portfolio, not 20%. Getting this wrong doubles or halves every cost
    number downstream.
    """
    return float(np.abs(w_target - w_current).sum() / 2.0)
