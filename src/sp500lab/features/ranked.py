"""Pre-ranked features: the cross-sectional ranking, done once instead of ten thousand times.

The observation
---------------
`rank_pct(feature, ctx.tradable)` depends on the feature, the date and the tradable mask -
and on nothing about the strategy. Two individuals in a genetic algorithm's population
therefore compute exactly the same ranks and then combine them with different weights. In
a 4,000-evaluation search each rank is recomputed 4,000 times to produce the same numbers.

So they are computed once, here, into a panel of the same shape, and a fitness evaluation
becomes a gather and a weighted sum. Measured: 0.29s per evaluation down to 0.16s, which
is nine minutes off a 4,000-individual search and correspondingly more off a larger one.

Why the names change
--------------------
A ranked column is called `mom_12_1__rank`, not `mom_12_1`. Same values in the same place
would let a strategy expecting raw values silently receive ranks - producing a portfolio
that is wrong in a way that still looks like a portfolio. Renaming turns that mistake into
a KeyError at the first rebalance, which is the whole design rule this codebase runs on:
make the failure mode impossible to express rather than remembering not to write it.

The mask has to match
---------------------
Ranks are taken within `panel.tradable(liquidity_floor)`, the same mask the engine hands a
strategy as `ctx.tradable`. A run using a different liquidity floor than the one the ranks
were built with would be ranking against a different universe, so the floor is recorded in
the metadata and `evolve/engine.py` passes the same value to both.
"""

from __future__ import annotations

import logging

import numpy as np

from ..strategies.signals import rank_pct
from .panel import FeaturePanel

log = logging.getLogger(__name__)

#: Suffix marking a column as an in-date percentile rank rather than a raw value.
RANK_SUFFIX = "__rank"


def rank_panel(features: FeaturePanel, panel, names: tuple[str, ...],
               *, keep_raw: tuple[str, ...] = (),
               liquidity_floor: float = 0.0) -> FeaturePanel:
    """A FeaturePanel of percentile ranks, plus any raw columns asked for.

    `keep_raw` is for features that must not be ranked. Macro columns are one value
    broadcast across every security, so ranking them within a date produces a constant -
    the regime gate needs the level, not its rank among identical copies.
    """
    missing = [n for n in tuple(names) + tuple(keep_raw) if not features.has(n)]
    if missing:
        raise KeyError(f"cannot rank features that do not exist: {missing}")

    rows = np.asarray(features.rows, dtype=np.int64)
    tradable = panel.tradable(liquidity_floor)[rows]
    n_rows, n_sec = len(rows), features.values.shape[1]

    out_names = tuple(f"{n}{RANK_SUFFIX}" for n in names) + tuple(keep_raw)
    values = np.empty((n_rows, n_sec, len(out_names)), dtype=np.float32)

    for f, name in enumerate(names):
        raw = features.matrix(name)
        for i in range(n_rows):
            values[i, :, f] = rank_pct(raw[i].astype(np.float64), tradable[i])
    for j, name in enumerate(keep_raw):
        values[:, :, len(names) + j] = features.matrix(name)

    meta = dict(features.meta) | {
        "ranked": True,
        "ranked_from": list(names),
        "raw_passthrough": list(keep_raw),
        "liquidity_floor": float(liquidity_floor),
    }
    log.info("features: pre-ranked %d column(s) over %d dates", len(names), n_rows)
    return FeaturePanel(dates=features.dates, rows=features.rows,
                        security_ids=features.security_ids, names=out_names,
                        values=values, meta=meta)
