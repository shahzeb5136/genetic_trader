"""The feature layer: every competitor's inputs, computed once and versioned.

    from sp500lab.features import build_features
    from sp500lab.backtest import run_backtest

    feats = build_features()
    run_backtest("quality_value", features=feats)

Reading order
-------------
    panel.py        the FeaturePanel container, the cache, and the leakage test
    price.py        momentum, volatility, beta, liquidity - from the price panel alone
    events.py       index membership transitions and dividend behaviour
    fundamental.py  bitemporal XBRL: filed_date, never period_end
    macro.py        unrevised daily macro series and market state

The one command that decides whether any of it can be trusted:

    python -m sp500lab features check

It rebuilds the whole matrix from a panel that physically ends at a past date, with every
filing after that date deleted, and asserts the earlier rows are bit-identical. See
docs/FEATURES.md and ADR-030.
"""

from __future__ import annotations

from .panel import (FEATURE_VERSION, FeaturePanel, build_features, check_leakage,
                    clear_memo, format_leakage, truncate_panel)

__all__ = [
    "FEATURE_VERSION", "FeaturePanel", "build_features", "check_leakage",
    "clear_memo", "format_leakage", "truncate_panel",
]
