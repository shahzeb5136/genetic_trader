"""Strategy implementations. Importing this package registers them by name.

    baselines.py   the null hypotheses every competitor must beat
    evolvable.py   genome-parameterised strategies - the genetic algorithm's substrate
    learned.py     model-driven strategies - the neural-network path

The three families are separated by how a strategy is *specified*, not by what it does.
All of them implement the same `target_weights(ctx)` and the engine cannot tell them
apart, which is the point: see docs/BACKTEST.md.

Add a module here and import it below; `sp500lab backtest --list` and
`strategy.get_strategy()` pick it up automatically.
"""

from __future__ import annotations

from . import baselines, evolvable, learned  # noqa: F401 - @register side effects

__all__ = ["baselines", "evolvable", "learned"]
