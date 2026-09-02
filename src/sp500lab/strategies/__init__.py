"""Strategy implementations. Importing this package registers them by name.

    baselines.py   the null hypotheses every competitor must beat
    alpha.py       twelve hypotheses, each with a different economic mechanism
    custom.py      YOURS. Add strategies here; nothing else writes to that file.
    signals.py     the shared grammar: ranking, blending, conditional sorts
    evolvable.py   genome-parameterised strategies - the genetic algorithm's substrate
    learned.py     model-driven strategies - the neural-network path

The three families are separated by how a strategy is *specified*, not by what it does.
All of them implement the same `target_weights(ctx)` and the engine cannot tell them
apart, which is the point: see docs/BACKTEST.md.

Add a module here and import it below; `sp500lab backtest --list` and
`strategy.get_strategy()` pick it up automatically.
"""

from __future__ import annotations

from . import alpha, baselines, custom, evolvable, frontier, learned  # noqa: F401 - side effects
from . import signals  # noqa: F401

#: Named sets, so the CLI can run "the null hypotheses" or "the hypotheses" without
#: anyone having to remember which is which. `spy_buy_hold` is in none of them: it is
#: evaluated by accept.replicate_benchmark(), not through the security panel, and it
#: appears in every scoreboard as the benchmark column instead of as a row.
GROUPS: dict[str, tuple[str, ...]] = {
    "baselines": ("equal_weight", "momentum_12_1", "low_vol", "short_reversal",
                  "random_weight", "cash"),
    "alpha": ("residual_momentum", "frog_in_the_pan", "pead_drift", "accrual_quality",
              "quality_value", "restatement_averse", "illiquidity_carry",
              "index_entry_drift", "lottery_averse", "dividend_grower",
              "defensive_regime", "multi_factor"),
    # The second wave (frontier.py): timing, calendar and ensemble mechanisms the first
    # twelve do not cover. Added 2026-08 AFTER the 2022-2026 forward test was read -
    # see the module docstring and ADR-037 for what that contaminates.
    "frontier": ("overnight_momentum", "week52_breakout", "div_month", "vol_managed",
                 "ensemble_rank"),
    "learned": ("rolling_ridge", "shallow_mlp"),
    "evolved": ("evolved_blend",),
    # Yours. Deliberately NOT folded into "all": a strategy you are still working on
    # should not silently join the scoreboard everyone else is measured against. Move it
    # up when you mean it.
    "custom": ("my_first_idea",),
}
GROUPS["all"] = tuple(n for g in ("baselines", "alpha", "frontier", "learned", "evolved")
                      for n in GROUPS[g])

__all__ = ["alpha", "baselines", "custom", "evolvable", "frontier", "learned",
           "signals", "GROUPS"]
