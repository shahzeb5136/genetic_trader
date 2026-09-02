"""Forward testing: what the strategies actually did after the research window ended.

    python -m sp500lab forward window                  # what is available, and what it proves
    python -m sp500lab forward seal low_vol --rationale "best d_sharpe in the suite"
    python -m sp500lab forward run low_vol             # <- this spends the holdout
    python -m sp500lab forward scoreboard

Every number this project has produced so far comes from 2007-04 to 2021-12. Strategies
were written, tuned, evolved and ranked inside that window, and ADR-032 says plainly
what the genetic algorithm's folds are: a measure of consistency, not of generalisation.
The only out-of-sample evidence available is the reserved period from 2022-01-01 onward
(ADR-025), and this package is the machinery for spending it properly.

Reading order
-------------
    windows.py   the three windows, the vintage arithmetic, and what a short window
                 can and cannot prove. Pure - no I/O, no panel.
    compare.py   research against forward: the deltas, their standard errors, the
                 named checks and the verdict rules. Also pure.
    legs.py      the seam: a `BacktestResult` reduced to a comparable `Leg`.
    seal.py      pre-registration. Write the prediction down before the look.
    store.py     what is kept, and the read API a forward report will be built on.
    engine.py    the orchestration: measure, seal, look, compare, record.
    cli.py       `sp500lab forward ...`

The three ideas that make this more than "run the backtest with a later start date"
------------------------------------------------------------------------------------

**1. A forward test is a paired comparison, not a number.** The research window made a
prediction. The forward window is the only place it can be checked. So every record
carries both legs and the difference between them, next to the standard error of that
difference - which on 56 monthly observations is around 0.5 on a Sharpe, and is the
reason a forward test here can refute a strategy but cannot confirm one.

**2. The holdout protects against fitting, not against choosing.** Nothing in ADR-025
stops somebody forward-testing twenty strategies and reporting the best three. So
candidates are *sealed* before the look, the seal records whether it was declared in
advance or written at the moment of the look, and `store.selection_bar()` applies the
same best-of-N correction to the forward window that the deflated Sharpe applies to a
search.

**3. Out-of-sample data keeps arriving.** The forward window grows by a month every
month. A second look a year later is mostly a re-reading of data already seen - except
for the twelve new months, which are genuine fresh evidence. Every record stores the
data vintage it ran against so `fresh_months` can say which part of a later look is
actually new.

Where the reports live
----------------------
Not here. `store.py` exposes seven pure read functions returning DataFrames and
dataclasses; `reporting/forward_views.py` composes them into an executive summary, one
technical report per candidate, a cross-sectional decay analysis and an honesty page,
and `sp500lab report forward` writes the set. That separation is the one ADR-028 draws
for everything else, and it is what lets a forward report be rebuilt years later from
the record alone, with no panel and no re-run.

See docs/FORWARD_TEST.md, ADR-033, ADR-034 and ADR-035.
"""

from __future__ import annotations

# NOTE: the `compare()` FUNCTION is deliberately not re-exported here. Binding it in
# the package namespace would shadow the `sp500lab.forward.compare` MODULE, so
# `import sp500lab.forward.compare as c` would hand back a function. Import it from
# its own module: `from sp500lab.forward.compare import compare`.
from .compare import Comparison, Leg
from .engine import (BASELINE_STUDY, DEFAULT_COSTS, FORWARD_STUDY, ForwardError,
                     ForwardTest, Outcome, forward_suite, forward_test, seal_candidate)
from .legs import leg_from_dict, leg_from_result, leg_from_slice
from .seal import Seal, create_seal
from .store import ForwardRecord
from .windows import (MIN_FORWARD_MONTHS, Window, describe_power, forward_window,
                      freshness, research_window, sharpe_band)

__all__ = [
    "forward_test", "forward_suite", "seal_candidate",
    "ForwardTest", "Outcome", "ForwardError", "ForwardRecord",
    "Seal", "create_seal",
    "Comparison", "Leg",
    "Window", "forward_window", "research_window", "freshness",
    "sharpe_band", "describe_power", "MIN_FORWARD_MONTHS",
    "leg_from_result", "leg_from_slice", "leg_from_dict",
    "FORWARD_STUDY", "BASELINE_STUDY", "DEFAULT_COSTS",
]
