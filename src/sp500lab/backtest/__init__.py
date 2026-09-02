"""Backtest engine: one harness, one set of rules, every model family scored the same.

    from sp500lab.backtest import run_backtest
    res = run_backtest("momentum_12_1", start="2010-01-01", costs="realistic")
    print(res.summary())

Reading order if you are new to this
------------------------------------
    context.py     the point-in-time view, and why leakage is structurally impossible
    engine.py      the rebalance loop, execution timing and the accounting
    costs.py       what a trade costs under the ADR-016 mandate
    portfolio.py   score -> weights, shared by every competitor
    trades.py      the order-by-order ledger - the evidence for an equity curve
    accept.py      the four checks that decide if any of this means anything
    registry.py    the trial log and the holdout guard - read before searching

Full narrative: docs/BACKTEST.md, and docs/EXPERIMENTS.md for the research discipline.
Design decisions: docs/DECISIONS.md ADR-017..ADR-026.

Two defaults worth knowing before your first run: every backtest is logged as a trial,
and every backtest stops before the 2022-01-01 holdout.

Before trusting a single number, run:

    python -m sp500lab backtest accept
"""

from __future__ import annotations

from .context import Context, LookaheadError, PanelView
from .costs import (FREE, OPTIMISTIC, PESSIMISTIC, REALISTIC, CostBreakdown, CostModel,
                    all_settings, get_cost_model)
from .engine import EngineError, run_all_cost_settings, run_backtest
from .metrics import Performance, deflate_result, deflated_sharpe, probabilistic_sharpe
from .panel import Panel, build_panel
from .registry import HOLDOUT_START, deflate, deflate_best, study
from .portfolio import Construction, build_weights, validate_weights
from .results import BacktestResult, compare, format_compare
from .strategy import (BaseStrategy, FunctionStrategy, SignalStrategy, Strategy,
                       get_strategy, list_strategies, register)
from .trades import TradeLedger, holdings, reconcile, write_csv

__all__ = [
    "Context", "LookaheadError", "PanelView",
    "Panel", "build_panel",
    "HOLDOUT_START", "study", "deflate", "deflate_best",
    "Strategy", "BaseStrategy", "SignalStrategy", "FunctionStrategy",
    "register", "get_strategy", "list_strategies",
    "Construction", "build_weights", "validate_weights",
    "CostModel", "CostBreakdown", "get_cost_model", "all_settings",
    "OPTIMISTIC", "REALISTIC", "PESSIMISTIC", "FREE",
    "run_backtest", "run_all_cost_settings", "EngineError",
    "BacktestResult", "compare", "format_compare",
    "Performance", "probabilistic_sharpe", "deflated_sharpe", "deflate_result",
    "TradeLedger", "reconcile", "write_csv", "holdings",
]
