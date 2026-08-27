"""Test-suite setup.

The experiment registry logs every backtest by default, which is exactly what you want
in real use and exactly what you do not want from a test run: 80 unit tests would append
80 fake trials to a real search and quietly inflate `n_trials`, which is the input to the
deflated Sharpe. An over-counted trial makes a genuine result look worse, so the damage
is silent in the direction that is hardest to notice.

Disabling it here, at import time, means an individual test never has to remember to.
Tests that exercise the registry itself point it at a tmp_path instead - see
tests/test_registry.py.

The holdout ledger is deliberately NOT disabled by this: nothing in the test suite should
be touching the holdout, and if something starts to, the ledger is how it gets noticed.
"""

from __future__ import annotations

import os

os.environ.setdefault("SP500LAB_REGISTRY", "off")
