"""The second-wave strategies and the features they stand on.

Unit tests where the logic is isolatable (the dividend calendar, the ensemble roster,
the MLP's label boundary); short real-panel runs where only the full stack can prove
a strategy holds the contract (long-only, tradable-only, finite numbers).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sp500lab.features.events import _div_due
from sp500lab.strategies import GROUPS
from sp500lab.strategies.frontier import ENSEMBLE_MEMBERS
from sp500lab.strategies.learned import ShallowMLP


def test_the_ensemble_roster_is_the_alpha_group():
    # frontier.py names the members locally to avoid a circular import; this is the
    # pin that keeps the two lists from drifting when a hypothesis is added.
    assert ENSEMBLE_MEMBERS == GROUPS["alpha"]


def test_frontier_strategies_are_in_the_all_group():
    for name in GROUPS["frontier"]:
        assert name in GROUPS["all"]
    assert "shallow_mlp" in GROUPS["all"]


# --------------------------------------------------------------------------
# The dividend calendar
# --------------------------------------------------------------------------

def _fake_panel(dates: list[str], n_securities: int = 1):
    return SimpleNamespace(dates=np.asarray(dates, dtype="<U10"),
                           n_securities=n_securities)


def test_div_due_predicts_a_quarterly_cadence():
    sessions = [str(d.date()) for d in pd.bdate_range("2020-01-01", "2021-12-31")]
    panel = _fake_panel(sessions)
    # Quarterly ex-dates: mid-Jan, mid-Apr, mid-Jul, mid-Oct 2020, mid-Jan 2021.
    ev_dates = ["2020-01-15", "2020-04-15", "2020-07-15", "2020-10-15", "2021-01-15"]
    date_pos = {d: i for i, d in enumerate(sessions)}
    ev_row = np.array([date_pos[d] for d in ev_dates], dtype=np.int64)
    ev_col = np.zeros(len(ev_row), dtype=np.int64)

    grid_dates = ["2020-02-28", "2020-03-31", "2020-06-30", "2020-09-30",
                  "2020-12-31", "2021-03-31"]
    rows = np.array([date_pos[d] for d in grid_dates], dtype=np.int64)
    out = _div_due(panel, rows, ev_row, ev_col)[:, 0]

    # Through June 2020 at most two payments are known - a cadence needs two gaps,
    # so three payments, before it is a cadence -> NaN, "no opinion".
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[2])
    # End of September: Jan/Apr/Jul known, cadence ~91d, last Jul 15 -> next due
    # ~Oct 14, two weeks out -> due.
    assert out[3] == 1.0
    # End of December: last payment Oct 15 -> next due mid-January -> due.
    assert out[4] == 1.0
    # End of March 2021: last payment Jan 15 -> next due mid-April -> due.
    assert out[5] == 1.0


def test_div_due_is_zero_between_payments_and_nan_without_history():
    sessions = [str(d.date()) for d in pd.bdate_range("2020-01-01", "2021-12-31")]
    panel = _fake_panel(sessions, n_securities=2)
    date_pos = {d: i for i, d in enumerate(sessions)}
    ev_dates = ["2020-01-15", "2020-04-15", "2020-07-15", "2020-10-15"]
    ev_row = np.array([date_pos[d] for d in ev_dates], dtype=np.int64)
    ev_col = np.zeros(len(ev_row), dtype=np.int64)

    # Mid-cycle: last payment Jul 15, cadence ~91d -> next due ~Oct 14. At the
    # Aug 31 close that is 44 days out - beyond the 31-day horizon -> 0.0, not due.
    rows = np.array([date_pos["2020-08-31"]], dtype=np.int64)
    out = _div_due(panel, rows, ev_row, ev_col)
    assert out[0, 0] == 0.0
    # Security 1 never paid: NaN, which is "no opinion", not "not due".
    assert np.isnan(out[0, 1])


# --------------------------------------------------------------------------
# The MLP's one correctness argument
# --------------------------------------------------------------------------

def test_mlp_training_rows_never_reach_the_as_of_date():
    m = ShallowMLP(train_points=10, horizon=21)
    m._fp = SimpleNamespace(rows=np.arange(0, 2000, 21, dtype=np.int64))
    for t in (500, 999, 1500):
        rows = m._train_rows(t)
        assert len(rows) <= 10
        assert (rows + m.horizon <= t).all()


def test_mlp_instance_reuse_cannot_leak_a_prior_runs_nets():
    """Running one instance twice must give identical curves.

    Revision 1 kept its fitted nets across runs, so a second run's early rebalances
    scored on nets trained years past their as-of dates - the forward harness runs
    six backtests on one instance and its realistic research leg tripled. The reset
    in on_start is the fix; this pins it.
    """
    from sp500lab.backtest import run_backtest
    from sp500lab.backtest.panel import build_panel
    from sp500lab.strategies.learned import ShallowMLP
    try:
        panel = build_panel()
    except Exception:                                          # noqa: BLE001
        pytest.skip("price panel not buildable here")
    strat = ShallowMLP(n_seeds=1, epochs=5)
    kw = dict(panel=panel, start="2011-01-01", end="2012-06-30", costs="realistic",
              log_run=False, record_trades=False)
    first = run_backtest(strat, **kw).equity
    second = run_backtest(strat, **kw).equity
    pd.testing.assert_series_equal(first, second)


def test_mlp_trains_and_predicts_on_synthetic_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    y = X[:, 0] * 0.5 - X[:, 3] * 0.3 + rng.normal(0, 0.1, 400)
    from sp500lab.strategies.learned import _forward, _train_mlp
    net = _train_mlp(X, y, (16, 8), seed=0, epochs=60, lr=1e-2, l2=1e-4)
    pred = _forward(net, X)
    corr = float(np.corrcoef(pred, y)[0, 1])
    assert corr > 0.8, f"the net failed to fit an easy linear signal (corr={corr:.2f})"
    # Same seed, same weights - the ensemble's determinism rests on this.
    net2 = _train_mlp(X, y, (16, 8), seed=0, epochs=60, lr=1e-2, l2=1e-4)
    np.testing.assert_array_equal(net["params"]["W0"], net2["params"]["W0"])


# --------------------------------------------------------------------------
# The features exist, documented, in the built panel
# --------------------------------------------------------------------------

def test_new_features_are_in_the_panel_and_catalogued():
    from sp500lab.features import build_features
    from sp500lab.features.catalog import FEATURE_DOCS
    try:
        fp = build_features()
    except Exception:                                          # noqa: BLE001
        pytest.skip("feature panel not buildable here")
    for name in ("mom_on_12_1", "mom_id_12_1", "on_minus_id_252d", "div_due_1m"):
        assert name in fp.names, name
        assert name in FEATURE_DOCS, name
        mat = fp.matrix(name)
        assert np.isfinite(mat).any(), f"{name} is entirely NaN"


def test_overnight_and_intraday_momentum_compose_to_total():
    from sp500lab.features import build_features
    try:
        fp = build_features()
    except Exception:                                          # noqa: BLE001
        pytest.skip("feature panel not buildable here")
    on = fp.matrix("mom_on_12_1")
    intra = fp.matrix("mom_id_12_1")
    total = fp.matrix("mom_12_1")
    ok = np.isfinite(on) & np.isfinite(intra) & np.isfinite(total)
    assert ok.sum() > 10_000
    # (1 + on) * (1 + id) = 1 + total wherever both legs cover the same sessions.
    # Names with patchy histories have legs summed over slightly different session
    # sets (rolling-sum NaN handling), so the identity is checked on the typical
    # case, not the tail: the median absolute gap must be tiny.
    gap = np.abs((1 + on[ok]) * (1 + intra[ok]) - (1 + total[ok]))
    assert float(np.median(gap)) < 1e-6


# --------------------------------------------------------------------------
# The strategies hold the contract on the real panel
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def short_window_results():
    """Each frontier strategy over 2010-2012, once, shared across assertions."""
    from sp500lab.backtest import run_backtest
    from sp500lab.backtest.panel import build_panel
    try:
        panel = build_panel()
    except Exception:                                          # noqa: BLE001
        pytest.skip("price panel not buildable here")
    out = {}
    for name in GROUPS["frontier"]:
        out[name] = run_backtest(name, panel=panel, start="2010-01-01",
                                 end="2012-12-31", costs="realistic",
                                 log_run=False, record_trades=False)
    return out


def test_frontier_strategies_produce_finite_results(short_window_results):
    for name, res in short_window_results.items():
        p = res.performance
        assert np.isfinite(p.cagr), name
        assert np.isfinite(p.sharpe), name
        assert p.max_drawdown <= 0, name
        assert not res.diagnostics.get("!! ruined"), name


def test_vol_managed_actually_derisks(short_window_results):
    # 2011 H2 was turbulent; the overlay must have run below full investment
    # somewhere, and never above it (long-only, no leverage).
    res = short_window_results["vol_managed"]
    gross = res.weights.sum(axis=1)
    assert float(gross.max()) <= 1.0 + 1e-9
    assert float(gross.min()) < 0.85
