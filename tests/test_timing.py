"""The calendar lab: leg compilation, the walk, costs, and the two identities.

Most tests run on small synthetic TimingData where every expected number can be done
by hand. The two acceptance identities also run against the real benchmarks table,
because they are the seam between this engine and the monthly one - if either drifts,
every timing number is void (see timing/engine.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp500lab.backtest.costs import FREE, CostModel
from sp500lab.timing.data import TimingData
from sp500lab.timing.engine import (TimingEngineError, run_timing_backtest,
                                    timing_accept)
from sp500lab.timing.strategies import (PreHoliday, TurnOfMonth, VixOvernight,
                                        Weekend, _from_return_days)

#: Zero spread but a real commission, for tests that count the dollars.
COMMISSION_ONLY = CostModel(name="test", per_share=0.0, min_commission=1.0,
                            max_commission_pct=0.0, spread_multiple=0.0,
                            fallback_half_spread=0.0)


def make_data(dates: list[str], opens=None, closes=None, vix=None) -> TimingData:
    d = pd.to_datetime(pd.Series(dates))
    n = len(dates)
    closes = np.asarray(closes if closes is not None
                        else np.linspace(100.0, 100.0 + n - 1, n), dtype=np.float64)
    opens = np.asarray(opens if opens is not None else closes - 0.25,
                       dtype=np.float64)
    return TimingData(
        ticker="TEST",
        dates=np.asarray(dates, dtype="<U10"),
        adj_open=opens, adj_close=closes,
        raw_open=opens, raw_close=closes,
        half_spread=np.full(n, 1e-4),
        day_of_week=d.dt.dayofweek.to_numpy(dtype=np.int64),
        month=d.dt.month.to_numpy(dtype=np.int64),
        vix=np.asarray(vix, dtype=np.float64) if vix is not None
        else np.full(n, np.nan),
    )


def weekdays(start: str, n: int) -> list[str]:
    return [str(d.date()) for d in pd.bdate_range(start, periods=n)]


# --------------------------------------------------------------------------
# Leg compilation
# --------------------------------------------------------------------------

def test_from_return_days_compiles_the_leg_convention():
    # Session 2's close-to-close return spans the overnight leg of 1 and the
    # intraday leg of 2 - the whole translation from paper-speak to engine-speak.
    mask = np.array([False, False, True, True, False])
    on, intra = _from_return_days(mask)
    assert intra.tolist() == [False, False, True, True, False]
    assert on.tolist() == [False, True, True, False, False]


def test_weekend_rule_enters_only_at_week_boundaries():
    data = make_data(weekdays("2021-01-04", 10))       # two clean Mon-Fri weeks
    on, intra = Weekend().legs(data)
    assert not intra.any()
    held = data.dates[on].tolist()
    assert held == ["2021-01-08"]                      # the first Friday only:
    # the second Friday's overnight leg has no following session in the data.


def test_weekend_rule_handles_a_friday_holiday():
    # A week that ends on Thursday: the standing order enters at Thursday's close.
    dates = ["2021-03-29", "2021-03-30", "2021-03-31", "2021-04-01",  # Fri 4/2 closed
             "2021-04-05", "2021-04-06"]
    on, _ = Weekend().legs(make_data(dates))
    assert data_held(dates, on) == ["2021-04-01"]


def test_pre_holiday_flags_closures_but_not_weekends():
    # Wed 6/30 -> Thu 7/1 -> Fri 7/2 -> Mon 7/5 closed -> Tue 7/6.
    dates = ["2021-06-30", "2021-07-01", "2021-07-02", "2021-07-06", "2021-07-07"]
    _, intra = PreHoliday().legs(make_data(dates))
    assert data_held(dates, intra) == ["2021-07-02"]   # a 4-day gap, not a weekend

    plain = weekdays("2021-01-04", 10)
    _, intra = PreHoliday().legs(make_data(plain))
    assert not intra.any()                             # Fridays are not holidays


def test_turn_of_month_holds_last_one_and_first_three():
    dates = weekdays("2021-01-04", 40)                 # spans Jan..Feb 2021
    _, intra = TurnOfMonth().legs(make_data(dates))
    held = data_held(dates, intra)
    assert "2021-01-29" in held                        # last session of January
    assert {"2021-02-01", "2021-02-02", "2021-02-03"} <= set(held)
    assert "2021-02-04" not in held
    assert "2021-01-28" not in held


def test_vix_gate_reads_only_the_prior_session():
    n = 300
    dates = weekdays("2015-01-01", n)
    vix = np.full(n, 15.0)
    vix[260] = 90.0                                    # one spike, after warmup
    on, _ = VixOvernight().legs(make_data(dates, vix=vix))
    assert not on[260]                                 # the spike day cannot know itself
    assert on[261]                                     # the day after can


def data_held(dates: list[str], mask: np.ndarray) -> list[str]:
    return [d for d, m in zip(dates, mask) if m]


# --------------------------------------------------------------------------
# The walk
# --------------------------------------------------------------------------

def test_buy_hold_reproduces_the_close_series_gross():
    dates = weekdays("2021-01-04", 30)
    closes = 100.0 * np.cumprod(1 + np.sin(np.arange(30)) * 0.01)
    data = make_data(dates, closes=closes, opens=closes * 0.999)
    res = run_timing_backtest("tm_buy_hold", data=data, start=dates[0],
                              end=dates[-1], costs=FREE, track_gross=False,
                              log_run=False)
    expected = closes / closes[0] * 100_000.0
    np.testing.assert_allclose(res.equity.to_numpy(), expected, rtol=1e-12)


def test_overnight_times_intraday_equals_buy_and_hold():
    dates = weekdays("2021-01-04", 30)
    rng = np.random.default_rng(7)
    closes = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, 30))
    opens = closes * (1 + rng.normal(0, 0.004, 30))
    data = make_data(dates, closes=closes, opens=opens)
    kw = dict(data=data, start=dates[0], end=dates[-1], costs=FREE,
              track_gross=False, log_run=False)
    bh = run_timing_backtest("tm_buy_hold", **kw).equity.to_numpy()
    on = run_timing_backtest("tm_overnight", **kw).equity.to_numpy()
    intra = run_timing_backtest("tm_intraday", **kw).equity.to_numpy()
    np.testing.assert_allclose((on / on[0]) * (intra / intra[0]), bh / bh[0],
                               rtol=1e-12)


def test_a_round_trip_is_charged_on_both_sides():
    dates = weekdays("2021-01-04", 3)
    data = make_data(dates, closes=np.array([100.0, 100.0, 100.0]),
                     opens=np.array([100.0, 100.0, 100.0]))
    res = run_timing_backtest("tm_weekend", data=data, start=dates[0],
                              end=dates[-1], costs=COMMISSION_ONLY,
                              track_gross=False, log_run=False)
    # Flat prices, no weekend in range: nothing traded, nothing charged.
    assert res.costs.total == 0.0
    assert res.equity.iloc[-1] == 100_000.0

    res = run_timing_backtest("tm_overnight", data=data, start=dates[0],
                              end=dates[-1], costs=COMMISSION_ONLY,
                              track_gross=False, log_run=False)
    # Two overnight legs (0->1, 1->2), each a buy at the close and a sell at the
    # open: four orders, $1 minimum each, prices flat so that is the whole P&L.
    assert res.costs.n_orders == 4
    assert res.costs.total == pytest.approx(4.0)
    assert res.equity.iloc[-1] == pytest.approx(100_000.0 - 4.0)


def test_the_final_overnight_leg_is_trimmed_at_the_window_end():
    dates = weekdays("2021-01-04", 10)
    closes = np.full(10, 100.0)
    opens = np.full(10, 100.0)
    opens[6] = 150.0        # a huge gap into session 6
    data = make_data(dates, closes=closes, opens=opens)
    res = run_timing_backtest("tm_overnight", data=data, start=dates[0],
                              end=dates[5], costs=FREE, track_gross=False,
                              log_run=False)
    # The window ends at session 5; the 5->6 overnight leg would need session 6's
    # open, which under a holdout is reserved data. It must not be taken.
    assert res.equity.iloc[-1] == pytest.approx(100_000.0)
    assert str(res.equity.index[-1]) == dates[5]


def test_a_hole_in_the_price_series_raises_rather_than_skips():
    dates = weekdays("2021-01-04", 6)
    closes = np.full(6, 100.0)
    closes[3] = np.nan
    data = make_data(dates, closes=closes, opens=np.full(6, 100.0))
    with pytest.raises(TimingEngineError, match="hole"):
        run_timing_backtest("tm_buy_hold", data=data, start=dates[0],
                            end=dates[-1], costs=FREE, track_gross=False,
                            log_run=False)


def test_the_result_carries_registry_compatible_config():
    dates = weekdays("2021-01-04", 15)
    data = make_data(dates)
    res = run_timing_backtest("tm_overnight", data=data, start=dates[0],
                              end=dates[-1], costs="realistic",
                              log_run=False)
    cfg = res.config
    for key in ("start", "end", "cost_model", "initial_capital", "holdout_mode",
                "touched_holdout", "strategy_detail", "panel", "seed",
                "liquidity_floor"):
        assert key in cfg, key
    assert cfg["strategy_detail"]["class"] == "Overnight"
    assert res.gross_equity is not None
    assert res.performance.cost_drag is not None
    assert res.performance.cost_drag > 0


# --------------------------------------------------------------------------
# The identities, on the real data
# --------------------------------------------------------------------------

def test_timing_engine_accepts_on_real_data():
    try:
        report = timing_accept()
    except KeyError:
        pytest.skip("benchmarks table not available")
    assert report["calibration_bp_per_year"] < 1.0
    assert report["decomposition_max_rel_err"] < 1e-9
    # The decomposition's substantive claim, pinned loosely: overnight carried more
    # of SPY's 2007-2021 return than intraday did. If a data refresh flips this the
    # whole family's motivation needs re-reading, so it should fail loudly here.
    assert report["overnight_cagr"] > report["intraday_cagr"]
