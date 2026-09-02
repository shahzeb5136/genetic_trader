"""The feature layer: is every window trailing, and is every join point-in-time?

Two kinds of test, because features fail in two ways.

**Unit tests on synthetic panels** pin the arithmetic. A world with a known constant
return has a known momentum, a known volatility and a known beta, so a failure points at
the formula rather than at the data.

**The leakage test on real data** is the one that decides whether any of it can be
trusted. It rebuilds the whole matrix from a panel that physically ends at a past date,
with every filing published after that date deleted, and asserts the earlier rows are
bit-identical. A rolling window that centred instead of trailing, or a fundamental joined
on `period_end` instead of `filed_date`, cannot survive it. It is slower than a unit test
and it is the reason the rest of the suite means anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from sp500lab.features.panel import FeaturePanel, _as_grid, truncate_panel
from sp500lab.strategies.signals import blend, conditional, rank_pct, require, zscore

from test_backtest import make_panel


# --------------------------------------------------------------------------
# The scoring grammar
# --------------------------------------------------------------------------

def test_rank_pct_spans_zero_to_one_over_the_eligible_names():
    x = np.array([5.0, 1.0, 3.0, 99.0])
    mask = np.array([True, True, True, False])
    r = rank_pct(x, mask)
    assert r[1] == pytest.approx(0.0)
    assert r[2] == pytest.approx(0.5)
    assert r[0] == pytest.approx(1.0)
    assert np.isnan(r[3]), "an ineligible name must not receive a rank"


def test_rank_pct_averages_ties():
    """Equal values get equal scores; the portfolio's tie-break is elsewhere (ADR-024)."""
    x = np.array([1.0, 2.0, 2.0, 3.0])
    r = rank_pct(x, np.ones(4, dtype=bool))
    assert r[1] == pytest.approx(r[2])
    assert r[1] == pytest.approx(0.5)


def test_rank_pct_is_invariant_to_monotone_transforms():
    """The property the pre-ranked feature panel relies on: ranking a rank is identity."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    mask = np.ones(50, dtype=bool)
    once = rank_pct(x, mask)
    assert rank_pct(once, mask) == pytest.approx(once)
    assert rank_pct(np.exp(x), mask) == pytest.approx(once)


def test_rank_pct_ignores_non_finite_values():
    x = np.array([1.0, np.nan, 3.0, np.inf, 2.0])
    r = rank_pct(x, np.ones(5, dtype=bool))
    assert np.isnan(r[1]) and np.isnan(r[3])
    assert r[0] == pytest.approx(0.0)
    assert r[4] == pytest.approx(0.5)
    assert r[2] == pytest.approx(1.0)


def test_rank_pct_abstains_when_too_few_names_are_eligible():
    """Two names do not make a cross-section; ranking them would invent a spread."""
    r = rank_pct(np.array([1.0, 2.0, np.nan]), np.ones(3, dtype=bool))
    assert np.isnan(r).all()


def test_blend_averages_only_the_components_a_name_has():
    """A name with two of three signals is judged on two, not sent to last place."""
    a = np.array([1.0, 1.0])
    b = np.array([0.0, np.nan])
    out = blend([a, b])
    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(1.0), "the missing component must not drag it down"


def test_blend_respects_min_components():
    out = blend([np.array([1.0]), np.array([np.nan])], min_components=2)
    assert np.isnan(out[0])


def test_blend_renormalises_weights():
    out = blend([np.array([1.0]), np.array([np.nan])], weights=[0.25, 0.75])
    assert out[0] == pytest.approx(1.0)


def test_conditional_keeps_only_the_selected_slice():
    primary = np.array([1.0, 2.0, 3.0, 4.0])
    condition = np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.ones(4, dtype=bool)
    top = conditional(primary, condition, mask, keep=0.5, high=True)
    assert np.isnan(top[0]) and np.isfinite(top[3])
    bottom = conditional(primary, condition, mask, keep=0.5, high=False)
    assert np.isfinite(bottom[0]) and np.isnan(bottom[3])


def test_zscore_is_winsorised():
    x = np.concatenate([np.zeros(99), [1e9]])
    z = zscore(x, np.ones(100, dtype=bool))
    assert z.max() <= 4.0 + 1e-9


def test_require_narrows_the_mask():
    mask = np.ones(3, dtype=bool)
    out = require(mask, np.array([1.0, np.nan, 3.0]))
    assert out.tolist() == [True, False, True]


# --------------------------------------------------------------------------
# Price features, against arithmetic
# --------------------------------------------------------------------------

def _price_features(panel, rows=None):
    from sp500lab.features import price
    rows = panel.rebalance_index if rows is None else rows
    return price.compute(panel, np.asarray(rows, dtype=np.int64))


def test_momentum_matches_the_price_ratio_it_claims_to_be():
    """12-1 momentum is close[t-21]/close[t-273] - 1, and nothing else."""
    p = make_panel(n_days=600, n_sec=2, daily_return=0.001)
    rows = np.array([500], dtype=np.int64)
    out = _price_features(p, rows)
    expected = p.adj_close[500 - 21, 0] / p.adj_close[500 - 273, 0] - 1.0
    assert out["mom_12_1"][0, 0] == pytest.approx(expected, rel=1e-12)


def test_volatility_of_a_constant_return_series_is_zero():
    p = make_panel(n_days=400, n_sec=2, daily_return=0.001)
    out = _price_features(p, np.array([350], dtype=np.int64))
    assert out["vol_126d"][0, 0] == pytest.approx(0.0, abs=1e-9)


def test_features_before_their_window_exists_are_nan_not_zero():
    """'No opinion' and 'zero' are different, and portfolio.py treats them differently."""
    p = make_panel(n_days=400, n_sec=2, daily_return=0.001)
    out = _price_features(p, np.array([5], dtype=np.int64))
    assert np.isnan(out["mom_12_1"][0, 0])
    assert np.isnan(out["vol_126d"][0, 0])


def test_no_price_feature_reads_past_its_row():
    """Delete the future, recompute, and the past must not move.

    The synthetic version of `check_leakage`, fast enough to run on every commit. A
    `center=True` or a negative shift anywhere in price.py fails here.
    """
    p = make_panel(n_days=600, n_sec=4, daily_return=0.001)
    rows = np.array([300, 400, 500], dtype=np.int64)
    full = _price_features(p, rows)

    cut = truncate_panel(p, 500)
    partial = _price_features(cut, rows)

    for name, matrix in full.items():
        a = np.nan_to_num(matrix, nan=-9e30)
        b = np.nan_to_num(partial[name], nan=-9e30)
        assert np.array_equal(a, b), f"{name} changed when the future was removed"


def test_truncated_panel_forgets_how_the_story_ended():
    """A truncated panel must not leak the delisting outcome it has not reached yet."""
    p = make_panel(n_days=300, n_sec=2)
    p.delist_return[0] = -1.0
    p.delist_reason[0] = "bankruptcy"
    cut = truncate_panel(p, 100)
    assert cut.dates[-1] == p.dates[100]
    assert cut.adj_close.shape[0] == 101
    assert float(cut.delist_return[0]) == 0.0
    assert str(cut.delist_reason[0]) == "unknown"
    assert int(cut.last_bar_index[0]) <= 100


# --------------------------------------------------------------------------
# The container
# --------------------------------------------------------------------------

def _panel_of(names, values, rows=(10, 20)):
    return FeaturePanel(dates=np.array(["2020-01-31", "2020-02-28"], dtype="<U10"),
                        rows=np.array(rows, dtype=np.int32),
                        security_ids=np.array(["A", "B"], dtype="<U16"),
                        names=tuple(names), values=np.asarray(values, dtype=np.float32))


def test_at_returns_the_row_for_a_panel_index():
    fp = _panel_of(["x"], np.arange(4, dtype=np.float32).reshape(2, 2, 1))
    assert fp.at(20).ravel().tolist() == [2.0, 3.0]


def test_at_refuses_a_row_it_does_not_have():
    """An interpolated feature is a lookahead bug wearing a convenience API."""
    fp = _panel_of(["x"], np.zeros((2, 2, 1), dtype=np.float32))
    with pytest.raises(KeyError, match="no features stored"):
        fp.at(15)


def test_unknown_feature_name_says_what_is_available():
    fp = _panel_of(["x"], np.zeros((2, 2, 1), dtype=np.float32))
    with pytest.raises(KeyError, match="unknown feature"):
        fp.matrix("nope")


def test_macro_vectors_broadcast_across_securities():
    """One number per date becomes one column, so ctx.feature() works the same for both."""
    out = _as_grid(np.array([1.0, 2.0]), 2, 3, "vix")
    assert out.shape == (2, 3)
    assert out[0].tolist() == [1.0, 1.0, 1.0]


def test_wrongly_shaped_feature_is_rejected():
    with pytest.raises(ValueError, match="expected"):
        _as_grid(np.zeros((3, 3)), 2, 3, "bad")


def test_coverage_reports_per_feature_fill_rates():
    values = np.array([[[1.0], [np.nan]], [[1.0], [1.0]]], dtype=np.float32)
    cov = _panel_of(["x"], values).coverage()
    assert cov.loc[0, "feature"] == "x"
    assert cov.loc[0, "overall"] == pytest.approx(0.75)


# --------------------------------------------------------------------------
# Real data: the test that decides whether the rest matters
# --------------------------------------------------------------------------

def _have_data() -> bool:
    try:
        from sp500lab.storage import silver_exists
        return silver_exists("market/daily_bars_adjusted")
    except Exception:                                             # noqa: BLE001
        return False


needs_data = pytest.mark.skipif(not _have_data(),
                                reason="silver layer not built; run `sp500lab ingest all`")


@needs_data
def test_leakage_check_passes_on_the_real_feature_panel():
    """Delete the future from the real data and the past must be bit-identical.

    Slow - it builds the whole feature matrix twice. It is also the only test in this
    file that can catch a fundamental joined on the wrong date column, which is the
    single most expensive mistake available in this project.
    """
    from sp500lab.features import check_leakage
    report = check_leakage(cut_at="2016-12-30")
    assert report["ok"], f"features reading the future: {report['failed']}"


@needs_data
def test_pre_ranked_features_produce_an_identical_backtest():
    """The GA's speed optimisation must be exactly that, and not a change of answer."""
    import numpy as np

    from sp500lab.backtest import run_backtest
    from sp500lab.backtest.panel import build_panel
    from sp500lab.features import build_features
    from sp500lab.features.ranked import rank_panel
    from sp500lab.strategies.evolvable import from_vector
    from sp500lab.strategies.genome import PRESETS, REGIME_FEATURES, alpha_genome

    panel = build_panel()
    features = build_features(panel=panel)
    ranked = rank_panel(features, panel, PRESETS["price"],
                        keep_raw=REGIME_FEATURES + ("vol_126d",))

    vector = alpha_genome("price").random(np.random.default_rng(3))
    common = dict(panel=panel, log_run=False, record_trades=False, benchmark=None)
    raw = run_backtest(from_vector(vector, "price"), features=features, **common)
    pre = run_backtest(from_vector(vector, "price", pre_ranked=True),
                       features=ranked, **common)
    assert pre.performance.cagr == pytest.approx(raw.performance.cagr, rel=1e-12)


@needs_data
def test_market_cap_is_the_right_order_of_magnitude():
    """The split trap produces a clean 4x or 10x error, so this is a real check.

    Apple is worth trillions. If `log_market_cap` says billions, `cum_split` is being
    applied at the wrong date or in the wrong direction, and every ratio that divides by
    market cap is wrong by the same factor.
    """
    import numpy as np

    from sp500lab.backtest.panel import build_panel
    from sp500lab.features import build_features

    panel = build_panel()
    fp = build_features(panel=panel)
    caps = np.exp(fp.matrix("log_market_cap")[-1])
    biggest = np.nanmax(caps)
    assert 5e11 < biggest < 2e13, f"largest market cap is ${biggest:,.0f}"
    finite = caps[np.isfinite(caps)]
    assert (finite > 1e8).all(), "an S&P 500 member worth under $100M is a data error"
