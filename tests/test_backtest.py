"""Tests for the backtest engine.

Two tiers, deliberately separated:

**Unit tests** build a synthetic panel by hand and assert exact numbers. A world where
every price is known lets the accounting be checked against arithmetic done on paper
rather than against another implementation - which is the only way to know that both
are not wrong in the same direction.

**Integration tests** run the real acceptance suite against the ingested data and skip
cleanly when it is absent, so a fresh clone still gets a green test run.

Every test here corresponds to a way a backtest lies. The names say which one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp500lab.backtest.context import Context, LookaheadError, PanelView
from sp500lab.backtest.costs import FREE, OPTIMISTIC, PESSIMISTIC, REALISTIC, CostModel
from sp500lab.backtest.delisting import classify_reason
from sp500lab.backtest.engine import EngineError, run_backtest
from sp500lab.backtest.metrics import (deflated_sharpe, expected_max_sharpe, norm_cdf,
                                       norm_ppf, probabilistic_sharpe)
from sp500lab.backtest.panel import Panel
from sp500lab.backtest.portfolio import (Construction, build_weights, stable_tiebreak,
                                         turnover, validate_weights)
from sp500lab.backtest.spreads import corwin_schultz, tick_floor
from sp500lab.backtest.strategy import BaseStrategy, normalize_weights
from sp500lab.normalize.splits import cumulative_split_ratio, tick_size


# --------------------------------------------------------------------------
# Synthetic panel
# --------------------------------------------------------------------------

def make_panel(n_days: int = 260, n_sec: int = 3, daily_return: float = 0.001,
               rebalance_every: int = 21) -> Panel:
    """A world with no dividends, no splits and a constant daily return.

    Constant growth makes every expected NAV computable in closed form, so a failure
    points at the accounting rather than at the data.
    """
    base = pd.Timestamp("2020-01-01")
    dates = np.array([(base + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
                      for i in range(n_days)], dtype="<U10")

    growth = (1.0 + daily_return) ** np.arange(n_days)
    close = np.outer(growth, 100.0 * np.arange(1, n_sec + 1))
    open_ = close / (1.0 + daily_return)  # today's open == yesterday's close

    sids = np.array([f"SID{i:06d}" for i in range(n_sec)], dtype="<U16")
    ones = np.ones((n_days, n_sec))
    reb = np.arange(rebalance_every - 1, n_days - 1, rebalance_every, dtype=np.int32)

    return Panel(
        dates=dates, security_ids=sids,
        tickers=np.array([f"T{i}" for i in range(n_sec)], dtype="<U16"),
        tiebreak=stable_tiebreak(sids),
        adj_close=close, adj_open=open_, raw_close=close, cum_split=ones,
        dollar_volume=np.full((n_days, n_sec), 1e9, dtype=np.float32),
        half_spread=np.zeros((n_days, n_sec), dtype=np.float32),
        in_index=np.ones((n_days, n_sec), dtype=bool),
        has_price=np.ones((n_days, n_sec), dtype=bool),
        first_bar_index=np.zeros(n_sec, dtype=np.int32),
        last_bar_index=np.full(n_sec, n_days - 1, dtype=np.int32),
        delist_return=np.zeros(n_sec), delist_reason=np.full(n_sec, "none", dtype="<U24"),
        rebalance_index=reb, index_size=np.full(n_days, n_sec, dtype=np.int32),
        meta={"synthetic": True},
    )


class HoldFirst(BaseStrategy):
    """All-in on security 0 while it is tradable, cash once it is not.

    Tradability-aware on purpose: the engine refuses an allocation to a name outside
    the point-in-time universe, so a strategy that ignores `ctx.tradable` cannot run -
    which is itself one of the guarantees under test.
    """

    name = "hold_first"

    def target_weights(self, ctx):
        w = ctx.empty_weights()
        if ctx.tradable[0]:
            w[0] = 1.0
        return w


class EqualAll(BaseStrategy):
    name = "equal_all"

    def target_weights(self, ctx):
        w = ctx.empty_weights()
        w[:] = 1.0 / len(w)
        return w


# --------------------------------------------------------------------------
# Accounting - the numbers must match arithmetic, not another implementation
# --------------------------------------------------------------------------

def test_constant_growth_compounds_exactly():
    """In a world growing 0.1%/day, the NAV must grow 0.1%/day. Nothing else."""
    p = make_panel(n_days=260, daily_return=0.001)
    res = run_backtest(HoldFirst(), panel=p, start=p.dates[0], costs=FREE,
                       benchmark=None, track_gross=False, initial_capital=1000.0)
    eq = res.equity
    growth = eq.iloc[-1] / eq.iloc[0]
    n = len(eq) - 1
    assert growth == pytest.approx((1.001) ** n, rel=1e-9)


def test_no_dividend_double_count_in_synthetic_world():
    """With no corporate actions the NAV must track the price exactly - no extra yield.

    A dividend credited on top of a total-return-adjusted price would show up here as
    NAV growth exceeding price growth in a world that pays no dividends at all.
    """
    p = make_panel(n_days=130, daily_return=0.002)
    res = run_backtest(HoldFirst(), panel=p, start=p.dates[0], costs=FREE,
                       benchmark=None, track_gross=False, initial_capital=1000.0)
    first_exec = p.date_index(str(res.rebalances["exec_date"].iloc[0]))
    price_growth = p.adj_close[-1, 0] / p.adj_open[first_exec, 0]
    assert res.equity.iloc[-1] / 1000.0 == pytest.approx(price_growth, rel=1e-9)


def test_execution_is_at_next_open_not_signal_close():
    """The first fill price must be the open AFTER the first rebalance date."""
    p = make_panel(n_days=100, daily_return=0.01)
    res = run_backtest(HoldFirst(), panel=p, start=p.dates[0], costs=FREE,
                       benchmark=None, track_gross=False, initial_capital=1000.0)
    first = res.rebalances.iloc[0]
    signal_row = p.date_index(first["date"])
    exec_row = p.date_index(first["exec_date"])
    assert exec_row == signal_row + 1

    # NAV at the close of the execution day = capital * close/open of that day.
    expected = 1000.0 * p.adj_close[exec_row, 0] / p.adj_open[exec_row, 0]
    assert res.equity.loc[first["exec_date"]] == pytest.approx(expected, rel=1e-9)


def test_cash_strategy_never_changes_nav():
    p = make_panel(n_days=100)
    res = run_backtest("cash", panel=p, start=p.dates[0], costs=REALISTIC,
                       benchmark=None, track_gross=False, initial_capital=5000.0)
    assert res.equity.nunique() == 1
    assert res.equity.iloc[-1] == pytest.approx(5000.0)


def test_costs_reduce_nav_monotonically():
    """optimistic >= realistic >= pessimistic, always, for any turnover."""
    p = make_panel(n_days=260)
    p.half_spread[:] = 0.001  # 10bp half-spread everywhere
    finals = []
    for cm in (OPTIMISTIC, REALISTIC, PESSIMISTIC):
        res = run_backtest(EqualAll(), panel=p, start=p.dates[0], costs=cm,
                           benchmark=None, track_gross=False)
        finals.append(res.equity.iloc[-1])
    assert finals[0] >= finals[1] >= finals[2]


def test_delisting_return_is_actually_booked():
    """A bankrupt holding must take the position to zero, not silently vanish."""
    p = make_panel(n_days=130, n_sec=2, daily_return=0.0)
    die_at = 60
    p.has_price[die_at + 1:, 0] = False
    p.adj_close[die_at + 1:, 0] = np.nan
    p.adj_open[die_at + 1:, 0] = np.nan
    p.last_bar_index[0] = die_at
    p.in_index[die_at + 1:, 0] = False
    p.delist_return[0] = -1.0
    p.delist_reason[0] = "bankruptcy"

    res = run_backtest(HoldFirst(), panel=p, start=p.dates[0], costs=FREE,
                       benchmark=None, track_gross=False, initial_capital=1000.0)
    assert len(res.exits) == 1, f"expected one forced exit, got {res.exits}"
    assert res.exits.iloc[0]["reason"] == "bankruptcy"
    assert res.exits.iloc[0]["proceeds"] == pytest.approx(0.0)
    # Everything was in that one name, so the portfolio is wiped out.
    assert res.equity.iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_index_removal_is_not_a_bankruptcy():
    """The same mechanics with delist_return=0 must preserve capital."""
    p = make_panel(n_days=130, n_sec=2, daily_return=0.0)
    die_at = 60
    for arr in (p.adj_close, p.adj_open):
        arr[die_at + 1:, 0] = np.nan
    p.has_price[die_at + 1:, 0] = False
    p.last_bar_index[0] = die_at
    p.in_index[die_at + 1:, 0] = False
    p.delist_return[0] = 0.0
    p.delist_reason[0] = "index_removal"

    res = run_backtest(HoldFirst(), panel=p, start=p.dates[0], costs=FREE,
                       benchmark=None, track_gross=False, initial_capital=1000.0)
    assert res.equity.iloc[-1] == pytest.approx(1000.0, rel=1e-9)


def test_strategy_cannot_buy_a_non_member():
    """Allocating outside the point-in-time universe is refused, not clipped."""
    p = make_panel(n_days=100, n_sec=3)
    p.in_index[:, 2] = False

    class BuyExcluded(BaseStrategy):
        name = "buy_excluded"

        def target_weights(self, ctx):
            w = ctx.empty_weights()
            w[2] = 1.0
            return w

    with pytest.raises(EngineError, match="not tradable"):
        run_backtest(BuyExcluded(), panel=p, start=p.dates[0], costs=FREE,
                     benchmark=None, track_gross=False)


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------

def test_panel_view_ends_at_as_of():
    p = make_panel(n_days=100)
    v = PanelView(p, 40)
    assert len(v.close) == 41
    assert v.dates[-1] == p.dates[40]
    with pytest.raises(IndexError):
        _ = v.close[41]


def test_price_on_future_date_raises():
    p = make_panel(n_days=100)
    v = PanelView(p, 40)
    with pytest.raises(LookaheadError):
        v.date_index(str(p.dates[41]))
    assert v.date_index(str(p.dates[40])) == 40


def test_context_holds_no_reference_to_the_panel():
    """The reach-through escape route must not exist."""
    p = make_panel(n_days=100)
    v = PanelView(p, 40)
    assert not hasattr(v, "panel")
    ctx = Context(as_of=str(p.dates[40]), t=40, view=v,
                  universe=p.in_index[40], tradable=p.in_index[40],
                  positions=np.zeros(p.n_securities), cash=0.0, nav=0.0)
    assert not hasattr(ctx, "panel")


def test_trailing_return_uses_only_past_data():
    p = make_panel(n_days=300, daily_return=0.001)
    ctx = Context(as_of=str(p.dates[280]), t=280, view=PanelView(p, 280),
                  universe=p.in_index[280], tradable=p.in_index[280],
                  positions=np.zeros(p.n_securities), cash=0.0, nav=0.0)
    r = ctx.trailing_return(252, skip=21)
    expected = (1.001) ** 252 - 1
    assert r[0] == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------
# Portfolio construction
# --------------------------------------------------------------------------

def test_long_only_rejects_negative_weights():
    with pytest.raises(ValueError, match="long-only"):
        validate_weights(np.array([0.5, -0.1, 0.6]))


def test_leverage_is_rejected():
    with pytest.raises(ValueError, match="leverage"):
        validate_weights(np.array([0.7, 0.7]))


def test_top_k_ties_break_deterministically():
    scores = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    elig = np.ones(5, dtype=bool)
    ids = np.array(["SID000004", "SID000003", "SID000002", "SID000001", "SID000000"])
    c = Construction(top_k=2, weighting="equal", min_names=1)
    a = build_weights(scores, elig, c, security_ids=ids)
    b = build_weights(scores, elig, c, security_ids=ids)
    assert np.array_equal(a, b)
    assert (a > 0).sum() == 2


def test_stable_tiebreak_is_reproducible_across_processes():
    """The builtin hash() is salted per process; blake2b is not."""
    ids = np.array(["SID000001", "SID000002", "SID000003"])
    assert np.array_equal(stable_tiebreak(ids), stable_tiebreak(ids))
    assert stable_tiebreak(ids).tolist() == [
        int.from_bytes(__import__("hashlib").blake2b(i.encode(), digest_size=8).digest(),
                       "big") for i in ids.tolist()]


def test_tiebreak_is_uncorrelated_with_security_id_order():
    """The survivorship bug in ADR-024. security_id order correlates with survival.

    Measured on the real panel: 99.0% of the low half of the id range is still priced
    today, against 61.1% of the high half. So a tie-break that sorts on the id hands
    every tie to a survivor. This asserts the replacement does not.
    """
    ids = np.array([f"SID{i:06d}" for i in range(600)])
    tb = stable_tiebreak(ids)
    rank_by_id = np.arange(600)
    rank_by_tb = np.argsort(np.argsort(tb))
    corr = float(np.corrcoef(rank_by_id, rank_by_tb)[0, 1])
    assert abs(corr) < 0.15, f"tie-break still tracks id order (corr={corr:.3f})"


def test_zero_signal_strategy_cannot_win_on_the_tie_break():
    """A strategy with no opinion must not systematically pick the same names.

    This is the regression test for the 17.65%/yr result an all-zero-weight strategy
    posted before ADR-024: with 600 tied names and top_k=50, two different id orderings
    must not select the same set just because of how the ids happen to sort.
    """
    n = 600
    ids = np.array([f"SID{i:06d}" for i in range(n)])
    scores = np.zeros(n)
    c = Construction(top_k=50, weighting="equal", min_names=1)
    picked = np.flatnonzero(build_weights(scores, np.ones(n, dtype=bool), c,
                                          security_ids=ids) > 0)
    # If ties fell back to array order the winners would be exactly 0..49.
    assert not np.array_equal(picked, np.arange(50))
    assert picked.mean() > n * 0.25, "selection is bunched at the low end of the id range"


def test_nan_score_is_no_opinion_not_zero():
    scores = np.array([np.nan, -5.0, -6.0])
    c = Construction(top_k=3, weighting="equal", min_names=1)
    w = build_weights(scores, np.ones(3, dtype=bool), c)
    assert w[0] == 0.0
    assert w[1] > 0 and w[2] > 0


def test_max_weight_cap_redistributes():
    scores = np.array([10.0, 1.0, 1.0, 1.0, 1.0])
    c = Construction(top_k=5, weighting="score", max_weight=0.3, min_names=1)
    w = build_weights(scores, np.ones(5, dtype=bool), c)
    assert w.max() <= 0.3 + 1e-12
    assert w.sum() == pytest.approx(1.0)


def test_abstains_below_min_names():
    c = Construction(top_k=10, min_names=5)
    w = build_weights(np.array([1.0, 2.0]), np.ones(2, dtype=bool), c)
    assert w.sum() == 0.0


def test_turnover_is_halved():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert turnover(a, b) == pytest.approx(1.0)  # a full turn, not 2.0


def test_normalize_weights_rejects_unknown_ids():
    ids = np.array(["SID000000", "SID000001"])
    with pytest.raises(KeyError, match="not in the panel"):
        normalize_weights(pd.Series({"AAPL": 1.0}), ids)


# --------------------------------------------------------------------------
# Costs
# --------------------------------------------------------------------------

def test_minimum_commission_binds_at_retail_size():
    """40 shares at $0.005 is $0.20; the $1 minimum is what actually gets charged."""
    cm = CostModel(per_share=0.005, min_commission=1.0, max_commission_pct=0.01,
                   spread_multiple=0.0)
    b = cm.charge(np.array([2000.0]), np.array([50.0]), np.array([0.0]))
    assert b.commission == pytest.approx(1.0)
    assert b.n_min_commission == 1


def test_one_percent_cap_binds_on_tiny_trades():
    cm = CostModel(per_share=0.005, min_commission=1.0, max_commission_pct=0.01,
                   spread_multiple=0.0)
    b = cm.charge(np.array([20.0]), np.array([50.0]), np.array([0.0]))
    assert b.commission == pytest.approx(0.20)
    assert b.n_cap_commission == 1


def test_spread_scales_with_the_setting():
    traded, px, hs = np.array([10_000.0]), np.array([100.0]), np.array([0.001])
    o = OPTIMISTIC.charge(traded, px, hs)
    r = REALISTIC.charge(traded, px, hs)
    p = PESSIMISTIC.charge(traded, px, hs)
    assert o.spread == 0.0
    assert r.spread == pytest.approx(10.0)     # 10bp of 10k
    assert p.spread == pytest.approx(20.0)


def test_missing_spread_uses_the_fallback_and_counts_it():
    b = REALISTIC.charge(np.array([10_000.0]), np.array([100.0]), np.array([np.nan]))
    assert b.n_spread_fallback == 1
    assert b.spread > 0


def test_free_model_charges_nothing():
    b = FREE.charge(np.array([10_000.0]), np.array([100.0]), np.array([0.01]))
    assert b.total == 0.0


# --------------------------------------------------------------------------
# Splits and spreads
# --------------------------------------------------------------------------

def test_cumulative_split_ratio_is_strictly_before_the_ex_date():
    dates = np.array(["2024-06-07", "2024-06-10", "2024-06-11"], dtype="<U10")
    splits = pd.DataFrame({"security_id": ["SID1"], "date": ["2024-06-10"], "value": [10.0]})
    out = cumulative_split_ratio(dates, {"SID1": 0}, splits)
    assert out[0, 0] == 10.0   # before the ex-date: quoted pre-split
    assert out[1, 0] == 1.0    # on the ex-date: already the new price
    assert out[2, 0] == 1.0


def test_tick_size_changes_at_decimalisation():
    d = np.array(["2001-04-06", "2001-04-09"], dtype="<U10")
    t = tick_size(d)
    assert t[0] == pytest.approx(0.0625)
    assert t[1] == pytest.approx(0.01)


def test_tick_floor_falls_as_price_rises():
    d = np.array(["2020-01-02"], dtype="<U10")
    f = tick_floor(d, np.array([[10.0, 200.0]]))
    assert f[0, 0] > f[0, 1]
    assert f[0, 1] == pytest.approx(2.0 * 0.01 / 2.0 / 200.0)


def test_corwin_schultz_detects_a_wide_spread():
    """A wide simulated spread must produce a larger estimate than a narrow one."""
    rng = np.random.default_rng(0)
    n = 400
    mid = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))

    def series(spread):
        half = mid * spread / 2
        intraday = np.abs(rng.normal(0, 0.005, n)) * mid
        return (mid + intraday + half, mid - intraday - half, mid)

    wide = np.nanmean(corwin_schultz(*[x[:, None] for x in series(0.02)]))
    narrow = np.nanmean(corwin_schultz(*[x[:, None] for x in series(0.0)]))
    assert wide > narrow


# --------------------------------------------------------------------------
# Delisting classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Lehman Brothers filed for bankruptcy.", "bankruptcy"),
    ("Airgas was acquired by Air Liquide.", "acquisition"),
    ("CVS acquired Aetna.", "acquisition"),
    ("Market capitalization changes.", "index_removal"),
    ("ConocoPhillips completed the corporate spin-off of Phillips 66.", "index_removal"),
    ("WarnerMedia and Discovery merged to create Warner Bros. Discovery.", "acquisition"),
    ("", "unresolved"),
    (None, "unresolved"),
])
def test_reason_classification(text, expected):
    category, assumption = classify_reason(text)
    assert category == expected
    assert assumption  # never silently empty - every row records what was assumed


def test_bankruptcy_and_removal_are_not_conflated():
    """The distinction the whole module exists for."""
    assert classify_reason("filed for bankruptcy")[0] != \
           classify_reason("Market capitalization changes.")[0]


# --------------------------------------------------------------------------
# Metrics and multiple-testing control
# --------------------------------------------------------------------------

def test_norm_ppf_inverts_norm_cdf():
    for p in (0.01, 0.1, 0.5, 0.9, 0.975, 0.999):
        assert norm_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-8)


def test_more_trials_raises_the_bar():
    """The whole point of the deflated Sharpe: searching harder demands more evidence."""
    a = expected_max_sharpe(10, 0.5)
    b = expected_max_sharpe(10_000, 0.5)
    assert b > a > 0


def test_deflated_sharpe_falls_as_trials_rise():
    args = dict(sharpe=0.30, n_obs=230, skew=-0.2, kurtosis=4.0)
    few = deflated_sharpe(**args, n_trials=5, trial_sharpe_std=0.15)
    many = deflated_sharpe(**args, n_trials=10_000, trial_sharpe_std=0.15)
    assert few > many
    assert 0.0 <= many <= 1.0


def test_probabilistic_sharpe_rises_with_sample_size():
    a = probabilistic_sharpe(0.15, 30, 0.0, 3.0)
    b = probabilistic_sharpe(0.15, 500, 0.0, 3.0)
    assert b > a


def test_negative_skew_and_fat_tails_reduce_confidence():
    normal = probabilistic_sharpe(0.2, 200, 0.0, 3.0)
    ugly = probabilistic_sharpe(0.2, 200, -1.5, 12.0)
    assert ugly < normal


# --------------------------------------------------------------------------
# Integration - needs the ingested data
# --------------------------------------------------------------------------

def _have_data() -> bool:
    try:
        from sp500lab.storage import silver_exists
        return silver_exists("market/daily_bars_adjusted")
    except Exception:  # noqa: BLE001
        return False


needs_data = pytest.mark.skipif(not _have_data(),
                                reason="silver layer not built; run `sp500lab ingest all`")


@needs_data
def test_acceptance_suite_passes():
    """The gate. If this fails, no downstream number from this engine is meaningful."""
    from sp500lab.backtest import accept
    checks = accept.run_all()
    failed = [c.line() for c in checks if not c.passed]
    assert not failed, "\n".join(failed)


@needs_data
def test_backtest_is_fast_enough_for_a_genetic_algorithm():
    """A GA run is 10,000 of these. Anything above ~1s makes evolution impractical."""
    import time
    from sp500lab.backtest import build_panel, run_backtest as run
    p = build_panel()
    run("momentum_12_1", panel=p, benchmark=None, track_gross=False)  # warm caches
    t0 = time.perf_counter()
    for _ in range(3):
        run("momentum_12_1", panel=p, benchmark=None, track_gross=False)
    per_run = (time.perf_counter() - t0) / 3
    assert per_run < 1.0, f"{per_run:.2f}s per backtest is too slow for a GA"


@needs_data
def test_result_roundtrips_through_disk(tmp_path):
    from sp500lab.backtest import run_backtest as run
    from sp500lab.backtest.results import BacktestResult
    res = run("momentum_12_1", start="2015-01-01", costs="realistic")
    res.save(tmp_path / "r")
    back = BacktestResult.load(tmp_path / "r")
    assert back.strategy == res.strategy
    assert back.performance.cagr == pytest.approx(res.performance.cagr, abs=1e-6)
    assert np.allclose(back.equity.to_numpy(), res.equity.to_numpy())
