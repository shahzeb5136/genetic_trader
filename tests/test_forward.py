"""Tests for the forward-testing harness.

These matter for the same reason the registry's do: every failure mode here is
*silent* and most of them are *irreversible*.

  * A forward window measured from the wrong boundary reads research data and reports
    it as out-of-sample. Nothing errors; the number is simply a lie.
  * A "fresh months" count that is too high claims evidence that has already been seen.
    The second look then looks like a confirmation of the first.
  * A verdict computed from a raw Sharpe drop rather than from its standard error will
    call noise a decay on a 54-month window roughly a third of the time.
  * A deflated Sharpe quoted against a trial count of zero is the most flattering
    number this project can produce, and it looks exactly like a real one.

So the tests below check the guarantees rather than checking that the code runs.

Three tiers:

**Pure** - windows, statistics, comparison and verdict rules, on hand-built inputs.
No disk, no data, no panel. Every arithmetic claim in the module docstrings is pinned
here.

**Storage** - seals, records and curves round-tripped through `tmp_path`. Nothing in
this file writes to `data/experiments/`.

**Integration** - real forward tests on the ingested panel, with every ledger redirected
into `tmp_path`. Skipped when the silver layer is absent, and **skipped by default even
when it is present**: they read reserved data, and `conftest.py` is explicit that nothing
in the suite should. See `spends_holdout` below. Enable with:

    SP500LAB_FORWARD_TESTS=1 python -m pytest tests/test_forward.py
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest

from sp500lab.backtest import metrics
from sp500lab.backtest.registry import HOLDOUT_START, research_end
import sp500lab.forward.compare as C          # the module; `compare` is the function
import sp500lab.forward.seal as seal_module
import sp500lab.forward.store as store_module
import sp500lab.forward.windows as W
from sp500lab.forward.compare import Leg, compare

# --------------------------------------------------------------------------
# Fixtures and builders
# --------------------------------------------------------------------------


@pytest.fixture()
def forward_store(tmp_path, monkeypatch):
    """The forward store and the seal ledger, writing into tmp_path."""
    monkeypatch.setattr(store_module, "FORWARD_LOG", tmp_path / "forward_runs.jsonl")
    monkeypatch.setattr(store_module, "FORWARD_CURVE_LOG", tmp_path / "curves.jsonl")
    monkeypatch.setattr(seal_module, "SEAL_LOG", tmp_path / "seals.jsonl")
    return store_module


def leg(label="forward", n_months=54, sharpe=0.8, sharpe_monthly=None, cagr=0.09,
        bench_sharpe=0.6, bench_cagr=0.08, max_drawdown=-0.20, turnover=2.0,
        ruined=False, start="2022-02-01", end="2026-08-26") -> Leg:
    """A Leg with everything a comparison reads, and nothing it does not."""
    return Leg(
        label=label, start=start, end=end, n_months=n_months,
        cagr=cagr, sharpe=sharpe,
        sharpe_monthly=sharpe if sharpe_monthly is None else sharpe_monthly,
        skew_monthly=0.0, kurtosis_monthly=0.0, ann_vol=0.15,
        max_drawdown=max_drawdown, ann_turnover=turnover, cost_drag=0.01,
        hit_rate=0.55, avg_positions=50.0,
        bench_cagr=bench_cagr, bench_sharpe=bench_sharpe, ruined=ruined)


def curve(start: str, months: int, monthly_return: float = 0.01) -> pd.Series:
    """A daily NAV curve with a constant monthly compounding rate."""
    idx = pd.bdate_range(start, periods=months * 21)
    rate = (1 + monthly_return) ** (1 / 21) - 1
    values = np.cumprod(np.full(len(idx), 1 + rate)) / (1 + rate)
    return pd.Series(values, index=[d.strftime("%Y-%m-%d") for d in idx])


# ==========================================================================
# Pure: the windows
# ==========================================================================

def test_a_window_cannot_end_before_it_starts():
    with pytest.raises(ValueError):
        W.Window("2023-01-01", "2022-01-01")


@pytest.mark.parametrize("start,end,expected", [
    ("2022-01-01", "2022-12-31", 12),
    ("2022-01-01", "2022-01-30", 0),        # January's month end is the 31st
    ("2022-01-31", "2022-01-31", 1),
    ("2022-01-01", "2022-01-31", 1),
    ("2026-08-01", "2026-08-26", 0),        # the month has not ended yet
    ("2023-01-01", "2022-01-01", 0),        # backwards is empty, not negative
])
def test_month_ends_are_counted_exactly(start, end, expected):
    assert W.month_ends_between(start, end) == expected


def test_research_window_is_clamped_to_the_holdout_boundary():
    """Asking for more research data than exists must not silently widen training."""
    w = W.research_window("2007-04-01", end="2025-01-01")
    assert w.end == research_end()
    assert w.end < HOLDOUT_START


def test_forward_window_is_floored_at_the_holdout_boundary():
    """A forward test that began earlier would report research data as out-of-sample."""
    w = W.forward_window("2026-08-26", start="2015-01-01")
    assert w.start == HOLDOUT_START


def test_forward_window_refuses_when_the_data_stops_before_the_holdout():
    with pytest.raises(ValueError, match="no forward data"):
        W.forward_window("2021-06-30")


def test_first_look_is_entirely_fresh():
    w = W.forward_window("2026-08-26")
    fresh, months = W.freshness(w, None)
    assert fresh == w
    assert months == w.n_months


def test_a_repeat_look_at_unchanged_data_is_not_new_evidence():
    """Re-running against the same vintage is one measurement printed twice."""
    w = W.forward_window("2026-08-26")
    fresh, months = W.freshness(w, "2026-08-26")
    assert fresh is None
    assert months == 0


def test_only_the_months_that_arrived_since_the_last_look_are_fresh():
    w = W.Window("2022-01-01", "2026-08-26")
    fresh, months = W.freshness(w, "2024-12-31")
    assert fresh.start == "2025-01-01"
    assert fresh.end == "2026-08-26"
    assert months == W.month_ends_between("2025-01-01", "2026-08-26") == 19


def test_a_previous_look_older_than_the_window_leaves_everything_fresh():
    w = W.Window("2022-01-01", "2026-08-26")
    fresh, months = W.freshness(w, "2019-01-01")
    assert fresh == w and months == w.n_months


# ==========================================================================
# Pure: what a short window can prove
# ==========================================================================

def test_sharpe_standard_error_matches_the_closed_form_for_normal_returns():
    """Lo (2002): SE = sqrt((1 + SR^2/2) / (n - 1)) when skew is 0 and kurtosis is 3."""
    sr, n = 0.25, 60
    expected = math.sqrt((1 + sr ** 2 / 2) / (n - 1))
    assert metrics.sharpe_standard_error(sr, n) == pytest.approx(expected, rel=1e-12)


def test_the_probabilistic_sharpe_is_the_standard_error_it_prints_next_to():
    """PSR == norm_cdf((SR - SR0) / SE). If these two drift apart, both are wrong."""
    sr, n, skew, kurt = 0.3, 80, -0.4, 5.0
    se = metrics.sharpe_standard_error(sr, n, skew, kurt)
    for benchmark in (0.0, 0.1, 0.25):
        assert metrics.probabilistic_sharpe(sr, n, skew, kurt, benchmark) == \
            pytest.approx(metrics.norm_cdf((sr - benchmark) / se), rel=1e-12)


def test_the_confidence_interval_brackets_the_estimate_and_narrows_with_data():
    lo_short, hi_short = W.sharpe_band(1.0, 24)
    lo_long, hi_long = W.sharpe_band(1.0, 240)
    assert lo_short < 1.0 < hi_short
    assert lo_long < 1.0 < hi_long
    assert (hi_short - lo_short) > (hi_long - lo_long)


def test_fifty_odd_months_cannot_separate_a_good_sharpe_from_a_poor_one():
    """The headline fact about forward testing here, pinned so it cannot be softened."""
    lo, hi = W.sharpe_band(1.0, 54)
    assert lo < 0.2 and hi > 1.8


def test_the_detectable_gap_is_root_two_standard_errors_at_95_percent():
    n, sr = 54, 1.0
    se = metrics.sharpe_standard_error(sr / math.sqrt(12), n) * math.sqrt(12)
    expected = metrics.norm_ppf(0.975) * math.sqrt(2) * se
    assert W.detectable_sharpe_gap(n, sr) == pytest.approx(expected, rel=1e-12)
    assert W.detectable_sharpe_gap(n, sr) > 1.0     # bigger than most real effects


def test_describe_power_says_inconclusive_below_the_minimum():
    assert "inconclusive" in W.describe_power(W.MIN_FORWARD_MONTHS - 1)
    assert "inconclusive" not in W.describe_power(W.MIN_FORWARD_MONTHS + 12)


# ==========================================================================
# Pure: the comparison
# ==========================================================================

def test_a_leg_measures_itself_against_its_own_windows_benchmark():
    x = leg(cagr=0.12, bench_cagr=0.10, sharpe=0.9, bench_sharpe=0.7)
    assert x.excess == pytest.approx(0.02)
    assert x.d_sharpe == pytest.approx(0.2)


def test_decays_carry_the_sign_of_the_change():
    c = compare(leg("research", n_months=175, sharpe=1.0, cagr=0.14, max_drawdown=-0.40),
                leg("forward", n_months=54, sharpe=0.6, cagr=0.05, max_drawdown=-0.25))
    assert c.decay_sharpe == pytest.approx(-0.4)
    assert c.decay_cagr == pytest.approx(-0.09)
    assert c.decay_max_drawdown == pytest.approx(0.15)      # shallower is positive


def test_the_decay_standard_error_combines_both_windows():
    r = leg("research", n_months=175, sharpe=1.0)
    f = leg("forward", n_months=54, sharpe=0.4)
    c = compare(r, f)
    assert c.decay_se == pytest.approx(math.sqrt(r.sharpe_se ** 2 + f.sharpe_se ** 2))
    assert c.decay_z == pytest.approx(c.decay_sharpe_monthly / c.decay_se)
    # The forward window, being the shorter one, dominates the uncertainty.
    assert f.sharpe_se > r.sharpe_se


def test_the_forward_window_is_where_the_uncertainty_lives():
    """A 0.6 Sharpe drop over 54 months is barely one sigma. That is the whole problem."""
    c = compare(leg("research", n_months=175, sharpe=1.0),
                leg("forward", n_months=54, sharpe=0.4))
    assert -1.5 < c.decay_z < -1.0


def test_psr_against_research_is_stricter_than_psr_against_zero():
    c = compare(leg("research", n_months=175, sharpe=1.0),
                leg("forward", n_months=54, sharpe=0.6, bench_sharpe=0.3))
    assert c.psr_vs_zero > c.psr_vs_benchmark > c.psr_vs_research


# ---- the verdict rules, one test each ------------------------------------

def test_too_short_a_window_gets_no_verdict_at_all():
    c = compare(leg("research", n_months=175, sharpe=1.0),
                leg("forward", n_months=12, sharpe=3.0, cagr=0.40))
    assert c.verdict == "inconclusive"
    assert c.passed("enough_data") is False


def test_ruin_fails_whatever_else_the_numbers_say():
    c = compare(leg("research", n_months=175, sharpe=1.0),
                leg("forward", n_months=54, sharpe=2.0, cagr=0.30, ruined=True))
    assert c.verdict == "failed"
    assert "zero NAV" in c.verdict_reason


def test_losing_money_out_of_sample_fails():
    c = compare(leg("research", n_months=175, sharpe=1.0, cagr=0.14),
                leg("forward", n_months=54, sharpe=-0.2, cagr=-0.03))
    assert c.verdict == "failed"
    assert c.passed("made_money") is False


def test_a_statistically_significant_collapse_fails():
    """Long windows on both sides shrink the error until a real drop is detectable."""
    c = compare(leg("research", n_months=600, sharpe=2.0, cagr=0.20),
                leg("forward", n_months=600, sharpe=0.0, cagr=0.02))
    assert c.decay_z < -C.DECAY_FATAL_SIGMA
    assert c.verdict == "failed"


def test_losing_an_edge_it_used_to_have_is_a_decay():
    c = compare(leg("research", n_months=175, sharpe=0.9, bench_sharpe=0.7, cagr=0.12),
                leg("forward", n_months=54, sharpe=0.8, bench_sharpe=1.0, cagr=0.10))
    assert c.passed("kept_its_edge") is False
    assert c.verdict == "decayed"
    assert "no longer made it better than buying the index" in c.verdict_reason


def test_a_drop_beyond_one_sigma_is_a_decay_even_with_no_edge_to_lose():
    c = compare(leg("research", n_months=175, sharpe=1.0, bench_sharpe=1.2, cagr=0.12),
                leg("forward", n_months=54, sharpe=0.4, bench_sharpe=1.2, cagr=0.05))
    assert c.passed("kept_its_edge") is None      # it never had one
    assert c.passed("decay_within_noise") is False
    assert c.verdict == "decayed"


def test_holding_up_is_reported_as_not_refuted_rather_than_as_confirmed():
    c = compare(leg("research", n_months=175, sharpe=0.9, bench_sharpe=0.7, cagr=0.12,
                    max_drawdown=-0.40),
                leg("forward", n_months=54, sharpe=0.95, bench_sharpe=0.8, cagr=0.13,
                    max_drawdown=-0.20))
    assert c.verdict == "held"
    assert "not refuted" in c.verdict_reason


def test_a_strategy_that_never_beat_the_index_is_not_penalised_for_still_not_doing_so():
    """`kept_its_edge` is about keeping an edge, so it abstains when there was none."""
    r = leg("research", sharpe=0.4, bench_sharpe=0.7)
    f = leg("forward", sharpe=0.4, bench_sharpe=0.7)
    assert C._kept_edge(r, f) is None


def test_a_doubling_of_turnover_is_flagged_as_a_change_of_behaviour():
    c = compare(leg("research", n_months=175, turnover=2.0),
                leg("forward", n_months=54, turnover=5.0))
    assert c.turnover_ratio == pytest.approx(2.5)
    assert c.passed("turnover_held") is False


def test_checks_abstain_rather_than_guess_when_a_number_is_missing():
    """A missing benchmark must not read as 'did not beat the benchmark'."""
    c = compare(leg("research", n_months=175),
                leg("forward", n_months=54, bench_sharpe=float("nan"),
                    bench_cagr=float("nan")))
    assert c.passed("beat_benchmark") is None
    assert c.passed("positive_excess") is None


def test_the_comparison_flattens_to_scalars_for_a_table():
    c = compare(leg("research", n_months=175), leg("forward", n_months=54))
    flat = c.as_flat_dict()
    for key in ("decay_sharpe_monthly", "decay_z", "decay_p", "psr_vs_research",
                "verdict", "forward_band_low", "turnover_ratio"):
        assert key in flat
    assert isinstance(flat["checks"], dict)


# ==========================================================================
# Storage: seals
# ==========================================================================

def test_the_seal_id_is_stable_across_processes_and_dict_ordering():
    a = seal_module.seal_id_for(strategy_class="X", params={"a": 1, "b": 2},
                                construction=None, cost_model="realistic",
                                initial_capital=1e5, liquidity_floor=0.0, seed=0)
    b = seal_module.seal_id_for(strategy_class="X", params={"b": 2, "a": 1},
                                construction=None, cost_model="realistic",
                                initial_capital=1e5, liquidity_floor=0.0, seed=0)
    assert a == b


@pytest.mark.parametrize("change", [
    {"params": {"a": 2}}, {"cost_model": "pessimistic"}, {"seed": 1},
    {"initial_capital": 50_000.0}, {"construction": {"top_k": 10}},
])
def test_anything_that_changes_the_strategy_changes_the_seal_id(change):
    base = dict(strategy_class="X", params={"a": 1}, construction=None,
                cost_model="realistic", initial_capital=1e5, liquidity_floor=0.0, seed=0)
    assert seal_module.seal_id_for(**base) != seal_module.seal_id_for(**(base | change))


def test_a_seal_cannot_be_built_from_a_run_that_saw_the_holdout(forward_store):
    """A prediction measured on the period it is about to be tested against is not one."""
    from test_registry import make_result
    result = make_result(holdout_mode="only", touched=True)
    with pytest.raises(ValueError, match="saw holdout data"):
        seal_module.create_seal(result, rationale="nope")


def test_the_earliest_seal_binds_so_a_disappointment_cannot_rewrite_it(forward_store):
    from test_registry import make_result
    first = seal_module.create_seal(make_result(strategy="s"), rationale="written first")
    seal_module.record(first)
    later = seal_module.create_seal(make_result(strategy="s"), rationale="written after")
    later.sealed_at = "2099-01-01T00:00:00Z"
    seal_module.record(later)

    assert first.seal_id == later.seal_id
    assert seal_module.get(first.seal_id).rationale == "written first"
    assert len(seal_module.history(first.seal_id)) == 2


def test_an_unattributed_candidate_reports_no_deflated_sharpe(forward_store):
    """Zero trials must never be dressed up as a survived deflation - it is a PSR."""
    from test_registry import make_result
    s = seal_module.create_seal(make_result(), rationale="no study")
    assert s.study is None
    assert s.n_trials == 0
    assert math.isnan(s.deflated_sharpe)


def test_seals_load_flat_with_the_prediction_alongside(forward_store):
    from test_registry import make_result
    seal_module.record(seal_module.create_seal(make_result(strategy="alpha"),
                                               rationale="why not"))
    df = seal_module.load_seals()
    assert len(df) == 1
    assert df.loc[0, "strategy"] == "alpha"
    assert "research_sharpe" in df.columns and "research_n_months" in df.columns


# ==========================================================================
# Storage: records and curves
# ==========================================================================

def record(store, *, strategy="s", seal_id="seal-1", cost_model="realistic",
           data_end="2026-08-26", forward_end="2026-08-26", mode="paired",
           forward_sharpe=0.8, verdict="held"):
    r = leg("research", n_months=175, sharpe=1.0)
    f = leg("forward", n_months=54, sharpe=forward_sharpe, end=forward_end)
    c = compare(r, f)
    return store.record(store.ForwardRecord(
        forward_id=store.new_forward_id(), logged_at="2026-08-28T00:00:00Z",
        batch_id="batch-1", seal_id=seal_id, seal_mode="auto", strategy=strategy,
        cost_model=cost_model, mode=mode, data_end=data_end,
        research=r.as_dict(), research_recomputed=r.as_dict(), forward=f.as_dict(),
        comparison=c.as_flat_dict(), verdict=verdict))


def test_a_record_round_trips_and_flattens_both_legs(forward_store):
    written = record(forward_store, strategy="momo")
    back = forward_store.get(written.forward_id)
    assert back.strategy == "momo"
    assert back.forward_leg().sharpe == pytest.approx(0.8)

    df = forward_store.load()
    assert len(df) == 1
    for col in ("research_sharpe", "forward_sharpe", "recomputed_sharpe",
                "decay_z", "verdict", "checks_failed"):
        assert col in df.columns


def test_look_number_counts_per_candidate_per_cost_setting(forward_store):
    """Three cost settings of one strategy are one experiment, not three looks."""
    for cost in ("optimistic", "realistic", "pessimistic"):
        record(forward_store, cost_model=cost)
    assert forward_store.look_number("seal-1", "realistic") == 2
    assert forward_store.look_number("seal-1", "optimistic") == 2
    assert forward_store.look_number("seal-1", "realistic", mode="continuous") == 1
    assert forward_store.look_number("seal-other", "realistic") == 1


def test_the_vintage_is_what_the_last_look_SAW_not_what_the_panel_held(forward_store):
    """A run stopped early must not retire months no look has ever covered."""
    record(forward_store, data_end="2026-08-26", forward_end="2024-12-31")
    assert forward_store.previous_data_end("seal-1", "realistic") == "2024-12-31"


def test_curves_round_trip_rebased_per_window(forward_store):
    nav_r, nav_f = curve("2007-05-01", 60), curve("2022-02-01", 30, 0.005)
    forward_store.save_curves("fwd-1", strategy="s", seal_id="seal-1",
                              cost_model="realistic",
                              research={"nav": nav_r}, forward={"nav": nav_f})
    got = forward_store.load_curves(["fwd-1"])["fwd-1"]
    assert set(got) == {"research", "forward"}
    for window in got.values():
        assert window["nav"].iloc[0] == pytest.approx(1.0)


def test_the_curve_keeps_the_opening_level_when_the_window_starts_mid_month(
        forward_store):
    """Rebasing to the first month end would silently drop the opening month."""
    nav = curve("2022-02-01", 12, 0.01)
    forward_store.save_curves("fwd-1", strategy="s", seal_id="seal-1",
                              cost_model="realistic", research={}, forward={"nav": nav})
    stored = forward_store.load_curves(["fwd-1"])["fwd-1"]["forward"]
    assert stored.index[0] == "2022-02-01"
    assert stored["nav"].iloc[0] == pytest.approx(1.0)


def test_the_stitched_curve_is_continuous_across_the_join(forward_store):
    nav_r, nav_f = curve("2007-05-01", 60), curve("2022-02-01", 30, 0.005)
    forward_store.save_curves("fwd-1", strategy="s", seal_id="seal-1",
                              cost_model="realistic",
                              research={"nav": nav_r}, forward={"nav": nav_f})
    spliced = forward_store.stitched_curve("fwd-1")
    join = spliced.attrs["join_date"]
    assert join == "2022-02-01"
    research = forward_store.load_curves(["fwd-1"])["fwd-1"]["research"]["nav"]
    # The forward leg is spliced onto the research leg's last value, so the join
    # neither jumps nor resets to 1.0.
    assert spliced.loc[join] == pytest.approx(float(research.iloc[-1]))
    assert spliced.iloc[0] == pytest.approx(1.0)
    assert spliced.index.is_monotonic_increasing


def test_the_annual_table_keeps_the_partial_first_year(forward_store):
    """2022 is the most informative year in the current holdout. It must not vanish."""
    nav = curve("2022-02-01", 30, 0.01)
    forward_store.save_curves("fwd-1", strategy="s", seal_id="seal-1",
                              cost_model="realistic", research={}, forward={"nav": nav})
    stored = forward_store.load_curves(["fwd-1"])["fwd-1"]["forward"]["nav"]
    in_2022 = [d for d in stored.index if d.startswith("2022")]

    table = forward_store.annual_table("fwd-1")
    assert list(table["year"])[0] == "2022"
    # The curve is rebased to 1.0 at the opening session, so 2022's return is simply
    # its December level minus one. Measuring from the first month end instead would
    # drop February, and the two differ.
    assert table.loc[0, "strategy"] == pytest.approx(float(stored[in_2022[-1]]) - 1.0)
    from_first_month_end = float(stored[in_2022[-1]]) / float(stored[in_2022[1]]) - 1
    assert table.loc[0, "strategy"] != pytest.approx(from_first_month_end)


def test_the_selection_bar_counts_candidates_not_runs(forward_store):
    """Three cost settings of one candidate are one hypothesis about the forward window."""
    for cost in ("optimistic", "realistic", "pessimistic"):
        record(forward_store, cost_model=cost, seal_id="seal-1")
    assert forward_store.selection_bar(cost_model="realistic")["n_forward_tests"] == 1


def test_the_selection_bar_rises_as_more_candidates_are_looked_at(forward_store):
    for i, sharpe in enumerate([0.2, 0.5, 0.9, 1.3, 1.7]):
        record(forward_store, seal_id=f"seal-{i}", forward_sharpe=sharpe)
    bar = forward_store.selection_bar(cost_model="realistic")
    assert bar["n_forward_tests"] == 5
    assert bar["bar"] > 0
    assert bar["spread"] > 0


def test_an_empty_store_answers_rather_than_raising(forward_store):
    assert forward_store.load().empty
    assert forward_store.scoreboard().empty
    assert forward_store.selection_bar()["n_forward_tests"] == 0
    assert forward_store.get("nope") is None
    assert forward_store.stitched_curve("nope") is None


def test_the_scoreboard_ranks_on_the_index_relative_column(forward_store):
    """Ranking on the raw Sharpe across differing windows ranks the market."""
    record(forward_store, seal_id="a", strategy="weak", forward_sharpe=0.3)
    record(forward_store, seal_id="b", strategy="strong", forward_sharpe=1.4)
    board = forward_store.scoreboard()
    assert list(board["strategy"]) == ["strong", "weak"]


# ==========================================================================
# Integration - the real panel, with every ledger redirected
# ==========================================================================

def _have_data() -> bool:
    try:
        from sp500lab.storage import silver_exists
        return silver_exists("market/daily_bars_adjusted")
    except Exception:  # noqa: BLE001
        return False


needs_data = pytest.mark.skipif(not _have_data(),
                                reason="silver layer not built; run `sp500lab ingest all`")


def _forward_tests_enabled() -> bool:
    return os.environ.get("SP500LAB_FORWARD_TESTS", "").strip().lower() in (
        "1", "on", "true", "yes")


#: Opt-in, and the reason is a principle rather than runtime.
#:
#: `tests/conftest.py` says it plainly: "nothing in the test suite should be touching the
#: holdout, and if something starts to, the ledger is how it gets noticed." A test that
#: runs `forward_test` DOES touch it. Redirecting the ledger into `tmp_path` keeps the
#: project's record clean, which is necessary - but it also removes the very mechanism
#: that was supposed to notice, and a look that happened is a look that happened whether
#: or not anybody wrote it down.
#:
#: So these are off by default. `pytest` on a fresh clone exercises 57 tests covering
#: every window rule, every statistic, every verdict and every storage guarantee, and
#: reads no reserved data at all. Turn them on deliberately when changing the engine:
#:
#:     SP500LAB_FORWARD_TESTS=1 python -m pytest tests/test_forward.py
#:
#: They assert on structure - boundaries, ledger entries, round-trips - and never on a
#: performance figure, so running them tells you the harness works without telling you
#: how any strategy did.
spends_holdout = pytest.mark.skipif(
    not _forward_tests_enabled(),
    reason="reads reserved holdout data; set SP500LAB_FORWARD_TESTS=1 to run")


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Every append-only log this package writes, redirected into tmp_path.

    Including the holdout ledger, which nothing else in the suite is allowed to
    redirect - and that is exactly why it is redirected here. A test run must never
    consume the project's one-shot holdout, and a forward test genuinely does look at
    it, so the only safe way to exercise the real path is to move the ledger.
    """
    from sp500lab.backtest import registry
    from sp500lab.forward import engine

    monkeypatch.setattr(registry.store, "EXPERIMENT_LOG", tmp_path / "runs.jsonl")
    monkeypatch.setattr(registry.store, "HOLDOUT_LOG", tmp_path / "holdout.jsonl")
    monkeypatch.setattr(registry.store, "CURVE_LOG", tmp_path / "curves.jsonl")
    monkeypatch.setattr(store_module, "FORWARD_LOG", tmp_path / "forward.jsonl")
    monkeypatch.setattr(store_module, "FORWARD_CURVE_LOG", tmp_path / "fcurves.jsonl")
    monkeypatch.setattr(seal_module, "SEAL_LOG", tmp_path / "seals.jsonl")
    monkeypatch.setattr(engine, "FORWARD_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(engine, "_REGISTRY_SNAPSHOT", None)
    monkeypatch.setenv("SP500LAB_REGISTRY", "on")
    return engine


@needs_data
@spends_holdout
def test_a_forward_test_records_exactly_one_look_per_cost_setting(sandbox):
    from sp500lab.backtest import registry

    test = sandbox.forward_test("momentum_12_1", costs=("realistic",), save=False,
                                rationale="integration test")
    assert len(test.outcomes) == 1
    assert registry.holdout_touch_count() == 1

    touch = registry.holdout_touches().iloc[0]
    assert touch["mode"] == "only"
    assert touch["start"] == HOLDOUT_START


@needs_data
@spends_holdout
def test_the_research_leg_never_touches_the_holdout(sandbox):
    """The prediction has to be measurable without looking at the answer."""
    test = sandbox.forward_test("momentum_12_1", costs=("realistic",), save=False)
    outcome = test.outcomes[0]
    assert outcome.research.config["touched_holdout"] is False
    assert outcome.research.config["end"] <= research_end()
    assert outcome.forward.config["start"] >= HOLDOUT_START


@needs_data
@spends_holdout
def test_the_two_legs_do_not_overlap_by_a_single_session(sandbox):
    test = sandbox.forward_test("momentum_12_1", costs=("realistic",), save=False)
    c = test.outcomes[0].comparison
    assert c.research.end < c.forward.start


@needs_data
@spends_holdout
def test_a_declared_seal_binds_and_the_forward_run_does_not_replace_it(sandbox):
    seals = sandbox.seal_candidate("low_vol", rationale="declared first",
                                   costs=("realistic",))
    assert seals[0].seal_mode == "declared"

    test = sandbox.forward_test("low_vol", costs=("realistic",), save=False,
                                rationale="ignored")
    assert test.seal.seal_mode == "declared"
    assert test.seal.rationale == "declared first"
    # Nothing was looked at while sealing, so the ledger has one entry, not two.
    from sp500lab.backtest import registry
    assert registry.holdout_touch_count() == 1


@needs_data
def test_sealing_alone_spends_nothing(sandbox):
    from sp500lab.backtest import registry
    sandbox.seal_candidate("low_vol", rationale="just the prediction",
                           costs=("realistic",))
    assert registry.holdout_touch_count() == 0


@needs_data
@spends_holdout
def test_a_second_look_at_unchanged_data_reports_no_fresh_evidence(sandbox):
    sandbox.forward_test("low_vol", costs=("realistic",), save=False)
    again = sandbox.forward_test("low_vol", costs=("realistic",), save=False)
    rec = again.outcomes[0].record
    assert rec.look_number == 2
    assert rec.fresh_months == 0


@needs_data
@spends_holdout
def test_continuous_mode_covers_the_same_forward_window_as_paired(sandbox):
    paired = sandbox.forward_test("equal_weight", costs=("realistic",), save=False)
    cont = sandbox.forward_test("equal_weight", costs=("realistic",), save=False,
                                mode="continuous")
    a = paired.outcomes[0].comparison.forward
    b = cont.outcomes[0].comparison.forward
    assert a.end == b.end
    # The continuous leg inherits a book instead of buying one, so it starts a month
    # earlier and pays no entry cost. Same window, slightly different question.
    assert b.start < a.start


@needs_data
@spends_holdout
def test_the_stored_record_is_enough_to_rebuild_the_comparison(sandbox):
    """A report years from now must not need the panel. This is that guarantee."""
    test = sandbox.forward_test("low_vol", costs=("realistic",), save=False)
    written = test.outcomes[0]

    rec = store_module.get(written.record.forward_id)
    rebuilt = compare(rec.research_leg(), rec.forward_leg())
    assert rebuilt.verdict == written.comparison.verdict
    assert rebuilt.decay_z == pytest.approx(written.comparison.decay_z)

    curves = store_module.load_curves([rec.forward_id])[rec.forward_id]
    assert set(curves) == {"research", "forward"}
    assert not store_module.annual_table(rec.forward_id).empty
    assert store_module.stitched_curve(rec.forward_id) is not None
