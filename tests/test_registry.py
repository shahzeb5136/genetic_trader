"""Tests for the experiment registry and the holdout guard.

These matter more than most tests here, because both mechanisms fail *silently* and
*irreversibly*:

  * A trial that is not logged cannot be recovered, and an under-counted `n_trials`
    makes the deflated Sharpe too generous - it tells you a result survived a search
    it did not survive.
  * A look at the holdout that is not recorded cannot be recovered either, and you end
    up trusting a test period you have already seen.

Neither produces an error at the time. So the tests below check the guarantees
directly rather than checking that the code runs.

Every test redirects the log files into `tmp_path`, so nothing here touches a real
search. `conftest.py` disables logging globally for the rest of the suite; these tests
re-enable it deliberately.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sp500lab.backtest import registry
from sp500lab.backtest.costs import CostBreakdown
from sp500lab.backtest.metrics import Performance
from sp500lab.backtest.results import BacktestResult


@pytest.fixture()
def reg(tmp_path, monkeypatch):
    """A registry writing into tmp_path, with logging switched on."""
    monkeypatch.setattr(registry.store, "EXPERIMENT_LOG", tmp_path / "runs.jsonl")
    monkeypatch.setattr(registry.store, "HOLDOUT_LOG", tmp_path / "holdout.jsonl")
    monkeypatch.setenv("SP500LAB_REGISTRY", "on")
    return registry


def test_the_isolation_fixture_actually_isolates(reg, tmp_path):
    """The fixture must redirect the REAL logs, not a copy of their names.

    Every other test in this file trusts `reg` to keep it away from
    data/experiments/. Both files there are append-only and irreplaceable - a
    stray trial is noise in someone's n_trials forever, and a stray holdout line
    says a period was looked at when it was not.

    The failure this guards against is silent. When the registry was split into a
    package, re-exporting the log paths from `registry/__init__.py` would have made
    `monkeypatch.setattr(registry, "HOLDOUT_LOG", ...)` patch a copy while
    `registry.store` went on writing to the real ledger: green tests, polluted data.
    The paths are deliberately not re-exported, so a patch aimed at the wrong module
    raises AttributeError - but that only helps if something checks, which is this.
    """
    from sp500lab.paths import EXPERIMENT_LOG, HOLDOUT_LOG

    assert not hasattr(registry, "EXPERIMENT_LOG"),         "re-exporting the log paths makes a patch on the package silently useless"
    assert not hasattr(registry, "HOLDOUT_LOG")
    assert not hasattr(registry, "CURVE_LOG")

    # what the modules that do the writing will actually open
    assert registry.store.EXPERIMENT_LOG == tmp_path / "runs.jsonl"
    assert registry.store.HOLDOUT_LOG == tmp_path / "holdout.jsonl"

    before = {p: (p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None
              for p in (EXPERIMENT_LOG, HOLDOUT_LOG)}

    reg.log(make_result(strategy="isolation_probe"), study="isolation-probe")
    reg.record_holdout_touch(strategy="isolation_probe", study="isolation-probe",
                             mode="only", start="2022-01-01", end="2022-12-31")

    assert (tmp_path / "runs.jsonl").exists(), "the trial went somewhere else"
    assert (tmp_path / "holdout.jsonl").exists(), "the ledger line went somewhere else"
    for path, was in before.items():
        now = (path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else None
        assert now == was, f"the test wrote to the real {path.name}"


def make_result(strategy="s", sharpe=1.0, params=None, seed=0, cost_model="realistic",
                start="2010-01-01", end="2021-12-31", n_months=120,
                holdout_mode="exclude", touched=False) -> BacktestResult:
    """A BacktestResult with a synthetic equity curve of a chosen Sharpe.

    Built by hand rather than by running the engine so the tests stay offline and the
    statistics are known exactly.
    """
    rng = np.random.default_rng(abs(hash((strategy, seed, sharpe))) % 2**32)
    months = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    mu = sharpe * 0.04 / np.sqrt(12)
    rets = rng.normal(mu, 0.04, n_months)
    daily_idx = pd.date_range(months[0], months[-1], freq="B")
    monthly_eq = pd.Series(np.cumprod(1 + rets), index=months)
    eq = monthly_eq.reindex(daily_idx.union(months)).interpolate().reindex(
        daily_idx.union(months))
    eq.index = [d.strftime("%Y-%m-%d") for d in eq.index]
    eq = eq.dropna()

    perf = Performance(
        start=str(eq.index[0]), end=str(eq.index[-1]), years=n_months / 12.0,
        n_periods=len(eq) - 1, total_return=float(eq.iloc[-1] - 1), cagr=0.08,
        ann_vol=0.15, sharpe=sharpe, sortino=sharpe * 1.2, max_drawdown=-0.30,
        max_drawdown_start=str(eq.index[0]), max_drawdown_end=str(eq.index[-1]),
        calmar=0.27, hit_rate=0.55, skew=-0.2, kurtosis=1.5,
        best_period=0.1, worst_period=-0.1, time_under_water=0.8,
        var_95=-0.02, cvar_95=-0.03)

    return BacktestResult(
        strategy=strategy,
        config={"start": start, "end": end, "cost_model": cost_model,
                "initial_capital": 100_000.0, "liquidity_floor": 0.0, "seed": seed,
                "n_rebalances": n_months, "holdout_mode": holdout_mode,
                "touched_holdout": touched,
                "strategy_detail": {"class": strategy.title(),
                                    "params": params or {}, "warmup": 0},
                "panel": {"n_dates": 100, "n_securities": 10, "n_bars": 900,
                          "end": "2026-08-26", "format_version": 5}},
        equity=eq, performance=perf,
        rebalances=pd.DataFrame({"date": [], "turnover": []}),
        costs=CostBreakdown(commission=10.0, spread=5.0, traded_notional=1e5, n_orders=50),
        diagnostics={"price_coverage": "76.8% median, 54.7% worst (273/499 on X)",
                     "runtime_seconds": 0.1, "forced_exits": "3 - {'acquisition': 3}"})


# --------------------------------------------------------------------------
# The holdout boundary
# --------------------------------------------------------------------------

def test_exclude_stops_before_the_holdout():
    start, end, touched = registry.apply_holdout(
        "2007-04-01", None, "exclude", "2026-08-26")
    assert end < registry.HOLDOUT_START
    assert end == "2021-12-31"
    assert touched is False


def test_include_reaches_the_holdout_and_is_flagged():
    start, end, touched = registry.apply_holdout(
        "2007-04-01", None, "include", "2026-08-26")
    assert end == "2026-08-26"
    assert touched is True


def test_only_runs_the_holdout_alone():
    start, end, touched = registry.apply_holdout(
        "2007-04-01", None, "only", "2026-08-26")
    assert start == registry.HOLDOUT_START
    assert touched is True


def test_exclude_never_extends_a_shorter_end():
    _, end, _ = registry.apply_holdout("2010-01-01", "2015-06-30", "exclude", "2026-08-26")
    assert end == "2015-06-30"


def test_unknown_holdout_mode_is_rejected():
    with pytest.raises(ValueError, match="holdout must be"):
        registry.apply_holdout("2010-01-01", None, "ignore-it", "2026-08-26")


def test_holdout_touch_is_recorded_even_when_logging_is_off(reg, monkeypatch):
    """The asymmetry that matters: trials are optional, holdout looks are not."""
    monkeypatch.setenv("SP500LAB_REGISTRY", "off")
    assert reg.enabled() is False
    reg.record_holdout_touch(strategy="x", study="s", mode="only",
                             start="2022-01-01", end="2026-08-26")
    assert reg.holdout_touch_count() == 1
    assert len(reg.holdout_touches()) == 1


def test_every_touch_appends_rather_than_replacing(reg):
    for _ in range(3):
        reg.record_holdout_touch(strategy="x", study=None, mode="include",
                                 start="2007-01-01", end="2026-08-26")
    assert reg.holdout_touch_count() == 3


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def test_run_is_logged_and_reloadable(reg):
    rec = reg.log(make_result("momentum"), study="test-study")
    assert rec is not None
    df = reg.load("test-study")
    assert len(df) == 1
    assert df.iloc[0]["strategy"] == "momentum"
    assert df.iloc[0]["run_id"] == rec.run_id


def test_logging_is_disabled_by_the_env_var(reg, monkeypatch):
    monkeypatch.setenv("SP500LAB_REGISTRY", "off")
    assert reg.log(make_result()) is None
    assert reg.load().empty


def test_force_overrides_the_env_var(reg, monkeypatch):
    monkeypatch.setenv("SP500LAB_REGISTRY", "off")
    assert reg.log(make_result(), force=True) is not None


def test_study_context_manager_tags_runs(reg):
    with reg.study("ga-run-1"):
        reg.log(make_result("a"))
        reg.log(make_result("b"))
    reg.log(make_result("c"))
    df = reg.load()
    assert set(df[df["study"] == "ga-run-1"]["strategy"]) == {"a", "b"}
    assert df[df["strategy"] == "c"].iloc[0]["study"] == reg.ADHOC_STUDY


def test_study_context_manager_restores_the_previous_study(reg):
    assert reg.current_study() is None
    with reg.study("outer"):
        with reg.study("inner"):
            assert reg.current_study() == "inner"
        assert reg.current_study() == "outer"
    assert reg.current_study() is None


def test_log_survives_a_truncated_line(reg):
    reg.log(make_result("good"), study="s")
    with open(reg.store.EXPERIMENT_LOG, "a", encoding="utf-8") as fh:
        fh.write('{"run_id": "truncated"')          # interrupted write
    reg.log(make_result("also-good"), study="s")
    df = reg.load("s")
    assert len(df) == 2, "a partial final line must not make the whole log unreadable"


# --------------------------------------------------------------------------
# Fingerprints and trial counting - the inputs to the deflated Sharpe
# --------------------------------------------------------------------------

def _fp(**kw):
    base = dict(strategy_class="C", params={"a": 1}, construction=None,
                start="2010-01-01", end="2021-12-31", cost_model="realistic",
                initial_capital=100_000.0, liquidity_floor=0.0, seed=0)
    return registry.fingerprint(**{**base, **kw})


def test_identical_configurations_share_a_fingerprint():
    assert _fp() == _fp()


@pytest.mark.parametrize("change", [
    {"params": {"a": 2}}, {"seed": 1}, {"cost_model": "pessimistic"},
    {"start": "2011-01-01"}, {"liquidity_floor": 1e6}, {"strategy_class": "D"},
])
def test_any_configuration_change_is_a_new_trial(change):
    assert _fp(**change) != _fp()


def test_param_order_does_not_change_the_fingerprint():
    a = _fp(params={"x": 1, "y": 2})
    b = _fp(params={"y": 2, "x": 1})
    assert a == b


def test_rerunning_the_same_config_is_one_trial_not_two(reg):
    """Counting log lines instead of configurations would over-deflate."""
    for _ in range(4):
        reg.log(make_result("a", params={"k": 1}), study="s")
    df = reg.load("s")
    assert len(df) == 4
    assert reg.count_trials("s") == 1


def test_distinct_configs_count_as_distinct_trials(reg):
    for k in range(5):
        reg.log(make_result("a", params={"k": k}), study="s")
    assert reg.count_trials("s") == 5


def test_trials_are_scoped_to_their_study(reg):
    for k in range(3):
        reg.log(make_result("a", params={"k": k}), study="one")
    for k in range(7):
        reg.log(make_result("b", params={"k": k}), study="two")
    assert reg.count_trials("one") == 3
    assert reg.count_trials("two") == 7


def test_data_version_does_not_create_a_new_trial(reg):
    """Re-running one hypothesis on refreshed data is still one hypothesis.

    The fingerprint takes no panel metadata, so two runs of the same configuration on
    different vintages of the data count as one trial - which is right: you tested one
    idea. The vintage is kept separately in `data_fingerprint`.
    """
    a = make_result("a", params={"k": 1})
    b = make_result("a", params={"k": 1})
    b.config["panel"] = {"n_dates": 999, "n_securities": 42, "n_bars": 1,
                         "end": "2030-01-01", "format_version": 5}
    ra, rb = reg.log(a, study="s"), reg.log(b, study="s")
    assert ra.fingerprint == rb.fingerprint
    assert ra.data_fingerprint != rb.data_fingerprint
    assert reg.count_trials("s") == 1


# --------------------------------------------------------------------------
# Monthly statistics - why they are stored separately
# --------------------------------------------------------------------------

def test_monthly_stats_use_months_not_days(reg):
    """~120 monthly observations, not ~2600 daily ones. See the module docstring."""
    res = make_result(n_months=120)
    stats = reg.monthly_stats(res.equity)
    assert 100 <= stats["n_months"] <= 125
    assert len(res.equity) > 2000


def test_monthly_stats_degrade_gracefully_on_a_short_curve(reg):
    tiny = pd.Series([1.0, 1.1, 1.2], index=["2020-01-31", "2020-02-28", "2020-03-31"])
    stats = reg.monthly_stats(tiny)
    assert stats["n_months"] == 0
    assert np.isnan(stats["sharpe"])


# --------------------------------------------------------------------------
# Deflation
# --------------------------------------------------------------------------

def test_more_trials_lowers_the_deflated_sharpe(reg):
    """The whole point: a wider search demands more evidence from the winner."""
    for k in range(3):
        reg.log(make_result("a", params={"k": k}, sharpe=0.4 + 0.1 * k), study="small")
    for k in range(60):
        reg.log(make_result("b", params={"k": k}, sharpe=0.4 + 0.01 * k), study="large")

    small = reg.deflate_best("small")
    large = reg.deflate_best("large")
    assert small["n_trials"] == 3
    assert large["n_trials"] == 60
    assert large["expected_max_sharpe_annualised"] > small["expected_max_sharpe_annualised"]


def test_deflated_sharpe_is_never_above_the_undeflated_one(reg):
    for k in range(20):
        reg.log(make_result("a", params={"k": k}, sharpe=0.3 + 0.05 * k), study="s")
    out = reg.deflate_best("s")
    assert out["deflated_sharpe"] <= out["psr_vs_zero"] + 1e-9


def test_single_trial_study_gets_no_deflation(reg):
    """One trial is not a search, so there is nothing to deflate against."""
    reg.log(make_result("only-one", sharpe=1.0), study="s")
    out = reg.deflate_best("s")
    assert out["n_trials"] == 1
    assert out["expected_max_sharpe_annualised"] == 0.0
    assert out["deflated_sharpe"] == pytest.approx(out["psr_vs_zero"], abs=1e-9)


def test_deflate_uses_monthly_not_daily_observations(reg):
    rec = reg.log(make_result("a", n_months=120), study="s")
    out = reg.deflate(rec.run_id)
    assert out["n_months"] < 200, "must deflate on months, not ~2600 daily points"


def test_deflate_by_run_id_and_by_row_agree(reg):
    rec = reg.log(make_result("a"), study="s")
    row = reg.get(rec.run_id)
    assert reg.deflate(rec.run_id)["deflated_sharpe"] == \
           reg.deflate(row)["deflated_sharpe"]


def test_deflate_on_an_unknown_run_raises(reg):
    with pytest.raises(KeyError):
        reg.deflate("no-such-run")


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------

def test_studies_separates_runs_from_trials(reg):
    reg.log(make_result("a", params={"k": 1}), study="s")
    reg.log(make_result("a", params={"k": 1}), study="s")   # a re-run
    reg.log(make_result("a", params={"k": 2}), study="s")
    row = reg.studies().iloc[0]
    assert row["runs"] == 3
    assert row["trials"] == 2


def test_studies_counts_holdout_exposure(reg):
    reg.log(make_result("a", touched=True), study="s")
    reg.log(make_result("b"), study="s")
    assert int(reg.studies().iloc[0]["touched_holdout"]) == 1


def test_best_picks_the_top_sharpe(reg):
    for k, sr in enumerate([0.2, 0.9, 0.5]):
        reg.log(make_result(f"s{k}", params={"k": k}, sharpe=sr), study="s")
    assert reg.best("s")["strategy"] == "s1"


def test_empty_registry_returns_empty_frames_not_errors(reg):
    assert reg.load().empty
    assert reg.studies().empty
    assert reg.count_trials("nothing") == 0
    assert reg.trial_sharpe_std("nothing") == 0.0


def test_record_is_json_serialisable(reg):
    """inf and numpy scalars must not break the log."""
    res = make_result()
    res.performance.calmar = float("inf")
    rec = reg.log(res, study="s")
    json.dumps(rec.as_dict())
    assert np.isnan(rec.calmar)


def test_git_state_reports_dirtiness():
    commit, dirty = registry.git_state()
    assert isinstance(commit, str) and isinstance(dirty, bool)


# --------------------------------------------------------------------------
# End to end through the engine
# --------------------------------------------------------------------------

def _have_data() -> bool:
    try:
        from sp500lab.storage import silver_exists
        return silver_exists("market/daily_bars_adjusted")
    except Exception:  # noqa: BLE001
        return False


needs_data = pytest.mark.skipif(not _have_data(), reason="silver layer not built")


@needs_data
def test_engine_defaults_to_the_research_window():
    """The default must protect the holdout without anyone remembering to ask."""
    from sp500lab.backtest import run_backtest
    res = run_backtest("equal_weight", log_run=False)
    assert res.equity.index[-1] < registry.HOLDOUT_START
    assert res.config["touched_holdout"] is False


@needs_data
def test_engine_logs_a_run_and_stamps_it(reg):
    from sp500lab.backtest import run_backtest
    res = run_backtest("equal_weight", start="2015-01-01", study="engine-test")
    assert res.config["run_id"]
    assert res.config["study"] == "engine-test"
    assert reg.count_trials("engine-test") == 1


@needs_data
def test_holdout_only_starts_inside_the_holdout(reg):
    from sp500lab.backtest import run_backtest
    res = run_backtest("equal_weight", holdout="only", log_run=False)
    assert res.equity.index[0] >= registry.HOLDOUT_START
    assert reg.holdout_touch_count() == 1
