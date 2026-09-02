"""Running a forward test: measure the prediction, spend the holdout, write it down.

    from sp500lab.forward import forward_test
    test = forward_test("momentum_12_1", rationale="best d_sharpe in the suite")
    print(test.summary())

What one call does, in order
-----------------------------
1. **Backtest the research window** (`holdout="exclude"`, so it stops 2021-12-31). No
   look is spent here; this is the prediction.
2. **Find or write the seal.** A `declared` seal made earlier binds and its numbers are
   the prediction. Otherwise an `auto` seal is written from step 1 - honest numbers,
   unproven ordering. See `seal.py`.
3. **Backtest the forward window** (`holdout="only"`). This is the look. The existing
   holdout ledger records it before the run starts and cannot be silenced (ADR-025);
   nothing in this module reimplements that or gets a say in it.
4. **Compare** the two legs, with standard errors on the difference (`compare.py`).
5. **Store** the record, both month-end curves, and - by default for a single strategy -
   the whole forward result including its order-by-order trade ledger.

Two modes, and the difference is real money
--------------------------------------------
``paired`` *(default)*
    Two independent runs. The forward leg starts from a fresh 100k with an empty book on
    the first month end of 2022, so it pays entry costs for its whole portfolio in the
    first month. That makes the forward leg *slightly worse* than a continuous path, and
    it is the honest simulation of "I read the research and started trading in 2022".

``continuous``
    One unbroken backtest from 2007 to today, sliced at the boundary afterwards. No
    re-entry cost, and the forward leg inherits the exact book the research window ended
    with. That is the honest simulation of "I had been running this all along". It also
    costs the same single look at the holdout, because one run reaching into reserved
    data is one look however it is sliced.

Neither is more correct. They answer different questions, they differ by roughly one
month of entry cost, and reporting whichever came out better would be a choice made
after seeing the answer - so `mode` is recorded on the record.

Every cost setting in one look
-------------------------------
The default runs optimistic, realistic AND pessimistic in a single invocation. costs.py
insists all three always be reported, and this is the one place where fetching a missing
one later is genuinely expensive: the honest alternative would be a second look. Three
cost settings of one strategy are one hypothesis under three assumptions, so
`store.look_number` counts them separately per cost model rather than calling them three
looks.

What this module deliberately does NOT do
------------------------------------------
It does not search, tune, select or rank anything before the look. Everything it runs,
it runs on a configuration that already existed. The moment a forward result feeds back
into which strategy to build next, the holdout is a training set - and no code can
prevent that, which is why `store.selection_bar()` measures how much choosing has
already happened and prints it next to the scoreboard.

See ADR-033, ADR-034 and docs/FORWARD_TEST.md.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..backtest import registry
from ..backtest.engine import DEFAULT_START, run_backtest
from ..backtest.panel import Panel, build_panel
from ..backtest.registry import HOLDOUT_START, research_end
from ..backtest.results import BacktestResult
from ..backtest.strategy import Strategy, get_strategy
from ..paths import PROJECT_ROOT
from . import seal as seal_module
from . import store
from .compare import Comparison, Leg, compare
from .legs import leg_from_result, leg_from_slice, slice_curve
from .seal import Seal, create_seal
from .windows import (MIN_FORWARD_MONTHS, Window, describe_power, forward_window,
                      freshness)

log = logging.getLogger(__name__)

#: Study the forward runs are logged under. A real study name, not a bucket: it makes
#: `experiments deflate forward-test` compute exactly the right correction, because
#: n_trials there is the number of distinct candidates the forward window has been
#: asked about - which is the multiple-testing exposure of the holdout itself.
FORWARD_STUDY = "forward-test"

#: Study the research legs are logged under. Separate so a re-measurement taken to
#: establish a baseline never inflates the trial count of the search that produced the
#: candidate. Re-running one configuration is one fingerprint however often it happens
#: (ADR-026), so this study's trial count is the number of candidates, not the number of
#: forward tests.
BASELINE_STUDY = "forward-baseline"

#: All three, always. See the module docstring.
DEFAULT_COSTS: tuple[str, ...] = ("optimistic", "realistic", "pessimistic")

MODES = ("paired", "continuous")

#: Where a forward result's full artifacts land: equity, rebalances, weights, exits and
#: the order-by-order trade ledger as both parquet and CSV (`results.save`).
FORWARD_RESULTS_DIR = PROJECT_ROOT / "results" / "forward"


class ForwardError(RuntimeError):
    """The forward test cannot be run, or cannot be trusted. Never suppress this."""


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass
class Outcome:
    """One forward test under one cost setting: both runs, the comparison, the record."""

    cost_model: str
    record: store.ForwardRecord
    comparison: Comparison
    research: BacktestResult
    forward: BacktestResult

    @property
    def verdict(self) -> str:
        return self.comparison.verdict

    def summary(self) -> str:
        c = self.comparison
        f = c.forward
        lines = [
            f"  [{self.cost_model} costs]   verdict: {c.verdict.upper()}",
            "",
            c.summary(),
            "",
            f"  {'Sharpe 95% band':22s} [{c.forward_band_low:.2f}, "
            f"{c.forward_band_high:.2f}]  on {f.n_months} months of forward data",
            f"  {'decay significance':22s} {_sigma(c)}",
            f"  {'P(true SR > 0)':22s} {_prob(c.psr_vs_zero)}",
            f"  {'P(true SR > index)':22s} {_prob(c.psr_vs_benchmark)}",
            f"  {'P(true SR > research)':22s} {_prob(c.psr_vs_research)}",
            "",
            "  CHECKS",
        ]
        for chk in c.checks:
            lines.append(f"    {chk.mark:4s}  {chk.name:20s} {chk.detail}")
        lines += ["", f"  {c.verdict.upper()}: {c.verdict_reason}"]
        return "\n".join(lines)


@dataclass
class ForwardTest:
    """Everything one `forward_test()` call produced."""

    strategy: str
    seal: Seal
    mode: str
    window: Window
    fresh: Window | None
    fresh_months: int
    look_number: int
    outcomes: list[Outcome] = field(default_factory=list)
    elapsed: float = 0.0
    batch_id: str = ""

    @property
    def headline(self) -> Outcome | None:
        """The realistic-cost outcome, which is the one a summary should lead with."""
        for o in self.outcomes:
            if o.cost_model == "realistic":
                return o
        return self.outcomes[0] if self.outcomes else None

    def verdicts(self) -> dict[str, str]:
        return {o.cost_model: o.verdict for o in self.outcomes}

    def summary(self) -> str:
        head = self.headline
        lines = [
            "=" * 78,
            f"FORWARD TEST  {self.strategy}   mode={self.mode}",
            "=" * 78,
            f"  seal            {self.seal.seal_id}  ({self.seal.seal_mode})",
            f"  rationale       {self.seal.rationale or '(none given)'}",
            f"  research        {self.seal.research_leg().start}.."
            f"{self.seal.research_leg().end}  "
            f"({self.seal.research_leg().n_months} months, not a look)",
            f"  forward         {self.window.describe()} available; the first "
            f"rebalance is the",
            "                  first month end inside it and fills at the next open, "
            "so a run",
            "                  covers one month less - see the table below",
            f"  look number     #{self.look_number} at this candidate under each "
            f"cost setting",
        ]
        if self.look_number > 1:
            lines.append(
                "  NEW since last  "
                + (f"{self.fresh.describe()} - the only part of this result that is "
                   "fresh evidence" if self.fresh is not None else
                   "nothing. The data has not moved since the last look, so this is "
                   "the same measurement again, not a second confirmation."))
        if self.seal.study:
            lines.append(
                f"  search behind   study {self.seal.study}: {self.seal.n_trials} "
                f"trial(s), deflated Sharpe {_fmt(self.seal.deflated_sharpe)}")
        else:
            lines.append(
                "  search behind   UNKNOWN - the trial log has no earlier run of this "
                "exact configuration,")
            lines.append(
                "                  so the research Sharpe below is NOT corrected for "
                "the search that")
            lines.append(
                "                  produced the candidate. Pass --origin-study if you "
                "know it.")
        drift = head.record.seal_drift_sharpe if head else float("nan")
        if np.isfinite(drift) and abs(drift) > 0.01:
            lines.append(
                f"  !! seal drift   the research Sharpe now measures {drift:+.3f} "
                "against the sealed prediction; the data changed under it")
        months = (head.comparison.forward.n_months if head is not None
                  else self.window.n_months)
        lines += ["", f"  {describe_power(months)}", ""]

        for outcome in self.outcomes:
            lines += [outcome.summary(), ""]

        lines += [
            "  " + "-" * 74,
            "  Verdicts: " + "   ".join(f"{k}={v}" for k, v in self.verdicts().items()),
            "",
            "  'held' means fifty-odd months failed to refute it. It does not mean the",
            "  strategy works - see the Sharpe band above. Refutation is what a window",
            "  this short can do; confirmation is not.",
            f"  The holdout has now been looked at "
            f"{registry.holdout_touch_count()} time(s) in total "
            "(`sp500lab experiments holdout`).",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# The test
# --------------------------------------------------------------------------

def forward_test(
    strategy: Strategy | str,
    *,
    rationale: str = "",
    panel: Panel | None = None,
    research_start: str = DEFAULT_START,
    forward_start: str = HOLDOUT_START,
    forward_end: str | None = None,
    costs: tuple[str, ...] | list[str] | str = DEFAULT_COSTS,
    mode: str = "paired",
    initial_capital: float = 100_000.0,
    liquidity_floor: float = 0.0,
    seed: int = 0,
    benchmark: str = "SPY",
    study: str = FORWARD_STUDY,
    baseline_study: str = BASELINE_STUDY,
    origin_study: str | None = None,
    features=None,
    record_trades: bool = True,
    save: bool | str | Path = True,
    notes: str = "",
    strategy_kwargs: dict | None = None,
    min_forward_months: int = MIN_FORWARD_MONTHS,
    runner=None,
) -> ForwardTest:
    """Forward-test one strategy and record the result. This spends the holdout.

    Parameters
    ----------
    rationale       why this candidate is worth a look. Stored on the seal; the point of
                    asking is that a reason written before the answer reads differently
                    from one written after.
    costs           one name or several. Defaults to all three, in one look.
    mode            'paired' (two runs) or 'continuous' (one run, sliced). See above.
    origin_study    the search this candidate came out of, which decides the deflated
                    Sharpe carried on the seal. Left None it is recovered from the
                    registry by configuration fingerprint - see `_origin_study` - which
                    works when the candidate was run under its search's own name and
                    quietly gives up when it was not. Pass it when you know it.
    save            True writes the forward result's full artifacts under
                    `results/forward/<forward_id>/`; a path writes them there instead;
                    False skips them. The record and the curves are stored either way.
    min_forward_months
                    below this many months of forward data, every verdict is
                    'inconclusive'. Lower it only if you enjoy reading noise.
    runner          the backtest function, `run_backtest` by default. The calendar
                    lab passes its own leg engine here (timing/cli.py), which is how
                    a daily-granularity rule gets a real seal, a paired comparison
                    and a scoreboard row through exactly this machinery rather than
                    a parallel one that could drift. Any runner must accept the same
                    keyword surface and return a `BacktestResult`.

    Returns a `ForwardTest`. Raises `ForwardError` if there is no forward data at all -
    which is a data problem, not a result.
    """
    t0 = time.perf_counter()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    settings = (costs,) if isinstance(costs, str) else tuple(costs)
    if not settings:
        raise ValueError("need at least one cost setting")

    strat = (get_strategy(strategy, **(strategy_kwargs or {}))
             if isinstance(strategy, str) else strategy)
    name = getattr(strat, "name", type(strat).__name__)
    panel = panel or build_panel()
    data_end = str(panel.dates[-1])

    window = forward_window(data_end, start=forward_start, end=forward_end)
    if window.n_months < 1:
        raise ForwardError(
            f"the forward window {window.describe()} contains no month end. "
            "There is nothing to test yet.")

    batch_id = store.new_batch_id()
    outcomes: list[Outcome] = []
    seal: Seal | None = None
    fresh: Window | None = None
    fresh_months = window.n_months
    look = 1

    for cost_model in settings:
        outcome, seal, fresh, fresh_months, look = _one_setting(
            strat, name, cost_model, panel=panel, window=window, mode=mode,
            research_start=research_start, rationale=rationale,
            initial_capital=initial_capital, liquidity_floor=liquidity_floor,
            seed=seed, benchmark=benchmark, study=study,
            baseline_study=baseline_study, origin_study=origin_study,
            features=features,
            record_trades=record_trades, save=save, notes=notes,
            batch_id=batch_id, data_end=data_end,
            min_forward_months=min_forward_months, runner=runner or run_backtest)
        outcomes.append(outcome)

    assert seal is not None                     # at least one setting always runs
    return ForwardTest(
        strategy=name, seal=seal, mode=mode, window=window, fresh=fresh,
        fresh_months=fresh_months, look_number=look, outcomes=outcomes,
        elapsed=time.perf_counter() - t0, batch_id=batch_id)


def _one_setting(strat, name: str, cost_model: str, *, panel, window: Window, mode: str,
                 research_start: str, rationale: str, initial_capital: float,
                 liquidity_floor: float, seed: int, benchmark: str, study: str,
                 baseline_study: str, origin_study: str | None, features,
                 record_trades: bool, save, notes: str, batch_id: str, data_end: str,
                 min_forward_months: int, runner=run_backtest):
    """One cost setting, end to end. Returns (outcome, seal, fresh, fresh_months, look)."""
    common = dict(panel=panel, features=features, initial_capital=initial_capital,
                  liquidity_floor=liquidity_floor, seed=seed, benchmark=benchmark,
                  costs=cost_model)

    if mode == "continuous":
        research_result, forward_result, recomputed, forward_leg = _continuous(
            strat, common, research_start=research_start, window=window,
            study=study, baseline_study=baseline_study, notes=notes,
            record_trades=record_trades, runner=runner)
    else:
        research_result, forward_result, recomputed, forward_leg = _paired(
            strat, common, research_start=research_start, window=window,
            study=study, baseline_study=baseline_study, notes=notes,
            record_trades=record_trades, runner=runner)

    # ---- the seal: a declared one binds, otherwise write an auto one ---------
    bound = _bind_seal(research_result, rationale=rationale,
                       origin_study=origin_study,
                       exclude=(study, baseline_study))
    prediction = bound.research_leg()

    # ---- vintage: how much of this result is evidence nobody has seen -------
    # Measured against the window the run ACTUALLY covered, not the one that was
    # available. The first rebalance is the first month end inside the window and it
    # fills at the next open (ADR-019), so a run always starts a month after the
    # boundary - and "fresh months" has to mean months of this result, or the
    # scoreboard would claim one more month of evidence than the result contains.
    realised = _realised_window(forward_leg, window)
    previous = store.previous_data_end(bound.seal_id, cost_model, mode)
    fresh, fresh_months = freshness(realised, previous)
    look = store.look_number(bound.seal_id, cost_model, mode)

    comparison = compare(prediction, forward_leg,
                         min_forward_months=min_forward_months)

    forward_id = store.new_forward_id()
    saved_to = _save_artifacts(forward_result, forward_id, save)
    record = _build_record(
        forward_id=forward_id, batch_id=batch_id, seal=bound, mode=mode,
        cost_model=cost_model, comparison=comparison, prediction=prediction,
        recomputed=recomputed, forward_leg=forward_leg,
        research_result=research_result, forward_result=forward_result,
        study=study, notes=notes, data_end=data_end, previous_data_end=previous,
        fresh=fresh, fresh_months=fresh_months, look=look, saved_to=saved_to)
    store.record(record)
    store.save_curves(
        forward_id, strategy=name, seal_id=bound.seal_id, cost_model=cost_model,
        research=_curve_map(research_result),
        forward=_curve_map(forward_result, lo=window.start))

    return (Outcome(cost_model=cost_model, record=record, comparison=comparison,
                    research=research_result, forward=forward_result),
            bound, fresh, fresh_months, look)


def _paired(strat, common: dict, *, research_start: str, window: Window, study: str,
            baseline_study: str, notes: str, record_trades: bool,
            runner=run_backtest):
    """Two independent runs. The forward one starts flat, as a new deployment would."""
    research_result = runner(
        strat, start=research_start, end=research_end(), holdout="exclude",
        study=baseline_study, record_trades=False, log_curve=True,
        notes=f"forward-test baseline{': ' + notes if notes else ''}", **common)

    forward_result = runner(
        strat, start=window.start, end=window.end, holdout="only",
        study=study, record_trades=record_trades, log_curve=True,
        notes=f"forward test{': ' + notes if notes else ''}", **common)

    recomputed = leg_from_result(research_result, "research")
    forward_leg = leg_from_result(forward_result, "forward")
    return research_result, forward_result, recomputed, forward_leg


def _continuous(strat, common: dict, *, research_start: str, window: Window,
                study: str, baseline_study: str, notes: str, record_trades: bool,
                runner=run_backtest):
    """One unbroken run across the boundary, sliced afterwards.

    The research leg here is a slice of a run that DID see the holdout, so it cannot be
    used to build a seal - `create_seal` refuses it, correctly. The seal is therefore
    always taken from a separate research-only run, which costs one extra backtest and
    keeps the prediction provably free of forward data.
    """
    research_only = runner(
        strat, start=research_start, end=research_end(), holdout="exclude",
        study=baseline_study, record_trades=False, log_curve=True,
        notes=f"forward-test baseline (continuous){': ' + notes if notes else ''}",
        **common)

    whole = runner(
        strat, start=research_start, end=window.end, holdout="include",
        study=study, record_trades=record_trades, log_curve=True,
        notes=f"forward test, continuous{': ' + notes if notes else ''}", **common)

    recomputed = leg_from_slice(whole, "research", end=research_end())
    forward_leg = leg_from_slice(whole, "forward", start=window.start, end=window.end)
    return research_only, whole, recomputed, forward_leg


def _bind_seal(research_result: BacktestResult, *, rationale: str,
               origin_study: str | None, exclude: tuple[str, ...]) -> Seal:
    """The binding seal for this candidate: an earlier declared one, or a new auto one."""
    study = origin_study or _origin_study(
        str(research_result.config.get("fingerprint", "") or ""), exclude)
    candidate = create_seal(research_result, rationale=rationale, seal_mode="auto",
                            study=study)
    existing = seal_module.get(candidate.seal_id)
    if existing is not None:
        if rationale and not existing.rationale:
            log.info("seal %s already exists; the rationale given now is not recorded "
                     "against it - the earliest line binds", existing.seal_id)
        return existing
    return seal_module.record(candidate)


def _build_record(*, forward_id: str, batch_id: str, seal: Seal, mode: str,
                  cost_model: str, comparison: Comparison, prediction: Leg,
                  recomputed: Leg, forward_leg: Leg,
                  research_result: BacktestResult, forward_result: BacktestResult,
                  study: str, notes: str, data_end: str, previous_data_end: str | None,
                  fresh: Window | None, fresh_months: int, look: int,
                  saved_to: str) -> store.ForwardRecord:
    cfg = forward_result.config
    diag = forward_result.diagnostics
    costs = forward_result.costs
    commit, dirty = registry.git_state()

    return store.ForwardRecord(
        forward_id=forward_id,
        logged_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        batch_id=batch_id,
        seal_id=seal.seal_id,
        seal_mode=seal.seal_mode,
        strategy=seal.strategy,
        strategy_class=seal.strategy_class,
        params=seal.params,
        construction=seal.construction,
        mode=mode,
        cost_model=cost_model,
        initial_capital=float(cfg.get("initial_capital", 0.0)),
        liquidity_floor=float(cfg.get("liquidity_floor", 0.0)),
        seed=int(cfg.get("seed", 0)),
        benchmark=str(cfg.get("benchmark") or ""),
        rationale=seal.rationale,
        notes=notes,

        research=prediction.as_dict(),
        research_recomputed=recomputed.as_dict(),
        forward=forward_leg.as_dict(),
        comparison=comparison.as_flat_dict(),
        verdict=comparison.verdict,
        verdict_reason=comparison.verdict_reason,

        research_run_id=str(research_result.config.get("run_id", "") or ""),
        forward_run_id=str(cfg.get("run_id", "") or ""),

        data_end=data_end,
        previous_data_end=previous_data_end or "",
        fresh_start=fresh.start if fresh else "",
        fresh_months=int(fresh_months),
        look_number=int(look),

        study=seal.study,
        forward_study=study,
        n_trials=int(seal.n_trials),
        trial_sharpe_std=float(seal.trial_sharpe_std),
        deflated_sharpe_research=float(seal.deflated_sharpe),
        seal_drift_sharpe=_drift(recomputed, prediction),

        holdout_looks_total=registry.holdout_touch_count(),
        n_rebalances=int(cfg.get("n_rebalances", 0)),
        coverage_min=_coverage(diag, "min"),
        coverage_median=_coverage(diag, "median"),
        forced_exits=_count(diag.get("forced_exits")),
        unresolved_exits=_count(diag.get("!! unresolved_exits")),
        unfilled_orders=_count(diag.get("unfilled_orders")),
        spread_fallback_orders=int(diag.get("!! spread_fallback_orders", 0) or 0),
        ruined="!! ruined" in diag,
        total_cost=float(costs.total),
        commission=float(costs.commission),
        spread_cost=float(costs.spread),
        traded_notional=float(costs.traded_notional),
        n_orders=int(costs.n_orders),

        git_commit=commit,
        git_dirty=dirty,
        panel_key=f"{cfg.get('panel', {}).get('start', '')}.."
                  f"{cfg.get('panel', {}).get('end', '')}",
        data_fingerprint=registry.panel_fingerprint(cfg.get("panel", {})),
        runtime_seconds=float(diag.get("runtime_seconds", 0.0) or 0.0),
        saved_to=saved_to,
    )


# --------------------------------------------------------------------------
# Sealing on its own - pre-registration without spending anything
# --------------------------------------------------------------------------

def seal_candidate(strategy: Strategy | str, *, rationale: str,
                   panel: Panel | None = None, research_start: str = DEFAULT_START,
                   costs: str | tuple[str, ...] = "realistic",
                   initial_capital: float = 100_000.0,
                   liquidity_floor: float = 0.0, seed: int = 0,
                   benchmark: str = "SPY", study: str = BASELINE_STUDY,
                   origin_study: str | None = None, features=None,
                   strategy_kwargs: dict | None = None, runner=None) -> list[Seal]:
    """Pre-register a candidate. Runs the RESEARCH window only - no look is spent.

    Do this the day you decide what to test, and forward-test later. The gap between
    `sealed_at` and the forward record's `logged_at` is the evidence that the choice was
    not made after seeing the answer, and it is the only evidence of that there can be.

    One seal per cost setting, because `seal_id` includes the cost model - a prediction
    made under optimistic costs is a different prediction. Seal the settings you intend
    to forward-test; sealing all three costs three research backtests and nothing else.

    Returns the seals, in the order the cost settings were given.
    """
    strat = (get_strategy(strategy, **(strategy_kwargs or {}))
             if isinstance(strategy, str) else strategy)
    panel = panel or build_panel()
    settings = (costs,) if isinstance(costs, str) else tuple(costs)
    runner = runner or run_backtest

    out = []
    for cost_model in settings:
        result = runner(
            strat, panel=panel, features=features, start=research_start,
            end=research_end(), holdout="exclude", costs=cost_model,
            initial_capital=initial_capital, liquidity_floor=liquidity_floor,
            seed=seed, benchmark=benchmark, study=study, record_trades=False,
            notes=f"seal: {rationale}"[:200])
        origin = origin_study or _origin_study(
            str(result.config.get("fingerprint", "") or ""), (study,))
        candidate = create_seal(result, rationale=rationale, seal_mode="declared",
                                study=origin)
        existing = seal_module.get(candidate.seal_id)
        if existing is not None:
            log.info("%s is already sealed as %s (%s) - the earliest line binds, so "
                     "this one is not written again", getattr(strat, "name", strategy),
                     existing.seal_id, existing.sealed_at)
            out.append(existing)
            continue
        out.append(seal_module.record(candidate))
    return out


# --------------------------------------------------------------------------
# A whole group at once
# --------------------------------------------------------------------------

def forward_suite(names, origin_studies: dict[str, str] | None = None,
                  **kwargs) -> list[ForwardTest]:
    """Forward-test several candidates in one pass. Failures are reported, not fatal.

    `names` may hold registered strategy names, strategy objects, or both - an evolved
    winner lives in a search checkpoint rather than in the strategy registry, and it is
    a candidate like any other (see `evolve.winners`).

    `origin_studies` maps a candidate's name to the search it came out of, for the ones
    the trial log cannot identify by fingerprint. That is exactly the evolved winners,
    and they are the candidates where it matters most: a winner of 1,400 individuals and
    a hand-written baseline are different claims at identical Sharpes, and without the
    study name the record would carry `n_trials=0` for the two most heavily searched
    candidates in the set.

    Testing twenty candidates at once is defensible - it is one decision rather than
    twenty sequential peeks - but it is emphatically NOT twenty independent tests. The
    best of twenty forward Sharpes is inflated exactly the way the best of twenty
    research Sharpes is, and `store.selection_bar()` is the correction. Read it before
    reading the scoreboard.
    """
    tests, failed = [], []
    candidates = list(names)
    lookup = origin_studies or {}
    panel = kwargs.pop("panel", None) or build_panel()
    default_origin = kwargs.pop("origin_study", None)
    for candidate in candidates:
        label = (candidate if isinstance(candidate, str)
                 else getattr(candidate, "name", type(candidate).__name__))
        try:
            tests.append(forward_test(candidate, panel=panel,
                                      origin_study=lookup.get(label, default_origin),
                                      **kwargs))
        except Exception as exc:                                  # noqa: BLE001
            failed.append((label, str(exc)))
            log.warning("forward test of %s failed: %s", label, exc)
    if failed:
        log.warning("%d of %d candidates could not be forward-tested",
                    len(failed), len(candidates))
    for label, err in failed:
        log.warning("  %s: %s", label, err)
    return tests


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------

def _curve_map(result: BacktestResult, lo: str = "") -> dict:
    """The three series a stored curve carries, optionally clipped to a window start."""
    out = {"nav": result.equity, "nav_gross": result.gross_equity,
           "benchmark": result.benchmark}
    if not lo:
        return out
    return {k: (slice_curve(v, lo, "") if v is not None else None)
            for k, v in out.items()}


def _save_artifacts(result: BacktestResult, forward_id: str,
                    save: bool | str | Path) -> str:
    """Write the forward result's full artifacts, or nothing. Returns the path or ''.

    On by default because a forward test happens once. The registry can rebuild a
    research run at any time; it cannot rebuild the trade ledger of a look that has
    already been spent if the data underneath it later changes.
    """
    if save is False or save is None:
        return ""
    directory = (FORWARD_RESULTS_DIR / forward_id if save is True
                 else Path(save) / forward_id)
    try:
        return str(result.save(directory))
    except Exception as exc:                                      # noqa: BLE001
        log.warning("could not save forward artifacts to %s: %s", directory, exc)
        return ""


#: The trial log, read once per process. A snapshot is what `_origin_study` wants: the
#: run it is looking for was logged long before this process started, and re-reading a
#: ten-megabyte log once per candidate per cost setting would dominate the runtime of a
#: whole suite.
_REGISTRY_SNAPSHOT = None


def _origin_study(fingerprint: str, exclude: tuple[str, ...]) -> str | None:
    """Which search first ran this exact configuration, if the trial log remembers.

    A forward result is meaningless without the multiple testing behind the candidate:
    the winner of a 1,400-individual genetic search and the first thing anybody wrote
    down are different claims even at identical Sharpes. The study name is what carries
    that, and asking a user to retype it is how it goes missing.

    So it is recovered from `registry.fingerprint`, which is the same hash the trial log
    already keys on - the EARLIEST non-forward study that ran this configuration. That
    is right when the candidate was run under its search's own `--study`, and returns
    None when it was not, which is the honest answer rather than a guess.
    """
    global _REGISTRY_SNAPSHOT
    if not fingerprint:
        return None
    try:
        if _REGISTRY_SNAPSHOT is None:
            _REGISTRY_SNAPSHOT = registry.load()
        df = _REGISTRY_SNAPSHOT
        if df.empty or "fingerprint" not in df.columns:
            return None
        hits = df[(df["fingerprint"] == fingerprint) & (~df["study"].isin(exclude))]
        return str(hits.iloc[0]["study"]) if len(hits) else None
    except Exception as exc:                                      # noqa: BLE001
        log.debug("could not recover the origin study: %s", exc)
        return None


def _realised_window(forward_leg: Leg, nominal: Window) -> Window:
    """The dates the forward run really covered, falling back to what was available."""
    if forward_leg.start and forward_leg.end:
        return Window(forward_leg.start, forward_leg.end, "forward")
    return nominal


def _drift(recomputed: Leg, sealed: Leg) -> float:
    a, b = recomputed.sharpe_monthly, sealed.sharpe_monthly
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(a - b)


def _coverage(diag: dict, which: str) -> float:
    import re
    text = str(diag.get("price_coverage", ""))
    pattern = r"([\d.]+)% worst" if which == "min" else r"([\d.]+)% median"
    m = re.search(pattern, text)
    return float(m.group(1)) / 100.0 if m else float("nan")


def _count(value) -> int:
    import re
    if value is None:
        return 0
    if isinstance(value, (int, np.integer)):
        return int(value)
    m = re.match(r"\s*(\d+)", str(value))
    return int(m.group(1)) if m else 0


def _sigma(c: Comparison) -> str:
    if not np.isfinite(c.decay_z):
        return "not computable"
    return (f"{c.decay_z:+.2f} sigma (p={c.decay_p:.3f} that a drop this large or "
            "larger is sampling noise)")


def _prob(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.3f}"


def _fmt(x: float) -> str:
    return "n/a" if x is None or x != x else f"{x:.3f}"
