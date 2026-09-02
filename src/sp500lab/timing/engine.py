"""The leg engine: hold or don't hold, at every close and every open.

The representation
------------------
A session has two tradable legs, and any calendar rule is two boolean vectors:

    hold_overnight[t]   hold from the close of session t to the open of t+1
    hold_intraday[t]    hold from the open of session t to its close

Buy-and-hold is both always true. The overnight strategy is `hold_overnight` alone.
Every calendar claim in strategies.py compiles down to these two vectors, and the
engine below neither knows nor cares which claim produced them.

Two identities keep the accounting honest, and `timing_accept()` asserts both:

1.  **Calibration.** Both legs always on, zero costs, must reproduce the same SPY
    total return the monthly engine's acceptance test is calibrated to (8.32%/yr,
    ADR-018). The legs multiply back to the close-to-close return by construction -
    `(open[t+1]/close[t]) * (close[t+1]/open[t+1]) = close[t+1]/close[t]` - so any
    daylight between this engine and the benchmark series is an accounting bug, not a
    data difference.

2.  **Decomposition.** Overnight NAV x intraday NAV = buy-and-hold NAV, gross of
    costs, at every session. The two strategies partition every close-to-close return
    between them, so their curves must multiply back to the whole. This is the
    identity that makes "the overnight share of SPY's return" a measurement rather
    than an estimate.

Costs are charged at every transition - a rule that is in for the overnight leg only
trades twice per session, and that is the point: 252 round trips a year is where most
published calendar anomalies go to die, and the three cost settings measure exactly
how dead. The cost model is the shared one from `backtest/costs.py`, so "optimistic /
realistic / pessimistic" mean the same thing here as everywhere else.

Why signals-at-the-close is not lookahead here
----------------------------------------------
The monthly engine's central invariant is next-open execution (ADR-017): a signal
computed FROM the close cannot fill AT that close. Calendar rules have no signal - the
schedule was fixed years in advance - so a market-on-close order placed at 15:50 fills
at the price the rule was always going to trade. The one rule that reads data (the VIX
gate) conditions on the PRIOR session's close and is lagged accordingly. Any future
timing rule that conditions on same-session data must move its entry to the next open
or it is manufacturing returns; that rule is enforced by review, not by code, and it
is written here so the review has something to point at.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from ..backtest import metrics
from ..backtest.costs import FREE, CostBreakdown, CostModel, get_cost_model
from ..backtest.registry import HOLDOUT_START, apply_holdout, record_holdout_touch
from ..backtest.results import BacktestResult
from .data import TimingData, load_timing_data
from .strategies import TimingStrategy, get_timing_strategy

log = logging.getLogger(__name__)

#: First session with a following session in the benchmarks table, plus warmup for
#: rules with trailing windows (the VIX gate needs a year). Aligned with the monthly
#: engine's DEFAULT_START so the two report over comparable windows.
DEFAULT_START = "2007-04-01"


class TimingEngineError(RuntimeError):
    """The leg engine cannot produce a trustworthy number. Never suppress this."""


def run_timing_backtest(
    strategy: TimingStrategy | str,
    *,
    data: TimingData | None = None,
    start: str = DEFAULT_START,
    end: str | None = None,
    initial_capital: float = 100_000.0,
    costs: str | CostModel = "realistic",
    track_gross: bool = True,
    holdout: str = "exclude",
    study: str | None = None,
    log_run: bool = True,
    log_curve: bool = True,
    notes: str = "",
) -> BacktestResult:
    """Run one calendar rule over one period and return a standard BacktestResult.

    The result is the same object the monthly engine returns, built from the same
    metrics, logged to the same registry, guarded by the same holdout ledger - so the
    scoreboard, the deflated Sharpe and the forward harness treat a calendar rule
    exactly like any other competitor. `holdout` and `study` mean what they mean in
    `run_backtest`; the asymmetry that you may skip trial logging but never a holdout
    look is preserved.
    """
    t_start = time.perf_counter()
    if isinstance(strategy, str):
        strategy = get_timing_strategy(strategy)
    data = data or load_timing_data()
    cost_model = get_cost_model(costs)

    start, end, touched = apply_holdout(start, end, holdout, str(data.dates[-1]))
    if touched:
        record_holdout_touch(strategy=strategy.name, study=study, mode=holdout,
                             start=start, end=end, reason=notes)

    lo = data.date_index(start, side="next")
    hi = data.date_index(end, side="prev") if end else data.n_dates - 1
    if hi - lo < 2:
        raise TimingEngineError(f"only {hi - lo + 1} session(s) between {start} and "
                                f"{end}; nothing can be measured on that")

    hold_on, hold_id = strategy.legs(data)
    _validate_legs(hold_on, hold_id, data.n_dates)
    # The final session's overnight leg would reach past `hi` - under a holdout that
    # means reaching into reserved data, so it is trimmed, exactly as the monthly
    # engine drops a rebalance with no session to fill in.
    hold_on = hold_on.copy()
    hold_on[hi:] = False

    net = _walk(data, lo, hi, hold_on, hold_id, initial_capital, cost_model)
    gross = (_walk(data, lo, hi, hold_on, hold_id, initial_capital, FREE)
             if track_gross else None)

    equity = pd.Series(net["nav"], index=data.dates[lo:hi + 1], name="nav")
    gross_eq = (pd.Series(gross["nav"], index=data.dates[lo:hi + 1], name="nav_gross")
                if gross is not None else None)
    bench = _benchmark(data, equity)

    perf = metrics.compute(
        equity, benchmark=bench,
        turnover=pd.Series(net["turnover"], index=equity.index),
        positions=pd.Series(np.maximum(hold_on[lo:hi + 1], hold_id[lo:hi + 1])
                            .astype(float), index=equity.index),
        gross_equity=gross_eq,
    )

    elapsed = time.perf_counter() - t_start
    exposure = float(np.mean(hold_on[lo:hi] | hold_id[lo:hi]))
    diagnostics = {
        "sessions": hi - lo + 1,
        "legs_held": f"{int(hold_on[lo:hi].sum())} overnight, "
                     f"{int(hold_id[lo:hi + 1].sum())} intraday",
        "time_invested": f"{exposure:.1%} of sessions touched the market",
        "transitions": net["n_transitions"],
        "half_spread_model": data.meta["half_spread"],
        "runtime_seconds": round(elapsed, 3),
    }

    result = BacktestResult(
        strategy=strategy.name,
        config={
            "start": str(data.dates[lo]), "end": str(data.dates[hi]),
            "initial_capital": initial_capital,
            "cost_model": cost_model.name, "cost_params": cost_model.describe(),
            "liquidity_floor": 0.0, "seed": 0,
            "benchmark": data.ticker, "n_rebalances": net["n_transitions"],
            "holdout_mode": holdout, "touched_holdout": touched,
            "holdout_start": HOLDOUT_START,
            "engine": "timing.legs",
            "strategy_detail": strategy.describe(),
            "panel": data.meta,
        },
        equity=equity, gross_equity=gross_eq, benchmark=bench,
        performance=perf, rebalances=net["ledger"],
        costs=net["costs"], diagnostics=diagnostics,
    )
    if log_run:
        from ..backtest.registry import log as log_to_registry
        rec = log_to_registry(result, study=study, notes=notes, curve=log_curve)
        if rec is not None:
            result.config["run_id"] = rec.run_id
            result.config["fingerprint"] = rec.fingerprint
            result.config["study"] = rec.study

    log.info("%s: CAGR %.2f%%  Sharpe %.2f  maxDD %.1f%%  (%.2fs)",
             result.strategy, perf.cagr * 100, perf.sharpe,
             perf.max_drawdown * 100, elapsed)
    return result


# --------------------------------------------------------------------------
# The walk - one pass over the sessions, NAV marked at every close
# --------------------------------------------------------------------------

def _walk(data: TimingData, lo: int, hi: int, hold_on: np.ndarray,
          hold_id: np.ndarray, capital: float, cost_model: CostModel) -> dict:
    """NAV path from close(lo) to close(hi) under the leg schedule.

    The state machine has one bit - invested or in cash - toggled at two checkpoints
    per session. A transition trades the whole account and is charged on the whole
    account; holding through a checkpoint costs nothing. NaN in a price the schedule
    needs is an engine-refusing error rather than a silent skip, because SPY's series
    is complete and a hole in it means the data changed under us.
    """
    nav = np.full(hi - lo + 1, np.nan)
    turnover = np.zeros(hi - lo + 1)
    ledger_rows: list[dict] = []
    total = CostBreakdown()

    value = float(capital)
    invested = False           # position state entering the close of session t
    n_trans = 0

    for t in range(lo, hi + 1):
        i = t - lo
        # ---- the close of t: settle the intraday leg, then reposition -------------
        want_close = bool(hold_on[t]) if t < hi else False
        if want_close != invested:
            value, cost = _trade(data, t, "close", value, cost_model)
            total += cost
            n_trans += 1
            turnover[i] += 1.0
            ledger_rows.append(_row(data, t, "close", want_close, value, cost))
            invested = want_close
        nav[i] = value

        if t == hi:
            break

        # ---- the overnight leg into t+1 -------------------------------------------
        if invested:
            r = data.adj_open[t + 1] / data.adj_close[t]
            _require_finite(r, data, t, "overnight")
            value *= r

        # ---- the open of t+1: reposition for the intraday leg ---------------------
        want_open = bool(hold_id[t + 1])
        if want_open != invested:
            value, cost = _trade(data, t + 1, "open", value, cost_model)
            total += cost
            n_trans += 1
            turnover[i + 1] += 1.0
            ledger_rows.append(_row(data, t + 1, "open", want_open, value, cost))
            invested = want_open

        # ---- the intraday leg of t+1 ----------------------------------------------
        if invested:
            r = data.adj_close[t + 1] / data.adj_open[t + 1]
            _require_finite(r, data, t + 1, "intraday")
            value *= r

        if value <= 0:
            raise TimingEngineError(
                f"NAV reached {value:,.2f} on {data.dates[t + 1]}; impossible for an "
                "unlevered long-only position in an index fund, so this is a bug")

    ledger = pd.DataFrame(ledger_rows) if ledger_rows else pd.DataFrame(
        columns=["date", "checkpoint", "side", "nav", "cost"])
    return {"nav": nav, "turnover": turnover, "ledger": ledger,
            "costs": total, "n_transitions": n_trans}


def _trade(data: TimingData, t: int, checkpoint: str, value: float,
           cost_model: CostModel) -> tuple[float, CostBreakdown]:
    """Trade the whole account at one checkpoint; return the NAV net of the charge."""
    px = data.raw_open[t] if checkpoint == "open" else data.raw_close[t]
    cost = cost_model.charge(
        np.array([value]), np.array([px]), np.array([data.half_spread[t]]))
    if cost.total >= value:
        raise TimingEngineError(
            f"cost {cost.total:,.2f} exceeds NAV {value:,.2f} on {data.dates[t]} - "
            "the account is too small to trade this rule at all")
    return value - cost.total, cost


def _row(data: TimingData, t: int, checkpoint: str, entering: bool, nav: float,
         cost: CostBreakdown) -> dict:
    return {"date": str(data.dates[t]), "checkpoint": checkpoint,
            "side": "BUY" if entering else "SELL", "nav": nav,
            "traded_notional": cost.traded_notional, "cost": cost.total,
            "cost_bps": cost.total / nav * 1e4 if nav > 0 else 0.0,
            "turnover": 1.0, "n_positions": int(entering)}


def _require_finite(r: float, data: TimingData, t: int, leg: str) -> None:
    if not np.isfinite(r) or r <= 0:
        raise TimingEngineError(
            f"no usable {leg} price around {data.dates[t]} for {data.ticker}; "
            "the benchmark series has a hole where the schedule needs a price")


def _validate_legs(hold_on: np.ndarray, hold_id: np.ndarray, n: int) -> None:
    for name, v in (("hold_overnight", hold_on), ("hold_intraday", hold_id)):
        if not isinstance(v, np.ndarray) or v.dtype != bool or v.shape != (n,):
            raise TimingEngineError(
                f"{name} must be a ({n},) bool array; got "
                f"{getattr(v, 'shape', None)} {getattr(v, 'dtype', type(v))}")


def _benchmark(data: TimingData, equity: pd.Series) -> pd.Series:
    """Buy-and-hold the same instrument, rebased to the strategy's starting NAV.

    Built from the SAME adjusted close array the legs use, not from
    `benchmark_total_return`, so the comparison is bitwise like-for-like; the
    acceptance check is what pins the two series to each other.
    """
    lo = int(np.searchsorted(data.dates, equity.index[0]))
    hi = int(np.searchsorted(data.dates, equity.index[-1]))
    closes = data.adj_close[lo:hi + 1]
    return pd.Series(closes / closes[0] * float(equity.iloc[0]),
                     index=data.dates[lo:hi + 1], name=data.ticker)


def run_all_cost_settings(strategy: TimingStrategy | str, **kwargs) -> list[BacktestResult]:
    """The same rule under optimistic, realistic and pessimistic costs - always."""
    from ..backtest.costs import all_settings
    kwargs.pop("costs", None)
    return [run_timing_backtest(strategy, costs=m, **kwargs) for m in all_settings()]


# --------------------------------------------------------------------------
# Acceptance - the two identities from the module docstring
# --------------------------------------------------------------------------

def timing_accept(start: str = "2007-04-01", end: str = "2021-12-31",
                  tolerance_bp: float = 1.0) -> dict:
    """Assert the leg engine's two identities. Raises on failure, returns the numbers.

    Free costs and `log_run=False` throughout: these are calibration runs, not trials -
    the same exemption the monthly engine's acceptance suite uses.
    """
    data = load_timing_data()
    kw = dict(data=data, start=start, end=end, costs=FREE, track_gross=False,
              log_run=False)
    bh = run_timing_backtest("tm_buy_hold", **kw)
    on = run_timing_backtest("tm_overnight", **kw)
    intra = run_timing_backtest("tm_intraday", **kw)

    # Identity 1: both legs on == the adjusted close series itself.
    lo = data.date_index(bh.equity.index[0])
    hi = data.date_index(bh.equity.index[-1])
    ref = data.adj_close[lo:hi + 1] / data.adj_close[lo] * float(bh.equity.iloc[0])
    calib_err = float(np.nanmax(np.abs(bh.equity.to_numpy() / ref - 1.0)))
    years = bh.performance.years
    calib_bp_yr = abs((1 + calib_err) ** (1 / max(years, 1e-9)) - 1) * 1e4

    # Identity 2: overnight x intraday == buy-and-hold, at every session.
    prod = (on.equity.to_numpy() / on.equity.iloc[0]) * \
           (intra.equity.to_numpy() / intra.equity.iloc[0])
    whole = bh.equity.to_numpy() / bh.equity.iloc[0]
    decomp_err = float(np.nanmax(np.abs(prod / whole - 1.0)))

    report = {
        "window": f"{bh.performance.start}..{bh.performance.end}",
        "buy_hold_cagr": bh.performance.cagr,
        "overnight_cagr": on.performance.cagr,
        "intraday_cagr": intra.performance.cagr,
        "calibration_max_rel_err": calib_err,
        "calibration_bp_per_year": calib_bp_yr,
        "decomposition_max_rel_err": decomp_err,
    }
    if calib_bp_yr > tolerance_bp:
        raise TimingEngineError(
            f"calibration failed: buy-and-hold through the leg engine drifts "
            f"{calib_bp_yr:.2f}bp/yr from the adjusted close series ({report})")
    if decomp_err > 1e-9:
        raise TimingEngineError(
            f"decomposition failed: overnight x intraday differs from buy-and-hold "
            f"by up to {decomp_err:.2e} ({report})")
    return report
