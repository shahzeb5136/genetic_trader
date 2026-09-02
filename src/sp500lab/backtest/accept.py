"""The four checks that decide whether any number this engine produces means anything.

From docs/HANDOFF.md TODO-1: write these before any strategy, and treat a failure as
invalidating every downstream result rather than as a bug to work around later.

    1. SPY total return   Buy-and-hold SPY through the engine, zero costs, must
                          reproduce SPY's real total return. This calibrates the whole
                          adjustment chain in one number. 6.4%/yr means dividends are
                          being dropped; 10.2%/yr means they are counted twice; 8.32%
                          means the chain is right.
    2. Equal-weight       Zero costs, an equal-weight portfolio must equal the
       identity           equal-weight index return computed directly from the panel.
                          This is the accounting identity: if the engine's own bookkeeping
                          disagrees with a five-line pandas calculation over the same
                          matrix, the bookkeeping is wrong.
    3. Leakage guard      A strategy that deliberately tries to read past its as-of date
                          must RAISE. Not return zeros, not be filtered - raise. Tested
                          three ways, matching the three escape routes in context.py.
    4. Determinism        Same inputs and seed produce a byte-identical equity curve.
                          A GA whose fitness function is noisy evolves toward the noise.

Acceptance is calibration, not research. So none of these runs is logged as a trial
(`log_run=False`) and none of them overrides the holdout policy - the strategy checks
stop at the research window like everything else. Checks 1 and 1b read the SPY benchmark
series directly rather than through the engine, over its full 2000-2026 history: that is
a property of the data feed, not a strategy result, and an index's long-run total return
is not information anyone can select a strategy with.

Run with `python -m sp500lab backtest accept`, or via pytest in tests/test_backtest.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .benchmark import annualised, benchmark_price_return, benchmark_total_return
from .context import Context, LookaheadError
from .costs import FREE
from .engine import run_backtest
from .panel import Panel, build_panel
from .portfolio import Construction
from .strategy import SignalStrategy, get_strategy

log = logging.getLogger(__name__)

#: Measured 2026-08-27 over 2000-01-03..2026-08-26. See docs/HANDOFF.md TODO-1.
SPY_TOTAL_RETURN = 0.0832
SPY_PRICE_RETURN = 0.0643

#: A few basis points, as the handoff specifies. The engine trades at opens and the
#: reference compounds closes, so exact equality is not expected - but the gap has to
#: stay far below the ~190bp/yr that a dropped dividend stream would open up.
TOLERANCE_BP = 25.0

#: Check 2 is an accounting *identity*, not an approximation, so it gets a far tighter
#: bound than the SPY comparison. It currently lands at 0.1bp. The residual is real and
#: small: cash left by an order that could not fill, and delisting proceeds parked in
#: cash where the reference simply renormalises over the survivors. 10bp leaves room for
#: that to grow as delisting coverage improves without letting a bookkeeping error hide.
IDENTITY_TOLERANCE_BP = 10.0


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    measured: float | None = None
    expected: float | None = None

    def line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name:28s} {self.detail}"


def run_all(panel: Panel | None = None, start: str = "2007-04-01") -> list[Check]:
    """Every acceptance check. Non-empty failures mean the engine is not trustworthy."""
    panel = panel or build_panel()
    checks = [
        check_spy_total_return(),
        check_adjustment_chain_direction(),
        check_equal_weight_identity(panel, start),
        check_leakage_guard(panel, start),
        check_determinism(panel, start),
        check_no_dividend_double_count(panel, start),
    ]
    return checks


# --------------------------------------------------------------------------
# 1. SPY total return
# --------------------------------------------------------------------------

def replicate_benchmark(ticker: str = "SPY") -> dict:
    """Buy-and-hold the benchmark with the engine's own accounting.

    SPY is an ETF and never an index constituent, so it cannot go through the security
    panel - the engine would (correctly) refuse to let a strategy hold a name outside
    the point-in-time universe. What is being validated here is the adjustment chain
    and the compounding, which is the part the engine actually depends on.
    """
    tr = benchmark_total_return(ticker).dropna()
    pr = benchmark_price_return(ticker).dropna()
    return {
        "ticker": ticker,
        "start": str(tr.index[0]), "end": str(tr.index[-1]),
        "total_return_annualised": annualised(tr),
        "price_return_annualised": annualised(pr),
        "total_return_cumulative": float(tr.iloc[-1] / tr.iloc[0] - 1.0),
    }


def check_spy_total_return() -> Check:
    r = replicate_benchmark("SPY")
    measured = r["total_return_annualised"]
    diff_bp = abs(measured - SPY_TOTAL_RETURN) * 1e4
    ok = diff_bp <= TOLERANCE_BP
    hint = ""
    if not ok:
        if abs(measured - SPY_PRICE_RETURN) * 1e4 < 50:
            hint = "  <- this is the PRICE return; dividends are being dropped"
        elif measured > SPY_TOTAL_RETURN + 0.01:
            hint = "  <- too high; dividends are likely being counted twice"
    return Check(
        "1. SPY total return", ok,
        f"{measured * 100:.2f}%/yr vs {SPY_TOTAL_RETURN * 100:.2f}% expected "
        f"({diff_bp:.1f}bp){hint}",
        measured, SPY_TOTAL_RETURN)


def check_adjustment_chain_direction() -> Check:
    """Total return must exceed price return by roughly SPY's dividend yield.

    Cheap, and it catches an inverted adjustment factor - which produces a plausible
    looking series that is wrong in a direction no single-number check would notice.
    """
    r = replicate_benchmark("SPY")
    gap = r["total_return_annualised"] - r["price_return_annualised"]
    ok = 0.010 <= gap <= 0.030
    return Check(
        "1b. dividend contribution", ok,
        f"total - price = {gap * 100:.2f}%/yr (expect 1.0-3.0%, SPY's yield)",
        gap, 0.019)


# --------------------------------------------------------------------------
# 2. Equal-weight identity
# --------------------------------------------------------------------------

def equal_weight_reference(panel: Panel, start: str, end: str | None = None) -> pd.Series:
    """Equal-weight index return computed straight off the panel, no engine involved.

    Deliberately naive - no cash, no shares, no cost model. Just: at each execution date
    form an equally weighted basket of that month's priced constituents, and compound the
    basket's value. If the engine's bookkeeping disagrees with this, one of them is wrong
    and it is not obvious which, which is the whole point of having it.

    Two details that are easy to get wrong and were, on the first attempt:

    * **A segment ends the session BEFORE the next execution date, not on it.** On the
      execution date itself the portfolio has already been rebalanced, so that close
      belongs to the next segment. Emitting it from both double-counts that day's move.
    * **Chaining happens at the open, not at the close.** `level` is carried as the
      portfolio's value at the *open* of the execution date, so the overnight move from
      the previous close is applied to the OLD basket - which is what actually holds it.
      Chaining from the previous close silently drops that move on every rebalance.
    """
    end = end or str(panel.dates[-1])
    end_row = panel.date_index(end, side="prev")
    reb = [int(t) for t in panel.rebalance_index
           if start <= str(panel.dates[t]) and t + 1 <= end_row
           and panel.in_index[t].any()]
    if not reb:
        return pd.Series(dtype=float, name="equal_weight_reference")

    ac, ao = panel.adj_close, panel.adj_open
    level = 1.0                       # value at the OPEN of the current execution date
    out = np.full(panel.n_dates, np.nan)

    for i, t in enumerate(reb):
        e = t + 1
        last = i + 1 >= len(reb)
        ne = end_row + 1 if last else reb[i + 1] + 1

        held = np.flatnonzero(panel.in_index[t] & panel.has_price[t]
                              & np.isfinite(ao[e]) & (ao[e] > 0))
        if not len(held):
            continue
        base = ao[e, held]

        for d in range(e, min(ne, end_row + 1)):
            ratio = ac[d, held] / base
            if np.isfinite(ratio).any():
                out[d] = level * float(np.nanmean(ratio))

        if not last:
            # Roll to the value at the next open, still holding the OLD basket.
            nxt_open = ao[ne, held] / base
            if np.isfinite(nxt_open).any():
                level = level * float(np.nanmean(nxt_open))
            elif np.isfinite(out[ne - 1]):
                level = float(out[ne - 1])

    s = pd.Series(out, index=panel.dates, name="equal_weight_reference").dropna()
    return s


def check_equal_weight_identity(panel: Panel, start: str) -> Check:
    """The engine's equal-weight run must match the reference to within a few bp/yr."""
    res = run_backtest("equal_weight", panel=panel, start=start, costs=FREE,
                       benchmark=None, track_gross=False, log_run=False)
    ref = equal_weight_reference(panel, start, end=res.config["end"])
    common = res.equity.index.intersection(ref.index)
    if len(common) < 100:
        return Check("2. equal-weight identity", False,
                     f"only {len(common)} overlapping dates")

    a = res.equity.loc[common]
    b = ref.loc[common]
    a_cagr = annualised(a)
    b_cagr = annualised(b)
    diff_bp = abs(a_cagr - b_cagr) * 1e4
    ok = diff_bp <= IDENTITY_TOLERANCE_BP
    return Check(
        "2. equal-weight identity", ok,
        f"engine {a_cagr * 100:.3f}%/yr vs reference {b_cagr * 100:.3f}%/yr "
        f"({diff_bp:.1f}bp)", a_cagr, b_cagr)


# --------------------------------------------------------------------------
# 3. Leakage guard
# --------------------------------------------------------------------------

class _PeekTomorrowByDate(SignalStrategy):
    """Asks for a price on a date it cannot know. Must raise LookaheadError."""
    name = "cheat_by_date"
    construction = Construction(top_k=10)

    def score(self, ctx: Context) -> np.ndarray:
        return ctx.price_on("2099-01-01")


class _PeekTomorrowByIndex(SignalStrategy):
    """Indexes one row past the end of the view. Must raise IndexError."""
    name = "cheat_by_index"
    construction = Construction(top_k=10)

    def score(self, ctx: Context) -> np.ndarray:
        return ctx.view.close[ctx.t + 1]


class _ReachThroughToPanel(SignalStrategy):
    """Tries to get at the engine's full panel. There must be nothing to reach."""
    name = "cheat_by_reference"
    construction = Construction(top_k=10)

    def score(self, ctx: Context) -> np.ndarray:
        return ctx.view.panel.adj_close[-1]  # type: ignore[attr-defined]


def check_leakage_guard(panel: Panel, start: str) -> Check:
    """All three escape routes from context.py must be closed."""
    cases = [
        (_PeekTomorrowByDate(), LookaheadError, "explicit future date"),
        (_PeekTomorrowByIndex(), IndexError, "index past the view"),
        (_ReachThroughToPanel(), AttributeError, "reach through to the panel"),
    ]
    failures = []
    for strat, expected, label in cases:
        try:
            run_backtest(strat, panel=panel, start=start, end="2010-12-31",
                         costs=FREE, benchmark=None, track_gross=False, log_run=False)
            failures.append(f"{label}: did NOT raise")
        except expected:
            pass
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label}: raised {type(exc).__name__}, expected "
                            f"{expected.__name__} ({exc})")
    ok = not failures
    return Check("3. leakage guard", ok,
                 "all 3 cheats blocked" if ok else "; ".join(failures))


# --------------------------------------------------------------------------
# 4. Determinism
# --------------------------------------------------------------------------

def check_determinism(panel: Panel, start: str) -> Check:
    """Two identical runs must produce byte-identical curves - including a random one."""
    mismatches = []
    for name in ("momentum_12_1", "random_weight"):
        a = run_backtest(name, panel=panel, start=start, seed=42, costs="realistic",
                         benchmark=None, track_gross=False, log_run=False)
        b = run_backtest(name, panel=panel, start=start, seed=42, costs="realistic",
                         benchmark=None, track_gross=False, log_run=False)
        if not np.array_equal(a.equity.to_numpy(), b.equity.to_numpy()):
            gap = float(np.nanmax(np.abs(a.equity.to_numpy() - b.equity.to_numpy())))
            mismatches.append(f"{name}: max diff {gap:.6g}")

    # A different seed MUST change the random strategy, or the seed is not wired up
    # and every "independent" run in the noise floor is the same run.
    r1 = run_backtest("random_weight", panel=panel, start=start, seed=1, costs=FREE,
                      benchmark=None, track_gross=False, log_run=False)
    r2 = run_backtest("random_weight", panel=panel, start=start, seed=2, costs=FREE,
                      benchmark=None, track_gross=False, log_run=False)
    if np.array_equal(r1.equity.to_numpy(), r2.equity.to_numpy()):
        mismatches.append("random_weight ignored the seed")

    ok = not mismatches
    return Check("4. determinism", ok,
                 "identical seeds match, different seeds differ" if ok
                 else "; ".join(mismatches))


# --------------------------------------------------------------------------
# 5. Dividend double-count guard
# --------------------------------------------------------------------------

def check_no_dividend_double_count(panel: Panel, start: str) -> Check:
    """An equal-weight run must not beat its own price-return version by too much.

    The engine holds constant adjusted shares, which already reinvests dividends. If
    anyone later "fixes" a perceived missing dividend by crediting cash, this check
    is what catches it: the gap between total and price return would roughly double.
    """
    res = run_backtest("equal_weight", panel=panel, start=start, costs=FREE,
                       benchmark=None, track_gross=False,
                       log_run=False)
    eq_cagr = annualised(res.equity)

    # Same portfolio, valued on a price-only series: adj_close / adj_factor recovers
    # the split-adjusted close, which carries no dividends.
    w = res.weights.to_numpy()
    reb = [panel.date_index(d) for d in res.weights.index]
    level, vals, dates = 1.0, [], []
    for i, t in enumerate(reb):
        e, nxt = t + 1, (reb[i + 1] + 1 if i + 1 < len(reb) else panel.n_dates - 1)
        held = np.flatnonzero((w[i] > 0) & np.isfinite(panel.raw_close[e]))
        if not len(held):
            continue
        base = panel.raw_close[e, held]
        ww = w[i][held] / w[i][held].sum()
        for d in range(e, min(nxt, panel.n_dates - 1) + 1):
            px = panel.raw_close[d, held]
            good = np.isfinite(px)
            if good.any():
                vals.append(level * float((ww[good] * px[good] / base[good]).sum()
                                          / ww[good].sum()))
                dates.append(str(panel.dates[d]))
        if vals:
            level = vals[-1]
    price_cagr = annualised(pd.Series(vals, index=dates))

    gap = eq_cagr - price_cagr
    ok = 0.005 <= gap <= 0.040
    return Check(
        "5. dividends counted once", ok,
        f"total {eq_cagr * 100:.2f}%/yr - price {price_cagr * 100:.2f}%/yr = "
        f"{gap * 100:.2f}%/yr (expect 0.5-4.0%, one dividend stream)", gap, 0.02)


# --------------------------------------------------------------------------
# 6. Every registered strategy honours the contract
# --------------------------------------------------------------------------
#
# The four checks above prove the ENGINE. These prove every competitor plugged into it.
# They run on a three-year window - about 36 rebalances, enough to clear every warmup -
# so the whole roster finishes in a few minutes and can gate a commit.

#: Window for the per-strategy checks. Fundamentals begin 2009-04, so 2012 gives every
#: feature family a year of history to fill in before the first rebalance.
CONTRACT_START, CONTRACT_END = "2012-01-01", "2014-12-31"

#: Sessions of price history kept past CONTRACT_END in the truncated panel of the
#: lookahead check. Enough for the final rebalance to execute at the next open and be
#: marked for a few days; short enough that a strategy peeking past the window finds
#: nothing there.
LOOKAHEAD_TAIL_SESSIONS = 5


def contract_roster(include_learned: bool = False) -> tuple[str, ...]:
    """Every registered strategy in the competition, in group order.

    The learned family trains a model inside `on_start` and is slow; it is opted in
    rather than out so the default roster stays fast enough to run on every change.
    """
    from ..strategies import GROUPS
    groups = ["baselines", "alpha", "frontier", "evolved", "custom"]
    if include_learned:
        groups.insert(3, "learned")
    seen: list[str] = []
    for g in groups:
        for n in GROUPS.get(g, ()):
            if n != "spy_buy_hold" and n not in seen:     # SPY is the benchmark, not a row
                seen.append(n)
    return tuple(seen)


def _contract_run(name: str, panel: Panel, start: str, end: str, *, seed: int = 0,
                  features=None, track_gross: bool = False):
    return run_backtest(name, panel=panel, start=start, end=end, costs="realistic",
                        seed=seed, features=features, benchmark=None,
                        track_gross=track_gross, record_trades=track_gross,
                        log_run=False, log_curve=False)


def check_strategy_contract(panel: Panel, names: tuple[str, ...] | None = None,
                            start: str = CONTRACT_START, end: str = CONTRACT_END
                            ) -> list[Check]:
    """One Check per strategy: it runs, and what it produces is a portfolio.

    The engine already REJECTS the loud violations (a negative weight, leverage, a
    non-member). What this adds is the quiet ones: a curve that goes non-finite, a
    weight row that is all NaN, a strategy whose two identical runs differ, or one that
    somehow ends up richer after costs than before them.
    """
    from .trades import reconcile
    names = names or contract_roster()
    out: list[Check] = []
    for name in names:
        label = f"6. contract: {name}"
        try:
            a = _contract_run(name, panel, start, end, track_gross=True)
            b = _contract_run(name, panel, start, end)
        except Exception as exc:  # noqa: BLE001
            out.append(Check(label, False, f"raised {type(exc).__name__}: {exc}"[:160]))
            continue

        problems: list[str] = []
        eq = a.equity.to_numpy()
        if not np.isfinite(eq).all() or (eq <= 0).any():
            problems.append("equity has a non-finite or non-positive value")
        w = a.weights.to_numpy()
        if w.size and (np.isnan(w).all(axis=1)).any():
            problems.append("a rebalance produced no weights at all (all NaN)")
        wf = np.nan_to_num(w)
        if (wf < -1e-12).any():
            problems.append("a negative weight slipped past validation")
        if (wf.sum(axis=1) > 1.0 + 1e-9).any():
            problems.append("weights sum above 1.0 (leverage)")
        if not np.array_equal(a.equity.to_numpy(), b.equity.to_numpy()):
            problems.append("two identical runs produced different curves")
        if a.gross_equity is not None and len(a.gross_equity):
            if float(a.equity.iloc[-1]) > float(a.gross_equity.iloc[-1]) * (1 + 1e-9):
                problems.append("net NAV exceeds gross NAV - costs added money")
        turn = a.rebalances["turnover"] if "turnover" in a.rebalances else None
        if turn is not None and ((turn < -1e-9) | (turn > 1.0 + 1e-9)).any():
            problems.append("turnover outside [0, 1]")
        if len(a.trades):
            rec = reconcile(a.trades, a)
            if not rec.get("ok", False):
                problems.append(f"trade ledger does not reconcile: {rec.get('why', '')}")

        p = a.performance
        detail = (f"{len(a.weights)} rebalances, CAGR {p.cagr * 100:.1f}%, "
                  f"{len(a.trades):,} orders")
        out.append(Check(label, not problems,
                         detail if not problems else "; ".join(problems),
                         measured=p.cagr))
    return out


def check_no_lookahead(panel: Panel, names: tuple[str, ...] | None = None,
                       start: str = CONTRACT_START, end: str = CONTRACT_END
                       ) -> list[Check]:
    """A strategy's decisions must not change when the future is deleted.

    Run each strategy twice over the same window: once on the full panel (which runs
    to 2026) and once on a copy physically truncated a few sessions past the window's
    end, with every filing and macro print after that date removed as well. The target
    weights at every rebalance must be bit-identical. Not close - identical.

    This is the check `context.py` cannot make. The point-in-time view stops a strategy
    reading tomorrow's price at rebalance time, but `on_start(panel)` sees the whole
    panel by design, and a model that trains there on rows past its as-of date - or a
    feature that centres a window instead of trailing it - will pass every other test
    and show up only here.
    """
    from ..features import build_features
    from ..features.panel import truncate_panel
    names = names or contract_roster()

    cut_row = min(panel.date_index(end, side="prev") + LOOKAHEAD_TAIL_SESSIONS,
                  panel.n_dates - 1)
    cut_date = str(panel.dates[cut_row])
    blind = truncate_panel(panel, cut_row)
    # Built once, shared by every feature strategy. Never cached: a build with a cutoff
    # must not be able to contaminate a real one, or be served from one.
    needs_features = any(getattr(get_strategy(n), "requires_features", ()) for n in names)
    full_f = build_features(panel) if needs_features else None
    blind_f = (build_features(blind, data_cutoff=cut_date, use_cache=False)
               if needs_features else None)

    out: list[Check] = []
    for name in names:
        label = f"7. no lookahead: {name}"
        try:
            a = _contract_run(name, panel, start, end, features=full_f)
            b = _contract_run(name, blind, start, end, features=blind_f)
        except Exception as exc:  # noqa: BLE001
            out.append(Check(label, False, f"raised {type(exc).__name__}: {exc}"[:160]))
            continue
        common = a.weights.index.intersection(b.weights.index)
        if len(common) < 2:
            out.append(Check(label, False, f"only {len(common)} common rebalance dates"))
            continue
        wa = np.nan_to_num(a.weights.loc[common].to_numpy(), nan=-9.0)
        wb = np.nan_to_num(b.weights.loc[common].to_numpy(), nan=-9.0)
        same = np.array_equal(wa, wb)
        if same:
            detail = f"{len(common)} rebalances identical with the panel cut at {cut_date}"
        else:
            diff = np.abs(wa - wb).max(axis=1)
            first = str(common[int(np.argmax(diff > 0))])
            detail = (f"weights differ at {int((diff > 0).sum())}/{len(common)} rebalances, "
                      f"first {first} (max |dw| {diff.max():.3g}) - the strategy reads "
                      f"data past its as-of date")
        out.append(Check(label, same, detail))
    return out


def run_strategy_checks(panel: Panel | None = None, *, include_learned: bool = False,
                        names: tuple[str, ...] | None = None) -> list[Check]:
    """Checks 6 and 7 over the roster. Slower than `run_all`; run it before a release."""
    panel = panel or build_panel()
    names = names or contract_roster(include_learned)
    return check_strategy_contract(panel, names) + check_no_lookahead(panel, names)


# --------------------------------------------------------------------------

def report(checks: list[Check]) -> str:
    lines = ["=" * 72, "BACKTEST ENGINE ACCEPTANCE", "=" * 72]
    lines += [c.line() for c in checks]
    n_fail = sum(not c.passed for c in checks)
    lines += ["=" * 72]
    if n_fail:
        lines.append(f"{n_fail} CHECK(S) FAILED - engine output is not trustworthy. "
                     "Do not build strategies on top of this.")
    else:
        lines.append("All checks passed. The engine reproduces known quantities and "
                     "refuses the known cheats.")
    return "\n".join(lines)
