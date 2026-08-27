"""The rebalance loop and the accounting.

This is the fitness function. Every number anyone quotes about any strategy in this
project comes out of the loop below, so the three decisions it encodes matter more
than anything in the strategies themselves.

1. Execution happens at the NEXT open, never at the signal close
------------------------------------------------------------------
A signal computed from data up to and including the close of day T cannot be filled at
the close of day T - that price was not knowable when the signal formed. It fills at
the open of T+1. This is encoded here, once, so no strategy can get it wrong and no
strategy has to remember to get it right. It is the single most common way a backtest
manufactures returns that do not exist.

2. Dividends are already in the prices - do NOT add them again
---------------------------------------------------------------
`adj_close` is total-return adjusted through `adj_factor` (splits + dividends,
ADR-006). Holding a constant number of adjusted shares therefore already reinvests
dividends. Crediting cash dividends on top double-counts them and inflates returns by
roughly the dividend yield - about 1.9pp/yr on SPY, which is more than enough to make
a bad strategy look good. There is deliberately no dividend accrual anywhere in this
file. Acceptance test 1 exists to catch a regression here: it fails at ~6.4%/yr if
dividends get dropped and at ~10.2%/yr if they get counted twice.

"Shares" below are therefore adjusted-space shares - notional, not the count on a
statement. Real share counts, needed for a per-share commission, are recovered through
`cum_split` at execution time.

3. A position that stops having prices must resolve to an outcome
------------------------------------------------------------------
Otherwise it silently vanishes and the strategy never books the result. See
delisting.py. Every such exit is recorded in `result.exits` with the assumption that
priced it, and the count of unresolved ones goes in the diagnostics.

Performance
-----------
The loop runs per rebalance (~230), not per session and never per security. Within a
holding period the share vector is constant, so the whole NAV path for that period is
one matrix-vector product. A full backtest is ~230 strategy calls plus ~230 matvecs.
That is what makes 10,000 GA fitness evaluations tractable.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from . import metrics
from .context import Context, PanelView
from .costs import CostBreakdown, CostModel, get_cost_model
from .panel import Panel, build_panel
from .portfolio import turnover as compute_turnover
from .portfolio import validate_weights
from .registry import HOLDOUT_START, apply_holdout, record_holdout_touch
from .results import BacktestResult
from .strategy import Strategy, get_strategy, normalize_weights

log = logging.getLogger(__name__)

#: Earliest usable rebalance. Membership intervals start at the 2007-03 snapshot, so
#: the first month-end session with a non-empty point-in-time universe is 2007-04-30.
DEFAULT_START = "2007-04-01"

#: Below this fraction of the index actually priced, a run is refused unless the
#: caller opts in. At 0.58 (2007) the backtest is over a 274-name subset of a 470-name
#: index; that may be acceptable, but it must be a decision rather than a surprise.
DEFAULT_MIN_COVERAGE = 0.0


class EngineError(RuntimeError):
    """The engine cannot produce a trustworthy number. Never suppress this."""


def run_backtest(
    strategy: Strategy | str,
    *,
    panel: Panel | None = None,
    start: str = DEFAULT_START,
    end: str | None = None,
    initial_capital: float = 100_000.0,
    costs: str | CostModel = "realistic",
    liquidity_floor: float = 0.0,
    features=None,
    seed: int = 0,
    benchmark: str | None = "SPY",
    track_gross: bool = True,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    strategy_kwargs: dict | None = None,
    holdout: str = "exclude",
    study: str | None = None,
    log_run: bool = True,
    notes: str = "",
) -> BacktestResult:
    """Run one strategy over one period and return everything it produced.

    Parameters
    ----------
    strategy        a Strategy object, or a registered name
    panel           prebuilt Panel; built (and cached) if omitted
    costs           'optimistic' | 'realistic' | 'pessimistic' | 'free' | CostModel
    liquidity_floor minimum trailing median dollar volume for a name to be buyable
    track_gross     also run the identical path with costs off, to measure cost drag
    min_coverage    refuse to run if any rebalance prices less than this share of the
                    index; 0.0 reports the number instead of refusing
    holdout         'exclude' (default) stops the day before HOLDOUT_START;
                    'include' runs straight through it; 'only' runs the holdout alone.
                    The last two are recorded in the holdout ledger and cannot be
                    silenced - see registry.py and ADR-025.
    study           name of the search this run belongs to. Decides `n_trials` for the
                    deflated Sharpe, so it should match the search you actually did.
    log_run         append to the experiment registry. On by default because a trial you
                    forgot to log cannot be recovered (ADR-026). The holdout ledger is
                    written regardless of this flag.
    notes           free text stored with the run

    Note the asymmetry between `holdout` and `log_run`: you may run something without
    logging it as a trial, but you may not look at the holdout without leaving a trace.
    """
    t_start = time.perf_counter()
    if isinstance(strategy, str):
        strategy = get_strategy(strategy, **(strategy_kwargs or {}))
    panel = panel or build_panel()
    cost_model = get_cost_model(costs)
    rng = np.random.default_rng(seed)

    start, end, touched = apply_holdout(start, end, holdout, str(panel.dates[-1]))
    if touched:
        record_holdout_touch(
            strategy=getattr(strategy, "name", type(strategy).__name__),
            study=study, mode=holdout, start=start, end=end, reason=notes)

    # The last session the run may see. Everything downstream is bounded by this row,
    # not by the panel's end - otherwise the final holding period would run to the end
    # of the data and a holdout="exclude" run would carry straight through the holdout.
    end_row = panel.date_index(end, side="prev")
    reb = _rebalance_rows(panel, start, end_row, getattr(strategy, "warmup", 0))
    if len(reb) < 2:
        raise EngineError(
            f"only {len(reb)} usable rebalance date(s) between {start} and {end}. "
            "Each needs a following session to execute in, and a warmup window.")

    tradable = panel.tradable(liquidity_floor)
    cov = _check_coverage(panel, reb, min_coverage)

    if hasattr(strategy, "on_start"):
        strategy.on_start(panel)

    state = _State(panel, initial_capital)
    gross = _State(panel, initial_capital) if track_gross else None
    from .costs import FREE

    ledger: list[dict] = []
    weight_rows: list[np.ndarray] = []
    total_costs = CostBreakdown()

    for i, t in enumerate(reb):
        exec_row = t + 1
        next_exec = reb[i + 1] + 1 if i + 1 < len(reb) else end_row

        # ---- signal, formed from data up to and including the close of t ----------
        ctx = Context(
            as_of=str(panel.dates[t]), t=t, view=PanelView(panel, t),
            universe=panel.in_index[t], tradable=tradable[t],
            positions=state.shares.copy(), cash=state.cash, nav=state.nav,
            features=features.at(t) if features is not None else None,
            feature_names=getattr(features, "names", ()),
            params=getattr(strategy, "params", {}), rng=rng,
        )
        w_target = validate_weights(
            normalize_weights(strategy.target_weights(ctx), panel.security_ids))
        _reject_untradable(w_target, tradable[t], panel, ctx.as_of)
        weight_rows.append(w_target)

        # ---- fill at the open of t+1 ----------------------------------------------
        rec = state.rebalance(exec_row, w_target, cost_model)
        total_costs += rec["cost"]
        if gross is not None:
            gross.rebalance(exec_row, w_target, FREE)

        # ---- carry the position to the next execution date ------------------------
        state.carry(exec_row, next_exec)
        if gross is not None:
            gross.carry(exec_row, next_exec)

        ledger.append({
            "date": str(panel.dates[t]),
            "exec_date": str(panel.dates[exec_row]),
            "nav": rec["nav_open"],
            "turnover": rec["turnover"],
            "n_positions": rec["n_positions"],
            "traded_notional": rec["cost"].traded_notional,
            "cost": rec["cost"].total,
            "cost_bps": (rec["cost"].total / rec["nav_open"] * 1e4
                         if rec["nav_open"] > 0 else 0.0),
            "cash": rec["cash"],
            "in_index": int(panel.index_size[t]),
            "priced": int((panel.in_index[t] & panel.has_price[t]).sum()),
        })

    equity = _series(panel.dates, state.nav_path)
    gross_equity = _series(panel.dates, gross.nav_path) if gross is not None else None
    ledger_df = pd.DataFrame(ledger)
    bench = _benchmark_series(benchmark, equity) if benchmark else None

    perf = metrics.compute(
        equity, benchmark=bench,
        turnover=ledger_df.set_index("date")["turnover"],
        positions=ledger_df.set_index("date")["n_positions"],
        gross_equity=gross_equity,
    )

    exits = pd.DataFrame(state.exits)
    elapsed = time.perf_counter() - t_start
    diagnostics = _diagnostics(panel, reb, cov, exits, total_costs, elapsed, state)

    result = BacktestResult(
        strategy=getattr(strategy, "name", type(strategy).__name__),
        config={
            "start": str(panel.dates[reb[0]]), "end": str(panel.dates[end_row]),
            "initial_capital": initial_capital,
            "cost_model": cost_model.name, "cost_params": cost_model.describe(),
            "liquidity_floor": liquidity_floor, "seed": seed,
            "benchmark": benchmark, "n_rebalances": len(reb),
            "holdout_mode": holdout, "touched_holdout": touched,
            "holdout_start": HOLDOUT_START,
            "strategy_detail": (strategy.describe() if hasattr(strategy, "describe")
                                else {"name": getattr(strategy, "name", "?")}),
            "panel": panel.meta,
        },
        equity=equity, gross_equity=gross_equity, benchmark=bench,
        performance=perf, rebalances=ledger_df,
        weights=pd.DataFrame(np.array(weight_rows),
                             index=pd.Index(ledger_df["date"], name="date"),
                             columns=panel.security_ids),
        exits=exits, costs=total_costs, diagnostics=diagnostics,
    )
    if log_run:
        from .registry import log as log_run_to_registry
        rec = log_run_to_registry(result, study=study, notes=notes)
        if rec is not None:
            result.config["run_id"] = rec.run_id
            result.config["fingerprint"] = rec.fingerprint
            result.config["study"] = rec.study

    log.info("%s: CAGR %.2f%%  Sharpe %.2f  maxDD %.1f%%  (%.2fs)",
             result.strategy, perf.cagr * 100, perf.sharpe,
             perf.max_drawdown * 100, elapsed)
    return result


# --------------------------------------------------------------------------
# Portfolio state - the accounting
# --------------------------------------------------------------------------

class _State:
    """Shares, cash and the NAV path for one run.

    Kept as a class rather than inlined because the engine runs two of these in
    lockstep - the real one and a zero-cost mirror - and any divergence between them
    other than costs would be a bug.
    """

    def __init__(self, panel: Panel, capital: float):
        self.p = panel
        self.shares = np.zeros(panel.n_securities, dtype=np.float64)
        self.cash = float(capital)
        self.nav = float(capital)
        self.nav_path = np.full(panel.n_dates, np.nan, dtype=np.float64)
        self.exits: list[dict] = []
        self.value_ok = panel.has_price
        self.unfilled = 0
        self.unfilled_notional = 0.0
        self.ruined = False

    # ------------------------------------------------------------- rebalance

    def rebalance(self, e: int, w_target: np.ndarray, cost_model: CostModel) -> dict:
        """Trade into `w_target` at the open of session `e`."""
        px = self.p.adj_open[e]
        held = self.shares != 0
        if held.any() and not np.isfinite(px[held]).all():
            bad = self.p.security_ids[held & ~np.isfinite(px)]
            raise EngineError(
                f"holding {list(bad[:5])} with no open price on {self.p.dates[e]}. "
                "carry() should have resolved these; this is an engine bug.")

        nav_open = self.cash + float(np.dot(np.nan_to_num(px), self.shares))

        # Ruin is a result, not a crash. A long-only portfolio CAN go to zero - hold a
        # single name through a bankruptcy and that is exactly what happens - and an
        # engine that raises instead of reporting it would hide the one outcome a
        # backtest most needs to show. The curve flatlines at zero from here.
        #
        # A NEGATIVE NAV is different: it is unreachable without leverage, so it means
        # the accounting is broken and it stays an error.
        if nav_open <= 0:
            if nav_open < -1e-9 * max(self.nav, 1.0):
                raise EngineError(
                    f"NAV went negative ({nav_open:,.2f}) on {self.p.dates[e]}; "
                    "impossible under a long-only mandate, so this is an engine bug.")
            self.ruined = True
            self.shares[:] = 0.0
            self.cash = 0.0
            return {"nav_open": 0.0, "cash": 0.0, "cost": CostBreakdown(),
                    "turnover": 0.0, "n_positions": 0}

        v_current = np.nan_to_num(px) * self.shares
        w_current = v_current / nav_open

        # Costs depend on the trade, the trade depends on how much is investable, and
        # investable depends on costs. Two passes close the loop to well under a cent.
        as_traded = self._as_traded_price(e)
        hs = self.p.half_spread[e].astype(np.float64)
        cost = CostBreakdown()
        v_target = w_target * nav_open
        for _ in range(2):
            traded = np.abs(v_target - v_current)
            cost = cost_model.charge(traded, as_traded, hs)
            v_target = w_target * (nav_open - cost.total)

        # A name can be priced at the signal date and have no bar at the execution
        # date - it halted, or that was its final session. Requiring a bar at t+1 to
        # be "tradable" at t would be lookahead: you cannot know on Monday that a
        # stock will not open on Tuesday. What actually happens to a market-on-open
        # order in that case is that it does not fill, so that is what happens here.
        # The intended notional stays in cash and the miss is recorded.
        fillable = np.isfinite(px) & (px > 0)
        unfilled = (v_target != 0) & ~fillable
        if unfilled.any():
            self.unfilled += int(unfilled.sum())
            self.unfilled_notional += float(np.abs(v_target[unfilled]).sum())
            v_target = np.where(fillable, v_target, 0.0)

        self.shares = np.where(fillable, v_target / np.where(fillable, px, 1.0), 0.0)
        self.cash = nav_open - float(v_target.sum()) - cost.total
        if self.cash < -1e-6 * nav_open:
            raise EngineError(f"cash went to {self.cash:,.2f} on {self.p.dates[e]}; "
                              "leverage is outside the mandate")

        return {"nav_open": nav_open, "cash": self.cash, "cost": cost,
                "turnover": compute_turnover(w_target, w_current),
                "n_positions": int((self.shares != 0).sum())}

    def _as_traded_price(self, e: int) -> np.ndarray:
        """The price that actually changed hands, for real share counts.

        Our stored close is split-adjusted (ADR-007), so `raw_close * cum_split` is
        what a broker would have printed. Only the per-share commission cares, and at
        retail size the per-order minimum usually dominates it anyway - but getting it
        backwards would inflate share counts by the split ratio, so it is done right.
        """
        return self.p.raw_close[e] * self.p.cum_split[e]

    # ----------------------------------------------------------------- carry

    def carry(self, e: int, next_e: int) -> None:
        """Hold the position from session `e` through `next_e`, filling the NAV path.

        Resolves any holding that stops being valuable before `next_e` - a delisting,
        or a gap in the price series. Without this the position would silently drop
        out of the NAV and the strategy would never book the outcome.
        """
        stop = min(next_e, self.p.n_dates - 1)
        held = np.flatnonzero(self.shares != 0)

        deaths: dict[int, list[int]] = {}
        if len(held):
            # Valuable at every session in [e, stop]? If not, the position must be
            # resolved at the last session before the break.
            ok = self.value_ok[e:stop + 1, held]
            broken = ~ok.all(axis=0)
            for j in np.flatnonzero(broken):
                first_bad = int(np.argmax(~ok[:, j])) + e
                deaths.setdefault(max(first_bad - 1, e - 1), []).append(int(held[j]))

        cursor = e
        for at in sorted(deaths):
            if at >= cursor:
                self._fill(cursor, at)
                cursor = at + 1
            self._resolve(at, deaths[at])
        self._fill(cursor, stop)
        self.nav = float(self.nav_path[stop])

    def _fill(self, lo: int, hi: int) -> None:
        """NAV for every session in [lo, hi] under a constant share vector."""
        if hi < lo:
            return
        block = np.nan_to_num(self.p.adj_close[lo:hi + 1])
        self.nav_path[lo:hi + 1] = block @ self.shares + self.cash

    def _resolve(self, at: int, cols: list[int]) -> None:
        """Liquidate positions that cannot be carried past session `at`."""
        for s in cols:
            px = self.p.adj_close[at, s]
            if not np.isfinite(px):
                px = self._last_finite(at, s)
            ret = float(self.p.delist_return[s])
            reason = str(self.p.delist_reason[s])
            # Bars resuming later means this was a gap in the feed, not a delisting.
            # Charging a bankruptcy return to a data gap would be a fabricated loss.
            if at < int(self.p.last_bar_index[s]):
                ret, reason = 0.0, "price_gap"
            proceeds = float(self.shares[s]) * float(px) * (1.0 + ret)
            self.exits.append({
                "date": str(self.p.dates[at]),
                "security_id": str(self.p.security_ids[s]),
                "ticker": str(self.p.tickers[s]),
                "reason": reason, "delist_return": ret,
                "last_price": float(px), "proceeds": proceeds,
            })
            self.cash += proceeds
            self.shares[s] = 0.0

    def _last_finite(self, at: int, s: int) -> float:
        col = self.p.adj_close[:at + 1, s]
        idx = np.flatnonzero(np.isfinite(col))
        if not len(idx):
            raise EngineError(f"no price ever for {self.p.security_ids[s]}")
        return float(col[idx[-1]])


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _rebalance_rows(panel: Panel, start: str, end_row: int, warmup: int) -> list[int]:
    """Month-end sessions in range that have a following session to execute in.

    `t + 1 <= end_row` rather than `t <= end_row`: a rebalance on the very last session
    of the window has nowhere to fill, and letting it execute one session later would
    reach past the window - which under a holdout policy means reaching into reserved
    data. Dropping it is the honest choice.
    """
    rows = [int(t) for t in panel.rebalance_index
            if start <= str(panel.dates[t])
            and t + 1 <= end_row and t >= warmup]
    # A rebalance with an empty point-in-time universe is not a rebalance, it is a
    # date before the membership history begins.
    return [t for t in rows if panel.in_index[t].any()]


def _reject_untradable(w: np.ndarray, tradable: np.ndarray, panel: Panel, as_of: str) -> None:
    """A weight on a name that was not tradable that day is a leak, not a rounding error."""
    bad = (w > 0) & ~tradable
    if bad.any():
        names = panel.security_ids[bad][:5].tolist()
        raise EngineError(
            f"strategy allocated to {int(bad.sum())} name(s) not tradable on {as_of}, "
            f"e.g. {names}. Either they were not in the index that day, or they had no "
            "price. Filter on ctx.tradable.")


def _check_coverage(panel: Panel, reb: list[int], min_coverage: float) -> dict:
    """Fraction of the point-in-time index that actually has prices, per rebalance.

    The denominator is `panel.index_size` - the real membership count - not the panel's
    own columns. See Panel.coverage(); measuring against the subset we have data for
    would report near-100% by construction.
    """
    idx = np.array(reb)
    in_idx = panel.index_size[idx]
    priced = (panel.in_index[idx] & panel.has_price[idx]).sum(axis=1)
    ratio = priced / np.maximum(in_idx, 1)
    worst = int(np.argmin(ratio))
    info = {
        "min": float(ratio.min()), "median": float(np.median(ratio)),
        "max": float(ratio.max()), "worst_date": str(panel.dates[idx[worst]]),
        "worst_priced": int(priced[worst]), "worst_in_index": int(in_idx[worst]),
    }
    if min_coverage > 0 and info["min"] < min_coverage:
        raise EngineError(
            f"price coverage falls to {info['min']:.1%} on {info['worst_date']} "
            f"({info['worst_priced']}/{info['worst_in_index']} names), below the "
            f"{min_coverage:.0%} floor. Raise `start`, or lower `min_coverage` to run "
            "anyway and read the diagnostics.")
    return info


def _diagnostics(panel: Panel, reb: list[int], cov: dict, exits: pd.DataFrame,
                 total: CostBreakdown, elapsed: float, state: _State) -> dict:
    d = {
        "rebalances": len(reb),
        "price_coverage": f"{cov['median']:.1%} median, {cov['min']:.1%} worst "
                          f"({cov['worst_priced']}/{cov['worst_in_index']} on "
                          f"{cov['worst_date']})",
        "runtime_seconds": round(elapsed, 3),
    }
    if len(exits):
        by = exits["reason"].value_counts().to_dict()
        d["forced_exits"] = f"{len(exits)} - {by}"
        unresolved = int((exits["reason"] == "unresolved").sum())
        if unresolved:
            d["!! unresolved_exits"] = (
                f"{unresolved} position(s) exited with no recorded reason; treated as "
                "an index removal at the last price. Pre-2010 coverage of "
                "sp500_changes is poor (ADR-010).")
    else:
        d["forced_exits"] = "0"
    if state.ruined:
        d["!! ruined"] = ("NAV reached zero - the portfolio was wiped out and every "
                          "later rebalance is a no-op")
    if state.unfilled:
        d["unfilled_orders"] = (f"{state.unfilled} order(s) had no opening bar and did "
                                f"not fill (${state.unfilled_notional:,.0f} intended)")
    if total.n_spread_fallback:
        d["!! spread_fallback_orders"] = total.n_spread_fallback
    if panel.meta.get("half_spread_source", "").startswith("MISSING"):
        d["!! half_spread"] = panel.meta["half_spread_source"]
    return d


def _series(dates: np.ndarray, nav: np.ndarray) -> pd.Series:
    """Trim to the span the run actually covered, both ends.

    Trailing NaNs matter now that a run can end before the panel does: leaving them in
    would make the curve claim a length it does not have.
    """
    s = pd.Series(nav, index=dates, name="nav")
    first, last = s.first_valid_index(), s.last_valid_index()
    return s.loc[first:last] if first is not None else s


def _benchmark_series(ticker: str, equity: pd.Series) -> pd.Series | None:
    """Benchmark rebased to the strategy's starting NAV, on the strategy's dates.

    Total-return adjusted the same way the strategy is, so the comparison is
    like-for-like. A price-return benchmark against a total-return strategy would
    hand every strategy a free ~1.9pp/yr of apparent alpha.
    """
    from .benchmark import benchmark_total_return
    try:
        b = benchmark_total_return(ticker)
    except Exception as exc:  # noqa: BLE001
        log.warning("benchmark %s unavailable: %s", ticker, exc)
        return None
    b = b.reindex(equity.index).ffill()
    if b.isna().all():
        return None
    first = b.first_valid_index()
    return b / b.loc[first] * float(equity.iloc[0])


def run_all_cost_settings(strategy: Strategy | str, **kwargs: Any) -> list[BacktestResult]:
    """The same strategy under optimistic, realistic and pessimistic costs.

    Always report all three. A strategy that only works under `optimistic` is a bet
    that the spread estimator is wrong in your favour, not a strategy (see costs.py).
    """
    from .costs import all_settings
    kwargs.pop("costs", None)
    return [run_backtest(strategy, costs=m, **kwargs) for m in all_settings()]
