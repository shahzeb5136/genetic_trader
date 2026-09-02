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
from .trades import TradeLedger

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
    record_trades: bool = True,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    strategy_kwargs: dict | None = None,
    holdout: str = "exclude",
    study: str | None = None,
    log_run: bool = True,
    log_curve: bool = True,
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
    record_trades   build the order-by-order ledger (`result.trades`). On by default,
                    because a result nobody can check the trades of is a claim rather
                    than a measurement. A large search should turn it off: see
                    trades.py for the sizing.
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
    log_curve       also store the month-end equity curve, so a report can plot this run
                    later without re-running it (~7 KB). A large search should turn this
                    off and re-run its winners; see registry.log_curve.
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

    features = _resolve_features(strategy, features, panel)
    # A strategy whose inputs did not exist before a date starts there, rather than
    # sitting in cash and reporting the flat stretch as performance. See
    # BaseStrategy.min_date.
    min_date = getattr(strategy, "min_date", "") or ""
    if min_date > start:
        log.info("%s: start moved %s -> %s (its inputs do not exist earlier)",
                 getattr(strategy, "name", "strategy"), start, min_date)
        start = min_date

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

    trade_ledger = (TradeLedger(security_ids=panel.security_ids, tickers=panel.tickers)
                    if record_trades else None)
    state = _State(panel, initial_capital, ledger=trade_ledger)
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
        rec = state.rebalance(exec_row, w_target, cost_model, signal_date=ctx.as_of)
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
    trades = trade_ledger.frame() if trade_ledger is not None else pd.DataFrame()
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
            "feature_version": (features.meta.get("feature_version")
                                if features is not None else None),
            "strategy_detail": (strategy.describe() if hasattr(strategy, "describe")
                                else {"name": getattr(strategy, "name", "?")}),
            "panel": panel.meta,
        },
        equity=equity, gross_equity=gross_equity, benchmark=bench,
        performance=perf, rebalances=ledger_df,
        weights=pd.DataFrame(np.array(weight_rows),
                             index=pd.Index(ledger_df["date"], name="date"),
                             columns=panel.security_ids),
        exits=exits, trades=trades, costs=total_costs, diagnostics=diagnostics,
    )
    if log_run:
        from .registry import log as log_run_to_registry
        rec = log_run_to_registry(result, study=study, notes=notes, curve=log_curve)
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

    def __init__(self, panel: Panel, capital: float, ledger=None):
        self.p = panel
        self.ledger = ledger
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

    def rebalance(self, e: int, w_target: np.ndarray, cost_model: CostModel,
                  signal_date: str = "") -> dict:
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

        # A name can be priced at the signal date and have no bar at the execution
        # date - it halted, or that was its final session. Requiring a bar at t+1 to
        # be "tradable" at t would be lookahead: you cannot know on Monday that a
        # stock will not open on Tuesday. What actually happens to a market-on-open
        # order in that case is that it does not fill, so that is what happens here.
        # The intended notional stays in cash and the miss is recorded.
        #
        # This is resolved BEFORE costs are priced, not after: an order that never
        # filled never paid a commission either. Charging it was a real (small) error
        # until the trade ledger made it visible - the cost total contained dollars
        # that no executed order could account for. See ADR-029.
        fillable = np.isfinite(px) & (px > 0)
        wanted = w_target * nav_open
        unfilled = (wanted != 0) & ~fillable
        if unfilled.any():
            self.unfilled += int(unfilled.sum())
            self.unfilled_notional += float(np.abs(wanted[unfilled]).sum())
            self._wanted = wanted
            w_target = np.where(fillable, w_target, 0.0)

        # Costs depend on the trade, the trade depends on how much is investable, and
        # investable depends on costs. Two passes close the loop to well under a cent.
        as_traded = self._as_traded_price(e)
        hs = self.p.half_spread[e].astype(np.float64)
        cost = CostBreakdown()
        commission = spread = fixed = np.zeros_like(px)
        v_target = w_target * nav_open
        for _ in range(2):
            traded = np.abs(v_target - v_current)
            cost, commission, spread, fixed = cost_model.charge_detail(
                traded, as_traded, hs)
            v_target = w_target * (nav_open - cost.total)

        shares_before = self.shares
        self.shares = np.where(fillable, v_target / np.where(fillable, px, 1.0), 0.0)
        self.cash = nav_open - float(v_target.sum()) - cost.total
        if self.cash < -1e-6 * nav_open:
            raise EngineError(f"cash went to {self.cash:,.2f} on {self.p.dates[e]}; "
                              "leverage is outside the mandate")

        if self.ledger is not None:
            self._record_orders(e, signal_date, nav_open, v_current, v_target,
                                w_current, px, as_traded, commission, spread, fixed,
                                shares_before, unfilled)

        return {"nav_open": nav_open, "cash": self.cash, "cost": cost,
                "turnover": compute_turnover(w_target, w_current),
                "n_positions": int((self.shares != 0).sum())}

    def _as_traded_price(self, e: int) -> np.ndarray:
        """The price that actually changed hands, for real share counts.

        Our stored prices are split-adjusted (ADR-007), so `raw x cum_split` is what a
        broker would have printed. The OPEN, because that is the bar orders fill in -
        the same price the trade ledger shows an outside reader, so the share count in
        the ledger and the share count the commission was charged on are the same
        number rather than two numbers that nearly agree (ADR-029).

        At retail size the per-order minimum dominates the per-share rate anyway, so
        this rarely moves a cost - but getting the split factor backwards would inflate
        share counts by the whole split ratio, so it is done properly.
        """
        return self.p.raw_open[e] * self.p.cum_split[e]

    def _record_orders(self, e: int, signal_date: str, nav_open: float,
                       v_current: np.ndarray, v_target: np.ndarray,
                       w_current: np.ndarray, px: np.ndarray, as_traded: np.ndarray,
                       commission: np.ndarray, spread: np.ndarray, fixed: np.ndarray,
                       shares_before: np.ndarray, unfilled: np.ndarray) -> None:
        """Write this rebalance into the trade ledger. See trades.py.

        `v_target` here is the post-unfill vector - what was actually bought, not what
        was wanted - so the cash flows tie to the cash column exactly. The wanted-but-
        unfillable orders are recorded separately with status 'unfilled', because an
        order that did not fill is a fact about the run and dropping it would make a
        month look like a decision nobody made.
        """
        delta = v_target - v_current
        charged = commission + spread + fixed
        # Union, not just the moves: an order can be priced on one pass of the cost
        # fixpoint and land on a near-zero delta on the next, and it still paid. Every
        # name that moved money OR was charged for money gets a row, which is what
        # makes the cash identity in trades.reconcile() exact rather than approximate.
        moved = np.flatnonzero((delta != 0.0) | (charged > 0.0))
        exec_date = str(self.p.dates[e])
        if len(moved):
            self.ledger.record_rebalance(
                signal_date=signal_date or exec_date, exec_date=exec_date, idx=moved,
                delta_value=delta[moved],
                adj_shares_delta=(self.shares - shares_before)[moved],
                adj_price=px[moved], as_traded_price=as_traded[moved],
                commission=commission[moved], spread=spread[moved],
                fixed=fixed[moved],
                weight_before=w_current[moved],
                weight_after=v_target[moved] / nav_open, nav=nav_open)

        miss = np.flatnonzero(unfilled)
        if len(miss):
            zeros = np.zeros(len(miss))
            self.ledger.record_rebalance(
                signal_date=signal_date or exec_date, exec_date=exec_date, idx=miss,
                delta_value=self._wanted[miss] - v_current[miss],
                adj_shares_delta=zeros, adj_price=px[miss],
                as_traded_price=as_traded[miss],
                commission=zeros, spread=zeros.copy(), fixed=zeros.copy(),
                weight_before=w_current[miss], weight_after=zeros.copy(), nav=nav_open,
                status="unfilled", reason="no_opening_bar")

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
        nav_at = (self.cash + float(np.dot(np.nan_to_num(self.p.adj_close[at]),
                                           self.shares))
                  if self.ledger is not None else float("nan"))
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
            if self.ledger is not None:
                self.ledger.record_exit(
                    date=str(self.p.dates[at]), security_index=s,
                    adj_shares=float(self.shares[s]), adj_price=float(px),
                    as_traded_price=float(self.p.raw_open[at, s]
                                          * self.p.cum_split[at, s]),
                    proceeds=proceeds, reason=reason, nav=nav_at)
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

def _resolve_features(strategy, features, panel):
    """Load the shared feature panel when a strategy needs one and none was passed.

    Automatic rather than an argument the caller must remember, because forgetting it
    does not fail - it hands the strategy a context with no features, every score comes
    back NaN, and the run reports a flat cash curve as if that were a result. Failing
    loudly on a missing feature NAME is the other half of the same argument.
    """
    needed = tuple(getattr(strategy, "requires_features", ()) or ())
    if not needed:
        return features
    if features is None:
        from ..features import build_features
        features = build_features(panel=panel)
    missing = [n for n in needed if n not in tuple(getattr(features, "names", ()))]
    if missing:
        raise EngineError(
            f"{getattr(strategy, 'name', 'strategy')} needs feature(s) {missing} that "
            "the feature panel does not have. Rebuild it: "
            "`python -m sp500lab features build --rebuild`.")
    return features


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
