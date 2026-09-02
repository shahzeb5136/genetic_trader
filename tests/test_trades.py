"""The trade ledger: does the list of orders actually add up to the equity curve?

Every test here is an identity, not a regression baseline. A ledger whose numbers merely
look plausible is worthless - the whole reason it exists is so an outside reader can
check the run, and a reader who finds the orders and the curve disagree has found that
one of them is fabricated. So the suite asserts the arithmetic:

    cash_after   = cash_before + sum(cash_flow)
    sum(cost)    = the cost the run reported
    notional     = |adjusted shares moved| x adjusted price
    n(rows)      = n(orders the cost model priced)

and the behavioural rules that make those identities meaningful - an unfilled order
moves no cash, a forced exit is still a sale, and recording changes no result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp500lab.backtest import run_backtest
from sp500lab.backtest.costs import FREE, REALISTIC
from sp500lab.backtest.trades import (BUY, COLUMNS, SELL, holdings, most_traded,
                                      reconcile, summarise, write_csv)

from sp500lab.backtest.strategy import BaseStrategy

from test_backtest import EqualAll, HoldFirst, make_panel


class EqualTradable(BaseStrategy):
    """Equal weight across whatever is tradable today. Trades when membership moves."""

    name = "equal_tradable"

    def target_weights(self, ctx):
        w = ctx.empty_weights()
        n = int(ctx.tradable.sum())
        if n:
            w[ctx.tradable] = 1.0 / n
        return w


def drifting_panel(n_days: int = 260, n_sec: int = 4):
    """A panel whose securities grow at DIFFERENT rates, so weights actually drift.

    `make_panel` gives every security the same daily return, which means an equal-weight
    portfolio never has to trade after the first rebalance - the weights are still
    exactly equal next month. That is a fine world for testing the accounting and a
    useless one for testing a trade ledger, because there are no trades in it.
    """
    p = make_panel(n_days=n_days, n_sec=n_sec, daily_return=0.0)
    rates = 0.0002 + 0.0004 * np.arange(n_sec)
    growth = np.cumprod(np.broadcast_to(1.0 + rates, (n_days, n_sec)), axis=0)
    base = 100.0 * np.arange(1, n_sec + 1)
    p.adj_close[:] = growth * base
    p.adj_open[:] = p.adj_close / (1.0 + rates)
    p.raw_close[:] = p.adj_close
    p.raw_open[:] = p.adj_open
    return p


def _run(strategy=None, **kw):
    p = kw.pop("panel", None) or make_panel(n_days=260, n_sec=4, daily_return=0.001)
    kw.setdefault("costs", REALISTIC)
    return p, run_backtest(strategy or EqualAll(), panel=p, start=p.dates[0],
                           benchmark=None, track_gross=False,
                           initial_capital=100_000.0, **kw)


# --------------------------------------------------------------------------
# The identities
# --------------------------------------------------------------------------

def test_ledger_reconciles_against_its_own_run():
    """The headline audit: costs attributed, cash replayed, order count matched."""
    _, res = _run()
    report = reconcile(res.trades, res)
    assert report["ok"], report


def test_every_charged_order_has_a_row():
    """A cost charged against an order nobody can see is the failure mode this prevents.

    Not `<=`: the cost model prices every order with non-zero notional, and the ledger
    records every order the cost model priced. Dropping even the sub-cent ones broke the
    cash identity by $1.26 on a real equal-weight run - see trades.RECORD_EVERY_CHARGED_ORDER.
    """
    _, res = _run()
    rebalance_rows = res.trades[res.trades["reason"] == "rebalance"]
    assert len(rebalance_rows) == res.costs.n_orders


def test_costs_sum_exactly_to_the_reported_total():
    _, res = _run()
    assert res.trades["cost"].sum() == pytest.approx(res.costs.total, abs=1e-9)
    assert res.trades["commission"].sum() == pytest.approx(res.costs.commission, abs=1e-9)
    assert res.trades["spread_cost"].sum() == pytest.approx(res.costs.spread, abs=1e-9)


def test_cash_flows_replay_the_cash_column():
    """Replaying the orders must land on the cash the engine recorded, every time.

    On a panel where every name grows at the same rate an equal-weight portfolio never
    trades after month one, so this uses a drifting panel - otherwise the test passes on
    an empty ledger and proves nothing.
    """
    _, res = _run(panel=drifting_panel())
    assert res.trades["block"].nunique() == len(res.rebalances)
    flows = (res.trades[res.trades["status"] == "filled"]
             .groupby("block")["cash_flow"].sum().sort_index().to_numpy())
    replayed = 100_000.0 + np.cumsum(flows)
    assert replayed == pytest.approx(res.rebalances["cash"].to_numpy(), abs=1e-6)


def test_notional_equals_adjusted_shares_times_adjusted_price():
    """The as-traded view and the adjusted view have to describe the same trade."""
    _, res = _run()
    t = res.trades[(res.trades["reason"] == "rebalance")
                   & (res.trades["status"] == "filled")]
    implied = (t["adj_shares"] * t["adj_price"]).abs()
    assert implied.to_numpy() == pytest.approx(t["notional"].to_numpy(), abs=1e-9)


def test_shares_times_price_equals_notional():
    """The as-traded pair - the one an outsider checks - is internally consistent too."""
    _, res = _run()
    t = res.trades[res.trades["status"] == "filled"]
    assert (t["shares"] * t["price"]).to_numpy() == pytest.approx(
        t["notional"].to_numpy(), rel=1e-9)


# --------------------------------------------------------------------------
# Sides, statuses and reasons
# --------------------------------------------------------------------------

def test_buys_take_cash_out_and_sells_put_it_back():
    _, res = _run(panel=drifting_panel())
    t = res.trades[(res.trades["reason"] == "rebalance")
                   & (res.trades["status"] == "filled")]
    assert (t.loc[t["side"] == BUY, "cash_flow"] < 0).all()
    # A sale is net of its own costs, so a tiny sale can still be cash-negative; what
    # must hold is that it moved stock the other way.
    assert (t.loc[t["side"] == SELL, "adj_shares"] <= 0).all()
    assert (t.loc[t["side"] == BUY, "adj_shares"] >= 0).all()


def test_first_rebalance_is_all_buys_from_an_empty_portfolio():
    _, res = _run()
    first = res.trades[res.trades["block"] == res.trades["block"].min()]
    assert set(first["side"]) == {BUY}
    assert (first["weight_before"] == 0).all()


def test_a_forced_exit_is_recorded_as_a_sale():
    """A delisting is not a decision, but it is a trade, and it must appear as one."""
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
    exits = res.trades[res.trades["reason"].str.startswith("exit:")]
    assert len(exits) == 1
    assert exits.iloc[0]["side"] == SELL
    assert exits.iloc[0]["reason"] == "exit:bankruptcy"
    # The whole point of a bankruptcy: the position was worth something and returned
    # nothing. Both halves of that are in the row.
    assert exits.iloc[0]["notional"] > 0
    assert exits.iloc[0]["cash_flow"] == pytest.approx(0.0)


def test_unfilled_orders_move_no_cash():
    """An order with no opening bar is recorded and charged nothing, because it did not
    happen. Recording it at all matters: the month otherwise looks like a decision
    nobody made."""
    p = drifting_panel(n_days=130, n_sec=3)
    # Security 1 joins the index at rebalance 2 and then has no opening bar on the
    # execution session. Not knowable on the signal date - you cannot know on Monday
    # that a stock will not open on Tuesday - so the order simply does not fill.
    signal = int(p.rebalance_index[2])
    p.in_index[:signal, 1] = False
    p.adj_open[signal + 1, 1] = np.nan

    res = run_backtest(EqualTradable(), panel=p, start=p.dates[0], costs=FREE,
                       benchmark=None, track_gross=False, initial_capital=1000.0)
    unfilled = res.trades[res.trades["status"] == "unfilled"]
    assert len(unfilled) >= 1
    assert (unfilled["cash_flow"] == 0).all()
    assert (unfilled["cost"] == 0).all()
    assert (unfilled["reason"] == "no_opening_bar").all()
    assert reconcile(res.trades, res)["ok"]


# --------------------------------------------------------------------------
# Recording must not change the result
# --------------------------------------------------------------------------

def test_recording_changes_nothing_about_the_run():
    """The ledger observes the backtest; it must not participate in it."""
    p = drifting_panel()
    common = dict(panel=p, start=p.dates[0], benchmark=None, track_gross=False,
                  initial_capital=100_000.0, costs=REALISTIC)
    with_ = run_backtest(EqualAll(), record_trades=True, **common)
    without = run_backtest(EqualAll(), record_trades=False, **common)
    assert with_.performance.cagr == without.performance.cagr
    assert with_.costs.total == without.costs.total
    assert with_.equity.to_numpy() == pytest.approx(without.equity.to_numpy(), abs=0)
    assert len(without.trades) == 0


def test_reconcile_accepts_a_strategy_that_never_traded():
    """An empty ledger is correct for a cash strategy and wrong for anything else."""
    from sp500lab.strategies.baselines import Cash
    p, res = _run(Cash())
    assert len(res.trades) == 0
    assert reconcile(res.trades, res)["ok"]


def test_reconcile_rejects_a_ledger_that_was_never_recorded():
    _, res = _run(record_trades=False)
    report = reconcile(res.trades, res)
    assert not report["ok"]
    assert "record_trades" in report["why"]


# --------------------------------------------------------------------------
# Export shape
# --------------------------------------------------------------------------

def test_exported_columns_are_stable_and_complete():
    _, res = _run()
    assert tuple(res.trades.columns) == COLUMNS


def test_csv_round_trips(tmp_path):
    _, res = _run()
    path = write_csv(res.trades, tmp_path / "trades.csv")
    back = pd.read_csv(path)
    assert len(back) == len(res.trades)
    assert back["notional"].sum() == pytest.approx(res.trades["notional"].sum(), rel=1e-9)
    assert back["cost"].sum() == pytest.approx(res.costs.total, rel=1e-9)


def test_summaries_do_not_invent_rows():
    _, res = _run()
    by_year = summarise(res.trades)
    assert by_year["orders"].sum() == len(res.trades)
    assert by_year["notional"].sum() == pytest.approx(res.trades["notional"].sum())
    assert len(most_traded(res.trades, top=2)) <= 2


def test_holdings_are_a_portfolio():
    _, res = _run()
    held = holdings(res)
    assert set(held.columns) == {"date", "security_id", "ticker", "weight"}
    per_date = held.groupby("date")["weight"].sum()
    assert (per_date <= 1.0 + 1e-9).all()
    assert per_date.iloc[0] == pytest.approx(1.0, abs=1e-9)
