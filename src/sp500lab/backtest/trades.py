"""The order-by-order record: what was bought, what was sold, when, and for how much.

Why a strategy is not trustworthy until this exists
---------------------------------------------------
An equity curve is a claim. A list of orders is the evidence for it. Every backtest in
this repo already reports coverage, costs and forced exits so that the ways it could be
lying are visible, but until now the one thing a sceptical reader most wants was
missing: show me the trades. Someone who does not trust the engine cannot check a CAGR -
there is nothing in it to check. They can check "on 2015-06-01 this bought 41 shares of
KO at the open", because that is a fact about the world, and it is either right or it is
not.

So the ledger is written for an outside reader, not for the engine:

  * the price column is the price a broker would have PRINTED that morning - the
    as-traded open, `raw_open x cum_split` - not the total-return-adjusted number the
    accounting runs on. Handing someone an adjusted price and asking them to verify it
    against a quote site guarantees a mismatch that means nothing.
  * the share count is the real one implied by that price, not the adjusted-share
    notional the engine carries internally.
  * the adjusted figures the engine actually used are kept alongside, so the two views
    can be reconciled rather than argued about.

Both are needed. The as-traded pair is checkable against a third party; the adjusted
pair is what the NAV was computed from. A ledger with only one of them is either
unverifiable or unreconcilable.

The identity that makes the ledger checkable
---------------------------------------------
For every rebalance:

    cash_after = cash_before + sum(cash_flow)

where `cash_flow` is negative for a buy, positive for a sale, and always net of the
commission and spread charged to that order. `reconcile()` asserts it against the run's
own cash column, and `sp500lab backtest trades --verify` prints the result. If the
ledger and the equity curve disagree, one of them is wrong and the disagreement is
visible rather than assumed away.

Costs are attributed, never spread
-----------------------------------
`CostModel.charge_detail` returns the per-order commission and half-spread it summed, so
every dollar in the headline cost figure lands on the order that caused it. Recomputing
the attribution here instead would let the two drift, and a cost total that no set of
orders adds up to is exactly the kind of thing this repo exists to make impossible.

What is deliberately in the ledger and not in `rebalances`
----------------------------------------------------------
Forced exits - delistings, and positions whose prices stop - appear as SELL rows with
the reason that priced them. They are not rebalance decisions and the strategy never
chose them, but they moved real money, and a reader reconciling "what did I hold last
month" against "what do I hold now" will not balance without them.

Cost of recording
-----------------
About 12,000 rows for a 50-name strategy over 232 rebalances. That is nothing for a
single run and it is real for a 10,000-individual genetic search, so
`run_backtest(record_trades=...)` defaults to on for ordinary use and the GA turns it
off. Re-running a winner with it on is the same trial - the fingerprint does not include
it - so nothing is lost by the asymmetry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BUY, SELL = "BUY", "SELL"

#: Column order of the exported ledger. Fixed so a diff between two exports is readable
#: and so a downstream consumer can rely on the layout.
COLUMNS = (
    "signal_date", "date", "security_id", "ticker", "side", "status", "reason",
    "shares", "price", "notional",
    "commission", "spread_cost", "cost", "cash_flow",
    "adj_shares", "adj_price", "weight_before", "weight_after", "nav", "block",
)

#: Every order the cost model charged is written down, however small - there is no
#: minimum. A $0.004 rebalancing crumb still pays the $1 commission minimum (capped at
#: 1% of its own value), and dropping it from the ledger was measured to break the cash
#: identity by $1.26 on `equal_weight`: real money charged against an order nobody could
#: see. A cost with no order attached to it is the exact failure this module exists to
#: rule out, so the noise stays and the arithmetic closes.
RECORD_EVERY_CHARGED_ORDER = True


@dataclass
class TradeLedger:
    """Every order a run placed, accumulated per rebalance and joined at the end.

    Rows are appended as small numpy arrays rather than dicts: a 400-name equal-weight
    strategy places ~90,000 orders over a full history, and building that many
    dictionaries costs more than the backtest does.
    """

    security_ids: np.ndarray
    tickers: np.ndarray
    _blocks: list[dict] = field(default_factory=list, repr=False)

    #: Append order. The engine records blocks in the exact order it executed them, and
    #: `reconcile()` walks that order rather than the date column - a forced exit can
    #: carry a date one session BEFORE the rebalance whose cash it actually follows
    #: (see engine._State.carry), and sorting by date would then break an identity that
    #: is not actually broken.
    _n_blocks: int = 0

    # ------------------------------------------------------------- recording

    def record_rebalance(
        self,
        *,
        signal_date: str,
        exec_date: str,
        idx: np.ndarray,
        delta_value: np.ndarray,
        adj_shares_delta: np.ndarray,
        adj_price: np.ndarray,
        as_traded_price: np.ndarray,
        commission: np.ndarray,
        spread: np.ndarray,
        fixed: np.ndarray,
        weight_before: np.ndarray,
        weight_after: np.ndarray,
        nav: float,
        status: str = "filled",
        reason: str = "rebalance",
    ) -> None:
        """One block of orders from a single rebalance. Every array is indexed by idx."""
        n = len(idx)
        if n == 0:
            return
        notional = np.abs(delta_value)
        cost = commission + spread + fixed
        with np.errstate(divide="ignore", invalid="ignore"):
            shares = np.where(np.isfinite(as_traded_price) & (as_traded_price > 0),
                              notional / as_traded_price, np.nan)
        self._n_blocks += 1
        self._blocks.append({
            "block": np.full(n, self._n_blocks, dtype=np.int64),
            "signal_date": np.full(n, signal_date),
            "date": np.full(n, exec_date),
            "security_id": self.security_ids[idx],
            "ticker": self.tickers[idx],
            "side": np.where(delta_value > 0, BUY, SELL),
            "status": np.full(n, status),
            "reason": np.full(n, reason),
            "shares": shares,
            "price": as_traded_price,
            "notional": notional,
            "commission": commission,
            "spread_cost": spread,
            "cost": cost,
            # Negative when money leaves the account. Costs are always a drain, so they
            # subtract on both sides of the trade rather than following its sign. An
            # order that never filled moved no cash at all, whatever it intended to.
            "cash_flow": (-delta_value - cost if status == "filled"
                          else np.zeros(n)),
            "adj_shares": adj_shares_delta,
            "adj_price": adj_price,
            "weight_before": weight_before,
            "weight_after": weight_after,
            "nav": np.full(n, float(nav)),
        })

    def record_exit(self, *, date: str, security_index: int, adj_shares: float,
                    adj_price: float, as_traded_price: float, proceeds: float,
                    reason: str, nav: float) -> None:
        """A position resolved outside a rebalance: a delisting, or a price gap.

        Recorded as a SELL, because that is what happened to the money, with the reason
        that priced it carried through so a reader can see which exits were assumptions
        rather than fills. The delisting return is folded into `proceeds` by the engine,
        so `notional` here is the position's last marked value and `cash_flow` is what
        was actually credited. The two differ by exactly the assumption, which is the
        point of showing both.
        """
        i = int(security_index)
        value = abs(float(adj_shares) * float(adj_price))
        shares = (value / as_traded_price
                  if np.isfinite(as_traded_price) and as_traded_price > 0 else np.nan)
        self._n_blocks += 1
        self._blocks.append({
            "block": np.array([self._n_blocks], dtype=np.int64),
            "signal_date": np.array([date]),
            "date": np.array([date]),
            "security_id": self.security_ids[[i]],
            "ticker": self.tickers[[i]],
            "side": np.array([SELL]),
            "status": np.array(["filled"]),
            "reason": np.array([f"exit:{reason}"]),
            "shares": np.array([shares], dtype=np.float64),
            "price": np.array([as_traded_price], dtype=np.float64),
            "notional": np.array([value], dtype=np.float64),
            "commission": np.zeros(1), "spread_cost": np.zeros(1), "cost": np.zeros(1),
            "cash_flow": np.array([float(proceeds)], dtype=np.float64),
            "adj_shares": np.array([-float(adj_shares)], dtype=np.float64),
            "adj_price": np.array([float(adj_price)], dtype=np.float64),
            "weight_before": np.array([np.nan]), "weight_after": np.zeros(1),
            "nav": np.array([float(nav)]),
        })

    # ---------------------------------------------------------------- output

    def frame(self) -> pd.DataFrame:
        """The whole ledger, one row per order, oldest first."""
        if not self._blocks:
            return pd.DataFrame(columns=list(COLUMNS))
        merged = {k: np.concatenate([b[k] for b in self._blocks]) for k in COLUMNS}
        return (pd.DataFrame(merged)
                .sort_values(["date", "block", "side", "ticker"], kind="stable")
                .reset_index(drop=True))


# --------------------------------------------------------------------------
# Export and audit
# --------------------------------------------------------------------------

def write_csv(trades: pd.DataFrame, path: str | Path) -> Path:
    """Write the ledger as plain CSV - the format anyone can open and check.

    No index column and no comment header: a leading `#` line is the fastest way to make
    a file every spreadsheet reads and half the parsers do not.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out, index=False)
    log.info("trades written: %s (%d rows)", out, len(trades))
    return out


def summarise(trades: pd.DataFrame) -> pd.DataFrame:
    """Per-year buys, sells, notional and cost. The shape of the trading, at a glance."""
    if trades.empty:
        return pd.DataFrame()
    df = trades.copy()
    df["year"] = df["date"].str.slice(0, 4)
    out = df.groupby("year").agg(
        orders=("side", "size"),
        buys=("side", lambda s: int((s == BUY).sum())),
        sells=("side", lambda s: int((s == SELL).sum())),
        names=("security_id", "nunique"),
        notional=("notional", "sum"),
        cost=("cost", "sum"),
    ).reset_index()
    out["cost_bps"] = np.where(out["notional"] > 0,
                               out["cost"] / out["notional"] * 1e4, 0.0).round(1)
    return out


def most_traded(trades: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """Which names the strategy actually spent its money and its costs on."""
    if trades.empty:
        return pd.DataFrame()
    out = trades.groupby(["ticker"]).agg(
        orders=("side", "size"),
        notional=("notional", "sum"),
        cost=("cost", "sum"),
        first=("date", "min"),
        last=("date", "max"),
    ).reset_index().sort_values("notional", ascending=False)
    return out.head(top).reset_index(drop=True)


def reconcile(trades: pd.DataFrame, result) -> dict:
    """Check the ledger against the run it came from. This is the audit.

    Three independent identities, each catching a different class of bug:

      1. every dollar of cost in the headline lands against exactly one order
      2. a rebalance's cash flows move cash from its before value to its after value,
         so the orders and the NAV path are the same story
      3. the ledger never claims more orders than the cost model was asked to price

    A failure on any of these means the ledger and the equity curve disagree, and
    nothing downstream of that is worth reading until it is resolved.
    """
    out: dict = {"n_orders_ledger": int(len(trades))}
    if trades.empty:
        # A strategy that holds cash forever has an empty ledger and is perfectly
        # consistent. An empty ledger against a run that DID trade is not.
        if int(result.costs.n_orders) == 0:
            return out | {"ok": True, "why": "the strategy never traded"}
        return out | {"ok": False,
                      "why": f"{result.costs.n_orders} orders were charged and none "
                             "recorded - was this run made with record_trades=False?"}

    filled = trades[trades["status"] == "filled"]
    rebal = filled[filled["reason"] == "rebalance"]

    charged = float(result.costs.total)
    ledger_cost = float(filled["cost"].sum())
    out["cost_charged"] = round(charged, 6)
    out["cost_in_ledger"] = round(ledger_cost, 6)
    out["cost_matches"] = bool(abs(charged - ledger_cost) <= max(1e-6, 1e-9 * charged))

    out["n_orders_charged"] = int(result.costs.n_orders)
    out["n_orders_recorded"] = int(len(rebal))
    # One ledger row per order the cost model priced. Fewer rows means an order was
    # charged and hidden; more means one was traded and never charged.
    out["order_count_ok"] = bool(len(rebal) == result.costs.n_orders)

    # Replay the cash account in the order the engine wrote the blocks, checkpointing
    # at each rebalance. Block order, not date order: a forced exit is resolved during
    # the carry that FOLLOWS a rebalance and can carry an earlier date than it, so
    # sorting by date would break an identity that holds perfectly well.
    blocks = (filled.groupby("block")
              .agg(flow=("cash_flow", "sum"), reason=("reason", "first"),
                   signal=("signal_date", "first"))
              .sort_index()
              .itertuples(index=False))
    blocks = list(blocks)

    worst, worst_date, i = 0.0, "", 0
    running = float(result.config.get("initial_capital", 0.0))
    for signal, exec_date, cash_after in zip(result.rebalances["date"],
                                             result.rebalances["exec_date"],
                                             result.rebalances["cash"].astype(float)):
        while i < len(blocks) and blocks[i].reason.startswith("exit:"):
            running += float(blocks[i].flow)
            i += 1
        if (i < len(blocks) and blocks[i].reason == "rebalance"
                and blocks[i].signal == signal):
            running += float(blocks[i].flow)
            i += 1
        gap = abs(running - float(cash_after))
        if gap > worst:
            worst, worst_date = gap, str(exec_date)
        running = float(cash_after)   # resync: one bad block must not poison the rest
    out["unconsumed_blocks"] = int(len(blocks) - i)
    out["worst_cash_gap"] = round(worst, 6)
    out["worst_cash_gap_date"] = worst_date
    # A cent of tolerance over ~230 rebalances, not a fraction of NAV: the identity is
    # exact arithmetic, so anything above rounding noise is a real disagreement.
    out["cash_reconciles"] = bool(worst < 0.01)

    out["ok"] = bool(out["cost_matches"] and out["cash_reconciles"]
                     and out["order_count_ok"])
    return out


def format_reconcile(report: dict) -> str:
    lines = ["=" * 68, "TRADE LEDGER RECONCILIATION", "=" * 68]
    for k, v in report.items():
        if k != "ok":
            lines.append(f"  {k:24s} {v}")
    lines += ["", "  PASS - the orders and the equity curve are the same run."
              if report.get("ok") else
              "  FAIL - the orders do not add up to the curve. Do not quote either."]
    return "\n".join(lines)


def holdings(result) -> pd.DataFrame:
    """Long-format target holdings per rebalance: date, security, ticker, weight.

    The companion to the trade ledger. Orders say what moved; this says what was held,
    and an outside reader needs both to reproduce a month.
    """
    w = getattr(result, "weights", None)
    if w is None or not len(w):
        return pd.DataFrame(columns=["date", "security_id", "ticker", "weight"])
    long = (w.stack().rename("weight").reset_index()
            .rename(columns={"level_1": "security_id"}))
    long = long[long["weight"] > 0].reset_index(drop=True)
    long["ticker"] = long["security_id"].map(_ticker_map(result)).fillna("")
    return long[["date", "security_id", "ticker", "weight"]]


def _ticker_map(result) -> dict:
    """security_id -> ticker, from whatever the result already carries."""
    trades = getattr(result, "trades", None)
    if trades is not None and len(trades):
        return dict(zip(trades["security_id"], trades["ticker"]))
    if len(result.exits):
        return dict(zip(result.exits["security_id"], result.exits["ticker"]))
    return {}
