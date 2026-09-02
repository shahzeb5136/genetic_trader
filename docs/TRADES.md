# Trades — the evidence for an equity curve

An equity curve is a claim. A list of orders is the evidence for it.

Everything else in this repository makes a *failure mode* visible — survivorship bias,
lookahead, ticker recycling, silent cache corruption. This makes the *result* checkable.
Somebody who does not trust this engine cannot check a CAGR; there is nothing in it to
check. They can check "on 2007-05-01 this bought 20.06 shares of AAPL at $99.59", because
that is a fact about the world and it is either right or it is not.

```bash
python -m sp500lab backtest trades momentum_12_1
```

```bash
python -m sp500lab report trades momentum_12_1 --open
```

The first writes `reports/trades/momentum_12_1/trades.csv` plus the holdings and any
forced exits. The second writes one self-contained HTML page with the CSV embedded in it,
so a report emailed on its own still carries its evidence.

---

## What is in a row

| column | what it is |
|---|---|
| `signal_date` | the month end whose data produced the decision |
| `date` | the session it executed in — always the **next** one (ADR-017) |
| `security_id`, `ticker` | the internal id survives ticker changes; the ticker is for reading |
| `side` | `BUY` or `SELL` |
| `status` | `filled`, or `unfilled` when there was no opening bar |
| `reason` | `rebalance`, `no_opening_bar`, or `exit:<why>` for a forced exit |
| `shares` | **real** share count — what a broker statement would show |
| `price` | **as-traded** open: `raw_open × cum_split`, the price printed that morning |
| `notional` | `shares × price` |
| `commission`, `spread_cost`, `cost` | this order's share of the run's costs |
| `cash_flow` | signed, net of cost. Negative when money left the account |
| `adj_shares`, `adj_price` | the total-return-adjusted figures the NAV was computed from |
| `weight_before`, `weight_after` | the position as a fraction of NAV, either side of the trade |
| `nav` | portfolio value at the execution open |
| `block` | the order the engine wrote things in. `reconcile()` walks this, not `date` |

### Why there are two prices

`price` is what a broker printed. `adj_price` is what the accounting used. They differ by
the dividend and split chain, and both have to be there:

* hand an outside reader only the **adjusted** price and every check against a quote site
  fails for reasons that have nothing to do with the strategy;
* hand them only the **as-traded** price and the ledger cannot be reconciled against the
  curve, because the curve is computed in adjusted space.

Verified against the source bars: `APH` on 2018-02-01 prints **$91.40** where the stored
split-adjusted close is $22.85 — two later 2:1 splits. `HON` prints **$159.00** against
$150.64, from the 2018 spin-offs. Both match the historical record. This is the check to
run first if you doubt anything here.

---

## The audit

Every export prints it, and it is arithmetic rather than judgement:

```
  cost_charged             18449.365321
  cost_in_ledger           18449.365321
  cost_matches             True
  n_orders_charged         2812
  n_orders_recorded        2812
  worst_cash_gap           0.0
  cash_reconciles          True

  PASS - the orders and the equity curve are the same run.
```

Three identities, each catching a different class of bug:

1. **every dollar of cost lands on exactly one order.** The cost model returns its
   per-order commission and half-spread rather than being re-derived here, so the two
   cannot drift apart.
2. **the cash flows replay the cash column.** For each rebalance,
   `cash_after = cash_before + Σ cash_flow`. Measured across every baseline and all twelve
   alpha strategies: **0.0**.
3. **no order traded without being charged, and none was charged without appearing.**

A `FAIL` means the trade list and the equity curve disagree. One of them is wrong, and
nothing downstream of either is worth reading until it is resolved.

### Why it replays `block` order rather than date order

A forced exit is resolved during the *carry* that follows a rebalance and can carry a date
one session **before** it. Sorting by date would break an identity that is not actually
broken. The `block` column is the engine's own execution order.

---

## Two things this exposed on the day it was built

**Costs charged to orders that did not exist.** Dropping sub-cent "dust" orders from the
ledger broke the cash identity by $1.26 on `equal_weight` — real money charged against
orders nobody could see. Every order the cost model prices now gets a row, however small.
A cost with no order attached to it is exactly the failure this is meant to rule out.

**Commission on unfilled orders.** Costs were priced *before* the fill check, so an order
for a name with no opening bar was charged a commission it never paid. The engine now
resolves fillability first. The effect on published numbers is under 0.01 basis points —
the point is not the magnitude, it is that a wrong thing became visible in an afternoon
because somebody could finally look at the orders. See ADR-029.

---

## Holdings

`holdings.csv` is the companion: `date, security_id, ticker, weight`, one row per position
per rebalance. Orders say what moved; holdings say what was held. An outside reader needs
both to reproduce a month.

---

## Cost of recording

About 12,000 rows for a 50-name strategy over 232 rebalances, and recording changes no
result — `tests/test_trades.py::test_recording_changes_nothing_about_the_run` pins that.

It is **on by default** for ordinary runs and **off inside the genetic algorithm**, where
12,000 rows per individual across 1,500 individuals is real memory. Re-running a winner
with it on is the *same trial* — the fingerprint does not include it — so nothing is lost:

```bash
python -m sp500lab evolve best ga-price-1 --trades reports/trades/ga-price-1
```

---

## From Python

```python
from sp500lab.backtest import run_backtest
from sp500lab.backtest.trades import reconcile, format_reconcile, write_csv, holdings

res = run_backtest("quality_value")
write_csv(res.trades, "trades.csv")
holdings(res).to_csv("holdings.csv", index=False)
print(format_reconcile(reconcile(res.trades, res)))
```

`result.save(dir)` writes both a `trades.parquet` for this codebase and a `trades.csv` for
whoever is checking it — who may have neither Python nor a reason to trust ours.
