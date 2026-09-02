# The backtest engine

The harness every strategy is scored on. Genetic algorithms, neural networks and
classical rules all implement one interface, see the same point-in-time view, pay the
same costs, and are measured by the same statistics. The scoreboard does not know who
is playing — that is the whole point of building it before building any competitor.

Read this before writing a strategy. Read §2 before trusting a number.

---

## 1. Run it

```bash
python -m sp500lab backtest accept
```

That is the gate. Six checks, and if any of them fails nothing downstream means
anything. Current state: all pass, SPY's total return reproduced to **0.2 basis
points**.

```bash
python -m sp500lab backtest baselines
```

```bash
python -m sp500lab backtest run momentum_12_1 --all-costs --annual --study my-idea
```

Every run is logged as a trial and stops before the reserved holdout by default. That is
`docs/EXPERIMENTS.md`, and it is worth reading before you start writing strategies.

First-time setup, in order — the engine reads two gold tables that must exist first:

```bash
python -m sp500lab backtest build-delisting
```

```bash
python -m sp500lab backtest build-spreads
```

---

## 2. Acceptance — what these checks buy you

| Check | What it catches | Measured |
|---|---|---|
| 1. SPY total return | The entire adjustment chain, in one number | 8.32%/yr vs 8.32% expected (0.2bp) |
| 1b. Dividend contribution | An inverted adjustment factor | total − price = 1.88%/yr |
| 2. Equal-weight identity | The engine's own bookkeeping | 11.298% vs 11.122% reference (17.6bp) |
| 3. Leakage guard | All three routes to the future | 3 of 3 cheats blocked |
| 4. Determinism | A noisy fitness function | Same seed identical, different seed differs |
| 5. Dividends counted once | A "fix" that double-counts them | total − price = 2.26%/yr |

Check 1 is the calibration instrument and it is unusually informative because the three
failure modes land on three different numbers:

- **~6.43%/yr** — dividends are being dropped. The engine is computing price return.
- **~8.32%/yr** — correct.
- **~10.2%/yr** — dividends are being counted twice.

You cannot get 8.32% by accident. That is why buy-and-hold SPY is the first thing the
engine has to reproduce, and why it is worth more than any number of unit tests on the
accounting.

Check 4 tests both directions. Identical seeds must match *and* different seeds must
diverge — a seed that is accepted and then ignored would make every "independent" run
of the noise floor the same run, and the noise floor is what the whole competition is
measured against.

---

## 3. The four decisions the engine encodes

Everything else is arithmetic.

### 3.1 Execution is at the next open. Always.

A signal computed from data up to and including the **close of day T** fills at the
**open of day T+1**. Filling at the close of T means trading on a price that was not
knowable when the signal formed, and it is the single most common way a backtest
manufactures returns that do not exist.

This lives in `engine.py`, once. No strategy can get it wrong and no strategy has to
remember to get it right.

### 3.2 Leakage is structural, not a rule

The important design decision in the whole engine. A strategy is never handed the panel
and asked to filter by date. It is handed `PanelView`, whose arrays are numpy **views**
that physically end at the as-of session:

```python
self.close = panel.adj_close[:t + 1]   # O(1), no copy, t+1 rows exist
```

`view.close[t + 1]` is an `IndexError`, not tomorrow's price. The future is not filtered
out of the object — it was never in it.

Three escape routes exist in principle, and all three are closed and tested:

| Route | What happens |
|---|---|
| Index past the end | `IndexError` — the rows do not exist |
| Ask for a future date | `LookaheadError` from `ctx.price_on()` |
| Reach through to the panel | `AttributeError` — `Context` holds no reference to it |

`accept.py` runs a deliberately cheating strategy for each. If any of them stops
raising, acceptance fails.

### 3.3 Dividends are already in the prices — never add them again

`adj_close` is total-return adjusted through `adj_factor` (splits + dividends, ADR-006).
Holding a constant number of adjusted shares **already reinvests dividends**. Crediting
cash dividends on top double-counts them and inflates returns by roughly the dividend
yield — about 1.9pp/yr on SPY, which is more than enough to make a bad strategy look
good.

There is deliberately no dividend accrual anywhere in `engine.py`. If you find yourself
about to add some because a position "isn't earning its dividends", run acceptance check
1 first — it is already earning them.

A consequence worth stating plainly: **"shares" in the engine are adjusted-space shares.**
They are notional, not the count on a brokerage statement. Real share counts, which the
per-share commission needs, are recovered at execution time through `cum_split`
(`normalize/splits.py`).

### 3.4 A position that stops having prices must resolve to an outcome

Otherwise it silently vanishes from the accounting and the strategy never books the
result. This is survivorship bias running backwards — it deletes the losses instead of
the losers — and it is worse than the usual kind because nothing errors and no row is
missing. The equity curve simply does not contain the day the company failed.

See §6.

---

## 4. Writing a strategy

### The contract

```python
def target_weights(self, ctx: Context) -> np.ndarray | pd.Series:
    """Non-negative weights summing to at most 1.0 (ADR-016)."""
```

The engine **rejects** a violation rather than normalising it. A strategy that returns a
negative weight has a bug, not a short position, and normalising it away would hide the
bug behind a plausible number.

### Prefer scoring to weighting

Almost every strategy should subclass `SignalStrategy` and implement `score(ctx)`
instead — a number per security, higher is better:

```python
from sp500lab.backtest import SignalStrategy, Construction, register

@register("my_momentum")
class MyMomentum(SignalStrategy):
    warmup = 273                                  # sessions of history needed
    construction = Construction(top_k=50, weighting="equal", max_weight=0.05)

    def score(self, ctx):
        return ctx.trailing_return(252, skip=21)  # 12-1 momentum
```

Portfolio construction — top-k selection, weighting scheme, per-name caps, tie-breaking,
long-only validation — is then shared with every other competitor. That matters more
than it looks: if each competitor writes its own, the competition partly measures who
wrote better position-sizing code rather than who has better signal.

**`NaN` means "no opinion" and the name is skipped. It is not the same as scoring zero,**
and only one of those should be able to earn a position.

### The tie-break is a modelling choice, not a detail

`portfolio.py` breaks ranking ties on a stable hash of the security_id. Ordering by the
id itself is deterministic — and quietly selects survivors, because ids are assigned in
first-observed order and **99.0% of the low half of the id range is still priced today
against 61.1% of the high half**.

That is not hypothetical. It is how ADR-024 was found: a strategy with all four signal
weights set to zero — no opinion about anything — scored **17.65%/yr at a Sharpe of
0.89**, beating every honest baseline and beating SPY, purely from the tie-break.
Nothing errored, and every acceptance check passed including determinism, because the
run *was* reproducible; it was reproducibly wrong.

If you write a strategy that produces many tied scores (a binary rule, a GA that drives
its weights toward zero), the tie-break is doing your portfolio construction. Use
`ctx.tiebreak`, which `SignalStrategy` already does for you.

### Where each model family plugs in

```python
# Traditional — an indicator over the bounded price view
def score(self, ctx):
    w = ctx.window(200)
    return ctx.latest() / np.nanmean(w, axis=0) - 1.0

# Genetic — the genome arrives in params; decode and evaluate
def score(self, ctx):
    lb, thresh = self.params["lookback"], self.params["threshold"]
    mom = ctx.trailing_return(lb, skip=21)
    return np.where(mom > thresh, mom, np.nan)

# Neural — a forward pass over point-in-time features
def on_start(self, panel):
    self.model = load_model(...)          # once, not per rebalance

def score(self, ctx):
    x = ctx.features                       # (S, F), knowable on ctx.as_of
    return self.model.predict(x)
```

`on_start` sees the whole panel, so anything computed there **must** be trailing-only.
It exists for loading a model or allocating buffers. When in doubt, compute inside
`score()`.

### What `Context` gives you

| Accessor | Shape | Notes |
|---|---|---|
| `ctx.close` | `(t+1, S)` | adjusted closes; last row **is** `as_of` |
| `ctx.latest()` | `(S,)` | the as-of row |
| `ctx.window(n)` | `(≤n, S)` | most recent `n` sessions; returns fewer near the panel start |
| `ctx.trailing_return(lb, skip)` | `(S,)` | total return over `lb`, ending `skip` sessions ago |
| `ctx.universe` | `(S,)` bool | index members on `as_of` — survivorship-free |
| `ctx.tradable` | `(S,)` bool | universe **and** priced **and** above the liquidity floor |
| `ctx.positions`, `ctx.cash`, `ctx.nav` | | current portfolio state |
| `ctx.rng` | | seeded generator — use this, never `np.random` |
| `ctx.prices` | DataFrame | convenience, ~1000× slower; never in a fitness function |

Allocating to a name outside `ctx.tradable` raises. That is not a nuisance — it is the
survivorship-bias guard, and a strategy that trips it is trying to buy a stock that was
not in the index that month.

---

## 5. Costs

Turnover is where backtests lie. Costs are charged by the engine, never by the strategy,
and **every result reports three settings**.

| Component | Treatment under ADR-016 |
|---|---|
| Commission | $0.005/share, $1.00 minimum, capped at 1% of trade value (IBKR-shaped) |
| Half-spread | Estimated — see below |
| Market impact | **Omitted.** Negligible below $100k on S&P 500 large caps |
| Borrow | **Omitted.** Long-only |

The omissions are consequences of the mandate, not shortcuts. Raise the capital and
impact stops being negligible; allow shorts and borrow stops being zero.

### The $1 minimum is the dominant cost at this scale

In a $100k portfolio holding 50 names, a full rebalance of one name is a ~$2,000 trade
of maybe 40 shares — $0.20 of per-share commission. **The $1 minimum binds instead**, so
commission is effectively a flat $1 per name traded: about 5bp on a $2,000 trade, and
much worse as the portfolio gets more concentrated in capital terms.

This makes `top_k` a real economic decision rather than a tuning parameter. The
`equal_weight` baseline holds 387 names at $100k — $258 per position — and pays 92.8bp
of traded notional in costs, almost all of it minimums. The engine reports how many
orders were priced by the minimum versus the 1% cap, so this shows up as a number rather
than as a mystery.

### Estimating a spread without quote data

Two estimators are implemented, and the interesting finding is that neither is sufficient
alone.

**Corwin & Schultz (2012)** compares a two-day high-low range against two one-day ranges;
volatility scales with interval length and the spread does not, so the volatility term
cancels. Two details matter:

- **Overnight gap adjustment.** Skipping it is the biggest source of overestimation.
- **Averaging convention.** Roughly half the two-day estimates come out negative for a
  liquid name. Truncating each to zero and *then* averaging gives `E[max(X,0)]` — a pure
  positive bias. **Measured here: that convention reports 36bp for AAPL in 2018-19,
  against a real quoted spread of about 1bp.** So this module averages the signed
  estimates and truncates the average.

**The tick floor.** Corwin-Schultz was built for a market with 50-100bp spreads; its
sample runs 1927-2006. For a mega cap in 2018 the true spread is 1-2bp — more than an
order of magnitude below the estimator's resolution, so it correctly returns zero. But
zero is still wrong, because a spread cannot be narrower than one tick:

```
half_spread ≥ MIN_SPREAD_TICKS × tick_size(date) / 2 / as_traded_price
```

`tick_size` is $0.01 after decimalisation (2001-04-09) and **$0.0625 before it**, which
makes the pre-2001 cost regime genuinely different rather than assumed away.
`MIN_SPREAD_TICKS = 2.0` is the one modelling assumption in the file, stated rather than
buried.

The as-traded price is required, not the stored close — ours is split-adjusted (ADR-007),
so a tick is a different fraction of it than of the price that actually traded.

Result: the floor binds for liquid modern names and the estimator takes over where it
can actually resolve. That division of labour is the design.

```
era          median half-spread   floor binds
2000-2001         15.58 bp            75%
2002-2007          4.47 bp            57%
2008-2009          6.72 bp            55%
2010-2015          3.35 bp            65%
2016+              2.48 bp            65%
```

Spot checks: AAPL 2024 **0.52bp**, MSFT 2024 **0.24bp**, KO 2024 **1.68bp**. 2008-09 is
wider than 2010-15, and pre-decimalisation is wider still. Both directions are right.

### The three settings, and why all three

| Setting | Charge |
|---|---|
| `optimistic` | commission only |
| `realistic` | commission + 1× estimated half-spread |
| `pessimistic` | commission + 2× estimated half-spread |

The spread estimate is the weakest number in the chain, so `pessimistic` doubles it
rather than adding a different term. **A strategy that survives only under `optimistic`
is not a strategy — it is a bet that the spread estimator is wrong in your favour.**

§8 shows exactly why this is not a formality.

---

## 6. Delisting returns

Three outcomes that are constantly conflated and are not the same event:

| Case | What happened | Handling | Return |
|---|---|---|---|
| `index_removal` | Still trading, just dropped from the index | Sell at the next open | 0 |
| `acquisition` | Bought for cash or stock | Exit at deal terms; approximated by the last price | 0 |
| `bankruptcy` | Equity wiped out | Liquidate at the last close | **−1.0** |

Treating a removal as a bankruptcy fabricates catastrophic losses. Treating a bankruptcy
as a removal silently deletes them. Both are large and they do not cancel.

Classification parses the free-text `reason` in `sp500_changes`, matched to a security by
ticker **and** date proximity to the membership end (a match more than a year away is
rejected — ticker alone is not an identifier, ADR-005). Current state:

```
era         acquisition  bankruptcy  index_removal  unresolved
2010+           120           2           212           78
pre-2010          7           1            51           47
```

Two rows classified as bankruptcy still had prices 90+ days later and were reclassified
as removals: the price series is stronger evidence than the wording. PG&E's 2019
Chapter 11 is the instructive case — it filed, and its equity was *not* wiped out.

**125 of 518 are unresolved** and default to an index removal at the last price. That is
the conservative choice for the common case and the wrong one for a bankruptcy, so the
count is reported per run rather than defaulted away. `sp500_changes` is under-recorded
before 2010 (ADR-010), which is why most unresolved rows are early-era. Every row carries
an `assumption` column recording in words what was assumed, so a reader can disagree with
a specific number rather than discover a silent one.

### The part that gets worse before it gets better

Only **4** delisting cases are visible today, because Yahoo carries almost no delisted
names — the bars are not there to be mishandled. With full coverage there would be
hundreds. **Buying the paid feed exposes this problem rather than solving it**, which is
why it is built now rather than during the EODHD migration.

### The exit buffer, and why it is 45 days

A name dropped from the index in month M is still in `universe_asof(M-end)`, so the
engine only sells it at the M+1 rebalance — up to ~35 calendar days later. With no
buffer that bar does not exist and a live company gets booked as a delisting.

Measured, not guessed: of 207 closed membership intervals that have prices, **203 keep
trading past 45 days and only 4 stop** — exactly the handful of genuine delistings in the
data. So 45 days separates "removed from the index, still listed" from "actually
delisted" cleanly, while staying an order of magnitude below the 365-day window
`quality/checks.py` uses to detect a reassigned symbol.

### Two outcomes that are not delistings

- **Price gap.** If bars resume later, the exit is recorded as `price_gap` with a zero
  return. Charging a bankruptcy return to a data gap would fabricate a loss. (There are
  24 interior gap-sessions across 3.7M bars, so this is rare by construction.)
- **Unfilled order.** A name priced at the signal date can have no bar at the execution
  date. Requiring a bar at T+1 to be "tradable" at T would be lookahead — you cannot know
  on Monday that a stock will not open on Tuesday. What actually happens to a
  market-on-open order is that it does not fill, so that is what the engine does. The
  intended notional stays in cash and the miss is reported.

---

## 7. Coverage is a result, not a footnote

**This is the largest single limitation on anything this engine produces.**

The point-in-time index is reconstructed from Wikipedia revisions and is essentially
complete. The *prices* are not. Yahoo does not carry most companies that have since
delisted, so the further back you go, the smaller the fraction of the real index that
can actually be traded:

| Rebalance | In the index | Priced | Coverage |
|---|---:|---:|---:|
| 2007-04-30 | 499 | 273 | **54.7%** |
| 2008-12-31 | 497 | 294 | 59.2% |
| 2011-06-30 | 499 | 313 | 62.7% |
| 2015-08-31 | 501 | 362 | 72.3% |
| 2019-10-31 | 503 | 430 | 85.5% |
| 2026-08-26 | 498 | 498 | 100% |

**343 index members have no usable price history at all.**

A backtest starting in 2007 is trading a 273-name subset of a 470-name index. That subset
is not random — it is the survivors — so the coverage gap is a *second* survivorship bias
sitting underneath the point-in-time universe that was so carefully built. The membership
data is honest about who was in the index; the price data is not honest about who could
be bought.

The engine therefore reports coverage on **every** run, measured against the true
membership count rather than against the panel's own columns. Measuring against the panel
would report ~99% by construction and hide the entire problem — which is exactly what an
earlier version of `Panel.coverage()` did before it was corrected.

`--min-coverage 0.8` refuses to run below a threshold. Use it when a result is going to
be quoted.

Closing this gap is TODO-8 (the EODHD paid migration), which mainly buys back 2007-2012.

---

## 8. What the baselines actually say

Every baseline, **2007-05-01 → 2021-12-31** (the research window — 2022 onward is the
holdout, ADR-025), $100k, monthly, long-only.

**Realistic costs**

```
strategy          CAGR     vol  Sharpe    maxDD  turnover  cost_drag  names
equal_weight    11.10%  22.67%    0.58  -58.25%    38.90%      0.96%    358
random_weight   10.24%  22.78%    0.54  -61.23%  1025.65%      3.55%     50
rolling_ridge    9.87%  24.63%    0.51  -46.48%   589.52%      2.48%     48
low_vol          9.84%  15.34%    0.69  -39.89%   191.26%      0.69%     50
evolved_blend    6.46%  21.66%    0.40  -51.70%   449.79%      1.50%     50
momentum_12_1    6.39%  23.24%    0.38  -55.53%   353.62%      1.63%     50
short_reversal   6.36%  30.47%    0.35  -77.65%  1003.95%      5.06%     50
cash             0.00%   0.00%    0.00    0.00%     0.00%      0.00%      0
```

`evolved_blend` runs its default genome — a starting individual, not a tuned one — and
`rolling_ridge` is the learned template. Neither has been searched or fitted, so treat
them as interface demonstrations rather than results.

And before believing any of it: with just **eight** trials in this study, the deflated
Sharpe puts `low_vol` at **0.952** against a 0.95 threshold — the luckiest of eight
worthless strategies would have posted a 0.37 Sharpe on its own. Eight. A GA run of
10,000 sets a far higher bar. See `docs/EXPERIMENTS.md`.

**Buy-and-hold SPY over the identical window: 10.42%/yr, Sharpe 0.59, maxDD −55.19%.**

Three results worth sitting with.

**The bar is roughly the index, and only two things clear it.** `equal_weight` beats SPY
on return (11.10% vs 10.42%) with more volatility and a worse drawdown; `low_vol` beats
it clearly on Sharpe (0.69 vs 0.59) with a much shallower drawdown. Nothing else does.
That is the bar every genetic algorithm and neural network has to clear, and if a model
beats it by a lot on the first try the overwhelmingly likely explanation is a bug in the
model, not an edge.

**The window matters more than you would like.** Over the *full* 2007–2026 history,
nothing beat SPY at all — the 2022–2026 stretch was unusually kind to cap-weighted
mega-caps, and it flatters the index against everything equal-weighted. That period is
now the holdout, so these numbers exclude it. Two honest tables, different conclusions,
same engine. It is a preview of exactly why the holdout is worth keeping: it contains a
regime the research window does not.

**Costs reorder the scoreboard.** Under `optimistic` costs, `random_weight` posts
**11.17%/yr and the second-best Sharpe of the entire suite** — beating 12-1 momentum.
Under `realistic` it drops to 8.87%; under `pessimistic`, 6.57%. Its cost drag runs
0.72% → 3.03% → 5.32%/yr, because random selection turns the portfolio over 1,037% a
year.

A strategy ranked on optimistic costs would conclude that randomness beats momentum. That
is not a hypothetical failure mode — it is what these numbers say, and it is why the
engine reports three settings and why `random_weight` is in the baseline suite at all.
Run it across several seeds: the spread is the width of the null distribution, and any
strategy inside it has demonstrated nothing.

---

## 9. Multiple-testing control, before the GA arrives

A genetic algorithm explicitly optimises the metric you report. Run 10,000 individuals
against 230 monthly observations and the best one will have a beautiful Sharpe ratio
**whether or not there is any signal in the data at all** — that is what a maximum over
10,000 draws does.

The **deflated Sharpe ratio** (Bailey & López de Prado, 2014) asks whether the winner
beats what the luckiest of N worthless strategies would have posted anyway:

```python
from sp500lab.backtest import deflate_result
deflate_result(result.performance, n_trials=10_000, trial_sharpe_std=0.15)
```

`n_trials` and `trial_sharpe_std` are properties of the **search**, not of the winner,
which is why the GA must log every individual it evaluates — losers included. Without the
trial count, a reported Sharpe is not optimistic or conservative; it is meaningless.

`metrics.py` also provides the probabilistic Sharpe ratio and corrects both for skew and
fat tails, which independently inflate the naive Sharpe's apparent precision.

No scipy dependency: the normal CDF and its inverse are implemented from standard
rational approximations, accurate to ~1e-9.

---

## 10. Performance

A full backtest — 232 rebalances over ~500 names — runs in **~0.17s**, and the
acceptance suite's fast path in 0.07s. A GA with population 200 over 50 generations is
10,000 evaluations, so this is roughly 28 minutes of fitness evaluation rather than the
28 hours a 10-second backtest would cost.

Three things buy that, and all three are load-bearing:

1. **The panel is built once** and cached on disk and in-process. A GA calling
   `build_panel()` 10,000 times gets the memoised object every time after the first.
2. **The loop is per rebalance, never per security.** Within a holding period the share
   vector is constant, so the whole NAV path for that period is one matrix-vector product.
3. **Context slices are views.** `panel.adj_close[:t+1]` allocates nothing.

The regression test asserts under 1s per backtest. If it starts failing, something in a
strategy is allocating per rebalance — usually `ctx.prices`.

---

## 11. Layout

```
src/sp500lab/backtest/
    panel.py       (date × security) matrices, built once, cached
    context.py     the bounded point-in-time view — read this first
    strategy.py    the interface, SignalStrategy, the registry
    portfolio.py   score → weights, shared by every competitor
    costs.py       commission + estimated spread, three settings
    spreads.py     Corwin-Schultz + tick floor → gold/
    delisting.py   exit assumptions → gold/
    benchmark.py   total-return benchmark series
    engine.py      the rebalance loop and the accounting
    metrics.py     performance stats, deflated Sharpe
    results.py     BacktestResult, save/load, comparison
    accept.py      the six acceptance checks
    cli.py         `sp500lab backtest ...`

src/sp500lab/strategies/
    baselines.py   the null hypotheses every competitor must beat
```

Gold tables the engine writes and reads:

| Table | Rows | What |
|---|---:|---|
| `gold/backtest/half_spread` | 3,706,372 | per (security, date) half-spread + which term bound |
| `gold/backtest/delisting_returns` | 518 | exit assumption per security, with reasoning text |
| `gold/backtest/panel/*.npz` | — | cached panel, keyed by build parameters |

---

## 12. What this engine does not do yet

Honest list. None of these block strategy work; all of them bound what a result means.

- **No feature layer** (TODO-4). `Context` carries a `features` slot and slices it by
  knowledge date, but `data/gold/` has no feature matrices yet. Until it does, every
  competitor computes its own inputs, and the competition partly measures who wrote
  better feature code. This is the next thing to build.
- **No walk-forward harness.** The engine runs one period. Purging, embargo and a
  touch-once holdout are the GA's scaffolding and are not built.
- ~~No experiment registry.~~ **Built** — see `docs/EXPERIMENTS.md`, ADR-025/026.
- **Price coverage** (§7). The binding constraint on the early years.
- **No point-in-time market cap or sectors** (TODO-5, TODO-6). Any cap-weighted or
  sector-neutral construction is currently unavailable or leaks.
- **Fundamentals start 2009-04.** Price-only strategies get the full window.
- **Monthly rebalancing only.** The schedule comes from `trading_calendar.is_month_end`.
  Anything faster would re-expose the sub-month membership dating error that ADR-016
  deliberately designed around.

---

## 13. Related reading

- `docs/EXPERIMENTS.md` — the trial log and the holdout. Read before searching.
- `docs/DECISIONS.md` ADR-016 through ADR-026 — why the engine is shaped this way
- `docs/HANDOFF.md` — project state and the remaining TODO list
- `docs/DATA_DICTIONARY.md` — the gold tables above
- `docs/ARCHITECTURE.md` — the data layer underneath all of this

---

## What was added after this document was first written

Three things, each with its own document, all of which run through this engine unchanged.

**Every order is now recorded** (`result.trades`, ADR-029). The engine writes a ledger with
the as-traded price and real share count next to the adjusted figures it computed the NAV
from, and `trades.reconcile()` replays the cash account against it. Building it exposed two
real bugs in this file: orders that never filled were being charged commission, and the
per-share commission was derived from the execution day's *close* rather than its *open*.
See [TRADES.md](TRADES.md).

**Strategies can read a shared feature layer** (`ctx.feature(name)`, ADR-030). A strategy
declaring `requires_features` gets the panel loaded automatically, and one declaring
`min_date` has its start moved forward rather than sitting in cash for three years and
reporting the flat stretch as performance. See [FEATURES.md](FEATURES.md).

**A genetic algorithm evaluates ~0.15s per individual** (ADR-031). Nothing in the engine
changed to make that possible; what changed is that features are precomputed and the
fitness function slices its folds out of the equity curve rather than re-running. See
[EVOLUTION.md](EVOLUTION.md).

And one new command that matters more than it looks:

```bash
python -m sp500lab backtest suite all
```

Strategies in this project do not all cover the same window — anything built on XBRL
fundamentals starts in 2010 — and SPY returned 10.42%/yr from 2007-04 against 15.66%/yr
from 2010-07. `suite` puts each strategy next to the index over **its own** dates and sorts
by the difference, because a leaderboard that sorts them together is actively misleading.
