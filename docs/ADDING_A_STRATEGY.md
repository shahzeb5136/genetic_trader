# Adding a strategy

Open [`src/sp500lab/strategies/custom.py`](../src/sp500lab/strategies/custom.py). That file
is yours — nothing else in the project writes to it, and it already contains one working
example you can edit or delete.

```python
@register("my_idea")
class MyIdea(FeatureStrategy):
    """One sentence somebody could argue with."""     # <- becomes the report's headline
    requires_features = ("gross_profitability", "vol_126d")
    min_date = "2010-07-01"          # only if the inputs do not exist earlier
    construction = STANDARD          # 50 names, equal weight, 5% cap

    def score(self, ctx):
        e = ctx.tradable
        return blend([rank_pct(self.f(ctx, "gross_profitability"), e),
                      rank_pct(-self.f(ctx, "vol_126d"), e)])
```

Add the name to `GROUPS["custom"]` in
[`strategies/__init__.py`](../src/sp500lab/strategies/__init__.py) and it appears
everywhere:

```bash
python -m sp500lab backtest run my_idea --all-costs --annual
```
```bash
python -m sp500lab backtest trades my_idea
```
```bash
python -m sp500lab report strategy my_idea --open
```
```bash
python -m sp500lab report all custom --open
```

Nothing else needs editing. The registry logs it, the holdout guard protects it, the cost
model charges it, the trade ledger records its orders and the report set picks it up.

---

## The contract, in four rules

**1. Return a score, not weights.** Higher is better, one number per security, aligned to
`ctx.security_ids`. `portfolio.py` turns it into a portfolio — the top-k cut, the per-name
cap, the long-only check and the unbiased tie-break are shared, so the scoreboard compares
your *idea* against everyone else's rather than your position sizing against theirs.

Return weights directly (override `target_weights`) only when the construction **is** the
idea — a risk-parity overlay, a regime switch that changes how much is invested. See
`defensive_regime` in `alpha.py`.

**2. NaN means "no opinion", and it is not zero.** A name scored NaN is passed over; a name
scored `0.0` is ranked last. Those are different statements and only one should be able to
earn a position.

**3. Only score names in `ctx.tradable`.** That mask is *in the index on this date* AND
*priced* AND *liquid enough*. The engine raises rather than quietly allocating outside it —
that refusal is the survivorship guard.

**4. You cannot see the future, structurally.** `ctx.close` is a numpy view that physically
ends at the as-of date, so `ctx.close[len(ctx.close)]` raises `IndexError` rather than
returning tomorrow's price. There is nothing to remember not to do (ADR-017).

---

## What you can read

### The shared feature layer — 75 columns

```bash
python -m sp500lab features list
```

`self.f(ctx, "mom_12_1")` returns one `(S,)` array. Declaring the name in
`requires_features` is what makes the engine load the panel for you and fail loudly on a
typo, instead of handing you a context with no features and letting every score come back
NaN and the run report a flat cash curve.

Everything in the layer is documented in [FEATURES.md](FEATURES.md) and in the generated
`reports/features.html`, which says what each one is, which end is historically good, and
how often it actually has a value.

### Or compute your own

```python
ctx.close                        # (t+1, S) adjusted closes; the last row IS today
ctx.window(126)                  # the last 126 sessions
ctx.trailing_return(252, skip=21)  # 12-1 momentum
ctx.latest()                     # today's row
ctx.universe, ctx.tradable       # point-in-time masks
ctx.positions, ctx.cash, ctx.nav # what you currently hold
ctx.rng                          # seeded; never use np.random, or runs stop reproducing
```

Both are fine. The feature layer exists so that two strategies ranking on momentum rank on
the *same* momentum — not to stop you writing your own.

⚠️ `ctx.prices` returns a DataFrame and is ~1000× slower. Never touch it in anything a
genetic algorithm will evaluate.

### The scoring grammar

From `strategies/signals.py`:

| helper | what it does |
|---|---|
| `rank_pct(x, mask)` | percentile rank in [0,1] within the date. **The workhorse** — a bad share count is worth "first place", not +25 standard deviations |
| `zscore(x, mask)` | winsorised z-score, when magnitude genuinely carries information |
| `blend([a, b], weights, min_components)` | averages the components a name *has*, rather than letting one NaN annihilate the row |
| `conditional(primary, condition, mask, keep)` | a double sort: score on one signal among the top slice of another |
| `require(mask, *features)` | narrow eligibility to names that have a value |

`blend` matters more than it looks. Summing and letting NaN propagate silently shrinks the
universe to names with complete fundamental data — which, because coverage correlates with
survival, is a survivorship filter dressed up as an arithmetic convention.

---

## Reading the result honestly

```bash
python -m sp500lab backtest suite custom
```

That puts your strategy next to **SPY over its own window**, which is the only comparison
here that means anything. Strategies in this project do not all cover the same period —
anything using SEC fundamentals starts in 2010 — and SPY returned 10.42%/yr from 2007-04
against 15.66%/yr from 2010-07. A raw CAGR ranks windows, not strategies.

Then, before believing anything:

```bash
python -m sp500lab experiments deflate adhoc
```

A strategy you wrote *after* looking at the scoreboard is a trial too. Every backtest is
logged automatically for exactly this reason — you cannot retroactively count an idea you
discarded, and the deflated Sharpe needs that count.

And check the orders:

```bash
python -m sp500lab backtest trades my_idea
```

Every export prints its own audit: for each rebalance `cash_after = cash_before +
Σ cash_flow`, and every dollar of cost attributed to exactly one order. If that fails, the
trade list and the equity curve are not the same run.

---

## Three things that will bite

**Your first result will be too good, and it will be a bug.** Nothing hand-written in this
repository beats the index by a wide margin. If yours does on its first run, look for the
bug before looking for the champagne — the usual causes are scoring a name outside
`ctx.tradable`, requiring a fundamental that only survivors have, or comparing against SPY
over a different window than the one you ran.

**A `min_date` is not optional if your inputs start late.** Without it the strategy sits in
cash from 2007 to 2010 and that flat stretch is averaged into its CAGR, which makes it look
worse and its Sharpe look better.

**Turnover is where backtests lie.** Run `--all-costs`. A strategy that only works under
`optimistic` is a bet that the half-spread estimator is wrong in your favour. `random_weight`
— which has no signal at all — posts the second-best Sharpe in the whole suite under
optimistic costs.

---

## Letting the genetic algorithm search near your idea

If your idea is a weighted combination of features, the GA already searches that space:

```bash
python -m sp500lab evolve run --study my-search --preset full --generations 25
```

To make it search a *different* set of features, edit `PRICE_FEATURES` or
`FUNDAMENTAL_FEATURES` in
[`strategies/genome.py`](../src/sp500lab/strategies/genome.py). Every feature you add
multiplies the search space and the trial count the deflated Sharpe has to discount, so add
them because you have a reason, not because they exist. [EVOLUTION.md](EVOLUTION.md) has
the rest.

---

## Twelve worked examples

[`strategies/alpha.py`](../src/sp500lab/strategies/alpha.py). Between them they cover a
plain ranking, a conditional double sort, an event-defined universe, a hard exclusion, a
multi-factor blend, and a regime switch that changes how much is invested. Whatever shape
your idea is, one of them is close to it.
