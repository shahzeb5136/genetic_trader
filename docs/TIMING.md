# The Calendar Lab

*WHEN the market pays, at daily granularity — the overnight/intraday decomposition and
nine calendar rules, costed three ways on SPY.*

Everything else in this project chooses **which** stocks to hold and rebalances monthly.
This lab tests a different family of claims: that returns are not uniform across the
clock or the calendar. The overnight session is not the trading day, Friday's close is
not Monday's open, and the turn of the month is not the middle. Design decisions:
[ADR-036](DECISIONS.md) for the lab, [ADR-047](DECISIONS.md) for its report set.

```bash
python -m sp500lab report timing --open
```

writes `reports/timing/`: an index — the two identities, the decomposition, every rule
under all three cost settings, the per-ticker table — and **one page per rule carrying
both windows**, because a rule's research numbers and its forward verdict are one story.
The rules are not on the monthly roster and never will be: a rule that sits in cash 80% of
the time gets a structural Sharpe boost, and sorting it into a scoreboard of thirty fully
invested stock pickers would rank it high for a reason that is not skill.

```bash
python -m sp500lab timing accept        # the two identities - run this first
python -m sp500lab timing suite         # every rule x three cost settings, vs SPY
python -m sp500lab timing decompose     # per-ticker overnight/intraday split
python -m sp500lab timing run tm_overnight --all-costs
python -m sp500lab timing seal all --rationale "..."     # pre-register, spends nothing
python -m sp500lab timing forward all                    # SPENDS LOOKS, recorded forever
```

---

## The machine

A session has two tradable legs, and any calendar rule is two boolean vectors:

    hold_overnight[t]   hold from the close of session t to the open of t+1
    hold_intraday[t]    hold from the open of session t to its close

Buy-and-hold is both always true; the overnight strategy is `hold_overnight` alone; and
"be invested close-to-close on day d" compiles to `hold_intraday[d]` plus
`hold_overnight[d-1]`. The engine (`timing/engine.py`) walks the sessions toggling one
bit of state — invested or cash — at two checkpoints per session, charging the shared
cost model on the whole account at every transition.

**Why a close fill is not lookahead here, when ADR-017 forbids it everywhere else.** The
monthly engine's next-open rule stops a signal trading on information from its own fill.
A calendar rule has no signal — its schedule was knowable years in advance — so a
market-on-close order placed at 15:50 fills at the price the rule was always going to
trade. The one rule that reads data (`tm_vix_overnight`) conditions on the **prior**
session's VIX close, lagged accordingly. Any future rule conditioning on same-session
data must move its entry to the next open, or it is manufacturing returns.

**Prices.** SPY raw OHLC from the benchmarks table, adjusted by the same
`compute_factors` chain the benchmark and the whole panel use (ADR-006). The adjustment
factor steps at the ex-date — between one close and the next open — so the dividend lands
in the overnight leg exactly as it does in the world. Get that convention backwards and
the "overnight premium" grows by SPY's entire dividend yield.

**Costs.** SPY has no row in `gold_half_spread` and Corwin-Schultz cannot resolve a
sub-basis-point spread anyway, so the half-spread is the ADR-020 tick floor:
`tick_size(date) / price` (~0.8bp in 2007, ~0.2bp today), plus the IBKR-shaped
commission. Optimistic / realistic / pessimistic mean what they mean everywhere else.

## The two identities

`sp500lab timing accept` asserts both; the unit tests hold them on synthetic data too.

1. **Calibration.** Both legs on, zero costs, must reproduce the adjusted SPY series —
   the same series the monthly engine's ADR-018 acceptance test is calibrated to.
   Measured drift: **0.00 bp/yr**.
2. **Decomposition.** `(open[t+1]/close[t]) × (close[t+1]/open[t+1]) = close[t+1]/close[t]`,
   so the overnight-only and intraday-only strategies *partition* buy-and-hold: their NAV
   product must equal it at every session, gross. Measured max error: **6×10⁻¹⁵**.

The second identity is what makes "the overnight share of SPY's return" a measurement
rather than an estimate.

## The headline measurement

2007-04 → 2021-12, gross of costs:

| leg | CAGR | Sharpe | maxDD |
|---|---:|---:|---:|
| overnight only (close→open) | **8.31%** | **0.71** | **−29.4%** |
| intraday only (open→close) | 2.21% | 0.22 | −46.7% |
| both (buy & hold) | 10.70% | 0.60 | −55.2% |

Overnight carried roughly four times intraday's return at a fraction of its drawdown.
That reproduces the standing literature result (Cooper, Cliff & Gulen 2008 and since) on
this project's own, survivorship-audited data.

**And then the costs.** The overnight rule trades ~500 times a year:

| costs | tm_overnight CAGR | cost drag |
|---|---:|---:|
| optimistic | 6.74% | 1.57%/yr |
| realistic | 3.66% | 4.65%/yr |
| pessimistic | 0.67% | 7.64%/yr |

**The gap between gross and net is the finding.** At retail size the anomaly is real and
mostly untradable, which is why the tradable expression of the same fact lives in the
monthly engine: `overnight_momentum` ranks the cross-section on the overnight *component*
of momentum and trades twelve times a year (ADR-037).

## The nine rules

Fixed schedules, zero fitted parameters — each is a published anomaly at its paper's
conventional definition, so the family is nine hypotheses, not nine hundred
parameterisations. Claims live in each rule's docstring (`timing/strategies.py`) and on
the report page; sample sizes differ by orders of magnitude and the honest reading
starts there:

| rule | claim | entries |
|---|---|---|
| `tm_buy_hold` | the bar, and the calibration instrument | 1 |
| `tm_overnight` | the market pays its night shift | 3,715 |
| `tm_intraday` | the control: the day carries risk for nothing | 3,715 |
| `tm_vix_overnight` | the overnight premium concentrates in high-VIX regimes | 1,721 |
| `tm_weekend` | Friday close → Monday open is special (French 1980) | 769 |
| `tm_turn_of_month` | last session + first three carry the premium (Lakonishok & Smidt) | 178 |
| `tm_month_end_drift` | the run-up INTO month end is the flow window | 177 |
| `tm_pre_holiday` | the session before a closure outperforms (Ariel 1990) | 133 |
| `tm_sell_in_may` | hold November–April (Bouman & Jacobsen 2002) | **16** |

**`entries` is the sample size, and it is computed, not asserted.** It counts the round
trips each rule makes over 2007-04 → 2021-12: the engine walks the legs in time order
(intraday[t], overnight[t], intraday[t+1], …) and the rising edges of that interleaved
vector are the entries. It is what the cost model charges for and what an independent
observation *is* for a schedule known years in advance — and it is not the session count.
`tm_sell_in_may` is invested across 1,806 sessions and enters sixteen times; sixteen is
what a Sharpe estimated from it is worth. `queries.rule_schedule()` produces the column
and `test_the_entries_column_reproduces_the_documented_sample_sizes` pins this table to
it, so the prose here cannot drift from the code (ADR-047).

`tm_turn_of_month` and `tm_month_end_drift` are designed so at most one can be true —
they split the same institutional-flow story at the month boundary. `tm_sell_in_may` is
included partly to demonstrate that daily machinery cannot rescue a seasonal hypothesis:
the effective sample is the number of seasons, not the number of sessions
(WHAT_TO_BUILD_NEXT.md's warning, honored).

**Result, realistic costs, research window: nothing beats buy-and-hold on ΔSharpe**, and
the family's best trial deflates to 0.69 against its own 27-trial search — below the
0.95 bar. That is the correct null result, published as such.

## The per-ticker decomposition

`sp500lab timing decompose` splits every index member's return into its two legs over
exactly the sessions it was in the point-in-time index (membership-clipped — invariant 3),
minimum 500 sessions. Gross of costs *by design*: trading a single name close-to-open
crosses its spread ~500 times a year, so the table says where the anomaly **lives**, and
the monthly `overnight_momentum` strategy is its costed expression. The full CSV is
embedded in the report page.

## Trials, holdout, forward tests

A timing backtest is a first-class citizen of the honesty machinery: it logs to the same
experiment registry (study `timing-1` holds the research runs), the holdout clamp and its
unsilenceable ledger apply unchanged, and the forward harness runs calendar rules through
the **same** seal → paired-comparison → verdict pipeline as every other candidate — the
forward engine takes a `runner` argument and the lab passes its leg engine
(`timing/cli.py::timing_runner`). Verdicts land in the same store and the same
`report forward` pages.

All nine rules were sealed and forward-tested on 2026-08-30 as part of the 2026-08 wave.
Read those verdicts with ADR-037's contamination note attached: the rules were *chosen*
after the first forward test's results were known.

Those verdicts live on each rule's own page in `reports/timing/`, rendered from the stored
record by the forward set's own sections (`forward_views.outcome_sections()`) rather than
by a second implementation. The rules stay out of `reports/forward/`'s pages and inside
its multiple-testing bar — a candidate that was looked at counts whether or not it has a
page — and that index links here.

## What would extend this honestly

- **More instruments through the same engine** (RSP for the equal-weight version of the
  overnight effect, IWM for small caps): `load_timing_data(ticker)` already takes the
  symbol; the missing piece is only the half-spread convention per instrument.
- **FOMC pre-announcement drift** (Lucca & Moench 2015) needs the scheduled announcement
  dates — a small reference table nothing in the current lake carries. Add the data
  first; the rule is then four lines.
- **Parameterised gates** (VIX thresholds, moving-average filters on the overnight leg)
  are searches, not rules: run them as a registered study with the deflation, or not at
  all.
