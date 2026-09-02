# Forward testing: what happened after the research window ended

Every number this project has produced so far comes from **2007-04 to 2021-12**.
Strategies were written, tuned, evolved and ranked inside that window. The genetic
algorithm's folds are inside it too, and [ADR-032](DECISIONS.md) says plainly what they
measure: consistency, not generalisation.

The only out-of-sample evidence that exists is the reserved period from **2022-01-01
onward** ([ADR-025](DECISIONS.md)). This document is about spending it properly, and the
`sp500lab forward` command tree is the machinery for doing so.

If you are returning to this after a break, read §1 and §8 and you will have what you
need.

---

## 1. The thirty-second version

```bash
python -m sp500lab forward window                             # what exists, what it proves
python -m sp500lab forward seal low_vol --rationale "..."     # pre-register; spends nothing
python -m sp500lab forward run low_vol --dry-run              # what a look would cost
python -m sp500lab forward run low_vol                        # <- THIS SPENDS THE HOLDOUT
python -m sp500lab forward scoreboard                         # prediction against outcome
```

- **`window` and `dry-run` are free.** So is `seal`. Everything else reads reserved data.
- **A forward test is a paired comparison**, not a number: the research window is the
  prediction, the forward window is the outcome, and the record carries both plus the
  standard error of the gap.
- **All three cost settings run in one look**, because fetching a missing one later would
  cost a second.
- **Every look is recorded** in the same holdout ledger `backtest run --holdout only`
  writes to, and it cannot be switched off.

---

## 2. What "forward testing" means here, precisely

Three windows, and the third is the one people forget.

```
2007-04 ....................... 2021-12 │ 2022-01 ................. today
<-------- RESEARCH (searchable) -------->│<------- FORWARD (reserved) ------->
                                         │           <--- fresh --->
                                         │        (only on look #2 and after)
```

| Window | What it is | Who may see it |
|---|---|---|
| **Research** | `2007-04-01` → `2021-12-31` | everything: searches, tuning, ranking |
| **Forward** | `2022-01-01` → end of data | one deliberate, recorded look |
| **Fresh** | the part of the forward window that arrived since this candidate was last tested | genuinely new evidence, even on a repeat look |

The forward window currently holds **55 month ends**. It grows by one every month
whether anybody runs anything or not, and that is the only mechanism by which this
project acquires new out-of-sample evidence.

### Two modes, and the difference is real money

| Mode | What it runs | The question it answers |
|---|---|---|
| `paired` *(default)* | two independent backtests; the forward one starts from a fresh $100k with an empty book | *"I read the research and started trading in 2022."* |
| `continuous` | one unbroken backtest 2007 → today, sliced at the boundary | *"I had been running this all along."* |

The paired forward leg pays entry costs for its whole portfolio in the first month, so it
comes out **slightly worse**. Neither mode is more correct; they answer different
questions, they cost the same single look, and `mode` is on the record so nobody can
report whichever came out better without it being visible.

---

## 3. The number that governs everything: 54 months is not much

This is the single most important fact in this document, and it is measured rather than
asserted. The standard error of a Sharpe ratio over *n* monthly observations is
roughly `sqrt((1 + SR²/2) / (n − 1))` (Mertens 2002; Lo 2002), which annualises to:

| months of forward data | 95% band on a Sharpe of 1.0 |
|---:|---|
| 24 | ±1.42 |
| 36 | ±1.15 |
| 54 | ±0.93 |
| 120 | ±0.62 |

**Fifty-four months cannot distinguish a Sharpe of 0.1 from a Sharpe of 1.9.** And a
*difference* between two windows carries √2 times the error of one, so the smallest gap
this window could resolve is about **1.35 Sharpe** — larger than almost any real effect.

Two consequences, and the whole design follows from them:

1. **A forward test here can refute a strategy and cannot confirm one.** A large negative
   result is informative *because it is large*. A modest positive result is
   indistinguishable from noise.
2. **So the verdict vocabulary is asymmetric.** `held` means "not refuted", never "works".
   Every summary prints the band next to the point estimate for exactly this reason.

`sp500lab forward window` prints this for the window you actually have.

---

## 4. Pre-registration, and why it is not bureaucracy

The holdout stops a strategy from being **fitted** to 2022 onward. It does nothing at all
about the far more common failure, which is **choosing what to test after seeing how it
did**.

Forward-test twenty strategies, report the three that worked, and the holdout has quietly
become a second research window. Every individual run was honest. The trial log would show
nothing wrong. The guarantee ADR-025 bought is gone anyway.

Medicine solved this with trial pre-registration and the reasoning transfers exactly. A
**seal** is a record, written before the look, of:

- which strategy, down to a configuration fingerprint;
- what its research window produced — **the prediction**;
- how many trials that research cost, so the prediction is already deflated;
- and *why* this candidate, in free text.

```bash
python -m sp500lab forward seal ga-price-1-best \
    --rationale "highest deflated Sharpe of the three GA runs; the only one above 0.95"
```

Sealing runs the research window only. **No holdout data is read and nothing is spent.**

### Declared or auto

Requiring a separate `seal` before every test would be the strict design and would be
routed around within a week. So `forward run` **auto-seals** anything it has not seen:
it measures the research window first, records that as the prediction, then looks.

| mode | what it proves |
|---|---|
| `declared` | sealed by its own command at a timestamp that precedes the look |
| `auto` | the numbers are clean — the prediction is still research-only — but the *ordering* is unproven |

A set of `declared` seals written on one day and tested on another is real
pre-registration. A pile of `auto` seals is a survey, and the scoreboard's `seal_mode`
column says which you have.

**The earliest line for a `seal_id` binds.** Seal ids are a hash of the configuration, not
a timestamp, so re-sealing the same candidate lands on the same id — and a seal rewritten
after a disappointing look cannot replace the prediction it failed to meet.

---

## 5. What a forward test actually does

`forward run` does five things in order:

1. **Backtests the research window** (`holdout="exclude"`, stops 2021-12-31). No look is
   spent. This is the prediction.
2. **Finds or writes the seal.** A declared seal binds; otherwise an auto seal is written.
3. **Backtests the forward window** (`holdout="only"`). *This is the look.* The existing
   holdout ledger records it before the run starts and cannot be silenced.
4. **Compares the two legs**, with standard errors on the difference.
5. **Stores** the record, both month-end curves, and — for a single strategy by default —
   the whole forward result including its order-by-order trade ledger.

### The comparison

```
                             research      forward       change
  months                          175           54
  CAGR                          9.84%        5.43%       -4.41%
  Sharpe (daily)                 0.69         0.49        -0.20
  Sharpe (monthly)               0.85         0.54        -0.31
  vs benchmark CAGR            -0.58%       -8.28%       -7.70%
  vs benchmark Sharpe            0.10        -0.34        -0.44
  max drawdown                -39.89%      -16.20%      +23.69%
  turnover                    191.26%      220.70%      +29.44%

  Sharpe 95% band        [-0.40, 1.48]  on 54 months of forward data
  decay significance     -0.56 sigma (p=0.289 that a drop this large is sampling noise)
  P(true SR > 0)         0.870
  P(true SR > index)     0.276
  P(true SR > research)  0.257
```

Three probabilities, three different questions:

| | question |
|---|---|
| `psr_vs_zero` | did it make risk-adjusted money at all? |
| `psr_vs_benchmark` | did it beat the index **over the same dates**? |
| `psr_vs_research` | did it live up to what research promised? |

The third is the one this harness exists for and the one nobody computes. It is the
probabilistic Sharpe with the *research* Sharpe as the benchmark — the same machinery the
deflated Sharpe uses with the expected-maximum as its benchmark; only the right-hand side
of the comparison changes.

### The checks and the verdict

Nine named conditions, each independently readable:

| check | passes when |
|---|---|
| `enough_data` | at least 24 months of forward data |
| `no_ruin` | NAV never reached zero |
| `made_money` | forward CAGR above zero |
| `beat_benchmark` | forward Sharpe above the index's, over the same dates |
| `positive_excess` | forward CAGR above the index's, over the same dates |
| `kept_its_edge` | it beat the index in research **and** still does *(abstains if it never did)* |
| `decay_within_noise` | the Sharpe drop is within one standard error of the difference |
| `drawdown_held` | the forward drawdown is no deeper than the research one |
| `turnover_held` | it trades no more than 1.5× as often as it used to |

A check can pass, fail, or **abstain** (`n/a`) when a number is missing. Abstention is not
failure: a strategy that never beat the index is not penalised for still not beating it.

The verdict is computed from those, in a fixed precedence:

| verdict | meaning |
|---|---|
| `inconclusive` | fewer than 24 months — no verdict is offered rather than a weak one |
| `failed` | ruin, or a loss out of sample, or a Sharpe collapse beyond 1.96σ |
| `decayed` | still works, but lost its edge over the index, or dropped beyond 1σ |
| `held` | **nothing here refutes it** — which is not the same as confirming it |

---

## 6. Data vintage: why a second look is mostly not new evidence

The forward window grows. A second look a year later is, for the most part, a re-reading
of data already seen — *except* for the twelve months that arrived in between, which are
genuinely untainted.

So every record stores the vintage it ran against, and the next look reports how much of
itself is new:

```
  look number     #2 at this candidate under each cost setting
  NEW since last  2025-01-01..2026-08-26  (19 months) - the only part of this
                  result that is fresh evidence
```

When nothing has changed, it says so: *"the data has not moved since the last look, so
this is the same measurement again, not a second confirmation."*

`forward window` lists every sealed candidate with the fresh data available for it, which
is the honest way to decide when a re-test is worth running.

The vintage is the **last date the previous look actually saw**, not the panel's end.
Those differ whenever a run was stopped early with `--forward-end`, and using the panel's
end there would retire months no look has ever covered.

---

## 7. Multiple testing on the forward window itself

Pick the best of twenty forward tests and the winner's Sharpe carries the same
best-of-N inflation the deflated Sharpe corrects for in a research search. The correction
is identical, so the same function computes it:

```
  MULTIPLE TESTING ON THE FORWARD WINDOW ITSELF [realistic costs]
    candidates looked at        12
    spread of forward Sharpes   0.4330
    luckiest-of-N bar           0.6212     <- what the luckiest of 12 worthless
    best forward Sharpe         0.9100        candidates would have posted
```

Counted per **candidate**, not per run: three cost settings of one strategy are one
hypothesis under three assumptions, and counting them as three would triple the apparent
search.

Because the forward runs are logged as a real study, the registry's own machinery works
on them directly:

```bash
python -m sp500lab experiments deflate forward-test
```

There, `n_trials` is the number of distinct candidates the forward window has been asked
about — which is exactly the multiple-testing exposure of the holdout.

---

## 8. The workflow

**Decide what to test, and write it down.** Ideally on a different day from the test.

```bash
python -m sp500lab forward window
python -m sp500lab forward seal ga-price-1-best --rationale "the only GA winner above 0.95 DSR"
python -m sp500lab forward seal multi_factor    --rationale "the untuned bar any model must beat"
python -m sp500lab forward seals
```

**Check what it would cost.**

```bash
python -m sp500lab forward run ga-price-1-best --dry-run
```

**Spend it, once.**

```bash
python -m sp500lab forward run ga-price-1-best --rationale "the pre-registered candidate"
```

**Read it.**

```bash
python -m sp500lab forward scoreboard
python -m sp500lab forward show fwd-20260828T141416-e5ed5e
python -m sp500lab experiments holdout
```

Then stop. If the result disappoints and you go back to searching, the holdout is spent —
and the ledger will show that it is.

### A whole group in one decision

```bash
python -m sp500lab forward suite alpha --rationale "the twelve hypotheses, all at once"
```

Defensible — it is one decision rather than twelve sequential peeks — but emphatically
**not** twelve independent tests. The selection bar in §7 is printed with the result for
exactly that reason.

---

## 9. What is stored

Three append-only JSONL files under `data/experiments/forward/`:

| File | Contents | Can be disabled |
|---|---|---|
| `seals.jsonl` | one line per pre-registration | **no** |
| `forward_runs.jsonl` | one line per forward test — both legs, the comparison, the verdict | **no** |
| `forward_curves.jsonl` | month-end curves for both windows, keyed by `forward_id` | **no** |

Plus, by default for a single `forward run`, the whole result under
`results/forward/<forward_id>/`: equity, rebalances, weights, exits, and the
order-by-order trade ledger as both parquet and CSV.

`SP500LAB_REGISTRY=off` silences the *trial* log. It does not silence any of these, for
the same reason it cannot silence the holdout ledger. Measured: 5.6 KB per record,
10.4 KB per curve pair, 1.5 KB per seal — and there will be tens of them ever.

Each record carries:

- **identity** — `forward_id`, `seal_id`, `seal_mode`, `batch_id`, `logged_at`
- **configuration** — strategy, class, params, construction, mode, cost model, capital,
  liquidity floor, seed, benchmark, rationale
- **the prediction** — the sealed research leg, nested under `research`
- **the re-measurement** — the research window as it measures *today*, under
  `research_recomputed`, with `seal_drift_sharpe` as the gap. Non-zero means the data
  changed under the prediction; a large value invalidates the comparison rather than
  adjusting it.
- **the outcome** — the forward leg, nested under `forward`
- **the comparison** — every delta, `decay_se`, `decay_z`, `decay_p`, the three PSRs, the
  Sharpe band, every named check, the verdict and its reason
- **vintage** — `data_end`, `previous_data_end`, `fresh_start`, `fresh_months`,
  `look_number`
- **the search behind the candidate** — `study`, `n_trials`, `trial_sharpe_std`,
  `deflated_sharpe_research`
- **honesty** — coverage, forced exits, unresolved exits, unfilled orders, spread
  fallbacks, ruin flag, and `holdout_looks_total` at the time of the run
- **provenance** — `git_commit`, `git_dirty`, `panel_key`, `data_fingerprint`, runtime,
  `saved_to`

Both legs are also ordinary logged runs, so `research_run_id` and `forward_run_id` link
straight into `runs.jsonl` and `curves.jsonl`.

### Curves keep the opening level

A forward window opening on 2022-01-01 first trades on 2022-02-01, and its first month
end is 2022-02-28. Rebasing to that month end would silently drop February from the
curve, the chart and the 2022 row of the annual table — about 2% of the evidence in a
54-month window. So the stored grid is month ends **plus the window's own first session**.

---

## 10. The reports

```bash
python -m sp500lab report forward --open
```

Writes a whole directory — `reports/forward_tests/` by default — and every page is
self-contained: no server, no CDN, no build step.

| File | What it is |
|---|---|
| `index.html` | the executive summary: verdicts, the paired scoreboard, the research-versus-forward scatter, every curve |
| `forward-<name>.html` | one per candidate: prediction against outcome, the significance of the gap, all nine checks, the curve, every year, the monthly grid, all three cost settings, full provenance, and the orders it placed |
| `decay-analysis.html` | the cross-section: did the research ranking predict the forward ranking? |
| `honesty.html` | what limits all of it |
| `EXECUTIVE_SUMMARY.md`, `DECAY_ANALYSIS.md`, `HONESTY.md`, `markdown/*.md` | the same pages as text, for reading without a browser |
| `data/*.csv` | the records, the curves and the seals, so the numbers are separable from the argument |

**Nothing in that path runs a backtest or reads the panel.** Every figure comes out of
`data/experiments/forward/`, which is the ADR-034 guarantee being exercised rather than
asserted — and `test_the_stored_record_is_enough_to_rebuild_the_comparison` holds it.

### The read API the reports are built on

`store.py` exposes seven pure read functions — DataFrames and dataclasses in, no markup
— and the views live in `reporting/forward_views.py` on the same seam
[ADR-028](DECISIONS.md) draws for everything else. Use them directly for anything the
pages do not already show.

```python
from sp500lab.forward import store

store.load()                       # every forward test, one flat row each
store.scoreboard()                 # the paired table, sorted, report-ready
store.selection_bar()              # the best-of-N bar for the forward window
store.get(forward_id)              # one record, structured
store.load_curves([forward_id])    # {'research': df, 'forward': df}
store.stitched_curve(forward_id)   # one continuous series, with `attrs['join_date']`
store.annual_table(forward_id)     # year by year, strategy against the index
```

`load()` flattens the nested legs into `research_*`, `recomputed_*` and `forward_*`
columns and adds `checks_failed` as a comma-separated list. The JSON stays structured so
code can rebuild a `Leg`; the DataFrame is flat so a table can render it.

`stitched_curve()` splices the forward leg onto the end of the research leg for one
readable chart. It is a **presentation, not a simulation** — the two legs are independent
runs — and `attrs["join_date"]` marks the boundary so a chart can draw it.

`sp500lab report forward` needs nothing from this package beyond those seven calls, and
nothing from the panel at all.

---

## 11. Reading order for the code

| Module | What lives there |
|---|---|
| `forward/windows.py` | the three windows, the vintage arithmetic, and what a short window can prove. Pure — no I/O |
| `forward/compare.py` | deltas, standard errors, the named checks, the verdict rules. Also pure |
| `forward/legs.py` | the seam: a `BacktestResult` reduced to a comparable `Leg`, whole or sliced |
| `forward/seal.py` | pre-registration |
| `forward/store.py` | what is kept, and the read API above |
| `forward/engine.py` | the orchestration: measure, seal, look, compare, record |
| `forward/cli.py` | `sp500lab forward ...` |
| `reporting/forward_views.py` | the four reports, composed from the store. Pure |
| `reporting/render/markdown.py` | the same reports as text — the second backend |

`windows.py` and `compare.py` are pure by design, which is what lets
`tests/test_forward.py` pin every verdict rule and every standard error against
hand-built inputs, with no data and no disk.

---

## 12. Things that surprise people

**Sealing spends nothing.** It runs the research window only. The refusal in
`create_seal` is load-bearing: a seal cannot be built from a run that saw the holdout,
because a prediction measured on the period it is about to be tested against is not a
prediction.

**The forward run starts a month after the boundary.** The first rebalance is the first
month end inside the window and it fills at the *next* open (ADR-019). So a 55-month
window produces a 54-month result, and both numbers appear in the output.

**Three cost settings are one look, not three.** `look_number` counts per (candidate,
cost model, mode). Three ledger entries appear because three runs read reserved data, but
the candidate has been looked at once.

**`held` is not a pass.** See §3. It means fifty-odd months failed to refute the research
claim, and §3 says how little that is worth.

**A missing study means no deflated Sharpe.** With no trial count, the "deflated" Sharpe
degenerates into the probabilistic Sharpe against zero — a real statistic under a
misleading name, and the most flattering possible reading of an unattributed candidate.
It reports `n/a` instead. Pass `--origin-study` when the recovery from the trial log
fails.

**The holdout-reading tests are off by default.** `conftest.py` is explicit that nothing
in the suite should touch the holdout, and an integration test of a forward harness
plainly does. Redirecting the ledger into `tmp_path` keeps the *record* clean but removes
the mechanism that was supposed to notice, and a look that happened is a look that
happened. So `pytest` on a fresh clone runs 58 tests — every window rule, every statistic,
every verdict, every storage guarantee — and reads no reserved data at all. The seven
integration tests need an explicit opt-in:

```bash
SP500LAB_FORWARD_TESTS=1 python -m pytest tests/test_forward.py
```

They assert on structure — boundaries, ledger entries, round-trips — never on a
performance figure, so running them tells you the harness works without telling you how
any strategy did.

---

## 13. Related

- `docs/DECISIONS.md` **ADR-033** (the design), **ADR-034** (what is stored) and
  **ADR-035** (the report set and the Markdown backend)
- `docs/EXPERIMENTS.md` — the trial log and the holdout ledger this builds on
- `docs/BACKTEST.md` — the engine both legs run through
- `docs/EVOLUTION.md` — ADR-032 on why the GA's folds are not this
- `src/sp500lab/forward/` — the implementation, documented at length
- `tests/test_forward.py` — 65 tests, mostly about the silent failure modes; the
  seven that read reserved data need `SP500LAB_FORWARD_TESTS=1`
