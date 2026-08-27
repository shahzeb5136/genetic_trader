# Experiments: the trial log and the holdout

Two mechanisms that keep a strategy search honest. Both had to exist **before** any
searching started, because both are lossy if added later — you cannot retroactively log
a trial you did not record, and you cannot un-see a test period.

If you are returning to this after a break, read §1 and §6 and you will have what you
need.

---

## 1. The thirty-second version

```bash
python -m sp500lab experiments studies          # what you have tried
python -m sp500lab experiments deflate NAME     # does the winner survive the search?
python -m sp500lab experiments holdout          # every look at the reserved period
```

- **Every backtest is logged automatically.** You do not have to remember.
- **Every backtest stops on 2021-12-31 by default.** 2022-01-01 onward is reserved.
- Group related runs with `--study NAME`. That name decides what the deflated Sharpe
  compares your winner against.
- Reaching into the holdout takes an explicit flag and **is always recorded**, even when
  trial logging is switched off.

---

## 2. Why a registry at all

A genetic algorithm explicitly optimises the metric you report. So does a person, more
slowly: you write a strategy, look at its Sharpe, change a lookback, look again, and
quietly abandon three ideas along the way. Every one of those is a trial, and **the best
of N trials has a good Sharpe whether or not there is any signal in the data** — that is
just what a maximum over N draws does.

The deflated Sharpe ratio (Bailey & López de Prado, 2014) corrects for exactly this. It
is implemented in `metrics.py` and needs two numbers that describe the *search*, not the
winner:

| Input | Meaning |
|---|---|
| `n_trials` | how many configurations were evaluated |
| `trial_sharpe_std` | how much their Sharpes varied |

Neither can be reconstructed afterwards. A discarded idea leaves no trace anywhere else
in this repo — not in git, not in the results directory, nowhere. So logging is on by
default, and the log is append-only in the same spirit as the bronze manifest: it
records what happened and nothing rewrites it.

### What a "trial" is

**Distinct configurations, not log lines.** Running the same strategy with the same
parameters four times is one hypothesis tested four times, and counting it as four would
over-deflate — making a real result look worse than it is.

The registry hashes each run's configuration into a `fingerprint`:

> strategy class · parameters · construction · start · end · cost model · capital ·
> liquidity floor · seed

`count_trials()` counts distinct fingerprints. The gap between the `runs` and `trials`
columns in `experiments studies` is re-run work.

The **data version is deliberately excluded** from the fingerprint. Re-running one
configuration after re-ingesting prices is still one hypothesis. The dataset is recorded
separately as `data_fingerprint`, for reproducing a run rather than for counting it.

### Studies are the scope

A `study` is a named search, and the study boundary decides which trials your winner has
to beat. Choose it to match the search you actually ran:

```bash
python -m sp500lab backtest run momentum_12_1 --study momentum-variants
```

```python
from sp500lab.backtest import registry, run_backtest

with registry.study("ga-run-3"):
    for genome in population:          # every run inside is tagged automatically
        run_backtest(EvolvedBlend(**decode(genome)))
```

Splitting one search across several study names understates `n_trials` and flatters the
result. That is the main way to lie to yourself with this tool, and it takes deliberate
effort — which is the best that can be done.

### Daily statistics for reading, monthly for deflating

The registry stores **both**, and this is not redundancy.

The headline Sharpe comes from the daily equity curve. The deflated Sharpe cannot use
it: the derivation assumes approximately independent observations, and a
monthly-rebalanced portfolio holds a constant share vector all month, so its daily
returns are strongly autocorrelated. Treating ~4,900 daily points as ~4,900 independent
observations, when there are really ~176 monthly ones, would overstate the precision of
every Sharpe in the log and make the deflation far too generous.

So `log()` resamples to month ends and stores `sharpe_monthly`, `skew_monthly`,
`kurtosis_monthly`, `n_months` alongside. **Read the daily Sharpe; deflate the monthly
one.** `deflate()` uses only the monthly figures.

---

## 3. Why a holdout, and why it is a constant

Everything from **2022-01-01** onward is reserved for the final test (ADR-025). Backtests
default to `holdout="exclude"` and stop the day before.

The reasoning: run thirty algorithms across 2007–2026, pick the best, and you have spent
the whole sample. There is no clean data left to check the winner against, and no amount
of later work recovers it. Freezing the last stretch costs nothing while you are still
building and is irreplaceable once you are not.

`HOLDOUT_START` is a module constant rather than a parameter, because **a holdout you can
move is not a holdout**. Moving it forward silently converts test data into training
data.

### The three modes

| Mode | What runs | Recorded as a look? |
|---|---|---|
| `exclude` *(default)* | 2007-05 → 2021-12 | no |
| `include` | the full history, holdout treated as ordinary data | **yes** |
| `only` | the holdout alone — this is the final test | **yes** |

```bash
python -m sp500lab backtest run my_algo --holdout only --study final-test
```

### The asymmetry that matters

Trial logging can be switched off for a scratch run:

```bash
SP500LAB_REGISTRY=off python -m sp500lab backtest run my_algo
```

**The holdout ledger cannot.** You may run something without logging it as a trial; you
may never look at the holdout without leaving a trace. Every look is appended to
`holdout_log.jsonl` with the date, strategy, study, mode and git commit, and a loud
warning is printed at the time.

That is deliberate. Each look degrades the holdout, and the ledger is the only way to
know how degraded it is when you come to trust a final number.

---

## 4. The workflow

**While researching** — nothing to remember. Runs are logged, the holdout is protected.

```bash
python -m sp500lab backtest run my_algo --study my-idea --notes "value tilt, 30 names"
```

**Check what you have tried:**

```bash
python -m sp500lab experiments studies
```

```
    study  runs  trials best_sharpe best_strategy  touched_holdout
baselines     8       8        0.69       low_vol                0
```

**Before believing anything:**

```bash
python -m sp500lab experiments deflate baselines
```

```
  n_trials                           8
  trial_sharpe_std                   0.2517
  n_months                           175
  sharpe_annualised_daily            0.6891
  sharpe_annualised_monthly          0.855
  expected_max_sharpe_annualised     0.3673
  psr_vs_zero                        0.9982
  deflated_sharpe                    0.952
```

Read that middle line carefully: **with eight trials, the luckiest of eight worthless
strategies would have posted a 0.37 Sharpe.** That is the bar `low_vol` had to clear, and
it cleared it — but only just, at 0.952 against a 0.95 threshold, and that is with a
mere eight trials. A GA run of 10,000 sets a far higher bar.

Read the DSR as a probability, not a score. Below ~0.95 the result is not distinguishable
from the best of N lucky draws, however good the raw Sharpe looks.

**Only at the very end**, with one final candidate:

```bash
python -m sp500lab backtest run winner --holdout only --study final-test
```

Then stop. If it disappoints and you go back to searching, the holdout is spent — and
the ledger will show that it is.

---

## 5. What is stored

Two append-only JSONL files under `data/experiments/`:

| File | Contents | Can be disabled |
|---|---|---|
| `runs.jsonl` | one line per backtest — 54 fields | yes, `SP500LAB_REGISTRY=off` |
| `holdout_log.jsonl` | one line per look at the holdout | **no** |

Each run record carries:

- **identity** — `run_id`, `fingerprint`, `study`, `logged_at`, `notes`
- **configuration** — strategy, class, params, construction, warmup, dates, holdout mode,
  cost model, capital, liquidity floor, seed
- **headline statistics** (daily) — CAGR, vol, Sharpe, Sortino, max drawdown, Calmar,
  turnover, positions, cost drag, IR, beta, alpha
- **monthly statistics** — `n_months`, `sharpe_monthly`, `skew_monthly`,
  `kurtosis_monthly`
- **honesty** — coverage min/median, forced exits, unresolved exits, spread fallbacks,
  unfilled orders, ruin flag
- **costs** — total, commission, spread, traded notional, order count
- **provenance** — `git_commit`, **`git_dirty`**, `panel_key`, `data_fingerprint`,
  runtime

`git_dirty` is worth its line: a result produced from a working tree with uncommitted
changes is not reproducible, and it is better to know that from the record than to
discover it while trying to rebuild the run.

These files are **not** in git — a GA run appends thousands of lines. They belong in the
backup set with `data/vault/` and `data/bronze/`: you cannot re-derive what you tried.
See RUNBOOK → Backups.

### Robustness

`_append` checks whether the file ends in a newline and inserts one if not. A process
killed mid-write leaves a partial line; appending straight onto it would concatenate two
records and destroy the *good* one along with the broken one. `_read` also skips
unparseable lines rather than failing the whole log. Same failure shape as the chunk-cache
bug in ADR-013 — intact by every check, and still wrong.

---

## 6. Using it from Python

```python
from sp500lab.backtest import registry, run_backtest

# runs are logged automatically; `study` groups them
res = run_backtest("momentum_12_1", study="momentum-variants")
res.config["run_id"]        # where it landed

registry.load("momentum-variants")      # DataFrame of every run in the study
registry.studies()                      # summary across all studies
registry.count_trials("momentum-variants")
registry.best("momentum-variants")
registry.deflate_best("momentum-variants")
registry.holdout_touches()              # the ledger

with registry.study("ga-run-3"):        # tag a whole search
    ...
```

Useful flags on `run_backtest`: `holdout=`, `study=`, `log_run=`, `notes=`.

---

## 7. Things that surprise people

**The default window changed.** Backtests now end 2021-12-31, so every number is over
~176 rebalances instead of ~232. Comparing against an older figure is comparing different
windows.

**The rankings changed with it.** Over 2007–2021, `equal_weight` (11.10%) beats SPY
(10.42%) and `low_vol` beats it on Sharpe — neither was true over the full period. The
2022–2026 stretch was unusually kind to cap-weighted mega-caps. That is a preview of why
the holdout is worth keeping: a regime it contains is a regime your research window does
not.

**The last rebalance of a window is dropped.** A rebalance on the final session has
nowhere to fill — execution is at the *next* open (ADR-019), which would be past the
window, and under a holdout policy that means reaching into reserved data. Dropping it is
the honest choice.

**Acceptance checks are not trials.** `backtest accept` passes `log_run=False` and leaves
the holdout policy alone. It is calibration, not research, so it does not pollute the
registry or the ledger. Checks 1 and 1b read the SPY series over its full 2000–2026
history: that is a property of the data feed, and an index's long-run total return is not
something anyone can select a strategy with.

**Tests do not pollute the registry.** `tests/conftest.py` sets `SP500LAB_REGISTRY=off`
at import. Over-counted trials would silently make every real result look worse.

---

## 8. Related

- `docs/DECISIONS.md` **ADR-025** (holdout) and **ADR-026** (registry)
- `docs/BACKTEST.md` — the engine these runs come out of
- `src/sp500lab/backtest/registry.py` — the implementation, documented at length
- `tests/test_registry.py` — 42 tests, mostly about the two silent failure modes
