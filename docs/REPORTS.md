# Reports

Static HTML over the experiment registry. One self-contained file per report — no server,
no CDN, no build step. Open it offline, keep it next to the commit that produced it, send
it to someone.

---

## 1. Run it

```bash
python -m sp500lab report study baselines --open
```

```bash
python -m sp500lab report run --study baselines
```

```bash
python -m sp500lab report registry
```

```bash
python -m sp500lab report compare momentum_12_1 low_vol equal_weight
```

```bash
python -m sp500lab report honesty
```

```bash
python -m sp500lab report forward --open
```

Output lands in `reports/` (gitignored — rebuildable from the registry at any time).
`-o path.html` overrides, `--open` launches a browser.

| Report | Answers |
|---|---|
| `study` | Which of these is better, and by how much |
| `run` | What did this one strategy actually do |
| `registry` | What have I tried, and does the winner survive the search |
| `compare` | Specific runs or strategies side by side |
| `honesty` | What would make me distrust all of the above |
| `forward` | Did any of it survive 2022 onward — and is the change bigger than the noise |

---

## 2. What a report shows

**Comparison** — equity curves on a log scale (rebased to the first date all of them
share, so nothing gets credit for starting earlier), per-strategy drawdown panels, a
scoreboard with the best value in each column highlighted, a risk/return scatter, and
cumulative performance relative to SPY.

**Deep dive** — KPI tiles, growth against both the benchmark and the same strategy with
costs switched off, drawdown, calendar-year bars, rolling 3-year Sharpe and 1-year
volatility, a monthly returns heatmap, and a cost breakdown.

**Registry** — study summaries, a deflated-Sharpe panel per study, and a sortable
leaderboard of every run.

**Honesty** — coverage by run, unresolved delisting exits, dirty working trees, and the
holdout ledger.

### Charts are interactive, within the page

- **Click a legend entry** to hide that series — on every chart in the report at once, so
  isolating one strategy on the equity chart also isolates it on drawdown.
- **Click a column header** to sort. Sorting is on the underlying number, never the
  rendered text — otherwise `9.84%` sorts above `11.10%`.
- **Hover** a bar, dot or heatmap cell for its exact value.

Light and dark both work from the same file, following the reader's system setting.

---

## 3. Two things the reports deliberately keep saying

**The caveats travel with the numbers.** Every report carries an honesty section, and the
deflation panel sits next to the scoreboard rather than behind a link. A report that shows
a 12% CAGR and mentions on page four that half the index was untradable has already misled
its reader.

**Drawdown charts are shallower than the truth.** They are drawn from month-end curves, so
an intra-month trough is invisible. The `maxDD` column comes from the daily curve and is
the number to quote. Every drawdown chart says so in its caption rather than leaving a
reader to assume.

---

## 4. Architecture

The important part is the seam, not the charts. A frontend is the most likely thing to get
rewritten, and if it reaches into `BacktestResult` internals or parses JSONL by hand, every
rewrite risks quietly changing what a number means.

```
src/sp500lab/reporting/
    theme.py     colours, number formatting, metric direction — the only file that
                 knows what things look like
    series.py    equity curves -> plot-ready arrays.        PURE
    tables.py    registry rows -> formatted tables.         PURE
    specs.py     what to draw, never how. The seam.
    views.py     composes registry data into a Report.      PURE
    forward_views.py
                 the same, over the forward-test store.     PURE
    render/
        charts.py    spec -> inline SVG    the only file that computes pixels
        html.py      Report -> one self-contained document
        markdown.py  Report -> text
    cli.py       `sp500lab report ...`
```

Everything up to and including `views.py` produces **no markup**. That is why
`tests/test_reporting.py` can assert that the shallowest drawdown is highlighted, or that
`align()` rebases to the shared window, without parsing a single tag — and why a change to
how a chart looks cannot quietly change what it shows.

To add a chart kind: a dataclass in `specs.py`, a branch in `charts.render()`, and one in
`html._block()`. Nothing in `views.py` has to learn about it.

To replace the frontend entirely: reimplement `render/`. Every view keeps working.

**That last claim has been cashed.** `render/markdown.py` renders the same `Report`
objects as text and not one line of any view changed to allow it. It earns its place
three ways: a summary that pastes into an email or an issue, a format still readable when
nothing renders today's SVG, and the cheapest regression test the seam can have — a view
that started emitting markup would render fine in HTML and be visibly broken in Markdown.
Charts that have an honest text form (bars, heatmaps — they are small grids of numbers)
are tabulated; line charts are described rather than faked, because an ASCII sparkline
would look like data while being an artefact of column width.

### Why hand-rolled SVG

A report has to be one file that opens with no network and no build step, which rules out
a CDN. That leaves embedding a charting library — hundreds of kilobytes in *every* report —
or emitting SVG. The vocabulary needed is five primitives of simple geometry, so emitting
it costs less than it saves, and it buys exact theme control: every colour is a CSS custom
property, which is how one stylesheet flips the whole report between light and dark.

Reports run 15–100 KB.

---

## 5. Where the data comes from

Reports read the registry, never the engine. Two sources:

| Source | Provides |
|---|---|
| `data/experiments/runs.jsonl` | every logged run and its 54 fields |
| `data/experiments/curves.jsonl` | month-end equity curves, keyed by `run_id` |

Curve storage was added for this (ADR-027) and is on by default. **Month-end, not daily:**
a daily curve is ~30 KB per run, so 10,000 GA individuals would cost ~300 MB; month-end is
~7 KB with all three series, so the same run costs ~75 MB. Since the strategy only trades
at month ends, that is where the information is — nothing a comparison chart shows is lost.

A large search should still pass `curve=False` and re-run its winners afterwards. The
fingerprint is unchanged, so re-running a winner is the same trial, not a new one.

A run with no stored curve appears in the tables and is named in a warning, rather than
being silently dropped from the charts.

---

## 6. From Python

```python
from sp500lab.backtest import registry
from sp500lab.reporting import comparison_report, run_report, write_html

runs = registry.load("my-study")
report = comparison_report(runs, title="My study")
write_html(report, "reports/my-study.html")

# a Report is inspectable before rendering — it is just dataclasses
[s.title for s in report.sections]
```

---

## 7. Things that surprise people

**`cash` wins columns.** Holding nothing genuinely has the lowest volatility, turnover,
cost and drawdown. The highlighting is correct, just uninteresting. It gets no drawdown
panel, because a full-width chart of a flat line at zero is noise.

**Position count is never highlighted.** A concentrated portfolio is not better or worse
than a diversified one, it is a different bet, and colouring one green would be an opinion
the table has no business having.

**Reports are not committed.** `reports/` is gitignored. They rebuild from the registry in
under a second, and a 100 KB HTML file per run would bloat the repo. The registry is the
thing that is backed up.

**A stale report does not warn you.** It is a snapshot. The footer carries the generation
time; rebuild after new runs.

---

## The forward-test set

`report forward` is the one command here that writes a **directory** rather than a file,
because the forward result is a set of paired comparisons rather than one run:

```
reports/forward_tests/
    index.html              the executive summary
    forward-<name>.html     one technical report per candidate
    decay-analysis.html     did the research ranking predict the forward ranking?
    honesty.html            what limits all of it
    EXECUTIVE_SUMMARY.md    the same pages as text
    DECAY_ANALYSIS.md
    HONESTY.md
    markdown/*.md
    data/*.csv              records, curves and seals — the numbers, separable
    README.md               a guide to the folder, generated from what is in it
```

Two things it does that the other reports do not.

**One caveat is printed on every page and is pinned by a test.** 54 monthly observations
put a ±0.9 band around an annualised Sharpe of 1.0, so a forward test can refute a
strategy and cannot confirm one. `held` therefore means *not refuted*, everywhere.

**The null hypotheses are used as the calibration.** `random_weight` picks its holdings
at random and also "held", which is the clearest available demonstration that a verdict
is a statement about matching a prediction rather than about quality.

Nothing in the path runs a backtest or reads the panel — every figure comes out of
`data/experiments/forward/`. See [FORWARD_TEST.md](FORWARD_TEST.md) and ADR-035.

---

## 8. Related

- `docs/EXPERIMENTS.md` — the registry these reports read, and the deflated Sharpe
- `docs/BACKTEST.md` — the engine the runs come from
- `docs/DECISIONS.md` **ADR-027** (curve storage) and **ADR-028** (static reports)
- `tests/test_reporting.py` — 47 tests, almost all of them on numbers rather than markup

---

## `report trades` — handing over the evidence

```bash
python -m sp500lab report trades momentum_12_1 --open
```

Every other report in this list argues that a number is trustworthy. This one hands over
the orders and invites the reader to disagree.

It is also the one place the "self-contained" rule earns its keep twice over. The trade
CSV is embedded **in the page** as a data URI rather than written beside it, because a
report emailed on its own must still carry its evidence — a link to a file that stayed on
somebody's laptop is worse than no link. `Download` is a block type like any other
(`specs.py`), so the view layer stays pure and the renderer decides how a downloadable
file looks.

Sections: what the strategy did and what it cost, the reconciliation table (does the
ledger add up to the curve?), the download, the most-traded names, and a sample of recent
orders. The reconciliation sits **above** the sample deliberately: "here are the trades"
and "here is the proof these are the trades that produced that curve" are different
claims, and the second is the one worth reading first.

Ledgers beyond `MAX_EMBEDDED_TRADES` (30,000 orders) embed a recent sample and say so in
the caption — base64 inflates a file by a third, and a 90,000-order equal-weight ledger
would make a 20 MB page. Use `sp500lab backtest trades` for the whole thing.

See [TRADES.md](TRADES.md).

---

## `report all` — the set

```bash
python -m sp500lab report all --open
```

One page per strategy, one for the feature layer, one index that links them, and the
registry and honesty pages alongside. Written as a folder, because that is what makes the
set travel: any single file still opens on its own, and the index is the only thing that
knows the others exist.

It runs each strategy under **all three cost settings** (the headline uses `--costs`,
default realistic), records the trade ledger, and computes the benchmark over each
strategy's own window. It also discovers the winner of every genetic-algorithm search in
`data/experiments/evolve/` and reports it beside the hand-written strategies — the engine
cannot tell them apart, and neither should the report set.

### The one distinction the scoreboard has to make

`found by` is either `written` or `evolved, N trials`. A written strategy's Sharpe is one
draw; an evolved one's is the maximum over however many configurations the search
evaluated, and the maximum of N draws is high whether or not there is any signal. Putting
both in one sorted table without saying which is which would be the single most misleading
thing this project could print, so the column exists and the deflated Sharpe sits next to
it.

An earlier version inferred "searched" from the trial count of the study a run was logged
under. That was wrong in the opposite direction: `report all` logs twenty strategies into
one study, so every hand-written strategy came out labelled "searched, 60 trials".
`tests/test_reports_set.py` pins both directions.

### What a strategy page contains, in order

The order is the argument:

1. **What this claims** — the first paragraph of the strategy's own docstring. Taken from
   the source rather than restated, so a strategy whose docstring stops explaining itself
   produces a report that stops explaining it too, which is the correct failure.
2. **Headline** — against the index over exactly these dates.
3. **What happened** — growth of 1.0, the ratio to SPY, drawdown from the DAILY curve, and
   every calendar year.
4. **Was it steady?** — rolling 3-year Sharpe, rolling volatility, monthly heatmap.
5. **What it cost** — all three cost settings, and where the money went.
6. **The orders** — the CSV, the reconciliation, and what it would hold today.
7. **What would make me distrust this** — coverage, forced exits, the deflated Sharpe.

Coverage is in section seven and also in the note under section two, deliberately. A report
that mentions on page four that half the index was untradable has already misled its reader.

### `report features`

What all 75 features are, family by family, with the reading (which end is historically
good) and how often each is actually populated. The leakage check goes **second**,
immediately after the overview and before a single feature is listed: a catalogue of 75
columns is a directory listing until somebody has established that none of them read the
future.

The catalogue itself lives in `features/catalog.py` as data, not prose, and
`tests/test_reports_set.py::test_every_feature_in_the_built_panel_is_documented` fails if a
feature exists without an entry. That is the only way a page like this stays true.

### Page weight

Measured on `low_vol`: 4.00 MB total, of which the embedded trade ledger is 3.65 MB and
every chart on the page together is 0.31 MB. So `MAX_EMBEDDED_TRADES` and
`MAX_EMBEDDED_BYTES` in `views.py`, not the charting, decide how large these files are.
Beyond the cap a page embeds its most recent orders and names the sibling CSV that holds
all of them — `report all` always writes the complete ledger to `reports/trades/`,
whatever the page chose to inline.
