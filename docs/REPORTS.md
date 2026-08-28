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

Output lands in `reports/` (gitignored — rebuildable from the registry at any time).
`-o path.html` overrides, `--open` launches a browser.

| Report | Answers |
|---|---|
| `study` | Which of these is better, and by how much |
| `run` | What did this one strategy actually do |
| `registry` | What have I tried, and does the winner survive the search |
| `compare` | Specific runs or strategies side by side |
| `honesty` | What would make me distrust all of the above |

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
    render/
        charts.py  spec -> inline SVG    the only file that computes pixels
        html.py    Report -> one document
    cli.py       `sp500lab report ...`
```

Everything up to and including `views.py` produces **no markup**. That is why
`tests/test_reporting.py` can assert that the shallowest drawdown is highlighted, or that
`align()` rebases to the shared window, without parsing a single tag — and why a change to
how a chart looks cannot quietly change what it shows.

To add a chart kind: a dataclass in `specs.py`, a branch in `charts.render()`, and one in
`html._block()`. Nothing in `views.py` has to learn about it.

To replace the frontend entirely: reimplement `render/`. Every view keeps working.

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

## 8. Related

- `docs/EXPERIMENTS.md` — the registry these reports read, and the deflated Sharpe
- `docs/BACKTEST.md` — the engine the runs come from
- `docs/DECISIONS.md` **ADR-027** (curve storage) and **ADR-028** (static reports)
- `tests/test_reporting.py` — 47 tests, almost all of them on numbers rather than markup
