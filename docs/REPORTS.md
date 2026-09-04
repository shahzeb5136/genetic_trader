# Reports

Static HTML over the experiment registry and the forward store. One self-contained file
per page — no server, no CDN, no build step. Open it offline, keep it next to the commit
that produced it, send it to someone.

---

## 1. One folder per lab, and what is in each

```bash
python -m sp500lab report backtest --open
```

```bash
python -m sp500lab report forward --open
```

```bash
python -m sp500lab report timing --open
```

```bash
python -m sp500lab report genetic --open
```

```
reports/
    backtest/                    the research window, 2007-04 → 2021-12
        index.html               every algorithm's headline statistics; click a name
        <algorithm>.html         one page per algorithm
    forward/                     the reserved period, 2022-01 onward
        index.html               prediction against outcome, every algorithm
        <algorithm>.html         one page per forward-tested algorithm
    timing/                      the calendar lab: WHEN rather than WHICH
        index.html               the decomposition and every rule costed three ways
        <rule>.html              one page per calendar rule, BOTH windows on each
    genetic_algorithm/           the search itself, three pages and no index
        methodology.html         how a search works, and every defence in it
        features.html            what it may read, and what it converged on
        evolved-algorithms.html  every search, its training, and its winner
```

That is the whole layout, and it is deliberate ([ADR-045](DECISIONS.md),
[ADR-046](DECISIONS.md), [ADR-047](DECISIONS.md)). The rule is **one folder per lab, and
a lab splits by window only when it is too big for one page per algorithm.** The monthly
roster is thirty algorithms across two windows, so it splits; the calendar and genetic
labs do not. Every set holds exactly two kinds of file — an index and the pages — and
nothing else is written into one. A rebuild removes pages an earlier build left behind,
so the folder always describes the last build. The indexes link to each other, and a page
is named by its algorithm, so `backtest/low-vol.html` and `forward/low-vol.html` are the
same strategy on either side of the boundary.

**The roster** is the same for both sets and is defined once, in `reporting/queries.py`:

| Who | Where it is decided |
|---|---|
| every built-in strategy: the baselines, the twelve hypotheses, the second wave, the learned models, `evolved_blend` | `GROUPS["all"]` in `strategies/__init__.py` |
| everything in the `custom` group — your own strategies | `GROUPS["custom"]`; `roster()` adds it, the engine's suite does not |
| the winners of the best three genetic-algorithm searches on disk, ranked by the research Sharpe of each search's best logged run | `ga_winners()`, `GA_WINNERS_SHOWN = 3`; `--ga-winners N` changes the count, `--no-evolved` drops them |

### The calendar lab

`report timing` writes an index and one page per calendar rule. These rules are not on
the roster above and are not meant to be: they run on a different engine — one
instrument, all-in or cash, on a schedule known years in advance — and sorting them into
a scoreboard of thirty fully invested stock pickers would rank a rule that sits in cash
80% of the time near the top for a reason that is not skill.

| Page | Answers |
|---|---|
| `index.html` | The leg engine and its two identities, where SPY's return actually happens, every rule under all three cost settings, and the per-ticker overnight/intraday split with the full CSV embedded |
| `<rule>.html` | One rule: what it claims in its own docstring, the schedule it trades, the research window costed three ways with its gross and net curves, and — when it was carried into the holdout — the whole forward test |

**Both windows on one page.** The monthly sets split by window because thirty algorithms
× two windows is too much for one page each. Nine rules is not, and a rule's research and
forward numbers are one story: `tm_weekend` went from a 0.04 Sharpe in research to −0.20
forward and is the only `failed` verdict in the entire forward set. The forward half of a
rule page is built by `forward_views.outcome_sections()` from the same stored record the
forward set reads, so the two cannot drift into disagreeing about a paired comparison.

**`entries` is the column to read first.** It counts the round trips a rule makes over the
research window, off its own leg vectors — the legs walk in time order (intraday, then
overnight, then the next intraday) and the rising edges of that sequence are the entries.
It is the sample size, and it is not the session count: `tm_sell_in_may` is invested
across 1,806 sessions and enters sixteen times. Sixteen is what a Sharpe from it is worth.

This is the one set that runs a backtest. A rule with no research-window row yet gets one,
logged under the study `reports`; the holdout is never touched.

### The genetic-algorithm lab

`report genetic` writes three pages and no index: a fourth page whose only content is
three links is a file, not navigation. Each page carries a link grid to the other two and
back to the backtest scoreboard, which is where an evolved winner sits beside everything
it was competing against.

| Page | Answers |
|---|---|
| `methodology.html` | What the search space is (nine prior-signed families, at most three live), what is being maximised (the worst quarter of twelve random sub-periods, net of pessimistic costs, minus a charge per rule), what a search hands on (an ensemble, not its champion), how the population moves, and the five defences against a search that would otherwise overfit every time |
| `features.html` | The nine families with their stories, members and signs; what was cut and why; the five presets and why they are short and frozen; and which families and features each search actually converged on — the champion's weights beside the share of the final population that agreed |
| `evolved-algorithms.html` | Every search on disk: its settings and objective, its fitness and diversity by generation (one line per seed), its champion decoded into a table, its ensemble — members, what they agree on, its own research and forward record — the deflated Sharpe, and the forward verdict of whatever it handed over |

It runs **no backtest and no search**. Every figure comes from the checkpoints in
`data/experiments/evolve/`, the trial log and the forward store, so the set rebuilds in a
second. A search whose checkpoint is gone is named on the last page rather than dropped:
its trials still count toward the deflated Sharpe of anything logged in the same study.

`report backtest baselines` or `report backtest low_vol,cash` narrows a set to a group or
a list. The forward set takes the same argument and writes a page only for roster members
that have a forward record; it prints the ones that do not. `-o DIR` writes a set
somewhere else (no pruning, no cross-link), and `--open` launches a browser on the index.
`report all` is an alias of `report backtest`.

The forward index still counts every candidate that was looked at, including the calendar
rules it does not show, and names them under *Not everything tested is shown* — with a
link to `timing/index.html`, since ADR-047, so the pointer goes somewhere.

### Everything else is on demand

The pages that used to sit beside the set still exist as commands. They land in
`reports/extra/` so the two sets stay clean, and `-o path.html` puts one anywhere.

| Command | Answers |
|---|---|
| `report strategy NAME` | one strategy in full, from a fresh run |
| `report features` | what every feature is, and whether any of it reads the future |
| `report algorithms` | the Algorithm Book: every competitor explained in its own words and scored on one page |
| `report registry` | everything tried, with the deflated Sharpe per study |
| `report honesty` | coverage, forced exits, dirty trees and the holdout ledger, across every run |
| `report study NAME` | every run in one study, side by side |
| `report run ID` | one logged run, from the registry |
| `report compare A B ...` | specific runs or strategies side by side |
| `report trades NAME` | a strategy's orders, embedded as a downloadable CSV |

The forward decay analysis and the forward honesty page have no command of their own:
`forward_decay_report()` and `forward_honesty_report()` in `reporting/forward_views.py`
still build them from the same records (§6).

Trade ledgers as files are the engine's business, not the reports': `sp500lab backtest
trades NAME` writes `results/trades/NAME/trades.csv` plus holdings and forced exits. Every
algorithm page still embeds its ledger as a download, up to `MAX_EMBEDDED_TRADES`, and
names that command when the ledger is too large to embed whole.

---

## 2. What a page shows

**The backtest index** — one scoreboard row per algorithm: CAGR, volatility, Sharpe, max
drawdown, Sharpe against the index over the algorithm's *own* window, and the deflated
Sharpe where a search produced it, sorted on the index-relative column. Then a card per
algorithm, every curve on one chart, and the honest summary: how many beat the index, and
which of them were evolved rather than written.

**A backtest page** — the order is the argument:

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

Coverage is in section seven and also in the note under section two, deliberately. A
report that mentions on page four that half the index was untradable has already misled
its reader.

**The forward index** — verdict counts, the paired scoreboard (research against forward,
the gap and its standard error), the research-versus-forward scatter, every forward
curve, and a card per algorithm. Two notes are pinned by tests: 54 monthly observations
put a ±0.9 band around a Sharpe of 1.0, so a forward test can refute and cannot confirm;
and `random_weight` also "held", which is the clearest demonstration that a verdict is
about matching a prediction rather than about quality.

**A forward page** — the claim and what happened to it, prediction against outcome in a
paired table, the curve (and the spliced research-plus-forward curve), every year, whether
the change is real, all nine checks, all three cost settings, provenance, and the forward
orders.

**The genetic pages** — the search space read off `alpha_genome` rather than restated, the
families and the cut list read off `strategies/genome.py`, the objective and every penalty
with the reason each one exists, the ensemble rule, the operators and what each defends
against, and then per search: what it was told to do, how its fitness and diversity moved
generation by generation and seed by seed, the champion as a table of families and
weighted features, the ensemble it hands on with what its members agree on, the
deflated-Sharpe panel, and the forward outcome of the deliverable.

### Charts are interactive, within the page

- **Click a legend entry** to hide that series — on every chart in the report at once, so
  isolating one strategy on the equity chart also isolates it on drawdown.
- **Click a column header** to sort. Sorting is on the underlying number, never the
  rendered text — otherwise `9.84%` sorts above `11.10%`.
- **Hover** a bar, dot or heatmap cell for its exact value.

Light and dark both work from the same file, following the reader's system setting.

---

## 3. Things the reports deliberately keep saying

**The caveats travel with the numbers.** Every report carries an honesty section, and the
deflation panel sits next to the scoreboard rather than behind a link. A report that shows
a 12% CAGR and mentions on page four that half the index was untradable has already misled
its reader.

**Drawdown charts from month-end curves are shallower than the truth.** An intra-month
trough is invisible to them. The `maxDD` column comes from the daily curve and is the
number to quote; a backtest page draws its drawdown from the daily curve and says so, and
every month-end drawdown chart says so in its caption rather than leaving a reader to
assume.

### The one distinction the scoreboard has to make

`found by` is either `written` or `evolved, N trials`. A written strategy's Sharpe is one
draw; an evolved one's is the maximum over however many configurations the search
evaluated, and the maximum of N draws is high whether or not there is any signal. Putting
both in one sorted table without saying which is which would be the single most misleading
thing this project could print, so the column exists and the deflated Sharpe sits next to
it.

An earlier version inferred "searched" from the trial count of the study a run was logged
under. That was wrong in the opposite direction: `report backtest` logs every strategy into
one study, so every hand-written strategy came out labelled "searched, 60 trials".
`tests/test_reports_set.py` pins both directions.

### An evolved result never appears without its trial count

A searched Sharpe is the maximum over every configuration the search evaluated, so the
genetic pages carry the trial count and the deflated Sharpe next to every winner's result
and the forward verdict wherever one exists. A page showing a 21% CAGR without saying it
was the best of 1,407 tries would be the most misleading thing this project could
publish.

The deflated Sharpe is a **probability**, so it is never printed as `1.000`. A value at or
above 0.9995 renders as `>0.999`: nothing on 136 monthly observations supports a claim of
certainty, and three decimal places of rounding should not manufacture one.
`test_a_deflated_sharpe_is_never_rendered_as_one` pins it.

### What the forward index counts, and what it shows

The forward store holds more than the roster — the calendar rules were forward-tested too.
The index shows the roster and *names* the rest, because the multiple-testing bar ("the
luckiest of N worthless candidates would have posted...") is computed over every candidate
that was looked at, whether or not it has a page. Hiding a look would understate N.
`test_the_index_names_tested_candidates_it_does_not_show` holds it.

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
    queries.py   what the pages need, as plain data — and the roster both monthly
                 sets share
    views.py     composes registry data into a Report.      PURE
    forward_views.py
                 the same, over the forward-test store.     PURE
    genetic_views.py
                 the same, over the search checkpoints.     PURE
    timing_views.py
                 the same, over the calendar lab. Its       PURE
                 forward half comes from forward_views.
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
objects as text and not one line of any view changed to allow it. The report sets no
longer write Markdown copies, but the backend stays: it is the cheapest regression test
the seam can have — a view that started emitting markup would render fine in HTML and be
visibly broken in Markdown — and `tests/test_forward_reports.py` runs every forward view
through it. Charts that have an honest text form (bars, heatmaps — they are small grids of
numbers) are tabulated; line charts are described rather than faked, because an ASCII
sparkline would look like data while being an artefact of column width.

### Why hand-rolled SVG

A report has to be one file that opens with no network and no build step, which rules out
a CDN. That leaves embedding a charting library — hundreds of kilobytes in *every* report —
or emitting SVG. The vocabulary needed is five primitives of simple geometry, so emitting
it costs less than it saves, and it buys exact theme control: every colour is a CSS custom
property, which is how one stylesheet flips the whole report between light and dark.

### Page weight

Measured on `low_vol`: 4.00 MB total, of which the embedded trade ledger is 3.65 MB and
every chart on the page together is 0.31 MB. So `MAX_EMBEDDED_TRADES` and
`MAX_EMBEDDED_BYTES` in `views.py`, not the charting, decide how large an algorithm page
is. Beyond the cap a page embeds its most recent orders and names the command that writes
all of them (`sp500lab backtest trades NAME`). Pages without a ledger run 15–150 KB.

---

## 5. Where the data comes from

Reports read the registry and the forward store, never the engine. The backtest set is
the one exception: `report backtest` *runs* each algorithm under all three cost settings
(logged under the study `reports`, the same trial every time) because a page shows the
daily curve, the orders and the diagnostics, which the registry does not keep. Everything
on a forward page comes out of `data/experiments/forward/` with no re-run at all.

| Source | Provides |
|---|---|
| `data/experiments/runs.jsonl` | every logged run and its 54 fields |
| `data/experiments/curves.jsonl` | month-end equity curves, keyed by `run_id` |
| `data/experiments/forward/` | every forward test: both legs, the comparison, the curves, the seals |

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
write_html(report, "reports/extra/my-study.html")

# a Report is inspectable before rendering — it is just dataclasses
[s.title for s in report.sections]
```

The forward pages that have no command:

```python
from sp500lab.forward import store
from sp500lab.reporting import forward_decay_report, forward_honesty_report, write_html

records = store.load()
write_html(forward_decay_report(records), "reports/extra/forward-decay.html")
write_html(forward_honesty_report(records), "reports/extra/forward-honesty.html")
```

---

## 7. Things that surprise people

**`cash` wins columns.** Holding nothing genuinely has the lowest volatility, turnover,
cost and drawdown. The highlighting is correct, just uninteresting. It gets no drawdown
panel, because a full-width chart of a flat line at zero is noise.

**Position count is never highlighted.** A concentrated portfolio is not better or worse
than a diversified one, it is a different bet, and colouring one green would be an opinion
the table has no business having.

**Reports are not committed.** `reports/` is gitignored. They rebuild from the registry
and the forward store, and a multi-megabyte HTML file per algorithm would bloat the repo.
The registry is the thing that is backed up.

**A stale report does not warn you.** It is a snapshot. The footer carries the generation
time; rebuild after new runs.

**A rebuild removes pages.** `report backtest baselines` leaves `reports/backtest/` holding
the index and the six baseline pages, and nothing else — the folder describes the last
build, not the union of every build. Use `-o` for a set you want to keep.

**`custom` is on the index.** The engine's suite leaves `GROUPS["custom"]` off the
scoreboard until a strategy is promoted; the report set includes it by default, because
that is where an idea of your own is meant to be read next to the rest.

---

## 8. Related

- `docs/EXPERIMENTS.md` — the registry these reports read, and the deflated Sharpe
- `docs/FORWARD_TEST.md` — the forward store, and what 54 months can prove
- `docs/EVOLUTION.md` and `docs/HOW_THE_GA_WORKS.md` — the search the genetic pages describe
- `docs/BACKTEST.md` — the engine the runs come from
- `docs/TRADES.md` — the ledger every page embeds, and the audit that ties it to the curve
- `docs/DECISIONS.md` **ADR-027** (curve storage), **ADR-028** (static reports),
  **ADR-035** (the forward views), **ADR-045** (two sets and nothing else),
  **ADR-046** (the genetic-algorithm lab)
- `tests/test_reporting.py`, `tests/test_reports_set.py`, `tests/test_forward_reports.py`,
  `tests/test_genetic_reports.py` — almost all of them on numbers rather than markup
