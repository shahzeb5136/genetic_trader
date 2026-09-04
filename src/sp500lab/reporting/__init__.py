"""Static HTML reports over the experiment registry and the forward store.

    python -m sp500lab report backtest --open     -> reports/backtest/
    python -m sp500lab report forward  --open     -> reports/forward/
    python -m sp500lab report timing   --open     -> reports/timing/
    python -m sp500lab report genetic  --open     -> reports/genetic_algorithm/

One folder per lab (ADR-047). The two monthly sets each hold exactly two kinds of file:
`index.html`, every algorithm's headline statistics on one scoreboard, and one
self-contained page per algorithm - no server, no CDN, no build step. Open a page
offline, keep it next to the commit that produced it, send it to someone.
`reports/timing/` is the calendar lab: an index and one page per calendar rule, each
carrying BOTH windows, because that family is nine rules and a rule's research-to-forward
arc is one story (ADR-047). `reports/genetic_algorithm/` is three pages on the search
itself, no index (ADR-046). Everything else (`report features`, `report registry`,
`report algorithms`, ...) is on demand and lands in `reports/extra/` (ADR-045).

Layout, in dependency order
---------------------------
    theme.py    colours and number formatting - the only file that knows what things
                look like
    series.py   equity curves -> plot-ready arrays. Pure.
    tables.py   registry rows -> formatted tables. Pure.
    specs.py    what to draw, never how. The seam between preparation and rendering.
    queries.py  what the pages need from the registry, the forward store and the
                strategy classes, as plain data - and the roster the two sets share.
    views.py    composes registry data into a Report of Sections and Blocks. Pure.
    forward_views.py
                the same, for the forward-test store. A sibling rather than more of
                views.py: a forward report asks whether an out-of-sample result matched
                a prediction, which needs different tables and a different voice.
    genetic_views.py
                the same, for the genetic algorithm: the search space, the objective,
                the operators, and every search with its winner decoded.
    timing_views.py
                the same, for the calendar lab: the leg decomposition, the rules costed
                three ways, and one page per rule. Its forward half is built by
                `forward_views.outcome_sections()` rather than a second time here.
    render/     Report -> SVG and HTML (and Markdown). The only files that compute
                pixels or emit tags.
    cli.py      `sp500lab report ...`

Everything up to and including `views.py` is testable by asserting on numbers, because
none of it produces markup. That is the point of the split: swap `render/` for a
different backend and every view keeps working, and a change to how a chart looks can
never quietly change what it shows.

See docs/REPORTS.md.
"""

from __future__ import annotations

from .forward_views import (
                            forward_decay_report,
                            forward_honesty_report,
                            forward_index_report,
                            forward_strategy_report,
)
from .genetic_views import features_report as genetic_features_report
from .genetic_views import methodology_report, searches_report
from .render.html import render as render_html
from .render.html import write as write_html
from .specs import Report, Section
from .timing_views import timing_report, timing_rule_report
from .views import (
                            comparison_report,
                            feature_report,
                            honesty_report,
                            index_report,
                            registry_report,
                            run_report,
                            strategy_report,
                            trades_report,
)

__all__ = [
    "Report", "Section",
    "comparison_report", "run_report", "registry_report", "honesty_report",
    "trades_report", "strategy_report", "feature_report", "index_report",
    "forward_index_report", "forward_strategy_report", "forward_decay_report",
    "forward_honesty_report",
    "methodology_report", "genetic_features_report", "searches_report",
    "timing_report", "timing_rule_report",
    "render_html", "write_html",
]
