"""Static HTML reports over the experiment registry.

    python -m sp500lab report study baselines
    python -m sp500lab report run 20260827T182514-586cb2
    python -m sp500lab report registry

One self-contained file per report: no server, no CDN, no build step. Open it offline,
keep it next to the commit that produced it, send it to someone.

Layout, in dependency order
---------------------------
    theme.py    colours and number formatting - the only file that knows what things
                look like
    series.py   equity curves -> plot-ready arrays. Pure.
    tables.py   registry rows -> formatted tables. Pure.
    specs.py    what to draw, never how. The seam between preparation and rendering.
    views.py    composes registry data into a Report of Sections and Blocks. Pure.
    forward_views.py
                the same, for the forward-test store. A sibling rather than more of
                views.py: a forward report asks whether an out-of-sample result matched
                a prediction, which needs different tables and a different voice.
    render/     Report -> SVG and HTML. The only files that compute pixels or emit tags.
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
from .render.html import render as render_html
from .render.html import write as write_html
from .specs import Report, Section
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
    "render_html", "write_html",
]
