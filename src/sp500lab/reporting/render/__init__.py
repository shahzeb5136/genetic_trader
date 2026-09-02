"""Rendering: specs in, strings out. The only place that computes pixels or emits tags.

    charts.py    chart spec -> inline SVG
    html.py      Report     -> one self-contained HTML document
    markdown.py  Report     -> Markdown, for reading without a browser

Two backends over one set of specs is the ADR-028 claim being cashed: `views.py` did not
change to make Markdown work, because it never emitted markup in the first place.

Nothing above this package imports it, and nothing in it reads the registry. That
boundary is what lets the presentation layer be replaced without touching a single
number.
"""

from __future__ import annotations

from . import charts, html, markdown

__all__ = ["charts", "html", "markdown"]
