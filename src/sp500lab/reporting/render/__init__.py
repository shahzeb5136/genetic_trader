"""Rendering: specs in, strings out. The only place that computes pixels or emits tags.

    charts.py   chart spec -> inline SVG
    html.py     Report     -> one self-contained HTML document

Nothing above this package imports it, and nothing in it reads the registry. That
boundary is what lets the presentation layer be replaced without touching a single
number.
"""

from __future__ import annotations

from . import charts, html

__all__ = ["charts", "html"]
