"""Chart *specifications* — what to draw, never how it looks.

This module is the seam between preparation and rendering, and it is why `views.py` can
be tested without asserting on a single tag of markup.

A view builds a `LineChart(series=[...], y_format="pct")`. The renderer decides that a
percentage axis gets one decimal place, that the benchmark is grey and dashed, and how
wide the left margin needs to be. Swap `render/` for a different backend — matplotlib,
Vega, a React component — and every view keeps working unchanged, because none of them
ever said anything about pixels.

The blocks below are the page vocabulary: a report is sections, a section is blocks, a
block is one of these. Adding a new kind of visual means adding a dataclass here and a
branch in the renderer, and nothing in `views.py` has to learn about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from .series import LineSeries, Point
from .tables import Table

#: How a renderer should format an axis or a value. Named rather than a format string so
#: the renderer keeps control of precision (see theme.py).
Format = str        # "pct" | "num" | "money" | "multiple" | "bp" | "count"


@dataclass
class LineChart:
    """One or more lines on a shared date axis."""

    series: list[LineSeries]
    title: str = ""
    subtitle: str = ""
    y_format: Format = "num"
    log_y: bool = False
    zero_line: bool = False
    height: int = 300
    legend: bool = True
    caption: str = ""


@dataclass
class AreaChart:
    """A single series filled to a baseline. Used for drawdown, which reads as depth."""

    series: LineSeries
    title: str = ""
    y_format: Format = "pct"
    baseline: float = 0.0
    invert_fill: bool = True      # fill downward: drawdown is a hole, not a hill
    height: int = 180
    caption: str = ""


@dataclass
class BarChart:
    labels: list[str]
    values: list[float]
    title: str = ""
    y_format: Format = "pct"
    diverging: bool = True        # colour negatives differently from positives
    height: int = 220
    caption: str = ""


@dataclass
class ScatterChart:
    points: list[Point]
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    x_format: Format = "pct"
    y_format: Format = "pct"
    height: int = 320
    caption: str = ""


@dataclass
class Heatmap:
    """A year x month grid. `values[row][col]`, None for a missing cell."""

    rows: list[str]
    cols: list[str]
    values: list[list[float | None]]
    title: str = ""
    value_format: Format = "pct"
    diverging: bool = True
    caption: str = ""
    highlight_last_col: bool = True    # the YTD column is a total, not a month


@dataclass
class Stat:
    """One number for a KPI tile."""

    label: str
    value: str
    note: str = ""
    emphasis: str = ""           # "" | "good" | "bad" | "warn"


@dataclass
class StatRow:
    stats: list[Stat]
    title: str = ""


@dataclass
class TableBlock:
    table: Table
    title: str = ""


@dataclass
class Note:
    """Prose. Use it where a number needs a caveat that a caption cannot carry."""

    text: str
    level: str = "info"          # info | warn | danger
    title: str = ""


Block = Union[LineChart, AreaChart, BarChart, ScatterChart, Heatmap,
              StatRow, TableBlock, Note]


@dataclass
class Section:
    title: str
    blocks: list[Block] = field(default_factory=list)
    blurb: str = ""
    anchor: str = ""

    def __post_init__(self) -> None:
        if not self.anchor:
            self.anchor = _slug(self.title)

    def add(self, block: Block | None) -> "Section":
        """Append, ignoring None so a view can skip a block without an if around it."""
        if block is not None:
            self.blocks.append(block)
        return self

    @property
    def empty(self) -> bool:
        return not self.blocks


@dataclass
class Report:
    title: str
    subtitle: str = ""
    sections: list[Section] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    generated_at: str = ""

    def add(self, section: Section | None) -> "Report":
        if section is not None and not section.empty:
            self.sections.append(section)
        return self

    def section(self, title: str) -> Section | None:
        return next((s for s in self.sections if s.title == title), None)


def _slug(text: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "-" for c in text)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or "section"
