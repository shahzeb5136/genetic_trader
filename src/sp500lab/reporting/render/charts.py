"""Chart specs to inline SVG. The only file that computes pixels.

Why hand-rolled rather than matplotlib or a JS library
------------------------------------------------------
A report has to be one file you can open with no network, no server and no build step,
and keep next to a commit. That rules out a CDN. It leaves either embedding a charting
library (hundreds of kilobytes per report, repeated in every file) or emitting SVG
directly.

The chart vocabulary here is small and fixed — five primitives, all of them simple
geometry — so emitting SVG costs less than it saves. It also buys exact control of
theming: every colour is a CSS custom property, so one stylesheet flips the whole report
between light and dark without regenerating anything.

Coordinates
-----------
Charts are drawn in a fixed user-space box and scaled by CSS (`width: 100%`,
`viewBox` set), so they stay sharp at any size and reflow on a phone. The y axis is
inverted in the usual SVG way: `_Scale.py()` maps a data value to a pixel with the
origin at the bottom.

Missing data is a break in the line, not a straight segment across the gap. A gap drawn
as a line is a claim about data that does not exist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .. import theme
from ..series import LineSeries
from ..specs import (AreaChart, BarChart, Heatmap, LineChart, ScatterChart)

#: User-space width. Height comes from the spec. CSS scales both.
WIDTH = 900
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 62, 18, 16, 34

_FORMATTERS = {
    "pct": lambda v: theme.pct(v, 1),
    "pct2": lambda v: theme.pct(v, 2),
    "num": theme.num,
    "money": theme.money,
    "multiple": theme.multiple,
    "bp": theme.bp,
    "count": theme.count,
}


def fmt(kind: str, value: float) -> str:
    return _FORMATTERS.get(kind, theme.num)(value)


# --------------------------------------------------------------------------
# Scales and ticks
# --------------------------------------------------------------------------

@dataclass
class _Scale:
    lo: float
    hi: float
    px0: float
    px1: float
    log: bool = False

    def __post_init__(self) -> None:
        if self.log:
            # A log axis cannot show zero or negatives; clamp to a small positive.
            self.lo = max(self.lo, 1e-9)
            self.hi = max(self.hi, self.lo * 1.0000001)
        if self.hi <= self.lo:
            self.hi = self.lo + 1.0

    def py(self, value: float) -> float:
        if self.log:
            value = max(value, 1e-9)
            t = (math.log(value) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        else:
            t = (value - self.lo) / (self.hi - self.lo)
        return self.px1 - t * (self.px1 - self.px0)


def _nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Round numbers spanning [lo, hi]. Ticks a reader can hold in their head."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(target, 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for mult in (1, 2, 2.5, 5, 10):
        if raw / mag <= mult:
            step = mult * mag
            break
    else:
        step = 10 * mag
    start = math.floor(lo / step) * step
    out, v = [], start
    while v <= hi + step * 0.5 and len(out) < 40:
        if v >= lo - step * 0.5:
            out.append(round(v, 10))
        v += step
    return out or [lo, hi]


def _log_ticks(lo: float, hi: float) -> list[float]:
    """Powers of ten with a 2/5 subdivision, which is how a growth chart is read."""
    if lo <= 0 or hi <= lo:
        return [max(lo, 1e-9)]
    out = []
    e = math.floor(math.log10(lo))
    while 10 ** e <= hi * 1.0001 and len(out) < 40:
        for m in (1, 2, 5):
            v = m * 10 ** e
            if lo * 0.999 <= v <= hi * 1.001:
                out.append(v)
        e += 1
    return out or [lo, hi]


def _bounds(values: list[float], pad: float = 0.06,
            include_zero: bool = False) -> tuple[float, float]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if math.isclose(lo, hi):
        span = abs(lo) or 1.0
        return lo - span * 0.1, hi + span * 0.1
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


# --------------------------------------------------------------------------
# SVG helpers
# --------------------------------------------------------------------------

def _open(height: int, cls: str = "") -> str:
    return (f'<svg class="chart {cls}" viewBox="0 0 {WIDTH} {height}" '
            f'preserveAspectRatio="xMidYMid meet" role="img">')


def _grid_and_axes(scale: _Scale, ticks: list[float], height: int, kind: str,
                   x_labels: list[tuple[float, str]]) -> str:
    parts = []
    for t in ticks:
        y = scale.py(t)
        if not (MARGIN_T - 1 <= y <= height - MARGIN_B + 1):
            continue
        parts.append(f'<line class="grid" x1="{MARGIN_L}" y1="{y:.1f}" '
                     f'x2="{WIDTH - MARGIN_R}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{MARGIN_L - 8}" y="{y + 4:.1f}" '
                     f'text-anchor="end">{_esc(fmt(kind, t))}</text>')
    for x, label in x_labels:
        parts.append(f'<text class="tick" x="{x:.1f}" y="{height - MARGIN_B + 20}" '
                     f'text-anchor="middle">{_esc(label)}</text>')
    return "".join(parts)


def _date_ticks(dates: list[str], px0: float, px1: float,
                target: int = 7) -> list[tuple[float, str]]:
    """Year labels spread across the axis, never more than `target` of them."""
    if not dates:
        return []
    years, seen = [], set()
    for i, d in enumerate(dates):
        y = str(d)[:4]
        if y not in seen:
            seen.add(y)
            years.append((i, y))
    if not years:
        return []
    step = max(1, round(len(years) / target))
    n = max(len(dates) - 1, 1)
    return [(px0 + (i / n) * (px1 - px0), label)
            for k, (i, label) in enumerate(years) if k % step == 0]


def _path(points: list[tuple[float, float] | None]) -> str:
    """Move-to on each break, so a gap in the data is a gap in the line."""
    out, pen_down = [], False
    for p in points:
        if p is None:
            pen_down = False
            continue
        cmd = "L" if pen_down else "M"
        out.append(f"{cmd}{p[0]:.1f},{p[1]:.1f}")
        pen_down = True
    return "".join(out)


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _series_class(s: LineSeries, index: int) -> str:
    if s.kind == "benchmark":
        return "s-benchmark"
    if s.kind == "gross":
        return "s-gross"
    return f"s{index % len(theme.SERIES_COLORS)}"


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def line_chart(spec: LineChart) -> str:
    live = [s for s in spec.series if len(s)]
    if not live:
        return _empty("no data")

    h = spec.height
    px0, px1 = MARGIN_L, WIDTH - MARGIN_R
    py0, py1 = MARGIN_T, h - MARGIN_B

    all_y = [v for s in live for v in s.finite_y]
    lo, hi = _bounds(all_y, include_zero=spec.zero_line and not spec.log_y)
    if spec.log_y:
        pos = [v for v in all_y if v > 0]
        lo, hi = (min(pos) * 0.95, max(pos) * 1.05) if pos else (1.0, 2.0)
    scale = _Scale(lo, hi, py0, py1, log=spec.log_y)
    ticks = _log_ticks(lo, hi) if spec.log_y else _nice_ticks(lo, hi)

    dates = max((s.x for s in live), key=len)
    parts = [_open(h), _grid_and_axes(scale, ticks, h, spec.y_format,
                                      _date_ticks(dates, px0, px1))]

    if spec.zero_line and lo < 0 < hi:
        yz = scale.py(0.0)
        parts.append(f'<line class="zero" x1="{px0}" y1="{yz:.1f}" '
                     f'x2="{px1}" y2="{yz:.1f}"/>')

    for i, s in enumerate(live):
        n = max(len(s) - 1, 1)
        pts: list[tuple[float, float] | None] = []
        for j, v in enumerate(s.y):
            if v is None or not math.isfinite(v):
                pts.append(None)
                continue
            pts.append((px0 + (j / n) * (px1 - px0), scale.py(v)))
        cls = _series_class(s, i)
        parts.append(f'<path class="line {cls}" data-series="{_esc(s.name)}" '
                     f'd="{_path(pts)}" fill="none"/>')

    parts.append("</svg>")
    svg = "".join(parts)
    if spec.legend and len(live) > 1:
        svg += _legend(live)
    return svg


def area_chart(spec: AreaChart) -> str:
    s = spec.series
    if not len(s):
        return _empty("no data")

    h = spec.height
    px0, px1 = MARGIN_L, WIDTH - MARGIN_R
    scale = _Scale(*_bounds(s.finite_y, pad=0.08, include_zero=True),
                   MARGIN_T, h - MARGIN_B)
    ticks = _nice_ticks(scale.lo, scale.hi, target=4)

    n = max(len(s) - 1, 1)
    pts = [(px0 + (j / n) * (px1 - px0), scale.py(v))
           for j, v in enumerate(s.y) if v is not None and math.isfinite(v)]
    if not pts:
        return _empty("no data")

    base = scale.py(spec.baseline)
    fill = (f'M{pts[0][0]:.1f},{base:.1f}'
            + "".join(f"L{x:.1f},{y:.1f}" for x, y in pts)
            + f'L{pts[-1][0]:.1f},{base:.1f}Z')

    cls = "dd" if spec.invert_fill else "pos"
    return "".join([
        _open(h),
        _grid_and_axes(scale, ticks, h, spec.y_format, _date_ticks(s.x, px0, px1)),
        f'<path class="area {cls}" d="{fill}"/>',
        f'<path class="line {cls}-line" d="{_path(list(pts))}" fill="none"/>',
        f'<line class="zero" x1="{px0}" y1="{base:.1f}" x2="{px1}" y2="{base:.1f}"/>',
        "</svg>",
    ])


def bar_chart(spec: BarChart) -> str:
    if not spec.values:
        return _empty("no data")
    h = spec.height
    px0, px1 = MARGIN_L, WIDTH - MARGIN_R
    scale = _Scale(*_bounds(spec.values, pad=0.12, include_zero=True),
                   MARGIN_T, h - MARGIN_B)
    ticks = _nice_ticks(scale.lo, scale.hi, target=4)

    n = len(spec.values)
    slot = (px1 - px0) / n
    bw = max(slot * 0.68, 1.0)
    base = scale.py(0.0)

    parts = [_open(h), _grid_and_axes(scale, ticks, h, spec.y_format, [])]
    for i, v in enumerate(spec.values):
        if v is None or not math.isfinite(v):
            continue
        x = px0 + slot * i + (slot - bw) / 2
        y = scale.py(v)
        top, height = (min(y, base), abs(base - y))
        cls = "neg" if (spec.diverging and v < 0) else "pos"
        parts.append(f'<rect class="bar {cls}" x="{x:.1f}" y="{top:.1f}" '
                     f'width="{bw:.1f}" height="{max(height, 0.6):.1f}">'
                     f'<title>{_esc(spec.labels[i])}: '
                     f'{_esc(fmt(spec.y_format, v))}</title></rect>')
    step = max(1, round(n / 12))
    for i, label in enumerate(spec.labels):
        if i % step:
            continue
        parts.append(f'<text class="tick" x="{px0 + slot * i + slot / 2:.1f}" '
                     f'y="{h - MARGIN_B + 20}" text-anchor="middle">{_esc(label)}</text>')
    parts.append(f'<line class="zero" x1="{px0}" y1="{base:.1f}" '
                 f'x2="{px1}" y2="{base:.1f}"/></svg>')
    return "".join(parts)


def scatter_chart(spec: ScatterChart) -> str:
    pts = [p for p in spec.points
           if math.isfinite(p.x) and math.isfinite(p.y)]
    if not pts:
        return _empty("no data")

    h = spec.height
    px0, px1 = MARGIN_L, WIDTH - MARGIN_R
    xs = _Scale(*_bounds([p.x for p in pts], pad=0.15, include_zero=True), px0, px1)
    ys = _Scale(*_bounds([p.y for p in pts], pad=0.15, include_zero=True),
                MARGIN_T, h - MARGIN_B)
    yticks = _nice_ticks(ys.lo, ys.hi, target=4)
    xticks = _nice_ticks(xs.lo, xs.hi, target=5)

    parts = [_open(h)]
    for t in yticks:
        y = ys.py(t)
        parts.append(f'<line class="grid" x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{px0 - 8}" y="{y + 4:.1f}" '
                     f'text-anchor="end">{_esc(fmt(spec.y_format, t))}</text>')
    for t in xticks:
        # _Scale maps low->bottom; for the x axis we want low->left, so mirror it.
        x = px0 + (px1 - xs.py(t))
        if not (px0 - 1 <= x <= px1 + 1):
            continue
        parts.append(f'<text class="tick" x="{x:.1f}" y="{h - MARGIN_B + 20}" '
                     f'text-anchor="middle">{_esc(fmt(spec.x_format, t))}</text>')

    placed = _place_labels([(px0 + (px1 - xs.py(p.x)), ys.py(p.y)) for p in pts])
    for i, p in enumerate(pts):
        x = px0 + (px1 - xs.py(p.x))
        y = ys.py(p.y)
        lx, ly = placed[i]
        cls = "s-benchmark" if p.kind == "benchmark" else f"s{i % len(theme.SERIES_COLORS)}"
        parts.append(f'<circle class="dot {cls}" cx="{x:.1f}" cy="{y:.1f}" r="6">'
                     f'<title>{_esc(p.label)}: {_esc(fmt(spec.x_format, p.x))} vol, '
                     f'{_esc(fmt(spec.y_format, p.y))}</title></circle>')
        if abs(ly - y) > 3:      # nudged: draw a leader so the label stays attached
            parts.append(f'<line class="leader" x1="{x + 7:.1f}" y1="{y:.1f}" '
                         f'x2="{lx - 2:.1f}" y2="{ly - 4:.1f}"/>')
        parts.append(f'<text class="dot-label" x="{lx:.1f}" y="{ly:.1f}">'
                     f'{_esc(p.label)}</text>')

    if spec.x_label:
        parts.append(f'<text class="axis-label" x="{(px0 + px1) / 2:.0f}" '
                     f'y="{h - 2}" text-anchor="middle">{_esc(spec.x_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def heatmap(spec: Heatmap) -> str:
    """A year x month grid. Colour is a diverging ramp centred on zero."""
    if not spec.rows or not spec.cols:
        return _empty("no data")

    flat = [v for row in spec.values for v in row
            if v is not None and math.isfinite(v)]
    if not flat:
        return _empty("no data")
    extent = max(abs(min(flat)), abs(max(flat))) or 1.0

    cell_w, cell_h, label_w = 58, 26, 52
    width = label_w + cell_w * len(spec.cols) + 8
    height = 22 + cell_h * len(spec.rows) + 6

    parts = [f'<svg class="chart heatmap" viewBox="0 0 {width} {height}" '
             f'preserveAspectRatio="xMidYMid meet" role="img">']
    for j, col in enumerate(spec.cols):
        parts.append(f'<text class="tick" x="{label_w + cell_w * j + cell_w / 2:.0f}" '
                     f'y="14" text-anchor="middle">{_esc(col)}</text>')

    for i, row in enumerate(spec.rows):
        y = 22 + cell_h * i
        parts.append(f'<text class="tick" x="{label_w - 8}" y="{y + 17}" '
                     f'text-anchor="end">{_esc(row)}</text>')
        for j, value in enumerate(spec.values[i]):
            x = label_w + cell_w * j
            is_total = spec.highlight_last_col and j == len(spec.cols) - 1
            if value is None or not math.isfinite(value):
                parts.append(f'<rect class="cell empty" x="{x}" y="{y}" '
                             f'width="{cell_w - 2}" height="{cell_h - 2}"/>')
                continue
            alpha = min(abs(value) / extent, 1.0) ** 0.65
            var = "--pos-rgb" if value >= 0 else "--neg-rgb"
            parts.append(
                f'<rect class="cell{" total" if is_total else ""}" x="{x}" y="{y}" '
                f'width="{cell_w - 2}" height="{cell_h - 2}" '
                f'fill="rgba(var({var}), {alpha:.3f})"><title>'
                f'{_esc(row)} {_esc(spec.cols[j])}: '
                f'{_esc(fmt(spec.value_format, value))}</title></rect>')
            parts.append(f'<text class="cell-label" x="{x + (cell_w - 2) / 2:.0f}" '
                         f'y="{y + cell_h / 2 + 4:.0f}" text-anchor="middle">'
                         f'{_esc(theme.pct(value, 1))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _place_labels(anchors: list[tuple[float, float]],
                  min_gap: float = 16.0) -> list[tuple[float, float]]:
    """Nudge scatter labels apart vertically so a cluster stays readable.

    Points that land close together - which is exactly what happens when several
    strategies have similar risk and return, i.e. the interesting case - would otherwise
    print their names on top of each other. Labels are pushed down in y until they clear
    the one above; where a label moves, the caller draws a leader line so it stays
    attached to its dot.
    """
    order = sorted(range(len(anchors)), key=lambda i: (anchors[i][1], anchors[i][0]))
    out = [(0.0, 0.0)] * len(anchors)
    last_y = float("-inf")
    for i in order:
        x, y = anchors[i]
        ly = max(y + 4.0, last_y + min_gap)
        out[i] = (x + 10.0, ly)
        last_y = ly
    return out


def _legend(series: list[LineSeries]) -> str:
    items = []
    for i, s in enumerate(series):
        cls = _series_class(s, i)
        items.append(
            f'<button class="legend-item {cls}" data-series="{_esc(s.name)}" '
            f'type="button"><span class="swatch"></span>{_esc(s.name)}</button>')
    return f'<div class="legend">{"".join(items)}</div>'


def _empty(message: str) -> str:
    return f'<div class="chart-empty">{_esc(message)}</div>'


def render(spec) -> str:
    """Dispatch a spec to its renderer. The only place block kinds are enumerated."""
    if isinstance(spec, LineChart):
        return line_chart(spec)
    if isinstance(spec, AreaChart):
        return area_chart(spec)
    if isinstance(spec, BarChart):
        return bar_chart(spec)
    if isinstance(spec, ScatterChart):
        return scatter_chart(spec)
    if isinstance(spec, Heatmap):
        return heatmap(spec)
    raise TypeError(f"no renderer for {type(spec).__name__}")
