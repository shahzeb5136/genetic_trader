"""Report to a single self-contained HTML file.

No CDN, no build step, no server. Open the file and it works, on a plane, in five years,
next to the commit that produced it. Everything — stylesheet, script, every chart — is
inline.

Interactivity is deliberately small: click a legend entry to isolate or hide a series,
click a column header to sort a table, hover a chart element for its exact value. That is
about a hundred lines of vanilla JavaScript and it covers what a reader actually does
with a backtest report. Anything more wants a real application, which is a different
decision (see docs/REPORTS.md).

Theming is CSS custom properties with a `prefers-color-scheme` override, so the same file
reads correctly in light and dark without regeneration. Chart colours are variables too,
which is why `charts.py` never writes a hex value.
"""

from __future__ import annotations

import base64 as _b64
import html as _html
import time

from ...paths import PROJECT_ROOT
from .. import theme
from ..specs import (AreaChart, BarChart, Download, Heatmap, LineChart, LinkGrid,
                     Note, Report, ScatterChart, Section, StatRow, TableBlock)
from ..tables import Table
from . import charts


def render(report: Report) -> str:
    """The whole document as one string."""
    body = [
        _header(report),
        _toc(report),
        "".join(_section(s) for s in report.sections),
        _footer(report),
    ]
    return _DOCUMENT.format(
        title=_esc(report.title),
        style=_STYLE,
        script=_SCRIPT,
        body="".join(body),
    )


def write(report: Report, path) -> "object":
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(report), encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# Page furniture
# --------------------------------------------------------------------------

def _header(report: Report) -> str:
    meta = "".join(
        f'<div class="meta-item"><span class="k">{_esc(k)}</span>'
        f'<span class="v">{_esc(v)}</span></div>'
        for k, v in report.meta.items())
    sub = f'<p class="subtitle">{_esc(report.subtitle)}</p>' if report.subtitle else ""
    return (f'<header class="page-header"><h1>{_esc(report.title)}</h1>{sub}'
            f'<div class="meta">{meta}</div></header>')


def _toc(report: Report) -> str:
    if len(report.sections) < 3:
        return ""
    links = "".join(f'<a href="#{s.anchor}">{_esc(s.title)}</a>'
                    for s in report.sections)
    return f'<nav class="toc">{links}</nav>'


def _footer(report: Report) -> str:
    when = report.generated_at or time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    return (f'<footer class="page-footer">Generated {_esc(when)} by sp500lab. '
            f'Rebuild with <code>python -m sp500lab report</code> from '
            f'<code>{_esc(PROJECT_ROOT.name)}</code>. '
            f'Every number here comes from the experiment registry — see '
            f'<code>docs/EXPERIMENTS.md</code>.</footer>')


def _section(section: Section) -> str:
    blurb = f'<p class="blurb">{_esc(section.blurb)}</p>' if section.blurb else ""
    blocks = "".join(_block(b) for b in section.blocks)
    return (f'<section id="{_esc(section.anchor)}"><h2>{_esc(section.title)}</h2>'
            f'{blurb}{blocks}</section>')


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------

def _block(block) -> str:
    if isinstance(block, Note):
        return _note(block)
    if isinstance(block, StatRow):
        return _stat_row(block)
    if isinstance(block, Download):
        return _download(block)
    if isinstance(block, LinkGrid):
        return _link_grid(block)
    if isinstance(block, TableBlock):
        return _table_block(block)
    if isinstance(block, (LineChart, AreaChart, BarChart, ScatterChart, Heatmap)):
        return _chart_block(block)
    raise TypeError(f"no HTML for block {type(block).__name__}")


def _chart_block(spec) -> str:
    title = getattr(spec, "title", "")
    subtitle = getattr(spec, "subtitle", "")
    caption = getattr(spec, "caption", "")
    head = f'<h3>{_esc(title)}</h3>' if title else ""
    if subtitle:
        head += f'<p class="chart-subtitle">{_esc(subtitle)}</p>'
    scroll = "scroll-x" if isinstance(spec, Heatmap) else ""
    cap = f'<p class="caption">{_esc(caption)}</p>' if caption else ""
    return (f'<figure class="block chart-block">{head}'
            f'<div class="chart-wrap {scroll}">{charts.render(spec)}</div>{cap}</figure>')


def _stat_row(block: StatRow) -> str:
    head = f'<h3>{_esc(block.title)}</h3>' if block.title else ""
    tiles = "".join(
        f'<div class="stat {_esc(s.emphasis)}">'
        f'<span class="stat-label">{_esc(s.label)}</span>'
        f'<span class="stat-value">{_esc(s.value)}</span>'
        + (f'<span class="stat-note">{_esc(s.note)}</span>' if s.note else "")
        + "</div>"
        for s in block.stats)
    return f'<div class="block">{head}<div class="stats">{tiles}</div></div>'


def _table_block(block: TableBlock) -> str:
    head = f'<h3>{_esc(block.title)}</h3>' if block.title else ""
    return f'<div class="block">{head}{_table(block.table)}</div>'


def _table(table: Table) -> str:
    if table.empty:
        cap = table.caption or "nothing to show"
        return f'<p class="caption empty">{_esc(cap)}</p>'

    sortable = " sortable" if table.sortable else ""
    # Built outside the f-string: a backslash escape inside one is Python 3.12+, and
    # pyproject declares >=3.11. It would have run fine here and failed on a 3.11 box.
    sort_attr = ' data-sort="0"' if table.sortable else ""
    head = "".join(
        '<th class="{}"{}>{}</th>'.format(
            table.aligns[i] if i < len(table.aligns) else "right", sort_attr, _esc(c))
        for i, c in enumerate(table.columns))

    body = []
    for row in table.rows:
        cells = []
        for i, cell in enumerate(row):
            align = table.aligns[i] if i < len(table.aligns) else "right"
            title = f' title="{_esc(cell.title)}"' if cell.title else ""
            key = _esc(str(cell.sort_key))
            body_text = (f'<a href="{_esc(cell.href)}">{_esc(cell.text)}</a>'
                         if cell.href else _esc(cell.text))
            cells.append(f'<td class="{align} {_esc(cell.emphasis)}" '
                         f'data-key="{key}"{title}>{body_text}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")

    cap = f'<p class="caption">{_esc(table.caption)}</p>' if table.caption else ""
    return (f'<div class="table-wrap"><table class="data{sortable}">'
            f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody>'
            f'</table></div>{cap}')


def _link_grid(block: LinkGrid) -> str:
    """Cards linking to sibling reports. Relative hrefs, so the set travels as a folder."""
    cards = []
    for c in block.cards:
        stats = "".join(
            f'<span class="card-stat"><span class="k">{_esc(k)}</span>'
            f'<span class="v">{_esc(v)}</span></span>'
            for k, v in c.stats)
        blurb = f'<p class="card-blurb">{_esc(c.blurb)}</p>' if c.blurb else ""
        cards.append(
            f'<a class="card {_esc(c.emphasis)}" href="{_esc(c.href)}">'
            f'<span class="card-title">{_esc(c.title)}</span>{blurb}'
            f'<span class="card-stats">{stats}</span></a>')
    head = f'<h3>{_esc(block.title)}</h3>' if block.title else ""
    cap = f'<p class="caption">{_esc(block.caption)}</p>' if block.caption else ""
    return f'<div class="block">{head}<div class="cards">{"".join(cards)}</div>{cap}</div>'


def _download(block: Download) -> str:
    """An <a download> whose href is the file itself. No network, no server.

    base64 rather than a percent-encoded plain-text URI: a CSV is full of commas,
    newlines and the occasional quote, and every one of them would need escaping to
    survive an attribute. Encoding once is cheaper than escaping correctly.
    """
    payload = _b64.b64encode(block.content.encode("utf-8")).decode("ascii")
    size = len(block.content) / 1024
    note = f'<span class="dl-note">{_esc(block.note)}</span>' if block.note else ""
    return (f'<div class="download"><a class="dl" download="{_esc(block.filename)}" '
            f'href="data:{_esc(block.mime)};base64,{payload}">{_esc(block.label)}</a>'
            f'<span class="dl-meta">{_esc(block.filename)} &middot; '
            f'{size:,.0f} KB</span>{note}</div>')


def _note(note: Note) -> str:
    title = f'<strong>{_esc(note.title)}</strong> ' if note.title else ""
    return f'<div class="note {_esc(note.level)}">{title}{_esc(note.text)}</div>'


def _esc(text) -> str:
    return _html.escape(str(text), quote=True)


# --------------------------------------------------------------------------
# Assets
# --------------------------------------------------------------------------

_SERIES_CSS = "".join(
    f"  .s{i} {{ --series: {light}; }}\n"
    for i, light in enumerate(theme.SERIES_COLORS))
_SERIES_CSS_DARK = "".join(
    f"    .s{i} {{ --series: {dark}; }}\n"
    for i, dark in enumerate(theme.SERIES_COLORS_DARK))

_STYLE = f"""
:root {{
  --bg: #ffffff; --panel: #fbfbfc; --ink: #1c1f23; --ink-soft: #5b6570;
  --line: #e3e6ea; --grid: #eef1f4; --accent: #4C78A8;
  --pos: {theme.POSITIVE}; --neg: {theme.NEGATIVE}; --warn: {theme.WARNING};
  --muted: {theme.MUTED}; --benchmark: {theme.BENCHMARK_COLOR};
  --pos-rgb: 46,139,87; --neg-rgb: 192,57,43;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}
{_SERIES_CSS}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #14171a; --panel: #1b1f23; --ink: #e8eaed; --ink-soft: #9aa4af;
    --line: #2b3137; --grid: #232830; --accent: #6BA3D0;
    --pos: #46b37e; --neg: #e8635e; --warn: #d8ad3f;
    --pos-rgb: 70,179,126; --neg-rgb: 232,99,94;
  }}
{_SERIES_CSS_DARK}
}}

* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 0 20px 72px; background: var(--bg); color: var(--ink);
  font-family: var(--sans); font-size: 15px; line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}}
main, .page-header, .toc, .page-footer {{ max-width: 1040px; margin: 0 auto; }}

.page-header {{ padding: 40px 0 20px; border-bottom: 1px solid var(--line); }}
h1 {{ font-size: 27px; margin: 0 0 6px; letter-spacing: -0.01em; }}
.subtitle {{ margin: 0 0 14px; color: var(--ink-soft); font-size: 15px; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 8px 22px; margin-top: 10px; }}
.meta-item {{ font-size: 12.5px; }}
.meta-item .k {{ color: var(--ink-soft); margin-right: 6px; }}
.meta-item .v {{ font-family: var(--mono); }}

.toc {{
  display: flex; flex-wrap: wrap; gap: 6px 14px; padding: 14px 0;
  border-bottom: 1px solid var(--line); font-size: 13px;
}}
.toc a {{ color: var(--ink-soft); text-decoration: none; }}
.toc a:hover {{ color: var(--accent); text-decoration: underline; }}

section {{ padding: 34px 0 6px; border-bottom: 1px solid var(--line); }}
section:last-of-type {{ border-bottom: 0; }}
h2 {{ font-size: 20px; margin: 0 0 4px; letter-spacing: -0.01em; }}
h3 {{ font-size: 14.5px; margin: 0 0 8px; font-weight: 600; color: var(--ink); }}
.blurb {{ margin: 0 0 20px; color: var(--ink-soft); max-width: 76ch; }}
.block {{ margin: 0 0 30px; }}
figure.block {{ margin-inline: 0; }}

.chart-wrap {{ background: var(--panel); border: 1px solid var(--line);
               border-radius: 8px; padding: 12px 8px 6px; }}
.chart-wrap.scroll-x {{ overflow-x: auto; }}
svg.chart {{ display: block; width: 100%; height: auto; }}
svg.heatmap {{ min-width: 720px; }}
.chart-subtitle {{ margin: -4px 0 8px; font-size: 12.5px; color: var(--ink-soft); }}
.chart-empty {{ padding: 30px; text-align: center; color: var(--muted); font-size: 13px; }}
.caption {{ margin: 8px 0 0; font-size: 12.5px; color: var(--ink-soft); max-width: 80ch; }}
.caption.empty {{ font-style: italic; }}

.grid {{ stroke: var(--grid); stroke-width: 1; }}
.zero {{ stroke: var(--ink-soft); stroke-width: 1; stroke-dasharray: 3 3; opacity: .55; }}
.tick {{ fill: var(--ink-soft); font-size: 11px; font-family: var(--mono); }}
.axis-label {{ fill: var(--ink-soft); font-size: 11.5px; }}
.line {{ stroke: var(--series, var(--accent)); stroke-width: 1.9;
         stroke-linejoin: round; stroke-linecap: round; }}
.s-benchmark {{ --series: var(--benchmark); }}
.s-benchmark.line {{ stroke-dasharray: 5 4; stroke-width: 1.5; }}
/* Gross-of-costs shadows its own strategy, so it takes the strategy colour at low
   opacity rather than grey - grey is the benchmark, and two grey dashed lines on one
   chart are indistinguishable. */
.s-gross {{ --series: var(--accent); }}
.s-gross.line {{ stroke-dasharray: 1 3; stroke-width: 1.6; opacity: .55; }}
.area.dd {{ fill: var(--neg); opacity: .17; }}
.dd-line {{ stroke: var(--neg); stroke-width: 1.4; }}
.bar.pos {{ fill: var(--pos); opacity: .85; }}
.bar.neg {{ fill: var(--neg); opacity: .85; }}
.bar:hover {{ opacity: 1; }}
.dot {{ fill: var(--series, var(--accent)); stroke: var(--bg); stroke-width: 2; }}
.dot-label {{ fill: var(--ink-soft); font-size: 11.5px; }}
.leader {{ stroke: var(--ink-soft); stroke-width: .8; opacity: .45; }}
.cell {{ stroke: var(--bg); stroke-width: 1; }}
.cell.empty {{ fill: var(--grid); }}
.cell.total {{ stroke: var(--ink-soft); stroke-width: 1.2; }}
.cell-label {{ fill: var(--ink); font-size: 10px; font-family: var(--mono); }}

.legend {{ display: flex; flex-wrap: wrap; gap: 6px 14px; padding: 10px 4px 2px; }}
.legend-item {{
  display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px;
  background: none; border: 0; padding: 2px 4px; cursor: pointer;
  color: var(--ink); font-family: inherit; border-radius: 4px;
}}
.legend-item:hover {{ background: var(--grid); }}
.legend-item .swatch {{ width: 11px; height: 11px; border-radius: 3px;
                        background: var(--series, var(--accent)); }}
.legend-item.off {{ opacity: .35; }}
.legend-item.off .swatch {{ background: var(--muted); }}
path.line.dim {{ opacity: .12; }}

.stats {{ display: grid; gap: 10px;
          grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }}
.stat {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
         padding: 12px 14px; display: flex; flex-direction: column; gap: 2px; }}
.stat-label {{ font-size: 11.5px; color: var(--ink-soft); text-transform: uppercase;
               letter-spacing: .04em; }}
.stat-value {{ font-size: 21px; font-family: var(--mono); letter-spacing: -0.02em; }}
.stat-note {{ font-size: 11.5px; color: var(--ink-soft); }}
.stat.good .stat-value {{ color: var(--pos); }}
.stat.bad .stat-value {{ color: var(--neg); }}
.stat.warn .stat-value {{ color: var(--warn); }}

.table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }}
table.data {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
table.data th, table.data td {{ padding: 7px 12px; white-space: nowrap; }}
table.data thead th {{
  background: var(--panel); border-bottom: 1px solid var(--line);
  font-weight: 600; font-size: 12px; color: var(--ink-soft);
  position: sticky; top: 0;
}}
table.data.sortable thead th {{ cursor: pointer; user-select: none; }}
table.data.sortable thead th:hover {{ color: var(--accent); }}
table.data thead th::after {{ content: ""; }}
table.data thead th.asc::after {{ content: " \\2191"; }}
table.data thead th.desc::after {{ content: " \\2193"; }}
table.data tbody tr:nth-child(even) {{ background: color-mix(in srgb, var(--panel) 55%, transparent); }}
table.data tbody tr:hover {{ background: var(--grid); }}
table.data td {{ font-family: var(--mono); }}
table.data td.left {{ font-family: var(--sans); }}
.left {{ text-align: left; }} .right {{ text-align: right; }} .center {{ text-align: center; }}
td.good {{ color: var(--pos); font-weight: 600; }}
td.bad {{ color: var(--neg); font-weight: 600; }}
td.warn {{ color: var(--warn); }}
td.muted {{ color: var(--muted); }}

.note {{ border-left: 3px solid var(--muted); background: var(--panel);
         padding: 11px 15px; border-radius: 0 6px 6px 0; margin: 0 0 22px;
         font-size: 13.5px; max-width: 88ch; }}
.note.warn {{ border-left-color: var(--warn); }}
.note.danger {{ border-left-color: var(--neg); }}

.download {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px;
  margin: 14px 0; padding: 14px 16px; background: var(--panel);
  border: 1px solid var(--line); border-radius: 6px; }}
a.dl {{ display: inline-block; padding: 7px 14px; border-radius: 5px;
  background: var(--accent); color: #fff; text-decoration: none;
  font-size: 13.5px; font-weight: 600; }}
a.dl:hover {{ filter: brightness(1.08); }}
.dl-meta {{ font-family: var(--mono); font-size: 12px; color: var(--ink-soft); }}
.dl-note {{ font-size: 12.5px; color: var(--ink-soft); flex-basis: 100%; }}

.cards {{ display: grid; gap: 10px; margin: 12px 0;
  grid-template-columns: repeat(auto-fill, minmax(268px, 1fr)); }}
a.card {{ display: block; padding: 14px 16px; border: 1px solid var(--line);
  border-radius: 6px; background: var(--panel); color: var(--ink);
  text-decoration: none; border-left-width: 3px; }}
a.card:hover {{ border-color: var(--accent); border-left-color: var(--accent); }}
a.card.good {{ border-left-color: var(--pos); }}
a.card.bad {{ border-left-color: var(--neg); }}
a.card.warn {{ border-left-color: var(--warn); }}
.card-title {{ display: block; font-weight: 650; font-size: 14.5px;
  font-family: var(--mono); }}
.card-blurb {{ margin: 5px 0 9px; font-size: 12.5px; color: var(--ink-soft);
  line-height: 1.45; }}
.card-stats {{ display: flex; flex-wrap: wrap; gap: 4px 14px; }}
.card-stat .k {{ color: var(--ink-soft); font-size: 11px; margin-right: 5px; }}
.card-stat .v {{ font-family: var(--mono); font-size: 12.5px; }}
table.data td a {{ color: var(--accent); text-decoration: none; }}
table.data td a:hover {{ text-decoration: underline; }}

.page-footer {{ padding: 26px 0 0; color: var(--ink-soft); font-size: 12.5px;
                border-top: 1px solid var(--line); margin-top: 34px; }}
code {{ font-family: var(--mono); font-size: .92em;
        background: var(--grid); padding: 1px 5px; border-radius: 4px; }}

@media print {{
  body {{ padding: 0; }}
  section {{ break-inside: avoid; }}
  .toc, .legend {{ display: none; }}
}}
"""

_SCRIPT = """
// Legend toggles. Clicking a series hides it across every chart that draws it, so
// isolating one strategy on the equity chart also isolates it on drawdown.
document.querySelectorAll('.legend-item').forEach(function (btn) {
  btn.addEventListener('click', function () {
    var name = btn.dataset.series;
    var off = btn.classList.toggle('off');
    document.querySelectorAll('.legend-item[data-series="' + CSS.escape(name) + '"]')
      .forEach(function (b) { b.classList.toggle('off', off); });
    document.querySelectorAll('path.line[data-series="' + CSS.escape(name) + '"]')
      .forEach(function (p) { p.classList.toggle('dim', off); });
  });
});

// Sortable tables. Sorts on data-key (the underlying number), never on the rendered
// text - otherwise "9.84%" sorts above "11.10%".
document.querySelectorAll('table.data.sortable').forEach(function (table) {
  var headers = table.querySelectorAll('thead th');
  headers.forEach(function (th, index) {
    th.addEventListener('click', function () {
      var body = table.tBodies[0];
      var rows = Array.prototype.slice.call(body.rows);
      var desc = !th.classList.contains('desc');
      headers.forEach(function (h) { h.classList.remove('asc', 'desc'); });
      th.classList.add(desc ? 'desc' : 'asc');
      rows.sort(function (a, b) {
        var x = a.cells[index].dataset.key, y = b.cells[index].dataset.key;
        var nx = parseFloat(x), ny = parseFloat(y);
        var cmp = (!isNaN(nx) && !isNaN(ny)) ? nx - ny : String(x).localeCompare(String(y));
        return desc ? -cmp : cmp;
      });
      rows.forEach(function (r) { body.appendChild(r); });
    });
  });
});
"""

_DOCUMENT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{style}</style>
</head>
<body>
<main>
{body}
</main>
<script>{script}</script>
</body>
</html>
"""
