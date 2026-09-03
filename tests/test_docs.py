"""The project map has to stay true, and a diagram is the easiest thing to let rot.

`docs/PROJECT_MAP.md` claims where every part of the project lives. Nothing enforces that
by construction, so these tests do:

  * the fenced Mermaid block and `project-map.mmd` are the same text, because the picture
    is rendered from the file and GitHub renders the fence
  * every `src/sp500lab/...` path the diagram names actually exists
  * the checked-in SVG is portable - no `<foreignObject>`, which a browser refuses to draw
    when an SVG is loaded through an `<img>` tag
  * every relative link in the page resolves

Cheap to run, and they fail the moment somebody moves a module without redrawing the map.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
ROOT = DOCS.parent
MAP_MD = DOCS / "PROJECT_MAP.md"
MAP_MMD = DOCS / "project-map.mmd"
MAP_SVG = DOCS / "project-map.svg"


def _fences(text: str, lang: str = "mermaid") -> list[str]:
    return re.findall(rf"^```{lang}\n(.*?)^```", text, re.S | re.M)


@pytest.fixture(scope="module")
def page() -> str:
    return MAP_MD.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# The source and the fence are one thing
# --------------------------------------------------------------------------

def test_the_map_and_its_rendered_picture_both_exist():
    for path in (MAP_MD, MAP_MMD, MAP_SVG):
        assert path.exists(), f"{path.name} is missing"


def test_the_fenced_diagram_matches_the_file_the_svg_is_rendered_from(page):
    """Otherwise the picture and the fence drift and one of them starts lying."""
    fenced = [f.strip() for f in _fences(page)]
    source = MAP_MMD.read_text(encoding="utf-8").strip()
    assert source in fenced, (
        "the big mermaid fence in PROJECT_MAP.md is not project-map.mmd verbatim; "
        "re-paste it after editing the diagram")


def test_the_diagram_avoids_markup_that_breaks_a_standalone_svg():
    """`htmlLabels: false` renders <b> and <i> literally. Only <br/> works in both modes."""
    source = MAP_MMD.read_text(encoding="utf-8")
    assert not re.search(r"</?(b|i|strong|em)>", source), \
        "use plain text: bold and italic render as literal tags in the SVG"


# --------------------------------------------------------------------------
# The picture is portable
# --------------------------------------------------------------------------

def test_the_svg_carries_no_foreign_objects():
    """A browser will not draw foreignObject inside <img>, so the labels would vanish."""
    svg = MAP_SVG.read_text(encoding="utf-8")
    assert "<foreignObject" not in svg
    assert svg.count("<text") > 50, "the labels should be real SVG text elements"


def test_the_svg_keeps_the_id_its_own_stylesheet_is_scoped_to():
    """Every rule mermaid emits is `#my-svg .cluster rect {...}`. Drop the id and the
    whole diagram renders in default black."""
    svg = MAP_SVG.read_text(encoding="utf-8")
    root = svg[:svg.index(">") + 1]
    assert 'id="my-svg"' in root
    assert "#my-svg" in svg, "the scoped stylesheet is missing"


def test_the_svg_paints_its_own_background():
    """Dark text on a transparent ground is invisible in a dark-mode reader."""
    svg = MAP_SVG.read_text(encoding="utf-8")
    head = svg[:svg.index("</style>") if "</style>" in svg else 2000]
    assert re.search(r'<rect[^>]+fill="#ffffff"', head), \
        "no white background plate under the diagram"


def test_the_svg_is_a_standalone_document():
    svg = MAP_SVG.read_text(encoding="utf-8")
    assert svg.lstrip().startswith("<svg")
    assert 'xmlns="http://www.w3.org/2000/svg"' in svg
    assert "<title>" in svg, "an accessible title for screen readers"


# --------------------------------------------------------------------------
# The map describes a project that actually looks like this
# --------------------------------------------------------------------------

def test_every_code_folder_the_diagram_names_exists():
    """The point of the map is 'where do I find X'. A stale path is the one defect it
    can have while still looking fine."""
    source = MAP_MMD.read_text(encoding="utf-8")
    named = {m for m in re.findall(r"src/sp500lab/(\w+)/", source)}
    assert named, "the diagram names no code folders at all"
    missing = [n for n in sorted(named) if not (ROOT / "src" / "sp500lab" / n).is_dir()]
    assert not missing, f"the diagram names folders that do not exist: {missing}"


def test_every_code_folder_in_the_reference_table_exists(page):
    table = {m for m in re.findall(r"`(src/sp500lab/[\w/]*)`", page)}
    missing = [p for p in sorted(table) if not (ROOT / p).exists()]
    assert not missing, f"the reference table names paths that do not exist: {missing}"


def test_the_module_files_the_diagram_names_exist():
    """Names like `prices_yfinance` and `forward_views` are the reader's search terms."""
    source = MAP_MMD.read_text(encoding="utf-8")
    on_disk = {p.stem for p in (ROOT / "src" / "sp500lab").rglob("*.py")}
    claimed = {
        "wikipedia_sp500", "wikipedia_history", "sec_tickers", "sec_companyfacts",
        "prices_yfinance", "benchmarks", "fred", "fama_french", "http_cache",
        "adjustments", "splits", "catalog", "ranked", "baselines", "alpha", "frontier",
        "learned", "evolvable", "genome", "custom", "engine", "portfolio", "costs",
        "spreads", "delisting", "trades", "metrics", "fitness", "operators", "seal",
        "windows", "compare", "store", "queries", "views", "forward_views",
        "genetic_views", "algorithms_view", "timing_views", "html", "markdown",
    }
    named_in_diagram = {c for c in claimed if c in source}
    assert len(named_in_diagram) > 30, "the diagram stopped naming modules"
    missing = sorted(named_in_diagram - on_disk)
    assert not missing, f"the diagram names modules that no longer exist: {missing}"


def test_the_three_report_folders_are_on_the_map():
    source = MAP_MMD.read_text(encoding="utf-8")
    for folder in ("reports/backtest/", "reports/forward/",
                   "reports/genetic_algorithm/", "reports/extra/"):
        assert folder in source, f"{folder} is missing from the map"


def test_the_irreplaceable_layers_are_marked_as_such(page):
    """Bronze and the experiment ledgers are the two things a mistake cannot undo."""
    for path in ("data/bronze/", "data/experiments/", "data/vault/"):
        row = next((line for line in page.splitlines()
                    if line.startswith(f"| `{path}`")), None)
        assert row is not None, f"{path} is not in the data table"
        assert "**No**" in row, f"{path} must be marked as not rebuildable"


# --------------------------------------------------------------------------
# The links work
# --------------------------------------------------------------------------

def test_every_relative_link_on_the_page_resolves(page):
    targets = re.findall(r"\]\((?!https?://)([^)#]+)(?:#[^)]*)?\)", page)
    missing = sorted({t for t in targets if not (DOCS / t).exists()})
    assert not missing, f"broken relative links in PROJECT_MAP.md: {missing}"


def test_the_page_is_reachable_from_the_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "PROJECT_MAP.md" in readme, "nothing links to the map"
