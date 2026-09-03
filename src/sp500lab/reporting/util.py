"""The small shared helpers every view and table needs.

These four functions were previously copy-pasted across `views.py`,
`forward_views.py`, `tables.py`, `theme.py`, `specs.py` and `cli.py` — four
`slugify`s under three names, two `_gt`/`_lt` pairs and two `_finite`s. The
copies had drifted: `views._gt(inf, 0)` returned True while
`forward_views._gt(inf, 0)` returned False, so the same number could be
emphasised on one page and not on another.

The finite-checking behaviour wins, because a non-finite value is never
something a report should highlight as a threshold crossing.
"""

from __future__ import annotations

import math
import time
from typing import Any

__all__ = ["now", "finite", "gt", "lt", "pos", "slugify", "page_href"]


def now() -> str:
    """Stamp for a report footer. UTC, minute resolution."""
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())


def finite(x: Any) -> bool:
    """True only for a real, finite number. NaN, inf, None and text are all False."""
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def gt(x: Any, threshold: float) -> bool:
    """`x > threshold`, False for anything that is not a finite number."""
    return finite(x) and float(x) > threshold


def lt(x: Any, threshold: float) -> bool:
    """`x < threshold`, False for anything that is not a finite number."""
    return finite(x) and float(x) < threshold


def pos(x: Any) -> bool:
    return gt(x, 0.0)


def slugify(text: Any, fallback: str = "report") -> str:
    """Lowercase, alphanumerics kept, everything else collapsed to single dashes.

    Used for both page filenames and section anchors, so it has to be one
    function: a strategy page and the index link to it are generated in
    different modules and must agree character for character.

    `fallback` is what an empty or all-punctuation title becomes. It varies by
    call site only so existing anchors stay stable.
    """
    out = "".join(c.lower() if c.isalnum() else "-" for c in str(text))
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or fallback


def page_href(name: Any) -> str:
    """The file a report set writes for one algorithm: `<slug>.html`, beside its index.

    One function because the index and the page are written by different modules and
    must agree character for character - and because the backtest and forward sets name
    their pages identically, so `backtest/low-vol.html` and `forward/low-vol.html` are
    the same strategy on either side of the boundary (ADR-045).
    """
    return f"{slugify(name)}.html"
