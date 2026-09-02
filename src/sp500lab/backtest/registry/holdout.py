"""The holdout ledger: a period you may look at once, and every look recorded.

Everything from `HOLDOUT_START` onward is reserved for the final test. Backtests default
to `holdout="exclude"` and stop the day before it, so it takes a deliberate act to see
it.

**Every look is recorded, and the record cannot be switched off.** Trial logging can be
disabled for a scratch run (`SP500LAB_REGISTRY=off`); the holdout ledger cannot. That
asymmetry is the point: you are allowed to run something without logging it as a trial,
but you are never allowed to look at the holdout without leaving a trace. Each look
degrades it, and the ledger is the only way to know how degraded it is.

The boundary is a constant here rather than a parameter, because a holdout you can move
is not a holdout. See ADR-025 and docs/EXPERIMENTS.md.

The ledger path lives in `store` and is read through it (`store.HOLDOUT_LOG`) rather
than imported by value, so a test that redirects the ledger into tmp_path redirects it
for this module too. Getting that wrong would mean tests appending to the real,
irreplaceable ledger.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import store

log_ = logging.getLogger(__name__)


#: Everything from this date onward is reserved for the final test. Decided 2026-08-27
#: (ADR-025). ~176 months of research data before it, ~56 months held out - enough to be
#: a real test, and it contains a regime the research window does not (the 2022 drawdown
#: and the rate cycle).
#:
#: Do not move this to make a result look better. Moving it forward silently converts
#: test data into training data, and there is no way to undo having seen it.
HOLDOUT_START = "2022-01-01"

HOLDOUT_MODES = ("exclude", "include", "only")

#: How the caller wanted the holdout treated, in words, for the ledger.
_MODE_MEANING = {
    "exclude": "research window only; stopped the day before the holdout",
    "include": "ran straight through the holdout as if it were ordinary data",
    "only": "ran on the holdout alone - this is the final test",
}

def apply_holdout(start: str, end: str | None, mode: str,
                  panel_end: str) -> tuple[str, str, bool]:
    """Clamp a date range to the holdout policy. Returns (start, end, touched).

    `exclude` moves `end` back to the last session before HOLDOUT_START. `include` and
    `only` both reach into it and both count as a look.
    """
    if mode not in HOLDOUT_MODES:
        raise ValueError(f"holdout must be one of {HOLDOUT_MODES}, got {mode!r}")
    end = end or panel_end

    if mode == "exclude":
        return start, min(end, _day_before(HOLDOUT_START)), False
    if mode == "only":
        return max(start, HOLDOUT_START), end, True
    return start, end, True

def _day_before(date: str) -> str:
    return str((pd.Timestamp(date) - pd.Timedelta(days=1)).date())

def research_end() -> str:
    """The last date a search may see. Everything after it is the holdout.

    A named function rather than callers computing `HOLDOUT_START - 1 day` themselves,
    so that moving the boundary - which nobody should do - would at least move it
    everywhere at once.
    """
    return _day_before(HOLDOUT_START)

def record_holdout_touch(*, strategy: str, study: str | None, mode: str,
                         start: str, end: str, reason: str = "") -> None:
    """Append to the holdout ledger. Deliberately not disableable.

    Called BEFORE the run rather than after. A run that crashes partway may still have
    shown you something, and the conservative reading of "did I look at it" is the only
    one worth keeping.
    """
    store.append_jsonl(store.HOLDOUT_LOG, {
        "at": store.now_iso(),
        "strategy": strategy,
        "study": study,
        "mode": mode,
        "meaning": _MODE_MEANING[mode],
        "start": start,
        "end": end,
        "holdout_start": HOLDOUT_START,
        "reason": reason,
        "git_commit": store.git_state()[0],
    })
    log_.warning(
        "HOLDOUT TOUCHED by %s (%s): %s. This is look #%d and it is recorded in %s. "
        "Each look degrades the holdout.",
        strategy, mode, _MODE_MEANING[mode], holdout_touch_count(), store.HOLDOUT_LOG.name)

def holdout_touches() -> pd.DataFrame:
    """Every recorded look at the holdout, oldest first. Read this before the final test."""
    return pd.DataFrame(list(store.read_jsonl(store.HOLDOUT_LOG)))

def holdout_touch_count() -> int:
    return sum(1 for _ in store.read_jsonl(store.HOLDOUT_LOG))
