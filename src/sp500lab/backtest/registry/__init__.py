"""The experiment registry: every trial, every curve, every look at the holdout.

Four modules, split by what they are rather than by what calls them:

    stats.py      statistics of one equity curve. Pure, no I/O.
    store.py      the append-only trial log, the curve log, and the study context.
                  Owns the three log paths.
    deflation.py  what the best of N trials is worth (ADR-026).
    holdout.py    the one-shot holdout boundary and its un-disableable ledger (ADR-025).

The public surface below is unchanged from when all four were one 860-line module, so
`registry.log(...)`, `registry.deflate(...)` and `registry.apply_holdout(...)` all still
mean what they meant.

One thing did change, deliberately. The log PATHS are not re-exported here. A test that
redirects the registry into tmp_path must patch `registry.store`:

    monkeypatch.setattr(registry.store, "EXPERIMENT_LOG", tmp_path / "runs.jsonl")

Patching a re-exported copy on this package would leave `store` writing to the real
files while the test believed otherwise - and the trial log and the holdout ledger are
append-only and irreplaceable. Because the name is absent here, an out-of-date patch
raises AttributeError instead of silently appending to the real thing.

Documented at length in docs/EXPERIMENTS.md; the decisions are ADR-025 (holdout),
ADR-026 (registry) and ADR-027 (curves in their own log).
"""

from __future__ import annotations

from .deflation import deflate, deflate_best, trial_sharpe_std
from .holdout import (
                      HOLDOUT_MODES,
                      HOLDOUT_START,
                      apply_holdout,
                      holdout_touch_count,
                      holdout_touches,
                      record_holdout_touch,
                      research_end,
)
from .stats import monthly_stats, to_monthly
from .store import (
                      ADHOC_STUDY,
                      RunRecord,
                      append_jsonl,
                      best,
                      count_trials,
                      current_study,
                      enabled,
                      fingerprint,
                      get,
                      git_state,
                      has_curve,
                      jsonable,
                      load,
                      load_curve,
                      load_curves,
                      log,
                      log_curve,
                      panel_fingerprint,
                      read_jsonl,
                      studies,
                      study,
)

__all__ = [
    # the trial log
    "ADHOC_STUDY", "RunRecord", "current_study", "enabled", "fingerprint", "log",
    "load", "get", "best", "studies", "count_trials", "study",
    # curves
    "log_curve", "load_curve", "load_curves", "has_curve", "to_monthly",
    # statistics and deflation
    "monthly_stats", "trial_sharpe_std", "deflate", "deflate_best",
    # the holdout
    "HOLDOUT_START", "HOLDOUT_MODES", "apply_holdout", "research_end",
    "record_holdout_touch", "holdout_touches", "holdout_touch_count",
    # plumbing others reuse
    "append_jsonl", "read_jsonl", "git_state", "jsonable", "panel_fingerprint",
]
