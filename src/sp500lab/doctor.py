"""`sp500lab doctor` - every check that decides whether a number here can be trusted.

One command, one exit code. It exists because the checks live in five places and it is
too easy to run four of them:

    bronze     every artifact re-hashed against the manifest        storage.verify_manifest
    silver     the data-quality battery, strict on ERROR            quality.checks.run
    engine     the acceptance suite: SPY total return, the
               equal-weight identity, the leakage guard,
               determinism, dividends counted once                  backtest.accept.run_all
    legs       the timing engine's two identities                   timing.engine.timing_accept
    features   the feature panel rebuilt with the future deleted    features.check_leakage
    roster     every strategy runs, and none of them changes its
               weights when the panel is truncated (opt-in: slow)   backtest.accept.run_strategy_checks

A stage that raises is a failed stage, not a skipped one. The order is cheapest first,
so a corrupt bronze file is reported before ten minutes of backtests are spent on it.

    python -m sp500lab doctor              # bronze + silver + engine + legs + features
    python -m sp500lab doctor --fast       # skip the bronze re-hash and the feature rebuild
    python -m sp500lab doctor --roster     # also checks 6 and 7 over every strategy
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)


@dataclass
class Stage:
    name: str
    passed: bool
    seconds: float
    lines: list[str] = field(default_factory=list)

    def line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        head = f"[{mark}] {self.name:10s} {self.seconds:6.1f}s"
        return "\n".join([head] + [f"       {ln}" for ln in self.lines])


def _stage(name: str, fn: Callable[[], tuple[bool, list[str]]]) -> Stage:
    t0 = time.perf_counter()
    try:
        ok, lines = fn()
    except Exception as exc:  # noqa: BLE001 - a crash is a failure with a reason
        log.exception("doctor stage %s crashed", name)
        ok, lines = False, [f"crashed: {type(exc).__name__}: {exc}"[:200]]
    return Stage(name, ok, time.perf_counter() - t0, lines)


# ------------------------------------------------------------------- stages

def stage_bronze() -> tuple[bool, list[str]]:
    from .storage import verify_manifest
    v = verify_manifest()
    lines = [f"{v['ok']}/{v['artifacts']} artifacts match their checksums, "
             f"{v['retired']} retired"]
    lines += [f"{f['issue']}: {f['path']}" for f in v["failures"][:5]]
    return v["corrupt"] == 0 and v["missing"] == 0, lines


def stage_silver(verify_bronze: bool) -> tuple[bool, list[str]]:
    from .quality import checks
    rep = checks.run(verify_bronze=verify_bronze)
    counts = checks.summary(rep)
    lines = [f"ERROR={counts['ERROR']}  WARN={counts['WARN']}  INFO={counts['INFO']}"]
    errors = rep[rep["severity"] == checks.ERROR]
    lines += [f"ERROR {r.check}: {r.detail}"[:140] for r in errors.itertuples()]
    return counts["ERROR"] == 0, lines


def stage_engine(panel) -> tuple[bool, list[str]]:
    from .backtest.accept import run_all
    checks = run_all(panel=panel)
    return all(c.passed for c in checks), [c.line() for c in checks]


def stage_legs() -> tuple[bool, list[str]]:
    from .timing.engine import timing_accept
    r = timing_accept()                      # raises on failure
    return True, [
        f"buy&hold {r['buy_hold_cagr'] * 100:.2f}%/yr = overnight "
        f"{r['overnight_cagr'] * 100:.2f}% x intraday {r['intraday_cagr'] * 100:.2f}%",
        f"calibration {r['calibration_bp_per_year']:.2f} bp/yr, decomposition max rel "
        f"err {r['decomposition_max_rel_err']:.1e}",
    ]


def stage_features(panel) -> tuple[bool, list[str]]:
    from .features import check_leakage
    r = check_leakage(panel)
    lines = [f"{r['rows_compared']} rows x {len(r['features'])} features compared at "
             f"{r['cut_at']}"]
    lines += [f"LEAK {name}" for name in r.get("failed", [])[:10]]
    return bool(r["ok"]), lines


def stage_roster(panel, include_learned: bool) -> tuple[bool, list[str]]:
    from .backtest.accept import run_strategy_checks
    checks = run_strategy_checks(panel, include_learned=include_learned)
    failed = [c for c in checks if not c.passed]
    lines = [f"{len(checks) - len(failed)}/{len(checks)} checks passed over "
             f"{len(checks) // 2} strategies"]
    lines += [c.line() for c in failed[:10]]
    return not failed, lines


# ------------------------------------------------------------------- runner

def run(*, fast: bool = False, roster: bool = False, include_learned: bool = False
        ) -> list[Stage]:
    """Run every stage in cost order. Never stops early: a full picture beats a fast one."""
    stages: list[Stage] = []
    if not fast:
        stages.append(_stage("bronze", stage_bronze))
    stages.append(_stage("silver", lambda: stage_silver(verify_bronze=False)))

    panel = None
    try:
        from .backtest.panel import build_panel
        panel = build_panel()
    except Exception as exc:  # noqa: BLE001
        stages.append(Stage("panel", False, 0.0,
                            [f"could not build the panel: {type(exc).__name__}: {exc}"[:200]]))
        return stages

    stages.append(_stage("engine", lambda: stage_engine(panel)))
    stages.append(_stage("legs", stage_legs))
    if not fast:
        stages.append(_stage("features", lambda: stage_features(panel)))
    if roster:
        stages.append(_stage("roster", lambda: stage_roster(panel, include_learned)))
    return stages


def report(stages: list[Stage]) -> str:
    lines = ["=" * 72, "SP500LAB DOCTOR", "=" * 72]
    lines += [s.line() for s in stages]
    failed = [s.name for s in stages if not s.passed]
    lines.append("=" * 72)
    if failed:
        lines.append(f"{len(failed)} stage(s) failed: {', '.join(failed)}. "
                     "Nothing produced by this pipeline should be trusted until they pass.")
    else:
        lines.append("Every stage passed. The data reproduces known quantities, the "
                     "engine refuses the known cheats, and every strategy honours the "
                     "contract." if any(s.name == "roster" for s in stages) else
                     "Every stage passed. Add --roster to check every strategy too.")
    return "\n".join(lines)
