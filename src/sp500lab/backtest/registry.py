"""The experiment registry and the holdout guard.

Two jobs, both of which have to exist BEFORE any strategy search starts, because both
are lossy if you add them later.

---------------------------------------------------------------------------
1. The registry: every run is a trial, and trials must be counted
---------------------------------------------------------------------------
A genetic algorithm explicitly optimises the metric you report. So does a person, more
slowly: you write a strategy, look at its Sharpe, change something, look again, and
discard three ideas along the way. Every one of those is a trial, and the best of N
trials has a good Sharpe whether or not there is any signal in the data - that is simply
what a maximum over N draws does.

The deflated Sharpe ratio (Bailey & Lopez de Prado, 2014) corrects for exactly this. It
is already implemented in `metrics.py` and needs two inputs that describe the *search*
rather than the winner:

    n_trials            how many configurations were evaluated
    trial_sharpe_std    how much their Sharpes varied

Neither can be recovered after the fact. **You cannot retroactively log a trial you did
not record**, and a discarded idea leaves no trace anywhere else in the repo. So logging
is on by default and the registry is append-only, in the same spirit as the bronze
manifest: it records what happened, and nothing rewrites it.

Studies
-------
A `study` is a named search. `n_trials` for the deflated Sharpe is the number of
*distinct configurations* in one study, so the study boundary is what decides which
trials the winner has to beat. Choose it to match the search you actually did:

    study="momentum-variants"   twelve hand-written momentum ideas -> N = 12
    study="ga-run-3"            one GA run of 10,000 individuals   -> N = 10,000

Re-running an identical configuration is the same trial, not a new one - the count is
over distinct fingerprints, not over log lines.

Why monthly statistics are stored alongside the daily ones
-----------------------------------------------------------
The headline Sharpe is computed from the daily equity curve. The deflated Sharpe cannot
use it: its derivation assumes approximately independent observations, and the daily
returns of a monthly-rebalanced portfolio are strongly autocorrelated within each holding
period. Using 4,861 daily observations where there are really ~176 independent monthly
ones would overstate the precision of every Sharpe in the registry and make the
deflation far too generous.

So `log()` resamples the equity curve to month ends and stores `sharpe_monthly`,
`skew_monthly`, `kurtosis_monthly` and `n_months` next to the daily figures. `deflate()`
uses only the monthly ones. Read the daily Sharpe; deflate the monthly one.

---------------------------------------------------------------------------
2. The holdout: a period you may look at once
---------------------------------------------------------------------------
Everything from `HOLDOUT_START` onward is reserved for the final test. Backtests default
to `holdout="exclude"` and stop the day before it, so it takes a deliberate act to see
it.

**Every look is recorded, and the record cannot be switched off.** Trial logging can be
disabled for a scratch run (`SP500LAB_REGISTRY=off`); the holdout ledger cannot. That
asymmetry is the point: you are allowed to run something without logging it as a trial,
but you are never allowed to look at the holdout without leaving a trace. Each look
degrades it, and the ledger is the only way to know how degraded it is.

The boundary is a constant here rather than a parameter, because a holdout you can move
is not a holdout.

Everything in this module is documented at length in `docs/EXPERIMENTS.md`, and the two
decisions are ADR-025 (holdout) and ADR-026 (registry).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import pandas as pd

from ..paths import EXPERIMENT_LOG, HOLDOUT_LOG
from . import metrics

if TYPE_CHECKING:  # avoid a circular import at runtime
    from .results import BacktestResult

log_ = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# The holdout boundary
# --------------------------------------------------------------------------

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


def record_holdout_touch(*, strategy: str, study: str | None, mode: str,
                         start: str, end: str, reason: str = "") -> None:
    """Append to the holdout ledger. Deliberately not disableable.

    Called BEFORE the run rather than after. A run that crashes partway may still have
    shown you something, and the conservative reading of "did I look at it" is the only
    one worth keeping.
    """
    _append(HOLDOUT_LOG, {
        "at": _now_iso(),
        "strategy": strategy,
        "study": study,
        "mode": mode,
        "meaning": _MODE_MEANING[mode],
        "start": start,
        "end": end,
        "holdout_start": HOLDOUT_START,
        "reason": reason,
        "git_commit": git_state()[0],
    })
    log_.warning(
        "HOLDOUT TOUCHED by %s (%s): %s. This is look #%d and it is recorded in %s. "
        "Each look degrades the holdout.",
        strategy, mode, _MODE_MEANING[mode], holdout_touch_count(), HOLDOUT_LOG.name)


def holdout_touches() -> pd.DataFrame:
    """Every recorded look at the holdout, oldest first. Read this before the final test."""
    return pd.DataFrame(list(_read(HOLDOUT_LOG)))


def holdout_touch_count() -> int:
    return sum(1 for _ in _read(HOLDOUT_LOG))


# --------------------------------------------------------------------------
# The run record
# --------------------------------------------------------------------------

@dataclass
class RunRecord:
    """One logged backtest. Flat on purpose so the log loads straight into a DataFrame.

    The fields are grouped by what they are for: identity, configuration, the headline
    statistics you read, the monthly statistics the deflated Sharpe needs, the honesty
    diagnostics, and provenance.
    """

    # identity
    run_id: str
    fingerprint: str
    study: str | None
    logged_at: str
    notes: str = ""

    # configuration - everything that determines the result
    strategy: str = ""
    strategy_class: str = ""
    params: dict = field(default_factory=dict)
    construction: dict | None = None
    warmup: int = 0
    start: str = ""
    end: str = ""
    holdout_mode: str = "exclude"
    touched_holdout: bool = False
    n_rebalances: int = 0
    cost_model: str = ""
    initial_capital: float = 0.0
    liquidity_floor: float = 0.0
    seed: int = 0

    # headline statistics, from the DAILY equity curve - what you read
    cagr: float = float("nan")
    total_return: float = float("nan")
    ann_vol: float = float("nan")
    sharpe: float = float("nan")
    sortino: float = float("nan")
    max_drawdown: float = float("nan")
    calmar: float = float("nan")
    hit_rate: float = float("nan")
    ann_turnover: float | None = None
    avg_positions: float | None = None
    cost_drag: float | None = None
    information_ratio: float | None = None
    beta: float | None = None
    alpha: float | None = None

    # MONTHLY statistics - what deflate() uses. See the module docstring.
    n_months: int = 0
    sharpe_monthly: float = float("nan")
    skew_monthly: float = float("nan")
    kurtosis_monthly: float = float("nan")

    # honesty - the diagnostics that change what the headline means
    coverage_min: float = float("nan")
    coverage_median: float = float("nan")
    forced_exits: int = 0
    unresolved_exits: int = 0
    spread_fallback_orders: int = 0
    unfilled_orders: int = 0
    ruined: bool = False

    # costs
    total_cost: float = 0.0
    commission: float = 0.0
    spread_cost: float = 0.0
    traded_notional: float = 0.0
    n_orders: int = 0

    # provenance - what it would take to reproduce this
    git_commit: str = "unknown"
    git_dirty: bool = False
    panel_key: str = ""
    data_fingerprint: str = ""
    runtime_seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

#: Set by the `study()` context manager so a GA loop does not have to thread `study=`
#: through every call.
_DEFAULT_STUDY: str | None = None

#: Study name used when nothing else is given. Named so it is obvious in the log that
#: these were not part of a deliberate search - but they still count as trials, because
#: they were still things you looked at.
ADHOC_STUDY = "adhoc"


@contextmanager
def study(name: str, notes: str = "") -> Iterator[str]:
    """Tag every run inside the block as part of one named search.

        with registry.study("ga-run-3"):
            for genome in population:
                run_backtest(EvolvedBlend(**decode(genome)))

    The study boundary decides what `n_trials` means for the deflated Sharpe, so it
    should match the search you actually ran - not a folder you find convenient.
    """
    global _DEFAULT_STUDY
    previous = _DEFAULT_STUDY
    _DEFAULT_STUDY = name
    log_.info("study %r: runs in this block will be logged as trials of it%s",
              name, f" ({notes})" if notes else "")
    try:
        yield name
    finally:
        _DEFAULT_STUDY = previous


def current_study() -> str | None:
    return _DEFAULT_STUDY


def enabled() -> bool:
    """Whether trial logging is on.

    Disabled by `SP500LAB_REGISTRY=off` (also `0`, `false`, `none`). The test suite sets
    it so unit tests do not pollute a real search.

    Note that this only silences the TRIAL log. The holdout ledger is written
    regardless - see the module docstring.
    """
    return os.environ.get("SP500LAB_REGISTRY", "on").strip().lower() not in (
        "off", "0", "false", "none", "no")


def fingerprint(*, strategy_class: str, params: dict, construction: dict | None,
                start: str, end: str, cost_model: str, initial_capital: float,
                liquidity_floor: float, seed: int) -> str:
    """A stable id for one trial CONFIGURATION.

    Two runs share a fingerprint exactly when they are the same experiment, so
    `count_trials` can count distinct ideas rather than log lines.

    The data version is deliberately excluded. Re-running the same configuration after
    re-ingesting prices is not a second hypothesis - you tried one thing. The data
    version is recorded separately in `data_fingerprint` for reproducibility.
    """
    payload = json.dumps({
        "cls": strategy_class, "params": _jsonable(params),
        "construction": _jsonable(construction),
        "start": start, "end": end, "cost_model": cost_model,
        "capital": round(float(initial_capital), 2),
        "liquidity_floor": float(liquidity_floor), "seed": int(seed),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def log(result: "BacktestResult", *, study: str | None = None, notes: str = "",
        force: bool = False) -> RunRecord | None:
    """Append one backtest to the registry. Returns None if logging is disabled.

    Called automatically by `run_backtest`. Call it by hand only for a result you built
    some other way.
    """
    if not enabled() and not force:
        return None

    cfg = result.config
    detail = cfg.get("strategy_detail", {})
    perf = result.performance
    costs = result.costs
    diag = result.diagnostics
    monthly = monthly_stats(result.equity)
    commit, dirty = git_state()

    rec = RunRecord(
        run_id=_new_run_id(),
        fingerprint=fingerprint(
            strategy_class=detail.get("class", result.strategy),
            params=detail.get("params", {}),
            construction=detail.get("construction"),
            start=cfg.get("start", ""), end=cfg.get("end", ""),
            cost_model=cfg.get("cost_model", ""),
            initial_capital=cfg.get("initial_capital", 0.0),
            liquidity_floor=cfg.get("liquidity_floor", 0.0),
            seed=cfg.get("seed", 0)),
        study=study or _DEFAULT_STUDY or ADHOC_STUDY,
        logged_at=_now_iso(),
        notes=notes,

        strategy=result.strategy,
        strategy_class=detail.get("class", ""),
        params=_jsonable(detail.get("params", {})),
        construction=_jsonable(detail.get("construction")),
        warmup=int(detail.get("warmup", 0) or 0),
        start=cfg.get("start", ""), end=cfg.get("end", ""),
        holdout_mode=cfg.get("holdout_mode", "exclude"),
        touched_holdout=bool(cfg.get("touched_holdout", False)),
        n_rebalances=int(cfg.get("n_rebalances", 0)),
        cost_model=cfg.get("cost_model", ""),
        initial_capital=float(cfg.get("initial_capital", 0.0)),
        liquidity_floor=float(cfg.get("liquidity_floor", 0.0)),
        seed=int(cfg.get("seed", 0)),

        cagr=perf.cagr, total_return=perf.total_return, ann_vol=perf.ann_vol,
        sharpe=perf.sharpe, sortino=_finite(perf.sortino),
        max_drawdown=perf.max_drawdown, calmar=_finite(perf.calmar),
        hit_rate=perf.hit_rate, ann_turnover=perf.ann_turnover,
        avg_positions=perf.avg_positions, cost_drag=perf.cost_drag,
        information_ratio=perf.information_ratio, beta=perf.beta, alpha=perf.alpha,

        n_months=monthly["n_months"],
        sharpe_monthly=monthly["sharpe"],
        skew_monthly=monthly["skew"],
        kurtosis_monthly=monthly["kurtosis"],

        coverage_min=_coverage(diag, "min"), coverage_median=_coverage(diag, "median"),
        forced_exits=_int_from_diag(diag.get("forced_exits")),
        unresolved_exits=_int_from_diag(diag.get("!! unresolved_exits")),
        spread_fallback_orders=int(diag.get("!! spread_fallback_orders", 0) or 0),
        unfilled_orders=_int_from_diag(diag.get("unfilled_orders")),
        ruined="!! ruined" in diag,

        total_cost=costs.total, commission=costs.commission, spread_cost=costs.spread,
        traded_notional=costs.traded_notional, n_orders=costs.n_orders,

        git_commit=commit, git_dirty=dirty,
        panel_key=str(cfg.get("panel", {}).get("start", "")) + ".."
                  + str(cfg.get("panel", {}).get("end", "")),
        data_fingerprint=_data_fingerprint(cfg.get("panel", {})),
        runtime_seconds=float(diag.get("runtime_seconds", 0.0) or 0.0),
    )
    _append(EXPERIMENT_LOG, rec.as_dict())
    return rec


def monthly_stats(equity: pd.Series) -> dict:
    """Sharpe, skew and kurtosis of MONTH-END returns.

    The deflated Sharpe assumes approximately independent observations. Daily returns of
    a monthly-rebalanced portfolio are not - within a holding period the share vector is
    constant, so the daily series carries far fewer independent observations than it has
    rows. Deflating on ~4,900 daily points instead of ~176 monthly ones would make every
    result look far more significant than it is.
    """
    eq = equity.dropna().astype(float)
    if len(eq) < 40:
        return {"n_months": 0, "sharpe": float("nan"),
                "skew": float("nan"), "kurtosis": float("nan")}
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(eq.index))))
    m = pd.Series(eq.to_numpy(), index=idx).resample("ME").last().dropna()
    if len(m) < 6:
        return {"n_months": len(m), "sharpe": float("nan"),
                "skew": float("nan"), "kurtosis": float("nan")}
    m.index = m.index.strftime("%Y-%m-%d")
    try:
        p = metrics.compute(m)
    except Exception:  # noqa: BLE001 - a degenerate curve is not worth crashing over
        return {"n_months": len(m), "sharpe": float("nan"),
                "skew": float("nan"), "kurtosis": float("nan")}
    return {"n_months": p.n_periods, "sharpe": p.sharpe,
            "skew": p.skew, "kurtosis": p.kurtosis}


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def load(study: str | None = None) -> pd.DataFrame:
    """Every logged run, newest last. Filter to one study by name."""
    rows = list(_read(EXPERIMENT_LOG))
    if not rows:
        return pd.DataFrame(columns=[f.name for f in RunRecord.__dataclass_fields__.values()])
    df = pd.DataFrame(rows)
    if study is not None:
        df = df[df["study"] == study].reset_index(drop=True)
    return df


def studies() -> pd.DataFrame:
    """One row per study: runs, distinct trials, best Sharpe, holdout exposure.

    `trials` is the number the deflated Sharpe uses, and it is distinct fingerprints -
    not `runs`. The gap between the two columns is re-run work.
    """
    df = load()
    if df.empty:
        return pd.DataFrame(columns=["study", "runs", "trials", "best_sharpe",
                                     "best_strategy", "touched_holdout", "first", "last"])
    out = []
    for name, g in df.groupby("study"):
        best = g.loc[g["sharpe"].idxmax()] if g["sharpe"].notna().any() else None
        out.append({
            "study": name,
            "runs": len(g),
            "trials": int(g["fingerprint"].nunique()),
            "best_sharpe": float(g["sharpe"].max()) if best is not None else float("nan"),
            "best_strategy": str(best["strategy"]) if best is not None else "",
            "touched_holdout": int(g["touched_holdout"].sum()),
            "first": str(g["logged_at"].min()),
            "last": str(g["logged_at"].max()),
        })
    return pd.DataFrame(out).sort_values("last", ascending=False).reset_index(drop=True)


def count_trials(study: str) -> int:
    """Distinct configurations evaluated in a study. This is `n_trials` for the DSR.

    Distinct, not total: re-running the same configuration is the same hypothesis
    tested twice, and counting it twice would over-deflate.
    """
    df = load(study)
    return int(df["fingerprint"].nunique()) if len(df) else 0


def trial_sharpe_std(study: str) -> float:
    """Spread of monthly Sharpes across the distinct trials in a study.

    The second input the deflated Sharpe needs. A wide spread means the search had a lot
    of room to get lucky, so the winner has more to prove.
    """
    df = load(study)
    if df.empty:
        return 0.0
    per_trial = df.drop_duplicates("fingerprint")["sharpe_monthly"].dropna()
    return float(per_trial.std(ddof=1)) if len(per_trial) > 1 else 0.0


def get(run_id: str) -> pd.Series | None:
    df = load()
    hit = df[df["run_id"] == run_id] if len(df) else df
    return hit.iloc[0] if len(hit) else None


def best(study: str, by: str = "sharpe") -> pd.Series | None:
    """The winner of a study, by whichever column you name."""
    df = load(study)
    if df.empty or df[by].isna().all():
        return None
    return df.loc[df[by].idxmax()]


# --------------------------------------------------------------------------
# Deflation - the reason the registry exists
# --------------------------------------------------------------------------

def deflate(run: "pd.Series | str", study: str | None = None) -> dict:
    """Deflated Sharpe for one run, given the search it came from.

    Accepts a run_id or a row from `load()`. `study` defaults to the run's own study,
    which is almost always what you want - the winner has to beat the search that
    produced it, not some other search.

    Returns the DSR, the PSR against zero, the bar the search set (the Sharpe the
    luckiest of N worthless strategies would have posted), and the inputs, so the number
    can be argued with rather than just quoted.

    Read the result as a probability, not a score. Below ~0.95 the strategy is not
    distinguishable from the best of N lucky draws, however good its raw Sharpe looks.
    """
    row = get(run) if isinstance(run, str) else run
    if row is None:
        raise KeyError(f"no run with id {run!r}")
    study = study or row["study"]

    n_trials = count_trials(study)
    spread = trial_sharpe_std(study)
    sr_m = float(row["sharpe_monthly"])
    n_months = int(row["n_months"])

    if not np.isfinite(sr_m) or n_months < 6:
        return {"run_id": row["run_id"], "study": study, "deflated_sharpe": float("nan"),
                "why": "too few monthly observations to deflate"}

    # Per-observation, non-annualised: what the DSR formula expects.
    sr_obs = sr_m / np.sqrt(12.0)
    full_kurt = float(row["kurtosis_monthly"]) + 3.0
    bar = metrics.expected_max_sharpe(n_trials, spread / np.sqrt(12.0))

    return {
        "run_id": row["run_id"],
        "strategy": row["strategy"],
        "study": study,
        "n_trials": n_trials,
        "trial_sharpe_std": round(spread, 4),
        "n_months": n_months,
        "sharpe_annualised_daily": round(float(row["sharpe"]), 4),
        "sharpe_annualised_monthly": round(sr_m, 4),
        "expected_max_sharpe_annualised": round(bar * np.sqrt(12.0), 4),
        "psr_vs_zero": round(metrics.probabilistic_sharpe(
            sr_obs, n_months, float(row["skew_monthly"]), full_kurt), 4),
        "deflated_sharpe": round(metrics.deflated_sharpe(
            sr_obs, n_months, float(row["skew_monthly"]), full_kurt,
            n_trials, spread / np.sqrt(12.0)), 4),
    }


def deflate_best(study: str) -> dict:
    """Deflate the best run in a study. The usual thing you want after a search."""
    row = best(study)
    if row is None:
        raise KeyError(f"study {study!r} has no runs")
    return deflate(row, study)


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------

def _new_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append(path, record: dict) -> None:
    """Append one JSON line, healing a truncated previous write first.

    A process killed mid-write leaves a partial final line with no newline. Appending
    straight onto it would concatenate the two records and destroy the good one as well
    as the broken one - the same shape of silent corruption as the chunk-cache bug in
    ADR-013, where the artifact passed every integrity check and was still wrong.
    Checking the last byte costs one seek and bounds the damage to the record that was
    already lost.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            truncated = fh.read(1) != b"\n"
        if truncated:
            log_.warning("%s ended mid-line - a previous write was interrupted. "
                         "Starting a new line so this record is not lost too.", path.name)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("\n")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _read(path) -> Iterator[dict]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            # A truncated final line from an interrupted write. Skip it rather than
            # making the whole log unreadable - append-only means the rest is intact.
            log_.warning("skipping unparseable line in %s", path.name)


def git_state() -> tuple[str, bool]:
    """(commit, dirty). A result from a dirty tree is not reproducible; say so."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=5).stdout.strip() or "unknown"
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True, timeout=5).stdout.strip()
        return commit, bool(status)
    except Exception:  # noqa: BLE001
        return "unknown", False


def _coverage(diag: dict, which: str) -> float:
    """Pull a percentage out of the human-readable coverage diagnostic string."""
    text = str(diag.get("price_coverage", ""))
    import re
    if which == "min":
        m = re.search(r"([\d.]+)% worst", text)
    else:
        m = re.search(r"([\d.]+)% median", text)
    return float(m.group(1)) / 100.0 if m else float("nan")


def _int_from_diag(value: Any) -> int:
    """Diagnostics are strings for humans; pull the leading count back out."""
    if value is None:
        return 0
    if isinstance(value, (int, np.integer)):
        return int(value)
    import re
    m = re.match(r"\s*(\d+)", str(value))
    return int(m.group(1)) if m else 0


def _data_fingerprint(panel_meta: dict) -> str:
    """Short hash of the dataset the run saw, for reproducibility (not for trial counts)."""
    if not panel_meta:
        return ""
    payload = json.dumps({k: panel_meta.get(k) for k in
                          ("n_dates", "n_securities", "n_bars", "end",
                           "format_version")}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _finite(x: float | None) -> float:
    """inf is a real answer (a Calmar with no drawdown) but it does not survive JSON."""
    if x is None:
        return float("nan")
    return float(x) if np.isfinite(x) else float("nan")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)
