"""The append-only trial log: what was run, and the curve it produced.

Every backtest is a trial. You will try a momentum rule with a 6-month lookback, then a
12-month one, then discard three ideas along the way - and the best of N trials has a
good Sharpe whether or not there is any signal in the data. **You cannot retroactively
log a trial you did not record**, and a discarded idea leaves no trace anywhere else in
the repo. So logging is on by default and this log is append-only, in the same spirit as
the bronze manifest: it records what happened, and nothing rewrites it. `deflation.py`
is what reads it back.

This module owns the three log paths. Everything else in the package reaches them
through it (`store.HOLDOUT_LOG`, never a from-import), so a test that redirects them
into tmp_path redirects them for the whole package. They are deliberately NOT
re-exported from `registry/__init__.py`: patching a copy on the package would silently
do nothing, and the run would append to the real, irreplaceable log. Patch
`registry.store` instead, and an out-of-date patch fails loudly.

Month-end curves are split into their own log (ADR-027) so the searchable index stays
small. See docs/EXPERIMENTS.md and ADR-026.
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
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import pandas as pd

from ...paths import CURVE_LOG, EXPERIMENT_LOG, HOLDOUT_LOG
from .stats import monthly_stats, to_monthly

if TYPE_CHECKING:  # avoid a circular import at runtime
    from ..results import BacktestResult

log_ = logging.getLogger(__name__)

__all__ = ["CURVE_LOG", "EXPERIMENT_LOG", "HOLDOUT_LOG"]


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

#: The study every run inside a `study(...)` block is tagged with. Module-level state
#: on purpose: it has to reach `log()` without every caller in between passing it down.
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
        "cls": strategy_class, "params": jsonable(params),
        "construction": jsonable(construction),
        "start": start, "end": end, "cost_model": cost_model,
        "capital": round(float(initial_capital), 2),
        "liquidity_floor": float(liquidity_floor), "seed": int(seed),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def log(result: "BacktestResult", *, study: str | None = None, notes: str = "",
        force: bool = False, curve: bool = True) -> RunRecord | None:
    """Append one backtest to the registry. Returns None if logging is disabled.

    Called automatically by `run_backtest`. Call it by hand only for a result you built
    some other way.

    `curve` also stores the month-end equity curve so a report can plot this run later
    without re-running it. See `log_curve` for the sizing argument.
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
        logged_at=now_iso(),
        notes=notes,

        strategy=result.strategy,
        strategy_class=detail.get("class", ""),
        params=jsonable(detail.get("params", {})),
        construction=jsonable(detail.get("construction")),
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
        data_fingerprint=panel_fingerprint(cfg.get("panel", {})),
        runtime_seconds=float(diag.get("runtime_seconds", 0.0) or 0.0),
    )
    append_jsonl(EXPERIMENT_LOG, rec.as_dict())
    if curve:
        log_curve(rec.run_id, result)
    return rec

#: Values are rounded to this many decimals before storage. Curves are rebased to 1.0,
#: so six decimals is sub-basis-point - far finer than anything a chart can show, and it
#: roughly halves the file.
_CURVE_DECIMALS = 6

def log_curve(run_id: str, result: "BacktestResult") -> bool:
    """Store a run's MONTH-END equity curve, rebased to 1.0. Returns False if skipped.

    Why monthly, and why a separate file
    ------------------------------------
    A frontend needs curves; the registry index does not. Keeping them in `runs.jsonl`
    would make every `load()` parse data no query uses, and a GA appends thousands of
    rows.

    Monthly rather than daily is a sizing decision, measured rather than guessed: a daily
    curve is ~30 KB per run, so 10,000 GA individuals would cost ~300 MB. Month-end is
    ~1.4 KB, so the same run costs ~14 MB - and since the strategy only trades at month
    ends, the monthly series is where the information actually is. Nothing a comparison
    chart shows is lost.

    `nav_gross` and `benchmark` are stored when present, so cost drag and relative
    performance can be drawn without re-running anything. A gross curve identical to the
    net one (a zero-cost run) is skipped rather than stored twice.

    Measured cost: ~7 KB per run with all three series, so a 10,000-individual GA run is
    ~76 MB. That is affordable but not free, and `load_curves` parses the whole file. A
    large search should pass `curve=False` and re-run its winners afterwards - the
    fingerprint is unchanged, so re-running a winner is the same trial, not a new one.
    """
    eq = result.equity.dropna().astype(float)
    if len(eq) < 2:
        return False

    payload: dict[str, Any] = {"run_id": run_id, "freq": "M",
                               "strategy": result.strategy,
                               "study": result.config.get("study")}
    gross = result.gross_equity
    if gross is not None:
        aligned = gross.reindex(eq.index).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            same = np.nanmax(np.abs(aligned.to_numpy() / eq.to_numpy() - 1.0)) < 1e-9
        if same:
            gross = None       # a zero-cost run; storing it twice buys nothing

    dates = None
    for name, series in (("nav", result.equity),
                         ("nav_gross", gross),
                         ("benchmark", result.benchmark)):
        if series is None:
            continue
        m = to_monthly(series)
        if m is None or m.empty:
            continue
        if dates is None:
            dates = [str(d) for d in m.index]
            payload["dates"] = dates
        # Reindex onto the nav grid so every column shares one date axis.
        m = m.reindex(dates)
        payload[name] = [None if not np.isfinite(v) else round(float(v), _CURVE_DECIMALS)
                         for v in (m / m.dropna().iloc[0]).to_numpy()]

    if dates is None:
        return False
    append_jsonl(CURVE_LOG, payload)
    return True

def load_curves(run_ids: "list[str] | str | None" = None) -> dict[str, pd.DataFrame]:
    """{run_id: DataFrame} of stored curves, indexed by month end.

    Reads the file once however many runs are asked for, so drawing a ten-strategy
    comparison is a single pass rather than ten.
    """
    if isinstance(run_ids, str):
        run_ids = [run_ids]
    wanted = set(run_ids) if run_ids is not None else None

    out: dict[str, pd.DataFrame] = {}
    for rec in read_jsonl(CURVE_LOG):
        rid = rec.get("run_id")
        if wanted is not None and rid not in wanted:
            continue
        dates = rec.get("dates") or []
        if not dates:
            continue
        cols = {k: rec[k] for k in ("nav", "nav_gross", "benchmark")
                if k in rec and rec[k] is not None}
        if not cols:
            continue
        df = pd.DataFrame(cols, index=pd.Index(dates, name="date"))
        df.attrs["strategy"] = rec.get("strategy", "")
        df.attrs["study"] = rec.get("study")
        out[rid] = df                      # a later write supersedes an earlier one
    return out

def load_curve(run_id: str) -> pd.DataFrame | None:
    return load_curves([run_id]).get(run_id)

def has_curve(run_id: str) -> bool:
    return load_curve(run_id) is not None

def load(study: str | None = None) -> pd.DataFrame:
    """Every logged run, newest last. Filter to one study by name."""
    rows = list(read_jsonl(EXPERIMENT_LOG))
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

def _new_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def append_jsonl(path, record: dict) -> None:
    """Append one JSON line, healing a truncated previous write first.

    Public because the forward-test store (`forward/store.py`) keeps its own
    append-only logs and must not reimplement the healing below - two copies of a
    subtle recovery path is how one of them ends up wrong.

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

def read_jsonl(path) -> Iterator[dict]:
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

@lru_cache(maxsize=1)
def git_state() -> tuple[str, bool]:
    """(commit, dirty). A result from a dirty tree is not reproducible; say so.

    Cached for the life of the process. Two subprocesses per logged run is nothing for a
    single backtest and about 30 seconds of pure overhead across a 1,500-individual
    genetic search - and the commit does not change under a running process in any
    workflow that is not already lying about its own provenance.
    """
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

def panel_fingerprint(panel_meta: dict) -> str:
    """Short hash of the dataset the run saw, for reproducibility (not for trial counts).

    Public because the forward-test seal stores it too: a prediction made against one
    vintage of the price data and checked against another is a different experiment,
    and this is the only field that can show it.
    """
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

def jsonable(obj: Any) -> Any:
    """Coerce numpy scalars, arrays and stray objects into something json.dumps accepts.

    Public so the forward-test records serialise parameters exactly the way the trial
    log does - two different coercions would produce two different fingerprints for
    one strategy.
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)
