"""What a forward test leaves behind, and the read API a report will be built on.

Three append-only files under `data/experiments/forward/`:

======================  ====================================================
``seals.jsonl``         pre-registrations - written by `seal.py`
``forward_runs.jsonl``  one line per forward test: prediction, outcome, gap
``forward_curves.jsonl``  month-end curves for both windows, keyed by forward_id
======================  ====================================================

Why a separate store when the registry already logs every run
--------------------------------------------------------------
Both legs of a forward test *are* logged as ordinary runs, and their curves *are* in
`curves.jsonl` (ADR-027). This store does not duplicate that work for its own sake; it
exists for three reasons the trial log cannot serve.

1. **It cannot be switched off.** `SP500LAB_REGISTRY=off` silences the trial log by
   design, and a forward test is exactly the thing that must never be silenced. Same
   asymmetry as the holdout ledger in ADR-025.
2. **The record is a pair, not a run.** The interesting quantity is the difference
   between two runs plus the sampling error of that difference. Nothing about that fits
   in a `RunRecord`, and reconstructing it later would mean re-deriving which two runs
   belonged together.
3. **It is a different lifecycle.** `runs.jsonl` has 4,000 lines and grows by a thousand
   per genetic search. `forward_runs.jsonl` will have tens of lines, ever, and each one
   consumed a period that cannot be re-consumed. Mixing them would bury the second kind.

Measured cost: 5.6 KB per record, 10.4 KB per curve pair (both windows, three series
each) and 1.5 KB per seal. At tens of records that is nothing, and the alternative is
re-running a look that cannot be re-run.

Shape: structured on disk, flat in a DataFrame
-----------------------------------------------
The JSON keeps the legs nested under `research`, `research_recomputed` and `forward`, so
all three have exactly the same field names and code can rebuild a `compare.Leg` from any
of them. `load()` flattens them to `research_*`, `recomputed_*` and `forward_*` columns,
because that is what a table or a chart wants. Structured for code, flat for reading -
and only one of the two shapes is written down.

Building a report on this
--------------------------
Everything a forward report needs is here and is pure:

.. code-block:: python

    from sp500lab.forward import store

    store.load()                      # every forward test, one flat row each
    store.scoreboard()                # the paired table, sorted, report-ready
    store.selection_bar(store.load()) # the multiple-testing bar for the forward window
    store.get(forward_id)             # one record, structured
    store.load_curves([forward_id])   # {'research': df, 'forward': df} per id
    store.stitched_curve(forward_id)  # one continuous series across both windows
    store.annual_table(forward_id)    # year by year, strategy against the index

Those seven calls are the whole contract, they return DataFrames and dataclasses rather
than markup, and nothing in this module imports the reporting stack - which is the same
separation `reporting/views.py` relies on for every existing report (ADR-028).
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

import numpy as np
import pandas as pd

from ..backtest import metrics, registry
from ..paths import FORWARD_CURVE_LOG, FORWARD_LOG
from .compare import Leg
from .legs import leg_from_dict

log = logging.getLogger(__name__)

#: Bumped when the on-disk shape changes. Readers should check it before assuming a
#: column exists; `load()` tolerates missing ones.
FORWARD_FORMAT_VERSION = 1

#: Curve values are rounded to this many decimals. Same reasoning as ADR-027: curves are
#: rebased to 1.0, so six decimals is sub-basis-point and roughly halves the file.
_CURVE_DECIMALS = 6


@dataclass
class ForwardRecord:
    """One forward test. The prediction, the outcome, and everything needed to argue.

    Flat except for the two legs and the comparison, which are nested so that both
    windows share one field vocabulary. `as_flat_dict()` is what `load()` puts in a
    DataFrame.
    """

    # identity
    forward_id: str
    logged_at: str
    batch_id: str
    seal_id: str
    seal_mode: str
    strategy: str

    # configuration
    strategy_class: str = ""
    params: dict = field(default_factory=dict)
    construction: dict | None = None
    mode: str = "paired"
    cost_model: str = ""
    initial_capital: float = 0.0
    liquidity_floor: float = 0.0
    seed: int = 0
    benchmark: str = ""
    rationale: str = ""
    notes: str = ""

    # the two legs and what they say together
    #: The BINDING prediction - the research leg as it stood in the seal. This is what
    #: the comparison is against, because that is what pre-registration means.
    research: dict = field(default_factory=dict)
    #: The research leg as re-measured at test time. Identical to `research` for an
    #: auto-seal; different for a declared seal whose data has since been re-ingested.
    #: `seal_drift_sharpe` is the gap between the two, and a large one invalidates the
    #: comparison rather than adjusting it.
    research_recomputed: dict = field(default_factory=dict)
    forward: dict = field(default_factory=dict)
    comparison: dict = field(default_factory=dict)
    verdict: str = "inconclusive"
    verdict_reason: str = ""

    research_run_id: str = ""
    forward_run_id: str = ""

    # vintage - which part of this was new evidence
    data_end: str = ""
    previous_data_end: str = ""
    fresh_start: str = ""
    fresh_months: int = 0
    look_number: int = 1

    # the search behind the candidate, carried so a reader cannot lose it
    study: str | None = None
    forward_study: str = ""
    n_trials: int = 0
    trial_sharpe_std: float = 0.0
    deflated_sharpe_research: float = float("nan")
    #: Sealed research Sharpe minus the one recomputed at test time. Non-zero means the
    #: data changed under the prediction - a re-ingest, a restated fundamental, a fixed
    #: bug. Small values are noise in the adjustment chain; large ones invalidate the
    #: comparison and must be explained rather than averaged away.
    seal_drift_sharpe: float = float("nan")

    # honesty, for the forward leg specifically
    holdout_looks_total: int = 0
    n_rebalances: int = 0
    coverage_min: float = float("nan")
    coverage_median: float = float("nan")
    forced_exits: int = 0
    unresolved_exits: int = 0
    unfilled_orders: int = 0
    spread_fallback_orders: int = 0
    ruined: bool = False
    total_cost: float = 0.0
    commission: float = 0.0
    spread_cost: float = 0.0
    traded_notional: float = 0.0
    n_orders: int = 0

    # provenance
    git_commit: str = "unknown"
    git_dirty: bool = False
    panel_key: str = ""
    data_fingerprint: str = ""
    runtime_seconds: float = 0.0
    saved_to: str = ""
    format_version: int = FORWARD_FORMAT_VERSION

    # --------------------------------------------------------------- reading

    def research_leg(self) -> Leg:
        """The prediction the forward window was tested against."""
        return leg_from_dict(self.research, label="research")

    def recomputed_leg(self) -> Leg:
        """The research window as it measures today. Compare with `research_leg`."""
        return leg_from_dict(self.research_recomputed or self.research,
                             label="recomputed")

    def forward_leg(self) -> Leg:
        """What actually happened out of sample."""
        return leg_from_dict(self.forward, label="forward")

    def as_dict(self) -> dict:
        return asdict(self)

    def as_flat_dict(self) -> dict:
        """One row: nested legs become `research_*` / `forward_*`, checks drop out."""
        out = {k: v for k, v in self.as_dict().items()
               if k not in ("research", "research_recomputed", "forward", "comparison")}
        for prefix, leg in (("research", self.research),
                            ("recomputed", self.research_recomputed),
                            ("forward", self.forward)):
            for k, v in (leg or {}).items():
                out[f"{prefix}_{k}"] = v
        for k, v in (self.comparison or {}).items():
            if k == "checks":
                out["checks_failed"] = ",".join(
                    n for n, c in (v or {}).items() if c.get("passed") is False)
                continue
            out[k] = v
        return out


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def new_forward_id() -> str:
    """A sortable id for one forward test. Same shape as a run_id, prefixed so the two
    can never be confused in a table that carries both."""
    return "fwd-" + time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + \
        uuid.uuid4().hex[:6]


def new_batch_id() -> str:
    """Groups the records written by one command - e.g. three cost settings at once."""
    return "batch-" + time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + \
        uuid.uuid4().hex[:4]


def record(rec: ForwardRecord) -> ForwardRecord:
    """Append one forward test. Never gated on `SP500LAB_REGISTRY` - see the docstring."""
    registry.append_jsonl(FORWARD_LOG, rec.as_dict())
    log.info("forward test %s recorded: %s [%s] -> %s", rec.forward_id, rec.strategy,
             rec.cost_model, rec.verdict)
    return rec


def save_curves(forward_id: str, *, strategy: str, seal_id: str, cost_model: str,
                research: dict[str, pd.Series], forward: dict[str, pd.Series]) -> bool:
    """Store month-end curves for both windows. Returns False if there was nothing to store.

    `research` and `forward` are `{series_name: daily Series}` - typically `nav`,
    `nav_gross` and `benchmark`. Each window is resampled with `registry.to_monthly` so
    it lands on the same date grid the trial-log curves use, then rebased to 1.0 at its
    own start. Rebasing per window is deliberate: the two runs are independent, the
    forward one really did start from a fresh 100k, and a shared base would imply a
    continuity that did not happen. `stitched_curve()` re-imposes it for charting, and
    says so.
    """
    payload: dict[str, Any] = {
        "forward_id": forward_id, "strategy": strategy, "seal_id": seal_id,
        "cost_model": cost_model, "freq": "M",
        "format_version": FORWARD_FORMAT_VERSION,
    }
    wrote = False
    for window, series_map in (("research", research), ("forward", forward)):
        block = _curve_block(series_map)
        if block:
            payload[window] = block
            wrote = True
    if not wrote:
        return False
    registry.append_jsonl(FORWARD_CURVE_LOG, payload)
    return True


def _curve_block(series_map: dict[str, pd.Series]) -> dict | None:
    """{'dates': [...], 'nav': [...], ...} on the grid of whichever series comes first.

    The grid is month ends PLUS the window's own first session, because a run almost
    never starts on a month end. A forward window opening on 2022-01-01 first trades on
    2022-02-01 and its first month end is 2022-02-28; rebasing to that point would drop
    February's return from the curve, from the stitched chart and from the 2022 row of
    `annual_table` - about 2% of the evidence in a 54-month window, silently.
    """
    anchor = _anchor_date(series_map)
    dates: list[str] | None = None
    block: dict[str, Any] = {}
    for name, series in series_map.items():
        if series is None:
            continue
        monthly = _monthly_with_anchor(series, anchor)
        if monthly is None or monthly.empty:
            continue
        if dates is None:
            dates = [str(d) for d in monthly.index]
            block["dates"] = dates
        monthly = monthly.reindex(dates)
        base = monthly.dropna()
        if base.empty:
            continue
        block[name] = [None if not np.isfinite(v) else round(float(v), _CURVE_DECIMALS)
                       for v in (monthly / float(base.iloc[0])).to_numpy()]
    return block if dates else None


def _anchor_date(series_map: dict[str, pd.Series]) -> str:
    """The first session of the window, taken from the NAV series if there is one."""
    for name in ("nav", *series_map):
        series = series_map.get(name)
        if series is None:
            continue
        clean = series.dropna()
        if len(clean):
            return str(clean.index[0])
    return ""


def _monthly_with_anchor(series: pd.Series, anchor: str) -> pd.Series | None:
    """Month ends, with the window's opening level prepended when it is not one."""
    monthly = registry.to_monthly(series)
    if monthly is None or monthly.empty:
        return monthly
    if not anchor or anchor >= str(monthly.index[0]):
        return monthly
    clean = series.dropna()
    labels = [str(d) for d in clean.index]
    if anchor not in labels:
        return monthly
    opening = pd.Series([float(clean.iloc[labels.index(anchor)])], index=[anchor])
    return pd.concat([opening, monthly])


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def iter_records() -> Iterator[dict]:
    yield from registry.read_jsonl(FORWARD_LOG)


def load(strategy: str | None = None, seal_id: str | None = None) -> pd.DataFrame:
    """Every forward test, oldest first, one flat row each.

    The entry point for anything that draws a table. Columns are the record's own
    fields plus `research_*`, `forward_*` and the comparison's scalars; `checks_failed`
    is a comma-separated list of the named checks that did not pass.
    """
    rows = [_flatten(r) for r in iter_records()]
    if not rows:
        return pd.DataFrame(columns=_EMPTY_COLUMNS)
    df = pd.DataFrame(rows)
    if strategy is not None:
        df = df[df["strategy"] == strategy]
    if seal_id is not None:
        df = df[df["seal_id"] == seal_id]
    return df.reset_index(drop=True)


_EMPTY_COLUMNS = [
    "forward_id", "logged_at", "strategy", "seal_id", "seal_mode", "cost_model",
    "verdict", "look_number", "fresh_months",
    "research_cagr", "research_sharpe", "research_d_sharpe",
    "forward_cagr", "forward_sharpe", "forward_d_sharpe",
    "decay_sharpe_monthly", "decay_z", "psr_vs_research",
]


def get(forward_id: str) -> ForwardRecord | None:
    """The structured record for one forward test. The LAST line wins for an id.

    Last rather than first, unlike a seal: a forward_id is unique per run, so a repeat
    can only be a rewritten record, and there is no pre-registration guarantee at stake.
    """
    found = None
    for rec in iter_records():
        if rec.get("forward_id") == forward_id:
            found = rec
    return _from_dict(found) if found else None


def for_seal(seal_id: str, cost_model: str | None = None,
             mode: str | None = None) -> list[ForwardRecord]:
    """Every forward test of one sealed candidate, oldest first."""
    out = []
    for rec in iter_records():
        if rec.get("seal_id") != seal_id:
            continue
        if cost_model is not None and rec.get("cost_model") != cost_model:
            continue
        if mode is not None and rec.get("mode") != mode:
            continue
        out.append(_from_dict(rec))
    return out


def look_number(seal_id: str, cost_model: str, mode: str = "paired") -> int:
    """Which look this would be: 1 for the first, 2 for the next, and so on.

    Counted per (candidate, cost model, mode) rather than per candidate. Running one
    strategy under optimistic, realistic and pessimistic costs is three runs of one
    experiment - costs.py insists all three be reported together - and calling that
    three separate looks at the holdout would make every honest user look reckless.
    """
    return len(for_seal(seal_id, cost_model=cost_model, mode=mode)) + 1


def previous_data_end(seal_id: str, cost_model: str, mode: str = "paired") -> str | None:
    """The last date a previous look at this candidate actually SAW, or None if never.

    What `windows.freshness` needs to work out how much of the next look is new.

    The forward leg's own end, not `data_end`. Those differ whenever a run was stopped
    early with `forward_end` - reproducing an old vintage, say - and using the panel's
    end there would mark months as already-seen that no look has ever covered, which is
    the one direction this bookkeeping must not err in: it would silently retire
    evidence that is still fresh.
    """
    prior = for_seal(seal_id, cost_model=cost_model, mode=mode)
    ends = [(p.forward or {}).get("end") or p.data_end for p in prior]
    ends = [e for e in ends if e]
    return max(ends) if ends else None


def load_curves(forward_ids: "list[str] | str | None" = None
                ) -> dict[str, dict[str, pd.DataFrame]]:
    """{forward_id: {'research': df, 'forward': df}}, each indexed by month end.

    Reads the file once however many ids are asked for, the same way
    `registry.load_curves` does, so a comparison chart is one pass rather than N.
    """
    if isinstance(forward_ids, str):
        forward_ids = [forward_ids]
    wanted = set(forward_ids) if forward_ids is not None else None

    out: dict[str, dict[str, pd.DataFrame]] = {}
    for rec in registry.read_jsonl(FORWARD_CURVE_LOG):
        fid = rec.get("forward_id")
        if wanted is not None and fid not in wanted:
            continue
        windows: dict[str, pd.DataFrame] = {}
        for window in ("research", "forward"):
            df = _curve_frame(rec.get(window))
            if df is not None:
                df.attrs["strategy"] = rec.get("strategy", "")
                df.attrs["window"] = window
                windows[window] = df
        if windows:
            out[fid] = windows          # a later write supersedes an earlier one
    return out


def _curve_frame(block: dict | None) -> pd.DataFrame | None:
    if not block:
        return None
    dates = block.get("dates") or []
    cols = {k: block[k] for k in ("nav", "nav_gross", "benchmark")
            if block.get(k) is not None}
    if not dates or not cols:
        return None
    return pd.DataFrame(cols, index=pd.Index(dates, name="date"))


def stitched_curve(forward_id: str, column: str = "nav") -> pd.Series | None:
    """Research and forward as one continuous line, rebased to 1.0 at the start.

    **This is a presentation, not a simulation.** The two legs are independent runs: the
    forward one started from a fresh 100k with an empty book and paid entry costs the
    continuous path would not have. Splicing the forward curve onto the end of the
    research curve makes one readable chart out of that, and slightly *understates* the
    forward leg, because those entry costs land inside it. The alternative - running one
    unbroken backtest across the boundary - is available as `mode="continuous"` on the
    engine, and it is a different experiment rather than a better rendering of this one.

    The join date is the first index value of the forward leg, so a chart can mark it.
    """
    curves = load_curves([forward_id]).get(forward_id)
    if not curves:
        return None
    research = curves.get("research")
    forward = curves.get("forward")
    if forward is None or column not in forward:
        return None
    tail = forward[column].dropna()
    if research is None or column not in research:
        out = tail / float(tail.iloc[0])
        out.attrs["join_date"] = str(tail.index[0])
        return out
    head = research[column].dropna()
    if head.empty:
        return tail / float(tail.iloc[0])
    spliced = pd.concat([head, tail * float(head.iloc[-1])])
    spliced = spliced[~spliced.index.duplicated(keep="first")]
    spliced.attrs["join_date"] = str(tail.index[0])
    return spliced


def annual_table(forward_id: str) -> pd.DataFrame:
    """Year by year over the FORWARD window: strategy, benchmark, excess.

    Computed from the stored month-end curve rather than from a re-run, so a report can
    draw it years later with the panel gone.

    The first year is partial - a forward window opening on 2022-01-01 first trades on
    2022-02-01 - and it is reported from the curve's own opening level rather than
    dropped. Dropping it would remove the single most informative year in the current
    holdout, and measuring it from the first month end instead would quietly omit the
    opening month. `_curve_block` stores that opening level for exactly this reason.
    """
    curves = load_curves([forward_id]).get(forward_id) or {}
    df = curves.get("forward")
    if df is None or "nav" not in df:
        return pd.DataFrame(columns=["year", "strategy", "benchmark", "excess"])
    out = pd.DataFrame({"strategy": _annual_from_curve(df["nav"])})
    if "benchmark" in df:
        out["benchmark"] = _annual_from_curve(df["benchmark"])
        out["excess"] = out["strategy"] - out["benchmark"]
    return out.reset_index().rename(columns={"index": "year"})


def _annual_from_curve(curve: pd.Series) -> pd.Series:
    """Calendar-year returns from a month-end NAV curve, keeping the partial first year.

    `metrics.annual_returns` resamples to year ends and takes a pct_change, which drops
    the first year because it has no predecessor. Here the curve's own opening value is
    the predecessor, which is correct for a window that starts mid-year and is the
    difference between reporting 2022 and silently omitting it.
    """
    s = curve.dropna().astype(float)
    if len(s) < 2:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(s.index))))
    s = pd.Series(s.to_numpy(), index=idx)
    year_end = s.resample("YE").last()
    base = year_end.shift(1)
    base.iloc[0] = float(s.iloc[0])
    out = year_end / base - 1.0
    out.index = [str(d.year) for d in out.index]
    return out


# --------------------------------------------------------------------------
# The scoreboard, and the bar it has to clear
# --------------------------------------------------------------------------

def scoreboard(records: pd.DataFrame | None = None,
               cost_model: str | None = None) -> pd.DataFrame:
    """Prediction against outcome for every forward test, sorted by what held up best.

    Sorted on `forward_d_sharpe` - the forward Sharpe minus the index's over the same
    dates - for the same reason `results.suite` sorts on it: the windows differ, and a
    raw Sharpe ranked across different windows ranks the market rather than the
    strategies.
    """
    df = load() if records is None else records
    if df.empty:
        return pd.DataFrame(columns=[
            "strategy", "cost_model", "verdict", "research_sharpe", "forward_sharpe",
            "decay_sharpe_monthly", "decay_z", "forward_d_sharpe", "fresh_months",
            "look_number", "seal_mode"])
    if cost_model is not None:
        df = df[df["cost_model"] == cost_model]
    cols = [c for c in ("strategy", "cost_model", "verdict", "seal_mode", "look_number",
                        "fresh_months", "research_cagr", "forward_cagr",
                        "research_sharpe", "forward_sharpe", "research_d_sharpe",
                        "forward_d_sharpe", "decay_sharpe_monthly", "decay_z",
                        "psr_vs_research", "forward_id") if c in df.columns]
    out = df[cols].copy()
    if "forward_d_sharpe" in out.columns:
        out = out.sort_values("forward_d_sharpe", ascending=False)
    return out.reset_index(drop=True)


def selection_bar(records: pd.DataFrame | None = None,
                  cost_model: str | None = "realistic") -> dict:
    """The multiple-testing bar for the FORWARD window itself.

    The holdout protects against fitting. It does not protect against *choosing*: pick
    the best of twenty forward tests and the winner's Sharpe carries the same
    best-of-N inflation that the deflated Sharpe corrects for in a research search
    (ADR-026). The correction is identical, so the same function computes it -
    `metrics.expected_max_sharpe` over the spread of forward Sharpes actually observed.

    One cost model at a time by default: three cost settings of one strategy are one
    hypothesis, and counting them as three would triple the apparent search.

    Returns the count, the spread, the bar, and the best forward Sharpe, so the two can
    be read against each other.
    """
    df = load() if records is None else records
    if cost_model is not None and "cost_model" in df.columns:
        df = df[df["cost_model"] == cost_model]
    if df.empty or "forward_sharpe_monthly" not in df.columns:
        return {"n_forward_tests": 0, "spread": 0.0, "bar": 0.0,
                "best_sharpe_monthly": float("nan"), "cost_model": cost_model}

    per_candidate = df.drop_duplicates("seal_id")["forward_sharpe_monthly"].dropna()
    n = int(len(per_candidate))
    spread = float(per_candidate.std(ddof=1)) if n > 1 else 0.0
    root = np.sqrt(12.0)
    bar = metrics.expected_max_sharpe(n, spread / root) * root
    return {
        "n_forward_tests": n,
        "cost_model": cost_model,
        "spread": round(spread, 4),
        "bar": round(float(bar), 4),
        "best_sharpe_monthly": (round(float(per_candidate.max()), 4) if n else
                                float("nan")),
    }


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------

def _from_dict(rec: dict) -> ForwardRecord:
    known = {f for f in ForwardRecord.__dataclass_fields__}
    return ForwardRecord(**{k: v for k, v in rec.items() if k in known})


def _flatten(rec: dict) -> dict:
    return _from_dict(rec).as_flat_dict()
