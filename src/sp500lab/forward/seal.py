"""Pre-registration: write down the prediction before looking at the answer.

The problem this solves
-----------------------
The holdout (ADR-025) stops a strategy from being *fitted* to 2022 onward. It does not
stop the far more common failure, which is choosing what to test after seeing how it
did. Forward-test twenty strategies, report the three that worked, and the holdout has
been converted into a second research window - every guarantee ADR-025 bought is gone,
and nothing in the trial log would show it, because each individual run was honest.

Medicine solved this with trial pre-registration and the reasoning transfers exactly. A
**seal** is a record, written before the look, saying:

* which strategy, down to a configuration fingerprint,
* what its research window produced - the prediction,
* how many trials that research window cost, so the prediction is already deflated,
* and *why* this candidate was chosen, in free text, because a rationale written
  afterwards never reads like one written before.

The seal is appended to `data/experiments/forward/seals.jsonl` and never rewritten.

Declared or auto - and why both exist
--------------------------------------
Requiring a separate `forward seal` command before every forward test would be the
strict design, and it would be routed around within a week. So `forward run`
**auto-seals** anything it has not seen before: it measures the research window first,
records that as the prediction, and only then looks forward.

An auto-seal is honest about the numbers - the prediction is computed from research data
alone, exactly as a declared one is. What it cannot prove is *ordering*: that this
candidate was chosen before anybody saw how it did. So the mode is recorded and travels
with every downstream table:

``declared``  sealed by its own command, at a timestamp that precedes the look
``auto``      sealed at the moment of the look; the numbers are clean, the choice is not

A set of `declared` seals written on one day and tested on another is real
pre-registration. A pile of `auto` seals is a survey, and should be read as one.

The seal id is deterministic
-----------------------------
`seal_id` is a hash of the configuration, not a timestamp, so re-sealing the same
candidate lands on the same id. That is what lets `store.look_number` count how many
times one candidate has been forward-tested, and it means a second seal of the same
thing is a second *line* rather than a second candidate. The reader takes the
**earliest** line as binding: pre-registration means the first thing you wrote down,
and a later one written after a disappointing look is exactly what must not count.

See ADR-033 and docs/FORWARD_TEST.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

import pandas as pd

from ..backtest import registry
from ..paths import SEAL_LOG
from .compare import Leg
from .legs import leg_from_dict, leg_from_result

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..backtest.results import BacktestResult

log = logging.getLogger(__name__)

#: Bumped when the on-disk shape of a seal changes, so a reader can tell.
SEAL_FORMAT_VERSION = 1

SEAL_MODES = ("declared", "auto")


@dataclass
class Seal:
    """One pre-registered candidate. Append-only; the earliest line for an id binds."""

    seal_id: str
    sealed_at: str
    seal_mode: str
    strategy: str
    rationale: str

    # configuration - what would have to be true for a later run to be the same thing
    strategy_class: str = ""
    params: dict = field(default_factory=dict)
    construction: dict | None = None
    warmup: int = 0
    min_date: str = ""
    cost_model: str = ""
    initial_capital: float = 0.0
    liquidity_floor: float = 0.0
    seed: int = 0
    benchmark: str = ""

    #: The prediction: a `compare.Leg` for the research window, nested rather than
    #: flattened so the two legs of a forward record share one shape.
    research: dict = field(default_factory=dict)
    research_run_id: str = ""
    research_fingerprint: str = ""

    # the search this candidate came out of, carried so it cannot be forgotten later
    study: str | None = None
    n_trials: int = 0
    trial_sharpe_std: float = 0.0
    deflated_sharpe: float = float("nan")

    # provenance
    data_end: str = ""
    data_fingerprint: str = ""
    git_commit: str = "unknown"
    git_dirty: bool = False
    format_version: int = SEAL_FORMAT_VERSION

    def research_leg(self) -> Leg:
        return leg_from_dict(self.research, label="research")

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        leg = self.research_leg()
        lines = [
            f"  seal            {self.seal_id}   ({self.seal_mode})",
            f"  sealed at       {self.sealed_at}",
            f"  strategy        {self.strategy}  [{self.cost_model} costs]",
            f"  rationale       {self.rationale or '(none given)'}",
            f"  predicts        CAGR {leg.cagr * 100:.2f}%   Sharpe {leg.sharpe:.2f}   "
            f"maxDD {leg.max_drawdown * 100:.1f}%",
            f"  measured over   {leg.start}..{leg.end}  ({leg.n_months} months)",
            f"  from study      {self.study}  ({self.n_trials} trial(s), "
            f"deflated Sharpe {_fmt(self.deflated_sharpe)})",
            f"  data vintage    {self.data_end}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Building one
# --------------------------------------------------------------------------

def seal_id_for(*, strategy_class: str, params: dict, construction: dict | None,
                cost_model: str, initial_capital: float, liquidity_floor: float,
                seed: int) -> str:
    """A stable id for one candidate CONFIGURATION.

    Deliberately narrower than `registry.fingerprint`: the dates are excluded, because
    a candidate is the same candidate whether its research window was measured to
    2021-12-31 or to some earlier date, and a forward test of it is a test of the same
    idea. Everything that changes what the strategy DOES is in here.
    """
    payload = json.dumps({
        "cls": strategy_class, "params": _jsonable(params),
        "construction": _jsonable(construction), "cost_model": cost_model,
        "capital": round(float(initial_capital), 2),
        "liquidity_floor": float(liquidity_floor), "seed": int(seed),
    }, sort_keys=True)
    return "seal-" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def create_seal(result: "BacktestResult", *, rationale: str, seal_mode: str = "declared",
                study: str | None = None) -> Seal:
    """Build a seal from a RESEARCH-window backtest. Does not write anything.

    Refuses a result that saw holdout data: a prediction measured on the period it is
    about to be tested against is not a prediction. That check is the whole reason this
    function takes a `BacktestResult` rather than a set of numbers.

    `study` is the search the CANDIDATE came from, and the caller must resolve it -
    there is deliberately no fallback to the study the baseline run happened to be
    logged under. That fallback existed briefly and was actively dangerous: it made
    every unattributed candidate look like a one-trial study with a flattering deflated
    Sharpe, which is precisely the number the registry exists to prevent anyone from
    quoting. `None` means "unknown", carries `n_trials=0`, and prints as unknown.
    """
    if seal_mode not in SEAL_MODES:
        raise ValueError(f"seal_mode must be one of {SEAL_MODES}, got {seal_mode!r}")
    cfg = result.config
    if cfg.get("touched_holdout"):
        raise ValueError(
            f"cannot seal {result.strategy!r} from a run that saw holdout data "
            f"(holdout={cfg.get('holdout_mode')!r}). A seal records what the RESEARCH "
            "window predicted; sealing from a run that already looked forward would "
            "make the prediction and the test the same measurement.")

    detail = cfg.get("strategy_detail", {})
    strategy_class = detail.get("class", result.strategy)
    params = _jsonable(detail.get("params", {}))
    construction = _jsonable(detail.get("construction"))
    commit, dirty = registry.git_state()

    n_trials, spread, dsr = _search_context(study, cfg.get("run_id"))

    return Seal(
        seal_id=seal_id_for(
            strategy_class=strategy_class, params=params, construction=construction,
            cost_model=str(cfg.get("cost_model", "")),
            initial_capital=float(cfg.get("initial_capital", 0.0)),
            liquidity_floor=float(cfg.get("liquidity_floor", 0.0)),
            seed=int(cfg.get("seed", 0))),
        sealed_at=_now_iso(),
        seal_mode=seal_mode,
        strategy=result.strategy,
        rationale=rationale.strip(),
        strategy_class=strategy_class,
        params=params,
        construction=construction,
        warmup=int(detail.get("warmup", 0) or 0),
        min_date=str(detail.get("min_date", "") or ""),
        cost_model=str(cfg.get("cost_model", "")),
        initial_capital=float(cfg.get("initial_capital", 0.0)),
        liquidity_floor=float(cfg.get("liquidity_floor", 0.0)),
        seed=int(cfg.get("seed", 0)),
        benchmark=str(cfg.get("benchmark") or ""),
        research=leg_from_result(result, "research").as_dict(),
        research_run_id=str(cfg.get("run_id", "") or ""),
        research_fingerprint=str(cfg.get("fingerprint", "") or ""),
        study=study,
        n_trials=n_trials,
        trial_sharpe_std=spread,
        deflated_sharpe=dsr,
        data_end=str(cfg.get("panel", {}).get("end", "")),
        data_fingerprint=registry.panel_fingerprint(cfg.get("panel", {})),
        git_commit=commit,
        git_dirty=dirty,
    )


def _search_context(study: str | None, run_id: str | None) -> tuple[int, float, float]:
    """(n_trials, trial_sharpe_std, deflated_sharpe) for the search behind a candidate.

    Carried into the seal so a forward result can never be read without the multiple
    testing that produced the candidate. A forward test of the winner of a
    1,400-individual search is a different claim from a forward test of the first thing
    anybody wrote, and by the time somebody reads the forward table the study may be
    hard to find.

    Failures here are not fatal: a hand-written candidate with no study still deserves
    a seal, it just carries zeros.
    """
    if not study:
        return 0, 0.0, float("nan")
    try:
        n_trials = registry.count_trials(study)
        spread = registry.trial_sharpe_std(study)
        dsr = float("nan")
        # A deflated Sharpe needs a trial count. With none, `deflate` sets the
        # best-of-N bar to zero and hands back the probabilistic Sharpe against zero -
        # a real statistic under a misleading name, and the most flattering possible
        # reading of an unattributed candidate. Report nothing instead.
        if run_id and n_trials:
            row = registry.get(run_id)
            if row is not None:
                dsr = float(registry.deflate(row, study).get("deflated_sharpe",
                                                             float("nan")))
        return n_trials, spread, dsr
    except Exception as exc:                                      # noqa: BLE001
        log.warning("could not read the search context for study %r: %s", study, exc)
        return 0, 0.0, float("nan")


# --------------------------------------------------------------------------
# Storage - append-only, never disabled
# --------------------------------------------------------------------------

def record(seal: Seal) -> Seal:
    """Append a seal to the ledger.

    Not gated on `SP500LAB_REGISTRY`, for the same reason the holdout ledger is not
    (ADR-025): a scratch run may go unlogged, but nothing that consumes out-of-sample
    data may go unrecorded.
    """
    registry.append_jsonl(SEAL_LOG, seal.as_dict())
    log.info("sealed %s as %s (%s): %s", seal.strategy, seal.seal_id, seal.seal_mode,
             seal.rationale or "no rationale given")
    return seal


def iter_seals() -> Iterator[dict]:
    yield from registry.read_jsonl(SEAL_LOG)


def load_seals(strategy: str | None = None) -> pd.DataFrame:
    """Every seal ever written, oldest first, flattened for reading.

    Flattened, not raw: the nested `research` leg becomes `research_*` columns so the
    frame drops straight into a table or a report. `get(seal_id)` returns the structured
    record instead, which is what code should use.
    """
    rows = [_flatten(r) for r in iter_seals()]
    if not rows:
        return pd.DataFrame(columns=["seal_id", "sealed_at", "seal_mode", "strategy",
                                     "rationale", "research_sharpe", "research_cagr"])
    df = pd.DataFrame(rows)
    if strategy is not None:
        df = df[df["strategy"] == strategy].reset_index(drop=True)
    return df


def get(seal_id: str) -> Seal | None:
    """The BINDING seal for an id - the earliest line written for it.

    Earliest, not latest, and the distinction is the entire mechanism. A seal rewritten
    after a disappointing forward test would otherwise silently replace the prediction
    it failed to meet.
    """
    for rec in iter_seals():
        if rec.get("seal_id") == seal_id:
            return _from_dict(rec)
    return None


def history(seal_id: str) -> list[Seal]:
    """Every line written for one id, oldest first. The first one is what binds."""
    return [_from_dict(r) for r in iter_seals() if r.get("seal_id") == seal_id]


def find(strategy: str, cost_model: str | None = None) -> Seal | None:
    """The earliest seal for a strategy name, optionally under one cost model."""
    for rec in iter_seals():
        if rec.get("strategy") != strategy:
            continue
        if cost_model is not None and rec.get("cost_model") != cost_model:
            continue
        return _from_dict(rec)
    return None


def _from_dict(rec: dict) -> Seal:
    known = {f for f in Seal.__dataclass_fields__}
    return Seal(**{k: v for k, v in rec.items() if k in known})


def _flatten(rec: dict) -> dict:
    out = {k: v for k, v in rec.items() if k != "research"}
    for k, v in (rec.get("research") or {}).items():
        out[f"research_{k}"] = v
    return out


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fmt(x: float) -> str:
    return "n/a" if x is None or x != x else f"{x:.3f}"


def _jsonable(obj: Any) -> Any:
    """The registry's coercion, so a seal and a trial serialise params identically."""
    return registry.jsonable(obj)
