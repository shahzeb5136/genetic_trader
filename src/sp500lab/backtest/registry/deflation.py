"""The deflated Sharpe ratio: what the best of N trials is worth.

Run twenty variants of a momentum rule, report the best, and its Sharpe is a maximum
over twenty draws whether or not there is any signal in the data. The deflated Sharpe
(Bailey & Lopez de Prado, 2014) corrects for exactly that, and needs two inputs that
describe the *search* rather than the winner:

    n_trials            how many distinct configurations were evaluated
    trial_sharpe_std    how much their Sharpes varied

Neither can be recovered after the fact, which is why `store` logs every individual and
not just the winners. This module only reads that log; the arithmetic itself is in
`backtest/metrics.py`.

A `study` is a named search, and the study boundary is what decides which trials the
winner has to beat. The count is over distinct fingerprints, not over log lines:
re-running an identical configuration is the same trial, not a new one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import metrics
from . import store

if TYPE_CHECKING:
    import pandas as pd


def trial_sharpe_std(study: str) -> float:
    """Spread of monthly Sharpes across the distinct trials in a study.

    The second input the deflated Sharpe needs. A wide spread means the search had a lot
    of room to get lucky, so the winner has more to prove.
    """
    df = store.load(study)
    if df.empty:
        return 0.0
    per_trial = df.drop_duplicates("fingerprint")["sharpe_monthly"].dropna()
    return float(per_trial.std(ddof=1)) if len(per_trial) > 1 else 0.0


def deflate(run: "pd.Series | str", study: str | None = None) -> dict:
    """Deflated Sharpe for one run, given the search it came from.

    Accepts a run_id or a row from `store.load()`. `study` defaults to the run's own study,
    which is almost always what you want - the winner has to beat the search that
    produced it, not some other search.

    Returns the DSR, the PSR against zero, the bar the search set (the Sharpe the
    luckiest of N worthless strategies would have posted), and the inputs, so the number
    can be argued with rather than just quoted.

    Read the result as a probability, not a score. Below ~0.95 the strategy is not
    distinguishable from the best of N lucky draws, however good its raw Sharpe looks.
    """
    row = store.get(run) if isinstance(run, str) else run
    if row is None:
        raise KeyError(f"no run with id {run!r}")
    study = study or row["study"]

    n_trials = store.count_trials(study)
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
    row = store.best(study)
    if row is None:
        raise KeyError(f"study {study!r} has no runs")
    return deflate(row, study)
