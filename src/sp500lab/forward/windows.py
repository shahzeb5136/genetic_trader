"""The three windows a forward test involves, and how much any of them can prove.

Pure date arithmetic and pure statistics. Nothing here reads the panel, the registry or
the disk, which is what makes all of it testable without any ingested data.

The three windows
-----------------
::

    2007-04 .................. 2021-12 | 2022-01 .......... today
    <------- RESEARCH -------->        <------ FORWARD ------>
                                       <--- fresh --->
                                       (only on look #2 and after)

**Research** is everything a search was allowed to see: `DEFAULT_START` to
`registry.research_end()`. Every strategy in this project was written, tuned, evolved
and ranked inside it.

**Forward** is `HOLDOUT_START` to the end of the data. It is the only period in which a
number produced by this project is an out-of-sample number, and it grows by one month
every month whether anyone runs anything or not.

**Fresh** is the part of the forward window that did not exist the last time this
candidate was looked at. It is the important one and it is easy to miss. The first
forward test of a strategy spends the whole holdout; the second one, run a year later,
is *mostly* re-reading data it has already seen - except for the twelve months that
arrived in between, which are genuinely untainted. So a forward record stores the data
vintage it ran against, and `freshness()` recovers how much of a later look is new
evidence rather than a second reading of old evidence.

Why the sample-size functions live here too
--------------------------------------------
Because they are the reason this module is not just a date helper. The forward window
is short, and the standard error of a Sharpe over a short window is enormous:

======================  ============================
months of forward data  95% band on a Sharpe of 1.0
======================  ============================
24                      +- 1.42
36                      +- 1.15
56                      +- 0.92
120                     +- 0.62
======================  ============================

Fifty-six months cannot distinguish a Sharpe of 0.1 from one of 1.9. That is not a
caveat to add at the end of a forward report; it is the single most important fact
about forward testing on this dataset, and `describe_power()` exists so that every
verdict has to be printed next to it.

The consequence, stated once so the rest of the package can rely on it: **a forward
test on this data can refute a strategy, and cannot confirm one.** A large negative
result is informative because it is large. A modest positive result is indistinguishable
from noise, and calling it a pass would be the same error as reading an undeflated
Sharpe off a genetic search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ..backtest import metrics
from ..backtest.registry import HOLDOUT_START, research_end

#: Below this many months, a forward test is not reported as held or failed - only as
#: inconclusive. Two years is already far too short to be conclusive (the table in the
#: module docstring says +-1.42 on a Sharpe of 1.0); it is the point below which the
#: window cannot even establish the SIGN of a difference against a research Sharpe of
#: the size this project produces, so a verdict would be theatre.
MIN_FORWARD_MONTHS = 24

#: Months per year, spelled out because the annualisation of a Sharpe and of its
#: standard error must use the same constant or the interval will not bracket the point.
MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class Window:
    """A closed date range, plus how many month ends it contains.

    `n_months` is calendar month ends rather than sessions, because every statistic a
    forward test compares is computed on the monthly series (ADR-026) and the daily
    count would overstate the evidence by a factor of 21.
    """

    start: str
    end: str
    label: str = ""

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"window {self.label or ''} ends ({self.end}) before it "
                             f"starts ({self.start})")

    @property
    def n_months(self) -> int:
        return month_ends_between(self.start, self.end)

    @property
    def years(self) -> float:
        return (pd.Timestamp(self.end) - pd.Timestamp(self.start)).days / 365.25

    def contains(self, date: str) -> bool:
        return self.start <= date <= self.end

    def clip(self, start: str | None = None, end: str | None = None) -> "Window":
        """A sub-window, never wider than this one."""
        return Window(max(self.start, start or self.start),
                      min(self.end, end or self.end), self.label)

    def describe(self) -> str:
        return f"{self.start}..{self.end}  ({self.n_months} months, {self.years:.1f}y)"

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "label": self.label,
                "n_months": self.n_months, "years": round(self.years, 3)}


def month_ends_between(start: str, end: str) -> int:
    """How many calendar month ends fall in [start, end].

    Calendar, not sessions: this counts the number of *observations* a monthly
    statistic would have, which is the n every standard error in this module divides by.
    """
    if end < start:
        return 0
    return int(len(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="ME")))


def research_window(start: str, end: str | None = None) -> Window:
    """The period a search was allowed to see. Never reaches the holdout.

    `end` is clamped to `registry.research_end()` even when a caller asks for more, so
    a mis-typed date cannot silently widen the training period. Widening it is what a
    holdout exists to prevent, and this function is the last place that can refuse.
    """
    stop = min(end or research_end(), research_end())
    return Window(start, stop, label="research")


def forward_window(data_end: str, start: str = HOLDOUT_START,
                   end: str | None = None) -> Window:
    """The out-of-sample period available right now, given how far the data runs.

    `start` defaults to `HOLDOUT_START` and is *floored* at it: a forward test that
    began earlier would be reading research data and reporting it as out-of-sample,
    which is the exact failure this whole module is built to make impossible.
    """
    lo = max(start, HOLDOUT_START)
    hi = min(end or data_end, data_end)
    if hi < lo:
        raise ValueError(
            f"no forward data: the holdout opens on {lo} and the panel ends on {hi}. "
            "Ingest more prices before forward testing.")
    return Window(lo, hi, label="forward")


def freshness(window: Window, previous_data_end: str | None) -> tuple[Window | None, int]:
    """(fresh sub-window, months of it) - the part of `window` nobody has seen.

    `previous_data_end` is the data vintage of the last forward test of this same
    candidate, or None if there has never been one.

    On a first look everything is fresh. On a later look only the months that arrived
    since the previous one are, and they are the only part of the result that is
    genuinely new evidence. Re-running a forward test against the same data and
    reporting the same Sharpe again is not a second confirmation; it is the same
    measurement printed twice.
    """
    if previous_data_end is None or previous_data_end < window.start:
        return window, window.n_months
    if previous_data_end >= window.end:
        return None, 0
    fresh = Window(_day_after(previous_data_end), window.end, label="fresh")
    return fresh, fresh.n_months


def _day_after(date: str) -> str:
    return str((pd.Timestamp(date) + pd.Timedelta(days=1)).date())


# --------------------------------------------------------------------------
# What a window of this length can and cannot show
# --------------------------------------------------------------------------

def sharpe_band(sharpe_annualised: float, n_months: int, skew: float = 0.0,
                kurtosis: float = 3.0, confidence: float = 0.95) -> tuple[float, float]:
    """95% interval around an ANNUALISED Sharpe measured over `n_months` months.

    Converts to per-month units, uses the shared standard error in `metrics`, and
    converts back - so the interval printed next to a Sharpe always agrees with the
    probabilistic Sharpe printed under it.
    """
    if n_months < 3 or not math.isfinite(sharpe_annualised):
        return float("nan"), float("nan")
    root = math.sqrt(MONTHS_PER_YEAR)
    lo, hi = metrics.sharpe_confidence_interval(
        sharpe_annualised / root, n_months, skew, kurtosis, confidence)
    return lo * root, hi * root


def detectable_sharpe_gap(n_months: int, sharpe_annualised: float = 1.0,
                          confidence: float = 0.95) -> float:
    """Smallest ANNUALISED Sharpe difference two independent windows could resolve.

    Two independent estimates each carry their own error, so the difference carries
    sqrt(2) times one of them. This is the number that says whether "the forward Sharpe
    fell from 1.30 to 0.85" is a finding or a coin flip: on 56 months it is a coin flip,
    because the resolution is about 1.3.
    """
    if n_months < 3:
        return float("nan")
    root = math.sqrt(MONTHS_PER_YEAR)
    se = metrics.sharpe_standard_error(sharpe_annualised / root, n_months) * root
    return metrics.norm_ppf(0.5 + confidence / 2.0) * math.sqrt(2.0) * se


def describe_power(n_months: int, sharpe_annualised: float = 1.0) -> str:
    """One paragraph saying what a window this long can prove. Printed with every test.

    Deliberately returns prose rather than a number. The failure mode this guards
    against is a reader taking a forward Sharpe at face value, and a reader who skips
    a table will still read a sentence.
    """
    if n_months < 3:
        return ("Fewer than three months of forward data: no statistic computed over it "
                "means anything yet.")
    lo, hi = sharpe_band(sharpe_annualised, n_months)
    gap = detectable_sharpe_gap(n_months, sharpe_annualised)
    verdict = ("enough to refute a strategy that fails badly, and not enough to confirm "
               "one that does not")
    if n_months < MIN_FORWARD_MONTHS:
        verdict = ("too short for any verdict at all - reported as inconclusive "
                   f"below {MIN_FORWARD_MONTHS} months")
    return (
        f"{n_months} months of forward data. A Sharpe of {sharpe_annualised:.2f} "
        f"measured over it has a 95% interval of [{lo:.2f}, {hi:.2f}], and the smallest "
        f"difference from the research window this test could resolve is {gap:.2f} "
        f"Sharpe. That is {verdict}.")
