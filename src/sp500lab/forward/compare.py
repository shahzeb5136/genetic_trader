"""The paired comparison: what was predicted, what happened, and whether the gap is real.

A forward test is not a number, it is a *difference*. The research window produced a
claim - "this compounds at 14%/yr with a Sharpe of 1.3" - and the forward window is the
only place that claim can be checked. So everything here works on two `Leg`s at once
and the output is always a delta next to the sampling error of that delta.

Why the sampling error is not a footnote
-----------------------------------------
The tempting way to read a forward test is: research Sharpe 1.32, forward Sharpe 0.71,
therefore it halved, therefore it was overfitted. On 176 and 56 monthly observations the
standard errors are roughly 0.26 and 0.47, so the difference of 0.61 carries a standard
error of 0.54. It is not distinguishable from zero. It is also not distinguishable from
a total collapse. **Both** of those statements are true at once, and a comparison that
prints only the first is as misleading as one that prints only the second.

Hence `decay_z` and `decay_p` sit in the same dataclass as `decay_sharpe`, and the
verdict is computed from the z rather than from the raw drop.

Three probabilities, three different questions
-----------------------------------------------
==========================  ===============================================
``psr_vs_zero``             did it make risk-adjusted money at all?
``psr_vs_benchmark``        did it beat the index over the SAME window?
``psr_vs_research``         did it live up to what research promised?
==========================  ===============================================

The third is the one this module exists for and the one nobody computes. It uses the
research Sharpe as the benchmark inside the probabilistic Sharpe, which is exactly the
same machinery the deflated Sharpe uses with the expected-maximum as its benchmark
(`metrics.probabilistic_sharpe`) - the difference is only what you put on the right-hand
side of the comparison.

What a verdict means, and what it does not
-------------------------------------------
`held` does not mean the strategy works. It means fifty-odd months failed to refute it,
and `windows.describe_power()` says plainly how little that is worth. `failed` is the
informative outcome, because refuting something is what a short sample can actually do.
The asymmetry is real and the vocabulary is chosen to keep it visible.

Nothing in this module touches the disk, the panel or the registry. Two dataclasses in,
one dataclass out - which is what lets `tests/test_forward.py` pin every verdict rule
against hand-built inputs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from ..backtest import metrics
from .windows import MIN_FORWARD_MONTHS, MONTHS_PER_YEAR, sharpe_band

#: Decay smaller than this many standard errors is treated as sampling noise rather
#: than as evidence of anything. One SE, not two: the cost of wrongly calling a decayed
#: strategy healthy is larger here than the cost of wrongly flagging a healthy one, and
#: the flag is a word in a table rather than an automated decision.
DECAY_NOISE_SIGMA = 1.0

#: Decay beyond this many standard errors is a refutation rather than a wobble. The
#: conventional 1.96 - and note how rarely a 56-month window can reach it, which is the
#: honest state of affairs rather than a defect in the threshold.
DECAY_FATAL_SIGMA = 1.96

#: A forward turnover more than this multiple of the research figure means the strategy
#: is not doing what it did before, whatever its return says. Trading twice as often in
#: a new regime is a change of behaviour, and it is charged for twice: once in costs,
#: once in the model risk of the spread estimate (ADR-020).
TURNOVER_DRIFT_LIMIT = 1.5


@dataclass(frozen=True)
class Leg:
    """One window's results, reduced to the numbers a comparison needs.

    Built from a `BacktestResult` by `legs.leg_from_result`, or by hand in a test.
    Deliberately not a `Performance`: a Leg carries the MONTHLY statistics the standard
    errors need, the benchmark over the same dates, and nothing else - so a comparison
    cannot accidentally reach for the daily Sharpe, which is autocorrelated and would
    make every interval below three times too narrow (ADR-026).

    `n_months` counts monthly RETURN observations, which is one fewer than the number
    of month ends in the window - the first month end is an opening level, not a
    return. Every standard error here divides by that count, so it is the one that
    matters; `windows.Window.n_months` counts month ends and is the one that describes
    a date range. They differ by one and both are correct for their own job.
    """

    label: str = ""
    start: str = ""
    end: str = ""
    n_months: int = 0

    cagr: float = float("nan")
    sharpe: float = float("nan")              # annualised, from the DAILY curve
    sharpe_monthly: float = float("nan")      # annualised, from MONTH-END returns
    skew_monthly: float = float("nan")
    kurtosis_monthly: float = float("nan")    # excess, as pandas reports it
    ann_vol: float = float("nan")
    max_drawdown: float = float("nan")
    ann_turnover: float = float("nan")
    cost_drag: float = float("nan")
    hit_rate: float = float("nan")
    avg_positions: float = float("nan")

    bench_cagr: float = float("nan")
    bench_sharpe: float = float("nan")

    ruined: bool = False

    @property
    def excess(self) -> float:
        """CAGR minus the benchmark's CAGR over this leg's own dates."""
        return self.cagr - self.bench_cagr

    @property
    def d_sharpe(self) -> float:
        """Sharpe minus the benchmark's Sharpe over this leg's own dates.

        The only column in this project that compares two strategies fairly when their
        windows differ - see `benchmark.over_window`. Here the windows differ by
        construction, so it is the only column that compares the two legs fairly either.
        """
        return self.sharpe - self.bench_sharpe

    @property
    def sharpe_se(self) -> float:
        """Standard error of the ANNUALISED monthly Sharpe over this leg."""
        return _annualised_se(self.sharpe_monthly, self.n_months,
                              self.skew_monthly, self.kurtosis_monthly)

    def band(self, confidence: float = 0.95) -> tuple[float, float]:
        """95% interval around this leg's annualised monthly Sharpe."""
        return sharpe_band(self.sharpe_monthly, self.n_months,
                           _finite(self.skew_monthly, 0.0),
                           _full_kurtosis(self.kurtosis_monthly), confidence)

    def as_dict(self) -> dict:
        d = asdict(self)
        d |= {"excess": self.excess, "d_sharpe": self.d_sharpe,
              "sharpe_se": self.sharpe_se}
        return d


@dataclass(frozen=True)
class Check:
    """One named, testable condition. `passed=None` means "could not be evaluated"."""

    name: str
    passed: bool | None
    detail: str

    @property
    def mark(self) -> str:
        return {True: "ok", False: "FAIL", None: "n/a"}[self.passed]


@dataclass
class Comparison:
    """Research against forward: the deltas, the significance, the checks, the verdict."""

    research: Leg
    forward: Leg

    decay_cagr: float = float("nan")
    decay_sharpe: float = float("nan")
    decay_sharpe_monthly: float = float("nan")
    decay_d_sharpe: float = float("nan")
    decay_max_drawdown: float = float("nan")
    turnover_ratio: float = float("nan")

    decay_se: float = float("nan")
    decay_z: float = float("nan")
    decay_p: float = float("nan")

    psr_vs_zero: float = float("nan")
    psr_vs_benchmark: float = float("nan")
    psr_vs_research: float = float("nan")

    forward_band_low: float = float("nan")
    forward_band_high: float = float("nan")

    checks: list[Check] = field(default_factory=list)
    verdict: str = "inconclusive"
    verdict_reason: str = ""

    # ------------------------------------------------------------- reading

    def check(self, name: str) -> Check | None:
        return next((c for c in self.checks if c.name == name), None)

    def passed(self, name: str) -> bool | None:
        c = self.check(name)
        return c.passed if c else None

    def as_flat_dict(self) -> dict:
        """Everything as scalars, ready to be a row in the forward log."""
        out = {f"decay_{k}": v for k, v in {
            "cagr": self.decay_cagr, "sharpe": self.decay_sharpe,
            "sharpe_monthly": self.decay_sharpe_monthly,
            "d_sharpe": self.decay_d_sharpe,
            "max_drawdown": self.decay_max_drawdown,
            "se": self.decay_se, "z": self.decay_z, "p": self.decay_p,
        }.items()}
        out |= {
            "turnover_ratio": self.turnover_ratio,
            "psr_vs_zero": self.psr_vs_zero,
            "psr_vs_benchmark": self.psr_vs_benchmark,
            "psr_vs_research": self.psr_vs_research,
            "forward_band_low": self.forward_band_low,
            "forward_band_high": self.forward_band_high,
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "checks": {c.name: {"passed": c.passed, "detail": c.detail}
                       for c in self.checks},
        }
        return out

    def summary(self) -> str:
        f, r = self.forward, self.research
        lines = [
            f"  {'':22s} {'research':>12s} {'forward':>12s} {'change':>12s}",
            f"  {'window':22s} {r.start[:7]:>12s} {f.start[:7]:>12s}",
            f"  {'':22s} {r.end[:7]:>12s} {f.end[:7]:>12s}",
            f"  {'months':22s} {r.n_months:12d} {f.n_months:12d}",
            _row("CAGR", r.cagr, f.cagr, pct=True),
            _row("Sharpe (daily)", r.sharpe, f.sharpe),
            _row("Sharpe (monthly)", r.sharpe_monthly, f.sharpe_monthly),
            _row("vs benchmark CAGR", r.excess, f.excess, pct=True),
            _row("vs benchmark Sharpe", r.d_sharpe, f.d_sharpe),
            _row("max drawdown", r.max_drawdown, f.max_drawdown, pct=True),
            _row("turnover", r.ann_turnover, f.ann_turnover, pct=True),
            _row("cost drag", r.cost_drag, f.cost_drag, pct=True),
            _row("names held", r.avg_positions, f.avg_positions),
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# The comparison itself
# --------------------------------------------------------------------------

def compare(research: Leg, forward: Leg,
            min_forward_months: int = MIN_FORWARD_MONTHS) -> Comparison:
    """Everything the two legs say together. Pure: same inputs, same output, always."""
    c = Comparison(research=research, forward=forward)

    c.decay_cagr = forward.cagr - research.cagr
    c.decay_sharpe = forward.sharpe - research.sharpe
    c.decay_sharpe_monthly = forward.sharpe_monthly - research.sharpe_monthly
    c.decay_d_sharpe = forward.d_sharpe - research.d_sharpe
    c.decay_max_drawdown = forward.max_drawdown - research.max_drawdown
    c.turnover_ratio = _ratio(forward.ann_turnover, research.ann_turnover)

    # Two independent estimates, so the variances add. Both are on the ANNUALISED
    # monthly Sharpe, which is the only Sharpe in this project whose observations are
    # close enough to independent for a standard error to mean anything.
    se_r, se_f = research.sharpe_se, forward.sharpe_se
    if math.isfinite(se_r) and math.isfinite(se_f):
        c.decay_se = math.sqrt(se_r ** 2 + se_f ** 2)
        if c.decay_se > 0 and math.isfinite(c.decay_sharpe_monthly):
            c.decay_z = c.decay_sharpe_monthly / c.decay_se
            # One-sided: the probability of a drop AT LEAST this large under the null
            # that the two windows share one true Sharpe.
            c.decay_p = metrics.norm_cdf(c.decay_z)

    c.psr_vs_zero = _psr(forward, benchmark_annualised=0.0)
    c.psr_vs_benchmark = _psr(forward, benchmark_annualised=forward.bench_sharpe)
    c.psr_vs_research = _psr(forward, benchmark_annualised=research.sharpe_monthly)
    c.forward_band_low, c.forward_band_high = forward.band()

    c.checks = _checks(research, forward, c, min_forward_months)
    c.verdict, c.verdict_reason = _verdict(c, min_forward_months)
    return c


def _checks(r: Leg, f: Leg, c: Comparison, min_months: int) -> list[Check]:
    """The named conditions, each independently readable and independently testable."""
    out = [
        Check("enough_data", f.n_months >= min_months,
              f"{f.n_months} months of forward data "
              f"({'at least' if f.n_months >= min_months else 'fewer than'} "
              f"the {min_months} needed for any verdict)"),
        Check("no_ruin", not f.ruined,
              "the portfolio was wiped out" if f.ruined else "NAV never reached zero"),
        Check("made_money", _cmp(f.cagr, 0.0),
              f"forward CAGR {_pct(f.cagr)}"),
        Check("beat_benchmark", _cmp(f.d_sharpe, 0.0),
              f"forward Sharpe {_num(f.sharpe)} against the index's "
              f"{_num(f.bench_sharpe)} over the same dates "
              f"({_num(f.d_sharpe, sign=True)})"),
        Check("positive_excess", _cmp(f.excess, 0.0),
              f"forward CAGR {_pct(f.cagr)} against the index's {_pct(f.bench_cagr)} "
              f"({_pct(f.excess, sign=True)})"),
        Check("kept_its_edge", _kept_edge(r, f), _edge_detail(r, f)),
        Check("decay_within_noise", _within_noise(c.decay_z),
              _decay_detail(c)),
        Check("drawdown_held", _cmp(f.max_drawdown, r.max_drawdown, or_equal=True),
              f"worst drawdown {_pct(f.max_drawdown)} forward against "
              f"{_pct(r.max_drawdown)} in research"),
        Check("turnover_held", _cmp(TURNOVER_DRIFT_LIMIT, c.turnover_ratio,
                                    or_equal=True),
              f"traded {_ratio_text(c.turnover_ratio)} as much as in research "
              f"({_pct(f.ann_turnover, digits=0)}/yr against "
              f"{_pct(r.ann_turnover, digits=0)}/yr)"),
    ]
    return out


def _verdict(c: Comparison, min_months: int) -> tuple[str, str]:
    """One word and the sentence behind it. Precedence is explicit and ordered.

    Read the vocabulary literally:

    ``inconclusive``  the window is too short to say anything
    ``failed``        the forward window contradicts the research claim, detectably
    ``decayed``       it still works, but by less than research said, beyond noise
    ``held``          nothing here refutes it - which is NOT the same as confirming it
    """
    f = c.forward

    if c.passed("enough_data") is False:
        return "inconclusive", (
            f"only {f.n_months} months of forward data; below {min_months} months "
            "no comparison can separate skill from sampling error, so no verdict "
            "is offered rather than a weak one.")

    if c.passed("no_ruin") is False:
        return "failed", ("the portfolio reached zero NAV in the forward window. "
                          "Nothing else about the run matters.")

    if c.passed("made_money") is False:
        return "failed", (f"the strategy lost money out of sample: CAGR "
                          f"{_pct(f.cagr)} over {f.n_months} months. A research "
                          "window that said otherwise was fitting something that "
                          "was not there.")

    if _fatal_decay(c.decay_z):
        return "failed", (
            f"the Sharpe fell {abs(c.decay_sharpe_monthly):.2f} against a standard "
            f"error of {c.decay_se:.2f} on the difference ({c.decay_z:.1f} sigma, "
            f"p={c.decay_p:.3f}). A drop that large is not sampling noise on a window "
            "this short - it takes a real change to produce one.")

    if c.passed("kept_its_edge") is False:
        return "decayed", (
            "it beat the index over the research window and did not over the forward "
            f"one ({_num(c.research.d_sharpe, sign=True)} then "
            f"{_num(f.d_sharpe, sign=True)} Sharpe against the index). It still made "
            "money; it no longer made it better than buying the index.")

    if c.passed("decay_within_noise") is False:
        return "decayed", (
            f"the Sharpe fell {abs(c.decay_sharpe_monthly):.2f} against a standard "
            f"error of {c.decay_se:.2f} ({c.decay_z:.1f} sigma). That is more than "
            "noise comfortably explains, and less than this window can call a "
            "refutation.")

    return "held", (
        f"nothing in {f.n_months} months of out-of-sample data refutes the research "
        f"claim: Sharpe {_num(f.sharpe_monthly)} against the {_num(c.research.sharpe_monthly)} "
        f"predicted, inside the {c.decay_se:.2f} standard error of the difference. "
        "Read that as 'not refuted', not as 'confirmed' - a window this short cannot "
        "confirm anything.")


# --------------------------------------------------------------------------
# Small pure helpers
# --------------------------------------------------------------------------

def _psr(leg: Leg, benchmark_annualised: float) -> float:
    """P(this leg's true monthly Sharpe exceeds a benchmark Sharpe), both annualised."""
    if leg.n_months < 6 or not math.isfinite(leg.sharpe_monthly):
        return float("nan")
    if not math.isfinite(benchmark_annualised):
        return float("nan")
    root = math.sqrt(MONTHS_PER_YEAR)
    return metrics.probabilistic_sharpe(
        leg.sharpe_monthly / root, leg.n_months, _finite(leg.skew_monthly, 0.0),
        _full_kurtosis(leg.kurtosis_monthly), benchmark_annualised / root)


def _annualised_se(sharpe_annualised: float, n_months: int, skew: float,
                   excess_kurtosis: float) -> float:
    if n_months < 3 or not math.isfinite(sharpe_annualised):
        return float("nan")
    root = math.sqrt(MONTHS_PER_YEAR)
    se = metrics.sharpe_standard_error(sharpe_annualised / root, n_months,
                                       _finite(skew, 0.0),
                                       _full_kurtosis(excess_kurtosis))
    return se * root


def _full_kurtosis(excess: float) -> float:
    """pandas reports EXCESS kurtosis; every formula in metrics.py wants the full one."""
    return _finite(excess, 0.0) + 3.0


def _finite(x: float, default: float) -> float:
    return float(x) if x is not None and math.isfinite(x) else default


def _ratio(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b) or b == 0:
        return float("nan")
    return a / b


def _cmp(a: float, b: float, or_equal: bool = False) -> bool | None:
    """a > b (or >=), or None when either side is not a number."""
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    return a >= b if or_equal else a > b


def _kept_edge(r: Leg, f: Leg) -> bool | None:
    """Did it still beat the index? Only meaningful if it beat the index in research."""
    if not math.isfinite(r.d_sharpe) or not math.isfinite(f.d_sharpe):
        return None
    if r.d_sharpe <= 0:
        return None                # it never had an edge to keep; not a failure here
    return f.d_sharpe > 0


def _within_noise(z: float) -> bool | None:
    if not math.isfinite(z):
        return None
    return z >= -DECAY_NOISE_SIGMA


def _fatal_decay(z: float) -> bool:
    return math.isfinite(z) and z <= -DECAY_FATAL_SIGMA


def _edge_detail(r: Leg, f: Leg) -> str:
    """Says what abstention MEANS, because 'n/a' next to two numbers reads as a failure."""
    if not math.isfinite(r.d_sharpe) or not math.isfinite(f.d_sharpe):
        return "no benchmark over one of the two windows, so there is nothing to compare"
    if r.d_sharpe <= 0:
        return (f"not applicable - it did not beat the index in research either "
                f"({_num(r.d_sharpe, sign=True)} Sharpe then, "
                f"{_num(f.d_sharpe, sign=True)} now), so there was no edge to keep")
    return (f"beat the index in research by {_num(r.d_sharpe, sign=True)} Sharpe "
            f"and forward by {_num(f.d_sharpe, sign=True)}")


def _decay_detail(c: Comparison) -> str:
    if not math.isfinite(c.decay_z):
        return "not computable - one of the two windows has too few months"
    return (f"monthly Sharpe moved {c.decay_sharpe_monthly:+.2f} against a "
            f"{c.decay_se:.2f} standard error on the difference ({c.decay_z:+.1f} sigma)")


def _row(label: str, a: float, b: float, pct: bool = False) -> str:
    fmt = _pct if pct else _num
    delta = b - a if math.isfinite(a) and math.isfinite(b) else float("nan")
    return f"  {label:22s} {fmt(a):>12s} {fmt(b):>12s} {fmt(delta, sign=True):>12s}"


def _pct(x: float, sign: bool = False, digits: int = 2) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x * 100:{'+' if sign else ''}.{digits}f}%"


def _num(x: float, sign: bool = False) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x:{'+' if sign else ''}.2f}"


def _ratio_text(x: float) -> str:
    return "n/a" if not math.isfinite(x) else f"{x:.2f}x"
