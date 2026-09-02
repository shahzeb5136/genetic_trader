"""Performance statistics, including the ones that survive a search.

Most of this file is ordinary: CAGR, volatility, Sharpe, drawdown. The part that
matters for this project is at the bottom.

Why the deflated Sharpe ratio is not optional here
--------------------------------------------------
A genetic algorithm explicitly optimises the metric you report. Run 10,000 individuals
against 19 years of monthly data and the best one will have a beautiful Sharpe ratio
whether or not there is any signal in the data at all - that is what a maximum over
10,000 draws does. The reported Sharpe of the winner is not an estimate of its skill;
it is an estimate of the skill PLUS the selection effect, and the selection effect
grows with the number of trials.

The deflated Sharpe ratio (Bailey & Lopez de Prado, 2014) is the correction: it asks
whether the winner's Sharpe exceeds what the best of N random strategies would have
achieved anyway. It needs the trial count as an input, which is why the GA must log
every individual it evaluates and not just the winners. Without the trial count the
reported Sharpe is not conservative or optimistic - it is meaningless.

It also accounts for non-normality. Financial returns are skewed and fat-tailed, and
both inflate the naive Sharpe's precision.

No scipy dependency
-------------------
The normal CDF and its inverse are implemented here from standard rational
approximations, accurate to ~1e-9 - far beyond what any of these statistics justify.
The project runs on pandas, numpy, duckdb and pyarrow, and it stays that way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS = 12
EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------
# Normal distribution helpers (no scipy)
# --------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, |error| < 1.15e-9)."""
    if not 0.0 < p < 1.0:
        return -math.inf if p <= 0.0 else math.inf
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# --------------------------------------------------------------------------
# Core statistics
# --------------------------------------------------------------------------

@dataclass
class Performance:
    """A backtest's summary statistics. `as_dict()` for JSON, `summary()` to read."""

    start: str
    end: str
    years: float
    n_periods: int
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_start: str
    max_drawdown_end: str
    calmar: float
    hit_rate: float
    skew: float
    kurtosis: float
    best_period: float
    worst_period: float
    time_under_water: float
    var_95: float
    cvar_95: float
    beta: float | None = None
    alpha: float | None = None
    tracking_error: float | None = None
    information_ratio: float | None = None
    ann_turnover: float | None = None
    avg_positions: float | None = None
    cost_drag: float | None = None

    def as_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in vars(self).items()}

    def summary(self) -> str:
        def pct(x):
            return "     n/a" if x is None else f"{x * 100:7.2f}%"

        def num(x):
            return "     n/a" if x is None else f"{x:8.2f}"

        lines = [
            f"  period            {self.start} .. {self.end}  ({self.years:.1f}y, "
            f"{self.n_periods} periods)",
            f"  total return     {pct(self.total_return)}",
            f"  CAGR             {pct(self.cagr)}",
            f"  volatility       {pct(self.ann_vol)}",
            f"  Sharpe           {num(self.sharpe)}",
            f"  Sortino          {num(self.sortino)}",
            f"  max drawdown     {pct(self.max_drawdown)}   "
            f"({self.max_drawdown_start} .. {self.max_drawdown_end})",
            f"  Calmar           {num(self.calmar)}",
            f"  hit rate         {pct(self.hit_rate)}",
            f"  skew / kurtosis  {num(self.skew)} /{num(self.kurtosis)}",
            f"  VaR / CVaR 95    {pct(self.var_95)} /{pct(self.cvar_95)}",
            f"  time under water {pct(self.time_under_water)}",
        ]
        if self.beta is not None:
            lines += [
                f"  beta             {num(self.beta)}",
                f"  alpha (ann)      {pct(self.alpha)}",
                f"  tracking error   {pct(self.tracking_error)}",
                f"  info ratio       {num(self.information_ratio)}",
            ]
        if self.ann_turnover is not None:
            lines += [
                f"  turnover (ann)   {pct(self.ann_turnover)}",
                f"  avg positions    {num(self.avg_positions)}",
            ]
        if self.cost_drag is not None:
            lines.append(f"  cost drag (ann)  {pct(self.cost_drag)}")
        return "\n".join(lines)


def _periods_per_year(index: pd.Index) -> float:
    """Infer observation frequency from the index, so daily and monthly both work."""
    if len(index) < 3:
        return float(TRADING_DAYS)
    d = pd.to_datetime(pd.Series(list(index)))
    med = float(d.diff().dt.days.median())
    if med <= 0 or not np.isfinite(med):
        return float(TRADING_DAYS)
    if med <= 4:
        return float(TRADING_DAYS)
    if med <= 10:
        return 52.0
    if med <= 45:
        return float(MONTHS)
    return 4.0 if med <= 120 else 1.0


def compute(
    equity: pd.Series,
    *,
    benchmark: pd.Series | None = None,
    risk_free: float = 0.0,
    turnover: pd.Series | None = None,
    positions: pd.Series | None = None,
    gross_equity: pd.Series | None = None,
) -> Performance:
    """Statistics from an equity curve indexed by date string.

    `benchmark` is an equity curve on the same index. `gross_equity` is the same
    strategy run with zero costs; the difference is the cost drag, which is the number
    that decides whether a high-turnover strategy is real.
    """
    eq = equity.dropna().astype(float)
    if len(eq) < 3:
        raise ValueError("need at least 3 observations to compute performance")
    ret = eq.pct_change().dropna()
    ppy = _periods_per_year(eq.index)

    days = (pd.Timestamp(str(eq.index[-1])) - pd.Timestamp(str(eq.index[0]))).days
    years = max(days / 365.25, 1e-9)

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0)
    vol = float(ret.std(ddof=1) * math.sqrt(ppy))

    excess = ret - risk_free / ppy
    sharpe = float(excess.mean() / ret.std(ddof=1) * math.sqrt(ppy)) if ret.std(ddof=1) > 0 else 0.0

    downside = ret[ret < 0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(excess.mean() / dstd * math.sqrt(ppy)) if dstd > 0 else float("inf")

    dd, dd_start, dd_end, tuw = _drawdown(eq)
    calmar = float(cagr / abs(dd)) if dd < 0 else float("inf")

    perf = Performance(
        start=str(eq.index[0]), end=str(eq.index[-1]), years=round(years, 3),
        n_periods=len(ret),
        total_return=total_return, cagr=cagr, ann_vol=vol,
        sharpe=sharpe, sortino=sortino,
        max_drawdown=dd, max_drawdown_start=dd_start, max_drawdown_end=dd_end,
        calmar=calmar,
        hit_rate=float((ret > 0).mean()),
        skew=float(ret.skew()), kurtosis=float(ret.kurtosis()),
        best_period=float(ret.max()), worst_period=float(ret.min()),
        time_under_water=tuw,
        var_95=float(ret.quantile(0.05)),
        cvar_95=float(ret[ret <= ret.quantile(0.05)].mean()) if len(ret) > 20 else float(ret.min()),
    )

    if benchmark is not None:
        bench = benchmark.reindex(eq.index).dropna()
        common = ret.index.intersection(bench.pct_change().dropna().index)
        if len(common) > 2:
            r, b = ret.loc[common], bench.pct_change().dropna().loc[common]
            var_b = float(b.var(ddof=1))
            beta = float(np.cov(r, b, ddof=1)[0, 1] / var_b) if var_b > 0 else 0.0
            perf.beta = beta
            perf.alpha = float((r.mean() - beta * b.mean()) * ppy)
            active = r - b
            te = float(active.std(ddof=1) * math.sqrt(ppy))
            perf.tracking_error = te
            perf.information_ratio = float(active.mean() * ppy / te) if te > 0 else 0.0

    if turnover is not None and len(turnover):
        t = turnover.dropna()
        perf.ann_turnover = float(t.sum() / years)
    if positions is not None and len(positions):
        perf.avg_positions = float(positions.mean())
    if gross_equity is not None and len(gross_equity) > 2:
        g = gross_equity.dropna().astype(float)
        gross_cagr = float((g.iloc[-1] / g.iloc[0]) ** (1.0 / years) - 1.0)
        perf.cost_drag = gross_cagr - cagr

    return perf


def _drawdown(eq: pd.Series) -> tuple[float, str, str, float]:
    peak = eq.cummax()
    dd = eq / peak - 1.0
    end = dd.idxmin()
    trough = float(dd.min())
    prior = eq.loc[:end]
    start = prior.idxmax() if len(prior) else eq.index[0]
    return trough, str(start), str(end), float((dd < -1e-12).mean())


def drawdown_series(equity: pd.Series) -> pd.Series:
    eq = equity.dropna().astype(float)
    return eq / eq.cummax() - 1.0


def monthly_table(equity: pd.Series) -> pd.DataFrame:
    """Calendar month returns as a year x month grid, with a YTD column."""
    eq = equity.dropna().astype(float)
    idx = pd.to_datetime(pd.Series(list(eq.index)))
    s = pd.Series(eq.to_numpy(), index=pd.DatetimeIndex(idx))
    m = s.resample("ME").last().pct_change().dropna()
    if m.empty:
        return pd.DataFrame()
    tab = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.to_numpy()})
    grid = tab.pivot(index="year", columns="month", values="ret")
    grid.columns = [pd.Timestamp(2000, c, 1).strftime("%b") for c in grid.columns]
    grid["YTD"] = (1 + grid.fillna(0)).prod(axis=1) - 1
    return grid


def annual_returns(equity: pd.Series) -> pd.Series:
    eq = equity.dropna().astype(float)
    s = pd.Series(eq.to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(pd.Series(list(eq.index)))))
    return s.resample("YE").last().pct_change().dropna()


# --------------------------------------------------------------------------
# Sampling error - how much a Sharpe measured over N observations is worth
# --------------------------------------------------------------------------
#
# The multiple-testing correction below asks "could the best of N searches have got
# this by luck". These two ask the prior question: "could ONE measurement have got
# this by luck", which is what decides whether a short out-of-sample window can
# distinguish anything at all. It usually cannot, and the number is worth seeing.

def sharpe_standard_error(sharpe: float, n_obs: int, skew: float = 0.0,
                          kurtosis: float = 3.0) -> float:
    """Standard error of a Sharpe ratio, corrected for skew and fat tails.

    Mertens (2002); Lo (2002) is the normal special case. Units follow the rest of
    this section: `sharpe` is PER-OBSERVATION, not annualised, and `kurtosis` is the
    FULL kurtosis (3.0 for a normal), not the excess kurtosis pandas reports.

    The returned SE is in the same per-observation units, so annualise it the same way
    you annualise the Sharpe itself - multiply by sqrt(periods per year).

    Why it shares its algebra with `probabilistic_sharpe`
    ----------------------------------------------------
    The PSR is exactly `norm_cdf((SR - SR0) / SE)` with this SE, and the function below
    is written that way so the two can never drift apart. A confidence interval that
    disagreed with the probability printed next to it would be worse than having
    neither.

    Assumes serially uncorrelated returns. Month-end returns of a monthly-rebalanced
    portfolio are close enough; daily returns of one are not, which is the same reason
    the registry deflates on monthly statistics (ADR-026).
    """
    if n_obs < 3:
        return float("nan")
    variance = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe ** 2
    if variance <= 0:
        return float("nan")
    return math.sqrt(variance / (n_obs - 1))


def sharpe_confidence_interval(sharpe: float, n_obs: int, skew: float = 0.0,
                               kurtosis: float = 3.0,
                               confidence: float = 0.95) -> tuple[float, float]:
    """(low, high) interval for a Sharpe, in whatever units `sharpe` was given in.

    Wide. That is the point of computing it. Fifty-six months of monthly data put a
    +-0.9 band around an annualised Sharpe of 1.0, which means a forward test over that
    span cannot tell 0.2 from 1.8 - and any verdict drawn from one has to be read
    against that, not against the point estimate.
    """
    se = sharpe_standard_error(sharpe, n_obs, skew, kurtosis)
    if not math.isfinite(se):
        return float("nan"), float("nan")
    z = norm_ppf(0.5 + confidence / 2.0)
    return sharpe - z * se, sharpe + z * se


# --------------------------------------------------------------------------
# Multiple-testing control - the part that matters for a GA
# --------------------------------------------------------------------------

def probabilistic_sharpe(sharpe: float, n_obs: int, skew: float, kurtosis: float,
                         benchmark_sharpe: float = 0.0) -> float:
    """P(true Sharpe > benchmark_sharpe), correcting for skew and fat tails.

    `sharpe` and `benchmark_sharpe` must be per-observation, NOT annualised - divide
    an annual figure by sqrt(periods per year) first. `kurtosis` is the FULL kurtosis
    (3.0 for a normal), not the excess kurtosis pandas reports; convert with +3.
    """
    se = sharpe_standard_error(sharpe, n_obs, skew, kurtosis)
    if not math.isfinite(se) or se <= 0:
        return float("nan")
    return norm_cdf((sharpe - benchmark_sharpe) / se)


def expected_max_sharpe(n_trials: int, trial_sharpe_std: float) -> float:
    """The Sharpe the LUCKIEST of `n_trials` worthless strategies would post.

    This is the bar a searched strategy has to clear. It rises with the number of
    trials and with how much the trials differ from one another, which is why an
    unconstrained GA over a wide search space is so dangerous: it maximises both.
    """
    if n_trials < 2 or trial_sharpe_std <= 0:
        return 0.0
    n = float(n_trials)
    return trial_sharpe_std * ((1 - EULER_GAMMA) * norm_ppf(1 - 1 / n)
                               + EULER_GAMMA * norm_ppf(1 - 1 / (n * math.e)))


def deflated_sharpe(sharpe: float, n_obs: int, skew: float, kurtosis: float,
                    n_trials: int, trial_sharpe_std: float) -> float:
    """Probability the strategy's true Sharpe is positive, given the search that found it.

    Bailey & Lopez de Prado (2014). Same units convention as probabilistic_sharpe:
    per-observation Sharpe, full kurtosis.

    Read it as a probability, not a score. Below ~0.95 the result is not distinguishable
    from the best of N lucky draws, however good the raw Sharpe looks.

    `trial_sharpe_std` is the standard deviation of the Sharpe ratios across ALL
    individuals the search evaluated - which is why they must all be logged, winners
    and losers alike.
    """
    sr0 = expected_max_sharpe(n_trials, trial_sharpe_std)
    return probabilistic_sharpe(sharpe, n_obs, skew, kurtosis, benchmark_sharpe=sr0)


def deflate_result(perf: Performance, n_trials: int, trial_sharpe_std: float,
                   periods_per_year: float = MONTHS) -> dict:
    """Convenience wrapper: annualised Performance in, DSR/PSR out."""
    sr_obs = perf.sharpe / math.sqrt(periods_per_year)
    full_kurt = perf.kurtosis + 3.0
    return {
        "sharpe_annualised": round(perf.sharpe, 4),
        "sharpe_per_period": round(sr_obs, 4),
        "n_obs": perf.n_periods,
        "n_trials": n_trials,
        "expected_max_sharpe_per_period": round(
            expected_max_sharpe(n_trials, trial_sharpe_std), 4),
        "psr_vs_zero": round(
            probabilistic_sharpe(sr_obs, perf.n_periods, perf.skew, full_kurt), 4),
        "deflated_sharpe": round(
            deflated_sharpe(sr_obs, perf.n_periods, perf.skew, full_kurt,
                            n_trials, trial_sharpe_std), 4),
    }
