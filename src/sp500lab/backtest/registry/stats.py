"""Statistics of one equity curve. No I/O, no registry, no paths.

Separated out because both `store.log()` (which records these at log time) and
`deflation` (which consumes them) need them, and neither should have to import the
other to get there.

Why monthly rather than daily
------------------------------
The headline Sharpe is computed from the daily equity curve. The deflated Sharpe cannot
use it: its derivation assumes approximately independent observations, and the daily
returns of a monthly-rebalanced portfolio are strongly autocorrelated within each
holding period. Using 4,861 daily observations where there are really ~176 independent
monthly ones would overstate the precision of every Sharpe in the registry and make the
deflation far too generous. Read the daily Sharpe; deflate the monthly one.
"""

from __future__ import annotations

import pandas as pd

from .. import metrics


def to_monthly(series: pd.Series) -> pd.Series | None:
    """Month-end samples of a daily curve, indexed by 'YYYY-MM-DD' strings.

    Public for the same reason as `append_jsonl`: the forward store writes curves too,
    and both must land on the same date grid or a stitched research-to-forward chart
    would have two different month ends.
    """
    s = series.dropna().astype(float)
    if len(s) < 2:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(s.index))))
    m = pd.Series(s.to_numpy(), index=idx).resample("ME").last().dropna()
    m.index = m.index.strftime("%Y-%m-%d")
    return m

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
