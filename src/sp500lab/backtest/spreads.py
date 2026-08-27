"""Bid-ask spread estimated from daily OHLC, because we have no quote data.

The problem
-----------
Half the cost of a monthly rebalance is the spread you cross, and quote data costs more
than this entire project's budget. Omitting the spread is not neutral - it is a
systematic overstatement of every strategy's return, and it grows with turnover, so it
flatters exactly the strategies that deserve the most scepticism.

The estimate here is a floor combined with an estimator, and BOTH parts are load-bearing.

Part 1: Corwin & Schultz (2012), Journal of Finance
----------------------------------------------------
"A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices."

A day's high-low range reflects both volatility and the spread, but volatility scales
with the length of the interval while the spread does not. Compare a two-day range
against two one-day ranges and the volatility term cancels, leaving the spread.

    beta  = sum over two consecutive days of ln(H/L)^2
    gamma = ln(H(2-day) / L(2-day))^2
    alpha = (sqrt(2*beta) - sqrt(beta)) / k - sqrt(gamma / k),  k = 3 - 2*sqrt(2)
    S     = 2 * (exp(alpha) - 1) / (1 + exp(alpha))

Two implementation details from the paper matter more than they look:

* **Overnight adjustment.** If a day's low exceeds the previous close the price gapped
  overnight, and the observed range then contains a move that did not happen during
  trading. Shift the range by the gap. Skipping this is the biggest single source of
  overestimation.
* **Averaging convention.** Two-day estimates are extremely noisy and roughly half come
  out negative for a liquid name. Truncating each negative to zero and THEN averaging
  gives E[max(X,0)], which for a true spread near zero is a pure positive bias of about
  0.4 standard deviations. MEASURED here: truncate-then-average reports 36bp for AAPL
  in 2018-19, where the real quoted spread was 1-2 cents on a $190 stock - about 1bp.
  So this module averages the SIGNED estimates and truncates the average, which is
  unbiased and correctly reports "indistinguishable from zero" for mega caps.

Part 2: the tick floor, which is what actually binds for large caps
--------------------------------------------------------------------
Corwin-Schultz was built for a market with 50-100bp spreads; its sample runs 1927-2006.
For an S&P 500 mega cap in 2018 the true spread is 1-2bp, which is more than an order
of magnitude below the estimator's resolution. It returns zero, and zero is closer to
right than 36bp - but it is still wrong, because a spread cannot be narrower than one
tick.

So the estimate is floored by the market's own quantum:

    half_spread >= MIN_SPREAD_TICKS * tick_size(date) / 2 / as_traded_price

`tick_size` is $0.01 after decimalisation (2001-04-09) and $0.0625 before it, which
makes the pre-2001 cost regime genuinely different rather than assumed away.
`MIN_SPREAD_TICKS = 2.0` says a name quotes at least two ticks wide on average - the
one modelling assumption in this file, stated rather than buried.

The as-traded price is required, not the stored close: our close is split-adjusted
(ADR-007), so a tick is a different fraction of it than of the price that traded. See
normalize/splits.py.

Net effect: the floor binds for liquid modern names (~0.3-1bp), and Corwin-Schultz
takes over where it can actually resolve - 2008-09, and the illiquid tail. That
division of labour is the design, not a fudge.

Abdi & Ranaldo (2017) is computed alongside as an independent cross-check. It uses the
close relative to the high-low midpoint, so it fails differently; agreement is evidence
and disagreement is a flag.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..normalize.splits import cumulative_split_ratio, load_splits, tick_size
from ..query import connect
from ..storage import write_gold

log = logging.getLogger(__name__)

_K = 3.0 - 2.0 * np.sqrt(2.0)

#: Trailing sessions averaged into the stored estimate. One month of trading - long
#: enough to tame the noise, short enough to track a liquidity regime change.
DEFAULT_SMOOTH_WINDOW = 21

#: The one modelling assumption here: a name quotes at least this many ticks wide.
#: 2.0 reproduces the ~0.5bp half-spread that mega caps actually showed post-2010.
MIN_SPREAD_TICKS = 2.0

#: Used where nothing can be computed at all (no price, no history). Reported by
#: costs.py whenever it is hit, never silently applied.
FALLBACK_HALF_SPREAD = 0.0005

#: Nothing above this is believable for an S&P 500 name; it means bad data, not a
#: 40% spread. Clipped and counted.
MAX_PLAUSIBLE_SPREAD = 0.20


def corwin_schultz(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """(D,S) SIGNED two-day proportional spread estimates, NaN where undefined.

    Signed on purpose - see "Averaging convention" in the module docstring. Truncate
    the AVERAGE, not the individual estimates.

    The estimate at row t uses sessions t-1 and t, so it is knowable at the close of t.
    """
    h = _positive(high)
    lo = _positive(low)
    c = _positive(close)

    prev_c = np.roll(c, 1, axis=0)
    prev_c[0] = np.nan

    # Overnight gap adjustment (Corwin & Schultz section I.B).
    with np.errstate(invalid="ignore"):
        gap_up = np.where(lo > prev_c, lo - prev_c, 0.0)
        gap_dn = np.where(h < prev_c, h - prev_c, 0.0)
    shift = np.nan_to_num(gap_up) + np.nan_to_num(gap_dn)
    h_adj, l_adj = h - shift, lo - shift

    prev_h, prev_l = np.roll(h_adj, 1, axis=0), np.roll(l_adj, 1, axis=0)
    prev_h[0] = prev_l[0] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.log(h_adj / l_adj) ** 2 + np.log(prev_h / prev_l) ** 2
        gamma = np.log(np.fmax(h_adj, prev_h) / np.fmin(l_adj, prev_l)) ** 2
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
        spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

    return np.where(np.isfinite(spread), spread, np.nan)


def abdi_ranaldo(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 window: int = DEFAULT_SMOOTH_WINDOW) -> np.ndarray:
    """(D,S) proportional spread, Abdi & Ranaldo (2017). Trailing `window` mean.

    S^2 = 4 * E[(c_t - eta_t) * (c_t - eta_{t+1})], eta = midpoint of log high/low.

    A cross-check rather than the primary: it needs a forward midpoint, so the value
    at row t is only knowable at t+1. The rolling mean is shifted one session to keep
    that honest.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = (np.log(_positive(high)) + np.log(_positive(low))) / 2.0
        c = np.log(_positive(close))
    eta_next = np.roll(eta, -1, axis=0)
    eta_next[-1] = np.nan
    prod = (c - eta) * (c - eta_next)
    rolled = (pd.DataFrame(prod).shift(1)
              .rolling(window, min_periods=max(3, window // 2)).mean().to_numpy())
    return 2.0 * np.sqrt(np.maximum(rolled, 0.0))


def tick_floor(dates: np.ndarray, as_traded_price: np.ndarray) -> np.ndarray:
    """(D,S) the narrowest half-spread the tick grid physically permits."""
    tick = tick_size(dates)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        return MIN_SPREAD_TICKS * tick / 2.0 / as_traded_price


def build(smooth_window: int = DEFAULT_SMOOTH_WINDOW, write: bool = True) -> pd.DataFrame:
    """Estimate half-spreads for every security-session and write them to gold.

    Output `gold/backtest/half_spread`:

        security_id, date, half_spread, half_spread_cs, half_spread_ar,
        tick_floor, binding

    `half_spread` = max(Corwin-Schultz trailing mean, tick floor), halved. `binding`
    says which of the two produced it, so the split of labour is inspectable rather
    than assumed.
    """
    con = connect()
    log.info("spreads: reading OHLC")
    bars = con.execute("""
        SELECT security_id, date, adj_high AS high, adj_low AS low,
               adj_close AS close, close AS raw_close
        FROM daily_bars_adjusted
        WHERE adj_high > 0 AND adj_low > 0 AND adj_close > 0
        ORDER BY date
    """).df()
    if bars.empty:
        raise RuntimeError("no adjusted bars; run `sp500lab normalize` first")

    dates = np.sort(bars["date"].unique())
    sids = np.sort(bars["security_id"].unique())
    date_pos = {d: i for i, d in enumerate(dates.tolist())}
    sid_pos = {s: i for i, s in enumerate(sids.tolist())}
    row = bars["date"].map(date_pos).to_numpy(dtype=np.int64)
    col = bars["security_id"].map(sid_pos).to_numpy(dtype=np.int64)

    def _mat(name: str) -> np.ndarray:
        m = np.full((len(dates), len(sids)), np.nan)
        m[row, col] = bars[name].to_numpy(dtype=np.float64)
        return m

    high, low, close, raw_close = _mat("high"), _mat("low"), _mat("close"), _mat("raw_close")
    log.info("spreads: %d sessions x %d securities", len(dates), len(sids))

    # Average the SIGNED estimates, then truncate. See the module docstring.
    cs_raw = corwin_schultz(high, low, close)
    cs = np.maximum(
        pd.DataFrame(cs_raw).rolling(smooth_window,
                                     min_periods=max(5, smooth_window // 3))
        .mean().to_numpy(), 0.0)
    ar = abdi_ranaldo(high, low, close, window=smooth_window)

    as_traded = raw_close * cumulative_split_ratio(dates, sid_pos, load_splits(con))
    floor = tick_floor(dates, as_traded)

    cs_half = cs / 2.0
    combined = np.fmax(np.nan_to_num(cs_half, nan=0.0), floor)
    binding = np.where(np.nan_to_num(cs_half, nan=0.0) > floor, "estimator", "tick_floor")

    n_clipped = int(np.nansum(combined > MAX_PLAUSIBLE_SPREAD))
    if n_clipped:
        log.warning("spreads: clipped %d estimates above %.0f%%",
                    n_clipped, MAX_PLAUSIBLE_SPREAD * 100)
    combined = np.clip(combined, 0.0, MAX_PLAUSIBLE_SPREAD)

    valid = np.isfinite(combined) & np.isfinite(as_traded)
    r, c_ = np.nonzero(valid)
    out = pd.DataFrame({
        "security_id": sids[c_],
        "date": dates[r],
        "half_spread": combined[r, c_],
        "half_spread_cs": cs_half[r, c_],
        "half_spread_ar": np.where(np.isfinite(ar[r, c_]), ar[r, c_] / 2.0, np.nan),
        "tick_floor": floor[r, c_],
        "binding": binding[r, c_],
    })
    share = float((out["binding"] == "tick_floor").mean())
    log.info("spreads: %d estimates, median half-spread %.2f bp, "
             "tick floor binds %.0f%% of the time",
             len(out), float(out["half_spread"].median() * 1e4), share * 100)

    if write:
        write_gold(out, "backtest/half_spread")
    return out


def summarise(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Median half-spread in bp by era, with the estimator/floor split.

    This is the sanity check the module docstring promises. Expect single-digit bp for
    the modern era and a visible widening in 2008-09. If a mega cap shows 30bp, the
    averaging convention has regressed.
    """
    if df is None:
        df = connect().execute("SELECT * FROM gold_half_spread").df()
    d = df.copy()
    d["year"] = d["date"].str.slice(0, 4).astype(int)
    d["era"] = pd.cut(d["year"], [1999, 2001, 2007, 2009, 2015, 2100],
                      labels=["2000-2001", "2002-2007", "2008-2009",
                              "2010-2015", "2016+"])
    return (d.groupby("era", observed=True)
            .agg(n=("half_spread", "size"),
                 median_bp=("half_spread", lambda s: round(float(s.median() * 1e4), 2)),
                 p90_bp=("half_spread", lambda s: round(float(s.quantile(0.9) * 1e4), 2)),
                 cs_median_bp=("half_spread_cs",
                               lambda s: round(float(s.median() * 1e4), 2)),
                 ar_median_bp=("half_spread_ar",
                               lambda s: round(float(s.median() * 1e4), 2)),
                 floor_binds=("binding", lambda s: f"{(s == 'tick_floor').mean():.0%}"))
            .reset_index())


def _positive(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64)
    return np.where(np.isfinite(x) & (x > 0), x, np.nan)
