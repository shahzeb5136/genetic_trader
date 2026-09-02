"""Price, volume and liquidity features. Everything here comes from the panel alone.

Two rules govern this file and nothing else matters as much:

1. **Every window is trailing.** `rolling(252)` in pandas ends at the current row, and
   `shift(k)` moves a value FORWARD in time so the current row reads a k-sessions-old
   value. Both are backward-looking. A `center=True` anywhere in this file, or a shift
   with a negative argument, is a lookahead bug that would show up as spectacular
   performance and nothing else.

2. **The market is the equal-weighted point-in-time index, not SPY.** Betas and residual
   momentum need a market return, and using SPY would introduce a cap-weighted series
   whose largest constituents dominate it. For a cross-sectional strategy choosing among
   index members, the equal-weighted cross-section IS the thing being chosen from, so a
   residual against it is the part of a stock's return its peers do not explain. It is
   also computed from the same survivorship-free membership mask as everything else,
   which SPY is not.

What is here, and why each earns its place
-------------------------------------------
Not a list of every indicator that exists. Each of these is a documented cross-sectional
effect with a different economic story, so that a search over them is choosing between
explanations rather than between parameterisations of one:

    mom_12_1, mom_6_1     under-reaction to news over 6-12 months (Jegadeesh & Titman)
    rev_1m                over-reaction at one month, which reverses
    resid_mom_12_1        the same momentum with market beta stripped out; Blitz, Huij &
                          Martens found it survives where raw momentum crashes
    info_discreteness     Da, Gurun & Warachka: the SAME 12-month return drifts further
                          when it arrived in many small pieces than in a few jumps.
                          A signal about HOW a return happened, not how big it was.
    high_52w_ratio        George & Hwang: nearness to the 52-week high predicts, and it
                          is not the same information as the return that produced it
    vol_126d, beta_252d   the low-risk anomaly, in its two usual forms
    idio_vol_252d         Ang et al.: idiosyncratic volatility is priced NEGATIVELY,
                          which no risk model predicts
    max_ret_21d           Bali, Cakici & Whitelaw: the lottery preference that probably
                          drives the one above
    skew_252d             the same preference measured a different way
    vol_of_vol_252d       uncertainty about risk, distinct from risk
    amihud_illiq          the illiquidity premium - and simultaneously the thing this
                          project's cost model will charge you for harvesting it
    half_spread           the cost model's own input, exposed as a feature so a strategy
                          can decline to trade what it cannot afford to trade
    log_dollar_volume     size, without needing shares outstanding
    trend_200d            time-series trend, the one signal here that is not
                          cross-sectional in origin
    corr_mkt_252d         how much of the name is market; a diversification signal
    ret_1m..ret_12m       raw building blocks, for a genome that wants to reweight them
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Sessions in a year and a month. Named so a reader can tell a lookback from a lag.
YEAR, MONTH = 252, 21


def compute(panel, rows: np.ndarray, *, data_cutoff: str | None = None) -> dict:
    """(R, S) matrices for every price feature, sampled at `rows`.

    `data_cutoff` is accepted and ignored: price features read the panel and nothing
    else, so truncating the panel is already the whole of the restriction. The parameter
    exists so every builder has one signature.
    """
    close = panel.adj_close.astype(np.float64)
    rows = np.asarray(rows, dtype=np.int64)

    rets = _returns(close)
    mkt = _market_return(rets, panel.in_index & panel.has_price)
    out: dict[str, np.ndarray] = {}

    # ---- momentum family: point-to-point ratios, no rolling machinery needed ----
    out["mom_12_1"] = _ratio(close, rows, MONTH, YEAR + MONTH)
    out["mom_6_1"] = _ratio(close, rows, MONTH, 6 * MONTH + MONTH)
    out["mom_1m"] = _ratio(close, rows, 0, MONTH)
    out["rev_1m"] = -out["mom_1m"]
    out["ret_3m"] = _ratio(close, rows, 0, 3 * MONTH)
    out["ret_12m"] = _ratio(close, rows, 0, YEAR)

    # ---- rolling statistics -----------------------------------------------------
    r = pd.DataFrame(rets)
    out["vol_21d"] = _take(r.rolling(MONTH, min_periods=15).std() * np.sqrt(YEAR), rows)
    vol126 = r.rolling(126, min_periods=63).std() * np.sqrt(YEAR)
    out["vol_126d"] = _take(vol126, rows)
    out["vol_of_vol_252d"] = _take(vol126.rolling(YEAR, min_periods=126).std(), rows)
    out["skew_252d"] = _take(r.rolling(YEAR, min_periods=126).skew(), rows)
    out["max_ret_21d"] = _take(r.rolling(MONTH, min_periods=10).max(), rows)

    c = pd.DataFrame(close)
    out["trend_200d"] = _take(c / c.rolling(200, min_periods=100).mean() - 1.0, rows)
    out["high_52w_ratio"] = _take(c / c.rolling(YEAR, min_periods=126).max(), rows)

    # ---- market-relative: beta, residual momentum, residual volatility ----------
    beta, corr = _rolling_beta(r, mkt, YEAR)
    out["beta_252d"] = _take(beta, rows)
    out["corr_mkt_252d"] = _take(corr, rows)

    mkt_s = pd.Series(mkt, index=r.index)
    resid = r - beta.mul(mkt_s, axis=0)
    resid_sd = resid.rolling(YEAR, min_periods=126).std()
    out["idio_vol_252d"] = _take(resid_sd * np.sqrt(YEAR), rows)
    # Residual momentum: the residual return accumulated over the 12-1 window, divided
    # by its own volatility over the same window. Standardising is what makes it
    # comparable across names whose residuals have wildly different scale - a high-beta
    # tech name and a utility do not produce residuals of the same size.
    cum_resid = resid.rolling(YEAR, min_periods=126).sum().shift(MONTH)
    out["resid_mom_12_1"] = _take(
        cum_resid / (resid_sd.shift(MONTH) * np.sqrt(YEAR)), rows)

    # ---- information discreteness (Da, Gurun & Warachka 2014) -------------------
    # sign(12-1 return) x (share of down days - share of up days) over the same window.
    # A stock that ground steadily upward has many small up days: %pos is high, so ID is
    # NEGATIVE and large in magnitude. One that gapped on two announcements has ID near
    # zero. The finding is that continuous information under-reacts more, so the drift is
    # stronger where ID is more negative.
    # `.where(r.notna())` matters: without it a session with no bar counts as neither
    # up nor down, which quietly deflates both shares for exactly the names whose
    # history is patchy - the delisted ones.
    valid = r.notna()
    pos = _take(r.gt(0).astype(np.float64).where(valid)
                .rolling(YEAR, min_periods=126).mean().shift(MONTH), rows)
    neg = _take(r.lt(0).astype(np.float64).where(valid)
                .rolling(YEAR, min_periods=126).mean().shift(MONTH), rows)
    out["info_discreteness"] = np.sign(out["mom_12_1"]) * (neg - pos)

    # ---- overnight / intraday decomposition (Lou, Polk & Skouras 2019) ----------
    # Every close-to-close return is two different trades glued together: the overnight
    # leg (yesterday's close to today's open, where earnings land and institutions
    # position) and the intraday leg (open to close, where retail flow and market-making
    # live). LPS's finding is that momentum lives almost entirely in the overnight
    # component, and that the two components PERSIST separately - which makes the
    # decomposition a feature, not a curiosity. Both series come from the same adjusted
    # panel, so the dividend lands in the overnight leg exactly as it does in the world:
    # the adjustment factor steps at the ex-date, between one close and the next open.
    on_ret, id_ret = _overnight_intraday(close, panel.adj_open.astype(np.float64))
    o = pd.DataFrame(on_ret)
    i = pd.DataFrame(id_ret)
    o_sum = o.rolling(YEAR, min_periods=126).sum()
    i_sum = i.rolling(YEAR, min_periods=126).sum()
    out["mom_on_12_1"] = np.expm1(_take(o_sum.shift(MONTH), rows))
    out["mom_id_12_1"] = np.expm1(_take(i_sum.shift(MONTH), rows))
    # The tug-of-war spread: trailing-year overnight minus intraday log return, no skip.
    # Positive means the name earns its keep while the market is closed.
    out["on_minus_id_252d"] = _take(o_sum - i_sum, rows)

    # ---- liquidity --------------------------------------------------------------
    dv = panel.dollar_volume.astype(np.float64)
    dv = np.where(np.isfinite(dv) & (dv > 0), dv, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        illiq = pd.DataFrame(np.abs(rets) / dv * 1e9)
        out["amihud_illiq"] = _take(
            np.log1p(illiq.rolling(63, min_periods=30).mean()), rows)
        out["log_dollar_volume"] = _take(pd.DataFrame(np.log(dv)), rows)

    hs = panel.half_spread.astype(np.float64)
    out["half_spread_bp"] = hs[rows] * 1e4

    _warn_on_empty(out)
    return out


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _returns(close: np.ndarray) -> np.ndarray:
    """(D, S) simple daily returns, NaN on the first row and across any price gap."""
    out = np.full_like(close, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = close[1:] / close[:-1] - 1.0
    # A 'return' spanning a gap in the price series is not a return, it is two prices
    # subtracted across an unknown interval. Leaving it in would put a 300% one-day move
    # into every volatility estimate that touches a resumed listing.
    out[~np.isfinite(out)] = np.nan
    return out


def _overnight_intraday(close: np.ndarray, open_: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """(D, S) log overnight and intraday returns, NaN where either bar is missing.

    Log returns rather than simple ones so that the two legs ADD back to the
    close-to-close log return where both exist - the decomposition is exact, and a
    rolling sum of each leg is a cumulative return rather than an approximation of one.

    Session t's overnight return runs from the close of t-1 to the open of t, so row 0
    has no overnight leg and is NaN. A tiny number of opens are backfilled from the
    close upstream (panel.meta['open_gaps_filled_from_close']); those sessions read as
    "the whole move happened overnight", which is the least-wrong statement available.
    """
    on = np.full_like(close, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        on[1:] = np.log(open_[1:] / close[:-1])
        intra = np.log(close / open_)
    on[~np.isfinite(on)] = np.nan
    intra[~np.isfinite(intra)] = np.nan
    return on, intra


def _market_return(rets: np.ndarray, member: np.ndarray) -> np.ndarray:
    """(D,) equal-weighted return of the point-in-time index members. See module docs."""
    r = np.where(member, rets, np.nan)
    with np.errstate(invalid="ignore"):
        counts = np.isfinite(r).sum(axis=1)
        total = np.nansum(r, axis=1)
    return np.where(counts >= 20, total / np.maximum(counts, 1), np.nan)


def _ratio(close: np.ndarray, rows: np.ndarray, skip: int, lookback: int) -> np.ndarray:
    """(R, S) total return from `lookback` sessions back to `skip` sessions back.

    `skip` is the classic momentum gap: 12-1 momentum measures the year up to one month
    ago, because the most recent month reverses rather than continuing.
    """
    end, start = rows - skip, rows - lookback
    out = np.full((len(rows), close.shape[1]), np.nan)
    ok = start >= 0
    if ok.any():
        with np.errstate(divide="ignore", invalid="ignore"):
            out[ok] = close[end[ok]] / close[start[ok]] - 1.0
    out[~np.isfinite(out)] = np.nan
    return out


def _rolling_beta(r: pd.DataFrame, mkt: np.ndarray, window: int
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trailing beta and correlation of each name against the market.

    Computed from rolling moments rather than a loop of regressions: 677 securities x
    4,900 sessions is 3.3 million regressions done the obvious way, and about six
    vectorised passes done this one.
    """
    m = pd.Series(mkt, index=r.index)
    minp = max(60, window // 4)
    mean_i = r.rolling(window, min_periods=minp).mean()
    mean_m = m.rolling(window, min_periods=minp).mean()
    mean_im = r.mul(m, axis=0).rolling(window, min_periods=minp).mean()
    var_m = m.pow(2).rolling(window, min_periods=minp).mean() - mean_m.pow(2)
    var_i = r.pow(2).rolling(window, min_periods=minp).mean() - mean_i.pow(2)

    cov = mean_im.sub(mean_i.mul(mean_m, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = cov.div(var_m.where(var_m > 0), axis=0)
        corr = cov.div(np.sqrt(var_i.clip(lower=0)).mul(np.sqrt(var_m.clip(lower=0)),
                                                        axis=0))
    return beta.replace([np.inf, -np.inf], np.nan), corr.clip(-1, 1)


def _take(df: pd.DataFrame | np.ndarray, rows: np.ndarray) -> np.ndarray:
    """(R, S) rows of a (D, S) frame, with infinities normalised to NaN.

    NaN rather than a fill value, everywhere: "no opinion" and "zero" are different
    statements, and portfolio.py already treats NaN as ineligible rather than neutral.
    """
    arr = df.to_numpy() if isinstance(df, pd.DataFrame) else np.asarray(df)
    out = arr[rows]
    return np.where(np.isfinite(out), out, np.nan)


def _warn_on_empty(out: dict) -> None:
    for name, mat in out.items():
        if not np.isfinite(mat).any():
            log.warning("feature %s is entirely NaN - check its inputs", name)
