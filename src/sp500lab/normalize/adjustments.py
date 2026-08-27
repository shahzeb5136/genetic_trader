"""Corporate-action adjustment factors, computed from prices + event records.

Why compute these ourselves
---------------------------
Vendor "adjusted close" columns are a moving target: every new dividend rewrites the
entire history of the series. A backtest run in January and re-run in June against
the same vendor gets different numbers with no code change, which makes results
irreproducible and obscures whether a strategy change actually helped.

This is not theoretical - it is measured in this repo. Comparing our factors against
yfinance's Adj Close over 2018-2024 gives a *constant* relative offset per ticker
(JNJ 4.7%, KO 4.3%, MMM 3.3%, NVDA 0.15%), each equal to that name's dividend yield
compounded over the ~1.7 years between the window end and today. The vendor series
had been rewritten by dividends paid after the window; ours is anchored and stable.
See docs/DECISIONS.md ADR-006.

Price conventions differ by vendor - this matters enormously
------------------------------------------------------------
"Unadjusted close" does not mean the same thing everywhere:

  AS_TRADED       Truly as-traded. A 10:1 split shows the pre-split price on the day
                  before. Splits AND dividends must be applied.
                  (EODHD raw, Norgate, most paid feeds.)

  SPLIT_ADJUSTED  Splits are ALREADY applied to OHLC and volume; only dividends are
                  left out. yfinance with auto_adjust=False behaves this way -
                  NVDA's 2024-06-06 close comes back as ~$121, not the ~$1,210 that
                  actually traded. Applying split factors again double-counts by 10x.

Getting this wrong is silent and catastrophic: prices jump by the split ratio at
every split, which looks like enormous alpha to a momentum model. The convention is
declared per source in SOURCE_CONVENTION and stamped into the output table.

Method (CRSP-style back-adjustment)
-----------------------------------
Walking backwards from the most recent bar, each action scales every *prior* price:

  split ratio r on ex-date s  -> prior prices divided by r   (AS_TRADED only)
  cash dividend D on ex-date s -> prior prices multiplied by (1 - D / C_prev)

The cumulative factor for date t is the product of every multiplier strictly after t,
so the newest bar has factor 1.0 and adjusted == raw. That anchoring is deliberate:
today's price is the one you would actually transact at.

Two factors are produced, because they are not interchangeable:

  adj_factor        splits + dividends -> use for RETURN calculations (total return)
  adj_factor_price  splits only        -> use for price levels and to scale volume
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..storage import read_silver, silver_exists, write_silver

log = logging.getLogger(__name__)

AS_TRADED = "as_traded"
SPLIT_ADJUSTED = "split_adjusted"

#: What each price source actually delivers in its OHLC columns.
SOURCE_CONVENTION = {
    "yfinance": SPLIT_ADJUSTED,   # verified empirically, see module docstring
    "eodhd": AS_TRADED,           # Phase 3 - re-verify on arrival before trusting
    "stooq": AS_TRADED,
}


def compute_factors(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    convention: str = AS_TRADED,
) -> pd.DataFrame:
    """One row per (security_id, date) with cumulative adjustment factors.

    `bars` needs security_id, ticker, date, close.
    `actions` needs security_id, date, action_type in {dividend, split}, value.
    """
    if convention not in (AS_TRADED, SPLIT_ADJUSTED):
        raise ValueError(f"unknown price convention: {convention!r}")

    bars = bars[["security_id", "ticker", "date", "close"]].copy()
    bars = bars.sort_values(["security_id", "date"]).reset_index(drop=True)

    if actions is None or len(actions) == 0:
        ev = pd.DataFrame(columns=["security_id", "date", "dividend", "split"])
    else:
        ev = (actions.pivot_table(index=["security_id", "date"], columns="action_type",
                                  values="value", aggfunc="sum").reset_index())
        ev.columns.name = None
    for col in ("dividend", "split"):
        if col not in ev.columns:
            ev[col] = np.nan

    df = bars.merge(ev[["security_id", "date", "dividend", "split"]],
                    on=["security_id", "date"], how="left")
    df["dividend"] = df["dividend"].fillna(0.0)
    df["split"] = df["split"].replace(0.0, np.nan).fillna(1.0)

    g = df.groupby("security_id", sort=False)
    df["prev_close"] = g["close"].shift(1)

    # Multipliers applied to every price STRICTLY BEFORE this date.
    if convention == AS_TRADED:
        split_mult = 1.0 / df["split"]
    else:
        # Splits already reflected in OHLC and volume - re-applying would double-count.
        split_mult = pd.Series(1.0, index=df.index)

    with np.errstate(divide="ignore", invalid="ignore"):
        div_mult = 1.0 - (df["dividend"] / df["prev_close"])
    # A dividend on the first bar has no prior close; guard nonsensical ratios too.
    div_mult = div_mult.where(df["prev_close"].notna() & (div_mult > 0), 1.0)

    df["day_mult"] = split_mult * div_mult
    df["day_mult_price"] = split_mult

    def _cum_after(s: pd.Series) -> pd.Series:
        """Product of all multipliers strictly after each row (reverse cumprod)."""
        rev = s.iloc[::-1]
        return rev.cumprod().shift(1).fillna(1.0).iloc[::-1]

    df["adj_factor"] = g["day_mult"].transform(_cum_after)
    df["adj_factor_price"] = g["day_mult_price"].transform(_cum_after)
    df["price_convention"] = convention

    return df[["security_id", "ticker", "date", "adj_factor",
               "adj_factor_price", "price_convention"]]


def apply_factors(bars: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Attach adjusted OHLC + volume to a raw bar frame."""
    out = bars.merge(factors, on=["security_id", "ticker", "date"], how="left")
    out["adj_factor"] = out["adj_factor"].fillna(1.0)
    out["adj_factor_price"] = out["adj_factor_price"].fillna(1.0)

    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[f"adj_{col}"] = out[col] * out["adj_factor"]
    if "volume" in out.columns:
        # Under SPLIT_ADJUSTED, adj_factor_price is 1.0 and volume passes through
        # unchanged - correct, because the vendor already scaled it.
        out["adj_volume"] = out["volume"] / out["adj_factor_price"].replace(0, np.nan)
    return out


def validate_against_vendor(adjusted: pd.DataFrame, tol: float = 1e-4) -> pd.DataFrame:
    """Cross-check our adjusted series against the vendor's, on RETURNS.

    Levels cannot be compared directly: the vendor rewrites its whole history as new
    dividends arrive, so its series differs from ours by a constant scale factor that
    grows with every payout. Daily returns are invariant to that constant, so they
    isolate genuine disagreement - a missed split, a wrong ex-date, a bad print.

    Returns one row per ticker; `flag` is REVIEW when typical return disagreement
    exceeds `tol` (default 1bp).
    """
    if "adj_close_vendor" not in adjusted.columns or "adj_close" not in adjusted.columns:
        return pd.DataFrame()

    d = adjusted.dropna(subset=["adj_close_vendor", "adj_close"]).copy()
    d = d[(d["adj_close_vendor"] > 0) & (d["adj_close"] > 0)]
    if d.empty:
        return pd.DataFrame()

    d = d.sort_values(["ticker", "date"])
    g = d.groupby("ticker")
    d["ret_ours"] = g["adj_close"].pct_change()
    d["ret_vendor"] = g["adj_close_vendor"].pct_change()
    d = d.dropna(subset=["ret_ours", "ret_vendor"])
    d["abs_diff"] = (d["ret_ours"] - d["ret_vendor"]).abs()

    rep = (d.groupby("ticker")
           .agg(bars=("abs_diff", "size"),
                median_ret_diff=("abs_diff", "median"),
                p99_ret_diff=("abs_diff", lambda s: s.quantile(0.99)),
                max_ret_diff=("abs_diff", "max"))
           .reset_index())
    rep["flag"] = np.where(rep["median_ret_diff"] > tol, "REVIEW", "ok")
    return rep.sort_values("median_ret_diff", ascending=False).reset_index(drop=True)


def run(convention: str | None = None) -> dict:
    """Build market/adjustment_factors and market/daily_bars_adjusted from silver."""
    bars = read_silver("market/daily_bars")
    actions = (read_silver("market/corporate_actions")
               if silver_exists("market/corporate_actions") else pd.DataFrame())

    if convention is None:
        srcs = set(bars["source"].unique()) if "source" in bars.columns else {"yfinance"}
        if len(srcs) > 1:
            raise ValueError(
                f"daily_bars mixes price sources {srcs}; adjust each source separately "
                "so the correct convention is applied to each")
        src = next(iter(srcs))
        convention = SOURCE_CONVENTION.get(src, AS_TRADED)
        log.info("price source=%s -> convention=%s", src, convention)

    factors = compute_factors(bars, actions, convention=convention)
    write_silver(factors, "market/adjustment_factors")

    adjusted = apply_factors(bars, factors)
    write_silver(adjusted, "market/daily_bars_adjusted")

    report = validate_against_vendor(adjusted)
    if len(report):
        write_silver(report, "quality/adjustment_vs_vendor")

    return {
        "convention": convention,
        "securities": int(factors["security_id"].nunique()),
        "rows": len(factors),
        "median_return_diff_vs_vendor": (round(float(report["median_ret_diff"].median()), 9)
                                         if len(report) else None),
        "tickers_flagged_for_review": int((report["flag"] == "REVIEW").sum())
                                       if len(report) else 0,
    }
