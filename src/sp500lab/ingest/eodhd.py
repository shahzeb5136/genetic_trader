"""EODHD ingestion, with a hard API-budget guard.

Why this module is budget-obsessed
----------------------------------
The free tier allows **20 API calls per day**. The paid tier allows 100k, but the
whole funding model for this project is a burst-buy: subscribe for one month, bulk
download everything, cancel. In both regimes an API call is a scarce, non-renewable
resource within its window, and a loop that accidentally re-fetches is not an
inefficiency - it is lost data you cannot get back today.

So every call goes through `_spend()`, which checks the live remaining budget before
issuing a request and refuses rather than overrunning. `/user` is free (verified: it
does not increment apiRequests), so the budget can be polled without cost.

Free-tier limits, measured not assumed
--------------------------------------
* 20 calls/day, plus a separate `extraLimit` allowance
* **1 year of history only** - requesting from=2000-01-01 returns ~251 bars starting
  one year back. Deep history requires the paid plan.
* `/user` is free; `/eod/{ticker}` costs 1; symbol lists cost 1 each.

Given that, the free tier's best use is NOT bulk price history - 20 tickers x 1 year
is worthless next to the 677 x 26 years already held from Yahoo. Its best use is
**universe metadata**: the active and delisted US symbol lists are one call each and
answer the strategic question directly - does EODHD actually carry the 296 delisted
S&P 500 members that Yahoo drops? That answer is what justifies (or refutes) the
annual purchase.

Price convention - MUST be verified before trusting
---------------------------------------------------
EODHD documents OHLC as "raw - adjusted for neither splits nor dividends" while
**volume is already split-adjusted**. That is a MIXED convention, unlike yfinance
(everything split-adjusted) or a pure as-traded feed. `verify_price_convention()`
tests it empirically against a known split rather than trusting the documentation.
Until that passes, do not feed EODHD volume into anything. See ADR-007 / ADR-014.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from ..config import get_settings
from ..http_cache import fetch
from ..storage import today_iso, write_silver, write_vault
from .base import IngestResult

log = logging.getLogger(__name__)

SOURCE = "eodhd"
BASE = "https://eodhd.com/api"

# Keep a couple of calls in reserve so an interactive check never hits a hard zero.
RESERVE_CALLS = 1


class BudgetExhausted(RuntimeError):
    """Raised instead of issuing a request that would exceed the daily allowance."""


def _token() -> str:
    tok = get_settings().secret("EODHD_API_TOKEN")
    if not tok:
        raise RuntimeError(
            "EODHD_API_TOKEN not set. Add it to .env (which is gitignored).")
    return tok


def account_status() -> dict:
    """Live account + budget state. This endpoint does NOT consume budget."""
    url = f"{BASE}/user?api_token={_token()}&fmt=json"
    # ttl=0 so the budget reading is always current, never a stale cache hit.
    data = fetch(url, source=SOURCE, ttl_seconds=0, force=True).json()
    used = int(data.get("apiRequests", 0))
    limit = int(data.get("dailyRateLimit", 0))
    return {
        "subscription": data.get("subscriptionType", "unknown"),
        "used": used,
        "limit": limit,
        "remaining": max(limit - used, 0),
        "extra_limit": int(data.get("extraLimit", 0)),
    }


def _spend(url: str, *, label: str, cost: int = 1, ttl_seconds: float | None = None):
    """Issue one budgeted request, refusing rather than overrunning the allowance.

    A cached response costs nothing, so the budget check runs only on a real miss.
    """
    from ..http_cache import _cache_paths, request_hash
    key = request_hash("GET", url)
    binp, metap = _cache_paths(SOURCE, key)
    if binp.exists() and metap.exists() and ttl_seconds is None:
        return fetch(url, source=SOURCE, ttl_seconds=None)  # free replay

    status = account_status()
    if status["remaining"] < cost + RESERVE_CALLS:
        raise BudgetExhausted(
            f"{label}: needs {cost} call(s) but only {status['remaining']} remain today "
            f"(limit {status['limit']}, reserve {RESERVE_CALLS}). Resets at 00:00 UTC.")
    log.info("  spending %d call(s) on %s  [%d/%d used]",
             cost, label, status["used"], status["limit"])
    return fetch(url, source=SOURCE, ttl_seconds=ttl_seconds)


# --------------------------------------------------------------------- universe

def run_symbol_lists(force: bool = False) -> IngestResult:
    """Active + delisted US symbol lists. Two calls, the highest-value spend.

    `delisted=1` REPLACES the result set rather than extending it, so both calls are
    required to see the whole universe.
    """
    res = IngestResult(source=SOURCE, dataset="symbol_lists")
    frames = []

    for delisted in (0, 1):
        label = "delisted" if delisted else "active"
        url = (f"{BASE}/exchange-symbol-list/US?api_token={_token()}&fmt=json"
               + ("&delisted=1" if delisted else ""))
        try:
            resp = _spend(url, label=f"US symbol list ({label})",
                          ttl_seconds=None if not force else 0)
        except BudgetExhausted as exc:
            res.errors.append(str(exc))
            continue

        res.fetched += 0 if resp.from_cache else 1
        res.from_cache += 1 if resp.from_cache else 0
        write_vault(source=SOURCE, dataset="symbol_lists",
                    filename=f"us_{label}.json", content=resp.content,
                    url=url.replace(_token(), "REDACTED"),
                    extra={"delisted": bool(delisted)})
        res.bronze_files += 1

        df = pd.DataFrame(resp.json())
        if df.empty:
            continue
        df.columns = [c.lower() for c in df.columns]
        df["is_delisted"] = bool(delisted)
        frames.append(df)

    if not frames:
        res.errors.append("no symbol lists retrieved")
        return res

    out = pd.concat(frames, ignore_index=True)
    out["retrieved_at"] = today_iso()
    write_silver(out, "reference/eodhd_us_symbols")

    res.rows = len(out)
    res.notes = {
        "active": int((~out["is_delisted"]).sum()),
        "delisted": int(out["is_delisted"].sum()),
        "columns": list(out.columns),
    }
    return res


def coverage_vs_missing() -> pd.DataFrame:
    """Does EODHD carry the S&P 500 members that Yahoo dropped?

    This is the purchase-decision report. It cross-references the ever-member
    universe against what we actually hold, and checks each gap against EODHD's
    symbol lists. A high hit rate here is the entire justification for the annual fee.
    """
    from ..storage import read_silver, silver_exists
    if not silver_exists("reference/eodhd_us_symbols"):
        raise FileNotFoundError("run `ingest eodhd-symbols` first")

    sym = read_silver("reference/eodhd_us_symbols")
    iv = read_silver("reference/sp500_membership_intervals")
    bars = read_silver("market/daily_bars")

    ever = set(iv["ticker"].unique())
    have = set(bars["ticker"].unique())
    missing = sorted(ever - have)

    # Share-class spelling differs three ways across sources: EODHD writes BRK-B, we
    # canonicalise to BRK.B, and some old Wikipedia revisions wrote BRKB with no
    # separator at all. Comparing on a separator-stripped form matches all three, so
    # a formatting difference is never mistaken for missing coverage.
    def squash(code: str) -> str:
        return str(code).upper().replace("-", "").replace(".", "").replace("/", "")

    eod_all = {squash(c) for c in sym["code"].astype(str)}
    eod_delisted = {squash(c) for c in sym.loc[sym["is_delisted"], "code"].astype(str)}

    rows = []
    for t in missing:
        k = squash(t)
        rows.append({
            "ticker": t,
            "in_eodhd": k in eod_all,
            "eodhd_delisted_list": k in eod_delisted,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- price data

def fetch_eod(ticker: str, start: str | None = None) -> pd.DataFrame:
    """One ticker's EOD history. Costs 1 call. Free tier returns ~1 year."""
    start = start or "2000-01-01"
    sym = ticker.replace(".", "-")
    url = (f"{BASE}/eod/{sym}.US?api_token={_token()}&fmt=json&period=d"
           f"&from={start}&order=a")
    resp = _spend(url, label=f"eod {ticker}", ttl_seconds=None)
    write_vault(source=SOURCE, dataset="eod", filename=f"{ticker}.json",
                content=resp.content, url=url.replace(_token(), "REDACTED"),
                extra={"ticker": ticker, "from": start})
    df = pd.DataFrame(resp.json())
    if df.empty:
        return df
    df["ticker"] = ticker
    return df


def verify_price_convention(ticker: str = "NVDA", split_date: str = "2024-06-10") -> dict:
    """Empirically determine what EODHD's OHLC actually represents.

    Documentation says raw/as-traded, but the cost of being wrong is a silent 10x
    error that looks like alpha (see ADR-007), so it gets tested against a known
    split rather than trusted.

    NOTE: on the free tier this returns only ~1 year of history, so a 2024 split is
    out of range and the test is INCONCLUSIVE. It is written now so that the check
    runs automatically the moment a paid plan makes the window available.
    """
    df = fetch_eod(ticker)
    if df.empty:
        return {"verdict": "no data"}
    df = df.sort_values("date")
    window = df[(df["date"] >= "2024-06-05") & (df["date"] <= "2024-06-11")]
    if window.empty:
        return {
            "verdict": "inconclusive",
            "reason": f"{split_date} outside available window "
                      f"({df['date'].min()}..{df['date'].max()}) - free tier gives ~1yr",
            "action": "re-run after upgrading; do not trust SOURCE_CONVICTION until then",
        }
    pre = window[window["date"] < split_date]["close"]
    if pre.empty:
        return {"verdict": "inconclusive", "reason": "no pre-split bar"}
    val = float(pre.iloc[-1])
    return {
        "verdict": "as_traded" if val > 600 else "split_adjusted",
        "pre_split_close": val,
        "note": "NVDA traded ~$1210 pre-split; ~$121 means the feed pre-adjusts splits",
    }


def run(force: bool = False) -> IngestResult:
    """Default EODHD job: account status + universe metadata (2 calls)."""
    status = account_status()
    log.info("EODHD account: %s  %d/%d calls used today",
             status["subscription"], status["used"], status["limit"])
    res = run_symbol_lists(force=force)
    res.notes["account"] = status
    return res
