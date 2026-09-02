"""The panel: every backtest input as a dense (date x security) matrix.

Why a matrix and not a DataFrame
--------------------------------
A genetic algorithm with population 200 over 50 generations is 10,000 fitness
evaluations. Each evaluation is a full backtest. If a backtest costs 10 seconds the
run costs 28 hours; if it costs 50 milliseconds the run costs 8 minutes. That is the
entire difference between "we can evolve strategies" and "we cannot", and it is
decided here rather than in the GA.

So the panel is built ONCE, cached on disk and in-process, and every backtest after
that is arithmetic over numpy arrays that are already resident. A strategy asking for
"the last 252 closes" gets a slice - O(1), no copy, no allocation. Nothing in the hot
loop touches DuckDB, pandas groupby, or the filesystem.

Why this is also the leakage guard
----------------------------------
Because prices live in one rectangular array indexed by session number, "data up to
and including as_of" is exactly `close[:t + 1]` - a numpy view whose last row IS the
as-of date. A strategy handed that view cannot address row t+1: the memory is not in
the object. See context.py, which is the only thing allowed to construct these views.

What the columns mean
---------------------
Every matrix is (n_sessions, n_securities), aligned to the same date index (the NYSE
sessions in `trading_calendar`) and the same security index. NaN means "no bar" -
either not listed yet, or already gone. That is load-bearing: `last_bar_index` is
derived from it and drives delisting resolution in the engine.

Three separate masks, which are NOT the same thing and get conflated constantly:

  in_index[d, s]   membership per sp500_membership_intervals - the strategy's universe
  has_price[d, s]  a bar exists - the engine can mark this position to market
  tradable[d, s]   in_index AND has_price AND passes the liquidity floor - buyable

`in_index & ~has_price` is the coverage gap, and it is large in the early years
(58% priced in 2007, 98% in 2025). The engine reports it per rebalance rather than
silently backtesting a 274-name subset of a 470-name index and calling it "the S&P
500". See docs/BACKTEST.md, "Coverage is a result, not a footnote".
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..normalize.splits import cumulative_split_ratio, load_splits
from .portfolio import stable_tiebreak
from ..paths import GOLD_DIR
from ..query import connect, prices_clipped_to_membership

log = logging.getLogger(__name__)

#: Sessions of price history kept before a name enters the index, so indicators have
#: warmup. 400 calendar days ~ 275 sessions, enough for a 12-month lookback.
DEFAULT_WARMUP_DAYS = 400

#: Calendar days of price history kept AFTER membership ends. A name dropped from the
#: index in month M is still in `universe_asof(M-end)`, so the engine only sells it at
#: the M+1 rebalance - up to ~35 calendar days later. Without a buffer that bar does not
#: exist and a live company would be booked as a delisting.
#:
#: MEASURED, not guessed: of 207 closed membership intervals that have prices, 203 keep
#: trading past 45 days and only 4 stop - which is exactly the handful of genuine
#: delistings Yahoo carries. So 45 days separates "removed from the index, still listed"
#: from "actually delisted" cleanly, and stays an order of magnitude below the 365-day
#: window quality/checks.py uses to detect a reassigned symbol.
DEFAULT_EXIT_DAYS = 45

PANEL_CACHE_DIR = GOLD_DIR / "backtest" / "panel"

#: Bumped whenever the on-disk layout changes, so a stale cache is never loaded.
PANEL_FORMAT_VERSION = 6


@dataclass
class Panel:
    """Aligned matrices for one universe over one date range.

    Treat as immutable. The engine holds the whole thing; a Context holds only
    slices of it (see context.PanelView).
    """

    dates: np.ndarray            # (D,)  '<U10' YYYY-MM-DD, ascending, NYSE sessions
    security_ids: np.ndarray     # (S,)  '<U16'
    tickers: np.ndarray          # (S,)  '<U16'
    tiebreak: np.ndarray         # (S,)  uint64, stable hash - see portfolio.py

    adj_close: np.ndarray        # (D,S) float64, total-return adjusted, NaN where no bar
    adj_open: np.ndarray         # (D,S) float64, execution price
    raw_close: np.ndarray        # (D,S) float64, as-stored close (split-adjusted, ADR-007)
    raw_open: np.ndarray         # (D,S) float64, as-stored open - the execution print
    cum_split: np.ndarray        # (D,S) float64, ratio to recover as-traded share counts

    dollar_volume: np.ndarray    # (D,S) float32, trailing median, knowable at date
    half_spread: np.ndarray      # (D,S) float32, proportional half-spread estimate

    in_index: np.ndarray         # (D,S) bool
    has_price: np.ndarray        # (D,S) bool

    first_bar_index: np.ndarray  # (S,) int32, -1 if never priced
    last_bar_index: np.ndarray   # (S,) int32, -1 if never priced
    delist_return: np.ndarray    # (S,) float64, terminal return applied at last bar
    delist_reason: np.ndarray    # (S,) '<U24'

    rebalance_index: np.ndarray  # (R,) int32, row indices of month-end sessions
    index_size: np.ndarray       # (D,) int32, TRUE index membership count per session
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- lookups

    def __post_init__(self) -> None:
        self._date_pos = {d: i for i, d in enumerate(self.dates.tolist())}
        self._sid_pos = {s: i for i, s in enumerate(self.security_ids.tolist())}

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    @property
    def n_securities(self) -> int:
        return len(self.security_ids)

    def date_index(self, date: str, side: str = "exact") -> int:
        """Row index for a date.

        side='exact'  KeyError if the date is not a session
        side='prev'   the latest session <= date
        side='next'   the earliest session >= date
        """
        if side == "exact":
            return self._date_pos[date]
        pos = int(np.searchsorted(self.dates, date, side="right" if side == "prev" else "left"))
        if side == "prev":
            pos -= 1
        if pos < 0 or pos >= len(self.dates):
            raise KeyError(f"no session {side} of {date!r} in panel range "
                           f"{self.dates[0]}..{self.dates[-1]}")
        return pos

    def security_index(self, security_id: str) -> int:
        return self._sid_pos[security_id]

    def tradable(self, liquidity_floor: float = 0.0) -> np.ndarray:
        """(D,S) bool - in the index, priced, and liquid enough to buy.

        Recomputed rather than stored because the floor is a strategy-level choice.
        """
        mask = self.in_index & self.has_price
        if liquidity_floor > 0:
            mask = mask & (self.dollar_volume >= liquidity_floor)
        return mask

    def coverage(self) -> pd.DataFrame:
        """Per rebalance date: true index size, how many are priced, and the ratio.

        The denominator is `index_size`, the count from the membership table itself -
        NOT the number of panel columns flagged in_index. Those differ by exactly the
        names we hold no prices for, which is the entire quantity this diagnostic
        exists to surface. Dividing by the panel would report ~99% coverage in 2008 by
        construction and hide that a third of the index was missing.
        """
        idx = self.rebalance_index
        true_n = self.index_size[idx]
        priced = (self.in_index[idx] & self.has_price[idx]).sum(axis=1)
        return pd.DataFrame({
            "date": self.dates[idx],
            "in_index": true_n,
            "priced": priced,
            "coverage": np.where(true_n > 0, priced / np.maximum(true_n, 1), np.nan),
        })

    # ------------------------------------------------------------ persistence

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            dates=self.dates, security_ids=self.security_ids, tickers=self.tickers,
            tiebreak=self.tiebreak,
            adj_close=self.adj_close, adj_open=self.adj_open, raw_close=self.raw_close,
            raw_open=self.raw_open,
            cum_split=self.cum_split, dollar_volume=self.dollar_volume,
            half_spread=self.half_spread, in_index=self.in_index, has_price=self.has_price,
            first_bar_index=self.first_bar_index, last_bar_index=self.last_bar_index,
            delist_return=self.delist_return, delist_reason=self.delist_reason,
            rebalance_index=self.rebalance_index, index_size=self.index_size,
            meta=np.array(json.dumps(self.meta)),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "Panel":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(**{k: z[k] for k in z.files if k != "meta"}, meta=meta)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

def _cache_key(start: str, end: str | None, warmup_days: int, exit_days: int,
               liquidity_window: int) -> str:
    payload = json.dumps({"v": PANEL_FORMAT_VERSION, "start": start, "end": end,
                          "warmup": warmup_days, "exit": exit_days,
                          "liqwin": liquidity_window}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


_MEMO: dict[str, Panel] = {}


def build_panel(
    start: str = "2000-01-01",
    end: str | None = None,
    *,
    warmup_days: int = DEFAULT_WARMUP_DAYS,
    exit_days: int = DEFAULT_EXIT_DAYS,
    liquidity_window: int = 20,
    use_cache: bool = True,
    rebuild: bool = False,
) -> Panel:
    """Assemble the panel from silver, with a disk + in-process cache.

    Called once per process in normal use. A GA that evaluates 10,000 individuals
    calls it 10,000 times and gets the memoized object every time after the first -
    which is the point.
    """
    key = _cache_key(start, end, warmup_days, exit_days, liquidity_window)
    if use_cache and not rebuild and key in _MEMO:
        return _MEMO[key]

    cache_path = PANEL_CACHE_DIR / f"{key}.npz"
    if use_cache and not rebuild and cache_path.exists():
        log.info("panel: loading cache %s", cache_path.name)
        panel = Panel.load(cache_path)
        _MEMO[key] = panel
        return panel

    panel = _build(start, end, warmup_days=warmup_days, exit_days=exit_days,
                   liquidity_window=liquidity_window)
    if use_cache:
        panel.save(cache_path)
        log.info("panel: cached to %s (%.1f MB)", cache_path.name,
                 cache_path.stat().st_size / 1e6)
    _MEMO[key] = panel
    return panel


def clear_memo() -> None:
    """Drop the in-process panel cache. Tests use this; nothing else should need it."""
    _MEMO.clear()


def _build(start: str, end: str | None, *, warmup_days: int, exit_days: int,
           liquidity_window: int) -> Panel:
    con = connect()

    end = end or con.execute("SELECT max(date) FROM trading_calendar").fetchone()[0]
    cal = con.execute(
        "SELECT date, is_month_end FROM trading_calendar "
        "WHERE date >= ? AND date <= ? ORDER BY date", [start, end]).df()
    if cal.empty:
        raise ValueError(f"no trading sessions between {start} and {end}")
    dates = cal["date"].to_numpy().astype("<U10")
    date_pos = {d: i for i, d in enumerate(dates.tolist())}
    D = len(dates)

    # Bars, clipped to membership. The exit buffer exists only so a removal can be
    # sold at the next open; see query.prices_clipped_to_membership.
    log.info("panel: reading clipped bars (%s..%s)", start, end)
    bars = prices_clipped_to_membership(con, warmup_days=warmup_days, exit_days=exit_days)
    bars = bars[(bars["date"] >= start) & (bars["date"] <= end)]
    # One security can hold several membership intervals (re-entry, or a reassigned
    # symbol under the same id); the clip join then emits a bar once per interval.
    bars = bars.drop_duplicates(subset=["security_id", "date"], keep="first")

    sids = np.sort(bars["security_id"].unique()).astype("<U16")
    sid_pos = {s: i for i, s in enumerate(sids.tolist())}
    S = len(sids)
    log.info("panel: %d sessions x %d securities", D, S)

    tickers = (bars.sort_values("date").groupby("security_id")["ticker"].last()
               .reindex(sids).fillna("").to_numpy().astype("<U16"))

    row = bars["date"].map(date_pos)
    col = bars["security_id"].map(sid_pos)
    keep = row.notna().to_numpy()
    if (~keep).any():
        log.warning("panel: dropped %d bars on non-session dates", int((~keep).sum()))
        bars, row, col = bars[keep], row[keep], col[keep]
    row = row.to_numpy(dtype=np.int64)
    col = col.to_numpy(dtype=np.int64)

    def _mat(series: pd.Series) -> np.ndarray:
        m = np.full((D, S), np.nan, dtype=np.float64)
        m[row, col] = series.to_numpy(dtype=np.float64)
        return m

    adj_close = _mat(bars["adj_close"])
    adj_open = _mat(bars["adj_open"])
    raw_close = _mat(bars["close"])
    raw_open = _mat(bars["open"])
    volume = _mat(bars["volume"])

    has_price = np.isfinite(adj_close) & (adj_close > 0)
    # An open is needed to execute. Where it is missing but the close is not, fall back
    # to the close: a strategy is not allowed to profit from a data gap, but it must
    # still be able to exit. Counted so the fallback never happens silently.
    open_gap = has_price & ~(np.isfinite(adj_open) & (adj_open > 0))
    n_open_gap = int(open_gap.sum())
    if n_open_gap:
        adj_open = np.where(open_gap, adj_close, adj_open)
        # Fill the as-traded open the same way, from the same bar, so the price the
        # trade ledger prints is always the price the engine actually executed at.
        raw_open = np.where(open_gap, raw_close, raw_open)
        log.warning("panel: %d bars had no usable open; filled from close", n_open_gap)

    # Trailing median dollar volume - trailing, so it is knowable at every date.
    dv = pd.DataFrame(np.where(has_price, raw_close * volume, np.nan))
    dollar_volume = (dv.rolling(liquidity_window, min_periods=max(1, liquidity_window // 2))
                     .median().to_numpy().astype(np.float32))

    cum_split = cumulative_split_ratio(dates, sid_pos, load_splits(con))
    cum_split[~has_price] = np.nan
    in_index = _membership_matrix(con, dates, sid_pos)
    half_spread = _load_half_spread(con, dates, sids, date_pos, sid_pos)

    first_bar_index = np.full(S, -1, dtype=np.int32)
    last_bar_index = np.full(S, -1, dtype=np.int32)
    any_price = has_price.any(axis=0)
    first_bar_index[any_price] = np.argmax(has_price[:, any_price], axis=0)
    last_bar_index[any_price] = D - 1 - np.argmax(has_price[::-1, any_price], axis=0)

    delist_return, delist_reason = _load_delisting(con, sids)

    rebalance_index = np.array(
        [date_pos[d] for d in cal.loc[cal["is_month_end"], "date"]], dtype=np.int32)
    index_size = _true_index_size(con, dates)

    meta = {
        "format_version": PANEL_FORMAT_VERSION,
        "start": start, "end": end,
        "warmup_days": warmup_days, "exit_days": exit_days,
        "liquidity_window": liquidity_window,
        "n_dates": D, "n_securities": S, "n_bars": int(has_price.sum()),
        "index_members_never_priced": int(_never_priced(con, sid_pos)),
        "n_rebalances": len(rebalance_index),
        "open_gaps_filled_from_close": n_open_gap,
        "half_spread_source": ("gold/backtest/half_spread"
                               if bool(np.isfinite(half_spread).any())
                               else "MISSING - run `sp500lab backtest build-spreads`"),
    }
    return Panel(
        dates=dates, security_ids=sids, tickers=tickers,
        tiebreak=stable_tiebreak(sids),
        adj_close=adj_close, adj_open=adj_open, raw_close=raw_close, raw_open=raw_open,
        cum_split=cum_split,
        dollar_volume=dollar_volume, half_spread=half_spread,
        in_index=in_index, has_price=has_price,
        first_bar_index=first_bar_index, last_bar_index=last_bar_index,
        delist_return=delist_return, delist_reason=delist_reason,
        rebalance_index=rebalance_index, index_size=index_size, meta=meta,
    )


def _membership_matrix(con, dates: np.ndarray, sid_pos: dict[str, int]) -> np.ndarray:
    """(D,S) bool from the point-in-time membership intervals.

    Built from `sp500_membership_intervals` and nothing else. Reconstructing it from
    `sp500_current` is the survivorship-bias mistake this whole repo exists to avoid.
    """
    D, S = len(dates), len(sid_pos)
    mask = np.zeros((D, S), dtype=bool)
    iv = con.execute("""
        SELECT security_id, start_date, end_date, end_is_open
        FROM sp500_membership_intervals
    """).df()
    for r in iv.itertuples(index=False):
        s = sid_pos.get(r.security_id)
        if s is None:
            continue
        lo = int(np.searchsorted(dates, r.start_date, side="left"))
        hi = D if (r.end_is_open or not r.end_date) else int(
            np.searchsorted(dates, r.end_date, side="right"))
        if hi > lo:
            mask[lo:hi, s] = True
    return mask


def _load_half_spread(con, dates, sids, date_pos, sid_pos) -> np.ndarray:
    """Proportional half-spread per (date, security) from gold, or NaN if not built.

    NaN is deliberate rather than a silent default: costs.py substitutes an explicit
    fallback and records that it did. A backtest that quietly assumed zero spread is
    exactly the kind of lie this repo is built to prevent.
    """
    D, S = len(dates), len(sids)
    out = np.full((D, S), np.nan, dtype=np.float32)
    try:
        df = con.execute("SELECT security_id, date, half_spread FROM gold_half_spread").df()
    except Exception:  # noqa: BLE001 - table not built yet
        log.warning("panel: gold/backtest/half_spread not built; costs will use a fallback")
        return out
    row = df["date"].map(date_pos)
    col = df["security_id"].map(sid_pos)
    keep = (row.notna() & col.notna()).to_numpy()
    out[row[keep].to_numpy(dtype=np.int64), col[keep].to_numpy(dtype=np.int64)] = \
        df.loc[keep, "half_spread"].to_numpy(dtype=np.float32)
    return out


def _load_delisting(con, sids) -> tuple[np.ndarray, np.ndarray]:
    """(S,) terminal return and reason category, from gold/backtest/delisting_returns.

    Default is 0.0 / 'unresolved': liquidate at the last observed close. That is the
    conservative choice for an index removal and the WRONG one for a bankruptcy, which
    is why delisting.py exists and why the engine reports how many positions exited
    with an unresolved reason.
    """
    S = len(sids)
    ret = np.zeros(S, dtype=np.float64)
    reason = np.full(S, "unresolved", dtype="<U24")
    try:
        df = con.execute(
            "SELECT security_id, delist_return, reason_category FROM gold_delisting_returns"
        ).df()
    except Exception:  # noqa: BLE001 - table not built yet
        log.warning("panel: gold/backtest/delisting_returns not built; "
                    "every delisting will resolve at last close")
        return ret, reason
    pos = {s: i for i, s in enumerate(sids.tolist())}
    for r in df.itertuples(index=False):
        i = pos.get(r.security_id)
        if i is not None:
            ret[i] = float(r.delist_return)
            reason[i] = str(r.reason_category)[:24]
    return ret, reason


def _true_index_size(con, dates: np.ndarray) -> np.ndarray:
    """(D,) how many securities were in the index on each session.

    Counted from `sp500_membership_intervals` directly, so it includes members we hold
    no prices for. That difference IS the coverage gap, and it has to be measured
    against the real index rather than against the subset we happen to have data for.
    """
    D = len(dates)
    out = np.zeros(D, dtype=np.int32)
    iv = con.execute("""
        SELECT start_date, end_date, end_is_open FROM sp500_membership_intervals
    """).df()
    for r in iv.itertuples(index=False):
        lo = int(np.searchsorted(dates, r.start_date, side="left"))
        hi = D if (r.end_is_open or not r.end_date) else int(
            np.searchsorted(dates, r.end_date, side="right"))
        if hi > lo:
            out[lo:hi] += 1
    return out


def _never_priced(con, sid_pos: dict[str, int]) -> int:
    """Index members with no usable price history at all - the hard coverage floor."""
    members = con.execute(
        "SELECT DISTINCT security_id FROM sp500_membership_intervals").df()["security_id"]
    return int((~members.isin(list(sid_pos))).sum())
