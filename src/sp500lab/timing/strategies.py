"""The calendar rules: nine standing claims about WHEN the market pays.

Every rule here compiles to the same two boolean vectors (see engine.py), holds one
instrument, and has zero fitted parameters - each is a published anomaly implemented
from its paper's recipe with conventional definitions, so the trial count stays what
it looks like: nine hypotheses, not nine hundred parameterisations. The moment one of
these grows a tunable threshold it becomes a search, and belongs in a study with a
deflated Sharpe like any other search.

The family portrait:

    tm_buy_hold       both legs, always. The calibration instrument and the bar.
    tm_overnight      close -> next open, every session. Cliff Asness calls this
                      "the market's night shift"; Cooper, Cliff & Gulen documented it.
    tm_intraday       open -> close, every session. The other half of the partition.
    tm_weekend        Friday close -> Monday open. French (1980) found Monday falls;
                      the modern claim is the weekend gap is where that lives.
    tm_turn_of_month  the last session and first three of each month (Lakonishok &
                      Smidt 1988; McConnell & Xu 2008 found it is ALL of the equity
                      premium in their window).
    tm_pre_holiday    only the session before an exchange holiday (Ariel 1990 - once
                      a third of the year's return in a handful of days).
    tm_sell_in_may    hold November through April (Bouman & Jacobsen 2002). The
                      weakest sample here: ~15 independent cycles, and it is listed
                      as much to show the sample-size problem as to test the claim.
    tm_vix_overnight  the overnight leg only when the PRIOR session's VIX is above
                      its trailing-year median - the documented concentration of the
                      overnight premium in high-fear regimes.
    tm_month_end_drift  hold the last five sessions of the month; the institutional
                      flow window (pension rebalancing, 401k contributions land).

A warning that applies to the whole family, from this project's own planning document
(WHAT_TO_BUILD_NEXT.md): predicting the MARKET is sample-starved - 177 months, three
regimes. These rules live at daily granularity, so tm_overnight rests on ~3,700
observations rather than 177 - but tm_sell_in_may rests on ~15, and honesty about
which rule has which sample is half the reason the family exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TimingData

#: Winter months for the Halloween indicator, per Bouman & Jacobsen.
_WINTER = (11, 12, 1, 2, 3, 4)


class TimingStrategy:
    """One calendar rule. Subclasses set `name` and implement `legs(data)`."""

    name: str = "unnamed"

    def legs(self, data: TimingData) -> tuple[np.ndarray, np.ndarray]:
        """(hold_overnight, hold_intraday), both (D,) bool over data.dates."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "class": type(self).__name__, "params": {},
                "construction": None, "warmup": 0,
                "instrument": "single benchmark instrument, all-in or cash"}


def _from_return_days(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compile 'be invested close-to-close on these sessions' into the two legs.

    A session d's close-to-close return spans the overnight leg of d-1 and the
    intraday leg of d, so: hold the intraday leg where the mask is on, and hold the
    overnight leg where the NEXT session's mask is on. This is the whole translation
    between how the anomaly literature states its windows and how the engine trades.
    """
    hold_id = mask.astype(bool).copy()
    hold_on = np.zeros_like(hold_id)
    hold_on[:-1] = hold_id[1:]
    return hold_on, hold_id


class BuyHold(TimingStrategy):
    """Own the index every hour the market allows. The bar every rule must beat.

    Also the calibration instrument: both legs on with zero costs must reproduce the
    same SPY total return the monthly engine is calibrated to (ADR-018), which pins
    this engine's accounting to the other one's.
    """

    name = "tm_buy_hold"

    def legs(self, data: TimingData) -> tuple[np.ndarray, np.ndarray]:
        ones = np.ones(data.n_dates, dtype=bool)
        return ones.copy(), ones


class Overnight(TimingStrategy):
    """Hold only while the market is closed: buy every close, sell every open.

    The overnight anomaly is the largest and strangest calendar effect on record:
    since at least the early 1990s, close-to-open has carried essentially ALL of US
    equities' return while open-to-close has carried the risk (Cooper, Cliff & Gulen
    2008 and a large literature since). Candidate mechanisms - earnings land overnight,
    market-makers charge for holding inventory through the close, retail buys at the
    open - disagree about whether it should persist, which is what makes it worth
    a standing test rather than an argument.

    It is also the family's designated cost casualty: two trades a session is ~500
    round trips a year, and whether any overnight premium survives the spread bill at
    retail size is exactly what the three cost settings print. Published results are
    gross; this one is net, and the gap between the two IS the finding.
    """

    name = "tm_overnight"

    def legs(self, data: TimingData) -> tuple[np.ndarray, np.ndarray]:
        return (np.ones(data.n_dates, dtype=bool),
                np.zeros(data.n_dates, dtype=bool))


class Intraday(TimingStrategy):
    """Hold only while the market is open: buy every open, sell every close.

    The control for tm_overnight - the two partition every close-to-close return, and
    the engine asserts their product equals buy-and-hold. If overnight carries the
    return, this carries the volatility for nothing, and seeing BOTH curves is what
    makes the decomposition legible.
    """

    name = "tm_intraday"

    def legs(self, data: TimingData) -> tuple[np.ndarray, np.ndarray]:
        return (np.zeros(data.n_dates, dtype=bool),
                np.ones(data.n_dates, dtype=bool))


class Weekend(TimingStrategy):
    """Hold only from the last close of the week to the first open of the next.

    French (1980): Monday's close-to-close return was reliably negative, and the gap
    from Friday's close already carried the weekend's news risk for nothing. The
    modern form asks whether the weekend gap specifically - two days of news with no
    trading - is priced differently from any overnight gap. One trade in, one out,
    ~52 times a year, so unlike tm_overnight this survives its own cost bill easily;
    the question is only whether there is anything left to collect.

    "Week boundary" is read off the sessions themselves (the next session's weekday
    wraps backwards), so a Friday holiday makes Thursday the entry and a Monday
    holiday makes Tuesday the exit, exactly as a standing order would have behaved.
    """

    name = "tm_weekend"

    def legs(self, data: TimingData) -> tuple[np.ndarray, np.ndarray]:
        hold_on = np.zeros(data.n_dates, dtype=bool)
        hold_on[:-1] = data.day_of_week[1:] <= data.day_of_week[:-1]
        return hold_on, np.zeros(data.n_dates, dtype=bool)


class TurnOfMonth(TimingStrategy):
    """Hold the last session of each month and the first three of the next.

    Lakonishok & Smidt (1988) found the four sessions around the month boundary carry
    a disproportionate share of the market's return; McConnell & Xu (2008) found that
    in their 1987-2005 window they carried ALL of it. The usual mechanism story is
    mechanical flow - payrolls, 401(k) contributions and pension rebalances land on a
    calendar, and the buying they force is not price-sensitive.

    Invested roughly 4 sessions in 21 (~19% of the time), so its return per unit of
    time in the market is the number to watch, and its Sharpe gets a structural boost
    from sitting in cash through most of the volatility. Both readings are printed;
    neither is adjusted away.
    """

    name = "tm_turn_of_month"

    def legs(self, data: TimingData) -> tuple[np.ndarray, np.ndarray]:
        m = data.month
        is_last = np.zeros(data.n_dates, dtype=bool)
        is_last[:-1] = m[1:] != m[:-1]
        pos = _position_in_month(m)
        return _from_return_days(is_last | (pos <= 3))


class PreHoliday(TimingStrategy):
    """Hold only the trading session immediately before an exchange holiday.

    Ariel (1990): the day before holidays earned many times the average day - at one
    point a third of the year's return in about eight sessions. Stories range from
    short-covering into a closure to plain mood. It is the smallest net a calendar can
    cast (~9 sessions a year), which makes it cheap to trade and quick to refute:
    with so few observations a decade of data moves the estimate visibly, and that is
    working as intended.

    Holidays are read from the session calendar itself: any gap to the next session
    that is not a plain weekend means the exchange was closed on a weekday.
    """

    name = "tm_pre_holiday"

    def legs(self, data: TimingData) -> tuple[np.ndarray, np.ndarray]:
        d = pd.to_datetime(pd.Series(data.dates.tolist()))
        gap = np.zeros(data.n_dates, dtype=np.int64)
        gap[:-1] = (d.diff().dt.days.to_numpy()[1:]).astype(np.int64)
        plain_weekend = (data.day_of_week == 4) & (gap == 3)
        holiday_ahead = (gap > 1) & ~plain_weekend
        return _from_return_days(holiday_ahead)


class SellInMay(TimingStrategy):
    """Hold November through April, sit in cash May through October.

    Bouman & Jacobsen (2002) documented the Halloween effect across 36 of 37 markets;
    it remains the calendar anomaly with the best out-of-sample record ACROSS
    COUNTRIES and the worst statistical footing WITHIN one: a 2007-2021 research
    window contains about fifteen independent winter/summer cycles, which is no
    sample at all. It is included at daily machinery mostly to demonstrate that a
    seasonal claim cannot be rescued by daily data - the effective sample is the
    number of seasons, not the number of sessions - and any reading of its result
    should start from that.
    """

    name = "tm_sell_in_may"

    def legs(self, data: TimingData) -> tuple[np.ndarray, np.ndarray]:
        return _from_return_days(np.isin(data.month, _WINTER))


class VixOvernight(TimingStrategy):
    """The overnight leg, taken only when yesterday's VIX topped its trailing median.

    The one rule in the family that reads market data, so the one rule with a lag
    discipline: the gate is the PRIOR session's VIX close against the median of the
    year ending at that same prior session - nothing the standing order could not
    have known at 15:50. The claim being isolated is documented in the overnight
    literature: the close-to-open premium concentrates in high-fear regimes, when
    market-makers charge the most for carrying inventory through the dark. If true,
    this keeps most of tm_overnight's return on roughly half its nights - and half
    the nights means half the round trips, so it is also the version of the overnight
    trade with the best chance of surviving its own costs.

    The median-of-trailing-year is a conventional split, not a searched threshold;
    move it and this becomes a parameter sweep that belongs in a registered study.
    """

    name = "tm_vix_overnight"

    def legs(self, data: TimingData) -> tuple[np.ndarray, np.ndarray]:
        vix = pd.Series(data.vix)
        med = vix.rolling(252, min_periods=126).median()
        gate = (vix > med).to_numpy()
        hold_on = np.zeros(data.n_dates, dtype=bool)
        hold_on[1:] = gate[:-1]          # decided at t from the close of t-1
        hold_on &= np.isfinite(data.vix)
        return hold_on, np.zeros(data.n_dates, dtype=bool)


class MonthEndDrift(TimingStrategy):
    """Hold the last five sessions of every month, nothing else.

    The institutional-flow companion to tm_turn_of_month, splitting the same story at
    the boundary: this rule owns only the run-up INTO month end (window-dressing,
    benchmark-tracking flows, and the monthly rebalance bid this project's own
    mandate produces), while tm_turn_of_month owns the boundary and the start. If the
    flow story is right, the two should not both work - the drift should live on
    whichever side the forced buyers actually trade. A pair of rules designed so at
    most one can be true is worth more than either alone.
    """

    name = "tm_month_end_drift"

    def legs(self, data: TimingData) -> tuple[np.ndarray, np.ndarray]:
        pos_from_end = _position_from_month_end(data.month)
        return _from_return_days(pos_from_end <= 5)


def _position_in_month(month: np.ndarray) -> np.ndarray:
    """(D,) 1-indexed session position within its calendar month."""
    pos = np.ones(len(month), dtype=np.int64)
    for t in range(1, len(month)):
        pos[t] = pos[t - 1] + 1 if month[t] == month[t - 1] else 1
    return pos


def _position_from_month_end(month: np.ndarray) -> np.ndarray:
    """(D,) 1-indexed session position counting back from its month's last session."""
    pos = np.ones(len(month), dtype=np.int64)
    for t in range(len(month) - 2, -1, -1):
        pos[t] = pos[t + 1] + 1 if month[t] == month[t + 1] else 1
    return pos


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_TIMING_REGISTRY: dict[str, type[TimingStrategy]] = {
    cls.name: cls
    for cls in (BuyHold, Overnight, Intraday, Weekend, TurnOfMonth, PreHoliday,
                SellInMay, VixOvernight, MonthEndDrift)
}

#: Named sets, mirroring strategies.GROUPS. `tm_buy_hold` is the benchmark twin and
#: is listed so it can be run, but scoreboards should treat it as the bar, not a row.
TIMING_GROUPS: dict[str, tuple[str, ...]] = {
    "timing": ("tm_overnight", "tm_intraday", "tm_weekend", "tm_turn_of_month",
               "tm_pre_holiday", "tm_sell_in_may", "tm_vix_overnight",
               "tm_month_end_drift"),
    "all": ("tm_buy_hold", "tm_overnight", "tm_intraday", "tm_weekend",
            "tm_turn_of_month", "tm_pre_holiday", "tm_sell_in_may",
            "tm_vix_overnight", "tm_month_end_drift"),
}


def get_timing_strategy(name: str) -> TimingStrategy:
    if name not in _TIMING_REGISTRY:
        raise KeyError(f"unknown timing strategy {name!r}; "
                       f"available: {sorted(_TIMING_REGISTRY)}")
    return _TIMING_REGISTRY[name]()


def list_timing_strategies() -> list[str]:
    return sorted(_TIMING_REGISTRY)
