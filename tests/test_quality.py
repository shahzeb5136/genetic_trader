"""Every data-quality check, exercised on synthetic frames that contain the defect.

The checks exist to turn silent corruption into a visible report, so each one is tested
two ways: a clean frame produces nothing, and a frame with exactly the defect the check
is named for produces one finding at the right severity. A check that fires on clean
data is noise; a check that stays quiet on its own defect is worse than no check.

Everything here is offline and needs no ingested data. The two `@needs_data` tests at
the bottom run the whole battery against the real lake and pin what it must not say.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp500lab.quality import checks as Q
from sp500lab.quality.checks import ERROR, INFO, WARN


# --------------------------------------------------------------------- fixtures

def _bars(n=30, ticker="AAA", sid="SID1", start="2020-01-01", close=None, **cols):
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d")
    c = np.asarray(close if close is not None else np.linspace(100, 110, n), dtype=float)
    df = pd.DataFrame({
        "security_id": sid, "ticker": ticker, "date": dates,
        "open": c * 0.999, "high": c * 1.01, "low": c * 0.99, "close": c,
        "volume": 1_000_000.0, "adj_close_vendor": c, "source": "yfinance",
        "ingest_date": "2026-01-01",
    })
    for k, v in cols.items():
        df[k] = v
    return df


def _calendar(start="2020-01-01", n=60):
    dates = pd.bdate_range(start, periods=n)
    dt = pd.Series(dates)
    return pd.DataFrame({
        "date": dt.dt.strftime("%Y-%m-%d"), "year": dt.dt.year, "month": dt.dt.month,
        "day_of_week": dt.dt.dayofweek,
        "is_month_end": dt.dt.to_period("M") != dt.shift(-1).dt.to_period("M"),
        "is_quarter_end": dt.dt.to_period("Q") != dt.shift(-1).dt.to_period("Q"),
        "is_year_end": dt.dt.year != dt.shift(-1).dt.year,
        "session_index": range(n), "calendar_source": "SPY",
    })


def _intervals(rows):
    return pd.DataFrame(rows, columns=["security_id", "ticker", "start_date",
                                       "end_date", "end_is_open", "source"])


def _sev(findings, check):
    return [f["severity"] for f in findings if f["check"] == check]


# ----------------------------------------------------------------- schema

def test_schema_passes_on_a_conforming_frame():
    assert Q.check_schema("market/daily_bars", _bars()) == []


def test_schema_flags_a_missing_column():
    f = Q.check_schema("market/daily_bars", _bars().drop(columns=["volume"]))
    assert _sev(f, "schema") == [ERROR]
    assert "volume" in f[0]["detail"]


def test_schema_flags_a_datetime_date_column():
    b = _bars()
    b["date"] = pd.to_datetime(b["date"])
    f = Q.check_schema("market/daily_bars", b)
    assert any("YYYY-MM-DD" in x["detail"] for x in f)


def test_schema_flags_a_malformed_date_string():
    b = _bars()
    b.loc[0, "date"] = "01/02/2020"
    f = Q.check_schema("market/daily_bars", b)
    assert _sev(f, "schema") == [ERROR]


def test_schema_ignores_unknown_datasets():
    assert Q.check_schema("nothing/here", _bars()) == []


# ------------------------------------------------------------------- bars

def test_bar_integrity_is_quiet_on_clean_bars():
    assert Q.check_bar_integrity(_bars()) == []


@pytest.mark.parametrize("col,factor,label", [
    ("high", 0.5, "high < low"), ("low", 2.0, "low > open"), ("close", -1.0, "close <= 0"),
])
def test_bar_integrity_catches_each_impossible_relationship(col, factor, label):
    b = _bars()
    b.loc[5, col] = b.loc[5, col] * factor
    f = Q.check_bar_integrity(b)
    assert any(label in x["detail"] and x["severity"] == ERROR for x in f), f


def test_bar_integrity_catches_a_zero_open():
    b = _bars()
    b.loc[3, "open"] = 0.0
    f = Q.check_bar_integrity(b)
    assert any("open <= 0" in x["detail"] and x["severity"] == ERROR for x in f)


def test_bar_integrity_catches_negative_volume():
    b = _bars()
    b.loc[3, "volume"] = -1.0
    f = Q.check_bar_integrity(b)
    assert any("negative volume" in x["detail"] for x in f)


def test_known_bad_bars_are_reported_as_warn_not_error():
    """The allowlist downgrades exactly the reviewed rows and nothing else."""
    (ticker, date), _ = next(iter(Q.KNOWN_BAD_BARS.items()))
    b = _bars(ticker=ticker, start=date, n=3)
    # A hair above the open: trips `low > open` and nothing else.
    b.loc[0, "low"] = b.loc[0, "open"] * 1.0005        # the reviewed defect
    b.loc[2, "low"] = b.loc[2, "open"] * 1.0005        # a NEW one two days later
    f = Q.check_bar_integrity(b)
    sev = sorted(x["severity"] for x in f if x["check"] == "bar_integrity")
    assert sev == [ERROR, WARN]
    err = next(x for x in f if x["severity"] == ERROR)
    assert b.loc[2, "date"] in err["sample"] and date not in err["sample"]


def test_primary_key_catches_a_duplicate_bar():
    b = pd.concat([_bars(), _bars().iloc[[4]]], ignore_index=True)
    assert _sev(Q.check_primary_key(b), "primary_key") == [ERROR]
    assert Q.check_primary_key(_bars()) == []


def test_identity_mapping_is_one_to_one():
    assert Q.check_identity_mapping(_bars()) == []
    two_tickers = pd.concat([_bars(), _bars(ticker="BBB")])          # same sid
    assert _sev(Q.check_identity_mapping(two_tickers), "identity") == [ERROR]
    two_sids = pd.concat([_bars(), _bars(sid="SID2", start="2021-01-01")])  # same ticker
    assert _sev(Q.check_identity_mapping(two_sids), "identity") == [ERROR]


def test_extreme_move_flags_an_unexplained_jump_but_not_a_split():
    close = np.full(30, 100.0)
    close[10:] = 250.0
    b = _bars(close=close)
    assert _sev(Q.check_extreme_moves(b, pd.DataFrame()), "extreme_move") == [WARN]
    split = pd.DataFrame([{"security_id": "SID1", "date": b.loc[10, "date"],
                           "action_type": "split", "value": 2.5}])
    assert Q.check_extreme_moves(b, split) == []


def test_stale_prices_flags_a_flat_run_and_names_the_ticker():
    close = np.linspace(100, 110, 30)
    close[10:20] = 105.0
    f = Q.check_stale_prices(_bars(close=close, ticker="FLAT"))
    assert _sev(f, "stale_prices") == [WARN]
    assert "FLAT:10" in f[0]["sample"]
    assert Q.check_stale_prices(_bars()) == []


def test_calendar_gaps_counts_only_holes_inside_the_active_span():
    cal = _calendar(n=40)
    b = _bars(n=30)
    assert Q.check_calendar_gaps(b, cal) == []           # ends early: not a gap
    holed = b.drop(index=[7, 8])
    f = Q.check_calendar_gaps(holed, cal)
    assert _sev(f, "calendar_gaps") == [WARN] and f[0]["rows"] == 2


def test_off_calendar_bars_are_an_error():
    cal = _calendar(n=40)
    b = _bars(n=5)
    b.loc[2, "date"] = "2020-01-04"                        # a Saturday
    assert _sev(Q.check_off_calendar_bars(b, cal), "off_calendar") == [ERROR]
    assert Q.check_off_calendar_bars(_bars(n=5), cal) == []


def test_ticker_recycling_counts_impostor_bars_and_finds_phantoms():
    iv = _intervals([
        ("SID1", "OLD", "2018-01-01", "2019-06-30", False, "wiki"),   # left long ago
        ("SID2", "GHOST", "2015-01-01", "2016-01-01", False, "wiki"),  # left, never priced
        ("SID3", "CUR", "2018-01-01", None, True, "wiki"),             # current member
    ])
    bars = pd.concat([
        _bars(sid="SID1", ticker="OLD", start="2019-01-01", n=10),    # inside
        _bars(sid="SID1", ticker="OLD", start="2022-01-01", n=7),     # impostor
        _bars(sid="SID2", ticker="GHOST", start="2021-01-01", n=5),   # all impostor
        _bars(sid="SID3", ticker="CUR", start="2024-01-01", n=5),
    ], ignore_index=True)
    f = Q.check_ticker_recycling(bars, iv)
    rec = next(x for x in f if x["check"] == "ticker_recycling")
    assert rec["rows"] == 12 and "OLD(7 bars)" in rec["sample"]
    ph = next(x for x in f if x["check"] == "phantom_history")
    assert ph["rows"] == 1 and ph["sample"] == "GHOST"


def test_universe_coverage_does_not_count_a_phantom_as_priced():
    iv = _intervals([
        ("SID1", "AAA", "2019-01-01", None, True, "wiki"),
        ("SID2", "GHOST", "2015-01-01", "2016-01-01", False, "wiki"),
        ("SID3", "NEVER", "2015-01-01", "2016-01-01", False, "wiki"),
    ])
    bars = pd.concat([_bars(sid="SID1", ticker="AAA", start="2020-01-01"),
                      _bars(sid="SID2", ticker="GHOST", start="2021-01-01")])
    f = Q.check_universe_coverage(bars, iv)[0]
    assert f["detail"].startswith("1/3")
    assert "1 of those have bars on disk" in f["detail"]
    assert "GHOST" in f["sample"] and "NEVER" in f["sample"]


# ------------------------------------------------------------ adjustments

def _factors(n=10, last=1.0):
    dates = pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d")
    fac = np.linspace(0.9, 1.0, n)
    fac[-1] = last
    return pd.DataFrame({"security_id": "SID1", "ticker": "AAA", "date": dates,
                         "adj_factor": fac, "adj_factor_price": 1.0,
                         "price_convention": "split_adjusted"})


def test_adjustment_factors_must_anchor_at_one():
    assert Q.check_adjustment_factors(_factors()) == []
    f = Q.check_adjustment_factors(_factors(last=0.98))
    assert any("!= 1.0" in x["detail"] and x["severity"] == ERROR for x in f)


def test_adjustment_factors_must_be_positive_and_finite():
    fac = _factors()
    fac.loc[2, "adj_factor"] = np.nan
    fac.loc[3, "adj_factor_price"] = 0.0
    f = Q.check_adjustment_factors(fac)
    assert sum(1 for x in f if "not a positive finite" in x["detail"]) == 2


def test_mixed_price_conventions_are_an_error():
    fac = _factors()
    fac.loc[0, "price_convention"] = "as_traded"
    assert any("mixed price conventions" in x["detail"] for x in Q.check_adjustment_factors(fac))


def test_adjusted_bars_must_equal_raw_times_factor():
    b = _bars(n=10)
    b["adj_factor"] = np.linspace(0.9, 1.0, 10)
    b["adj_close"] = b["close"] * b["adj_factor"]
    b["adj_volume"] = b["volume"]
    assert Q.check_adjusted_bars(b) == []
    b.loc[4, "adj_close"] *= 1.01
    f = Q.check_adjusted_bars(b)
    assert any("adj_close != close * adj_factor" in x["detail"] for x in f)


def test_adjusted_bars_flag_a_lost_volume():
    b = _bars(n=5)
    b["adj_factor"], b["adj_close"] = 1.0, b["close"]
    b["adj_volume"] = [1e6, np.nan, 1e6, -5.0, 1e6]
    f = Q.check_adjusted_bars(b)
    assert any(x["rows"] == 2 and "adj_volume" in x["detail"] for x in f)


# ---------------------------------------------------- benchmarks & calendar

def _bench(ticker="SPY", n=30, start="2020-01-01"):
    b = _bars(n=n, ticker=ticker, start=start)
    b["dividend"] = 0.0
    b["split_ratio"] = 0.0
    if ticker == "SPY":
        b.loc[10, "dividend"] = 1.5
    return b.drop(columns=["security_id", "ingest_date"])


def test_benchmarks_are_quiet_when_clean():
    assert Q.check_benchmarks(pd.concat([_bench("SPY"), _bench("^GSPC")]), _calendar()) == []


def test_benchmarks_require_spy_dividends():
    b = _bench("SPY")
    b["dividend"] = 0.0
    f = Q.check_benchmarks(b, None)
    assert any("no dividend events" in x["detail"] and x["severity"] == ERROR for x in f)


def test_benchmarks_flag_bad_ohlc_and_duplicates():
    b = _bench("SPY")
    b.loc[2, "high"] = 1.0
    b = pd.concat([b, b.iloc[[5]]])
    f = Q.check_benchmarks(b, None)
    assert any("impossible OHLC" in x["detail"] for x in f)
    assert any("duplicate" in x["detail"] for x in f)


def test_benchmarks_report_off_calendar_rows_as_info():
    b = _bench("^VIX", n=5)
    b.loc[1, "date"] = "2020-01-04"
    f = Q.check_benchmarks(b, _calendar())
    assert [x["severity"] for x in f if "outside" in x["detail"]] == [INFO]


def test_calendar_is_quiet_when_clean():
    assert Q.check_trading_calendar(_calendar(), None) == []


def test_calendar_catches_a_weekend_session():
    c = _calendar(n=5)
    c.loc[2, "date"] = "2020-01-04"
    assert any("weekend" in x["detail"] for x in Q.check_trading_calendar(c, None))


def test_calendar_catches_a_broken_session_index():
    c = _calendar()
    c.loc[10, "session_index"] = 99
    assert any("session_index" in x["detail"] for x in Q.check_trading_calendar(c, None))


def test_calendar_catches_a_wrong_month_end_flag():
    c = _calendar()
    i = c.index[c["is_month_end"]][0]
    c.loc[i, "is_month_end"] = False
    assert any("is_month_end" in x["detail"] for x in Q.check_trading_calendar(c, None))


def test_calendar_tolerates_the_september_2001_closure_but_not_longer():
    """9/10 -> 9/17 is a closure. 9/10 -> 9/18 has never happened."""
    dates = pd.to_datetime(["2001-09-07", "2001-09-10", "2001-09-17", "2001-09-18"])
    c = _calendar(n=4)
    c["date"] = dates.strftime("%Y-%m-%d")
    c["is_month_end"] = [False, False, False, True]
    assert not any("gaps" in x["detail"] for x in Q.check_trading_calendar(c, None))
    c.loc[2, "date"] = "2001-09-18"
    c.loc[3, "date"] = "2001-09-19"
    assert any("gaps" in x["detail"] for x in Q.check_trading_calendar(c, None))


def test_calendar_reports_sessions_with_no_bars():
    f = Q.check_trading_calendar(_calendar(n=40), _bars(n=30))
    assert any(x["rows"] == 10 and "no bar" in x["detail"] for x in f)


# ---------------------------------------------------------------- identity

def _master(rows):
    return pd.DataFrame(rows, columns=["security_id", "cik", "ticker", "name",
                                       "exchange", "first_seen", "last_seen"])


def test_security_master_uniqueness():
    ok = _master([("SID1", 1, "AAA", "A", "N", "2020", "2021"),
                  ("SID2", 2, "BBB", "B", "N", "2020", "2021")])
    assert Q.check_security_master(ok) == []
    dup = pd.concat([ok, ok.iloc[[0]]])
    assert any("duplicate security_id" in x["detail"] for x in Q.check_security_master(dup))


def test_security_master_flags_a_pair_under_two_ids():
    m = _master([("SID1", 1, "AAA", "A", "N", "2020", "2021"),
                 ("SID9", 1, "AAA", "A", "N", "2020", "2021")])
    assert any("same (cik, ticker)" in x["detail"] for x in Q.check_security_master(m))


def test_security_master_reports_missing_cik_as_info_and_junk_ticker_as_warn():
    m = _master([("SID1", None, "NONE.", "?", "N", "2020", "2021")])
    f = Q.check_security_master(m)
    assert INFO in _sev(f, "security_master") and WARN in _sev(f, "security_master")


def test_referential_integrity_finds_an_orphan():
    master = _master([("SID1", 1, "AAA", "A", "N", "2020", "2021")])
    f = Q.check_referential_integrity(master, {"market/daily_bars": _bars(sid="SID7")})
    assert f[0]["severity"] == ERROR and "SID7" in f[0]["sample"]
    assert Q.check_referential_integrity(master, {"x": _bars(sid="SID1")}) == []


def test_membership_intervals_invariants():
    assert Q.check_membership_intervals(_intervals([
        ("SID1", "AAA", "2018-01-01", "2019-01-01", False, "w"),
        ("SID1", "AAA", "2020-01-01", None, True, "w")])) == []
    inverted = _intervals([("SID1", "AAA", "2019-01-01", "2018-01-01", False, "w")])
    assert any("end before they start" in x["detail"] for x in Q.check_membership_intervals(inverted))
    two_open = _intervals([("SID1", "AAA", "2018-01-01", None, True, "w"),
                           ("SID1", "AAA", "2020-01-01", None, True, "w")])
    assert any("more than one open" in x["detail"] for x in Q.check_membership_intervals(two_open))
    overlap = _intervals([("SID1", "AAA", "2018-01-01", "2019-06-01", False, "w"),
                          ("SID1", "AAA", "2019-01-01", None, True, "w")])
    assert any("overlapping" in x["detail"] for x in Q.check_membership_intervals(overlap))


# -------------------------------------------------------- corporate actions

def _actions(rows):
    return pd.DataFrame(rows, columns=["security_id", "ticker", "date", "action_type",
                                       "value", "source"])


def test_corporate_actions_invariants():
    b = _bars(n=10)
    d = b.loc[5, "date"]
    ok = _actions([("SID1", "AAA", d, "dividend", 0.5, "y")])
    assert Q.check_corporate_actions(ok, _calendar(), b) == []
    bad = _actions([("SID1", "AAA", d, "dividend", -0.5, "y"),
                    ("SID1", "AAA", d, "split", 1.0, "y"),
                    ("SID1", "AAA", d, "split", 5000.0, "y"),
                    ("SID1", "AAA", "2020-01-04", "dividend", 0.1, "y"),
                    ("SID1", "AAA", d, "merger", 1.0, "y")])
    f = Q.check_corporate_actions(bad, _calendar(), b)
    details = " | ".join(x["detail"] for x in f)
    for phrase in ("non-positive", "ratio 1.0", "decimal place", "non-sessions",
                   "unknown action_type", "duplicate"):
        assert phrase in details, phrase


def test_corporate_actions_flag_a_dividend_bigger_than_half_the_price():
    b = _bars(n=10)
    d = b.loc[5, "date"]
    big = _actions([("SID1", "AAA", d, "dividend", 80.0, "y")])
    f = Q.check_corporate_actions(big, None, b)
    assert any("half the prior close" in x["detail"] for x in f)


# ------------------------------------------------------------ fundamentals

def _xbrl(n=2000):
    """`n` filler facts under one tag plus one fact per core tag, all clean."""
    rows = []
    for i in range(n):
        rows.append({"security_id": "SID1", "ticker": "AAA", "cik": 1,
                     "tag": "OperatingIncomeLoss", "unit": "USD",
                     "period_start": "2020-01-01",
                     "period_end": f"2020-{3 * (i % 4) + 1:02d}-01",
                     "value": 1e9 + i, "form": "10-Q", "filed_date": "2021-01-15",
                     "accession": f"acc{i}"})
    for t in Q.CORE_XBRL_TAGS:
        rows.append({**rows[0], "tag": t, "accession": f"acc-{t}"})
    return pd.DataFrame(rows)


def test_fundamentals_are_quiet_when_clean():
    assert Q.check_fundamentals(_xbrl()) == []


def test_fundamentals_catch_missing_dates_values_and_core_tags():
    x = _xbrl()
    x.loc[0, "filed_date"] = None
    x.loc[1, "value"] = np.nan
    x = x[x["tag"] != "Assets"]
    details = " | ".join(f["detail"] for f in Q.check_fundamentals(x))
    assert "no filed_date" in details and "no value" in details and "Assets" in details


def test_fundamentals_grade_early_filings_by_share():
    x = _xbrl()
    x.loc[0, "filed_date"] = "2019-01-01"                  # 1 in 2000: cover-page noise
    assert _sev(Q.check_fundamentals(x), "fundamentals") == [WARN]
    x["filed_date"] = "2019-01-01"                          # all of them: swapped dates
    assert ERROR in _sev(Q.check_fundamentals(x), "fundamentals")


def test_fundamentals_flag_duplicates_and_absurd_magnitudes():
    x = pd.concat([_xbrl(), _xbrl().iloc[[0]]])
    x.loc[x.index[1], "value"] = 5e14
    details = " | ".join(f["detail"] for f in Q.check_fundamentals(x))
    assert "duplicate" in details and "scale error" in details


def test_fundamentals_coverage_is_info_above_85_percent():
    f = Q.check_fundamentals(_xbrl(), universe={"SID1"})
    assert [x["severity"] for x in f if "have XBRL facts" in x["detail"]] == [INFO]
    f = Q.check_fundamentals(_xbrl(), universe={"SID1", "SID2", "SID3"})
    assert [x["severity"] for x in f if "have XBRL facts" in x["detail"]] == [WARN]


# ------------------------------------------------------------------- macro

def _macro(series="DGS10", values=(1.0, 1.5, 2.0), start="2020-01-01", revised=False):
    dates = pd.bdate_range(start, periods=len(values)).strftime("%Y-%m-%d")
    return pd.DataFrame({"series_id": series, "date": dates, "value": list(values),
                         "description": series, "revised": revised})


def test_macro_integrity_is_quiet_when_clean():
    assert Q.check_macro_integrity(pd.concat([_macro(), _macro("USREC", (0, 1, 0))])) == []


def test_macro_integrity_catches_range_units_and_indicator_errors():
    m = pd.concat([_macro("DGS10", (1.0, 150.0, 2.0)),         # percent -> basis points?
                   _macro("USREC", (0, 2, 0)),                  # not 0/1
                   _macro("VIXCLS", (np.nan, np.nan, np.nan))])  # empty
    details = " | ".join(f["detail"] for f in Q.check_macro_integrity(m))
    assert "DGS10" in details and "0/1" in details and "no values" in details


def test_macro_integrity_catches_duplicates_and_mixed_revised_flags():
    m = pd.concat([_macro(), _macro().iloc[[0]]])
    m.loc[m.index[-1], "revised"] = True
    details = " | ".join(f["detail"] for f in Q.check_macro_integrity(m))
    assert "duplicate" in details and "revised=True and revised=False" in details


def test_macro_staleness_only_watches_daily_series():
    m = pd.concat([_macro("DGS10", start="2020-01-01"), _macro("UNRATE", start="2020-01-01")])
    f = Q.check_macro_staleness(m, as_of="2020-03-01")
    assert len(f) == 1 and "DGS10" in f[0]["sample"] and "UNRATE" not in f[0]["sample"]
    assert Q.check_macro_staleness(m, as_of="2020-01-08") == []


# ------------------------------------------------------------ cross-source

def test_vix_cross_source_agrees_disagrees_and_diverges():
    n = 300
    vix = np.random.default_rng(0).uniform(12, 30, n)
    fred = _macro("VIXCLS", vix, start="2019-01-01")
    yahoo = _bench("^VIX", n=n, start="2019-01-01")
    yahoo["close"] = vix
    assert _sev(Q.check_vix_cross_source(fred, yahoo), "cross_source_vix") == [INFO]
    yahoo.loc[:10, "close"] += 1.0                          # a burst of shifted days
    assert _sev(Q.check_vix_cross_source(fred, yahoo), "cross_source_vix") == [WARN]
    yahoo["close"] = vix * 10                               # a different series entirely
    assert _sev(Q.check_vix_cross_source(fred, yahoo), "cross_source_vix") == [ERROR]


def test_spy_must_track_the_index():
    n = 400
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, n)
    spy = _bench("SPY", n=n)
    gspc = _bench("^GSPC", n=n)
    spy["close"] = 100 * np.cumprod(1 + r)
    gspc["close"] = 3000 * np.cumprod(1 + r + rng.normal(0, 0.0005, n))
    assert _sev(Q.check_spy_tracks_index(pd.concat([spy, gspc])), "cross_source_spy") == [INFO]
    gspc["close"] = 3000 * np.cumprod(1 + rng.normal(0, 0.01, n))
    assert _sev(Q.check_spy_tracks_index(pd.concat([spy, gspc])), "cross_source_spy") == [ERROR]


# -------------------------------------------------------------------- gold

def test_half_spread_invariants():
    ok = pd.DataFrame({"security_id": "S", "date": "2020-01-01",
                       "half_spread": [0.001, 0.002], "binding": ["estimator", "tick_floor"]})
    assert Q.check_half_spread(ok) == []
    bad = ok.copy()
    bad.loc[0, "half_spread"] = -0.1
    bad.loc[1, "binding"] = "guess"
    details = " | ".join(f["detail"] for f in Q.check_half_spread(bad))
    assert "outside" in details and "unknown binding" in details


def test_delisting_invariants():
    ok = pd.DataFrame({"reason_category": ["bankruptcy", "acquisition", "unresolved"],
                       "delist_return": [-1.0, 0.0, 0.0]})
    f = Q.check_delisting_returns(ok)
    assert [x["severity"] for x in f] == [INFO]              # the unresolved count
    bad = ok.copy()
    bad.loc[0, "delist_return"] = -0.5                       # a bankruptcy that kept half
    bad.loc[1, "delist_return"] = 3.0
    bad.loc[2, "reason_category"] = "vanished"
    details = " | ".join(x["detail"] for x in Q.check_delisting_returns(bad))
    assert "total loss" in details and "outside" in details and "unknown reason" in details


# ------------------------------------------------------------------ runner

def test_a_crashing_check_becomes_a_finding_not_an_exception():
    findings: list[dict] = []
    Q._safe(findings, "boom", lambda: 1 / 0)
    assert findings[0]["severity"] == ERROR and "ZeroDivisionError" in findings[0]["detail"]


def test_summary_counts_every_severity():
    rep = pd.DataFrame({"severity": [ERROR, WARN, WARN, INFO]})
    assert Q.summary(rep) == {"ERROR": 1, "WARN": 2, "INFO": 1}


# ----------------------------------------------------------- the real lake

def _have_lake() -> bool:
    from sp500lab.storage import silver_exists
    return silver_exists("market/daily_bars") and silver_exists("reference/trading_calendar")


needs_data = pytest.mark.skipif(not _have_lake(), reason="silver layer not built")


@needs_data
def test_the_real_lake_has_no_unreviewed_errors():
    """Anything structurally impossible in the real data is either reviewed
    (KNOWN_BAD_BARS) or a regression. Bronze re-hashing is skipped for speed; the
    manifest has its own test in test_storage.py."""
    rep = Q.run(verify_bronze=False)
    errors = rep[rep["severity"] == ERROR]
    assert errors.empty, errors[["check", "detail"]].to_string()


@needs_data
def test_the_real_lake_cross_sources_agree():
    rep = Q.run(verify_bronze=False)
    by = rep.set_index("check")["severity"]
    assert by.get("cross_source_vix") == INFO
    assert by.get("cross_source_spy") == INFO
    assert by.get("adjustment_vs_vendor") == INFO
