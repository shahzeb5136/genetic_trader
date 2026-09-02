"""Unit tests for the invariants that are expensive to get wrong.

These run offline against synthetic fixtures - no network, no ingested data required.
Each test corresponds to a failure mode that actually occurred during development.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sp500lab.ingest.wikipedia_history import (
    _extract_ticker,
    _find_ticker_column,
    build_intervals,
    parse_constituents,
)
from sp500lab.normalize.adjustments import AS_TRADED, SPLIT_ADJUSTED, apply_factors, compute_factors
from sp500lab.registry import SecurityRegistry

# --------------------------------------------------------------- security master

def test_share_classes_get_distinct_ids():
    """GOOG and GOOGL share a CIK but are different securities."""
    reg = SecurityRegistry()
    a = reg.resolve_or_assign(1652044, "GOOGL")
    b = reg.resolve_or_assign(1652044, "GOOG")
    assert a != b


def test_same_pair_is_stable():
    reg = SecurityRegistry()
    a = reg.resolve_or_assign(320193, "AAPL")
    b = reg.resolve_or_assign(320193, "AAPL")
    assert a == b
    assert len(reg) == 1


def test_recycled_ticker_splits_by_issuer():
    """WM was Washington Mutual, then Waste Management - two securities."""
    reg = SecurityRegistry()
    wamu = reg.resolve_or_assign(933136, "WM")
    wm = reg.resolve_or_assign(823768, "WM")
    assert wamu != wm


@pytest.mark.parametrize("raw,expected", [
    ("BRK-B", "BRK.B"), ("BRK/B", "BRK.B"), ("brk.b", "BRK.B"), (" AAPL ", "AAPL"),
])
def test_ticker_normalisation(raw, expected):
    assert SecurityRegistry.normalize_ticker(raw) == expected


def test_ids_are_not_reused_after_reload():
    reg = SecurityRegistry()
    first = reg.resolve_or_assign(1, "AAA")
    reloaded = SecurityRegistry(reg.df)
    second = reloaded.resolve_or_assign(2, "BBB")
    assert first != second
    assert reloaded.resolve_or_assign(1, "AAA") == first


# ------------------------------------------------------------- wikitext parsing

@pytest.mark.parametrize("cell,expected", [
    ("{{NyseSymbol|MMM}}", "MMM"),
    ("{{NASDAQ|AAPL}}", "AAPL"),
    ("{{NYSE|BRK.B}}", "BRK.B"),
    ("[[3M|MMM]]", "MMM"),
    ("MMM", "MMM"),
    ("reports", None),                    # SEC-filings column, not a ticker
    ("[[Abbott Laboratories]]", None),    # company name, not a ticker
    ("", None),
    # Agilent's ticker is a single letter. It sat on the not-a-ticker list from the
    # first commit and the company was missing from every snapshot (ADR-044).
    ("A", "A"),
    ("{{NYSE|A}}", "A"),
    ("[[Agilent Technologies|A]]", "A"),
    ("N/A", None),                        # normalises to N.A - boilerplate, not a ticker
    ("NA", None),
    ("N", None),
    # Cboe lists on its own BZX exchange; Wikipedia switched its row to this template in
    # 2019-01 and the parser dropped it for seven years of snapshots (ADR-044).
    ("{{BZX link|CBOE}}", "CBOE"),
    ("{{Cboe|CBOE}}", "CBOE"),
])
def test_extract_ticker(cell, expected):
    assert _extract_ticker(cell) == expected


TABLE_TICKER_FIRST = """{| class="wikitable sortable"
|-
! [[Ticker symbol]] !! Company !! GICS Sector
|-
| {{NyseSymbol|MMM}} || [[3M Co.]] || Industrials
|-
| {{NyseSymbol|ABT}} || [[Abbott]] || Health Care
|}"""

# The 2007-era layout: Company first, Ticker second. Assuming column 0 here
# silently harvested company names and yielded ~4 tickers instead of 500.
TABLE_COMPANY_FIRST = """{| class="wikitable"
|+ S & P 500 component stocks
! Company !! [[Ticker symbol]] !! [[SEC filing|SEC filings]] !! Industry
|-
|\t[[3M Company]]\t || \tMMM\t || \t[http://sec.gov reports]\t || \tConglomerates
|-
|\t[[ACE Limited]]\t || \tACE\t || \t[http://sec.gov reports]\t || \tInsurance
|}"""

# Modern layout: newline-separated cells.
TABLE_NEWLINE_CELLS = """{| class="wikitable sortable"
|-
![[Ticker symbol|Symbol]]
! Security !! GICS Sector
|-
|| {{NyseSymbol|MMM}}
|| [[3M]]
|| Industrials
|-
|| {{NyseSymbol|AOS}}
|| [[A. O. Smith]]
|| Industrials
|}"""


def test_ticker_column_detected_from_header():
    assert _find_ticker_column(TABLE_TICKER_FIRST) == 0
    assert _find_ticker_column(TABLE_COMPANY_FIRST) == 1
    assert _find_ticker_column(TABLE_NEWLINE_CELLS) == 0


@pytest.mark.parametrize("table,expected", [
    (TABLE_TICKER_FIRST, ["MMM", "ABT"]),
    (TABLE_COMPANY_FIRST, ["MMM", "ACE"]),
    (TABLE_NEWLINE_CELLS, ["MMM", "AOS"]),
])
def test_parse_constituents_across_eras(table, expected):
    assert parse_constituents(table) == expected


# ------------------------------------------------------------------- intervals

def _snap(pairs):
    return pd.DataFrame(pairs, columns=["snapshot_date", "ticker"])


def test_intervals_mark_open_membership():
    snaps = _snap([("2020-01-31", "AAA"), ("2020-02-29", "AAA"), ("2020-03-31", "AAA")])
    iv = build_intervals(snaps)
    assert len(iv) == 1
    assert iv.loc[0, "start_date"] == "2020-01-31"
    assert bool(iv.loc[0, "end_is_open"]) is True
    assert iv.loc[0, "end_date"] is None


def test_intervals_close_when_membership_ends():
    snaps = _snap([("2020-01-31", "AAA"), ("2020-02-29", "AAA"),
                   ("2020-03-31", "BBB")])
    iv = build_intervals(snaps).set_index("ticker")
    assert iv.loc["AAA", "end_date"] == "2020-02-29"
    assert bool(iv.loc["AAA", "end_is_open"]) is False


def test_reentry_produces_two_intervals():
    """A name that leaves and returns must not be collapsed into one span."""
    snaps = _snap([("2020-01-31", "AAA"), ("2020-02-29", "BBB"),
                   ("2020-03-31", "AAA")])
    iv = build_intervals(snaps)
    aaa = iv[iv["ticker"] == "AAA"].sort_values("start_date").reset_index(drop=True)
    assert len(aaa) == 2
    assert aaa.loc[0, "end_date"] == "2020-01-31"
    assert bool(aaa.loc[1, "end_is_open"]) is True


# ----------------------------------------------------------------- adjustments

def _bars(closes, dates=None):
    dates = dates or [f"2024-01-{i + 1:02d}" for i in range(len(closes))]
    return pd.DataFrame({
        "security_id": ["SID1"] * len(closes), "ticker": ["AAA"] * len(closes),
        "date": dates, "close": closes, "open": closes, "high": closes,
        "low": closes, "volume": [1000] * len(closes),
    })


def _action(date, kind, value):
    return pd.DataFrame([{"security_id": "SID1", "ticker": "AAA", "date": date,
                          "action_type": kind, "value": value}])


def test_newest_bar_is_always_unadjusted():
    """Anchoring at the present is what makes the series reproducible."""
    bars = _bars([100.0, 101.0, 50.0])
    f = compute_factors(bars, _action("2024-01-03", "split", 2.0), convention=AS_TRADED)
    assert f["adj_factor"].iloc[-1] == pytest.approx(1.0)


def test_as_traded_split_halves_prior_prices():
    bars = _bars([100.0, 100.0, 50.0])
    f = compute_factors(bars, _action("2024-01-03", "split", 2.0), convention=AS_TRADED)
    adj = apply_factors(bars, f)
    # Pre-split bars scale down by the ratio, so the series has no phantom jump.
    assert adj["adj_close"].iloc[0] == pytest.approx(50.0)
    assert adj["adj_close"].iloc[2] == pytest.approx(50.0)


def test_split_adjusted_convention_does_not_reapply_splits():
    """yfinance pre-applies splits; re-applying them double-counts by the ratio."""
    bars = _bars([100.0, 100.0, 50.0])
    acts = _action("2024-01-03", "split", 2.0)
    f_as = compute_factors(bars, acts, convention=AS_TRADED)
    f_sa = compute_factors(bars, acts, convention=SPLIT_ADJUSTED)
    assert f_as["adj_factor"].iloc[0] == pytest.approx(0.5)
    assert f_sa["adj_factor"].iloc[0] == pytest.approx(1.0)


def test_dividend_reduces_prior_prices():
    bars = _bars([100.0, 100.0, 99.0])
    f = compute_factors(bars, _action("2024-01-03", "dividend", 1.0),
                        convention=SPLIT_ADJUSTED)
    # prior prices scale by (1 - 1.00/100.00) = 0.99
    assert f["adj_factor"].iloc[0] == pytest.approx(0.99)
    assert f["adj_factor"].iloc[-1] == pytest.approx(1.0)


def test_volume_scales_by_split_only():
    bars = _bars([100.0, 100.0, 50.0])
    acts = pd.concat([_action("2024-01-03", "split", 2.0),
                      _action("2024-01-02", "dividend", 1.0)], ignore_index=True)
    f = compute_factors(bars, acts, convention=AS_TRADED)
    adj = apply_factors(bars, f)
    # A dividend does not change share count, so adj_volume reflects the split alone.
    assert adj["adj_volume"].iloc[0] == pytest.approx(2000.0)


def test_no_actions_leaves_prices_untouched():
    bars = _bars([10.0, 11.0, 12.0])
    f = compute_factors(bars, pd.DataFrame(), convention=AS_TRADED)
    assert (f["adj_factor"] == 1.0).all()


def test_unknown_convention_rejected():
    with pytest.raises(ValueError, match="unknown price convention"):
        compute_factors(_bars([1.0]), pd.DataFrame(), convention="nonsense")
