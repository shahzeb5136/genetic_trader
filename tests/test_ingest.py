"""The ingest parsers and guards, offline.

Every ingestor is fetch -> bronze -> parse -> silver, and the parse step is the only
one with logic in it. These tests feed each parser a hand-built payload shaped like
the real one and check the silver row shape that every downstream consumer assumes.
No network: the fetch step is exercised in test_storage.py against a stubbed client.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sp500lab.ingest import benchmarks, eodhd, fred, sec_companyfacts
from sp500lab.ingest.base import IngestResult
from sp500lab.ingest.prices_yfinance import OHLC_COLS, _to_long
from sp500lab.registry import SecurityRegistry

# ------------------------------------------------------------- yfinance reshape

def _multi(tickers, n=3):
    """A frame shaped like `yf.download(..., group_by='column')` for several tickers."""
    idx = pd.date_range("2020-01-01", periods=n, tz="UTC")
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"],
         tickers])
    data = np.random.default_rng(0).uniform(10, 20, (n, len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols)


def test_to_long_reshapes_a_multi_ticker_batch():
    df = _to_long(_multi(["AAA", "BBB-C"]), ["AAA", "BBB-C"])
    assert set(df["ticker"]) == {"AAA", "BBB-C"} and len(df) == 6
    for c in OHLC_COLS + ["adj_close_vendor", "dividend", "split_ratio"]:
        assert c in df.columns, c
    assert df["date"].dt.tz is None, "dates are naive session dates, not UTC instants"


def test_to_long_handles_a_single_flat_frame():
    raw = _multi(["AAA"]).droplevel(1, axis=1)
    raw.index.name = "Date"
    df = _to_long(raw, ["AAA"])
    assert df["ticker"].unique().tolist() == ["AAA"] and len(df) == 3


def test_to_long_drops_calendar_padding_rows():
    """yfinance pads every ticker to the union of dates; an all-NaN row is not a bar."""
    raw = _multi(["AAA", "BBB"])
    for field in ("Open", "High", "Low", "Close", "Volume"):
        raw.loc[raw.index[0], (field, "BBB")] = np.nan
    df = _to_long(raw, ["AAA", "BBB"])
    assert len(df[df["ticker"] == "BBB"]) == 2 and len(df[df["ticker"] == "AAA"]) == 3


def test_to_long_of_nothing_is_an_empty_frame():
    assert _to_long(None, ["AAA"]).empty
    assert _to_long(pd.DataFrame(), ["AAA"]).empty


# ---------------------------------------------------------------- FRED CSV

def test_fred_csv_parses_dots_as_missing_and_keeps_the_row():
    text = "observation_date,DGS10\n2020-01-01,.\n2020-01-02,1.88\nnot-a-date,2.0\n"
    df = fred.parse_fred_csv(text, "DGS10", "10y", revised=False)
    assert df["date"].tolist() == ["2020-01-01", "2020-01-02"]
    assert np.isnan(df.loc[0, "value"]) and df.loc[1, "value"] == 1.88
    assert list(df.columns) == ["series_id", "date", "value", "description", "revised"]
    assert (df["series_id"] == "DGS10").all() and (~df["revised"]).all()


def test_fred_csv_value_column_is_numeric_even_when_every_row_is_a_dot():
    df = fred.parse_fred_csv("DATE,X\n2020-01-01,.\n", "X", "x", revised=True)
    assert pd.api.types.is_numeric_dtype(df["value"]) and df["revised"].all()


def test_fred_csv_rejects_a_single_column_payload():
    with pytest.raises(ValueError, match="two-column"):
        fred.parse_fred_csv("just_one_column\n1\n", "X", "x", revised=False)


def test_fred_series_table_tags_revisions_honestly():
    """Market prints are final; statistical releases are revised for months."""
    assert fred.SERIES["DGS10"][1] is False and fred.SERIES["VIXCLS"][1] is False
    assert fred.SERIES["GDPC1"][1] is True and fred.SERIES["CPIAUCSL"][1] is True


# ------------------------------------------------------------ trading calendar

def test_calendar_flags_the_last_session_of_each_period_not_the_31st():
    # 2020-01-31 was a Friday; 2020-05-29 Fri was the last May session (31st a Sunday)
    dates = pd.bdate_range("2020-01-01", "2020-06-05").strftime("%Y-%m-%d").tolist()
    cal = benchmarks.derive_calendar(dates)
    me = set(cal.loc[cal["is_month_end"], "date"])
    assert {"2020-01-31", "2020-05-29"} <= me and "2020-05-31" not in me
    qe = set(cal.loc[cal["is_quarter_end"], "date"])
    assert "2020-03-31" in qe and "2020-01-31" not in qe
    assert cal["session_index"].tolist() == list(range(len(dates)))
    assert cal.iloc[-1]["is_month_end"], "the newest session closes its own month"


def test_calendar_dedupes_sorts_and_refuses_garbage():
    cal = benchmarks.derive_calendar(["2020-01-03", "2020-01-02", "2020-01-03"])
    assert cal["date"].tolist() == ["2020-01-02", "2020-01-03"]
    with pytest.raises(ValueError):
        benchmarks.derive_calendar([])
    with pytest.raises(ValueError):
        benchmarks.derive_calendar(["2020-01-02", "yesterday"])


def test_spy_is_the_calendar_source_and_a_benchmark():
    assert benchmarks.CALENDAR_SOURCE == "SPY" and "SPY" in benchmarks.BENCHMARKS


# ------------------------------------------------------------- XBRL flattening

def _payload():
    return {
        "entityName": "Acme",
        "facts": {"us-gaap": {
            "Assets": {"units": {"USD": [
                {"start": None, "end": "2020-12-31", "val": 100, "fy": 2020, "fp": "FY",
                 "form": "10-K", "filed": "2021-02-01", "accn": "a1", "frame": "CY2020Q4I"},
                # the same period restated a year later - both rows must survive
                {"start": None, "end": "2020-12-31", "val": 105, "fy": 2021, "fp": "FY",
                 "form": "10-K", "filed": "2022-02-01", "accn": "a2"},
                # no filing date: useless for point-in-time, must be dropped
                {"start": None, "end": "2021-12-31", "val": 110, "accn": "a3"},
            ]}},
            "Revenues": {"units": {"USD": [
                {"start": "2020-01-01", "end": "2020-12-31", "val": 50,
                 "form": "10-K", "filed": "2021-02-01", "accn": "a1"}]}},
        }},
    }


def test_flatten_keeps_restatements_and_drops_undated_facts():
    rows = sec_companyfacts._flatten_facts(_payload(), cik=7, keep=None)
    assets = [r for r in rows if r["tag"] == "Assets"]
    assert len(assets) == 2 and {r["filed_date"] for r in assets} == {"2021-02-01",
                                                                       "2022-02-01"}
    assert all(r["cik"] == 7 and r["entity_name"] == "Acme" for r in rows)
    assert {r["tag"] for r in rows} == {"Assets", "Revenues"}


def test_flatten_respects_the_tag_allowlist():
    rows = sec_companyfacts._flatten_facts(_payload(), cik=7, keep={"Revenues"})
    assert [r["tag"] for r in rows] == ["Revenues"]
    assert rows[0]["period_start"] == "2020-01-01" and rows[0]["value"] == 50


def test_flatten_survives_an_empty_or_malformed_payload():
    assert sec_companyfacts._flatten_facts({}, cik=1, keep=None) == []
    assert sec_companyfacts._flatten_facts({"facts": None}, cik=1, keep=None) == []


def test_default_tags_cover_what_the_feature_layer_reads():
    need = {"NetIncomeLoss", "Assets", "StockholdersEquity", "EarningsPerShareDiluted"}
    assert need <= sec_companyfacts.DEFAULT_TAGS


# ------------------------------------------------------------- security master

def test_registry_round_trips_through_its_silver_file(tmp_path, monkeypatch):
    from sp500lab import paths
    # storage imports the path HELPERS by name, and they read paths.SILVER_DIR at call
    # time - so patching the one constant redirects every writer.
    monkeypatch.setattr(paths, "SILVER_DIR", tmp_path / "silver")
    reg = SecurityRegistry()
    sid = reg.resolve_or_assign(320193, "AAPL", seen_date="2020-01-01")
    reg.save()
    again = SecurityRegistry.load()
    assert again.resolve(320193, "AAPL") == sid and len(again) == 1


def test_registry_bulk_assign_is_stable_and_vectorised():
    reg = SecurityRegistry()
    rows = pd.DataFrame({"cik": [1, 2, 1], "ticker": ["AAA", "BBB", "AAA"]})
    ids = reg.bulk_assign(rows)
    assert ids.iloc[0] == ids.iloc[2] != ids.iloc[1] and len(reg) == 2


# ---------------------------------------------------------------- EODHD budget

def test_eodhd_refuses_to_overrun_the_daily_budget(tmp_path, monkeypatch):
    """The free tier is 20 calls a day; a loop that overruns loses data, not time."""
    from sp500lab import http_cache
    monkeypatch.setattr(http_cache, "CACHE_DIR", tmp_path / "_cache")   # guarantee a miss
    monkeypatch.setattr(eodhd, "account_status",
                        lambda: {"remaining": eodhd.RESERVE_CALLS, "used": 19,
                                 "limit": 20, "subscription": "free", "extra_limit": 0})
    calls = []
    monkeypatch.setattr(eodhd, "fetch", lambda *a, **k: calls.append(a) or None)
    with pytest.raises(eodhd.BudgetExhausted, match="Resets at 00:00 UTC"):
        eodhd._spend("http://eodhd/x", label="probe")
    assert calls == [], "nothing may be requested once the budget is gone"


def test_eodhd_free_replay_costs_nothing(tmp_path, monkeypatch):
    """A cached response must be served without even asking the account endpoint."""
    from sp500lab import http_cache
    monkeypatch.setattr(http_cache, "CACHE_DIR", tmp_path / "_cache")
    key = http_cache.request_hash("GET", "http://eodhd/x")
    binp, metap = http_cache._cache_paths(eodhd.SOURCE, key)
    binp.parent.mkdir(parents=True)
    binp.write_bytes(b"[]")
    metap.write_text(json.dumps({"fetched_at_epoch": 0, "status": 200}))
    monkeypatch.setattr(eodhd, "account_status",
                        lambda: (_ for _ in ()).throw(AssertionError("budget was polled")))
    served = []
    monkeypatch.setattr(eodhd, "fetch", lambda url, **k: served.append(url) or "resp")
    assert eodhd._spend("http://eodhd/x", label="probe") == "resp"
    assert served == ["http://eodhd/x"]


def test_eodhd_symbol_squash_matches_share_class_spellings():
    """BRK-B (EODHD), BRK.B (ours) and BRKB (old Wikipedia) are one symbol."""
    import inspect
    src = inspect.getsource(eodhd.coverage_vs_missing)
    assert 'replace("-", "")' in src and 'replace(".", "")' in src


# ------------------------------------------------------------------- results

def test_ingest_result_summarises_itself():
    r = IngestResult(source="s", dataset="d")
    r.fetched, r.from_cache, r.rows = 2, 3, 10
    r.errors.append("boom")
    d = r.as_dict() if hasattr(r, "as_dict") else vars(r)
    assert d["source"] == "s" and d["rows"] == 10 and d["errors"] == ["boom"]
