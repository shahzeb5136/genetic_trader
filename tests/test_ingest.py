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


# --------------------------------------------------------------- Fama-French

FF5_TEXT = """This file was created by CMPT_ME_BEME_OP_INV_RETS_DAILY using the 202508 CRSP database.
The 1-month TBill return is from Ibbotson and Associates, Inc.

,Mkt-RF,SMB,HML,RMW,CMA,RF
19630701,-0.67,0.02,-0.35,0.03,0.13,0.012
19630702,0.79,-0.28,0.28,-0.08,-0.21,0.012
19630703,0.63,-0.18,-0.10,0.13,-0.25,0.012
19630705,-99.99,0.09,-0.10,0.15,-0.05,0.012

Copyright 2025 Kenneth R. French
"""

MOM_TEXT = """This file was created using the 202508 CRSP database.

,Mom
19261103,0.56
19630701,-0.11
19630702,0.18
"""


def test_ff_parser_finds_the_header_and_converts_percent_to_decimal():
    from sp500lab.ingest.fama_french import FILES, parse_factor_csv
    df = parse_factor_csv(FF5_TEXT, FILES["F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"])
    assert df["date"].tolist() == ["1963-07-01", "1963-07-02", "1963-07-03", "1963-07-05"]
    assert df.loc[0, "mkt_rf"] == pytest.approx(-0.0067)
    assert df.loc[0, "rf"] == pytest.approx(0.00012)
    assert np.isnan(df.loc[3, "mkt_rf"]), "-99.99 is the library's missing marker"
    assert list(df.columns) == ["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]


def test_ff_parser_stops_at_the_copyright_trailer_and_rejects_the_wrong_file():
    from sp500lab.ingest.fama_french import parse_factor_csv
    mom = parse_factor_csv(MOM_TEXT, ("Mom",))
    assert len(mom) == 3 and mom.loc[0, "mom"] == pytest.approx(0.0056)
    with pytest.raises(ValueError, match="no header row"):
        parse_factor_csv(MOM_TEXT, ("Mkt-RF", "SMB"))
    with pytest.raises(ValueError, match="no data rows"):
        parse_factor_csv(",Mom\nCopyright\n", ("Mom",))


def test_ff_zip_reader_takes_the_csv_inside():
    import io
    import zipfile

    from sp500lab.ingest.fama_french import _csv_from_zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("F-F_Momentum_Factor_daily.CSV", MOM_TEXT)
    assert "19261103,0.56" in _csv_from_zip(buf.getvalue())
    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w") as z:
        z.writestr("readme.txt", "nothing")
    with pytest.raises(ValueError, match="no CSV"):
        _csv_from_zip(empty.getvalue())


def test_ff_files_and_columns_agree():
    from sp500lab.ingest.fama_french import COLUMNS, FILES
    for expect in FILES.values():
        assert all(c in COLUMNS for c in expect)


# ------------------------------------------------------- the integrity gate

def _series(ticker, n=100, start="2015-01-01", close=None, **cols):
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d")
    c = np.asarray(close if close is not None else np.linspace(50, 60, n), dtype=float)
    df = pd.DataFrame({"ticker": ticker, "date": dates, "open": c * 0.999, "high": c * 1.01,
                       "low": c * 0.99, "close": c, "volume": 1e6, "split_ratio": 0.0})
    for k, v in cols.items():
        df[k] = v
    return df


def _intervals(rows):
    return pd.DataFrame(rows, columns=["security_id", "ticker", "start_date", "end_date",
                                       "end_is_open", "source"])


def _cal(start="2014-01-01", n=800):
    return pd.DataFrame({"date": pd.bdate_range(start, periods=n).strftime("%Y-%m-%d")})


def test_gate_passes_a_clean_series():
    from sp500lab.ingest.prices_yfinance import series_integrity
    iv = _intervals([("S1", "AAA", "2015-01-01", None, True, "w")])
    v = series_integrity(_series("AAA"), iv, _cal())
    assert len(v) == 1 and not v.loc[0, "reject"] and v.loc[0, "live"]


def test_gate_tolerates_a_few_bad_prints_on_a_live_member_and_none_on_a_dead_one():
    """A current constituent with a couple of vendor glitches must still enter silver
    (its bad rows become ERRORs for review, ADR-040); a delisted series with even one
    impossible bar is rejected, because rejecting it loses nothing."""
    from sp500lab.ingest.prices_yfinance import MAX_BAD_OHLC_ROWS, series_integrity
    live = _intervals([("S1", "LIVE", "2015-01-01", None, True, "w")])
    dead = _intervals([("S1", "DEAD", "2015-01-01", "2018-01-01", False, "w")])

    b = _series("LIVE")
    b.loc[:MAX_BAD_OHLC_ROWS, "low"] = b["open"] * 1.0005      # one more than tolerated
    v = series_integrity(b, live, None)
    assert v.loc[0, "reject"] and "impossible OHLC" in v.loc[0, "reasons"]
    ok = _series("LIVE")
    ok.loc[:MAX_BAD_OHLC_ROWS - 1, "low"] = ok["open"] * 1.0005  # exactly tolerated
    assert not series_integrity(ok, live, None).loc[0, "reject"]

    d = _series("DEAD")
    d.loc[0, "low"] = d.loc[0, "open"] * 1.0005                   # a single bad print
    v = series_integrity(d, dead, None)
    assert v.loc[0, "reject"] and v.loc[0, "reasons"] == "1 impossible OHLC rows"
    assert not series_integrity(_series("DEAD"), dead, None).loc[0, "reject"]


def test_gate_shares_the_reviewed_allowlist_with_the_quality_battery():
    """A delisted name with one REVIEWED bad print (KNOWN_BAD_BARS) keeps its history;
    the same print on an unreviewed date rejects it. One allowlist, two consumers."""
    from sp500lab.ingest.prices_yfinance import series_integrity
    from sp500lab.quality.checks import KNOWN_BAD_BARS
    (ticker, date), _ = next(iter(KNOWN_BAD_BARS.items()))
    dead = _intervals([("S1", ticker, "2010-01-01", "2024-12-31", False, "w")])
    b = _series(ticker, n=3, start=date)
    b.loc[0, "low"] = b.loc[0, "open"] * 1.0005            # the reviewed row
    assert not series_integrity(b, dead, None).loc[0, "reject"]
    b.loc[2, "low"] = b.loc[2, "open"] * 1.0005            # an unreviewed one
    v = series_integrity(b, dead, None)
    assert v.loc[0, "reject"] and v.loc[0, "reasons"] == "1 impossible OHLC rows"


def test_gate_rejects_a_wild_day_inside_membership_but_not_outside_it():
    """The panel clips to membership, so a shell's post-delisting pennies do not matter;
    a 5,000% day while the name was in the index is corrupt data."""
    from sp500lab.ingest.prices_yfinance import series_integrity
    close = np.full(100, 50.0)
    close[80:] = 5000.0                                          # the jump is at bar 80
    dates = pd.bdate_range("2015-01-01", periods=100).strftime("%Y-%m-%d")
    left_before_jump = _intervals([("S1", "SHELL", "2015-01-01", dates[50], False, "w")])
    still_in = _intervals([("S1", "SHELL", "2015-01-01", None, True, "w")])
    b = _series("SHELL", close=close)
    assert not series_integrity(b, left_before_jump, None).loc[0, "reject"]
    v = series_integrity(b, still_in, None)
    assert v.loc[0, "reject"] and "inside membership" in v.loc[0, "reasons"]


def test_gate_forgives_a_jump_on_a_recorded_split_day():
    from sp500lab.ingest.prices_yfinance import series_integrity
    close = np.full(100, 50.0)
    close[80:] = 500.0                                           # a 10:1 reverse split
    b = _series("RS", close=close)
    b.loc[80, "split_ratio"] = 0.1
    iv = _intervals([("S1", "RS", "2015-01-01", None, True, "w")])
    assert not series_integrity(b, iv, None).loc[0, "reject"]


def test_gate_rejects_a_stub_but_never_a_live_member_for_coverage():
    from sp500lab.ingest.prices_yfinance import series_integrity
    dead = _intervals([("S1", "STUB", "2011-01-01", "2016-01-31", False, "w")])
    live = _intervals([("S1", "STUB", "2011-01-01", None, True, "w")])
    stub = _series("STUB", n=100).iloc[::4]                     # 25% of its own span
    assert series_integrity(stub, dead, _cal()).loc[0, "reject"]
    v = series_integrity(stub, live, _cal())
    assert not v.loc[0, "reject"] and v.loc[0, "coverage"] < 0.5, \
        "a hole in a live constituent is reported, never dropped"


def test_replay_rebuilds_silver_from_a_bronze_partition_without_the_network(tmp_path,
                                                                         monkeypatch):
    """`ingest prices --from-bronze DATE` is the rollback (ADR-043). It must read every
    chunk that pull wrote, whatever the request keys were, run the gate, carry nothing
    forward from the silver it is replacing, and never touch the network."""
    import sys

    from sp500lab import http_cache, paths, query, storage
    from sp500lab.ingest import prices_yfinance as P
    from sp500lab.registry import SecurityRegistry

    root = tmp_path / "data"
    layout = {"DATA_DIR": root, "BRONZE_DIR": root / "bronze", "SILVER_DIR": root / "silver",
              "GOLD_DIR": root / "gold", "VAULT_DIR": root / "vault",
              "CACHE_DIR": root / "_cache", "MANIFEST_DIR": root / "_manifest",
              "INGEST_LOG": root / "_manifest/ingest_log.jsonl"}
    for mod in (paths, storage, http_cache, query):
        for name, value in layout.items():
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, value)
    monkeypatch.setattr(P.SecurityRegistry, "load", classmethod(lambda cls: SecurityRegistry()))

    class NoNetwork:
        @staticmethod
        def download(*a, **k):
            raise AssertionError("a replay must not fetch")
    monkeypatch.setitem(sys.modules, "yfinance", NoNetwork)

    # the universe the caller asks for, and a stale silver that must NOT leak through
    storage.write_silver(pd.DataFrame({"security_id": ["S1", "S2", "S3"],
                                       "ticker": ["AAA", "BAD", "GONE"]}),
                         "reference/sp500_current")
    stale = _series("GONE", n=5).assign(security_id="S3", source="yfinance",
                                        ingest_date="2026-01-01").drop(columns=["split_ratio"])
    storage.write_silver(stale, "market/daily_bars")

    # two chunk files under an old partition with arbitrary request keys
    part = paths.bronze_path(P.SOURCE, P.DATASET, "2026-01-01", "x").parent
    part.mkdir(parents=True)
    good = _series("AAA", n=40, dividend=0.0)
    bad = _series("BAD", n=40, dividend=0.0)
    bad["close"] = -1.0                                        # every bar impossible
    for i, frame in enumerate((good, bad), 1):
        frame = frame.assign(date=pd.to_datetime(frame["date"]))
        frame.to_parquet(part / f"bars_chunk_00{i}_deadbeef{i}.parquet", index=False)

    res = P.run(universe="current", ingest_date="2026-01-01")

    assert not res.errors, res.errors
    assert res.from_cache == 2 and res.fetched == 0
    bars = storage.read_silver("market/daily_bars")
    assert set(bars["ticker"]) == {"AAA"}, "BAD is rejected; GONE is not carried forward"
    assert len(bars) == 40 and (bars["ingest_date"] == "2026-01-01").all()
    assert res.notes["rejected_tickers"] == {"BAD": "40 impossible OHLC rows"}
    assert res.notes["carried_forward_tickers"] == []
    assert (part / "rejected_tickers.json").exists()


def test_gate_report_has_one_row_per_ticker_with_the_reason_text():
    from sp500lab.ingest.prices_yfinance import series_integrity
    bad = _series("BAD")
    bad["close"] = -1.0
    v = series_integrity(pd.concat([_series("AAA"), bad]), None, None).set_index("ticker")
    assert list(v.index) == ["AAA", "BAD"]
    assert v.loc["BAD", "bad_ohlc"] == 100 and v.loc["BAD", "reject"]
    assert v.loc["AAA", "reasons"] == ""
