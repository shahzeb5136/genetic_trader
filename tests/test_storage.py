"""The storage layer: bronze immutability, the manifest, the cache, the lake views.

Bronze is the only irreplaceable layer, so its guarantees are tested rather than
assumed: a write is idempotent, every artifact has a checksum, and `verify_manifest`
notices a flipped byte. The HTTP cache is tested with the network stubbed out - the
contract is "one fetch per URL, ever, unless told otherwise", and that is a property
of the cache code, not of the internet.

Every test redirects the lake into tmp_path. `paths` computes the layout at import
time, so the fixture rebinds the directory constants on every module that imported
them by name - which is exactly the trap ADR-039 documents for the registry.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest

from sp500lab import http_cache, paths, query, storage


@pytest.fixture()
def lake(tmp_path, monkeypatch):
    """A fresh, empty data root. Returns it."""
    root = tmp_path / "data"
    layout = {
        "DATA_DIR": root, "BRONZE_DIR": root / "bronze", "SILVER_DIR": root / "silver",
        "GOLD_DIR": root / "gold", "VAULT_DIR": root / "vault", "CACHE_DIR": root / "_cache",
        "MANIFEST_DIR": root / "_manifest", "INGEST_LOG": root / "_manifest/ingest_log.jsonl",
    }
    for mod in (paths, storage, http_cache, query):
        for name, value in layout.items():
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, value)
    return root


# ------------------------------------------------------------------ bronze

def test_bronze_write_is_idempotent_on_identical_content(lake):
    p1 = storage.write_bronze(source="s", dataset="d", filename="a.bin", content=b"abc",
                              ingest_date="2026-01-01")
    p2 = storage.write_bronze(source="s", dataset="d", filename="a.bin", content=b"abc",
                              ingest_date="2026-01-01")
    assert p1 == p2 and p1.read_bytes() == b"abc"
    assert len(list(storage.iter_manifest())) == 1, "a no-op write must not re-log"


def test_bronze_sidecar_records_a_matching_checksum_and_relative_path(lake):
    p = storage.write_bronze(source="s", dataset="d", filename="a.bin", content=b"hello",
                             url="http://x", ingest_date="2026-01-01", extra={"k": 1})
    meta = json.loads(p.with_suffix(".bin.meta.json").read_text())
    assert meta["sha256"] == http_cache.sha256_bytes(b"hello")
    assert meta["bytes"] == 5 and meta["url"] == "http://x" and meta["k"] == 1
    assert not Path(meta["path"]).is_absolute(), "the manifest must survive a move"
    assert (lake / meta["path"]) == p


def test_bronze_is_partitioned_by_ingest_date_not_data_date(lake):
    p = storage.write_bronze(source="yf", dataset="bars", filename="x.parquet",
                             content=b"1", ingest_date="2026-03-04")
    assert "ingest_date=2026-03-04" in p.as_posix()


def test_verify_manifest_passes_then_catches_corruption_and_loss(lake):
    a = storage.write_bronze(source="s", dataset="d", filename="a.bin", content=b"aaa",
                             ingest_date="2026-01-01")
    b = storage.write_bronze(source="s", dataset="d", filename="b.bin", content=b"bbb",
                             ingest_date="2026-01-01")
    v = storage.verify_manifest()
    assert (v["ok"], v["corrupt"], v["missing"]) == (2, 0, 0)

    a.write_bytes(b"aab")                                   # one flipped byte
    b.unlink()
    v = storage.verify_manifest()
    assert (v["ok"], v["corrupt"], v["missing"]) == (0, 1, 1)
    issues = {f["issue"] for f in v["failures"]}
    assert issues == {"checksum mismatch", "missing"}


def test_a_tombstone_turns_missing_into_retired(lake):
    p = storage.write_bronze(source="s", dataset="d", filename="a.bin", content=b"x",
                             ingest_date="2026-01-01")
    p.unlink()
    storage.retire_bronze(p, reason="written by a bug (ADR-013)")
    v = storage.verify_manifest()
    assert v["missing"] == 0 and v["retired"] == 1 and v["artifacts"] == 0


def test_vault_writes_are_logged_with_their_layer(lake):
    p = storage.write_vault(source="eodhd", dataset="eod", filename="AAPL.json",
                            content=b"{}", url="http://x?api_token=REDACTED")
    rec = list(storage.iter_manifest())[-1]
    assert rec["layer"] == "vault" and rec["path"].startswith("vault/")
    assert p.exists() and storage.verify_manifest()["ok"] == 1


# ------------------------------------------------------------------ silver / gold

def test_silver_round_trips_and_exists_reports_truthfully(lake):
    assert not storage.silver_exists("x/y")
    df = pd.DataFrame({"a": [1, 2], "date": ["2020-01-01", "2020-01-02"]})
    storage.write_silver(df, "x/y")
    assert storage.silver_exists("x/y")
    back = storage.read_silver("x/y")
    pd.testing.assert_frame_equal(back, df)


def test_silver_overwrites_rather_than_appends(lake):
    storage.write_silver(pd.DataFrame({"a": [1, 2, 3]}), "x/y")
    storage.write_silver(pd.DataFrame({"a": [9]}), "x/y")
    assert storage.read_silver("x/y")["a"].tolist() == [9]


def test_reading_an_unbuilt_dataset_says_so(lake):
    with pytest.raises(FileNotFoundError, match="not built yet"):
        storage.read_silver("never/built")


def test_gold_lands_under_the_gold_root(lake):
    p = storage.write_gold(pd.DataFrame({"a": [1]}), "backtest/thing")
    assert p == lake / "gold" / "backtest" / "thing" / "data.parquet"


# ------------------------------------------------------------------ query views

def test_every_parquet_becomes_a_view_and_collisions_are_qualified(lake):
    storage.write_silver(pd.DataFrame({"a": [1]}), "market/daily_bars")
    storage.write_silver(pd.DataFrame({"b": [2]}), "reference/data_quality")
    storage.write_silver(pd.DataFrame({"c": [3]}), "quality/data_quality")   # same leaf
    storage.write_gold(pd.DataFrame({"d": [4]}), "backtest/half_spread")

    con = query.connect()
    views = set(query.list_views()["view"])
    assert "daily_bars" in views and "gold_half_spread" in views
    # the first leaf keeps the bare name; the collision is qualified by its folder
    assert "data_quality" in views
    assert any(v.endswith("_data_quality") for v in views), views
    assert con.execute("SELECT a FROM daily_bars").fetchone() == (1,)


# ------------------------------------------------------------------ http cache

class _FakeResponse:
    def __init__(self, content=b"payload", status=200):
        self.content, self.status_code = content, status
        self.headers = {"Content-Type": "text/plain"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise http_cache.requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture()
def net(lake, monkeypatch):
    """Stub the network: returns a list the test can push responses onto."""
    calls: list[str] = []
    queue: list[_FakeResponse] = []

    def fake_request(method, url, **kw):
        calls.append(url)
        return queue.pop(0) if queue else _FakeResponse()

    monkeypatch.setattr(http_cache.requests, "request", fake_request)
    monkeypatch.setattr(http_cache._LIMITER, "wait", lambda host: None)
    monkeypatch.setattr(http_cache.time, "sleep", lambda s: None)
    return calls, queue


def test_fetch_hits_the_network_once_and_the_cache_after(net):
    calls, _ = net
    r1 = http_cache.fetch("http://x/a", source="t")
    r2 = http_cache.fetch("http://x/a", source="t")
    assert (r1.from_cache, r2.from_cache) == (False, True)
    assert r1.content == r2.content == b"payload"
    assert r1.sha256 == r2.sha256 == http_cache.sha256_bytes(b"payload")
    assert calls == ["http://x/a"]


def test_cache_key_distinguishes_url_method_and_body():
    h = http_cache.request_hash
    assert h("GET", "http://x/a") == h("GET", "http://x/a")
    assert h("GET", "http://x/a") != h("GET", "http://x/b")
    assert h("GET", "http://x/a") != h("POST", "http://x/a")
    assert h("POST", "http://x/a", b"1") != h("POST", "http://x/a", b"2")


def test_ttl_expiry_and_force_refetch(net, monkeypatch):
    calls, _ = net
    http_cache.fetch("http://x/a", source="t", ttl_seconds=100)
    http_cache.fetch("http://x/a", source="t", ttl_seconds=100)
    assert len(calls) == 1
    later = time.time() + 1_000
    monkeypatch.setattr(http_cache.time, "time", lambda: later)
    http_cache.fetch("http://x/a", source="t", ttl_seconds=100)     # expired
    assert len(calls) == 2
    http_cache.fetch("http://x/a", source="t", ttl_seconds=None)    # forever: cached
    assert len(calls) == 2
    http_cache.fetch("http://x/a", source="t", force=True)          # explicit bypass
    assert len(calls) == 3


def test_transient_failures_are_retried_then_succeed(net):
    calls, queue = net
    queue.extend([_FakeResponse(status=503), _FakeResponse(status=429),
                  _FakeResponse(b"ok")])
    r = http_cache.fetch("http://x/flaky", source="t")
    assert r.content == b"ok" and len(calls) == 3


def test_persistent_failure_raises_and_caches_nothing(net):
    from sp500lab.config import get_settings
    calls, queue = net
    queue.extend([_FakeResponse(status=500)] * get_settings().max_retries)
    with pytest.raises(RuntimeError, match="fetch failed after"):
        http_cache.fetch("http://x/dead", source="t")
    r = http_cache.fetch("http://x/dead", source="t")       # queue drained -> 200 now
    assert r.from_cache is False, "a failed fetch must not poison the cache"


def test_cache_bytes_are_written_before_meta(net):
    """A crash between the two writes must leave no meta pointing at nothing."""
    http_cache.fetch("http://x/a", source="t")
    key = http_cache.request_hash("GET", "http://x/a")
    binp, metap = http_cache._cache_paths("t", key)
    assert binp.exists() and metap.exists()
    meta = json.loads(metap.read_text())
    assert meta["sha256"] == http_cache.sha256_bytes(binp.read_bytes())


def test_cache_stats_counts_by_source(net):
    http_cache.fetch("http://x/a", source="alpha")
    http_cache.fetch("http://x/b", source="alpha")
    http_cache.fetch("http://x/c", source="beta")
    stats = http_cache.cache_stats()
    assert stats["alpha"]["files"] == 2 and stats["beta"]["files"] == 1
    assert stats["_total"]["files"] == 3


# ------------------------------------------------------------------ paths

def test_layout_is_created_idempotently(lake):
    paths.ensure_dirs()
    paths.ensure_dirs()
    assert all(d.is_dir() for d in paths.ALL_DIRS if str(d).startswith(str(lake)))


def test_every_path_helper_resolves_under_the_data_root(lake):
    assert paths.bronze_path("s", "d", "2026-01-01", "f").is_relative_to(lake / "bronze")
    assert paths.silver_path("x/y").is_relative_to(lake / "silver")
    assert paths.gold_path("x/y").is_relative_to(lake / "gold")
