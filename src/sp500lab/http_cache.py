"""Fetch-once HTTP client with on-disk cache, per-host rate limiting and retries.

Design rules this enforces (see docs/ARCHITECTURE.md):

1. **Fetch once.** Every response is written to data/_cache keyed by a hash of the
   request. Re-running an ingest job replays from disk and never re-hits the API.
   This matters little on free tiers with 100k req/day, but it is the habit that
   makes a metered burst-buy month survivable.
2. **Write before parse.** Bytes land on disk before anything tries to interpret
   them, so a parser bug never costs a re-download.
3. **Rate limit per host.** SEC publishes a 10 req/s ceiling and blocks abusers;
   we run under it via a token bucket, configured in settings.toml.
4. **Retry with backoff and jitter** on 429/5xx and transport errors.

The cache is keyed on (method, url, body). It deliberately ignores headers so that
rotating a User-Agent does not invalidate the whole cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .config import get_settings
from .paths import CACHE_DIR

log = logging.getLogger(__name__)


@dataclass
class Response:
    content: bytes
    url: str
    status: int
    content_type: str
    from_cache: bool
    fetched_at: str
    sha256: str

    def text(self, encoding: str = "utf-8") -> str:
        return self.content.decode(encoding, errors="replace")

    def json(self):
        return json.loads(self.content)


class _RateLimiter:
    """Token bucket per host. Thread-safe, process-local."""

    def __init__(self) -> None:
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        min_interval = 1.0 / get_settings().rate_limit_for(host)
        with self._lock:
            now = time.monotonic()
            last = self._last.get(host, 0.0)
            sleep_for = min_interval - (now - last)
            if sleep_for > 0:
                time.sleep(sleep_for)
                now = time.monotonic()
            self._last[host] = now


_LIMITER = _RateLimiter()


def request_hash(method: str, url: str, body: bytes | None = None) -> str:
    h = hashlib.sha256()
    h.update(method.upper().encode())
    h.update(b"\x00")
    h.update(url.encode())
    if body:
        h.update(b"\x00")
        h.update(body)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cache_paths(source: str, key: str) -> tuple[Path, Path]:
    d = CACHE_DIR / source
    return d / f"{key}.bin", d / f"{key}.meta.json"


def fetch(
    url: str,
    *,
    source: str,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    ttl_seconds: float | None = None,
    force: bool = False,
) -> Response:
    """Fetch a URL, serving from cache when possible.

    Parameters
    ----------
    source
        Logical source name ("sec", "wikipedia", ...). Only used to shard the
        cache directory so it stays browsable.
    ttl_seconds
        None means cache forever - correct for immutable history. Set a TTL for
        endpoints whose content changes (e.g. "today's constituent list").
    force
        Bypass the cache and overwrite. Use when you know upstream changed.
    """
    settings = get_settings()
    key = request_hash(method, url, body)
    bin_path, meta_path = _cache_paths(source, key)

    if not force and bin_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        age = time.time() - meta.get("fetched_at_epoch", 0)
        if ttl_seconds is None or age < ttl_seconds:
            content = bin_path.read_bytes()
            return Response(
                content=content, url=url, status=meta.get("status", 200),
                content_type=meta.get("content_type", ""), from_cache=True,
                fetched_at=meta.get("fetched_at", ""), sha256=meta.get("sha256", ""),
            )

    host = requests.utils.urlparse(url).netloc
    hdrs = {"User-Agent": settings.user_agent, "Accept-Encoding": "gzip, deflate"}
    if headers:
        hdrs.update(headers)

    last_exc: Exception | None = None
    for attempt in range(settings.max_retries):
        _LIMITER.wait(host)
        try:
            resp = requests.request(
                method, url, data=body, headers=hdrs,
                timeout=settings.timeout_seconds,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()

            content = resp.content
            digest = sha256_bytes(content)
            now = time.time()
            fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))

            bin_path.parent.mkdir(parents=True, exist_ok=True)
            # Write bytes BEFORE meta so a crash never leaves meta pointing at nothing.
            bin_path.write_bytes(content)
            meta_path.write_text(json.dumps({
                "url": url, "method": method, "status": resp.status_code,
                "content_type": resp.headers.get("Content-Type", ""),
                "bytes": len(content), "sha256": digest,
                "fetched_at": fetched_at, "fetched_at_epoch": now,
            }, indent=2), encoding="utf-8")

            return Response(
                content=content, url=url, status=resp.status_code,
                content_type=resp.headers.get("Content-Type", ""), from_cache=False,
                fetched_at=fetched_at, sha256=digest,
            )
        except Exception as exc:  # noqa: BLE001 - retry any transport/status failure
            last_exc = exc
            if attempt == settings.max_retries - 1:
                break
            delay = settings.backoff_base_seconds ** (attempt + 1)
            delay += random.uniform(0, delay * 0.3)  # jitter, avoid lockstep retries
            log.warning("fetch failed (%s/%s) %s: %s - retrying in %.1fs",
                        attempt + 1, settings.max_retries, url, exc, delay)
            time.sleep(delay)

    raise RuntimeError(f"fetch failed after {settings.max_retries} attempts: {url}") from last_exc


def cache_stats() -> dict[str, object]:
    """Size and file count of the response cache, by source."""
    out: dict[str, object] = {}
    total_bytes = total_files = 0
    if CACHE_DIR.exists():
        for src_dir in sorted(p for p in CACHE_DIR.iterdir() if p.is_dir()):
            files = list(src_dir.glob("*.bin"))
            nbytes = sum(f.stat().st_size for f in files)
            out[src_dir.name] = {"files": len(files), "mb": round(nbytes / 1e6, 2)}
            total_files += len(files)
            total_bytes += nbytes
    out["_total"] = {"files": total_files, "mb": round(total_bytes / 1e6, 2)}
    return out
