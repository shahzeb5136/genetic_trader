"""Bronze/silver/gold writers with checksum provenance.

Bronze artifacts are immutable. Every write emits a sidecar `.meta.json` and appends
a row to data/_manifest/ingest_log.jsonl recording url, sha256, byte count and
fetch time. `verify_manifest()` re-hashes every artifact to detect bit-rot or an
accidental edit - important because once a paid subscription lapses, a corrupted
bronze file is a permanent loss, not an inconvenience.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .config import get_settings
from .http_cache import sha256_bytes
from .paths import (DATA_DIR, INGEST_LOG, MANIFEST_DIR, VAULT_DIR, bronze_path,
                    gold_path, silver_path)

log = logging.getLogger(__name__)


def today_iso() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _append_manifest(record: dict[str, Any]) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(INGEST_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def write_bronze(
    *,
    source: str,
    dataset: str,
    filename: str,
    content: bytes,
    url: str = "",
    ingest_date: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist one raw artifact byte-for-byte, with provenance.

    Idempotent: if a file with identical content already exists at the same path,
    the write is skipped and the manifest is not double-appended.
    """
    ingest_date = ingest_date or today_iso()
    path = bronze_path(source, dataset, ingest_date, filename)
    digest = sha256_bytes(content)

    if path.exists() and sha256_bytes(path.read_bytes()) == digest:
        log.debug("bronze unchanged, skipping: %s", path.name)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)

    meta = {
        "source": source, "dataset": dataset, "filename": filename,
        # Stored relative to DATA_DIR so the manifest survives relocating data/.
        "path": path.relative_to(DATA_DIR).as_posix(),
        "url": url, "sha256": digest, "bytes": len(content),
        "ingest_date": ingest_date,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **(extra or {}),
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    _append_manifest(meta)
    log.info("bronze  %-10s %-24s %8.1f KB  %s", source, dataset, len(content) / 1024, filename)
    return path


def write_silver(df: pd.DataFrame, dataset: str, *, filename: str = "data.parquet") -> Path:
    """Write a normalized table. Overwrites - silver is always rebuildable from bronze."""
    path = silver_path(dataset, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression=get_settings().compression, index=False)
    log.info("silver  %-34s %7d rows  %6.2f MB", dataset, len(df), path.stat().st_size / 1e6)
    return path


def write_vault(
    *,
    source: str,
    dataset: str,
    filename: str,
    content: bytes,
    url: str = "",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist a vendor-licensed artifact to the vault.

    Vault vs bronze is about RE-FETCHABILITY, not about money. Bronze holds data we
    can pull again freely; vault holds data acquired under a subscription or a hard
    rate limit, where re-fetching is constrained or impossible. EODHD lands here even
    on the free tier, because 20 calls/day makes every response expensive - and when
    the paid plan arrives and later lapses, the same directory is already the thing
    that must be backed up.
    """
    path = VAULT_DIR / source / dataset / filename
    digest = sha256_bytes(content)
    if path.exists() and sha256_bytes(path.read_bytes()) == digest:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    meta = {
        "source": source, "dataset": dataset, "filename": filename,
        "path": path.relative_to(DATA_DIR).as_posix(), "url": url,
        "sha256": digest, "bytes": len(content), "layer": "vault",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **(extra or {}),
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    _append_manifest(meta)
    log.info("vault   %-10s %-24s %8.1f KB  %s", source, dataset, len(content) / 1024, filename)
    return path


def read_silver(dataset: str, *, filename: str = "data.parquet") -> pd.DataFrame:
    path = silver_path(dataset, filename)
    if not path.exists():
        raise FileNotFoundError(
            f"silver dataset '{dataset}' not built yet - expected {path}")
    return pd.read_parquet(path)


def silver_exists(dataset: str, *, filename: str = "data.parquet") -> bool:
    return silver_path(dataset, filename).exists()


def write_gold(df: pd.DataFrame, dataset: str, *, filename: str = "data.parquet") -> Path:
    path = gold_path(dataset, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression=get_settings().compression, index=False)
    log.info("gold    %-34s %7d rows", dataset, len(df))
    return path


def iter_manifest() -> Iterator[dict[str, Any]]:
    if not INGEST_LOG.exists():
        return
    for line in INGEST_LOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def retire_bronze(path: Path | str, reason: str) -> None:
    """Record that a bronze artifact was deliberately removed.

    The manifest is append-only, so a deleted file would otherwise be reported as
    `missing` by verify_manifest() forever - indistinguishable from data loss. A
    tombstone says "this absence is intentional, and here is why", which keeps the
    audit trail honest without weakening the integrity check.

    Deletion from bronze should be rare. The legitimate case is an artifact written
    by a bug (see ADR-013), not routine cleanup.
    """
    p = Path(path)
    relpath = p.relative_to(DATA_DIR).as_posix() if p.is_absolute() else Path(path).as_posix()
    _append_manifest({
        "path": relpath, "retired": True, "reason": reason,
        "retired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    log.info("retired bronze artifact: %s (%s)", relpath, reason)


def verify_manifest() -> dict[str, Any]:
    """Re-hash every bronze artifact against its recorded checksum.

    Returns counts and the list of failures. Run this after moving the data
    directory or restoring from backup. Artifacts with a tombstone (see
    retire_bronze) are counted separately rather than reported as missing.
    """
    seen: dict[str, dict[str, Any]] = {}
    for rec in iter_manifest():
        seen[rec["path"]] = rec  # last write wins, so a tombstone supersedes a write

    retired = [p for p, r in seen.items() if r.get("retired")]
    seen = {p: r for p, r in seen.items() if not r.get("retired")}

    ok = missing = corrupt = 0
    failures: list[dict[str, str]] = []

    for relpath, rec in seen.items():
        p = Path(relpath)
        if not p.is_absolute():
            p = DATA_DIR / relpath
        if not p.exists():
            missing += 1
            failures.append({"path": relpath, "issue": "missing"})
            continue
        if sha256_bytes(p.read_bytes()) != rec["sha256"]:
            corrupt += 1
            failures.append({"path": relpath, "issue": "checksum mismatch"})
            continue
        ok += 1

    return {"artifacts": len(seen), "ok": ok, "missing": missing,
            "corrupt": corrupt, "retired": len(retired), "failures": failures}
