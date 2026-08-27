"""Common contract for ingestors.

Every ingestor exposes `run(force: bool = False) -> IngestResult` and follows the
same three-step shape:

    1. fetch  -> bytes, via http_cache (cached, rate-limited, retried)
    2. bronze -> write raw bytes verbatim with checksum provenance
    3. silver -> parse into a normalized DataFrame and write parquet

Step 3 is always rebuildable from step 2, so a parser bug is fixed by re-running
the normalize step, never by re-downloading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IngestResult:
    source: str
    dataset: str
    rows: int = 0
    bronze_files: int = 0
    from_cache: int = 0
    fetched: int = 0
    errors: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        bits = [f"{self.source}/{self.dataset}", f"{self.rows} rows"]
        if self.bronze_files:
            bits.append(f"{self.bronze_files} artifacts")
        if self.fetched or self.from_cache:
            bits.append(f"net={self.fetched} cache={self.from_cache}")
        if self.errors:
            bits.append(f"ERRORS={len(self.errors)}")
        return "  |  ".join(bits)
