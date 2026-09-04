"""Canonical filesystem layout.

Every path in the project resolves through this module. Nothing else builds paths
by hand, so the layout can be relocated (external SSD, object store mount) by
changing SP500LAB_DATA_DIR alone.

Layout
------
data/bronze/   Raw vendor payloads, byte-for-byte as received. Append-only, never
               rewritten. This is the only irreplaceable layer once a paid
               subscription lapses - see docs/ARCHITECTURE.md "Why bronze is sacred".
data/silver/   Normalized, deduplicated, conformed to internal IDs. Rebuildable
               from bronze by re-running the normalize step.
data/gold/     Analysis-ready panels and features. Rebuildable from silver.
data/vault/    Downloads made during a *paid* subscription window. Physically
               separate from bronze so it is obvious what cannot be re-fetched.
data/_cache/   HTTP response cache keyed by request hash (fetch-once discipline).
data/_manifest/Append-only ingestion log with checksums and provenance.
data/experiments/ Append-only record of every backtest run, and the holdout ledger.
               Provenance for research rather than for ingestion: you cannot
               re-derive what you tried, so it is treated like bronze - append-only,
               never rewritten, and backed up. See docs/EXPERIMENTS.md.
data/experiments/forward/
               The out-of-sample record: pre-registrations, forward tests and their
               curves. The most irreplaceable directory in the project - a forward
               test consumes a period that only exists once. See docs/FORWARD_TEST.md.
reports/       Generated pages, one folder per lab: `backtest/` and `forward/`, an
               index and one page per algorithm each; `timing/`, the calendar rules;
               `genetic_algorithm/`, three pages on the search itself; and `extra/` for
               anything asked for on its own. Disposable. See docs/REPORTS.md, ADR-045,
               ADR-046 and ADR-047.
results/       Saved run artifacts: `forward/<forward_id>/` and `trades/<strategy>/`.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/sp500lab/paths.py -> src/sp500lab -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "config"
DOCS_DIR = PROJECT_ROOT / "docs"
LOGS_DIR = PROJECT_ROOT / "logs"

DATA_DIR = Path(os.environ.get("SP500LAB_DATA_DIR", PROJECT_ROOT / "data")).resolve()

BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
VAULT_DIR = DATA_DIR / "vault"
CACHE_DIR = DATA_DIR / "_cache"
MANIFEST_DIR = DATA_DIR / "_manifest"

EXPERIMENTS_DIR = DATA_DIR / "experiments"

INGEST_LOG = MANIFEST_DIR / "ingest_log.jsonl"
#: Every backtest run. One JSON object per line, append-only.
EXPERIMENT_LOG = EXPERIMENTS_DIR / "runs.jsonl"
#: Every time a run was allowed to see holdout data. Never disabled - see ADR-025.
HOLDOUT_LOG = EXPERIMENTS_DIR / "holdout_log.jsonl"
#: Month-end equity curves, one line per run, keyed by run_id. Split out from runs.jsonl
#: so the searchable index stays small - see ADR-027.
CURVE_LOG = EXPERIMENTS_DIR / "curves.jsonl"

#: Forward testing - the out-of-sample record. Kept in its own directory rather than
#: mixed into the trial log because these files are a different KIND of thing: there
#: will be a handful of them ever, each one spends a resource that cannot be replaced,
#: and none of them may be silenced by SP500LAB_REGISTRY=off. See ADR-033/034.
FORWARD_DIR = EXPERIMENTS_DIR / "forward"
#: One line per pre-registered candidate: what was predicted, before the look.
SEAL_LOG = FORWARD_DIR / "seals.jsonl"
#: One line per forward test: the prediction, the outcome, and the gap between them.
FORWARD_LOG = FORWARD_DIR / "forward_runs.jsonl"
#: Month-end curves for both windows of a forward test, keyed by forward_id.
FORWARD_CURVE_LOG = FORWARD_DIR / "forward_curves.jsonl"

#: Generated HTML reports. Disposable: rebuildable from the registry at any time.
#: One folder per lab, and nothing else (ADR-045, ADR-047). `backtest/` and `forward/`
#: are the monthly roster on either side of the 2022 boundary and each holds an index and
#: one page per algorithm. A page asked for on its own lands in `extra/`.
REPORTS_DIR = PROJECT_ROOT / "reports"
BACKTEST_REPORTS_DIR = REPORTS_DIR / "backtest"
FORWARD_REPORTS_DIR = REPORTS_DIR / "forward"
#: The calendar lab: the overnight/intraday decomposition and one page per calendar rule,
#: each carrying BOTH windows, because the family is small enough that a rule's research
#: and forward numbers are one story (ADR-047).
TIMING_REPORTS_DIR = REPORTS_DIR / "timing"
#: The genetic-algorithm lab: how the search works, what it may read, and what it found.
#: Three pages, no index - each one links to the other two (ADR-046).
GENETIC_REPORTS_DIR = REPORTS_DIR / "genetic_algorithm"
EXTRA_REPORTS_DIR = REPORTS_DIR / "extra"

#: Saved run artifacts that are not reports: forward-test results under `forward/`,
#: exported trade ledgers under `trades/<strategy>/`. Gitignored, reproducible.
RESULTS_DIR = PROJECT_ROOT / "results"

ALL_DIRS = [
    CONFIG_DIR, DOCS_DIR, LOGS_DIR, DATA_DIR,
    BRONZE_DIR, SILVER_DIR, GOLD_DIR, VAULT_DIR, CACHE_DIR, MANIFEST_DIR,
    EXPERIMENTS_DIR, FORWARD_DIR,
]


def ensure_dirs() -> None:
    """Create the full layout. Idempotent."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def bronze_path(source: str, dataset: str, ingest_date: str, filename: str) -> Path:
    """Path for one raw artifact.

    Partitioned by ingest_date (not the data's own date) because bronze records
    *when we fetched*, which is what makes re-derivation auditable.
    """
    return BRONZE_DIR / source / dataset / f"ingest_date={ingest_date}" / filename


def silver_path(dataset: str, filename: str = "data.parquet") -> Path:
    return SILVER_DIR / dataset / filename


def gold_path(dataset: str, filename: str = "data.parquet") -> Path:
    return GOLD_DIR / dataset / filename
