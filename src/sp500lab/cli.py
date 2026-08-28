"""Command line entrypoint.

    python -m sp500lab <command> [options]

Run `python -m sp500lab --help` for the full list. Everything is idempotent: re-running
a command replays from cache and rewrites the same silver tables, so it is always safe
to run again after a crash or an interruption.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*utcnow.*")

log = logging.getLogger("sp500lab.cli")


# ------------------------------------------------------------------ helpers

def _print_result(res) -> None:
    print()
    print(res.summary())
    if res.notes:
        print(json.dumps(res.notes, indent=2, default=str))
    if res.errors:
        print(f"\n{len(res.errors)} error(s); first few:")
        for e in res.errors[:10]:
            print(f"  - {e}")


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


# ----------------------------------------------------------------- commands

def cmd_init(args) -> int:
    from .paths import ALL_DIRS, ensure_dirs
    ensure_dirs()
    print("Created project layout:")
    for d in ALL_DIRS:
        print(f"  {d}")
    return 0


def cmd_ingest(args) -> int:
    from .ingest import (benchmarks, fred, prices_yfinance, sec_companyfacts,
                         sec_tickers, wikipedia_history, wikipedia_sp500)

    jobs = {
        "sec-tickers":  lambda: sec_tickers.run(force=args.force),
        "wiki-current": lambda: wikipedia_sp500.run(force=args.force),
        "wiki-history": lambda: wikipedia_history.run(force=args.force, start=args.start,
                                                      limit=args.limit),
        "benchmarks":   lambda: benchmarks.run(force=args.force, start=args.start),
        "fred":         lambda: fred.run(force=args.force),
        "prices":       lambda: prices_yfinance.run(force=args.force,
                                                    universe=args.universe,
                                                    start=args.start, limit=args.limit),
        "fundamentals": lambda: sec_companyfacts.run(force=args.force,
                                                     universe=args.universe,
                                                     limit=args.limit,
                                                     all_tags=args.all_tags),
    }

    # Dependency-ordered: identity -> universe -> calendar -> prices -> fundamentals
    ALL_ORDER = ["sec-tickers", "wiki-current", "wiki-history",
                 "benchmarks", "fred", "prices", "fundamentals"]

    targets = ALL_ORDER if args.dataset == "all" else [args.dataset]
    failed = 0
    for name in targets:
        print(f"\n{'=' * 72}\n== {name}\n{'=' * 72}")
        try:
            _print_result(jobs[name]())
        except Exception as exc:  # noqa: BLE001
            failed += 1
            log.exception("%s failed", name)
            print(f"  FAILED: {exc}")
    return 1 if failed else 0


def cmd_normalize(args) -> int:
    from .normalize import adjustments
    print(json.dumps(adjustments.run(convention=args.convention), indent=2, default=str))
    return 0


def cmd_quality(args) -> int:
    from .quality import checks
    rep = checks.run()
    print()
    worst = 0
    for r in rep.itertuples():
        print(f"[{r.severity:5s}] {r.check:22s} {r.detail}")
        if isinstance(r.sample, str) and r.sample:
            print(f"          e.g. {r.sample[:160]}")
        if r.severity == "ERROR":
            worst = 1
    return worst if args.strict else 0


def cmd_verify(args) -> int:
    from .storage import verify_manifest
    v = verify_manifest()
    print(json.dumps({k: v[k] for k in ("artifacts", "ok", "missing", "corrupt", "retired")},
                     indent=2))
    if v["failures"]:
        print("\nfailures:")
        for f in v["failures"][:20]:
            print(f"  {f['issue']:20s} {f['path']}")
        return 1
    print("\nAll bronze artifacts match their recorded checksums.")
    return 0


def cmd_status(args) -> int:
    from .http_cache import cache_stats
    from .paths import BRONZE_DIR, DATA_DIR, VAULT_DIR
    from .query import list_views

    print("=" * 72)
    print(f"data root: {DATA_DIR}")
    print("=" * 72)

    views = list_views()
    if views.empty:
        print("\nNo datasets built yet. Run:  python -m sp500lab ingest all")
        return 0

    print("\nDATASETS")
    print(views[["view", "rows", "mb"]].to_string(index=False))

    bronze_bytes = sum(f.stat().st_size for f in BRONZE_DIR.rglob("*") if f.is_file())
    vault_bytes = sum(f.stat().st_size for f in VAULT_DIR.rglob("*") if f.is_file())
    print("\nSTORAGE")
    print(f"  bronze (raw, immutable)   {_human(bronze_bytes)}")
    print(f"  silver (normalized)       {_human(views['mb'].sum() * 1e6)}")
    print(f"  vault  (paid-window)      {_human(vault_bytes)}")
    print(f"  http cache                {cache_stats()['_total']['mb']:,.1f} MB")

    from .storage import silver_exists
    if silver_exists("quality/data_quality"):
        from .storage import read_silver
        q = read_silver("quality/data_quality")
        counts = q["severity"].value_counts().to_dict()
        print("\nQUALITY  " + "  ".join(f"{k}={v}" for k, v in counts.items())
              + "   (run `quality` for detail)")
    return 0


def cmd_query(args) -> int:
    from .query import connect, list_views
    if args.list:
        print(list_views().to_string(index=False))
        return 0
    if not args.sql:
        print("provide SQL, or --list to see available views")
        return 2
    con = connect()
    df = con.sql(args.sql).df()
    if len(df) <= args.limit:
        print(df.to_string(index=False))
    else:
        print(df.head(args.limit).to_string(index=False))
        print(f"... {len(df) - args.limit} more rows")
    return 0


# -------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sp500lab",
        description="Survivorship-bias-aware S&P 500 research platform: "
                    "data layer + backtest engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
typical first run:
  python -m sp500lab init
  python -m sp500lab ingest all
  python -m sp500lab normalize
  python -m sp500lab quality
  python -m sp500lab status

then, to backtest:
  python -m sp500lab backtest build-delisting   # exit assumptions -> gold/
  python -m sp500lab backtest build-spreads     # cost inputs      -> gold/
  python -m sp500lab backtest accept            # MUST pass before trusting anything
  python -m sp500lab backtest baselines

research discipline (docs/EXPERIMENTS.md):
  every run is logged as a trial, and backtests stop before the reserved holdout
  python -m sp500lab experiments studies        # what you have tried
  python -m sp500lab experiments deflate NAME   # does the winner survive the search?
  python -m sp500lab experiments holdout        # every look at the reserved period

visualise it (docs/REPORTS.md):
  python -m sp500lab report study baselines --open
  python -m sp500lab report registry
""")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the data/ layout").set_defaults(func=cmd_init)

    i = sub.add_parser("ingest", help="fetch a dataset into bronze + silver")
    i.add_argument("dataset", choices=["all", "sec-tickers", "wiki-current", "wiki-history",
                                       "benchmarks", "fred", "prices", "fundamentals"])
    i.add_argument("--force", action="store_true", help="bypass cache and re-fetch")
    i.add_argument("--start", default=None, help="start date (YYYY-MM-DD)")
    i.add_argument("--limit", type=int, default=None, help="cap items processed (testing)")
    i.add_argument("--universe", default="ever", choices=["ever", "current"],
                   help="'ever' = survivorship-free; 'current' = today's members only")
    i.add_argument("--all-tags", action="store_true",
                   help="fundamentals: keep every XBRL tag, not the curated set")
    i.set_defaults(func=cmd_ingest)

    n = sub.add_parser("normalize", help="compute corporate-action adjustment factors")
    n.add_argument("--convention", default=None, choices=["as_traded", "split_adjusted"],
                   help="override the price convention inferred from the source")
    n.set_defaults(func=cmd_normalize)

    q = sub.add_parser("quality", help="run data quality checks")
    q.add_argument("--strict", action="store_true", help="exit 1 if any ERROR found")
    q.set_defaults(func=cmd_quality)

    sub.add_parser("verify", help="re-hash bronze against the manifest"
                   ).set_defaults(func=cmd_verify)
    sub.add_parser("status", help="what exists, how big, quality summary"
                   ).set_defaults(func=cmd_status)

    from .backtest.cli import add_experiments_parser, add_parser as add_backtest_parser
    from .reporting.cli import add_parser as add_report_parser
    add_backtest_parser(sub)
    add_experiments_parser(sub)
    add_report_parser(sub)

    s = sub.add_parser("query", help="run SQL against the parquet lake")
    s.add_argument("sql", nargs="?", help="SQL to execute")
    s.add_argument("--list", action="store_true", help="list available views")
    s.add_argument("--limit", type=int, default=50, help="max rows to print")
    s.set_defaults(func=cmd_query)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .logging_setup import configure_logging
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
