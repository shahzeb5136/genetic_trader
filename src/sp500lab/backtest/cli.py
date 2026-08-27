"""`sp500lab backtest ...` - the command surface of the engine.

Kept out of the top-level cli.py so that importing the CLI stays cheap. numpy, the
panel and the strategy registry only load when a backtest is actually requested.

    sp500lab backtest list
    sp500lab backtest accept
    sp500lab backtest run momentum_12_1 --costs realistic
    sp500lab backtest run momentum_12_1 --all-costs --top-k 30
    sp500lab backtest baselines
    sp500lab backtest build-spreads
    sp500lab backtest build-delisting
    sp500lab backtest coverage
"""

from __future__ import annotations

import json
import sys

from ..paths import PROJECT_ROOT


def add_parser(sub) -> None:
    """Register the `backtest` subcommand tree on the main parser."""
    p = sub.add_parser("backtest", help="run the backtest engine")
    bs = p.add_subparsers(dest="backtest_command", required=True)

    r = bs.add_parser("run", help="backtest one strategy")
    r.add_argument("strategy")
    r.add_argument("--start", default="2007-04-01")
    r.add_argument("--end", default=None)
    r.add_argument("--costs", default="realistic",
                   choices=["optimistic", "realistic", "pessimistic", "free"])
    r.add_argument("--all-costs", action="store_true",
                   help="report all three cost settings, as costs.py insists")
    r.add_argument("--capital", type=float, default=100_000.0)
    r.add_argument("--top-k", type=int, default=None, help="override the holding count")
    r.add_argument("--max-weight", type=float, default=None, help="per-name cap")
    r.add_argument("--weighting", default=None,
                   choices=["equal", "score", "score_rank", "inverse_vol"])
    r.add_argument("--liquidity-floor", type=float, default=0.0,
                   help="min trailing median dollar volume to be buyable")
    r.add_argument("--min-coverage", type=float, default=0.0,
                   help="refuse to run below this priced share of the index")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--benchmark", default="SPY")
    r.add_argument("--save", default=None, help="directory to write the result into")
    r.add_argument("--annual", action="store_true", help="print the year-by-year table")
    r.set_defaults(func=cmd_run)

    b = bs.add_parser("baselines", help="run every baseline and print the scoreboard")
    b.add_argument("--start", default="2007-04-01")
    b.add_argument("--end", default=None)
    b.add_argument("--costs", default="realistic")
    b.add_argument("--save", default=None)
    b.set_defaults(func=cmd_baselines)

    a = bs.add_parser("accept", help="the acceptance checks - run before trusting anything")
    a.add_argument("--start", default="2007-04-01")
    a.set_defaults(func=cmd_accept)

    ls = bs.add_parser("list", help="registered strategies")
    ls.set_defaults(func=cmd_list)

    c = bs.add_parser("coverage", help="priced share of the index, per rebalance")
    c.add_argument("--annual", action="store_true", help="collapse to year ends")
    c.set_defaults(func=cmd_coverage)

    sp = bs.add_parser("build-spreads", help="estimate half-spreads into gold/")
    sp.add_argument("--window", type=int, default=21)
    sp.set_defaults(func=cmd_build_spreads)

    dl = bs.add_parser("build-delisting", help="build delisting returns into gold/")
    dl.set_defaults(func=cmd_build_delisting)

    pn = bs.add_parser("build-panel", help="rebuild the cached panel")
    pn.add_argument("--start", default="2000-01-01")
    pn.set_defaults(func=cmd_build_panel)


# ----------------------------------------------------------------- commands

def cmd_run(args) -> int:
    from . import run_backtest
    from .results import compare, format_compare
    from .strategy import get_strategy

    strat = get_strategy(args.strategy)
    _apply_construction_overrides(strat, args)

    common = dict(start=args.start, end=args.end, initial_capital=args.capital,
                  liquidity_floor=args.liquidity_floor, seed=args.seed,
                  benchmark=args.benchmark, min_coverage=args.min_coverage)

    if args.all_costs:
        results = [run_backtest(strat, costs=c, **common)
                   for c in ("optimistic", "realistic", "pessimistic")]
        for res in results:
            print(res.summary())
            print()
        print(format_compare(compare(results)))
        _warn_if_only_optimistic(results)
        result = results[1]
    else:
        result = run_backtest(strat, costs=args.costs, **common)
        print(result.summary())

    if args.annual:
        print("\nANNUAL RETURNS")
        tab = result.annual_table()
        print((tab * 100).round(2).to_string())

    if args.save:
        out = result.save(args.save)
        print(f"\nsaved -> {out}")
    return 0


def cmd_baselines(args) -> int:
    from . import run_backtest
    from .results import compare, format_compare
    from .strategy import list_strategies

    names = [n for n in list_strategies() if n != "spy_buy_hold"]
    results = []
    for n in names:
        try:
            results.append(run_backtest(n, start=args.start, end=args.end,
                                        costs=args.costs))
        except Exception as exc:  # noqa: BLE001
            print(f"  {n}: FAILED - {exc}", file=sys.stderr)
    if not results:
        return 1

    print("=" * 78)
    print(f"BASELINES   {args.start} .. {args.end or 'latest'}   [{args.costs} costs]")
    print("=" * 78)
    print(format_compare(compare(results, benchmark_name="equal_weight")))
    print()
    print(results[0].diagnostics.get("price_coverage", ""))
    print("\nBuy-and-hold SPY, the benchmark every one of these has to beat:")
    from .accept import replicate_benchmark
    spy = replicate_benchmark("SPY")
    print(f"  SPY total return {spy['total_return_annualised'] * 100:.2f}%/yr "
          f"({spy['start']} .. {spy['end']})")

    if args.save:
        for res in results:
            res.save(f"{args.save}/{res.strategy}")
        print(f"\nsaved -> {args.save}")
    return 0


def cmd_accept(args) -> int:
    from .accept import report, run_all
    checks = run_all(start=args.start)
    print(report(checks))
    return 1 if any(not c.passed for c in checks) else 0


def cmd_list(args) -> int:
    from .strategy import get_strategy, list_strategies
    print(f"{'name':20s} {'class':22s} warmup  description")
    print("-" * 78)
    for n in list_strategies():
        s = get_strategy(n)
        doc = (s.__doc__ or "").strip().splitlines()
        print(f"{n:20s} {type(s).__name__:22s} {getattr(s, 'warmup', 0):6d}  "
              f"{doc[0] if doc else ''}")
    return 0


def cmd_coverage(args) -> int:
    from .panel import build_panel
    cov = build_panel().coverage().dropna()
    if args.annual:
        cov = cov[cov["date"].str.slice(5, 7) == "12"]
    out = cov.copy()
    out["coverage"] = (out["coverage"] * 100).round(1).astype(str) + "%"
    print("Point-in-time index members vs how many we actually hold prices for.")
    print("This is the ceiling on what any backtest over the early years can mean.\n")
    print(out.to_string(index=False))
    return 0


def cmd_build_spreads(args) -> int:
    from . import spreads
    df = spreads.build(smooth_window=args.window)
    print()
    print(spreads.summarise(df).to_string(index=False))
    print("\nSanity check: single-digit bp for modern large caps, wider in 2008-09 and "
          "wider again before decimalisation in 2001.")
    return 0


def cmd_build_delisting(args) -> int:
    from . import delisting
    df = delisting.build()
    print()
    print(delisting.summarise(df).to_string(index=False))
    unresolved = int((df["reason_category"] == "unresolved").sum())
    print(f"\n{unresolved} of {len(df)} securities have no recorded reason and default "
          "to an index removal at the last price.")
    print("sp500_changes is under-recorded before 2010 (ADR-010), so most of those are "
          "early-era. Each row carries its assumption in the `assumption` column.")
    return 0


def cmd_build_panel(args) -> int:
    from .panel import build_panel
    p = build_panel(start=args.start, rebuild=True)
    print(json.dumps(p.meta, indent=2))
    return 0


# ------------------------------------------------------------------ helpers

def _apply_construction_overrides(strat, args) -> None:
    """Let the CLI retune top_k / weighting / cap without editing the strategy."""
    from dataclasses import replace
    c = getattr(strat, "construction", None)
    if c is None:
        return
    changes = {}
    if args.top_k is not None:
        changes["top_k"] = args.top_k
    if args.max_weight is not None:
        changes["max_weight"] = args.max_weight
    if args.weighting is not None:
        changes["weighting"] = args.weighting
    if changes:
        strat.construction = replace(c, **changes)


def _warn_if_only_optimistic(results) -> None:
    """The check costs.py exists to make: does this survive being charged properly?"""
    opt, pess = results[0].performance.cagr, results[2].performance.cagr
    if opt > 0 and pess <= 0:
        print("\n!! This strategy is profitable only under `optimistic` costs. "
              "That is a bet on the spread estimator being wrong in your favour, "
              "not a strategy. See docs/BACKTEST.md.")


def default_results_dir() -> str:
    return str(PROJECT_ROOT / "results")
