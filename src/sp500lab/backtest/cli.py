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

    sp500lab experiments studies
    sp500lab experiments list --study momentum-variants
    sp500lab experiments deflate momentum-variants
    sp500lab experiments holdout
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
    _add_research_args(r)
    r.set_defaults(func=cmd_run)

    b = bs.add_parser("baselines", help="run every baseline and print the scoreboard")
    b.add_argument("--start", default="2007-04-01")
    b.add_argument("--end", default=None)
    b.add_argument("--costs", default="realistic")
    b.add_argument("--save", default=None)
    _add_research_args(b, default_study="baselines")
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


def _add_research_args(parser, default_study: str | None = None) -> None:
    """The two flags that keep a search honest. See docs/EXPERIMENTS.md."""
    from .registry import ADHOC_STUDY, HOLDOUT_START
    parser.add_argument(
        "--study", default=default_study or ADHOC_STUDY,
        help="name of the search this run belongs to. Decides n_trials for the "
             "deflated Sharpe, so make it match the search you actually ran.")
    parser.add_argument(
        "--holdout", default="exclude", choices=["exclude", "include", "only"],
        help=f"'exclude' (default) stops the day before {HOLDOUT_START}; 'include' runs "
             "through it; 'only' runs the final test. The last two are recorded in the "
             "holdout ledger and cannot be silenced.")
    parser.add_argument("--no-log", action="store_true",
                        help="do not record this run as a trial (the holdout ledger is "
                             "still written)")
    parser.add_argument("--notes", default="", help="free text stored with the run")


def add_experiments_parser(sub) -> None:
    """Register the `experiments` subcommand tree."""
    p = sub.add_parser("experiments",
                       help="the trial log and the holdout ledger")
    es = p.add_subparsers(dest="experiments_command", required=True)

    st = es.add_parser("studies", help="one row per search: runs, trials, best")
    st.set_defaults(func=cmd_studies)

    ls = es.add_parser("list", help="logged runs")
    ls.add_argument("--study", default=None)
    ls.add_argument("--sort", default="sharpe")
    ls.add_argument("--top", type=int, default=25)
    ls.set_defaults(func=cmd_experiments_list)

    df = es.add_parser("deflate",
                       help="deflated Sharpe for a study's best run - read this before "
                            "believing any searched result")
    df.add_argument("study")
    df.add_argument("--run-id", default=None, help="deflate a specific run instead")
    df.set_defaults(func=cmd_deflate)

    sh = es.add_parser("show", help="everything recorded about one run")
    sh.add_argument("run_id")
    sh.set_defaults(func=cmd_show)

    ho = es.add_parser("holdout", help="every recorded look at the holdout period")
    ho.set_defaults(func=cmd_holdout)


# ----------------------------------------------------------------- commands

def cmd_run(args) -> int:
    from . import run_backtest
    from .results import compare, format_compare
    from .strategy import get_strategy

    strat = get_strategy(args.strategy)
    _apply_construction_overrides(strat, args)

    common = dict(start=args.start, end=args.end, initial_capital=args.capital,
                  liquidity_floor=args.liquidity_floor, seed=args.seed,
                  benchmark=args.benchmark, min_coverage=args.min_coverage,
                  holdout=args.holdout, study=args.study,
                  log_run=not args.no_log, notes=args.notes)

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
    _print_run_footer(result)

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
                                        costs=args.costs, holdout=args.holdout,
                                        study=args.study, log_run=not args.no_log,
                                        notes=args.notes))
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
    print()
    print("Buy-and-hold SPY over the SAME window, which is the bar all of these")
    print("have to clear - a full-history SPY figure would not be comparable:")
    from . import metrics
    from .benchmark import benchmark_total_return
    span = results[0].equity.index
    spy = benchmark_total_return("SPY").reindex(span).ffill().dropna()
    perf = metrics.compute(spy)
    print(f"  SPY  CAGR {perf.cagr * 100:.2f}%   vol {perf.ann_vol * 100:.2f}%   "
          f"Sharpe {perf.sharpe:.2f}   maxDD {perf.max_drawdown * 100:.2f}%")
    print(f"       {span[0]} .. {span[-1]}")

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


# ------------------------------------------- experiments / registry commands

def cmd_studies(args) -> int:
    from . import registry
    df = registry.studies()
    if df.empty:
        print("No runs logged yet. Every backtest is logged automatically unless you "
              "pass --no-log or set SP500LAB_REGISTRY=off.")
        return 0
    print("Every search you have run. `trials` is what the deflated Sharpe uses -")
    print("distinct configurations, not log lines. See docs/EXPERIMENTS.md.")
    print()
    out = df.copy()
    out["best_sharpe"] = out["best_sharpe"].map(lambda v: f"{v:.2f}")
    print(out.to_string(index=False))
    touches = registry.holdout_touch_count()
    print()
    print(f"Holdout ({registry.HOLDOUT_START} onward) has been looked at "
          f"{touches} time(s).")
    return 0


def cmd_experiments_list(args) -> int:
    from . import registry
    df = registry.load(args.study)
    if df.empty:
        scope = f" for study {args.study}" if args.study else ""
        print(f"No runs logged{scope}.")
        return 0
    cols = [c for c in ("run_id", "study", "strategy", "cost_model", "start", "end",
                        "cagr", "sharpe", "sharpe_monthly", "max_drawdown",
                        "ann_turnover", "touched_holdout") if c in df.columns]
    if args.sort in df.columns:
        df = df.sort_values(args.sort, ascending=False)
    out = df[cols].head(args.top).copy()
    for c in ("cagr", "max_drawdown", "ann_turnover"):
        if c in out.columns:
            out[c] = out[c].map(
                lambda v: "n/a" if v is None or v != v else f"{v * 100:.2f}%")
    for c in ("sharpe", "sharpe_monthly"):
        if c in out.columns:
            out[c] = out[c].map(lambda v: "n/a" if v != v else f"{v:.2f}")
    print(out.to_string(index=False))
    print()
    print(f"{len(df)} run(s), {df['fingerprint'].nunique()} distinct trial(s).")
    return 0


def cmd_deflate(args) -> int:
    """The number that decides whether a searched result means anything."""
    from . import registry
    try:
        res = (registry.deflate(args.run_id, args.study) if args.run_id
               else registry.deflate_best(args.study))
    except KeyError as exc:
        print(f"  {exc}")
        return 1

    print("=" * 72)
    print(f"DEFLATED SHARPE   study={res['study']}")
    print("=" * 72)
    for k, v in res.items():
        print(f"  {k:34s} {v}")

    dsr = res.get("deflated_sharpe")
    print()
    if dsr is None or dsr != dsr:
        print("  Not enough monthly observations to deflate.")
    elif dsr >= 0.95:
        print("  >= 0.95: the result survives the search that produced it.")
    else:
        print(f"  {dsr:.3f} < 0.95: NOT distinguishable from the best of "
              f"{res['n_trials']} lucky draws, however good the raw Sharpe looks.")
    print()
    print("  Read this as a probability, not a score. `n_trials` counts distinct")
    print("  configurations in the study - if that undercounts what you actually")
    print("  tried, the number above is too generous.")
    return 0


def cmd_show(args) -> int:
    from . import registry
    row = registry.get(args.run_id)
    if row is None:
        print(f"  no run with id {args.run_id!r}")
        return 1
    print(json.dumps({k: (v.item() if hasattr(v, "item") else v)
                      for k, v in row.to_dict().items()}, indent=2, default=str))
    return 0


def cmd_holdout(args) -> int:
    from . import registry
    df = registry.holdout_touches()
    print(f"Holdout period: {registry.HOLDOUT_START} onward (ADR-025).")
    print("Everything before it is the research window.")
    print()
    if df.empty:
        print("The holdout has never been looked at. Keep it that way until you have a")
        print("final candidate - every look degrades it, and a look cannot be undone.")
        return 0
    print(f"!! {len(df)} recorded look(s):")
    print()
    cols = [c for c in ("at", "strategy", "study", "mode", "start", "end", "reason")
            if c in df.columns]
    print(df[cols].to_string(index=False))
    print()
    print("Each of these saw reserved data. The more entries here, the less the")
    print("holdout is worth as an independent test.")
    return 0


# ------------------------------------------------------------------ helpers

def _print_run_footer(result) -> None:
    """Tell the user where the run was recorded, so the log is never a surprise."""
    cfg = result.config
    if cfg.get("run_id"):
        print(f"\nlogged as {cfg['run_id']}  study={cfg.get('study')}  "
              f"fingerprint={cfg.get('fingerprint')}")
    if cfg.get("touched_holdout"):
        print("!! this run saw HOLDOUT data and is recorded in the holdout ledger "
              "(`sp500lab experiments holdout`)")


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
