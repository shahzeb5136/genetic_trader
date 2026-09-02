"""CLI for the calendar lab: `sp500lab timing ...`.

    timing accept                     the engine's two identities - run this first
    timing run tm_overnight           one rule, one cost setting
    timing run tm_overnight --all-costs
    timing suite                      every rule x three cost settings, scored vs SPY
    timing decompose                  per-ticker overnight/intraday split, as CSV

Heavy imports stay inside the handlers, same as every other CLI here.
"""

from __future__ import annotations

import argparse
import logging

log = logging.getLogger(__name__)


def add_parser(sub) -> None:
    p = sub.add_parser("timing", help="calendar/timing strategies on daily legs")
    ts = p.add_subparsers(dest="timing_command", required=True)

    a = ts.add_parser("accept", help="assert the leg engine's calibration and "
                                     "decomposition identities")
    a.set_defaults(func=cmd_accept)

    r = ts.add_parser("run", help="backtest one calendar rule")
    r.add_argument("name", help="rule name, e.g. tm_overnight")
    _research_args(r)
    r.add_argument("--all-costs", action="store_true",
                   help="run optimistic, realistic and pessimistic together")
    r.set_defaults(func=cmd_run)

    s = ts.add_parser("suite", help="every rule under all three cost settings, "
                                    "scored against buy-and-hold")
    _research_args(s)
    s.set_defaults(func=cmd_suite)

    d = ts.add_parser("decompose", help="per-ticker overnight vs intraday split "
                                        "across the point-in-time index")
    d.add_argument("--start", default="2007-04-01")
    d.add_argument("--end", default="2021-12-31")
    d.add_argument("--out", default=None, help="write the full table as CSV here")
    d.add_argument("--top", type=int, default=15, help="rows to print each way")
    d.set_defaults(func=cmd_decompose)

    se = ts.add_parser("seal", help="pre-register rules for a forward test - "
                                    "runs the research window only, spends nothing")
    se.add_argument("name", help="rule name, or 'all' for the whole family")
    se.add_argument("--rationale", required=True,
                    help="why these are worth a look - written BEFORE the answer")
    se.set_defaults(func=cmd_seal)

    fw = ts.add_parser("forward", help="forward-test rules on 2022+ - THIS SPENDS "
                                       "LOOKS and is permanently recorded")
    fw.add_argument("name", help="rule name, or 'all' for the whole family")
    fw.add_argument("--rationale", default="")
    fw.set_defaults(func=cmd_forward)

    ts.add_parser("list", help="the rules and their claims").set_defaults(func=cmd_list)


def _research_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--start", default="2007-04-01")
    p.add_argument("--end", default=None)
    p.add_argument("--costs", default="realistic",
                   choices=("optimistic", "realistic", "pessimistic", "free"))
    p.add_argument("--holdout", default="exclude",
                   choices=("exclude", "include", "only"),
                   help="anything but 'exclude' is permanently recorded")
    p.add_argument("--study", default=None, help="log runs as trials of this study")
    p.add_argument("--no-log", action="store_true", help="skip the trial log "
                   "(the holdout ledger is never skipped)")
    p.add_argument("--notes", default="")


def cmd_accept(args) -> int:
    from .engine import timing_accept
    rep = timing_accept()
    print("TIMING ENGINE ACCEPTANCE")
    print(f"  window                     {rep['window']}")
    print(f"  buy-and-hold CAGR          {rep['buy_hold_cagr'] * 100:.2f}%")
    print(f"  overnight-only CAGR        {rep['overnight_cagr'] * 100:.2f}%")
    print(f"  intraday-only CAGR         {rep['intraday_cagr'] * 100:.2f}%")
    print(f"  calibration drift          {rep['calibration_bp_per_year']:.3f} bp/yr "
          f"(tolerance 1.0)")
    print(f"  decomposition max error    {rep['decomposition_max_rel_err']:.2e}")
    print("  PASS - overnight x intraday multiplies back to buy-and-hold, and "
          "buy-and-hold matches the adjusted series.")
    return 0


def cmd_run(args) -> int:
    from .engine import run_all_cost_settings, run_timing_backtest

    kw = dict(start=args.start, end=args.end, holdout=args.holdout,
              study=args.study, log_run=not args.no_log, notes=args.notes)
    if args.all_costs:
        results = run_all_cost_settings(args.name, **kw)
    else:
        results = [run_timing_backtest(args.name, costs=args.costs, **kw)]
    for res in results:
        print(res.summary())
        print()
    return 0


def cmd_suite(args) -> int:
    """Every rule x three cost settings; the realistic rows scored against SPY."""
    from ..backtest.results import format_suite, suite
    from .data import load_timing_data
    from .engine import run_timing_backtest
    from .strategies import TIMING_GROUPS

    data = load_timing_data()
    kw = dict(data=data, start=args.start, end=args.end, holdout=args.holdout,
              study=args.study, log_run=not args.no_log, notes=args.notes)
    realistic = []
    for name in TIMING_GROUPS["all"]:
        for costs in ("optimistic", "realistic", "pessimistic"):
            res = run_timing_backtest(name, costs=costs, **kw)
            if costs == "realistic":
                realistic.append(res)
            p = res.performance
            print(f"  {name:20s} {costs:12s} CAGR {p.cagr * 100:7.2f}%  "
                  f"Sharpe {p.sharpe:5.2f}  maxDD {p.max_drawdown * 100:6.1f}%  "
                  f"cost drag {(p.cost_drag or 0) * 100:5.2f}%")
    print()
    print("REALISTIC COSTS, EACH AGAINST BUY-AND-HOLD SPY OVER ITS OWN WINDOW")
    print(format_suite(suite(realistic)))
    print()
    print("Read `d_sharpe`, nothing else: a rule in cash most of the time gets a "
          "structural Sharpe boost, and CAGR ranks time-in-market, not skill.")
    return 0


def cmd_decompose(args) -> int:
    from .decompose import decompose_members, summarise

    df = decompose_members(start=args.start, end=args.end)
    if df.empty:
        print("no securities with enough in-index sessions")
        return 1
    s = summarise(df)
    print(f"{s['names']} securities with >=500 in-index sessions, "
          f"{args.start}..{args.end}  (gross of costs - see docs/TIMING.md)")
    print(f"  median overnight return    {s['median_overnight_ann'] * 100:7.2f}%/yr  "
          f"({s['overnight_positive']:.0%} of names positive)")
    print(f"  median intraday return     {s['median_intraday_ann'] * 100:7.2f}%/yr  "
          f"({s['intraday_positive']:.0%} of names positive)")
    print(f"  overnight beats intraday   {s['overnight_beats_intraday']:.0%} of names")

    cols = ["ticker", "sessions", "overnight_ann", "intraday_ann", "total_ann",
            "overnight_share"]
    fmt = df[cols].copy()
    for c in ("overnight_ann", "intraday_ann", "total_ann"):
        fmt[c] = fmt[c].map(lambda v: f"{v * 100:.1f}%")
    import pandas as pd
    fmt["overnight_share"] = fmt["overnight_share"].map(
        lambda v: "n/a" if pd.isna(v) else f"{v:+.2f}")
    print(f"\nTOP {args.top} BY OVERNIGHT RETURN")
    print(fmt.head(args.top).to_string(index=False))
    print(f"\nBOTTOM {args.top} BY OVERNIGHT RETURN")
    print(fmt.tail(args.top).to_string(index=False))

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nfull table ({len(df)} rows) written to {args.out}")
    return 0


def timing_runner(strategy, *, panel=None, features=None, liquidity_floor=0.0,
                  seed=0, benchmark="SPY", record_trades=True, **kw):
    """`run_backtest`'s keyword surface, mapped onto the leg engine.

    This is what lets a calendar rule flow through the REAL forward harness -
    seals, paired comparison, verdicts, the store - rather than a parallel one.
    The dropped keywords are the ones a single-instrument schedule has no use for:
    there is no panel to pass, no features, no liquidity floor, no randomness.
    """
    from .engine import run_timing_backtest
    kw.pop("strategy_kwargs", None)
    return run_timing_backtest(strategy, **kw)


def _resolve_rules(name: str) -> list:
    from .strategies import TIMING_GROUPS, get_timing_strategy
    names = TIMING_GROUPS["all"] if name == "all" else [name]
    return [get_timing_strategy(n) for n in names]


def cmd_seal(args) -> int:
    from ..forward.engine import DEFAULT_COSTS, seal_candidate

    for strat in _resolve_rules(args.name):
        seals = seal_candidate(strat, rationale=args.rationale, costs=DEFAULT_COSTS,
                               origin_study="timing-1", runner=timing_runner)
        for s in seals:
            print(f"  sealed {strat.name:20s} {s.cost_model:12s} {s.seal_id}  "
                  f"({s.seal_mode})")
    print("\nNo look was spent. `timing forward` runs the test; the gap between "
          "now and then is the evidence the choice preceded the answer.")
    return 0


def cmd_forward(args) -> int:
    from ..forward.engine import forward_test

    for strat in _resolve_rules(args.name):
        try:
            t = forward_test(strat, rationale=args.rationale,
                             origin_study="timing-1", save=False,
                             runner=timing_runner)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  {strat.name}: FAILED - {exc}")
            continue
        print(t.summary())
        print()
    return 0


def cmd_list(args) -> int:
    from .strategies import _TIMING_REGISTRY

    for name in sorted(_TIMING_REGISTRY):
        doc = (_TIMING_REGISTRY[name].__doc__ or "").strip().splitlines()[0]
        print(f"  {name:20s} {doc}")
    return 0
